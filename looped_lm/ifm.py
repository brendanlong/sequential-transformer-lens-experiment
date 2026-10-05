# xLLM and datasets live in the separate IFM env (looped_lm/jobs/ifm_env.sh).
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
"""Logit lens, a least-squares tuned lens, and readout geometry on IFM's LoopedLM suite.

IFM's "Towards Looped Models Done Right" Part I trains every recipe on the same
336B TxT360 tokens with the same tokenizer, width, seed and schedule, with
loss on the final output only. All loop recipes store 28 blocks and execute
112; ``dense-d112`` executes 112 distinct blocks, ``dense-d28`` 28.

Records the residual stream after every block *execution* (so a looped model
yields 112 states, like dense-d112) at sampled positions of FineWeb-Edu text,
then for each state reports:

- logit lens: KL(final ‖ lens), top-1 agreement with the final output, and
  next-token top-1 accuracy;
- a linear translator fit by ridge regression from that state to the final
  state, read out through the model's own head (a closed-form tuned lens): how
  much of the final prediction is linearly recoverable;
- variance in each quarter of the readout's singular directions, and relative
  visibility (see ``lens_metrics``);
- sparse reconstruction from the 250k token directions vs a Gaussian null, and
  whether the first token picked is the current token, the next token, or the
  model's final prediction;
- lens logit outliers.

Runs in an environment with xLLM-Loop's dependencies and its checkout on
``PYTHONPATH``; ``--offload`` streams one block at a time to the GPU, so
dense-d112 (7.4 GB of bf16 weights) runs on an 8 GB card.

    PYTHONPATH=.:../ext/xllm-loop python -m looped_lm.ifm \
        --artifact ../ext/ifm/dense-ouro-336b \
        --out results/looped-lm/ifm/dense-ouro-336b.json
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from looped_lm import lens_metrics as lm
from looped_lm import xllm_compat


def load_text_tokens(tokenizer: object, n_seq: int, seq_len: int) -> Tensor:
    """The first seq_len tokens (with BOS) of n_seq long-enough FineWeb-Edu docs."""
    from datasets import load_dataset

    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )
    rows = []
    for doc in stream:
        ids = tokenizer.encode(doc["text"], bos=True, eos=False)  # type: ignore[attr-defined]
        if len(ids) >= seq_len:
            rows.append(ids[:seq_len])
            if len(rows) == n_seq:
                break
    return torch.tensor(rows)


class Recorder:
    """Forward hooks that keep every block execution's output at chosen positions."""

    def __init__(self, model: nn.Module, device: torch.device, offload: bool) -> None:
        self.states: list[Tensor] = []
        self.layer_ids: list[int] = []
        self.positions: Tensor | None = None
        self.handles = [model.embed.register_forward_hook(self._embed_hook)]
        for i, layer in enumerate(model.layers):
            if offload:
                self.handles.append(
                    layer.register_forward_pre_hook(lambda mod, _args: mod.to(device))
                )
            self.handles.append(
                layer.register_forward_hook(self._layer_hook(i, offload))
            )

    def _take(self, hidden: Tensor) -> Tensor:
        assert self.positions is not None
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        return hidden[rows, self.positions.to(hidden.device)].detach().cpu()

    def _embed_hook(self, _mod: nn.Module, _args: object, out: Tensor) -> None:
        self.states.append(self._take(out))
        self.layer_ids.append(-1)

    def _layer_hook(self, i: int, offload: bool):  # noqa: ANN202
        def hook(mod: nn.Module, _args: object, out: tuple) -> None:
            self.states.append(self._take(out[0]))
            self.layer_ids.append(i)
            if offload:
                mod.to("cpu")

        return hook


@torch.no_grad()
def record(
    model: nn.Module,
    tokens: Tensor,
    positions: Tensor,
    device: torch.device,
    offload: bool,
    batch: int,
) -> tuple[Tensor, list[int]]:
    """(executions + 1, n_seq, positions, dim) bf16 states, and each one's block id."""
    rec = Recorder(model, device, offload)
    chunks = []
    for start in range(0, len(tokens), batch):
        rec.states, rec.layer_ids = [], []
        rec.positions = positions[start : start + batch]
        model(tokens[start : start + batch].to(device), multi_segments=False)
        chunks.append(torch.stack(rec.states))
    for h in rec.handles:
        h.remove()
    return torch.cat(chunks, dim=1), rec.layer_ids


@torch.no_grad()
def effective_gain(norm: nn.Module, dim: int, eps: float) -> Tensor:
    """The RMSNorm's per-feature scale, read off a vector of ones.

    xLLM stores the scale as ``weight + 1``; asking the module avoids depending
    on that convention.
    """
    param = next(norm.parameters())
    ones = torch.ones(1, dim, dtype=torch.float32, device=param.device)
    return (norm(ones.to(param.dtype)).float() * (1 + eps) ** 0.5)[0]


def rms_normed(x: Tensor, eps: float) -> Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def ridge(x: Tensor, y: Tensor, rel_lambda: float = 1e-3) -> Tensor:
    """argmin_A ‖x A − y‖² + λ‖A‖², λ = rel_lambda · tr(xᵀx)/dim."""
    gram = x.T.double() @ x.double()
    lam = rel_lambda * gram.trace() / gram.shape[0]
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + lam * eye, x.T.double() @ y.double()).float()


@torch.no_grad()
def lens_stats(
    logits_fn: object, z: Tensor, final_logp: Tensor, next_tok: Tensor, chunk: int
) -> dict[str, float]:
    kl = agree = next_acc = 0.0
    for s in range(0, len(z), chunk):
        logp = F.log_softmax(logits_fn(z[s : s + chunk]), dim=-1)  # type: ignore[operator]
        fp = final_logp[s : s + chunk]
        kl += (fp.exp() * (fp - logp)).sum().item()
        top = logp.argmax(-1)
        agree += (top == fp.argmax(-1)).sum().item()
        next_acc += (top == next_tok[s : s + chunk]).sum().item()
    n = len(z)
    return {
        "kl_to_final": kl / n,
        "agree_final_top1": agree / n,
        "next_token_top1": next_acc / n,
    }


def analyse(args: argparse.Namespace) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, tokenizer, cfg = xllm_compat.load(
        args.artifact, device="cpu" if args.offload else device
    )
    if args.offload:
        for name, child in model.named_children():
            if name != "layers":
                child.to(device)

    tokens = load_text_tokens(tokenizer, args.n_seq, args.seq_len)
    rng = random.Random(args.seed)
    positions = torch.tensor(
        [
            sorted(rng.sample(range(args.min_pos, args.seq_len - 1), args.n_pos))
            for _ in range(args.n_seq)
        ]
    )
    next_tok = tokens.gather(1, positions + 1)
    cur_tok = tokens.gather(1, positions)
    states, layer_ids = record(
        model, tokens, positions, device, args.offload, args.batch
    )
    print(f"recorded {states.shape} states", flush=True)

    head = model.output
    eps = getattr(head.final_norm, "eps", cfg.rmsnorm_eps)
    gain = effective_gain(head.final_norm, cfg.model_dim, eps).to(device)
    unembed = head.output.weight.detach().float().to(device)
    del model, head
    torch.cuda.empty_cache()

    def logits_from_z(z: Tensor) -> Tensor:
        return (z * gain) @ unembed.T

    readout = lm.folded_readout(unembed, gain, centers=False)
    s, vh = lm.singular_basis(readout)
    atoms = F.normalize(readout, dim=-1)
    del readout

    n_train = args.n_seq // 2
    flat = states.flatten(1, 2)  # (executions + 1, n_seq * n_pos, dim)
    split = n_train * args.n_pos
    next_eval = next_tok.flatten()[split:].to(device)
    cur_eval = cur_tok.flatten()[split:].to(device)
    z_final_train = rms_normed(flat[-1, :split].to(device), eps)
    z_final_eval = rms_normed(flat[-1, split:].to(device), eps)
    final_logp = torch.cat(
        [
            F.log_softmax(logits_from_z(z_final_eval[c : c + args.chunk]), dim=-1)
            for c in range(0, len(z_final_eval), args.chunk)
        ]
    )
    final_top1 = final_logp.argmax(-1)
    generator = torch.Generator().manual_seed(args.seed)

    rows = []
    for layer in range(flat.shape[0]):
        z_train = rms_normed(flat[layer, :split].to(device), eps)
        z_eval = rms_normed(flat[layer, split:].to(device), eps)
        translator = ridge(z_train, z_final_train)
        lens = lens_stats(logits_from_z, z_eval, final_logp, next_eval, args.chunk)
        tuned = lens_stats(
            lambda z, a=translator: logits_from_z(z @ a),
            z_eval,
            final_logp,
            next_eval,
            args.chunk,
        )
        zc = z_eval - z_eval.mean(0)
        sub = zc[: args.n_omp]
        r2, chosen = lm.omp_r2(sub, atoms, args.k_max)
        r2_null, _ = lm.omp_r2(lm.gaussian_null(sub, generator), atoms, args.k_max)
        first = chosen[:, 0]
        outliers = torch.cat(
            [
                lm.outlier_counts(logits_from_z(z_eval[c : c + args.chunk]))
                for c in range(0, len(z_eval), args.chunk)
            ]
        )
        rows.append(
            {
                "execution": layer,
                "block": layer_ids[layer],
                "lens": lens,
                "tuned_ridge": tuned,
                "energy_bands_raw": lm.energy_by_singular_band(z_eval, vh).tolist(),
                "energy_bands_centered": lm.energy_by_singular_band(zc, vh).tolist(),
                "relative_visibility_raw": lm.relative_visibility(z_eval, s, vh),
                "relative_visibility_centered": lm.relative_visibility(zc, s, vh),
                "omp_r2": r2.tolist(),
                "omp_r2_null": r2_null.tolist(),
                "omp_first_atom": {
                    "current_token": (first == cur_eval[: args.n_omp])
                    .float()
                    .mean()
                    .item(),
                    "next_token": (first == next_eval[: args.n_omp])
                    .float()
                    .mean()
                    .item(),
                    "final_top1": (first == final_top1[: args.n_omp])
                    .float()
                    .mean()
                    .item(),
                },
                "outliers_mean": outliers.float().mean().item(),
                "outliers_median": outliers.float().median().item(),
            }
        )
        print(
            f"{layer:>3} block {layer_ids[layer]:>3} KL {lens['kl_to_final']:.2f} "
            f"tuned {tuned['kl_to_final']:.2f} agree {lens['agree_final_top1']:.0%} "
            f"R2(1) {r2[0]:.2f}/{r2_null[0]:.2f} "
            f"outliers {outliers.float().mean():.1f}",
            flush=True,
        )
    return {
        "artifact": str(args.artifact),
        "arch": cfg.arch,
        "num_layers": cfg.num_layers,
        "loop_times": cfg.loop_times,
        "singular_values": s.tolist(),
        "n_eval_tokens": len(z_final_eval),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-seq", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-pos", type=int, default=64)
    parser.add_argument("--min-pos", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--n-omp", type=int, default=512)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = analyse(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
