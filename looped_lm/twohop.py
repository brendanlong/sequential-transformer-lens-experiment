"""Lens vs probes on Kohli et al.'s looped and vanilla two-hop models.

Checkpoints and data are from "Loop, Think, & Generalize" (Kohli et al. 2026,
arXiv 2604.07822, github.com/OSU-NLP-Group/Loop-Think-Generalize). Each model
reads ``<h><r1><r2>`` and predicts the tail ``t = r2(r1(h))``; the bridge
``b = r1(h)`` is never a training target at any position (loss is on the last
position only). ``r<R>_l<L>`` is an L-layer GPT-2 block stack applied R times,
so ``r2_l4`` and ``r1_l8`` have the same effective depth (8) and FLOPs.

For every effective layer (0 = embeddings) and each of bridge@r1, bridge@r2,
target@r2, on the train / test-ID / test-OOD inferred splits, reports:

- logit-lens top-1 accuracy (whole vocabulary, as in their Figure 4, and
  restricted to entity tokens);
- a linear probe and a tuned-lens-style probe (affine map into the frozen
  readout), trained on training compositions;
- how much of the residual's variance sits in the readout's strong vs weak
  singular directions, and probes restricted to each half;
- sparse reconstruction from token directions vs a Gaussian null, and which
  tokens OMP picks first;
- lens logit outliers.

    uv run python -m looped_lm.twohop --root ext/kohli \
        --checkpoint checkpoints/systematicity/r2_l4/checkpoint_epoch_1201.pt
"""

import argparse
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from looped_lm import lens_metrics as lm

DATA = "data/composition.2000.200.7.2"
SPLITS = ("train_inferred", "test_inferred_iid", "test_inferred_ood")
TARGETS = {
    "bridge@r1": (1, "bridge"),
    "bridge@r2": (2, "bridge"),
    "target@r2": (2, "tail"),
}
TOKEN = re.compile(r"<[^>]+>")


class Conv1D(nn.Module):
    """GPT-2's transposed linear layer: weight is (in, out)."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.bias = nn.Parameter(torch.empty(n_out))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_attn = Conv1D(dim, 3 * dim)
        self.c_proj = Conv1D(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = (
            h.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
            for h in self.c_attn(x).split(d, dim=-1)
        )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.c_fc = Conv1D(dim, 4 * dim)
        self.c_proj = Conv1D(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """transformers' GPT2Block in eval mode, with the same parameter names."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, n_heads)
        self.ln_2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class RecurrentGPT2(nn.Module):
    """Kohli et al.'s RecurrentGPT2Block: n_layers blocks applied n_iterations times."""

    def __init__(
        self, vocab: int, dim: int, n_heads: int, n_layers: int, n_iterations: int
    ) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.token_embedding = nn.Embedding(vocab, dim)
        self.position_embedding = nn.Embedding(5, dim)
        self.ln_f = nn.LayerNorm(dim, eps=1e-5)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def residuals(
        self, ids: Tensor, edit: Callable[[int, Tensor], Tensor] | None = None
    ) -> list[Tensor]:
        """Residual stream after the embeddings and after every block execution.

        ``edit(layer, x)``, if given, replaces the residual at each effective
        layer (0 = embeddings) before the model continues from it.
        """
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(pos)
        if edit is not None:
            x = edit(0, x)
        out = [x]
        for _ in range(self.n_iterations):
            for block in self.blocks:
                x = block(x)
                if edit is not None:
                    x = edit(len(out), x)
                out.append(x)
        return out

    def lens(self, x: Tensor) -> Tensor:
        return self.lm_head(self.ln_f(x))


def load_model(path: Path, device: torch.device) -> tuple[RecurrentGPT2, int]:
    """Model and its training epoch; R and L come from the ``r<R>_l<L>`` directory."""
    match = re.fullmatch(r"r(\d+)_l(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"expected a r<R>_l<L> directory, got {path.parent}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()
    }
    vocab, dim = state["token_embedding.weight"].shape
    model = RecurrentGPT2(vocab, dim, 12, int(match[2]), int(match[1]))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), ckpt["epoch"]


@dataclass
class Examples:
    ids: Tensor  # (n, 3): h, r1, r2
    bridge: Tensor
    tail: Tensor


def parse_tokens(item: dict, vocab: dict[str, int]) -> list[int]:
    return [vocab[t] for t in TOKEN.findall(item["target_text"]) if t != "</a>"]


def load_atomic(root: Path, vocab: dict[str, int]) -> dict[tuple[int, int], int]:
    """Every atomic fact (h, r) -> t; all are in the training set."""
    rows = json.loads((root / DATA / "train.json").read_text())
    parsed = (parse_tokens(x, vocab) for x in rows)
    return {(h, r): t for h, r, t in (x for x in parsed if len(x) == 3)}


def load_data(
    root: Path, n_probe: int, n_eval: int, seed: int
) -> tuple[dict[str, int], Examples, dict[str, Examples]]:
    """Vocab, probe-training compositions, and the evaluation splits."""
    data = root / DATA
    tokens = json.loads((data / "vocab.json").read_text())
    if "<pad>" not in tokens:
        tokens = ["<pad>", *tokens]
    vocab = {tok: i for i, tok in enumerate(tokens)}

    def parse(item: dict) -> list[int]:
        return parse_tokens(item, vocab)

    train = [parse(x) for x in json.loads((data / "train.json").read_text())]
    atomic = {(h, r): t for h, r, t in (x for x in train if len(x) == 3)}

    def examples(rows: list[list[int]]) -> Examples:
        return Examples(
            ids=torch.tensor([r[:3] for r in rows]),
            bridge=torch.tensor([atomic[r[0], r[1]] for r in rows]),
            tail=torch.tensor([r[3] for r in rows]),
        )

    rng = random.Random(seed)
    test = json.loads((data / "test.json").read_text())
    evals = {}
    for split in SPLITS:
        rows = [parse(x) for x in test if x["type"] == split]
        evals[split] = examples(rng.sample(rows, min(n_eval, len(rows))))
    # The residual at r1 depends only on (h, r1): a probe trained on a
    # composition sharing an evaluation example's (h, r1) has seen its input.
    held_out = {(h, r1) for ex in evals.values() for h, r1 in ex.ids[:, :2].tolist()}
    composed = [x for x in train if len(x) == 4 and (x[0], x[1]) not in held_out]
    probe_set = examples(rng.sample(composed, n_probe))
    return vocab, probe_set, evals


@torch.no_grad()
def collect(model: RecurrentGPT2, ids: Tensor, batch: int = 2048) -> Tensor:
    """(layers, n, positions 1-2, dim) raw residuals, on the CPU."""
    device = model.lm_head.weight.device
    chunks = [
        torch.stack(
            [r[:, 1:3].cpu() for r in model.residuals(ids[i : i + batch].to(device))]
        )
        for i in range(0, len(ids), batch)
    ]
    return torch.cat(chunks, dim=1)


def normed(x: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm without its gain and bias: what the readout's z is."""
    return F.layer_norm(x.float(), x.shape[-1:], eps=eps)


def analyse(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    root = Path(args.root)
    ckpt_path = root / args.checkpoint
    model, epoch = load_model(ckpt_path, device)
    vocab, probe_set, evals = load_data(root, args.n_probe, args.n_eval, args.seed)
    entities = torch.tensor(sorted(i for t, i in vocab.items() if t.startswith("<e_")))
    entity_index = torch.full((len(vocab),), -1, dtype=torch.long)
    entity_index[entities] = torch.arange(len(entities))
    entities_dev = entities.to(device)

    embed = model.token_embedding.weight.detach()
    readout = lm.folded_readout(embed, model.ln_f.weight.detach(), centers=True)
    readout[0] = 0  # <pad> is never an output
    s, vh = lm.singular_basis(readout)
    dim = vh.shape[1]
    top_half, bottom_half = vh[: dim // 2], vh[dim // 2 :]
    entity_readout = readout[entities_dev]
    atoms = F.normalize(readout[1:], dim=-1)  # OMP dictionary, ids offset by 1

    res_probe = collect(model, probe_set.ids)
    res_eval = {split: collect(model, ex.ids) for split, ex in evals.items()}
    n_layers = res_probe.shape[0]
    generator = torch.Generator().manual_seed(args.seed)

    def labels(ex: Examples, field: str) -> Tensor:
        return entity_index[getattr(ex, field)].to(device)

    result: dict = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "iterations": model.n_iterations,
        "block_layers": len(model.blocks),
        "singular_values": s.tolist(),
        "layers": [],
    }
    for layer in range(n_layers):
        row: dict = {"layer": layer, "targets": {}, "positions": {}}
        for name, (pos, field) in TARGETS.items():
            col = pos - 1
            x_tr = normed(res_probe[layer, :, col].to(device))
            y_tr = labels(probe_set, field)
            probes = {
                "linear": lm.train_probe(x_tr, y_tr, len(entities), steps=args.steps),
                "readout": lm.train_probe(
                    x_tr, y_tr, len(entities), readout=entity_readout, steps=args.steps
                ),
            }
            if field == "bridge":
                for half, basis in (("top", top_half), ("bottom", bottom_half)):
                    probes[f"linear_{half}_half"] = _ProjectedProbe(
                        basis,
                        lm.train_probe(
                            x_tr @ basis.T, y_tr, len(entities), steps=args.steps
                        ),
                    )
            per_split: dict = {
                "probe_train": {
                    f"probe_{k}": lm.probe_accuracy(p, x_tr, y_tr)
                    for k, p in probes.items()
                }
            }
            for split, ex in evals.items():
                raw = res_eval[split][layer, :, col].to(device)
                y = labels(ex, field)
                logits = model.lens(raw)
                z = normed(raw)
                outliers = lm.outlier_counts(logits[:, 1:])
                rank = (
                    logits > logits.gather(1, getattr(ex, field).to(device)[:, None])
                ).sum(-1)
                per_split[split] = {
                    "lens_top1": (logits.argmax(-1).cpu() == getattr(ex, field))
                    .float()
                    .mean()
                    .item(),
                    "lens_entity_top1": (logits[:, entities_dev].argmax(-1) == y)
                    .float()
                    .mean()
                    .item(),
                    "label_in_outliers": (rank < outliers).float().mean().item(),
                    **{
                        f"probe_{k}": lm.probe_accuracy(p, z, y)
                        for k, p in probes.items()
                    },
                }
            row["targets"][name] = per_split
        for pos in (1, 2):
            split = "test_inferred_iid"
            raw = res_eval[split][layer, :, pos - 1].to(device)
            z = normed(raw)
            zc = z - z.mean(0)
            r2, chosen = lm.omp_r2(zc, atoms, args.k_max)
            r2_null, _ = lm.omp_r2(lm.gaussian_null(zc, generator), atoms, args.k_max)
            first = chosen[:, 0].cpu() + 1
            ex = evals[split]
            row["positions"][f"r{pos}"] = {
                "energy_bands": lm.energy_by_singular_band(zc, vh).tolist(),
                "relative_visibility": lm.relative_visibility(zc, s, vh),
                "omp_r2": r2.tolist(),
                "omp_r2_null": r2_null.tolist(),
                "omp_first_atom": {
                    "head": (first == ex.ids[:, 0]).float().mean().item(),
                    "r1": (first == ex.ids[:, 1]).float().mean().item(),
                    "r2": (first == ex.ids[:, 2]).float().mean().item(),
                    "bridge": (first == ex.bridge).float().mean().item(),
                    "tail": (first == ex.tail).float().mean().item(),
                },
                "outliers_mean": lm.outlier_counts(model.lens(raw)[:, 1:])
                .float()
                .mean()
                .item(),
            }
        result["layers"].append(row)
        print(f"layer {layer} done", flush=True)
    return result


class _ProjectedProbe(nn.Module):
    basis: Tensor

    def __init__(self, basis: Tensor, probe: nn.Module) -> None:
        super().__init__()
        self.register_buffer("basis", basis)
        self.probe = probe

    def forward(self, x: Tensor) -> Tensor:
        return self.probe(x @ self.basis.T)


def print_summary(result: dict) -> None:
    print(
        f"\n{result['checkpoint']} (epoch {result['epoch']}, "
        f"{result['block_layers']} layers x {result['iterations']})"
    )
    for name in TARGETS:
        for split in SPLITS:
            print(f"\n{name} on {split}")
            keys = list(result["layers"][0]["targets"][name][split])
            print(f"{'L':>3}" + "".join(f"{k[:14]:>15}" for k in keys))
            for row in result["layers"]:
                vals = row["targets"][name][split]
                print(f"{row['layer']:>3}" + "".join(f"{vals[k]:>15.0%}" for k in keys))
    for pos in ("r1", "r2"):
        print(
            f"\nposition {pos} (test ID): variance by singular band, visibility, OMP R²"
        )
        for row in result["layers"]:
            p = row["positions"][pos]
            bands = " ".join(f"{b:.0%}" for b in p["energy_bands"])
            r2 = " ".join(
                f"{a:.2f}/{b:.2f}"
                for a, b in zip(p["omp_r2"][:4], p["omp_r2_null"][:4], strict=True)
            )
            first = " ".join(f"{k}={v:.0%}" for k, v in p["omp_first_atom"].items())
            print(
                f"{row['layer']:>3} [{bands}] vis={p['relative_visibility']:.2f} "
                f"R2(k=1..4) model/null {r2} | first atom {first} "
                f"| outliers {p['outliers_mean']:.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", default="ext/kohli")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-probe", type=int, default=20000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyse(args)
    print_summary(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
