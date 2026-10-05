"""Is the token-shaped bridge at r1 what the second hop reads?

For each test composition ``<h><r1><r2>`` (bridge ``b = r1(h)``, answer
``t = r2(b)``), pick another entity ``b'`` that also has relation ``r2`` (so
``t' = r2(b')`` exists and differs from ``t``). After effective layer L, edit
the residual at the r1 position only, let the model run on, and read the
answer at r2:

- **token swap**: move the residual's component along b's lens readout
  direction onto b''s (``x += c (ŵ_b' − ŵ_b)``, ``c = x·ŵ_b``). This changes
  exactly what the logit lens reads there; the rest of the residual, including
  anything a probe finds in other directions, is left alone.
- **token steer**: the same swap pushed hard, with magnitude equal to the
  residual's own norm (``x += ‖x‖ (ŵ_b' − ŵ_b)``), so it cannot fail just
  because the token component is still small.
- **token ablation**: remove that component (``x −= c ŵ_b``).
- **full patch**: replace the whole residual with the one from a donor input
  ``<h'><r1'>`` whose bridge is b'. The reference for "the r1 state is used at
  all from this layer on".

If the second hop reads the token-shaped component, the token swap flips the
answer to t'.

    uv run python -m looped_lm.twohop_causal --root ext/kohli \
        --checkpoint checkpoints/systematicity/r4_l4/checkpoint_epoch_401.pt
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from looped_lm import lens_metrics as lm
from looped_lm.twohop import SPLITS, Examples, load_atomic, load_data, load_model

R1 = 1


def counterfactuals(
    ex: Examples, atomic: dict[tuple[int, int], int], seed: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Per example: b', t' = r2(b'), and donor ids <h'><r1'><r2> with r1'(h') = b'."""
    by_relation: dict[int, list[int]] = defaultdict(list)
    sources: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (h, r), t in atomic.items():
        by_relation[r].append(h)
        sources[t].append((h, r))
    rng = random.Random(seed)
    b_alt, t_alt, donors = [], [], []
    for (_h, _r1, r2), b, t in zip(
        ex.ids.tolist(), ex.bridge.tolist(), ex.tail.tolist(), strict=True
    ):
        while True:
            cand = rng.choice(by_relation[r2])
            if cand != b and atomic[cand, r2] != t and sources[cand]:
                break
        b_alt.append(cand)
        t_alt.append(atomic[cand, r2])
        h2, r1_2 = rng.choice(sources[cand])
        donors.append([h2, r1_2, r2])
    return torch.tensor(b_alt), torch.tensor(t_alt), torch.tensor(donors)


class Edit:
    """One intervention on the r1 residual after effective layer ``layer``."""

    def __init__(
        self,
        kind: str,
        layer: int,
        w_hat: Tensor,
        b: Tensor,
        b_alt: Tensor,
        donor_r1: Tensor,
        model: torch.nn.Module,
    ) -> None:
        self.kind, self.layer, self.w_hat = kind, layer, w_hat
        self.b, self.b_alt, self.donor_r1, self.model = b, b_alt, donor_r1, model
        self.lens_top1 = torch.empty(0)

    def __call__(self, at: int, x: Tensor) -> Tensor:
        if at != self.layer:
            return x
        x = x.clone()
        r = x[:, R1]
        c = (r * self.w_hat[self.b]).sum(-1, keepdim=True)
        if self.kind == "token_swap":
            r = r + c * (self.w_hat[self.b_alt] - self.w_hat[self.b])
        elif self.kind == "token_steer":
            norm = r.norm(dim=-1, keepdim=True)
            r = r + norm * (self.w_hat[self.b_alt] - self.w_hat[self.b])
        elif self.kind == "token_ablate":
            r = r - c * self.w_hat[self.b]
        else:
            r = self.donor_r1
        x[:, R1] = r
        self.lens_top1 = self.model.lens(r).argmax(-1)  # type: ignore[operator]
        return x


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    model, epoch = load_model(root / args.checkpoint, device)
    vocab, _probe, evals = load_data(root, 1, args.n_eval, args.seed)
    atomic = load_atomic(root, vocab)
    readout = lm.folded_readout(
        model.token_embedding.weight.detach(), model.ln_f.weight.detach(), centers=True
    )
    w_hat = F.normalize(readout, dim=-1)
    n_layers = 1 + len(model.blocks) * model.n_iterations

    def answer(ids: Tensor, edit: Edit | None = None) -> Tensor:
        return model.lens(model.residuals(ids, edit)[-1][:, 2]).argmax(-1)

    result: dict = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "iterations": model.n_iterations,
        "block_layers": len(model.blocks),
        "splits": {},
    }
    for split in args.splits:
        ex = evals[split]
        ids = ex.ids.to(device)
        b, t = ex.bridge.to(device), ex.tail.to(device)
        b_alt, t_alt, donor_ids = (
            x.to(device) for x in counterfactuals(ex, atomic, args.seed)
        )
        donor = model.residuals(donor_ids)
        clean = answer(ids)
        rows = []
        for layer in range(n_layers):
            row: dict = {"layer": layer}
            for kind in ("token_swap", "token_steer", "token_ablate", "full_patch"):
                edit = Edit(kind, layer, w_hat, b, b_alt, donor[layer][:, R1], model)
                out = answer(ids, edit)
                row[kind] = {
                    "answer_original": (out == t).float().mean().item(),
                    "answer_counterfactual": (out == t_alt).float().mean().item(),
                    "lens_reads_b": (edit.lens_top1 == b).float().mean().item(),
                    "lens_reads_b_alt": (edit.lens_top1 == b_alt).float().mean().item(),
                }
            rows.append(row)
        result["splits"][split] = {
            "n": len(ids),
            "clean_accuracy": (clean == t).float().mean().item(),
            "layers": rows,
        }
        print(
            f"\n{args.checkpoint} {split}: clean accuracy "
            f"{result['splits'][split]['clean_accuracy']:.0%}"
        )
        print(
            f"{'L':>3} {'swap->t2':>9} {'swap->t':>8} {'lens->b2':>9} "
            f"{'steer->t2':>10} "
            f"{'ablate->t':>10} {'patch->t2':>10} {'patch->t':>9}"
        )
        for r in rows:
            ts, ta, fp = r["token_swap"], r["token_ablate"], r["full_patch"]
            st = r["token_steer"]
            print(
                f"{r['layer']:>3} {ts['answer_counterfactual']:>9.0%} "
                f"{ts['answer_original']:>8.0%} {ts['lens_reads_b_alt']:>9.0%} "
                f"{st['answer_counterfactual']:>10.0%} "
                f"{ta['answer_original']:>10.0%} {fp['answer_counterfactual']:>10.0%} "
                f"{fp['answer_original']:>9.0%}"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", default="ext/kohli")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS[1:]))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
