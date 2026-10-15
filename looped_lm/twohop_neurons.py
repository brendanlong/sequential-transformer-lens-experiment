"""Are the two-hop models' MLP neurons interpretable, and does looping change that?

Each neuron is an MLP hidden unit (after GELU). For every block *execution*
(so a looped model's shared neuron is measured once per iteration) and each
input position, the neuron's activations over training compositions are
explained by each known variable (head ``h``, relations ``r1``/``r2``, bridge
``b = r1(h)``, answer ``t = r2(b)``) with an adjusted eta²: the share of the
activation variance that the variable's categories account for, corrected for
the share an unrelated variable with as many categories would get by chance
((k − 1)/(N − k) of the within-group variance).

A neuron execution is

- **inactive** if its activation hardly varies (std below ``--min-std``);
- **one-variable** if its best variable explains at least ``--clean`` of the
  variance;
- otherwise **mixed / unexplained** by any single variable.

Position ``h`` only ever sees the head token, so it is skipped; ``r1`` sees
``h, r1, b``; ``r2`` sees all five. For looped models, each neuron's best
(position, variable) at one iteration is compared with the next iteration's.

Also keeps the top-activating examples of a few clean neurons per model for the
report.

    uv run python -m looped_lm.twohop_neurons --root ext/kohli \
        --checkpoint checkpoints/systematicity/r4_l4/checkpoint_epoch_401.pt \
        --out OUT.json
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from looped_lm.twohop import Examples, load_data, load_model

POSITIONS = {
    "r1": (1, ("h", "r1", "bridge")),
    "r2": (2, ("h", "r1", "r2", "bridge", "answer")),
}


def variables(ex: Examples) -> dict[str, Tensor]:
    return {
        "h": ex.ids[:, 0],
        "r1": ex.ids[:, 1],
        "r2": ex.ids[:, 2],
        "bridge": ex.bridge,
        "answer": ex.tail,
    }


@torch.no_grad()
def executions(model: torch.nn.Module, ids: Tensor, batch: int = 8192):  # noqa: ANN201
    """Yield (execution, MLP hidden activations at positions 1-2) block by block.

    Keeps only the current residual stream, so memory does not grow with depth.
    """
    pos = torch.arange(3, device=ids.device)
    x = model.token_embedding(ids) + model.position_embedding(pos)
    k = 0
    for _ in range(model.n_iterations):
        for block in model.blocks:
            acts, nxt = [], []
            for s in range(0, len(ids), batch):
                xb = x[s : s + batch]
                h = xb + block.attn(block.ln_1(xb))
                a = F.gelu(block.mlp.c_fc(block.ln_2(h)), approximate="tanh")
                nxt.append(h + block.mlp.c_proj(a))
                acts.append(a[:, 1:3].half())  # float32 is recovered per chunk
            x = torch.cat(nxt)
            k += 1
            yield k, torch.cat(acts)


def adjusted_eta2(acts: Tensor, labels: Tensor, n_cat: int) -> Tensor:
    """Per-neuron adjusted eta² of a categorical variable; acts is (N, neurons)."""
    n = acts.shape[0]
    counts = torch.zeros(n_cat, device=acts.device).index_add_(
        0, labels, torch.ones(n, device=acts.device)
    )
    sums = torch.zeros(n_cat, acts.shape[1], device=acts.device).index_add_(
        0, labels, acts
    )
    present = counts > 0
    k = int(present.sum())
    mean = acts.mean(0)
    sst = (acts - mean).pow(2).sum(0)
    ssb = (sums[present].pow(2) / counts[present, None]).sum(0) - n * mean.pow(2)
    ssw = sst - ssb
    adj = (ssb - (k - 1) * ssw / (n - k)) / sst.clamp_min(1e-12)
    return adj.clamp_min(0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--root", default="ext/kohli")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=60000)
    parser.add_argument("--clean", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.01)
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    model, epoch = load_model(root / args.checkpoint, device)
    vocab, probe_set, _evals = load_data(root, args.n, 10, args.seed)
    names = {i: t for t, i in vocab.items()}
    var = {k: v.to(device) for k, v in variables(probe_set).items()}
    n_hidden = model.blocks[0].mlp.c_fc.weight.shape[1]
    n_vocab = len(vocab)
    tops: dict[tuple[int, int], list[int]] = {}  # (execution, neuron) -> top examples

    rows = []
    best_label: list[list[str]] = []  # per execution, per neuron: "pos:var" or status
    for e1, a in executions(model, probe_set.ids.to(device)):
        e = e1 - 1
        row: dict = {"execution": e1, "block": e % len(model.blocks), "positions": {}}
        labels_e = ["inactive"] * n_hidden
        best_overall = torch.full((n_hidden,), -1.0, device=device)
        for pos_name, (col, var_names) in POSITIONS.items():
            x = a[:, col - 1]
            std = torch.cat(
                [x[:, c : c + 512].float().std(0) for c in range(0, n_hidden, 512)]
            )
            eta = torch.cat(
                [
                    torch.stack(
                        [
                            adjusted_eta2(x[:, c : c + 512].float(), var[v], n_vocab)
                            for v in var_names
                        ]
                    )
                    for c in range(0, n_hidden, 512)
                ],
                dim=1,
            )
            best, which = eta.max(0)
            active = std >= args.min_std
            clean = active & (best >= args.clean)
            counts = {
                "inactive": int((~active).sum()),
                "mixed": int((active & ~clean).sum()),
                **{
                    v: int((clean & (which == i)).sum())
                    for i, v in enumerate(var_names)
                },
            }
            row["positions"][pos_name] = {
                "counts": counts,
                "best_eta2_hist": torch.histc(best[active], bins=10, min=0, max=1)
                .int()
                .tolist(),
                "median_best_eta2": best[active].median().item()
                if active.any()
                else 0.0,
            }
            top_idx = torch.cat(
                [
                    x[:, c : c + 512].topk(8, dim=0).indices
                    for c in range(0, n_hidden, 512)
                ],
                dim=1,
            )  # (8, neurons)
            for j in torch.nonzero(active & (best > best_overall)).flatten().tolist():
                tops[(e, j)] = [col, *top_idx[:, j].tolist()]
                labels_e[j] = (
                    f"{pos_name}:{var_names[which[j]]}"
                    if best[j] >= args.clean
                    else f"{pos_name}:mixed"
                )
            best_overall = torch.where(
                active, torch.maximum(best_overall, best), best_overall
            )
        rows.append(row)
        best_label.append(labels_e)
        del a, x
        torch.cuda.empty_cache()
        print(f"exec {e1}: {row['positions']['r2']['counts']}", flush=True)

    # Same neuron across iterations (looped models only).
    consistency = []
    n_blocks = len(model.blocks)
    for e in range(n_blocks, len(best_label)):
        prev, cur = best_label[e - n_blocks], best_label[e]
        both = [
            i
            for i in range(n_hidden)
            if ":" in prev[i]
            and ":" in cur[i]
            and not prev[i].endswith("mixed")
            and not cur[i].endswith("mixed")
        ]
        same = sum(prev[i] == cur[i] for i in both)
        consistency.append(
            {
                "execution": e + 1,
                "clean_in_both": len(both),
                "same_job": same,
                "changed_job": len(both) - same,
            }
        )

    # A few clean neurons with their top activations, for the report.
    rng = random.Random(args.seed)
    examples = []
    clean_ids = [
        (e, j, lab)
        for e, labs in enumerate(best_label)
        for j, lab in enumerate(labs)
        if ":" in lab and not lab.endswith("mixed")
    ]
    for e, j, lab in rng.sample(clean_ids, min(args.n_examples, len(clean_ids))):
        _col, *top = tops[(e, j)]
        examples.append(
            {
                "execution": e + 1,
                "neuron": j,
                "label": lab,
                "top": [
                    {
                        "tokens": [names[t] for t in probe_set.ids[i].tolist()],
                        "bridge": names[int(probe_set.bridge[i])],
                        "answer": names[int(probe_set.tail[i])],
                    }
                    for i in top
                ],
            }
        )

    out = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "iterations": model.n_iterations,
        "block_layers": n_blocks,
        "hidden": n_hidden,
        "n": len(probe_set.ids),
        "clean_threshold": args.clean,
        "executions": rows,
        "consistency": consistency,
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
