"""Per-position critical-layer ablation (the writeup's "Sequential Ablation").

A one-step-per-layer algorithm needs the intermediate at ``<op>`` position j
to exist by a specific *critical* layer and never again after it. So:

- zeroing the residual at that position from ``critical + 1`` onward should
  leave accuracy intact (the state was already consumed), while
- zeroing from ``critical`` onward should hurt (the state is destroyed at the
  layer that needs it).

A model that is not sequential (the writeup's Model B) survives both.

``--sweep`` prints the full picture instead: for every element and ``<op>``
position, accuracy when it is zeroed from each layer onward, plus "all
element positions" / "all <op> positions" rows. A sequential model consumes
one operand per layer (the all-elements row climbs gradually); a parallel
one consumes them all at once (a single cliff, early).

Critical layers default to the layer where the logit lens reads each
intermediate best (the staircase's peak per position). For a model with no
staircase, pass them explicitly — the writeup applies Model A's layers to
Model B:

    uv run python -m lego.ablate_critical_layer \
        --checkpoint hf:lego/std_96d_6h_8L_kp2_s42_curriculum_FSL/step_97656.pt
    uv run python -m lego.ablate_critical_layer \
        --checkpoint hf:lego/std_96d_6h_8L_kp2_s42_fsl_only/step_156250.pt \
        --critical-layers 1 2 3 4 5
"""

import argparse

import torch
from torch import Tensor

from lego.analyze_logit_lens import analyze_op_positions
from lego.generator import generate_fixed_dataset
from lego.model import AnyModel
from lego.tokenizer import answer_position, encode
from lego.training import load_model


def op_positions(k: int) -> list[int]:
    """The ``<op>`` positions holding t[1]..t[k-1] (pos 2(j+1) for j = 1..k-1).

    t[k] lives at the ``<predict>`` position, which is the answer readout and
    is never ablated.
    """
    return [2 * (j + 1) for j in range(1, k)]


@torch.no_grad()
def accuracy_with_ablation(
    model: AnyModel,
    input_ids: Tensor,
    k: int,
    zero_from: dict[int, int],
) -> float:
    """Answer accuracy with ``zero_from[pos]`` = first layer whose output is zeroed.

    The residual at ``pos`` is zeroed after every layer ``>= zero_from[pos]``,
    so layer ``zero_from[pos]`` still *reads* the position (the hook runs on
    its output) but no later layer can.
    """

    def hook(x: Tensor, layer_idx: int) -> Tensor:
        positions = [p for p, cutoff in zero_from.items() if layer_idx >= cutoff]
        if not positions:
            return x
        x = x.clone()
        x[:, positions, :] = 0.0
        return x

    logits, _ = model.forward_with_layer_hooks(input_ids, hook)
    ans_pos = answer_position(k)
    preds = logits[:, ans_pos - 1, :].argmax(dim=-1)
    return (preds == input_ids[:, ans_pos]).float().mean().item()


def element_positions(k: int) -> list[int]:
    """Positions of the start element and the k operands (pos 2j+1 for j = 0..k)."""
    return [2 * j + 1 for j in range(k + 1)]


def print_layer_sweep(
    model: AnyModel, input_ids: Tensor, k: int, n_layers: int
) -> None:
    """Accuracy with each position zeroed from layer L onward, for every L."""
    rows: list[tuple[str, list[int]]] = [("e0 (pos 1)", [1])]
    rows += [(f"g{j} (pos {2 * j + 1})", [2 * j + 1]) for j in range(1, k + 1)]
    rows += [(f"t[{j}] <op> (pos {2 * j + 2})", [2 * j + 2]) for j in range(1, k)]
    rows += [
        ("all element positions", element_positions(k)),
        ("all <op> positions", op_positions(k)),
    ]
    print("\nAccuracy with the position zeroed from layer L onward (100% = consumed)")
    print(
        f"{'position':26s}" + "".join(f"{'L' + str(li):>7s}" for li in range(n_layers))
    )
    for label, pos in rows:
        accs = [
            accuracy_with_ablation(model, input_ids, k, dict.fromkeys(pos, li))
            for li in range(n_layers)
        ]
        print(f"{label:26s}" + "".join(f"{a:>7.1%}" for a in accs))


# A peak below this is not a staircase step: chance is 1/6 for S3 and the
# writeup's weakest real step (t[5] in Model A) peaks at 48%.
MIN_PEAK = 1 / 3


def critical_layers_from_lens(heatmap_op: Tensor, k: int) -> dict[int, int | None]:
    """Layer where the lens reads t[j] best at its <op> position, per position.

    ``None`` where the peak never clears :data:`MIN_PEAK` (no staircase step).
    """
    result: dict[int, int | None] = {}
    for j, pos in enumerate(op_positions(k)):
        column = heatmap_op[:, j]
        peak = int(column.argmax().item())
        result[pos] = peak if column[peak].item() >= MIN_PEAK else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Critical-layer ablation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Local checkpoint path or hf:<relpath> into the public dataset",
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--critical-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="L",
        help="Critical layer for t[1]..t[k-1] (k-1 values). Default: the layer "
        "where the logit lens peaks; required when a position has no peak.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Print accuracy for every position zeroed from every layer instead "
        "of the critical-layer table.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(args.checkpoint, device)
    k = args.k
    n_layers = model_config.n_layers
    positions = op_positions(k)

    arch = "WS" if model_config.weight_shared else "Std"
    print(f"Model: {arch} {model_config.dim}d/{model_config.n_heads}h/{n_layers}L")
    print(f"k={k}, n_examples={args.n_examples}, seed={args.seed}")

    examples = generate_fixed_dataset(k, args.n_examples, seed=args.seed)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples], dtype=torch.long, device=device
    )

    if args.sweep:
        print_layer_sweep(model, input_ids, k, n_layers)
        return

    if args.critical_layers is not None:
        if len(args.critical_layers) != len(positions):
            parser.error(
                f"--critical-layers needs {len(positions)} values for k={k}, "
                f"got {len(args.critical_layers)}"
            )
        critical = dict(zip(positions, args.critical_layers, strict=True))
        source = "given"
    else:
        found = critical_layers_from_lens(
            analyze_op_positions(model, examples, device), k
        )
        missing = [p for p, li in found.items() if li is None]
        if missing:
            parser.error(
                "no logit-lens staircase at <op> positions "
                f"{missing} (peak below {MIN_PEAK:.0%}); pass --critical-layers"
            )
        critical = {p: li for p, li in found.items() if li is not None}
        source = "from logit lens"

    baseline = accuracy_with_ablation(model, input_ids, k, {})
    print(f"\nBaseline accuracy: {baseline:.1%}")

    labels = [f"t[{j + 1}] (pos {p})" for j, p in enumerate(positions)]
    header = "".join(f"{lbl:>16s}" for lbl in labels) + f"{'All':>10s}"
    print(f"\nCritical layers ({source}):")
    print(f"{'':32s}{header}")
    print(
        f"{'Critical layer':32s}"
        + "".join(f"{'L' + str(critical[p]):>16s}" for p in positions)
        + f"{'—':>10s}"
    )

    def row(label: str, zero_from: dict[int, int]) -> None:
        per_pos = [
            accuracy_with_ablation(model, input_ids, k, {p: zero_from[p]})
            for p in positions
        ]
        together = accuracy_with_ablation(model, input_ids, k, zero_from)
        print(
            f"{label:32s}"
            + "".join(f"{a:>16.1%}" for a in per_pos)
            + f"{together:>10.1%}"
        )

    row("Zero from critical+1", {p: critical[p] + 1 for p in positions})
    row("Zero from critical", {p: critical[p] for p in positions})
    # Every layer: a model that never reads the <op> positions stays at 100%
    # here, which is what distinguishes "ablation-robust" from "unused".
    row("Zero from L0 (every layer)", dict.fromkeys(positions, 0))

    # The writeup's original footnote zeroed t[1]..t[k], i.e. the <predict>
    # readout too, which takes any model to chance and so says nothing.
    with_predict = accuracy_with_ablation(
        model, input_ids, k, dict.fromkeys([*positions, answer_position(k) - 1], 0)
    )
    print(f"\n(+ <predict> position zeroed at every layer: {with_predict:.1%})")


if __name__ == "__main__":
    main()
