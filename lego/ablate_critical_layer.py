"""Per-position critical-layer ablation (the writeup's "Sequential Ablation").

A one-step-per-layer algorithm needs the intermediate at ``<op>`` position j
to exist by a specific *critical* layer and never again after it. So:

- zeroing the residual at that position from ``critical + 1`` onward should
  leave accuracy intact (the state was already consumed), while
- zeroing from ``critical`` onward should hurt (the state is destroyed at the
  layer that needs it).

A model that is not sequential (the writeup's Model B) survives both.

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

    for label, offset in (("Zero from critical+1", 1), ("Zero from critical", 0)):
        per_pos = [
            accuracy_with_ablation(model, input_ids, k, {p: critical[p] + offset})
            for p in positions
        ]
        together = accuracy_with_ablation(
            model, input_ids, k, {p: critical[p] + offset for p in positions}
        )
        print(
            f"{label:32s}"
            + "".join(f"{a:>16.1%}" for a in per_pos)
            + f"{together:>10.1%}"
        )

    # The writeup's "ALL <op> positions zeroed from any layer" control zeroed
    # positions 4..2(k+1) — the t[1]..t[k] positions, which *include* the
    # <predict> readout — so it destroys accuracy trivially for any model.
    # The intermediates-only variant is the informative one.
    with_predict = accuracy_with_ablation(
        model, input_ids, k, {p: 0 for p in [*positions, answer_position(k) - 1]}
    )
    without_predict = accuracy_with_ablation(
        model, input_ids, k, {p: 0 for p in positions}
    )
    print(
        f"\nZero <op> positions t[1]..t[{k}] at every layer "
        f"(includes <predict>; the writeup's 'all <op>' control): {with_predict:.1%}"
    )
    print(
        f"Zero <op> positions t[1]..t[{k - 1}] at every layer "
        f"(intermediates only, <predict> intact): {without_predict:.1%}"
    )


if __name__ == "__main__":
    main()
