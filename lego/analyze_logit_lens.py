"""Logit lens analysis for the group composition model.

Projects the residual stream at each layer through the model's final norm and
tied unembedding and asks whether it decodes to the intermediate composition
states. Three views:

1. ``<op>`` positions across layers — the **staircase**. The ``<op>`` token
   after operand j is the first position that has causally seen everything
   needed to compute trajectory[j]; a one-step-per-layer algorithm shows
   t[j] appearing at layer ~j. This is the table in the writeup.
2. ``<predict>`` position across layers — does the answer position walk
   through successive trajectory states?
3. Element positions at every layer — do operand positions hold running
   state? (Consistently no: intermediates live at ``<op>`` positions.)

Usage:
    uv run python -m lego.analyze_logit_lens \
        --checkpoint hf:lego/std_96d_6h_8L_kp2_s42_curriculum_FSL/step_97656.pt
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from lego.generator import ELEMENTS, S3Example, generate_fixed_dataset
from lego.model import AnyModel
from lego.tokenizer import answer_position, element_token, encode
from lego.training import load_model

# A decode function maps (hidden_state, layer_idx) -> logits. The logit lens
# ignores layer_idx; a per-layer probe such as a tuned lens would use it.
DecodeFn = Callable[[Tensor, int], Tensor]


@torch.no_grad()
def logit_lens(
    residual: Tensor,
    final_norm: torch.nn.Module,
    embedding_weight: Tensor,
) -> Tensor:
    """Apply logit lens: norm → unembedding → logits."""
    return F.linear(final_norm(residual), embedding_weight)


def make_logit_lens_fn(model: AnyModel) -> DecodeFn:
    """Create a logit lens DecodeFn from a model."""
    final_norm = model.final_norm
    emb_weight = model.tok_emb.weight

    def decode(hidden_state: Tensor, _layer_idx: int) -> Tensor:
        return logit_lens(hidden_state, final_norm, emb_weight)

    return decode


def _targets(examples: list[S3Example], traj_idx: int, device: torch.device) -> Tensor:
    return torch.tensor(
        [element_token(ex.trajectory[traj_idx]) for ex in examples], device=device
    )


@torch.no_grad()
def probe_predict_position(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Top-1 match rate with each trajectory state at the <predict> position.

    Returns (n_layers, k+1): heatmap[L, j] is the fraction of examples where
    layer L's top-1 equals trajectory[j].
    """
    predict_pos = answer_position(k) - 1
    heatmap = torch.zeros(len(residuals), k + 1)
    for layer_idx, residual in enumerate(residuals):
        top1 = decode_fn(residual[:, predict_pos, :], layer_idx).argmax(dim=-1)
        for traj_pos in range(k + 1):
            targets = _targets(examples, traj_pos, device)
            heatmap[layer_idx, traj_pos] = (top1 == targets).float().mean().item()
    return heatmap


@torch.no_grad()
def probe_element_positions(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Top-1 match rate with trajectory[j] at element position j (pos 2j+1).

    Returns (n_layers, k+1).
    """
    heatmap = torch.zeros(len(residuals), k + 1)
    for layer_idx, residual in enumerate(residuals):
        for j in range(k + 1):
            top1 = decode_fn(residual[:, 2 * j + 1, :], layer_idx).argmax(dim=-1)
            targets = _targets(examples, j, device)
            heatmap[layer_idx, j] = (top1 == targets).float().mean().item()
    return heatmap


@torch.no_grad()
def probe_op_positions(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Top-1 match rate with trajectory[j+1] at the <op> position after operand j+1.

    Returns (n_layers, k): heatmap[L, j] is the fraction of examples where
    layer L's top-1 at position 2*(j+2) equals trajectory[j+1]. Column k-1
    (t[k]) is the <predict> position, i.e. the answer readout.
    """
    heatmap = torch.zeros(len(residuals), k)
    for layer_idx, residual in enumerate(residuals):
        for j in range(k):
            top1 = decode_fn(residual[:, 2 * (j + 2), :], layer_idx).argmax(dim=-1)
            targets = _targets(examples, j + 1, device)
            heatmap[layer_idx, j] = (top1 == targets).float().mean().item()
    return heatmap


@torch.no_grad()
def probe_op_probability_mass(
    residuals: list[Tensor],
    examples: list[S3Example],
    decode_fn: DecodeFn,
    k: int,
    device: torch.device,
) -> Tensor:
    """Like :func:`probe_op_positions` but mean softmax mass on the target token."""
    heatmap = torch.zeros(len(residuals), k)
    for layer_idx, residual in enumerate(residuals):
        for j in range(k):
            probs = F.softmax(decode_fn(residual[:, 2 * (j + 2), :], layer_idx), -1)
            targets = _targets(examples, j + 1, device)
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            heatmap[layer_idx, j] = target_probs.mean().item()
    return heatmap


@torch.no_grad()
def _residuals(
    model: AnyModel, examples: list[S3Example], device: torch.device
) -> list[Tensor]:
    input_ids = torch.tensor(
        [encode(ex) for ex in examples], dtype=torch.long, device=device
    )
    _logits, residuals = model.forward_with_residuals(input_ids)
    return residuals


@torch.no_grad()
def analyze_predict_position(
    model: AnyModel, examples: list[S3Example], device: torch.device
) -> Tensor:
    """Logit lens at the <predict> position across layers."""
    k = len(examples[0].ops)
    residuals = _residuals(model, examples, device)
    return probe_predict_position(
        residuals, examples, make_logit_lens_fn(model), k, device
    )


@torch.no_grad()
def analyze_element_positions(
    model: AnyModel, examples: list[S3Example], device: torch.device
) -> Tensor:
    """Logit lens at element positions across layers; see probe_element_positions."""
    k = len(examples[0].ops)
    residuals = _residuals(model, examples, device)
    return probe_element_positions(
        residuals, examples, make_logit_lens_fn(model), k, device
    )


@torch.no_grad()
def analyze_op_positions(
    model: AnyModel, examples: list[S3Example], device: torch.device
) -> Tensor:
    """Logit lens at <op> positions across layers (the staircase)."""
    k = len(examples[0].ops)
    residuals = _residuals(model, examples, device)
    return probe_op_positions(residuals, examples, make_logit_lens_fn(model), k, device)


@torch.no_grad()
def analyze_op_probability_mass(
    model: AnyModel, examples: list[S3Example], device: torch.device
) -> Tensor:
    """Softmax mass on the correct intermediate at <op> positions across layers."""
    k = len(examples[0].ops)
    residuals = _residuals(model, examples, device)
    return probe_op_probability_mass(
        residuals, examples, make_logit_lens_fn(model), k, device
    )


def print_heatmap(
    heatmap: Tensor,
    title: str,
    row_label: str = "Layer",
    col_labels: list[str] | None = None,
    fmt: str = ".0%",
) -> None:
    """Print a text heatmap."""
    n_rows, n_cols = heatmap.shape
    if col_labels is None:
        col_labels = [f"t[{j}]" for j in range(n_cols)]

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    col_width = max(6, max(len(c) for c in col_labels) + 1)
    header = f"{row_label:>6s}" + "".join(f"{c:>{col_width}s}" for c in col_labels)
    print(header)
    print("-" * len(header))

    for i in range(n_rows):
        row_vals = "".join(
            f"{heatmap[i, j].item():{col_width}{fmt}}" for j in range(n_cols)
        )
        print(f"  L{i:<3d} {row_vals}")


def draw_heatmap(
    ax: object,
    data: object,
    title: str,
    col_labels: list[str],
    row_labels: list[str],
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Draw an annotated heatmap on a matplotlib axis."""
    import numpy as np

    arr = np.asarray(data)
    n_rows, n_cols = arr.shape

    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)  # type: ignore[union-attr]
    ax.set_title(title, fontsize=9)  # type: ignore[union-attr]
    ax.set_xlabel("State")  # type: ignore[union-attr]
    ax.set_ylabel("Layer")  # type: ignore[union-attr]
    ax.set_xticks(range(n_cols))  # type: ignore[union-attr]
    ax.set_xticklabels(col_labels, fontsize=7)  # type: ignore[union-attr]
    ax.set_yticks(range(n_rows))  # type: ignore[union-attr]
    ax.set_yticklabels(row_labels, fontsize=7)  # type: ignore[union-attr]

    for i in range(n_rows):
        for j in range(n_cols):
            val = float(arr[i, j])
            ax.text(  # type: ignore[union-attr]
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if val < 0.5 else "black",
            )

    ax.figure.colorbar(im, ax=ax, shrink=0.8)  # type: ignore[union-attr]


def save_heatmap_figure(
    heatmap_predict: Tensor,
    heatmap_op: Tensor,
    heatmap_op_mass: Tensor,
    heatmap_elem: Tensor,
    k: int,
    n_layers: int,
    output_path: Path,
) -> None:
    """Save a matplotlib figure with the four logit-lens heatmaps."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes_arr = plt.subplots(1, 4, figsize=(24, 5))
    assert isinstance(axes_arr, np.ndarray)

    traj_labels = [f"traj[{j}]" for j in range(k + 1)]
    op_labels = [f"p{2 * (j + 2)}→t[{j + 1}]" for j in range(k)]
    layer_labels = [f"L{i}" for i in range(n_layers)]

    panels: list[tuple[object, object, str, list[str]]] = [
        (
            axes_arr[0],
            heatmap_op.numpy(),
            "Logit lens at <op> positions\n(top-1 — staircase pattern)",
            op_labels,
        ),
        (
            axes_arr[1],
            heatmap_op_mass.numpy(),
            "Logit lens at <op> positions\n(probability mass)",
            op_labels,
        ),
        (
            axes_arr[2],
            heatmap_predict.numpy(),
            "Logit lens at <predict> pos\n(top-1 match rate)",
            traj_labels,
        ),
        (
            axes_arr[3],
            heatmap_elem.numpy(),
            "Logit lens at element positions\n(top-1 match rate)",
            traj_labels,
        ),
    ]

    for ax, data, title, xlabels in panels:
        draw_heatmap(ax, data, title, xlabels, layer_labels)

    fig.suptitle(f"Logit Lens Analysis (k={k})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens analysis for the group composition model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Local checkpoint path or hf:<relpath> into the public dataset",
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the heatmap figure (default: no figure)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, model_config = load_model(args.checkpoint, device)

    n_layers = model_config.n_layers
    arch = "WS" if model_config.weight_shared else "Std"
    print(f"Model: {arch} {model_config.dim}d/{model_config.n_heads}h/{n_layers}L")

    print(f"Generating {args.n_examples} examples with k={args.k}")
    examples = generate_fixed_dataset(args.k, args.n_examples, seed=args.seed)

    print("\nSample examples:")
    for ex in examples[:3]:
        print("  " + " → ".join(ELEMENTS[s] for s in ex.trajectory))

    op_col_labels = [f"p{2 * (j + 2)}→t[{j + 1}]" for j in range(args.k)]

    heatmap_op = analyze_op_positions(model, examples, device)
    print_heatmap(
        heatmap_op,
        f"Logit lens at <op> positions (top-1 match rate, k={args.k})",
        col_labels=op_col_labels,
    )

    heatmap_op_mass = analyze_op_probability_mass(model, examples, device)
    print_heatmap(
        heatmap_op_mass,
        f"Logit lens at <op> positions (probability mass, k={args.k})",
        col_labels=op_col_labels,
    )

    heatmap_predict = analyze_predict_position(model, examples, device)
    print_heatmap(
        heatmap_predict,
        f"Logit lens at <predict> position (top-1 match rate, k={args.k})",
    )

    heatmap_elem = analyze_element_positions(model, examples, device)
    print_heatmap(
        heatmap_elem,
        f"Logit lens at element positions (top-1 match rate, k={args.k})",
        col_labels=["start"] + [f"op{j}" for j in range(1, args.k + 1)],
    )

    print(f"\n{'=' * 60}")
    print("Staircase summary (first layer where t[j] exceeds 50% at its <op> position)")
    print(f"{'=' * 60}")
    for j in range(args.k):
        pos = 2 * (j + 2)
        first_layer = next(
            (li for li in range(n_layers) if heatmap_op[li, j].item() > 0.5), None
        )
        final_val = heatmap_op[n_layers - 1, j].item()
        if first_layer is not None:
            print(
                f"  t[{j + 1}] at pos {pos}: first >50% at L{first_layer}, "
                f"final={final_val:.0%}"
            )
        else:
            print(f"  t[{j + 1}] at pos {pos}: never >50% (final={final_val:.0%})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_heatmap_figure(
            heatmap_predict,
            heatmap_op,
            heatmap_op_mass,
            heatmap_elem,
            args.k,
            n_layers,
            output_path,
        )


if __name__ == "__main__":
    main()
