"""Render figures/staircase.png: <op>-position logit lens for Model A and Model B.

Usage: uv run python scripts/make_readme_figure.py [--output figures/staircase.png]
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from lego.analyze_logit_lens import analyze_op_positions
from lego.generator import generate_fixed_dataset
from lego.training import load_model

MODELS = [
    (
        "Model A — answer-only → full-sequence",
        "hf:lego/std_96d_6h_8L_kp2_s42_curriculum_FSL/step_97656.pt",
    ),
    (
        "Model B — full-sequence only",
        "hf:lego/std_96d_6h_8L_kp2_s42_fsl_only/step_156250.pt",
    ),
]
# Single-hue sequential ramp (light -> dark), so the heatmap encodes magnitude only.
BLUES = [
    "#f4f8fd",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/staircase.png")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-examples", type=int, default=500)
    args = parser.parse_args()

    matplotlib.rcParams["font.family"] = "sans-serif"
    device = torch.device("cpu")
    examples = generate_fixed_dataset(args.k, args.n_examples, seed=999)
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUES)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    im = None
    for ax, (title, ckpt) in zip(axes, MODELS, strict=True):
        model, config = load_model(ckpt, device)
        heat = analyze_op_positions(model, examples, device).numpy()
        n_layers = config.n_layers
        im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(range(args.k))
        ax.set_xticklabels([f"t[{j + 1}]" for j in range(args.k)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_xlabel("intermediate state at its <op> position")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(n_layers):
            for j in range(args.k):
                v = float(heat[i, j])
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 0.55 else "#333333",
                )
    axes[0].set_ylabel("layer")
    assert im is not None
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("logit-lens top-1 match rate (chance = 17%)")
    cbar.outline.set_visible(False)
    fig.suptitle(
        "Logit lens at <op> positions, k = 6: "
        "one intermediate per layer (A) vs none until the answer (B)",
        fontsize=11,
        x=0.02,
        ha="left",
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
