"""Figures and summary tables from the looped_lm result JSONs.

uv run python -m looped_lm.report results/looped-lm
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots from the dataviz reference palette, in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#0b0b0b", "#8a8984"

TWOHOP_MAIN = {
    "r1_l8_2001": "vanilla 8L (ID stage)",
    "r2_l4_1201": "looped 4L×2 (ID stage)",
    "r2_l4_7101": "looped 4L×2 (OOD stage)",
}
# (ID-stage checkpoint, OOD-stage checkpoint or None, label); one color per model.
TWOHOP_DEPTH = [
    ("r1_l4_13501", None, "vanilla 4L"),
    ("r1_l8_2001", None, "vanilla 8L"),
    ("r2_l4_1201", "r2_l4_7101", "looped 4L×2"),
    ("r4_l4_401", "r4_l4_3501", "looped 4L×4"),
    ("r8_l4_301", "r8_l4_2901", "looped 4L×8"),
]
SPLIT_NAMES = {
    "train_inferred": "train",
    "test_inferred_iid": "test ID",
    "test_inferred_ood": "test OOD",
}
IFM_NAMES = {
    "dense-d112-336b": "D112 (112 distinct)",
    "dense-d28-336b": "D28 (28 distinct)",
    "dense-ouro-336b": "Ouro R28×4",
    "dense-ouro-raw-injection-336b": "Ouro R28×4 + inj.",
    "dense-middle-336b": "P8 R12×8 C8",
    "dense-huginn-336b": "Huginn P8 R12×8 C8 + inj.",
}


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def load(dir_: Path, names: dict[str, str]) -> dict[str, dict]:
    return {
        n: json.loads((dir_ / f"{n}.json").read_text())
        for n in names
        if (dir_ / f"{n}.json").exists()
    }


def twohop_lens_vs_probe(runs: dict[str, dict], out: Path) -> None:
    """Their Figure 4 (lens top-1) with the linear probe overlaid, per split."""
    targets = ["bridge@r1", "bridge@r2", "target@r2"]
    fig, axes = plt.subplots(
        len(targets), 3, figsize=(11, 8.5), sharex=True, sharey=True
    )
    for row, target in enumerate(targets):
        for col, split in enumerate(SPLIT_NAMES):
            ax = axes[row, col]
            style(ax)
            for i, (name, label) in enumerate(TWOHOP_MAIN.items()):
                if name not in runs:
                    continue
                layers = runs[name]["layers"]
                x = [r["layer"] for r in layers]
                vals = [r["targets"][target][split] for r in layers]
                ax.plot(
                    x,
                    [v["lens_top1"] for v in vals],
                    color=COLORS[i],
                    lw=2,
                    label=f"{label}: lens",
                )
                ax.plot(
                    x,
                    [v["probe_linear"] for v in vals],
                    color=COLORS[i],
                    lw=2,
                    ls=(0, (2, 2)),
                    label=f"{label}: probe",
                )
            ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(SPLIT_NAMES[split], fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"{target}\ntop-1 accuracy", fontsize=9, color=INK)
            if row == len(targets) - 1:
                ax.set_xlabel("effective layer (0 = embeddings)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "Two-hop models: logit lens (solid) vs linear probe (dashed)",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def twohop_geometry(runs: dict[str, dict], out: Path) -> None:
    """Per-position readout geometry on test ID, x = fraction of depth."""
    metrics = [
        (
            "relative visibility\n(1 = random direction)",
            lambda p: p["relative_visibility"],
        ),
        (
            "OMP R²(1) − null\n(variance from one token)",
            lambda p: p["omp_r2"][0] - p["omp_r2_null"][0],
        ),
        ("lens logit outliers\n(mean count)", lambda p: p["outliers_mean"]),
    ]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(10, 8), sharex=True)
    for col, pos in enumerate(["r1", "r2"]):
        for row, (label, fn) in enumerate(metrics):
            ax = axes[row, col]
            style(ax)
            for i, (id_name, ood_name, run_label) in enumerate(TWOHOP_DEPTH):
                for name, ls, suffix in (
                    (id_name, "-", "ID stage"),
                    (ood_name, ":", "OOD stage"),
                ):
                    if name not in runs:
                        continue
                    layers = runs[name]["layers"]
                    depth = len(layers) - 1
                    ax.plot(
                        [r["layer"] / depth for r in layers],
                        [fn(r["positions"][pos]) for r in layers],
                        color=COLORS[i],
                        lw=2,
                        ls=ls,
                        marker="o",
                        ms=3,
                        label=f"{run_label}, {suffix}",
                    )
            if row == 0:
                ax.set_title(f"position {pos}", fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
            if row == len(metrics) - 1:
                ax.set_xlabel("fraction of effective depth", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        "Two-hop readout geometry on test ID (solid: ID stage, dotted: OOD stage)",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def ifm_figure(runs: dict[str, dict], out: Path) -> None:
    panels = [
        ("KL(final ‖ logit lens), nats", lambda r: r["lens"]["kl_to_final"], True),
        (
            "KL(final ‖ ridge-tuned lens), nats",
            lambda r: r["tuned_ridge"]["kl_to_final"],
            True,
        ),
        (
            "logit lens top-1 = final top-1",
            lambda r: r["lens"]["agree_final_top1"],
            False,
        ),
        (
            "ridge-tuned lens top-1 = final top-1",
            lambda r: r["tuned_ridge"]["agree_final_top1"],
            False,
        ),
        (
            "share of raw ‖z‖² in the weakest-read quarter\nof readout directions "
            "(isotropic: 0.25)",
            lambda r: r["energy_bands_raw"][-1],
            False,
        ),
        (
            "relative visibility of the token-varying part\n(1 = random direction)",
            lambda r: r["relative_visibility_centered"],
            False,
        ),
        (
            "OMP R²(1) − Gaussian null\n(variance one token direction explains)",
            lambda r: r["omp_r2"][0] - r["omp_r2_null"][0],
            False,
        ),
        (
            "OMP R²(8) − Gaussian null",
            lambda r: r["omp_r2"][7] - r["omp_r2_null"][7],
            False,
        ),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True)
    for ax, (label, fn, log) in zip(axes.flat, panels, strict=True):
        style(ax)
        for i, (name, run_label) in enumerate(IFM_NAMES.items()):
            if name not in runs:
                continue
            rows = runs[name]["rows"]
            depth = len(rows) - 1
            x = [r["execution"] / depth for r in rows]
            y = [fn(r) for r in rows]
            if log:
                y = [max(v, 1e-2) for v in y]
            ax.plot(x, y, color=COLORS[i], lw=1.6, label=run_label)
        if log:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=9.5, color=INK)
        for boundary in (0.25, 0.5, 0.75):
            ax.axvline(boundary, color="#e6e5e0", lw=0.8, zorder=0)
    for ax in axes[-1]:
        ax.set_xlabel("fraction of executed depth (0 = embeddings)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "IFM LoopedLM (336B tokens): logit-lens readability by depth",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


TWOHOP_ALL = {
    "r1_l8_61": "vanilla 8L, train stage",
    "r1_l8_2001": "vanilla 8L, ID stage",
    "r1_l4_41": "vanilla 4L, train stage",
    "r1_l4_13501": "vanilla 4L, ID stage",
    "r2_l4_31": "looped 4L×2, train stage",
    "r2_l4_1201": "looped 4L×2, ID stage",
    "r2_l4_7101": "looped 4L×2, OOD stage",
    "r4_l4_41": "looped 4L×4, train stage",
    "r4_l4_401": "looped 4L×4, ID stage",
    "r4_l4_3501": "looped 4L×4, OOD stage",
    "r8_l4_71": "looped 4L×8, train stage",
    "r8_l4_301": "looped 4L×8, ID stage",
    "r8_l4_2901": "looped 4L×8, OOD stage",
}


def _first(values: list[float], threshold: float) -> str:
    return next((f"L{i}" for i, v in enumerate(values) if v >= threshold), "never")


def twohop_table(runs: dict[str, dict], split: str) -> None:
    """When each quantity becomes probe-decodable vs lens-readable, on one split."""
    print(f"\n### {SPLIT_NAMES[split]}\n")
    print(
        "| Model | answer acc. | bridge@r1: probe ≥ 90% / lens ≥ 90% | "
        "bridge@r1 lens: peak / last layer | target@r2: probe ≥ 90% / lens ≥ 90% "
        "| bridge@r2: best probe / best lens |"
    )
    print("|---|---|---|---|---|---|")
    for name, label in TWOHOP_ALL.items():
        if name not in runs:
            continue
        layers = runs[name]["layers"]
        last = len(layers) - 1

        def col(target: str, key: str, layers: list = layers) -> list[float]:
            return [r["targets"][target][split][key] for r in layers]

        b1_lens, b1_probe = (
            col("bridge@r1", "lens_top1"),
            col("bridge@r1", "probe_linear"),
        )
        t_lens, t_probe = (
            col("target@r2", "lens_top1"),
            col("target@r2", "probe_linear"),
        )
        b2_lens, b2_probe = (
            col("bridge@r2", "lens_top1"),
            col("bridge@r2", "probe_linear"),
        )
        peak = max(range(len(b1_lens)), key=b1_lens.__getitem__)
        print(
            f"| {label} (L{last}) | {t_lens[-1]:.0%} "
            f"| {_first(b1_probe, 0.9)} / {_first(b1_lens, 0.9)} "
            f"| {b1_lens[peak]:.0%} at L{peak} / {b1_lens[-1]:.0%} "
            f"| {_first(t_probe, 0.9)} / {_first(t_lens, 0.9)} "
            f"| {max(b2_probe):.0%} / {max(b2_lens):.0%} |"
        )


def _window(rows: list[dict], frac: float, fn: object, half: int = 3) -> float:
    """Mean of fn over executions within ±half of frac · depth.

    A single execution can land on a loop boundary, where injection models
    differ from their neighbours.
    """
    depth = len(rows) - 1
    centre = round(frac * depth)
    sel = rows[max(centre - half, 0) : centre + half + 1]
    return sum(fn(r) for r in sel) / len(sel)  # type: ignore[operator]


def ifm_table(runs: dict[str, dict]) -> None:
    fracs = (0.5, 0.75, 0.9)
    print("\n### IFM (each cell: mean over ±3 executions)\n")
    print(
        "| Model | final next-token top-1 | lens = final top-1 at "
        "50% / 75% / 90% depth | ridge-tuned lens = final top-1 at 50% / 75% / 90% "
        "| KL lens / tuned at 75% (nats) | raw energy in weakest quarter at 50% "
        "| OMP R²(1) − null, peak |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, label in IFM_NAMES.items():
        if name not in runs:
            continue
        rows = runs[name]["rows"]
        depth = len(rows) - 1

        def w(frac: float, fn: object, rows: list = rows) -> float:
            return _window(rows, frac, fn)

        lens = " / ".join(
            f"{w(f, lambda r: r['lens']['agree_final_top1']):.0%}" for f in fracs
        )
        tuned = " / ".join(
            f"{w(f, lambda r: r['tuned_ridge']['agree_final_top1']):.0%}" for f in fracs
        )
        kl = (
            f"{w(0.75, lambda r: r['lens']['kl_to_final']):.1f} / {w(0.75, lambda r: r[
                    'tuned_ridge'
                ]['kl_to_final']):.1f}"
        )
        dark = w(0.5, lambda r: r["energy_bands_raw"][-1])
        peak = max(rows, key=lambda r: r["omp_r2"][0] - r["omp_r2_null"][0])
        excess = peak["omp_r2"][0] - peak["omp_r2_null"][0]
        print(
            f"| {label} | {rows[-1]['lens']['next_token_top1']:.1%} | {lens} | {tuned} "
            f"| {kl} | {dark:.0%} | {excess:.3f} at {peak['execution'] / depth:.0%} |"
        )
    print("\nLens = final top-1 just before / after each Ouro loop boundary:\n")
    for name in ("dense-ouro-336b", "dense-ouro-raw-injection-336b"):
        if name in runs:
            rows = runs[name]["rows"]
            cells = ", ".join(
                f"{rows[e]['lens']['agree_final_top1']:.0%} → "
                f"{rows[e + 1]['lens']['agree_final_top1']:.0%}"
                for e in (28, 56, 84)
            )
            print(f"- {IFM_NAMES[name]}: {cells}")
    print("\nFirst depth (5% steps, window means) from which both standard models")
    print("beat every looped model at every later step:")
    standard = ("dense-d112-336b", "dense-d28-336b")
    looped = [n for n in IFM_NAMES if n not in standard and n in runs]
    for key in ("lens", "tuned_ridge"):
        first = None
        for step in range(1, 20):
            frac = step / 20

            def agree(r: dict, key: str = key) -> float:
                return r[key]["agree_final_top1"]

            lo = min(_window(runs[n]["rows"], frac, agree) for n in standard)
            hi = max(_window(runs[n]["rows"], frac, agree) for n in looped)
            if lo > hi and first is None:
                first = frac
            elif lo <= hi:
                first = None
        print(f"- {key}: {first}")


def twohop_omp_summary(runs: dict[str, dict]) -> None:
    """Middle-half-of-depth OMP statistics at r1 and r2 on test ID."""
    print("\n### Two-hop OMP, middle half of depth (layers at 25 to 75% of depth)\n")
    print(
        "| Model | r1: R²(1) − null | r1: first atom = bridge | "
        "r2: R²(1) − null | r2: first atom = answer |"
    )
    print("|---|---|---|---|---|")
    for name, label in TWOHOP_ALL.items():
        if name not in runs or "train" in label:
            continue
        layers = runs[name]["layers"]
        depth = len(layers) - 1
        mid = [r for r in layers if 0.25 <= r["layer"] / depth <= 0.75]

        def mean(pos: str, fn: object, mid: list = mid) -> float:
            return sum(fn(r["positions"][pos]) for r in mid) / len(mid)  # type: ignore[operator]

        def excess(p: dict) -> float:
            return p["omp_r2"][0] - p["omp_r2_null"][0]

        print(
            f"| {label} | {mean('r1', excess):.2f} "
            f"| {mean('r1', lambda p: p['omp_first_atom']['bridge']):.0%} "
            f"| {mean('r2', excess):.2f} "
            f"| {mean('r2', lambda p: p['omp_first_atom']['tail']):.0%} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    twohop = load(args.results / "twohop", dict(TWOHOP_ALL))
    if twohop:
        twohop_lens_vs_probe(twohop, figures / "twohop-lens-vs-probe.png")
        twohop_geometry(twohop, figures / "twohop-geometry.png")
        for split in SPLIT_NAMES:
            twohop_table(twohop, split)
        twohop_omp_summary(twohop)
    ifm = load(args.results / "ifm", IFM_NAMES)
    if ifm:
        ifm_figure(ifm, figures / "ifm.png")
        ifm_table(ifm)


if __name__ == "__main__":
    main()
