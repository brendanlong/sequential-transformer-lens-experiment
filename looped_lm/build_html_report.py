# ruff: noqa: E501, RUF001  (mostly HTML prose with typographic dashes)
"""Build the HTML report from results/looped-lm, with every number read from the JSONs.

    uv run python -m looped_lm.build_html_report results/looped-lm \
        --template TEMPLATE.html -o OUT.html

Published at https://looped-lens-a783c534.surge.sh (Lion Reader article
01a10db5-f7a7-7f88-b580-9a710c86e7e7, Research Results collection).
"""

import argparse
import html
import json
from collections.abc import Callable
from pathlib import Path

from looped_lm import lens_grid_html as lg

DOMAIN = "looped-lens-a783c534.surge.sh"
REPO = "https://github.com/brendanlong/sequential-transformer-lens-experiment"

# Light-theme literals (survive saving as an article); CSS classes swap in theme vars.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a8984"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2b9a2b", "#a3a29b"]
GRAY = 6

TWOHOP_ID = [
    ("r1_l8_2001", "vanilla 8L"),
    ("r2_l4_1201", "looped 4L×2"),
    ("r4_l4_401", "looped 4L×4"),
    ("r8_l4_301", "looped 4L×8"),
]
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
IFM = [
    ("dense-d112-336b", "D112 (standard, 112 layers)"),
    ("dense-d28-336b", "D28 (standard, 28 layers)"),
    ("dense-ouro-336b", "Ouro: 28 layers × 4"),
    ("dense-ouro-raw-injection-336b", "Ouro × 4 + input injection"),
    ("dense-middle-336b", "8 + 12 × 8 + 8"),
    ("dense-huginn-336b", "Huginn: 8 + 12 × 8 + 8 + injection"),
]
ID = "test_inferred_iid"


def esc(s: object) -> str:
    return html.escape(str(s))


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def line_chart(
    panels: list[dict],
    legend: list[tuple[int, str, bool]],
    *,
    width: int = 720,
    panel_h: int = 190,
    cols: int = 2,
    x_label: str,
    y_range: tuple[float, float],
    y_fmt: Callable[[float], str] = pct,
    caption: str,
) -> str:
    """Small-multiple line chart as inline SVG.

    Each panel: {"title", "lines": [(color_index, [(x, y)], dashed, label)]};
    legend entries are (color_index, name, dashed).
    """
    rows = -(-len(panels) // cols)
    pad_l, pad_r, pad_t, pad_b = 44, 10, 26, 34
    pw = (width - (cols - 1) * 18) / cols
    legend_h = 22 * (-(-len(legend) // 2)) + 10
    height = rows * (panel_h + pad_t + pad_b) + legend_h
    y0, y1 = y_range
    out = [
        f'<figure><svg viewBox="0 0 {width} {height:.0f}" role="img" '
        f'aria-label="{esc(caption)}" font-family="system-ui, sans-serif" font-size="11">'
    ]
    for i, p in enumerate(panels):
        ox = (i % cols) * (pw + 18)
        oy = (i // cols) * (panel_h + pad_t + pad_b)
        iw, ih = pw - pad_l - pad_r, panel_h
        xs = [x for line in p["lines"] for x, _ in line[1]]
        x0, x1 = min(xs), max(xs)

        def sx(
            x: float, ox: float = ox, x0: float = x0, x1: float = x1, iw: float = iw
        ) -> float:
            return ox + pad_l + (x - x0) / ((x1 - x0) or 1) * iw

        def sy(y: float, oy: float = oy, ih: float = ih) -> float:
            y = min(max(y, y0), y1)
            return oy + pad_t + ih - (y - y0) / (y1 - y0) * ih

        out.append(
            f'<text x="{ox + pad_l}" y="{oy + 14}" font-weight="600" fill="#0b0b0b" '
            f'class="ink">{esc(p["title"])}</text>'
        )
        for k in range(5):
            yv = y0 + (y1 - y0) * k / 4
            out.append(
                f'<line x1="{ox + pad_l}" x2="{ox + pad_l + iw}" y1="{sy(yv):.1f}" '
                f'y2="{sy(yv):.1f}" stroke="#e1e0d9" class="grid"/>'
                f'<text x="{ox + pad_l - 5}" y="{sy(yv) + 4:.1f}" text-anchor="end" '
                f'fill="#52514e" class="ink2">{y_fmt(yv)}</text>'
            )
        for k in range(5):
            xv = x0 + (x1 - x0) * k / 4
            label = f"{xv:.0%}" if x1 <= 1 else f"{xv:.0f}"
            out.append(
                f'<text x="{sx(xv):.1f}" y="{oy + pad_t + ih + 14}" text-anchor="middle" '
                f'fill="#52514e" class="ink2">{label}</text>'
            )
        out.append(
            f'<text x="{ox + pad_l + iw / 2}" y="{oy + pad_t + ih + 28}" '
            f'text-anchor="middle" fill="#52514e" class="ink2">{esc(x_label)}</text>'
        )
        for s_idx, pts, dashed, label in p["lines"]:
            d = " ".join(
                f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                for j, (x, y) in enumerate(pts)
            )
            dash = ' stroke-dasharray="4 3"' if dashed else ""
            out.append(
                f'<path d="{d}" fill="none" stroke="{LIGHT[s_idx]}" stroke-width="2" '
                f'class="s{s_idx + 1}"{dash}/>'
            )
            if len(pts) <= 40:
                for x, y in pts:
                    out.append(
                        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="transparent">'
                        f"<title>{esc(label)}: {f'{x:.0%}' if x1 <= 1 else f'L{x}'} → "
                        f"{y_fmt(y)}</title></circle>"
                    )
    ly = rows * (panel_h + pad_t + pad_b) + 8
    for i, (c_idx, name, dashed) in enumerate(legend):
        lx = (i % 2) * (width / 2)
        yy = ly + (i // 2) * 22
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        out.append(
            f'<line x1="{lx}" x2="{lx + 22}" y1="{yy + 6}" y2="{yy + 6}" '
            f'stroke="{LIGHT[c_idx]}" stroke-width="3" class="s{c_idx + 1}"{dash}/>'
            f'<text x="{lx + 28}" y="{yy + 10}" fill="#0b0b0b" class="ink">{esc(name)}</text>'
        )
    out.append(f"</svg><figcaption>{caption}</figcaption></figure>")
    return "".join(out)


def table(header: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def first_at(values: list[float], t: float = 0.9) -> int | None:
    return next((i for i, v in enumerate(values) if v >= t), None)


def window(rows: list[dict], frac: float, fn: Callable[[dict], float]) -> float:
    depth = len(rows) - 1
    half = round(3 * depth / 112)
    c = round(frac * depth)
    sel = rows[max(c - half, 0) : c + half + 1]
    return sum(fn(r) for r in sel) / len(sel)


CAUSAL_MODELS = [
    ("r1_l8_2001", "standard 8 layers"),
    ("r1_l4_13501", "standard 4 layers"),
    ("r2_l4_1201", "looped 4×2, ID stage"),
    ("r2_l4_7101", "looped 4×2, OOD stage"),
    ("r4_l4_401", "looped 4×4, ID stage"),
    ("r4_l4_3501", "looped 4×4, OOD stage"),
    ("r8_l4_301", "looped 4×8, ID stage"),
    ("r8_l4_2901", "looped 4×8, OOD stage"),
]
LENS, PROBE, PATCH, SWAP = 0, GRAY, 2, 1


def build(results: Path) -> tuple[str, str]:
    th = {
        n: json.loads((results / "twohop" / f"{n}.json").read_text())["layers"]
        for n in TWOHOP_ALL
    }
    causal = {
        n: json.loads((results / "twohop-causal" / f"{n}.json").read_text())["splits"]
        for n in TWOHOP_ALL
    }
    ifm = {
        n: json.loads((results / "ifm" / f"{n}.json").read_text())["rows"]
        for n, _ in IFM
    }

    def col(name: str, target: str, key: str, split: str = ID) -> list[float]:
        return [r["targets"][target][split][key] for r in th[name]]

    def ccol(name: str, kind: str, key: str, split: str = ID) -> list[float]:
        return [r[kind][key] for r in causal[name][split]["layers"]]

    def read_window(name: str, split: str = ID) -> list[int]:
        """Layers at which replacing the r1 residual redirects most answers."""
        patch = ccol(name, "full_patch", "answer_counterfactual", split)
        return [i for i, v in enumerate(patch) if v >= 0.5]

    def summary(name: str, split: str = ID) -> dict:
        win = read_window(name, split)
        lens = col(name, "bridge@r1", "lens_top1", split)
        probe = col(name, "bridge@r1", "probe_linear", split)
        swap = ccol(name, "token_swap", "answer_counterfactual", split)
        ablate = ccol(name, "token_ablate", "answer_original", split)
        steer = ccol(name, "token_steer", "answer_counterfactual", split)
        return {
            "acc": causal[name][split]["clean_accuracy"],
            "window": win,
            "lens": max(lens[i] for i in win) if win else 0.0,
            "probe": max(probe[i] for i in win) if win else 0.0,
            "swap": max(swap[i] for i in win) if win else 0.0,
            "ablate": min(ablate[i] for i in win) if win else 1.0,
            "steer": max(steer),
            "lens_90": first_at(lens),
        }

    S = {n: summary(n) for n, _ in CAUSAL_MODELS}
    S_ood = {n: summary(n, "test_inferred_ood") for n, _ in CAUSAL_MODELS}

    def rng(values: list[float]) -> str:
        lo, hi = pct(min(values)), pct(max(values))
        return lo if lo == hi else f"{lo}–{hi}"

    def span(win: list[int]) -> str:
        return (
            f"L{win[0]}–L{win[-1]}" if len(win) > 1 else (f"L{win[0]}" if win else "—")
        )

    # ---- headline figure: lens vs probe vs causal, per model, test ID
    panels = []
    for n, label in CAUSAL_MODELS:
        layers = list(range(len(th[n])))
        panels.append(
            {
                "title": label,
                "lines": [
                    (
                        PATCH,
                        list(
                            zip(
                                layers,
                                ccol(n, "full_patch", "answer_counterfactual"),
                                strict=True,
                            )
                        ),
                        False,
                        "replacing the whole r1 state redirects",
                    ),
                    (
                        PROBE,
                        list(
                            zip(
                                layers, col(n, "bridge@r1", "probe_linear"), strict=True
                            )
                        ),
                        True,
                        "probe finds the bridge",
                    ),
                    (
                        LENS,
                        list(
                            zip(layers, col(n, "bridge@r1", "lens_top1"), strict=True)
                        ),
                        False,
                        "logit lens shows the bridge",
                    ),
                    (
                        SWAP,
                        list(
                            zip(
                                layers,
                                ccol(n, "token_swap", "answer_counterfactual"),
                                strict=True,
                            )
                        ),
                        False,
                        "swapping only what the lens sees redirects",
                    ),
                ],
            }
        )
    fig_main = line_chart(
        panels,
        [
            (LENS, "logit lens shows the bridge", False),
            (PATCH, "model reads it here (whole-state swap)", False),
            (SWAP, "lens-visible swap changes the answer", False),
            (PROBE, "probe finds the bridge", True),
        ],
        cols=2,
        panel_h=110,
        x_label="layer (0 = embeddings)",
        y_range=(0, 1),
        caption=(
            "All at the first relation's position, on held-out compositions of seen "
            "facts. Green marks the layers where the second hop takes the bridge from "
            "this position: replacing the whole residual there with one whose bridge is "
            "another entity changes the final answer to that entity's. Blue is what the "
            "logit lens shows. Orange is the causal test of the lens: moving only the "
            "component the lens reads onto the other entity's token direction. Where "
            "orange rises with green, the lens is showing the state the model uses."
        ),
    )

    t_rows = []
    for n, label in CAUSAL_MODELS:
        s = S[n]
        t_rows.append(
            [
                esc(label),
                pct(s["acc"]),
                span(s["window"]),
                pct(s["lens"]),
                pct(s["swap"]),
                pct(s["ablate"]),
                pct(s["probe"]),
            ]
        )
    main_table = table(
        [
            "Model",
            "answer accuracy",
            "layers where the model reads the bridge",
            "lens shows it there (best layer)",
            "swap what the lens sees: answers redirected",
            "remove it: accuracy left",
            "probe there (best layer)",
        ],
        t_rows,
    )
    ood_rows = []
    for n, label in CAUSAL_MODELS:
        s = S_ood[n]
        ood_rows.append(
            [
                esc(label),
                pct(s["acc"]),
                span(s["window"]),
                pct(s["lens"]),
                pct(s["swap"]),
                pct(s["ablate"]),
            ]
        )
    ood_table = table(
        [
            "Model",
            "answer accuracy",
            "read layers",
            "lens shows it",
            "swap redirects",
            "remove: accuracy left",
        ],
        ood_rows,
    )

    # ---- readout appendix numbers
    b2_probe = [max(col(n, "bridge@r2", "probe_linear")) for n, _ in CAUSAL_MODELS]
    b2_lens_max = max(
        r["targets"]["bridge@r2"][s]["lens_top1"]
        for n in TWOHOP_ALL
        for r in th[n]
        for s in ("train_inferred", ID, "test_inferred_ood")
    )

    def mid_omp(n: str) -> float:
        layers = th[n]
        d = len(layers) - 1
        v = [
            r["positions"]["r1"]["omp_r2"][0] - r["positions"]["r1"]["omp_r2_null"][0]
            for r in layers
            if 0.25 <= r["layer"] / d <= 0.75
        ]
        return sum(v) / len(v)

    omp_rows = [
        [esc(label), f"{mid_omp(n) * 100:.0f} points"] for n, label in CAUSAL_MODELS
    ]
    omp_table = table(
        [
            "Model",
            "variance one token direction explains at r1, "
            "middle half of depth, above noise",
        ],
        omp_rows,
    )

    def onset(n: str, target: str) -> str:
        p = first_at(col(n, target, "probe_linear"))
        q = first_at(col(n, target, "lens_top1"))
        return (
            f"{'L' + str(p) if p is not None else 'never'} / "
            f"{'L' + str(q) if q is not None else 'never'}"
        )

    all_rows = []
    for n, label in TWOHOP_ALL.items():
        all_rows.append(
            [
                esc(label),
                pct(col(n, "target@r2", "lens_top1")[-1]),
                onset(n, "bridge@r1"),
                onset(n, "target@r2"),
                f"{pct(max(col(n, 'bridge@r2', 'probe_linear')))} / "
                f"{pct(max(col(n, 'bridge@r2', 'lens_top1')))}",
            ]
        )
    all_table = table(
        [
            "Model",
            "answer acc.",
            "bridge at r1: probe ≥ 90% / lens ≥ 90%",
            "answer at r2: probe ≥ 90% / lens ≥ 90%",
            "bridge at r2: best probe / best lens",
        ],
        all_rows,
    )

    # ---- lens examples
    examples = json.loads((results / "twohop-lens-examples.json").read_text())
    ckpt = {
        n: f"checkpoints/systematicity/{n.rsplit('_', 1)[0]}/checkpoint_epoch_{n.rsplit('_', 1)[1]}.pt"
        for n, _ in CAUSAL_MODELS
    }
    reads = {ckpt[n]: set(read_window(n)) for n, _ in CAUSAL_MODELS}
    p0 = examples["prompts"][0]
    trio = [
        ("r1_l8_2001", "standard 8 layers"),
        ("r2_l4_1201", "looped 4×2, ID stage"),
        ("r2_l4_7101", "looped 4×2, OOD stage"),
    ]
    static_grids = "".join(
        lg.twohop_grid(
            examples["models"][ckpt[n]], 0, p0, [1, 2], label, reads[ckpt[n]]
        )
        for n, label in trio
    )
    twohop_examples = (
        lg.STYLE + lg.legend() + f'<div class="lgwrap">{static_grids}</div>'
    )
    explorer = lg.twohop_explorer(
        examples, [(ckpt[n], label) for n, label in CAUSAL_MODELS], reads
    )
    lm_dir = results / "ifm-lens-examples"
    lm_models = [
        (n, label, json.loads((lm_dir / f"{n}.json").read_text()))
        for n, label in IFM
        if (lm_dir / f"{n}.json").exists()
    ]
    lm_section = ""
    if lm_models:
        by = {n: d for n, _, d in lm_models}
        statics = "".join(
            f"<p><strong>{esc(label)}</strong></p>" + lg.lm_table(by[n]["prompts"][0])
            for n, label in IFM
            if n in by and n in ("dense-d112-336b", "dense-ouro-336b")
        )
        lm_section = f"""<h3>What the lens looks like on text</h3>
<p>Each column is a position in the prompt and each row a depth; the cell is the logit
lens's top token for the <em>next</em> position, shaded by its probability (darker =
more confident). Hover a cell for the top five. Below: the standard 112-layer model and
the Ouro-style looped model (28 layers × 4) on the same sentence.</p>
{statics}
<p>In the standard model the continuation (“Tower”, “located”, “France”, “Paris”)
shows up from about 70% of depth. In the looped model the middle loops show the same
few generic tokens (“a”, “in”, “The”) at every position, often confidently, and the
real prediction appears only in roughly the last 15% of depth. The bottom rows are noise
in both: the lens on the raw embeddings, and a band of “import” at 7% of depth in the
standard model.</p>
<p>Any model and prompt (two of the prompts are random FineWeb-Edu snippets):</p>
{lg.lm_explorer(lm_models)}"""

    # ---- IFM
    def ifm_lines(fn: Callable[[dict], float]) -> list:
        out = []
        for i, (n, label) in enumerate(IFM):
            rows = ifm[n]
            d = len(rows) - 1
            out.append((i, [(r["execution"] / d, fn(r)) for r in rows], False, label))
        return out

    ifm_legend = [(i, label, False) for i, (_, label) in enumerate(IFM)]
    fig_ifm = line_chart(
        [
            {
                "title": "Logit lens agrees with the final prediction",
                "lines": ifm_lines(lambda r: r["lens"]["agree_final_top1"]),
            },
            {
                "title": "Tuned lens (reference) agrees",
                "lines": ifm_lines(lambda r: r["tuned_ridge"]["agree_final_top1"]),
            },
        ],
        ifm_legend,
        cols=2,
        panel_h=200,
        x_label="fraction of executed depth",
        y_range=(0, 1),
        caption=(
            "Share of 4,096 FineWeb-Edu tokens where the lens's top-1 token matches "
            "the model's final top-1, after every executed block. All six models "
            "were trained on the same 336B tokens with the same recipe."
        ),
    )
    fig_dark = line_chart(
        [
            {
                "title": "Raw energy in the weakest-read quarter of directions",
                "lines": ifm_lines(lambda r: r["energy_bands_raw"][-1]),
            },
            {
                "title": "Variance one token direction explains, above noise",
                "lines": ifm_lines(lambda r: r["omp_r2"][0] - r["omp_r2_null"][0]),
            },
        ],
        ifm_legend,
        cols=2,
        panel_h=170,
        x_label="fraction of executed depth",
        y_range=(0, 1),
        caption=(
            "Left: a random vector would put 25% there. Right: on a 0–100% scale "
            "to compare with the two-hop models; it never exceeds 5%."
        ),
    )

    def agree(key: str) -> Callable[[dict], float]:
        return lambda r: r[key]["agree_final_top1"]

    ifm_rows = []
    for n, label in IFM:
        rows = ifm[n]
        ifm_rows.append(
            [
                esc(label),
                f"{rows[-1]['lens']['next_token_top1']:.1%}",
                " / ".join(
                    pct(window(rows, f, agree("lens"))) for f in (0.5, 0.75, 0.9)
                ),
                " / ".join(
                    pct(window(rows, f, agree("tuned_ridge"))) for f in (0.5, 0.75, 0.9)
                ),
            ]
        )
    ifm_table = table(
        [
            "Model",
            "final next-token top-1",
            "logit lens agrees at 50 / 75 / 90% depth",
            "tuned lens agrees at 50 / 75 / 90%",
        ],
        ifm_rows,
    )
    std = ("dense-d112-336b", "dense-d28-336b")
    loops = [n for n, _ in IFM if n not in std]
    at90_std = [window(ifm[n], 0.9, agree("lens")) for n in std]
    at90_loop = [window(ifm[n], 0.9, agree("lens")) for n in loops]
    ouro, ouro_inj = ifm["dense-ouro-336b"], ifm["dense-ouro-raw-injection-336b"]

    def bump(rows: list) -> str:
        return ", ".join(
            f"{pct(rows[e]['lens']['agree_final_top1'])} → "
            f"{pct(rows[e + 1]['lens']['agree_final_top1'])}"
            for e in (28, 56, 84)
        )

    dark_mid = [
        r["energy_bands_raw"][-1]
        for n, _ in IFM
        for r in ifm[n]
        if 0.25 <= r["execution"] / (len(ifm[n]) - 1) <= 0.75
    ]
    omp_peak = max(r["omp_r2"][0] - r["omp_r2_null"][0] for n, _ in IFM for r in ifm[n])

    v8, v4 = S["r1_l8_2001"], S["r1_l4_13501"]
    l2, l4 = S["r2_l4_1201"], S["r4_l4_401"]
    causal_looped = [
        S[n]
        for n in ("r2_l4_7101", "r4_l4_401", "r4_l4_3501", "r8_l4_301", "r8_l4_2901")
    ]
    style = (
        "<style>"
        + "".join(
            f".s{i + 1}{{stroke:light-dark({a},{b})}}"
            for i, (a, b) in enumerate(zip(LIGHT, DARK, strict=True))
        )
        + (
            "svg .ink{fill:var(--ink)} svg .ink2{fill:var(--ink-2)} "
            "svg .grid{stroke:var(--grid)} figure{margin:1.5rem 0} "
            "figcaption{color:var(--ink-2);font-size:.9rem} "
            ".tldr{background:var(--surface);padding:.8rem 1.2rem;border-radius:8px}</style>"
        )
    )

    body = f"""{style}
<p>The question: <strong>does weight sharing (looping) make the logit lens show the
intermediate state a model actually uses?</strong> The logit lens is the tool we care about,
because it needs no labels and works when we don't know what to look for. Two checks
tell us whether to trust what it shows: a <em>causal test</em> (does changing what the lens
reads change the model's answer?) and a <em>linear probe</em> (is the lens missing
information that is there?).</p>

<p>Our own S3 experiment couldn't answer this because our looped and standard models
learned different algorithms. Here I use public model pairs trained on the same data by
other groups. Code and raw results: <a href="{REPO}/pull/6">PR #6</a>.</p>

<div class="tldr">
<p><strong>Short answer</strong></p>
<ul>
<li><strong>On a two-hop reasoning task, yes, once the model loops enough or trains long
enough.</strong> In looped models with 4 or 8 iterations, and in the 2-iteration model
once it generalizes to unseen fact pairs, the logit lens shows the intermediate
(“bridge”) entity at {rng([x["lens"] for x in causal_looped])} at the layers where the model
reads it. Swapping just what the lens sees redirects {rng([x["swap"] for x in causal_looped])}
of answers (best layer per model), and removing it drops accuracy to
{rng([x["ablate"] for x in causal_looped])}. So the lens is showing the state the model uses.</li>
<li><strong>In standard (non-looped) models, no.</strong> The model reads the bridge at
{span(v8["window"])} (8 layers) or {span(v4["window"])} (4 layers), when the lens shows it
{pct(v8["lens"])} / {pct(v4["lens"])} of the time, and changing what the lens sees redirects
{pct(max(v8["swap"], v4["swap"]))} of answers. The lens only shows the bridge clearly later
(from L{v8["lens_90"]} in the 8-layer model), after the model is done with it, and what it
shows then is not used.</li>
<li><strong>Looping isn't sufficient on its own.</strong> The 2-iteration model at its
earlier training stage behaves like the standard ones (swap redirects
{pct(l2["swap"])}).</li>
<li><strong>It doesn't carry over to language models.</strong> In six 1.5–3.7B models
trained on the same 336B tokens, looped models are, if anything, <em>less</em> readable
late in the network (lens agrees with the final prediction {pct(min(at90_loop))}–{pct(max(at90_loop))}
vs {pct(min(at90_std))}–{pct(max(at90_std))} at 90% depth). There is no known intermediate
there, so no causal test.</li>
<li><strong>Caveats:</strong> one training run per configuration, two-hop checkpoints matched
by training stage rather than step, and a single task.</li>
</ul>
</div>

<h2>The two-hop models</h2>
<p>From <a href="https://arxiv.org/abs/2604.07822">Loop, Think &amp; Generalize</a> (Kohli
et al. 2026): GPT-2 blocks trained on a made-up knowledge graph of 2,000 entities. The
input is a head entity and two relations, <code>&lt;h&gt;&lt;r1&gt;&lt;r2&gt;</code>; the
answer is <code>r2(r1(h))</code>. The <strong>bridge</strong> <code>b = r1(h)</code> is an
ordinary entity token. The model is trained to output it for the one-hop question
<code>&lt;h&gt;&lt;r1&gt;</code>, but never as part of a two-hop question. So the hope is
that the model reuses that token as its intermediate state. “4×2” is a 4-layer block run
twice, with the same depth and compute as the standard 8-layer model. Checkpoints come
from two training stages: generalizing to new pairs of seen facts (“ID”), and later to pairs
of facts never composed in training (“OOD”, which only looped models reach). Everything
below is measured on 1,000–2,000 held-out pairs of seen facts, which every model answers
at {pct(min(S[n]["acc"] for n, _ in CAUSAL_MODELS))}–{pct(max(S[n]["acc"] for n, _ in CAUSAL_MODELS))}.</p>

<h3>What the lens looks like</h3>
<p>One held-out prompt, <code>{esc(" ".join(p0["tokens"]))}</code>: the bridge is
<code>{esc(p0["bridge"])}</code> and the answer <code>{esc(p0["answer"])}</code>. Each cell
is the logit lens at one layer (row) and position (column): the label is its top token and
probability, and the bar shows its top three tokens and the rest of the probability. The
green edge marks the layers where the causal test below says the model reads the bridge.
Three models of the same depth:</p>
{twohop_examples}
<p>Typical of every prompt: the lens puts almost all of its non-uniform mass on a single
token, so a cell is basically “one token at some confidence”. In the standard model the
bridge appears at the first relation only after the green layers; in the looped OOD-stage
model it is the lens's confident top token for most of them.</p>
<p>Any model and prompt:</p>
{explorer}

<p>Because attention is causal, the second hop has to get the bridge from the first
relation's position, so that is where we look. To find <em>when</em> it takes it, I replace
that position's whole residual at one layer with the residual from a different question
whose bridge is another entity <code>b'</code>, and check whether the answer becomes
<code>r2(b')</code>. To test the <em>lens</em>, I instead move only the component the lens
reads (along the bridge token's readout direction) onto <code>b'</code>'s direction,
leaving everything else alone.</p>

{fig_main}

<details open><summary>Summary per model (held-out pairs of seen facts)</summary>
{main_table}
<p>“Layers where the model reads the bridge”: replacing the whole residual redirects at
least half the answers. The other columns are the best (or, for accuracy, worst) value
over those layers.</p>
</details>

<p>Reading the figure: in the standard models the green read window closes before the
blue lens curve rises, and orange stays at zero. In the 4×4 and 4×8 models, and in the
4×2 model at its OOD stage, blue is already near 100% inside the read window and orange
rises with it. Forcing a large push along the other entity's token direction works for
looped models even at their earlier stage (up to {pct(l2["steer"])} for 4×2,
{pct(l4["steer"])} for 4×4), but never for standard models ({pct(max(v8["steer"], v4["steer"]))}).
So the looped models' second hop is built to read token directions, while the standard
models' second hop isn't.</p>

<details><summary>The same on pairs of facts never composed in training</summary>
{ood_table}
<p>Only the OOD-stage looped models answer these; the read window still exists in the
others, but the answer is wrong either way.</p></details>

<h3>What the lens misses</h3>
<ul>
<li><strong>Timing in standard models.</strong> A probe finds the bridge during the read
window ({pct(v8["probe"])} in the 8-layer model) while the lens doesn't, so the lens
misses the intermediate entirely there.</li>
<li><strong>The copy at the second relation.</strong> After the hand-off, the bridge is
also linearly decodable at the second relation's position
({pct(min(b2_probe))}–{pct(max(b2_probe))}) but never lens-readable (at most
{pct(b2_lens_max)}) in any model. It isn't hidden in weakly read directions; it's just not
written along the token's own direction.</li>
<li><strong>How token-like the state is.</strong> Mid-depth, one token direction explains
much more of the residual in looped models than in standard ones:</li>
</ul>
{omp_table}

<details><summary>Readout timing for all 13 checkpoints, including the train-fit stage</summary>
{all_table}
<p>First layer at which the probe / the lens reach 90% top-1 on held-out pairs of seen
facts.</p></details>

<h2>Does it carry over to language models?</h2>
<p>IFM's <a href="https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/">Towards
Looped Models Done Right</a> trained six models on the same 336B tokens with the same
tokenizer, width, seed and schedule, with loss only on the final output. Every looped
model stores 28 blocks and executes 112. The controls are a standard 112-block model
(same executed depth) and a standard 28-block model (same parameters). There is no known
intermediate in text, so the measure is how early the lens agrees with the model's own
final prediction. A tuned lens (a least-squares map to the final layer) is the reference
for what is linearly there.</p>

{lm_section}

{fig_ifm}
<details open><summary>Summary (each cell averages ±2.7% of depth)</summary>{ifm_table}</details>

<p>From 75% of depth on, both standard models lead every looped model, and the models
that loop 8 times are generally the least readable. Loop boundaries don't help: in the
Ouro-style model the end of each loop feeds the output head directly, yet the lens is no
better there than one block later ({bump(ouro)}). With input injection it is
<em>worse</em> just after each injection ({bump(ouro_inj)}).</p>

<details><summary>Dark subspace and token reconstruction in the language models</summary>
{fig_dark}
<p>{pct(min(dark_mid))}–{pct(max(dark_mid))} of the residual's raw energy through the
middle half of depth sits in the quarter of directions the output matrix reads most
weakly: a large component shared by every token, invisible to the lens. The part that
varies between tokens is spread roughly evenly, and the residual is never close to a few
token directions (the best single token explains at most {omp_peak:.1%} more variance
than noise).</p></details>

<h2>Caveats</h2>
<ul>
<li>One training run per configuration in both families; no error bars on the
language-model numbers (4,096 tokens from 64 documents).</li>
<li>Two-hop checkpoints are at different epochs, matched by training stage. What we see
changes a lot between stages (4×2: not causal at the ID stage, causal at the OOD stage).</li>
<li>One task. The two-hop task has an intermediate that is also a trained output token,
which is the most favorable case for the lens.</li>
<li>The causal test moves a single readout direction per entity. A model could read the
bridge in a token-like format that isn't exactly that direction; the strong push and the
removal test bound this from both sides.</li>
</ul>

<h2>Reproduce</h2>
<p>Branch <code>brendanlong/looped-lm-lens</code> of
<a href="{REPO}">sequential-transformer-lens-experiment</a>
(<a href="{REPO}/pull/6">PR #6</a>). Write-up: <code>results/looped-lm.md</code>; raw JSONs:
<code>results/looped-lm/</code>.</p>
<pre>uv sync --group looped
# two-hop readout and causal tests (downloads Kohli et al.'s checkpoints)
uv run --group looped snakemake -s looped_lm/Snakefile --cores 1 --config kohli_root=ext/kohli
# language models: needs xLLM-Loop's env (looped_lm/jobs/ifm_env.sh)
bash looped_lm/jobs/ifm.sh
# this page
uv run python -m looped_lm.build_html_report results/looped-lm --template TEMPLATE -o page.html</pre>
"""
    return "Does looping make the logit lens show the intermediate state?", body


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    title, body = build(args.results)
    page = args.template.read_text().replace("{{TITLE}}", esc(title))
    args.out.write_text(page.replace("{{BODY}}", body))


if __name__ == "__main__":
    main()
