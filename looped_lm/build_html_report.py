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

DOMAIN = "looped-lens-a783c534.surge.sh"
REPO = "https://github.com/brendanlong/sequential-transformer-lens-experiment"

# Light-theme literals (survive saving as an article); CSS classes swap in theme vars.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2b9a2b"]

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
    series_names: list[str],
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

    Each panel: {"title", "lines": [(series_index, [(x, y)], dashed, label)]}.
    """
    rows = -(-len(panels) // cols)
    pad_l, pad_r, pad_t, pad_b = 44, 10, 26, 34
    pw = (width - (cols - 1) * 18) / cols
    legend_h = 22 * (-(-len(series_names) // 3)) + 10
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
    for i, name in enumerate(series_names):
        lx = (i % 3) * (width / 3)
        yy = ly + (i // 3) * 22
        out.append(
            f'<line x1="{lx}" x2="{lx + 22}" y1="{yy + 6}" y2="{yy + 6}" '
            f'stroke="{LIGHT[i]}" stroke-width="3" class="s{i + 1}"/>'
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


def build(results: Path) -> tuple[str, str]:
    th = {
        n: json.loads((results / "twohop" / f"{n}.json").read_text())["layers"]
        for n in TWOHOP_ALL
    }
    ifm = {
        n: json.loads((results / "ifm" / f"{n}.json").read_text())["rows"]
        for n, _ in IFM
    }

    def col(name: str, target: str, key: str, split: str = ID) -> list[float]:
        return [r["targets"][target][split][key] for r in th[name]]

    # ---- two-hop numbers used in prose
    lag = {}
    for n, _ in TWOHOP_ID:
        lag[n] = (
            first_at(col(n, "bridge@r1", "probe_linear")),
            first_at(col(n, "bridge@r1", "lens_top1")),
        )
    b2_probe = [
        max(col(n, "bridge@r2", "probe_linear", s))
        for n in TWOHOP_ALL
        if "train" not in TWOHOP_ALL[n]
        for s in (ID,)
    ]
    b2_lens_max = max(
        r["targets"]["bridge@r2"][s]["lens_top1"]
        for n in TWOHOP_ALL
        for r in th[n]
        for s in ("train_inferred", ID, "test_inferred_ood")
    )
    l2_last = col("r2_l4_1201", "bridge@r1", "lens_top1")
    l2_peak = max(l2_last)

    def mid_omp(n: str, pos: str, key: str | None = None) -> float:
        layers = th[n]
        d = len(layers) - 1
        mid = [r for r in layers if 0.25 <= r["layer"] / d <= 0.75]
        if key is None:
            v = [
                r["positions"][pos]["omp_r2"][0] - r["positions"][pos]["omp_r2_null"][0]
                for r in mid
            ]
        else:
            v = [r["positions"][pos]["omp_first_atom"][key] for r in mid]
        return sum(v) / len(v)

    # ---- two-hop figure 1: bridge@r1 and bridge@r2 lens vs probe
    panels = []
    for target, where in (
        ("bridge@r1", "first relation (r1)"),
        ("bridge@r2", "second relation (r2)"),
    ):
        for i, (n, label) in enumerate(TWOHOP_ID):
            lens = col(n, target, "lens_top1")
            probe = col(n, target, "probe_linear")
            panels.append(
                {
                    "title": f"{label} — bridge at {where}",
                    "lines": [
                        (i, list(enumerate(lens)), False, f"{label} lens"),
                        (i, list(enumerate(probe)), True, f"{label} probe"),
                    ],
                }
            )
    # reorder panels so r1 and r2 for each model sit side by side
    panels_pairs = []
    for i in range(len(TWOHOP_ID)):
        panels_pairs += [panels[i], panels[i + len(TWOHOP_ID)]]
    fig1 = line_chart(
        panels_pairs,
        [label for _, label in TWOHOP_ID],
        cols=2,
        panel_h=110,
        x_label="effective layer (0 = embeddings)",
        y_range=(0, 1),
        caption=(
            "Top-1 accuracy for the bridge entity on held-out compositions of "
            "seen facts (ID-stage checkpoints). Solid: logit lens. Dashed: linear "
            "probe. Left: at the first relation, where the bridge is computed. "
            "Right: at the second relation, where the second hop reads it."
        ),
    )

    # ---- two-hop figure 2: OMP R2(1)-null at r1 by depth fraction
    lines = []
    for i, (n, label) in enumerate(TWOHOP_ID):
        layers = th[n]
        d = len(layers) - 1
        lines.append(
            (
                i,
                [
                    (
                        r["layer"] / d,
                        r["positions"]["r1"]["omp_r2"][0]
                        - r["positions"]["r1"]["omp_r2_null"][0],
                    )
                    for r in layers
                ],
                False,
                label,
            )
        )
    fig2 = line_chart(
        [
            {
                "title": "Variance of the r1 residual explained by one token direction, "
                "above a noise baseline",
                "lines": lines,
            }
        ],
        [label for _, label in TWOHOP_ID],
        cols=1,
        panel_h=200,
        x_label="fraction of effective depth",
        y_range=(0, 1),
        caption=(
            "Orthogonal matching pursuit over the unembedding rows, R²(1) minus "
            "the same for Gaussian noise with matched covariance, ID-stage "
            "checkpoints, test ID. Layer 0 is the input token's own embedding."
        ),
    )

    # ---- IFM figures
    def ifm_lines(fn: Callable[[dict], float]) -> list:
        out = []
        for i, (n, label) in enumerate(IFM):
            rows = ifm[n]
            d = len(rows) - 1
            out.append((i, [(r["execution"] / d, fn(r)) for r in rows], False, label))
        return out

    fig3 = line_chart(
        [
            {
                "title": "Logit lens agrees with final prediction",
                "lines": ifm_lines(lambda r: r["lens"]["agree_final_top1"]),
            },
            {
                "title": "Ridge-tuned lens agrees with final prediction",
                "lines": ifm_lines(lambda r: r["tuned_ridge"]["agree_final_top1"]),
            },
        ],
        [label for _, label in IFM],
        cols=2,
        panel_h=200,
        x_label="fraction of executed depth",
        y_range=(0, 1),
        caption=(
            "Share of 4,096 FineWeb-Edu tokens where the lens's top-1 token "
            "matches the model's final top-1, after every executed block. All six "
            "models were trained on the same 336B tokens with the same recipe."
        ),
    )
    fig4 = line_chart(
        [
            {
                "title": "Raw residual energy in the weakest-read quarter of directions",
                "lines": ifm_lines(lambda r: r["energy_bands_raw"][-1]),
            },
            {
                "title": "Variance one token direction explains, above noise",
                "lines": ifm_lines(lambda r: r["omp_r2"][0] - r["omp_r2_null"][0]),
            },
        ],
        [label for _, label in IFM],
        cols=2,
        panel_h=200,
        x_label="fraction of executed depth",
        y_range=(0, 1),
        caption=(
            "Left: an isotropic vector would put 25% of its energy there. "
            "Right: on the same 0–100% scale as the two-hop figure; the LMs never "
            "rise above 5%."
        ),
    )

    # ---- tables
    def onset(n: str, target: str) -> str:
        p = first_at(col(n, target, "probe_linear"))
        l_ = first_at(col(n, target, "lens_top1"))
        return (
            f"{'L' + str(p) if p is not None else 'never'} / "
            f"{'L' + str(l_) if l_ is not None else 'never'}"
        )

    t_rows = []
    for n, label in TWOHOP_ALL.items():
        lens = col(n, "bridge@r1", "lens_top1")
        peak = max(range(len(lens)), key=lens.__getitem__)
        t_rows.append(
            [
                esc(label),
                pct(col(n, "target@r2", "lens_top1")[-1]),
                onset(n, "bridge@r1"),
                f"{pct(lens[peak])} at L{peak} / {pct(lens[-1])}",
                onset(n, "target@r2"),
                f"{pct(max(col(n, 'bridge@r2', 'probe_linear')))} / "
                f"{pct(max(col(n, 'bridge@r2', 'lens_top1')))}",
            ]
        )
    twohop_table = table(
        [
            "Model",
            "answer acc.",
            "bridge@r1: probe ≥ 90% / lens ≥ 90%",
            "bridge@r1 lens: peak / last",
            "answer@r2: probe ≥ 90% / lens ≥ 90%",
            "bridge@r2: best probe / best lens",
        ],
        t_rows,
    )
    omp_rows = []
    for n, label in TWOHOP_ALL.items():
        if "train" in label:
            continue
        omp_rows.append(
            [
                esc(label),
                f"{mid_omp(n, 'r1'):.2f}",
                pct(mid_omp(n, "r1", "bridge")),
                f"{mid_omp(n, 'r2'):.2f}",
                pct(mid_omp(n, "r2", "tail")),
            ]
        )
    omp_table = table(
        [
            "Model",
            "r1: R²(1) − null",
            "r1: first token = bridge",
            "r2: R²(1) − null",
            "r2: first token = answer",
        ],
        omp_rows,
    )

    def agree(key: str) -> Callable[[dict], float]:
        return lambda r: r[key]["agree_final_top1"]

    ifm_rows = []
    for n, label in IFM:
        rows = ifm[n]
        d = len(rows) - 1
        pk = max(rows, key=lambda r: r["omp_r2"][0] - r["omp_r2_null"][0])
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
                pct(window(rows, 0.5, lambda r: r["energy_bands_raw"][-1])),
                f"{pk['omp_r2'][0] - pk['omp_r2_null'][0]:.3f} at {pk['execution'] / d:.0%}",
            ]
        )
    ifm_table = table(
        [
            "Model",
            "final next-token top-1",
            "lens agrees at 50 / 75 / 90% depth",
            "tuned lens agrees at 50 / 75 / 90%",
            "raw energy in weakest quarter at 50%",
            "one-token R² above noise, peak",
        ],
        ifm_rows,
    )

    std = ("dense-d112-336b", "dense-d28-336b")
    loops = [n for n, _ in IFM if n not in std]
    at90_std = [window(ifm[n], 0.9, agree("lens")) for n in std]
    at90_loop = [window(ifm[n], 0.9, agree("lens")) for n in loops]
    t90_std = [window(ifm[n], 0.9, agree("tuned_ridge")) for n in std]
    t90_loop = [window(ifm[n], 0.9, agree("tuned_ridge")) for n in loops]
    ouro = ifm["dense-ouro-336b"]
    ouro_inj = ifm["dense-ouro-raw-injection-336b"]

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

    v8, l2, l4, _l8 = (lag[n] for n, _ in TWOHOP_ID)
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
            ".tldr{background:var(--surface);padding:.8rem 1.2rem;border-radius:8px}"
            "</style>"
        )
    )

    body = f"""{style}
<p>Our own S3 experiment was abandoned because the looped and standard models we
trained learned different algorithms, so we couldn't tell whether weight sharing
itself made the logit lens work better. This report asks the same question of two
public model families where someone else trained looped and standard models on the
same data. Code and raw results are in <a href="{REPO}/pull/6">PR #6</a>.</p>

<div class="tldr">
<p><strong>Summary</strong></p>
<ul>
<li><strong>Small two-hop models: looping makes the intermediate readable sooner.</strong>
The intermediate ("bridge") entity can be decoded by a linear probe from layer
L{v8[0]} in both an 8-layer standard model and a depth-matched 4-layer block looped
twice. The logit lens reads it from L{v8[1]} in the standard model and L{l2[1]} in the
looped one, and immediately in models looped 4 or 8 times.</li>
<li><strong>But the copy the model actually uses is invisible to the lens in every
model.</strong> Where the second step reads the bridge, a probe decodes it at
{pct(min(b2_probe))}–{pct(max(b2_probe))}; the lens reads it at no more than
{pct(b2_lens_max)}, looped or not.</li>
<li><strong>Language models (1.5–3.7B, same 336B tokens): looping does not help.</strong>
At 90% of depth the standard models' lens matches the final prediction
{pct(min(at90_std))}–{pct(max(at90_std))} of the time versus
{pct(min(at90_loop))}–{pct(max(at90_loop))} for looped models.</li>
<li><strong>Caveats:</strong> one training run per configuration, no error bars on
the language-model numbers, and two-hop checkpoints that are matched by training
stage rather than training step. The two-hop result is suggestive, not settled.</li>
</ul>
</div>

<h2>What was measured</h2>
<ul>
<li><strong>Logit lens:</strong> apply the model's final normalization and output
matrix to the residual stream after every layer (every executed block, for looped
models) and read the top token.</li>
<li><strong>Linear probe:</strong> the reference for whether information is present at
all. For the two-hop models, a classifier over the 2,000 entities, trained on
compositions that share no (head, first relation) pair with the test examples. For the
language models, a least-squares map from each layer to the final layer, read out
through the model's own head (a "tuned lens").</li>
<li><strong>Dark subspace:</strong> the share of the residual's energy in the quarter
of directions that the output matrix reads most weakly. A random vector puts 25% there.</li>
<li><strong>How many tokens:</strong> how much of the residual's variance the best 1–8
token directions explain (orthogonal matching pursuit), minus the same for random noise
with the same covariance.</li>
</ul>

<h2>Two-hop composition (Kohli et al. 2026)</h2>
<p>Models from <a href="https://arxiv.org/abs/2604.07822">Loop, Think &amp;
Generalize</a>: GPT-2 blocks of width 768 trained on a made-up knowledge graph with
2,000 entities. The input is a head entity and two relations; the answer is
relation2(relation1(head)), and the bridge relation1(head) is an ordinary vocabulary
token that is never a training target. "4L×2" is a 4-layer block run twice; it has the
same depth and compute as the 8-layer standard model. Each configuration has
checkpoints at three training stages: fitting the training set, generalising to new
pairs of seen facts ("ID"), and generalising to pairs of facts never composed in
training ("OOD"; only looped models get there). Evaluation uses 2,000 held-out
compositions of seen facts.</p>

{fig1}

<p>The lag between "a probe can decode it" and "the lens can read it" is
{v8[1] - v8[0]} layers in the standard 8-layer model, {l2[1] - l2[0]} in its looped twin,
and {max(l4[1] - l4[0], 0)} in the 4×4 model (the 4×8 model's lens gets there a layer
before the probe). Once readable, the bridge stays readable to the end in the 4×4 and
4×8 models. The twice-looped model peaks at {pct(l2_peak)} and fades to
{pct(l2_last[-1])} by its last layer while the probe stays at 100%, the same pattern as
our S3 model. The standard 4-layer model lags by only one layer, so the lag isn't purely
a function of looping.</p>

<p>At the second relation, where the second step needs it, the bridge is
probe-decodable and lens-invisible in every model. It isn't hidden in weakly read
directions either: a probe restricted to the strongly read half of directions decodes it
about as well as one restricted to the weak half. It just isn't written along the
bridge token's own output direction.</p>

{fig2}

<p>At mid-depth, one token direction (the bridge, in 100% of examples) explains
{mid_omp("r4_l4_401", "r1") * 100:.0f} percentage points more variance than noise in the 4×4
and 4×8 models, against {mid_omp("r1_l8_2001", "r1") * 100:.0f} in the standard 8-layer
model. When a token shows up at all
it is one token: adding more directions barely helps.</p>

<details><summary>All 13 checkpoints: when the bridge and answer become decodable vs
readable (test ID)</summary>
{twohop_table}
<p>"Probe ≥ 90% / lens ≥ 90%": the first layer at which each reaches 90% top-1 (L0 =
embeddings).</p></details>
<details><summary>Token reconstruction, averaged over the middle half of depth</summary>
{omp_table}</details>

<h2>Language models (IFM LoopedLM)</h2>
<p>IFM's <a href="https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/">Towards
Looped Models Done Right</a> trained six models on the same 336B tokens with the same
tokenizer, width, seed and schedule, with loss only on the final output. Every looped
model stores 28 blocks and executes 112. The standard controls are D112 (112 separate
blocks, the same executed depth) and D28 (28 blocks, the same parameter count). Neither
IFM paper looks at the lens. I ran 128 FineWeb-Edu documents × 64 positions; the tuned
lens is fit on half the documents and everything is measured on the other half (4,096
tokens).</p>

{fig3}

<p>Through the middle of the network all the models are barely readable. From 75% of
depth on, where the prediction forms, both standard models lead every looped model
on both lenses (tuned lens at 90% depth: {pct(min(t90_std))}–{pct(max(t90_std))} vs
{pct(min(t90_loop))}–{pct(max(t90_loop))}). The models that loop 8 times are generally
the least readable. Loop boundaries don't help: in the Ouro-style model the state at
the end of each loop feeds the output head directly, yet the lens is no better there
than one block later ({bump(ouro)}). With input injection the state is
<em>less</em> readable just after each injection ({bump(ouro_inj)}).</p>

{fig4}

<p>In every model, {pct(min(dark_mid))}–{pct(max(dark_mid))} of the residual's raw
energy through the middle half of depth sits in the most weakly read quarter of
directions. That's a large component shared by every token. Input injection shows up as
a sawtooth: each injection pushes it back up. The part that varies from token to token is
spread roughly evenly, and the residual is never close to a few token directions: the
best single token explains at most {omp_peak:.1%} more variance than noise.</p>

<details open><summary>Summary table (each cell averages ±2.7% of depth, so no column
sits exactly on a loop boundary)</summary>{ifm_table}</details>

<h2>Caveats</h2>
<ul>
<li>One training run per configuration in both families, and no error bars on the
language-model numbers (4,096 tokens from 64 documents). Differences of a few points
are not established.</li>
<li>Two-hop checkpoints are at different epochs, and lens timing moves a lot with
training stage. For example, the twice-looped model reads the bridge from L5 at the ID
stage and L1 at the OOD stage.</li>
<li>Two-hop probes are trained on training compositions, so on never-composed fact
pairs a probe can score below the lens.</li>
<li>FineWeb-Edu overlaps the language models' training sources, equally for all six.
The D28 and Ouro-style checkpoints are mid-schedule (step 80,000 of 119,210), so all
six are compared there.</li>
</ul>

<h2>Reproduce</h2>
<p>Branch <code>brendanlong/looped-lm-lens</code> of
<a href="{REPO}">sequential-transformer-lens-experiment</a> (<a href="{REPO}/pull/6">PR #6</a>);
the full write-up is <code>results/looped-lm.md</code>, and the raw JSONs are in
<code>results/looped-lm/</code>.</p>
<pre>uv sync --group looped
# two-hop: downloads Kohli et al.'s checkpoints, then runs every config
uv run --group looped snakemake -s looped_lm/Snakefile --cores 1 --config kohli_root=ext/kohli
# IFM: needs xLLM-Loop's env (looped_lm/jobs/ifm_env.sh); gpuc job looped_lm/jobs/ifm.yaml
bash looped_lm/jobs/ifm.sh
# tables and this page
uv run python -m looped_lm.report results/looped-lm
uv run python -m looped_lm.build_html_report results/looped-lm --template TEMPLATE -o page.html</pre>
"""
    return "Are looped transformers more readable with the logit lens?", body


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
