# ruff: noqa: E501  (HTML/JS templates)
"""Logit-lens example grids for the HTML report.

Two-hop: one cell per (layer, position), drawn as a stacked bar of the lens's top
tokens colored by role (bridge, answer, input token, other) plus the rest of the
probability mass. Language models: one cell per (layer, position) showing the
top token, shaded by its probability, with the top five on hover.

Each view is rendered statically in Python (so a saved copy of the page keeps
it) and again by a small script that lets the reader pick the model and prompt.
"""

import html
import json

ROLE_COLORS = {
    "bridge": "#2a78d6",
    "answer": "#eb6834",
    "input": "#52514e",
    "other": "#b3b2ab",
}
REST = "#e6e5e0"


def esc(s: object) -> str:
    return html.escape(str(s))


def _pct(p: float) -> str:
    return "<1%" if p < 0.005 else f"{p:.0%}"


def _role(token: str, prompt: dict) -> tuple[str, str]:
    """(role, short label) of a token in a two-hop prompt."""
    if token == prompt["bridge"]:
        return "bridge", "bridge"
    if token == prompt["answer"]:
        return "answer", "answer"
    names = ("h", "r1", "r2")
    for name, tok in zip(names, prompt["tokens"], strict=True):
        if token == tok:
            return "input", f"input {name}"
    return "other", token.strip("<>")


def _bar(top: list, prompt: dict, k: int = 3) -> str:
    x, parts = 0.0, []
    for token, p in top[:k]:
        role, label = _role(token, prompt)
        parts.append(
            f'<rect x="{x:.2f}" width="{p * 100:.2f}" height="10" '
            f'fill="{ROLE_COLORS[role]}"><title>{esc(label)} {p:.1%}</title></rect>'
        )
        x += p * 100
    parts.append(f'<rect x="{x:.2f}" width="{100 - x:.2f}" height="10" fill="{REST}"/>')
    return (
        '<svg viewBox="0 0 100 10" preserveAspectRatio="none" class="lgbar">'
        + "".join(parts)
        + "</svg>"
    )


def _cell(top: list, prompt: dict) -> str:
    token, p = top[0]
    role, label = _role(token, prompt)
    weight = "600" if role in ("bridge", "answer") else "400"
    return (
        f'<div class="lgcell"><span class="lglbl lg-{role}" style="font-weight:{weight}">'
        f"{esc(label)} {_pct(p)}</span>{_bar(top, prompt)}</div>"
    )


def twohop_grid(
    model: dict,
    prompt_idx: int,
    prompt: dict,
    positions: list[int],
    title: str,
    read_layers: set[int],
) -> str:
    pos_names = ["h", "r1", "r2"]
    head = "".join(
        f"<th>{esc(pos_names[p])}<br><span class=lgtok>{esc(prompt['tokens'][p])}</span></th>"
        for p in positions
    )
    rows = []
    for layer, cells in enumerate(model["layers"]):
        mark = ' class="lgread"' if layer in read_layers else ""
        tds = "".join(
            f"<td>{_cell(cells[prompt_idx][p]['top'], prompt)}</td>" for p in positions
        )
        rows.append(f"<tr{mark}><th>L{layer}</th>{tds}</tr>")
    return (
        f'<div class="lggrid"><div class="lgtitle">{esc(title)}</div>'
        f'<table class="lgtable"><thead><tr><th></th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def legend() -> str:
    items = [
        ("bridge", "bridge entity (the intermediate)"),
        ("answer", "final answer"),
        ("input", "one of the prompt's own tokens"),
        ("other", "any other token"),
    ]
    out = "".join(
        f'<span class="lgkey"><svg width="14" height="10"><rect width="14" height="10" '
        f'fill="{c}"/></svg> {esc(t)}</span>'
        for role, t in items
        for c in [ROLE_COLORS[role]]
    )
    out += (
        f'<span class="lgkey"><svg width="14" height="10"><rect width="14" height="10" '
        f'fill="{REST}"/></svg> rest of the probability</span>'
    )
    return f'<p class="lglegend">{out}</p>'


STYLE = """<style>
.lgwrap{display:flex;gap:1rem;flex-wrap:wrap;align-items:flex-start}
.lggrid{flex:1 1 13rem;min-width:12rem}
.lgtitle{font-weight:600;font-size:.9rem;margin-bottom:.3rem}
.lgtable{font-size:.75rem;display:table;width:100%}
.lgtable th,.lgtable td{padding:.15rem .3rem;border-bottom:1px solid var(--grid);vertical-align:middle}
.lgtable thead th{font-weight:600;text-align:left}
.lgtok{font-weight:400;color:var(--ink-2)}
.lgcell{min-width:5.5rem}
.lglbl{display:block;white-space:nowrap;line-height:1.2}
.lgbar{display:block;width:100%;height:8px}
.lg-bridge{color:light-dark(#1f63b5,#6ea8ef)} .lg-answer{color:light-dark(#b8461b,#f08a5d)}
.lg-input,.lg-other{color:var(--ink-2)}
tr.lgread th:first-child{box-shadow:inset 3px 0 0 light-dark(#1baf7a,#199e70)}
.lglegend{font-size:.85rem;color:var(--ink-2)} .lgkey{margin-right:1rem;white-space:nowrap}
.lgcontrols{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0;font-size:.9rem}
.lgcontrols label{max-width:100%;display:flex;gap:.3rem;align-items:center}
.lgcontrols select{max-width:100%;min-width:0;font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--grid);border-radius:6px;padding:.1rem .3rem}
.lmtable{font-size:.72rem;display:block;overflow-x:auto}
.lmtable td,.lmtable th{padding:.1rem .25rem;white-space:pre;border-bottom:1px solid var(--grid)}
.lmtable thead th{font-weight:600}
</style>"""


def twohop_explorer(data: dict, models: list[tuple[str, str]], reads: dict) -> str:
    """Selector + grid for any model / prompt, drawn by JS from embedded data."""
    compact = {
        "prompts": data["prompts"],
        "models": [
            {
                "label": label,
                "read": sorted(reads[name]),
                "layers": [
                    [[cell["top"][:3] for cell in ex] for ex in layer]
                    for layer in data["models"][name]["layers"]
                ],
            }
            for name, label in models
        ],
        "colors": ROLE_COLORS,
        "rest": REST,
    }
    model_opts = "".join(
        f'<option value="{i}">{esc(label)}</option>'
        for i, (_, label) in enumerate(models)
    )
    prompt_opts = "".join(
        f'<option value="{i}">prompt {i + 1}: {esc(" ".join(p["tokens"]))}</option>'
        for i, p in enumerate(data["prompts"])
    )
    return f"""<div class="lgcontrols">
<label>Model <select id="lg-model">{model_opts}</select></label>
<label>Prompt <select id="lg-prompt">{prompt_opts}</select></label></div>
<div id="lg-explorer"><p><em>Interactive on the web page: pick a model and prompt to see
the lens at every layer and all three positions.</em></p></div>
<script type="application/json" id="lg-data">{json.dumps(compact, separators=(",", ":"))}</script>
<script>
(function () {{
  var D = JSON.parse(document.getElementById("lg-data").textContent);
  var mSel = document.getElementById("lg-model"), pSel = document.getElementById("lg-prompt");
  function esc(s) {{ return String(s).replace(/[&<>"]/g, function (c) {{ return {{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]; }}); }}
  function role(tok, p) {{
    if (tok === p.bridge) return ["bridge", "bridge"];
    if (tok === p.answer) return ["answer", "answer"];
    var names = ["h", "r1", "r2"];
    for (var i = 0; i < 3; i++) if (tok === p.tokens[i]) return ["input", "input " + names[i]];
    return ["other", tok.replace(/[<>]/g, "")];
  }}
  function cell(top, p) {{
    var r = role(top[0][0], p), x = 0, bars = "";
    top.forEach(function (t) {{
      var rr = role(t[0], p);
      bars += '<rect x="' + x + '" width="' + (t[1] * 100) + '" height="10" fill="' + D.colors[rr[0]] + '"><title>' + esc(rr[1]) + " " + (t[1] * 100).toFixed(1) + '%</title></rect>';
      x += t[1] * 100;
    }});
    bars += '<rect x="' + x + '" width="' + (100 - x) + '" height="10" fill="' + D.rest + '"/>';
    var w = (r[0] === "bridge" || r[0] === "answer") ? 600 : 400;
    return '<div class="lgcell"><span class="lglbl lg-' + r[0] + '" style="font-weight:' + w + '">' + esc(r[1]) + " " + (top[0][1] < 0.005 ? "&lt;1" : Math.round(top[0][1] * 100)) + '%</span><svg viewBox="0 0 100 10" preserveAspectRatio="none" class="lgbar">' + bars + "</svg></div>";
  }}
  function draw() {{
    var m = D.models[+mSel.value], pi = +pSel.value, p = D.prompts[pi], names = ["h", "r1", "r2"];
    var h = "<tr><th></th>" + [0, 1, 2].map(function (i) {{ return "<th>" + names[i] + '<br><span class="lgtok">' + esc(p.tokens[i]) + "</span></th>"; }}).join("") + "</tr>";
    var rows = m.layers.map(function (layer, li) {{
      var cls = m.read.indexOf(li) >= 0 ? ' class="lgread"' : "";
      return "<tr" + cls + "><th>L" + li + "</th>" + layer[pi].map(function (top) {{ return "<td>" + cell(top, p) + "</td>"; }}).join("") + "</tr>";
    }}).join("");
    document.getElementById("lg-explorer").innerHTML = '<div class="lggrid"><div class="lgtitle">' + esc(m.label) + " — bridge " + esc(p.bridge) + ", answer " + esc(p.answer) + '</div><table class="lgtable"><thead>' + h + "</thead><tbody>" + rows + "</tbody></table></div>";
  }}
  mSel.addEventListener("change", draw); pSel.addEventListener("change", draw); draw();
}})();
</script>"""


def _show(tok: str) -> str:
    return tok.replace("\n", "↵").replace("\t", "→")


def lm_table(prompt: dict, n_rows: int = 15) -> str:
    """Rows: executions at evenly spaced depths; columns: positions; cell: top-1."""
    layers = prompt["layers"]
    depth = len(layers) - 1
    picks = sorted({round(i * depth / (n_rows - 1)) for i in range(n_rows)})
    head = "".join(f"<th>{esc(_show(t))}</th>" for t in prompt["tokens"])
    rows = []
    for li in reversed(picks):
        tds = []
        for top in layers[li]:
            tok, p = top[0]
            alpha = min(max(p, 0.0), 1.0)
            hover = esc(" | ".join(f"{t!r} {q:.0%}" for t, q in top))
            tds.append(
                f'<td title="{hover}" style="background:rgba(42,120,214,{alpha * 0.8:.2f})">{esc(_show(tok))}</td>'
            )
        rows.append(f"<tr><th>{li / depth:.0%}</th>{''.join(tds)}</tr>")
    return (
        '<table class="lmtable"><thead><tr><th>depth</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def lm_explorer(models: list[tuple[str, str, dict]]) -> str:
    """Selector + logit-lens table for any IFM model / prompt."""
    compact = [
        {
            "label": label,
            "prompts": [
                {
                    "tokens": p["tokens"],
                    "layers": [[c[:5] for c in layer] for layer in p["layers"]],
                }
                for p in data["prompts"]
            ],
        }
        for _, label, data in models
    ]
    first = models[0][2]["prompts"]
    model_opts = "".join(
        f'<option value="{i}">{esc(label)}</option>'
        for i, (_, label, _) in enumerate(models)
    )
    prompt_opts = "".join(
        f'<option value="{i}">{esc("".join(p["tokens"])[:60])}…</option>'
        for i, p in enumerate(first)
    )
    return f"""<div class="lgcontrols">
<label>Model <select id="lm-model">{model_opts}</select></label>
<label>Prompt <select id="lm-prompt">{prompt_opts}</select></label></div>
<div id="lm-explorer"><p><em>Interactive on the web page: pick a model and prompt.</em></p></div>
<script type="application/json" id="lm-data">{json.dumps(compact, separators=(",", ":"))}</script>
<script>
(function () {{
  var D = JSON.parse(document.getElementById("lm-data").textContent);
  var mSel = document.getElementById("lm-model"), pSel = document.getElementById("lm-prompt");
  function esc(s) {{ return String(s).replace(/\\n/g, "↵").replace(/\\t/g, "→").replace(/[&<>"]/g, function (c) {{ return {{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]; }}); }}
  function draw() {{
    var p = D[+mSel.value].prompts[+pSel.value], depth = p.layers.length - 1, picks = [];
    for (var i = 0; i < 15; i++) {{ var v = Math.round(i * depth / 14); if (picks.indexOf(v) < 0) picks.push(v); }}
    var head = "<tr><th>depth</th>" + p.tokens.map(function (t) {{ return "<th>" + esc(t) + "</th>"; }}).join("") + "</tr>";
    var rows = picks.reverse().map(function (li) {{
      return "<tr><th>" + Math.round(li / depth * 100) + "%</th>" + p.layers[li].map(function (top) {{
        var hover = top.map(function (t) {{ return JSON.stringify(t[0]) + " " + Math.round(t[1] * 100) + "%"; }}).join(" | ");
        return '<td title="' + esc(hover) + '" style="background:rgba(42,120,214,' + (Math.min(top[0][1], 1) * 0.8).toFixed(2) + ')">' + esc(top[0][0]) + "</td>";
      }}).join("") + "</tr>";
    }}).join("");
    document.getElementById("lm-explorer").innerHTML = '<table class="lmtable"><thead>' + head + "</thead><tbody>" + rows + "</tbody></table>";
  }}
  mSel.addEventListener("change", draw); pSel.addEventListener("change", draw); draw();
}})();
</script>"""
