# How much does the logit lens see? (linear probes vs. the lens)

Added after the writeup; not part of the original experiment log.

## Question

The staircase tables report the logit lens's top-1 accuracy. That doesn't say
how much of what the residual holds the lens recovers. Here a supervised
linear probe is the reference for "linearly present": a multinomial logistic
regression on the final-normed residual, 5000 training / 2000 held-out
examples, k = 6. RESULTS.md (Phase 12) already noted that probes find more
than the lens. This measures by how much, where, and for which models.

The lens is scored two ways:

- **Whole vocabulary**, as in the staircase tables.
- **Element logits only**: the argmax over the six group-element logits.
  This is the fair comparison with a 6-way probe. In early layers the
  whole-vocabulary top-1 is often a structural token rather than an element
  (in Model A, at L1, only 46% of t[1] and 25% of t[2] predictions are
  elements).

The element-only lens tells elements apart through a 5-dimensional subspace:
the span of the six element embeddings minus their mean (softmax ignores a
shift shared by every logit). Everything else in the 96-dimensional residual
is exactly invisible to it ("dark"). So probes are also trained on:

- only the **visible** 5-d part;
- only the **dark** 91-d remainder;
- a **random** 5-d subspace (averaged over 5 draws), the baseline that says
  whether the visible directions are special.

```bash
uv run python -m lego.probe_lens_gap --checkpoint hf:lego/std_96d_6h_8L_kp2_s42_curriculum_FSL/step_97656.pt  # Model A
uv run python -m lego.probe_lens_gap --checkpoint hf:lego/std_96d_6h_8L_kp2_s42_fsl_only/step_156250.pt      # Model B
```

About a minute per model on CPU, deterministic. Full output:
[probe-lens-gap-output.txt](probe-lens-gap-output.txt).

## Model A (the staircase model)

At the `<op>` after operand j (the `t[6]` column is the `<predict>` position):

| | t[1] | t[2] | t[3] | t[4] | t[5] | t[6]/ans |
|---|---|---|---|---|---|---|
| Layer where the probe first reaches ≥ 99% | L1 | L1 | L3 | L4 | L5 | L6 |
| Layer where the element-only lens peaks | L1 | L1 | L3 | L4 | L5 | L6 |
| Element-only lens at that layer | 84% | 63% | 70% | 61% | 47% | 100% |
| Whole-vocabulary lens at that layer | 44% | 20% | 70% | 60% | 47% | 100% |
| Lens efficiency there | 0.81 | 0.55 | 0.64 | 0.54 | 0.37 | 1.00 |
| Probe at L7 | 100% | 100% | 100% | 100% | 100% | 100% |
| Element-only lens at L7 | 16% | 17% | 18% | 17% | 17% | 100% |

*Lens efficiency* = (element-only lens accuracy − chance) / (probe accuracy −
chance), with chance = 1/6. 1 means the lens recovers everything a linear
probe can; 0 means it does no better than guessing.

1. **The timing matches exactly.** Each intermediate becomes linearly
   decodable (≥ 99%) at the same layer where the element-only lens peaks.
   Before that layer the probe sometimes reads a partial signal the lens also
   partly shows (t[3] at L1–L2: probe 47–55%, element-only lens 37%).
2. **At that layer, the lens recovers 37–81% of the chance-adjusted signal**
   for intermediates, and all of it for the answer. Most of the gap between
   the writeup's staircase numbers and the probe at t[1]–t[2] is the
   whole-vocabulary lens picking a non-element token.
3. **After its layer, an intermediate is no longer readable by the lens, but
   it is still in the residual.** The lens drops to chance while the probe
   stays at 100% through L7. The residual's variance in the visible subspace
   shrinks over the next few layers: from 7–49% at each state's layer (5.2%
   for a random direction) to ≤ 0.5% by L5 for t[1]–t[4], and 0.0% at L6–L7
   for all five. It doesn't vanish completely. A probe on the visible part
   alone still reads 38–81% at L7, from very low-variance directions that the
   lens's fixed readout doesn't pick up. RESULTS.md calls these
   "transient wavefronts (computed at one layer, erased at the next)". That
   describes what the lens sees, not the residual. With the critical-layer
   ablation (zeroing after the critical layer is harmless), the states are
   kept but no longer used. A plausible reason they are hidden from the
   unembedding is full-sequence loss, which trains `<op>` positions to
   predict a uniform next element.
4. **The visible directions are modestly special, and only at the step's
   layer.** There, a probe on the visible 5-d subspace beats a random 5-d
   subspace: t[2] at L1 79% vs 59%, t[4] at L4 74% vs 45%, t[1] at L1 98% vs
   89%. By L7 it does no better than random, except t[1]. A probe on the dark
   91-d part alone is always within 3 points of the full probe. The model
   holds every state in directions the lens can't read; the lens-visible part
   is an extra copy.

**Weighting by how strongly the lens reads each direction** changes little
here. The *relative visibility* is ‖W y‖² / ‖y‖² over the residual's
variation y, divided by the same quantity for a random direction. W is the
mean-subtracted element unembedding, and 1 means "as visible as random". The
five nonzero singular values of W are fairly even (7.5–10.7), so this tracks
the visible-variance share: 1.3–10× random at each state's layer, ≤ 0.1 by L5
for t[1]–t[4].

At the `<predict>` position (the per-run table at the end of the output),
Model A's residual also linearly encodes t[3] (78–87% from L4), t[5] (92–96%
at L6–L7), and the last two pairs g5·g4 (68–83% from L1) and g6·g5
(83–92%). That says the information is present there. It doesn't say
whether `<predict>` computes it or copies it via attention from the `<op>`
positions.

## Model B (full-sequence loss only)

No intermediate is linearly decodable at any position probed:

- **`<op>` positions:** t[1] = g1·e0 (which takes one attention step) is 89%
  at L0 and 100% from L1. t[2]…t[5] stay near chance (≤ 24%) at every
  layer, in the full residual and in every subspace. The product of the
  operands without the start element is near chance for j ≥ 3. Adjacent pairs
  g_j·g_(j−1) are partly decodable (peaks of 35–72%). For j = 1 these
  alternative targets all reduce to g1 and are trivially 100%.
- **`<predict>` position:** among products of three or more consecutive
  inputs, none reaches 28% except the full answer. The answer goes from 44%
  at L3 to 89% at L4 and 99% at L5, the same L4 cliff as the ablation
  sweep. Single adjacent pairs peak at 32–58%. Here the element-only lens
  tracks the probe fairly well: 41% / 72% / 97% at L3 / L4 / L5, efficiency
  0.76–0.99.

So for intermediates, Model B's lens shows nothing because nothing is
linearly there to show, not because the lens is blind. The dark subspace
holds nothing either: the full-residual probe is also near chance. This fits
a model that computes the answer in one go around L3–L4 from pair-level
features.

Caveats: these are linear probes only. A partial product could be encoded
nonlinearly, for example split into S3's sign and rotation parts. Element
positions weren't probed for partial products. RESULTS.md (Phase 12) reports
that nonlinear MLP probes also failed on earlier full-sequence-loss models,
but that code isn't in this repo and wasn't run on these checkpoints.

## Takeaway for "how good is the logit lens"

On Model A, the element-only lens detects **when** a state is written: at
the same layer as the probe. It is only a partial measure of **how much**:
37–81% of the chance-adjusted signal at that layer. It says nothing about
**whether** a state is still present afterwards: at every later layer it
reads chance while the state remains fully decodable. Scoring the lens over
the whole vocabulary understates it further at early layers. For Model B's
intermediates, the probes say the lens isn't missing anything linear.
