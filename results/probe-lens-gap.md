# How much does the logit lens see? (linear probes vs. the lens)

Added after the writeup; not part of the original experiment log.

## Question

The staircase tables report the logit lens's top-1 accuracy. That doesn't say
how much of what the residual actually holds the lens recovers. Here a
supervised linear probe (multinomial logistic regression on the final-normed
residual, 5000 training / 2000 held-out examples, k = 6) is the reference for
"linearly present". The lens is compared against it.

The lens can only tell the six group elements apart through a 5-dimensional
subspace: the span of the six element embeddings, minus their mean (softmax
ignores a shift shared by every logit). Everything else in the 96-dimensional
residual is invisible to it ("dark"). So probes are also trained on:

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

The probe confirms the staircase. It also shows that the lens sees only part
of each step, and only at one layer.

At the `<op>` after operand j (the `t[6]` column is the `<predict>` position):

| | t[1] | t[2] | t[3] | t[4] | t[5] | t[6]/ans |
|---|---|---|---|---|---|---|
| Layer where the probe first reaches ≥ 99% | L1 | L1 | L3 | L4 | L5 | L6 |
| Layer where the lens peaks | L1 | L2–L3 | L3 | L4 | L5 | L6 |
| Lens accuracy at its peak | 44% | 55% | 70% | 60% | 47% | 100% |
| Probe accuracy at that layer | 100% | 100% | 100% | 99% | 100% | 100% |
| Lens efficiency at the peak | 0.33 | 0.46 | 0.64 | 0.53 | 0.37–0.41 | 1.00 |
| Probe accuracy at L7 | 100% | 100% | 100% | 100% | 100% | 100% |
| Lens accuracy at L7 | 16% | 17% | 18% | 17% | 17% | 100% |

*Lens efficiency* is (lens accuracy − chance) / (probe accuracy − chance),
with chance = 1/6. 1 means the lens recovers everything a linear probe can;
0 means it does no better than guessing.

1. **The timing matches.** Each intermediate becomes linearly decodable at
   the layer where the lens first sees it, or one layer earlier (t[2]). A
   probe sees no more steps than the lens does, and none earlier.
2. **Even at its best layer, the lens recovers only a third to two-thirds of
   what is there** for intermediates. For the final answer it recovers all of
   it.
3. **The lens's "erasure" is a rotation, not a deletion.** After its
   critical layer, each intermediate drops to chance under the lens. It stays
   100% linearly decodable through L7. The share of the residual's variance
   in the visible subspace falls from 7–49% at the peak (5.2% would be
   expected for a random direction) to 0.0–0.5%. RESULTS.md describes these
   states as "transient wavefronts (computed at one layer, erased at the
   next)". That is true of what the lens can see, not of the residual. With
   the critical-layer ablation (zeroing after the critical layer is
   harmless), the picture is: kept, but moved out of the readout directions
   and unused. A plausible reason is full-sequence loss: it trains `<op>`
   positions to predict a uniform next element, so the state has to be hidden
   from the unembedding.
4. **The visible directions are only modestly special.** At the peaks, a
   probe on the visible 5-d subspace beats a random 5-d subspace (t[4] at
   L4: 74% vs 45%; t[1] at L1: 98% vs 89%). A probe on the dark 91-d part
   alone always matches the full probe. The lens-visible signal is an extra
   copy of a state the model also holds in directions the lens can't read.

At the `<predict>` position (the per-run table at the end of the output),
Model A also builds up t[3] (78–87% from L4), t[5] (92–96% from L6) and the
last two pairs g5·g4 / g6·g5 (73–92% from L1). That position does part of
the work too.

## Model B (full-sequence loss only)

No intermediate is linearly decodable anywhere we looked:

- **`<op>` positions:** t[1] = g1·e0 (which takes one attention step) is
  100% from L0. t[2]…t[5] stay at 15–24% at every layer, in the full
  residual and in every subspace. The product of the operands without the
  start element is also at chance for j ≥ 3. Products of adjacent pairs
  g_j·g_(j−1) are partly decodable (peaks of 35–72%).
- **`<predict>` position:** among products of three or more consecutive
  inputs, none reaches 28% except the full answer. The answer goes from 44%
  at L3 to 89% at L4 and 99% at L5, the same L4 cliff as the ablation
  sweep. Single adjacent pairs peak at 32–58%.

So Model B's lens shows nothing at intermediate positions because nothing is
linearly there to show, not because the lens is blind. The intermediates
aren't hidden in the dark subspace either: the full-residual probe is also at
chance. This fits a model that computes the answer in one go around L3–L4
from pair-level features.

Caveats: these are linear probes only. A partial product could be encoded
nonlinearly, for example split into S3's sign and rotation parts. Element
positions weren't probed for partial products. RESULTS.md (Phase 12) reports
that nonlinear MLP probes also failed on earlier full-sequence-loss models,
but that code isn't in this repo and wasn't run on these checkpoints.

## Takeaway for "how good is the logit lens"

On Model A, the lens is a good detector of **when** a state is written:
same layer as the probe, or within one. It is a poor measure of **whether** a
state is present. At its best layer it recovers 33–64% of the
chance-adjusted signal for intermediates. At every later layer it reads
chance while the state is still fully decodable. For a model like Model B,
where the lens shows nothing, the probes say the lens isn't missing anything
linear.
