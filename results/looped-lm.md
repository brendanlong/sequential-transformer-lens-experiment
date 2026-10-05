# Are looped transformers more logit-lens-readable? Two public model pairs

Added after the writeup. Our own S3 comparison was abandoned because the looped
and standard models learned different algorithms, so any lens difference could
not be put down to weight sharing. Here the same measurements run on two
public families where someone else trained looped and standard models on the
same data:

- **Two-hop composition** (Kohli et al. 2026, [Loop, Think &
  Generalize](https://arxiv.org/abs/2604.07822)). GPT-2 blocks, width 768,
  answer-only loss, a 2,000-entity knowledge graph. The input is `<h><r1><r2>`,
  the answer is `t = r2(r1(h))`, and the *bridge* `b = r1(h)` is a vocabulary
  token that is never a training target. `r<R>_l<L>` is an L-layer stack run R
  times; `r2_l4` and the vanilla `r1_l8` have the same depth and FLOPs. On held-out
  compositions of *seen* facts ("test ID") both solve the task through the
  bridge; only looped models eventually generalise to compositions of facts
  never composed in training ("test OOD"). Their checkpoints mark three
  training stages: train fit, ID generalisation, OOD generalisation.
- **IFM LoopedLM Part I** ([blog](https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/),
  [checkpoints](https://huggingface.co/collections/IFM/towards-looped-models-done-right-6ab9f671bb1c97e33b27d14c)).
  Language models trained on the same 336B TxT360 tokens with the same
  tokenizer (250k vocabulary), width 1536, seed and schedule, loss on the final
  output only. Every loop recipe stores 28 blocks and executes 112:
  Ouro-style (all 28 blocks × 4), and prelude–loop–coda (8 + 12 × 8 + 8), each
  with and without re-injecting the input. Controls: D112 (112 distinct blocks,
  depth-matched) and D28 (28 blocks, parameter-matched). Neither IFM paper does
  any lens or probe analysis.

Code: `looped_lm/` (`twohop.py`, `ifm.py`, shared metrics in
`lens_metrics.py`, figures and tables in `report.py`, pipeline in
`looped_lm/Snakefile`). Raw results: `results/looped-lm/`.

## Summary

- **Small two-hop models (same data, same depth, same algorithm in
  distribution): looping makes the intermediate more lens-readable, earlier.**
  The bridge entity is linearly present at the first relation from L3 in both
  the vanilla 8-layer model and its looped 4×2 twin, but the lens can read it
  four layers later in the vanilla model and two layers later in the looped
  one; with 4 or 8 iterations there is no lag at all and it stays readable to
  the last layer. Mid-depth, the looped residual there is close to a single
  token direction — the bridge — while the vanilla residual is close to no
  token.
- **But the copy the second hop actually uses is dark in every model.** At the
  second relation the bridge is 95–100% linearly decodable and 0–2% readable
  by the lens, looped or not, and not because it sits in weakly read
  directions.
- **1.5–3B language models (same 336B tokens and recipe): looping does not
  help.** Past ~75% of depth both standard models are more lens- and
  tuned-lens-readable than every loop design; prelude–loop–coda models with 8
  iterations are the least readable; loop boundaries carry no readable
  intermediate prediction. In all six, most of the normed residual's raw
  energy sits in the most weakly read quarter of directions (a large component
  shared by every token), and the token-varying part is never close to a few
  token directions.

So the logit lens can favour weight sharing on a small algorithmic task where
both models use the same algorithm, but the effect does not carry over to the
language-model suite, and in neither family does looping make the
*consumed* intermediate state visible.

## Measurements

At every effective layer (every block *execution* for looped models):

- **Logit lens**: final norm + unembedding applied to the residual. For
  two-hop, top-1 accuracy for the bridge at the `r1` position (their Figure 4),
  the bridge at the `r2` position, and the answer at `r2`. For the LMs, KL from
  the final distribution and top-1 agreement with the final prediction.
- **Probes**, the reference for "linearly present". Two-hop: a 2,000-way
  linear probe on the normed residual, trained on 20,000 training compositions
  (also a probe through the frozen unembedding, which agrees within a few
  points). LMs: a ridge-regression map from each layer's normed residual to the
  final one, read out through the model's head — a closed-form tuned lens.
- **Dark subspace**: the share of the normed residual's energy in the quarter
  of directions the readout (unembedding with the norm gain folded in) reads
  most weakly. 25% for an isotropic vector.
- **How many tokens**: orthogonal matching pursuit of the batch-centered
  normed residual over the unit-norm unembedding rows, as R²(k) for k = 1…8,
  against a Gaussian null with the same covariance; plus which token OMP picks
  first.

## Two-hop results

![Lens vs probe on the two-hop models](looped-lm/figures/twohop-lens-vs-probe.png)

Test ID (held-out compositions of seen facts), 2,000 examples. "Probe ≥ 90%
/ lens ≥ 90%" is the first layer at which each reaches 90% top-1 (L0 =
embeddings); train and test-OOD versions are in
[looped-lm/twohop-tables.md](looped-lm/twohop-tables.md).

| Model | answer acc. | bridge@r1: probe ≥ 90% / lens ≥ 90% | bridge@r1 lens: peak / last layer | target@r2: probe ≥ 90% / lens ≥ 90% | bridge@r2: best probe / best lens |
|---|---|---|---|---|---|
| vanilla 8L, train stage (L8) | 1% | never / never | 74% at L8 / 74% | never / never | 21% / 2% |
| vanilla 8L, ID stage (L8) | 99% | L3 / L7 | 96% at L7 / 96% | L7 / L6 | 97% / 0% |
| vanilla 4L, train stage (L4) | 1% | never / never | 1% at L4 / 1% | never / never | 12% / 1% |
| vanilla 4L, ID stage (L4) | 98% | L2 / L3 | 100% at L4 / 100% | L3 / L3 | 95% / 0% |
| looped 4L×2, train stage (L8) | 1% | never / never | 17% at L8 / 17% | never / never | 10% / 1% |
| looped 4L×2, ID stage (L8) | 99% | L3 / L5 | 99% at L5 / 56% | L6 / L5 | 95% / 0% |
| looped 4L×2, OOD stage (L8) | 100% | L2 / L1 | 100% at L2 / 100% | L6 / L5 | 100% / 0% |
| looped 4L×4, train stage (L16) | 1% | never / never | 58% at L16 / 58% | never / never | 15% / 0% |
| looped 4L×4, ID stage (L16) | 99% | L4 / L4 | 100% at L5 / 100% | L9 / L9 | 99% / 0% |
| looped 4L×4, OOD stage (L16) | 100% | L4 / L3 | 100% at L4 / 100% | L8 / L7 | 98% / 0% |
| looped 4L×8, train stage (L32) | 2% | never / never | 46% at L32 / 46% | never / never | 17% / 0% |
| looped 4L×8, ID stage (L32) | 99% | L4 / L4 | 100% at L6 / 100% | L21 / L18 | 100% / 0% |
| looped 4L×8, OOD stage (L32) | 100% | L4 / L5 | 100% at L9 / 73% | L20 / L19 | 100% / 0% |

![Readout geometry on the two-hop models](looped-lm/figures/twohop-geometry.png)

1. **Their Figure 4 reproduces.** On the depth- and FLOP-matched pair at the ID
   stage, the lens finds the bridge at `r1` mid-depth in the looped model and
   only near the top in the vanilla one, and on OOD inputs the vanilla model
   recovers the bridge (69% at L7) but never the answer (1%).
2. **In the looped models the lens sees the bridge as soon as it is there; in
   the vanilla ones it lags.** The vanilla 8-layer model has the bridge linearly
   decodable (≥ 90%) from L3 but lens-readable only from L7: four layers in
   which it is present but invisible to the lens. Its depth-matched looped twin
   lags by two (L3 → L5), and with 4 or 8 iterations the lag is zero (L4 → L4).
   For the answer at `r2` the lens reaches 90% no later than the probe in
   every model, though in the vanilla 8-layer model the probe already reads a
   partial signal two layers earlier (34–44% at L4–5, lens 0–9%); in the looped
   twin the lens matches or leads the probe at every layer.
3. **Once readable, it stays readable — mostly.** With 4 or 8 iterations the
   bridge at `r1` stays at 100% lens accuracy to the last layer (the state
   settles into a fixed point). The 2-iteration ID-stage model behaves like our
   S3 Model A: the lens peaks at L5 (99%) and fades to 56% by L8 while the probe
   stays at 100%. The 8-iteration OOD-stage model also fades (100% → 73%).
4. **The bridge where it is used is dark in every model.** At the `r2`
   position, where the second hop has to read it, the bridge is linearly
   decodable at 95–100% in all ID-stage models and lens-readable in none (≤ 2%
   in all 13 checkpoints). It is not hidden in low-gain directions: a probe on
   only the half of the residual the readout reads most strongly decodes it as
   well as one on the weakly read half (e.g. 97% vs 98%, 4L×4 at L8). It is
   simply not written along the bridge's own unembedding row. Weight sharing
   does not change this.
5. **How many tokens: one, and only in the looped models mid-depth.** At `r1`,
   over the middle half of depth, the single best token direction explains
   43–46 points more variance than for Gaussian noise in the 4- and 8-iteration
   ID-stage models (17 for 2 iterations, 59–77 at the OOD stage), and it is the
   bridge for 71–100% of examples. In the vanilla 8-layer model it explains 3
   points (though it is still the bridge for 43% of examples); the residual is
   not near any token until the last layer (≈ 45).
   Where token content is present, R²(k) is flat after k = 1: one token, not a
   handful. At `r2` mid-depth the first token picked is the eventual answer in
   20–64% of examples depending on the model, but it explains little variance
   (≤ 0.37 at the OOD stage, ≤ 0.26 at the ID stage).
6. The token-varying part of the residual has relative visibility ≈ 1 through
   most of depth in every model (0.7–0.8 for the 4L×4 OOD-stage model at `r1`),
   so there is no systematic looped/vanilla difference in how much variance sits
   in weakly read directions.

## IFM results

![IFM per-depth lens metrics](looped-lm/figures/ifm.png)

128 FineWeb-Edu documents × 64 positions each (from token 16 on); ridge maps
fit on half, everything else measured on the other 4,096 tokens.

| Model | final next-token top-1 | lens = final top-1 at 50% / 75% / 90% depth | ridge-tuned lens = final top-1 at 50% / 75% / 90% | KL lens / tuned at 75% (nats) | raw energy in weakest quarter at 50% | OMP R²(1) − null, peak |
|---|---|---|---|---|---|---|
| D112 | 46.8% | 1% / 18% / 33% | 26% / 52% / 67% | 6.7 / 1.3 | 80% | 0.046 at 88% |
| D28 | 43.6% | 3% / 18% / 33% | 32% / 54% / 67% | 6.3 / 1.0 | 76% | 0.037 at 89% |
| Ouro R28×4 | 44.5% | 3% / 11% / 22% | 37% / 52% / 61% | 5.5 / 1.3 | 83% | 0.038 at 94% |
| Ouro R28×4 + injection | 45.0% | 9% / 17% / 15% | 33% / 46% / 49% | 4.8 / 1.8 | 64% | 0.039 at 96% |
| P8 R12×8 C8 | 43.9% | 3% / 6% / 12% | 29% / 40% / 49% | 7.7 / 2.1 | 82% | 0.037 at 96% |
| Huginn P8 R12×8 C8 + injection | 45.0% | 3% / 7% / 14% | 22% / 38% / 47% | 6.7 / 2.2 | 55% | 0.038 at 96% |

1. **Looping does not make the LMs more lens-readable overall.** Through the
   first ~60% of depth the Ouro-style models are slightly *more* readable than
   the standard ones (at 50% depth: lens agreement 3–9% vs 1–3%, ridge-tuned
   33–37% vs 26–32%), but everything is low there. From about 75–80% of depth
   on, where the prediction actually forms, both standard models lead every
   looped model (at 90% depth: lens 33% for D112 and D28 vs 12–22%; tuned 67%
   vs 47–61%). The prelude–loop–coda models, which loop more (8 iterations of 12
   blocks), are the least readable at every depth past the middle. The final
   models are within 3 points of each other on next-token accuracy.
2. **No readout at loop boundaries.** The Ouro-style model's state at the end
   of loops 1–3 is the output of the very block that feeds the head, yet the
   lens is no better there than at neighbouring blocks (agreement 2%, 3%, 11%
   at executions 28, 56, 84). With loss only on the final output nothing
   forces intermediate loops to be readable — unlike ByteDance's Ouro, which
   trains a loss at every loop.
3. **Dark subspace: the shared part of the residual is dark, the token-varying
   part is not.** In every model, 55–85% of the raw normed residual's energy
   through the middle of the network lies in the most weakly read quarter of
   directions — a large component common to all tokens that the lens is
   nearly blind to. With that common component removed, the variance is close
   to isotropic (relative visibility ≈ 1) until the last third, where it moves
   toward strongly read directions — earlier for D112/D28 than for the loops.
   Input injection shows up as a sawtooth: each injection pushes the dark
   share back up to ~85–90%, after which the blocks drain it.
4. **The residual is never close to a few tokens.** One token direction
   explains at most 4.6 points of variance more than for Gaussian noise of the
   same covariance (D112, at 88% depth); eight explain at most ~10 points
   more. When OMP's first pick is a meaningful token it is the eventual
   prediction (≤ 14% of positions) rather than the current token, and only in
   the last fifth of depth. The rise comes earliest in D112/D28, latest in the
   prelude–loop–coda models.
5. Lens logit outlier counts (logits more than 6 robust SDs above the median)
   run into the hundreds for most models mid-depth, so for a 250k vocabulary
   they mostly measure heavy tails rather than "tokens under consideration";
   they are in the JSONs but not used above.

## Caveats

- Two-hop probes are trained on training compositions; on OOD inputs a probe
  can score *below* the lens (the looped ID-stage model at L4–5), so on OOD
  "lens efficiency" is not bounded by 1.
- FineWeb-Edu overlaps TxT360's Common Crawl sources, so some evaluation text
  may have been seen in training. That affects all IFM models alike.
- One seed per recipe in both families. Kohli et al. released one run per
  configuration; IFM trained each recipe once.
- IFM D28 and the loop models are at step 80,000 of a 119,210-step schedule
  (mid-decay); that is the only checkpoint released for D28 and the Ouro-style
  models, so all six are compared there.
