# LEGO Composition: Experiment Results

> **Note for the public release.** This is the experiment log as it was
> written during the research (the provenance record for the writeup,
> [Training a Transformer to Compose One Step Per Layer (and Proving It)](https://www.lesswrong.com/posts/QEbC3t4XpLsiwhRqg/training-a-transformer-to-compose-one-step-per-layer-and)),
> kept verbatim. The commands were run in a private research monorepo; to
> map them onto this repo:
>
> - `experiments.lego.X` → `lego.X`; `./train.sh local lego -- ARGS` and
>   `./train.sh remote [--module M] lego -- ARGS` (SkyPilot wrappers) →
>   `uv run python -m lego.train ARGS` / `uv run python -m M ARGS`.
> - `--save-checkpoint` performed a private S3 upload and has no equivalent
>   here (checkpoints always save locally). `s3://…` URIs refer to that
>   private store; public copies of the checkpoints behind the writeup are on
>   the [HF dataset](https://huggingface.co/datasets/brendanlong/sequential-transformer-lens-experiment)
>   under friendlier run names (the mapping is in the dataset card and
>   `README.md`).
> - wandb run IDs link into the
>   [public project](https://wandb.ai/brendanlong-com/lego-reasoning).
> - **This release ships only the code the writeup needs**: the task
>   (`lego/generator.py`, `lego/tokenizer.py`, `lego/data.py`), the standard
>   and weight-shared models, `lego/train.py`, `lego/curriculum_ao_to_fsl.py`,
>   the logit lens (`lego/analyze_logit_lens.py`) and the two ablations
>   (`lego/ablate_critical_layer.py`, `lego/ablate_sequential.py`). Phases
>   that used other code — the tuned lens, per-head attribution, supervised
>   probes, the staircase / alignment / lens-aux auxiliary losses, the S4/S5
>   groups' training runs, the evar / entropy / principal-angle metrics,
>   and the 2026-08 hold-out generalization probes — are documented below for
>   completeness but their scripts are not included; `--staircase-loss`,
>   `--lens-aux`, `--align-loss`, `--repel-loss`, `--holdout-k`,
>   `--eval-k-max` and `--pos-encoding pope` are not flags in this repo.
>
> The writeup's Model A is the `Std 96d/6h/8L kp=2, seed 42` pair
> (`7gwt74kt` → `clzbke38`) and Model B is the `FSL-only kp=2, seed 42` run
> (`b5mcg9s9`), both in [Phase 16](results/13-k-weighted-curriculum.md).

Based on ["Unveiling Transformers with LEGO"](https://arxiv.org/abs/2206.04301) (Zhang et al., 2022).

## Current Best Understanding

1. **Weight sharing forces genuinely compositional algorithms.** Because the shared MLP can only learn the 36-entry S3 group table (one-step composition), multi-step memorization shortcuts are structurally impossible. Standard models with unique-per-layer MLPs can and do learn multi-step shortcuts when they have enough capacity. However, k-power=2 data weighting can also push standard models toward one-hop-per-layer composition.

2. **Full-sequence loss destroys intermediate state visibility regardless of architecture.** When every position is trained to predict its next token, gradient pressure pushes trajectory information orthogonal to the unembedding basis. This is architecture-independent -- even weight sharing cannot overcome it.

3. **Curriculum training (AO then FSL) preserves compositional structure.** Models trained first with answer-only loss and then switched to full-sequence loss retain the compositional algorithm. Intermediate states become transient wavefronts rather than persistent, but the sequential structure survives.

4. **Neither architecture keeps residuals "in" the embedding space.** Embedding explained variance (fraction of residual norm in the span of token embeddings) is only 16-30% for all models -- the vast majority of the residual stream is in directions invisible to the logit lens. The tuned lens gap reflects subspace rotation, not embedding proximity. Weight sharing does NOT produce more embedding-aligned representations (Phase 17).

5. **The model uses `<op>` tokens as compute registers.** Probing all positions reveals that intermediate states are stored exclusively at `<op>` positions, never at element positions. Element positions preserve raw input for later layers' attention. FSL models erase intermediates to a fixed attractor token (`r2s`) before converging to uniform.

6. **A standard model computes an unseen composition depth (2026-08-17).** With
   k=4 held out of training entirely, a 128d/4h/8L standard model with RoPE
   solves it at **94.4%** while every trained length sits at 100% — so the
   algorithm is not a per-k lookup table. Two cautions on reading this against
   point 1: the matched weight-shared arms did *not* generalize, but they also
   failed to fit their trained distribution, so they are not evidence about
   compositionality; and the result requires a **relative** position encoding
   (the same run with learned position embeddings gets 35.9%, and fails its
   trained lengths too). n=1 per cell.

7. **The short-chain curriculum must be contiguous (2026-08-17).** Every arm
   learns k<=3 perfectly and then collapses at k>=4 once k=4 is removed —
   including the trained k=5/k=6 in 3 of 4 cells. A hole in the curriculum
   blocks the lengths above it, which sharpens the existing finding that a
   short-chain curriculum is required for learnability at all.

## Task Design

S3 (symmetric group on 3 elements) is the smallest non-abelian group (6 elements: e, r, r2, s, rs, r2s). Each sequence is a chain of S3 elements to be composed left-to-right:

```
<start> e <op> r <op> s <op> r2 <predict> ANSWER
```

The model must compute g_k . ... . g_1 . start and predict the final element. Non-commutativity forces genuine sequential composition -- no shortcut possible.

We initially tried binary group composition but found the model exploited a parity shortcut (see [01-binary-group-pilot.md](results/01-binary-group-pilot.md)).

## Headline Models

Four models form the basis for most comparisons:

| Model | Config | Params | Role |
|-------|--------|--------|------|
| Large Standard | 128d/4h/8L | 1.59M | Baseline with ample capacity |
| Small Standard | 48d/3h/8L | 229K | Minimum viable standard model |
| Large WS | 256d/8h/8iter | 825K unique | WS with overcapacity (shows traveling wave) |
| Small WS | 96d/6h/8iter | 125K unique | Minimum viable WS model (most interpretable) |

All reach 100% accuracy on k=1 through k=6 under answer-only loss.

## Key Results by Experiment

### Initial S3 Training (Phase 2-3)

Standard and weight-shared transformers both solve 6-hop S3 composition. The standard model shows a persistent staircase at `<op>` positions (once an intermediate appears, it stays visible). The weight-shared 256d model shows a traveling wave (intermediates appear, peak, then decay as later iterations overwrite them). The capacity-constrained 192d WS model shows persistent states with early onset -- the best of both patterns.

**Details**: [results/02-s3-initial-training.md](results/02-s3-initial-training.md)

### Architecture Sweep (Phase 4 + 8)

Minimum viable architectures: WS needs 96d/6h/8iter (125K params) with 10M examples; standard needs 48d/3h/8L (229K params). For weight-shared models, head_dim=16-32 is critical -- too large (>=40) or too small (<=10) fails. Standard models are less sensitive to head_dim but require 8 layers. The capacity-constrained WS 192d/6h model produces the most interpretable logit lens patterns.

**Details**: [results/03-architecture-sweep.md](results/03-architecture-sweep.md)

### Convergence Analysis (Phase 5)

The traveling wave in the overcapacity WS 256d model gets *stronger* with continued training -- it is a genuine property, not an artifact. The capacity-constrained WS 192d model shows the opposite: states become more persistent with training. Overcapacity allows the shared block to be aggressive; capacity constraints force conservation.

**Details**: [results/04-convergence-analysis.md](results/04-convergence-analysis.md)

### Ablation Studies (Phase 6)

Freezing individual `<op>` positions barely affects accuracy, but freezing all simultaneously is catastrophic. Later-layer states at `<op>` positions serve as communication channels for the final-layer attention to read the answer. WS 256d is most robust to individual ablation due to redundant encoding.

**Details**: [results/05-ablation-studies.md](results/05-ablation-studies.md)

### Logit Lens Analysis (Phase 7 + 11)

Comprehensive comparison across all four headline models with both logit lens and tuned lens at op, predict, and element positions. Small WS (96d/6h) is the most interpretable: highest logit lens accuracy (44.8% avg at op positions) with the smallest tuned lens gap (+20.1%). Small Standard is most opaque (16.2% logit lens). WS models uniquely encode running state at element positions with a diagonal pattern.

**Details**: [results/06-logit-lens-analysis.md](results/06-logit-lens-analysis.md)

### Tuned Lens Analysis (Phase 9)

The tuned lens helps standard models ~2x more than weight-shared models (89.9% vs 44.0% improvement), confirming that WS models stay closer to the embedding space. The tuned lens does NOT eliminate the traveling wave in overcapacity WS models -- the information is genuinely lost, not just hidden.

**Details**: [results/07-tuned-lens-analysis.md](results/07-tuned-lens-analysis.md)

### Optimizer Comparison (Phase 10)

AdamW with weight_decay=0 is equivalent to Adam. Weight decay (0.01) helps large models slightly but hurts capacity-constrained models -- the Small WS drops from 99.8% to 95.2%. Recommendation: keep wd=0 as default.

**Details**: [results/08-optimizer-comparison.md](results/08-optimizer-comparison.md)

### Full-Sequence Loss (Phase 12 + 13)

Full-sequence loss completely destroys logit lens visibility of intermediate states in both architectures. The mechanism: `<op>` positions must predict random next tokens (uniform distribution), which actively pushes trajectory information orthogonal to the unembedding basis. This is confirmed by the grokking experiment (same 128d/4h/8L architecture, clear staircase under AO, nothing under FSL). Even nonlinear MLP probes cannot extract intermediate states from individual positions. Tiny model experiments show AO is ~13x more parameter-efficient than FSL, supporting the hypothesis that AO learns composition while FSL memorizes lookup tables.

**Details**: [results/09-full-sequence-loss.md](results/09-full-sequence-loss.md)

### Curriculum Training (Phase 14)

The compositional algorithm learned under AO is a stable attractor -- it survives 50M examples of FSL training. Intermediate states become transient wavefronts (computed at one layer, erased at the next) rather than persistent. Per-head analysis reveals the WS model uses a clean write-then-erase mechanism (H1 writes at L_j, erases at L_{j+1}), while the standard model defers all erasure to the final layers. Supervised linear probes show the "transient wavefront" from the logit lens partly underestimates actual information in the residual stream. The tuned lens is fundamentally wrong for FSL models because it learns to predict the next random token rather than the trajectory state.

**Details**: [results/10-curriculum-training.md](results/10-curriculum-training.md)

### Scaling to 24 Hops (Phase 15)

A standard 128d/4h/24L model (4.75M params) reliably learns ~20 of 24 hops with the best seed, though there is significant seed sensitivity (62.7% mean with a bad seed vs 96.1% with a good one). The bottleneck is capacity, not data. The 24L model's computation is even more opaque to the logit lens than the 8L model.

**Weight-shared models completely fail at 24 iterations.** Both 128d/4h (218K params) and 256d/8h (825K params) WS models -- which solve k=1-6 perfectly with 8 iterations -- cannot learn beyond k=1 with 24 iterations. The problem is not capacity but optimization difficulty: gradients through 24 applications of the same block appear too noisy to learn the compositional algorithm from scratch. A curriculum approach (gradually increasing depth) is a promising next step.

**Details**: [results/11-scaling-experiments.md](results/11-scaling-experiments.md)

### Minimum Layer Curriculum (Phase 15b)

The minimum viable WS model for verifiable one-hop-per-layer composition needs k+1 = 7 layers for k=6. With 6 layers, WS models show no staircase under AO -- they must use multi-step shortcuts. Standard 6L models (676K params) have enough MLP capacity for multi-step memorization. After FSL curriculum training, 7L WS models show the cleanest transient wavefront pattern (t[3] at 74% even after FSL).

**Details**: [results/12-minimum-layer-curriculum.md](results/12-minimum-layer-curriculum.md)

### K-Weighted Curriculum (Phase 16)

With k-power=2 data weighting (higher k sampled quadratically more), **standard models also learn one-hop-per-layer composition** -- the first standard transformer to show a clean staircase. This overturns the earlier claim that WS models are inherently more logit-lens-readable: the difference was partly driven by the data distribution favoring different algorithms, not purely by architecture. When both architectures learn the same algorithm, standard models' logit lens values are actually higher.

**Details**: [results/13-k-weighted-curriculum.md](results/13-k-weighted-curriculum.md)

### Enhanced Metrics Analysis (Phase 17)

New metrics reveal the logit lens is reading a tiny fraction of what the residual stream contains:

1. **Embedding explained variance (evar):** Only 16-30% of the residual's norm lies in the 10-dimensional span of token embeddings. The logit lens is projecting a ~96-dimensional vector onto a ~10-dimensional subspace. Both WS and standard models have similar evar -- weight sharing does NOT keep residuals closer to the embedding space.

2. **Logit entropy:** In FSL models, the entropy staircase reveals active erasure. At computation layers, entropy is low (18-30%). One layer later it jumps to 63-76%. The model computes, uses, then scrubs intermediates. After erasure, the logit lens converges to a fixed attractor token (`r2s`) before eventually reaching near-uniform.

3. **Principal angles:** Activation subspaces rotate gradually toward the final layer (cos θ = 0.5-0.7 at early layers → 1.0 at final layer). AO and FSL models show similar subspace rotation, and WS does not produce more stable subspaces.

4. **Position-level probing:** The model stores all intermediate states exclusively at `<op>` positions, never at element positions. Element positions carry zero trajectory signal in FSL models. This motivates the `<buf>` token experiment (Phase 18).

**Details**: [results/14-enhanced-metrics.md](results/14-enhanced-metrics.md)

### Buffer Token and 0-Hop Experiments (Phase 18)

**Buffer token (`<buf>`) — negative result.** Inserting `<buf>` tokens after each element to give element positions trivially predictable next tokens did NOT change where the model stores intermediate states. All computation remained at `<op>` positions; element and `<buf>` positions showed 0% trajectory signal. The model prefers `<op>` positions because they're structurally the last position before new input arrives, making them natural accumulator points for attention. The `<buf>` token was reverted.

**0-hop (k=0) — mildly positive.** Including k=0 (identity: answer = start element) in the training distribution produces slightly more persistent AO states and stronger FSL wavefronts, particularly for later trajectory positions (t[5]: 38% vs 20% prob mass in FSL). Now the default (k_min=0).

## Summary Table: Logit Lens Visibility

Average logit lens accuracy at `<op>` positions (layers 0-6, k=6):

| Model | Logit Lens | Tuned Lens | Gap |
|-------|-----------|-----------|-----|
| Small WS (96d/6h) | **44.8%** | 53.8% | +20.1% |
| Large WS (256d/8h) | 36.5% | 56.0% | +53.6% |
| Large Standard (128d/4h) | 32.7% | 51.2% | +56.7% |
| Small Standard (48d/3h) | 16.2% | 28.2% | +74.4% |

Weight-shared models are consistently more visible to the logit lens and have smaller tuned lens gaps, confirming they compute in a space closer to the embedding basis.

### S5 Group Experiments (Phase 19)

S5 (120 elements, non-solvable) forces genuine sequential composition -- no abelian decomposition shortcuts possible. Standard 512d/8h/8L models grok each hop sequentially (k=1 at step 38K, k=2 at 62K, k=3 at 82K) but **stall permanently at k=4** (300K steps, zero progress). Sharp dim threshold: models need dim >= 384 to learn even single-step S5 composition (vs 48 for S3). Weight-shared models are broken on S5 -- only dim=768 learns k=1, and only in isolation (not mixed-k training). Larger WS dims (1024-2048) paradoxically fail (non-monotonic).

**Details**: [results/19-s5-group-experiments.md](results/19-s5-group-experiments.md)

### S4 Group Experiments (Phase 20)

S4 (24 elements, 576-entry Cayley table) is the right difficulty level between S3 and S5. Both WS (256d/8h/8iter, 825K params) and standard (256d/8h/8L, ~6.5M params) transformers achieve 100% accuracy on k=0-6, but require ~500M streaming examples -- roughly 50x more data than S3 (which converged in ~10M, though at smaller model sizes). The grokking pattern is sequential hop-by-hop (k=0 → k=1 → ... → k=6), consistent with the model learning genuine sequential composition.

Architecture sweep showed a sharp capacity threshold: WS models below 256d could not fully grok k=1 within 50M examples (128d reached 17%, 192d reached 21%), while 256d solved it completely.

**Details**: [results/17-s4-group-experiments.md](results/17-s4-group-experiments.md)

### Residual-Embedding Alignment Loss (Phase 21)

**Negative result.** Two auxiliary losses -- soft alignment (logsumexp-based pressure to push residuals toward nearest embedding) and embedding repulsion (push element embeddings apart) -- were tested against baseline on S3 with the Large Standard model (128d/4h/8L). The alignment loss successfully pushes residuals closer to embeddings (nearest cosine 0.48 vs 0.22 at layer 7), but logit lens B top-1 accuracy **drops** from 27.3% (baseline) to 16.0% (align) and 14.7% (align+repel). Repulsion alone is a no-op (26.2%, within noise). Task accuracy is unaffected across all conditions.

The alignment loss aligns residuals to the wrong tokens -- PAD and structural tokens rather than intermediate trajectory values. Without supervision telling the model *which* token to align to, it picks the cheapest target. This parallels how full-sequence loss (Phase 12) destroys intermediate visibility: unsupervised gradient pressure pushes trajectory information orthogonal to the embedding basis. **Supervised alignment (staircase loss, Phase 16) remains the only known way to force logit lens readability.**

**Details**: [results/21-alignment-loss.md](results/21-alignment-loss.md)

## Open Questions

- **Logit lens on S4**: Does S4's larger Cayley table force cleaner sequential composition visible to the logit lens compared to S3? (Requires retraining WS model with `--save-checkpoint`.)
- **S5 k=4 wall**: Would more layers (12L, 16L), constant LR, or k-power weighting break through the k=4 barrier?
- **WS non-monotonicity**: Why does WS 768d work but 1024d+ fail on S5? Is this reproducible across seeds?
- **A5 vs S4**: A5 (60 elements, non-solvable) vs S4 (24 elements, solvable) at matched dim -- cleanest test of the solvability hypothesis independent of group size.
- **Curriculum depth scaling for WS models**: Train WS at k=1/1-iter, then incrementally add iterations and hops. WS models fail at 24 iterations from scratch but succeed at 8 -- gradual depth scaling may overcome the optimization barrier while preserving interpretability.
- Generalization: train on k<=5, test on k=6 to definitively prove compositional vs memorization strategies
- Why does the model converge to `r2s` as the erasure attractor? Is this arbitrary or does it reflect some property of the S3 group structure?
- Why does the model exclusively use `<op>` positions for computation despite element positions also having causal access to the needed information?

## 2026-07-22 — Lens-aux loss (grok_lens Phase 5): see grok_lens/RESULTS.md

Added `--lens-aux` (per-layer logit-lens CE against the final answer at
the `<predict>` position). Three matched runs (base / uniform / linear,
seed 42, jobs 69–71, wandb `lego-reasoning`, run names
`S3-std-8L-lensaux-*`): all 100% at k ≤ 6; the aux loss front-loads the
computation (answer readable at layer ~k/2 instead of the last layers)
without hiding intermediate states. Full entry, commands, and analysis
(`compare_lens_aux.py`) documented in
[grok_lens/RESULTS.md](https://github.com/brendanlong/lens-loss-grokking-experiment/blob/main/RESULTS.md).

## 2026-07-27 — VALIDITY CAVEAT for all S3 test-accuracy claims above

Post-release review of the grok_lens Phase 5 work found that this
experiment's test sets are i.i.d. samples from a space the training
stream effectively covers (only 335,922 possible chains for k ≤ 6 vs
millions of training draws), so "test accuracy" in the entries above
partly measures seen chains. A corrected pipeline (full enumeration,
disjoint per-k-stratified held-out split, per-k-uniform sampling — the
uniform-k curriculum turns out to be *required* for learnability) lives
in the public release repo
([lens-loss-grokking-experiment](https://github.com/brendanlong/lens-loss-grokking-experiment),
`lego/`), with re-run results in its RESULTS.md "Review re-runs II":
baselines genuinely generalize (100% held-out at k ≥ 2); the lens-aux
loss costs held-out accuracy on the smallest strata; lens front-loading
survives but is smaller than the leaky evaluation suggested.

## 2026-08-17 — S4 checkpoint recovery + generalization probe (queued, local GPU)

Six runs queued on the idle local RTX 3060 Ti (SkyPilot jobs 212, 213,
228–231). **Both S4 arms reached 100% on every k=0-6 and their checkpoints are now in S3**
(the point of the exercise). Outcomes, wandb IDs and canonical URIs below.

### Why: the Phase 20 S4 checkpoints do not exist

Phase 20 trained both S4 arms to 100% on k=0–6 (WS 256d ~11.4 h, standard
256d ~12.8 h of local GPU). Neither model survives as weights: the WS run
omitted `--save-checkpoint` and its local files were overwritten by the
standard run, whose own files lived in `data/lego/checkpoints/` and went with
the worktree. `s3://brendanlong-experiments/lego/checkpoints/` contains no
`S4`/`S5`/`A5` prefix — only wandb scalars remain. So the top open question
(**logit lens on S4**) was blocked purely on ~24 GPU-hours, which the idle
local GPU supplies for free.

The task YAML synced checkpoints only from an EXIT trap, which is how a reboot or
SIGKILL loses a run; it now also syncs every 5 min in the background to a
separate `in-progress/` subprefix. Separate for two reasons: train.py's final
`upload_checkpoint_to_s3` raises `FileExistsError` on an existing key (so a
mid-run sync into the canonical prefix would fail the job at the very end, after
all the GPU time), and `assert_run_name_free` lists the canonical prefix at
startup and would read a mid-run key as a reused run name.

These runs used `skypilot/train-lego.yaml`; that per-experiment file was
subsequently replaced upstream by the shared `skypilot/train-generic.yaml`, and
the periodic sync now lives there (via `start_periodic_sync` in
`skypilot/lib.sh`), so every experiment gets it rather than just lego. Retry
consequence to remember: a run that dies after its first checkpoint leaves
`in-progress/` objects, so re-running under the same `RUN_NAME` needs
`ALLOW_OVERWRITE=1`.

**Convergence is ~4x earlier than Phase 20 implied.** The WS arm reached 100%
on every k=0-6 at **step 224,000 = 114.7M examples** (first all-100% eval at
218,000; the `*** CONVERGED ***` line, which needs all k >= 99.9%, at 224,000).
Phase 20 reported being manually stopped at 677,500 steps (346.9M) and concluded
S4 "requires ~500M streaming examples" — that overstates the data requirement by
roughly 3-4x, and by extension the "~50x more data than S3" claim. On the full
500M schedule, **12.7 of the 16.5 hours train a model already at loss 0.0000**.
The standard arm was therefore re-queued (job 234, replacing the cancelled 213)
at `--generate-n 250000000` = 488,281 steps ~ 8.5 h, still 2.2x the WS
convergence step; Phase 20's own std/WS stop ratio was only 1.12x, so the
headroom is ample. Note `early_stop_patience=None` is hardcoded in `train.py`, so
these runs cannot stop themselves at convergence — the budget is the only knob.

```bash
# job 212 — WS arm (~16.4 h at the observed 16.5 steps/s)
RUN_NAME=S4-ws-256d-8h-8iter-k0-6-s42 ./train.sh local lego -- \
  --group S4 --weight-shared --dim 256 --n-heads 8 --n-layers 8 \
  --generate-n 500000000 --batch-size 512 \
  --wandb-run-name S4-ws-256d-8h-8iter-k0-6-s42 \
  --save-checkpoint --save-every-steps 50000

# job 234 — standard arm, right-sized budget (~8.5 h at ~16 steps/s).
# Job 213 was this same run at --generate-n 500000000 (~18 h); cancelled
# before it started once the WS arm showed convergence at 114.7M examples.
RUN_NAME=S4-std-256d-8h-8L-k0-6-s42 ./train.sh local lego -- \
  --group S4 --dim 256 --n-heads 8 --n-layers 8 \
  --generate-n 250000000 --batch-size 512 \
  --wandb-run-name S4-std-256d-8h-8L-k0-6-s42 \
  --save-checkpoint --save-every-steps 50000
```

Config: lego defaults — answer-only loss, lr 3e-4 cosine, seed 42,
torch.compile on, `--n-epochs` forced to 1 by streaming mode. The WS arm runs
the full 500M/512 = 976,562 steps (Phase 20 stopped both runs manually shortly
after convergence, hence its shorter recorded wall-clocks); the standard arm
runs 250M/512 = 488,281 steps. **One caveat on the cosine schedule:** shrinking
`--generate-n` also compresses the LR decay, so job 234 is not a pure truncation
of job 213 — it anneals to ~0 by 488k steps instead of 976k. Both still converge
well before their own annealing tail, but the two arms' schedules differ, which
matters if anyone compares their training curves rather than their endpoints.

**Periodic checkpoint sync verified in production:** 17 objects landed under
`.../S4-ws-256d-8h-8iter-k0-6-s42/in-progress/` at ~55 min intervals
(step_50000 through step_800000, 3.2 MiB each), so the crash-safety change works
rather than silently no-opping behind its `>/dev/null`.

### S4 outcome (both arms, checkpoints recovered)

| arm | wandb | steps | converged | wall-clock | final k=0-6 | checkpoint |
|---|---|---|---|---|---|---|
| WS 256d/8h/8iter | `eb50295p` | 976,562 | **224,000** (114.7M ex) | 16.5 h | 100% all | `s3://brendanlong-experiments/lego/checkpoints/S4-ws-256d-8h-8iter-k0-6-s42/step_976562.pt` |
| std 256d/8h/8L | `jrd85ege` | 488,281 | **281,000** (143.9M ex) | 8.5 h | 100% all | `s3://brendanlong-experiments/lego/checkpoints/S4-std-256d-8h-8L-k0-6-s42/step_488281.pt` |

Both carry a `.pt.json` metadata sidecar (`converged_step`, `total_steps`,
`mean_accuracy`) and both runs completed their cosine schedule, so the two are
comparable at their endpoints. Phase 20's result is reproduced — and this time
the weights survive, which unblocks the **logit lens on S4** open question.

The std/WS convergence ratio is **1.25x** (281,000 vs 224,000), close to Phase
20's 1.12x stop ratio, so the 250M budget chosen for the std arm had ample
headroom; ~175M would have sufficed. Intermediate checkpoints every 50,000 steps
are retained under each prefix (plus `in-progress/` copies from the periodic
sync), which makes a training-trajectory analysis possible without re-running.

### Generalization probe: composition vs per-k memorization

The open question "train on k<=5, test on k=6 to definitively prove
compositional vs memorization strategies" was not expressible in the harness —
test sets were built over the training range, so a held-out length was never
evaluated. Added `--holdout-k` and `--eval-k-max`.

**The literal phrasing is confounded.** `seq_len(k) = 2k+4` puts the k=6 answer
at position 15, while training at `k_max=5` never reaches past position 13. With
the default `--pos-encoding learned` that position embedding is still at init, so
failure at k=6 would measure an untrained parameter, not composition. Hence two
designs, run for both architectures:

```bash
# jobs 228, 229 — hop-count generalization: k=4 absent from training,
# padding still k_max=6, so every position carries real tokens in training
RUN_NAME=S3-std-8L-holdout-k4-s42 ./train.sh local lego -- \
  --wandb-run-name S3-std-8L-holdout-k4-s42 --group S3 --dim 128 --n-heads 4 \
  --n-layers 8 --generate-n 20000000 --batch-size 512 --save-checkpoint \
  --k-max 6 --holdout-k 4
RUN_NAME=S3-ws-8iter-holdout-k4-s42 ./train.sh local lego -- \
  --wandb-run-name S3-ws-8iter-holdout-k4-s42 --group S3 --dim 128 --n-heads 4 \
  --n-layers 8 --generate-n 20000000 --batch-size 512 --save-checkpoint \
  --k-max 6 --holdout-k 4 --weight-shared

# jobs 230, 231 — length extrapolation: train k<=5, test k=6, RoPE so
# position is relative rather than a per-position learned parameter
RUN_NAME=S3-std-8L-rope-k5to6-s42 ./train.sh local lego -- \
  --wandb-run-name S3-std-8L-rope-k5to6-s42 --group S3 --dim 128 --n-heads 4 \
  --n-layers 8 --generate-n 20000000 --batch-size 512 --save-checkpoint \
  --k-max 5 --eval-k-max 6 --pos-encoding rope
RUN_NAME=S3-ws-8iter-rope-k5to6-s42 ./train.sh local lego -- \
  --wandb-run-name S3-ws-8iter-rope-k5to6-s42 --group S3 --dim 128 --n-heads 4 \
  --n-layers 8 --generate-n 20000000 --batch-size 512 --save-checkpoint \
  --k-max 5 --eval-k-max 6 --pos-encoding rope --weight-shared
```

**Prediction (recorded before results).** If weight sharing forces a genuinely
compositional algorithm (Current Best Understanding #1) while standard models
can memorize multi-step shortcuts, the WS arms should retain high accuracy on
the untrained chain length and the standard arms should drop. A null result —
both generalize — would weaken the memorization reading of the standard model;
both failing would suggest per-k specialization is universal here.

### Results (jobs 228-233) — prediction refuted, and position encoding dominates

Chance on S3 is 16.7%. Starred columns are lengths the model never trained on.

| job (wandb) | arch | pos enc | design | trained k=5 / k=6 | probe |
|---|---|---|---|---|---|
| 228 (`1sazuwii`) | std | learned | holdout k=4 | 31.4% / 35.2% **✗** | k=4\* 35.9% |
| 229 (`29vpbrhd`) | WS | learned | holdout k=4 | 16.2% / 17.9% **✗** | k=4\* 17.5% |
| 232 (`7d3ciofe`) | std | **rope** | holdout k=4 | 100% / 100% ✓ | **k=4\* 94.4%** |
| 233 (`yl02d6ce`) | WS | rope | holdout k=4 | 32.4% / 32.3% **✗** | k=4\* 29.4% |
| 230 (`lszbsfhu`) | std | rope | k<=5 -> 6 | 100% / — ✓ | **k=6\* 70.2%** |
| 231 (`pedq7kq1`) | WS | rope | k<=5 -> 6 | 96.6% / — (~✓) | k=6\* 20.4% |

All six learn k=0-3 at 100%. Job 232 converged at step 12,000, job 230 at 11,000.

**1. Read only one holdout arm as a generalization result.** Three of the four
holdout arms (228, 229, 233) failed their own *trained* k=5 and k=6 — they never
fit the training distribution, so their held-out numbers measure a failed
optimization, not a failure to generalize. Only job 232 earns an interpretation.

**2. The clean positive: 94.4% on a hop count never trained.** Job 232 (std,
RoPE) holds k=4 out of training entirely and still solves it at 94.4%, with every
trained length at 100%. For this architecture the algorithm is not a per-k
lookup — an unseen composition depth is computed, not memorized. That is the
question the open item asked, answered affirmatively for the standard model.

**3. Removing k=4 usually breaks k=5 and k=6.** Every arm learns k<=3 perfectly
and then collapses at k>=4 — including the *trained* k=5/k=6 in three of four
cells. So the short-chain curriculum has to be **contiguous**: a hole at k=4
blocks the lengths above it. This extends the known result that a
short-chain curriculum is required for learnability at all (grok_lens
RESULTS.md "Review re-runs II": k-proportional sampling, 83% k=6, never lifted
off chance). It also means `--holdout-k` is a much more invasive intervention
than it looks.

**4. Position encoding dominates architecture here.** Same standard config, same
holdout: **35.9% with learned embeddings vs 94.4% with RoPE**, and trained
k=5/k=6 go from 31%/35% to 100%/100%. The review's predicted positional confound
was real and large — arm C was worth adding, and arm A alone would have produced
a badly wrong conclusion. Note the effect is *bigger* than the predicted
mechanism: learned position embeddings didn't just break the held-out readout,
they broke **training** at k>=4 under a gapped curriculum.

**5. The pre-registered prediction was wrong, in direction.** I predicted the WS
arms would retain accuracy on the untrained length and the standard arms would
drop, reasoning from "weight sharing forces genuinely compositional algorithms"
(Current Best Understanding #1). The opposite happened in every cell: standard
generalized (94.4% holdout, 70.2% extrapolation) and WS did not (29.4%, 20.4%).

The honest caveat is that this is **not** clean evidence against weight sharing:
WS failed to fit its trained distribution in three of four cells, and WS is
already documented as the more optimization-fragile architecture (it fails
entirely at 24 iterations; its S5 behavior is non-monotonic in width). The
defensible statement is "WS did not train under a gapped curriculum," not "WS
cannot generalize." Distinguishing those needs a WS arm that first converges —
e.g. holding out a length under the full k=0-6 curriculum at a width where WS is
known-good, or simply more seeds.

Further caveats: n=1 per cell; the WS arms are 128d/4h/8iter (head_dim 32, inside
the documented viable band) rather than the headline "Small WS" 96d/6h, so they
are not the most-interpretable configuration; and per caveat 3 below, trained-k
accuracy here is inflated by chain revisits, which makes the 232 result the
conservative one (its held-out stratum is clean by construction).

**Follow-ups this suggests**, in order of information value: (a) a WS arm that
converges, to make claim 5 decidable; (b) hold out k=5 or k=6 instead of k=4 —
if the collapse is really "the hole blocks everything above it," holding out the
*top* length should leave the rest intact and give a cleaner probe; (c) seeds.

### Caveats (a review pass found the first two after the jobs were queued)

1. **Arm A does not fully escape the positional confound.** Holding out k=4
   keeps every *position* trained (k=5 and k=6 chains cover positions 0–15), but
   not every *readout*: `<predict>` appears at `2k+2` and the answer at `2k+3`,
   so with k=4 absent the model is never supervised to emit anything at position
   10, and never sees `<predict>` there at all. With `--pos-encoding learned` a
   k=4 failure is therefore still partly attributable to an unsupervised readout
   position rather than to an inability to compose 4 hops. Compounding this, the
   4-hop *composition* is heavily trained regardless — every k=5/k=6 chain
   contains a k=4 prefix. So arm A measures "can it read out at an unsupervised
   hop count" more than "can it compose an unseen depth". Jobs 232/233 (below)
   are the disambiguation.
2. **RoPE substitutes its own extrapolation.** Training at `k_max=5` (length 14)
   exposes relative offsets ≤ 13; k=6 (length 16) needs 14–15. Only two novel
   offsets, but arm B is a relative-distance extrapolation test, not a
   confound-free one.
3. **Trained-k "test accuracy" is close to train accuracy.** S3 with k=0–6 has
   only 335,922 distinct chains total, so 20M streamed examples revisit each
   trained chain ~60× (the same leakage the public release repo fixed with a
   disjoint enumerated split — see grok_lens/RESULTS.md "Review re-runs II").
   The held-out stratum *is* clean by construction, so the trained-vs-held-out
   gap is inflated on the trained side.
4. Single seed per cell (n=1) — treat any WS-vs-standard gap as suggestive.
5. The RoPE arms differ from the headline learned-PE models, so arm B compares
   WS vs standard *within* RoPE, not against Phase 2 numbers.
6. `test_acc/mean` spans trained and untrained strata. `evaluate` now also emits
   `test_acc/trained_mean` and `test_acc/untrained_mean` whenever untrained k
   are present; read those, or the per-k values.

Also noted by that review and now blocked in `train.py`: `--holdout-k` with
`--staircase-loss` would have trained the held-out answers directly (the
staircase supervises `trajectory[j]` at every `j < k_max`, and for `j` in
`holdout_k` that target *is* the held-out answer on a held-out-length prefix).
None of the queued runs used it; the combination is now a `parser.error`.

### Arm C — the same holdout under RoPE (jobs 232, 233)

Added to separate caveat 1's two explanations: under RoPE there is no
per-position learned parameter for the unsupervised readout position, so if the
learned-PE arms fail at k=4 but the RoPE arms don't, the drop was positional
rather than compositional.

```bash
RUN_NAME=S3-std-8L-rope-holdout-k4-s42 ./train.sh local lego -- \
  --wandb-run-name S3-std-8L-rope-holdout-k4-s42 --group S3 --dim 128 \
  --n-heads 4 --n-layers 8 --generate-n 20000000 --batch-size 512 \
  --save-checkpoint --k-max 6 --holdout-k 4 --pos-encoding rope
RUN_NAME=S3-ws-8iter-rope-holdout-k4-s42 ./train.sh local lego -- \
  --wandb-run-name S3-ws-8iter-rope-holdout-k4-s42 --group S3 --dim 128 \
  --n-heads 4 --n-layers 8 --generate-n 20000000 --batch-size 512 \
  --save-checkpoint --k-max 6 --holdout-k 4 --pos-encoding rope --weight-shared
```
