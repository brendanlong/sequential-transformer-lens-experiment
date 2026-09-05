# Phase 20: Diagonal Evar Analysis

## Goal

Phase 17 (enhanced metrics) reported that embedding explained variance (evar) was ~25% for all models, but that analysis averaged across ALL layers and ALL op positions. For answer-only (AO) models, this is misleading: after the critical layer where each intermediate state is computed, the model has no loss pressure to maintain that state — the residual can drift off the embedding manifold freely. Averaging across all entries dilutes the signal from the positions/layers where computation is actually happening.

This analysis measures evar specifically on the **diagonal** — the layer where each op position's logit lens accuracy peaks — to see whether evar is meaningfully higher where it matters. We also compare AO-only vs AO+FSL (curriculum) checkpoints.

## Setup

Analysis script: `analyze_diagonal_evar.py`

Models analyzed (all 96d/6h, uniform curriculum):

Checkpoint bytes live in the S3 backup at
`s3://brendanlong-experiments/wandb_migrated/lego-reasoning/<name>/<version>/`
and resolve automatically via `analyze_diagonal_evar.py`.

| Model | Wandb (AO phase) | Wandb (FSL phase) | Checkpoint (artifact name) |
|-------|------|------|----------|
| WS 8iter AO | `84c0iqcs` | — | `lego-ws_96d_6h_8L_curriculum_AO:v0` |
| Std 8L AO | `lyy1ex3n` | — | `lego-std_96d_6h_8L_curriculum_AO:v0` |
| WS 8iter AO→FSL | `84c0iqcs` | `ac6tyb31` | `lego-ws_96d_6h_8L_curriculum_FSL:v0` |
| Std 8L AO→FSL | `lyy1ex3n` | `9sl0fov0` | `lego-std_96d_6h_8L_curriculum_FSL:v0` |

```bash
uv run python -m experiments.lego.analyze_diagonal_evar
```

Parameters: k=6, n_examples=500, seed=999.

## Method

For each model:
1. Compute the full evar heatmap (8 layers × 6 op positions)
2. Compute the top-1 accuracy heatmap (same shape)
3. Find the **critical layer** for each position = layer where top-1 accuracy peaks
4. Extract evar at the critical layer ("diagonal evar")
5. Compute "before-critical" evar (layers before the computation happens) vs "at-and-after-critical" evar

## Results

### 1. WS AO model: diagonal evar is notably higher than overall

| Metric | WS AO |
|--------|-------|
| Overall evar | 25.8% |
| **Diagonal evar** | **30.1%** |
| Off-diagonal evar | 25.2% |
| Before-critical evar | 18.2% |
| At-and-after-critical evar | 34.1% |

The WS AO model's staircase is clean (100% top-1 at all critical layers). The diagonal evar (30.1%) is meaningfully higher than the overall (25.8%), and the before/after split is even more dramatic: **18.2% before** vs **34.1% at-and-after** the critical layer. This confirms the hypothesis — before computation, the residual is mostly off-manifold; once the model has computed the trajectory state, ~34% of the residual lives in the embedding subspace.

Per-position breakdown on the diagonal:

| State | Critical Layer | Top-1 | Evar | Entropy |
|-------|---------------|-------|------|---------|
| t[1] | L2 | 100% | 36.6% | 1.0% |
| t[2] | L2 | 100% | 29.2% | 0.5% |
| t[3] | L3 | 100% | 27.3% | 0.0% |
| t[4] | L5 | 100% | 29.8% | 0.0% |
| t[5] | L6 | 100% | 30.1% | 0.0% |
| t[6] | L7 | 100% | 27.6% | 0.0% |

Even at the best layers, evar is only 27-37%. The model succeeds at encoding the correct trajectory state (100% top-1, near-zero entropy), but most of the residual norm (63-73%) is still orthogonal to the embedding subspace. The logit lens only reads the "tip" of the residual.

### 2. Std AO model: diagonal evar is LOWER than overall

| Metric | Std AO |
|--------|--------|
| Overall evar | 26.8% |
| **Diagonal evar** | **22.1%** |
| Off-diagonal evar | 27.5% |
| Before-critical evar | 28.1% |
| At-and-after-critical evar | 21.4% |

The standard model shows the opposite pattern: evar *decreases* in the late layers where computation happens (L5-L7). The standard model doesn't produce a clean staircase — it compresses computation into the last 2-3 layers with moderate accuracy (48-100%). The early layers have higher evar because they're still close to the input embeddings, but by the time the model is actively computing, the residual has rotated away from the embedding space.

### 3. FSL training reduces evar across the board

| Model | Overall | Diagonal | Before | At+After |
|-------|---------|----------|--------|----------|
| WS AO | 25.8% | 30.1% | 18.2% | 34.1% |
| WS AO→FSL | 18.0% | 21.5% | 15.4% | 20.7% |
| Std AO | 26.8% | 22.1% | 28.1% | 21.4% |
| Std AO→FSL | 21.4% | 17.3% | 23.4% | 14.8% |

FSL training makes evar *worse* everywhere, not better. This is surprising at first glance — FSL adds loss at intermediate positions, so one might expect the model to keep residuals more on-manifold. But the opposite happens because FSL trains `<op>` positions to predict the *next token* (which is a random group element), not the trajectory state. The model actively pushes trajectory information orthogonal to the unembedding to avoid interference with the next-token prediction target.

The WS AO→FSL model's staircase is also partially degraded:

| State | Critical Layer | Top-1 (AO) | Top-1 (AO→FSL) |
|-------|---------------|-----------|----------------|
| t[1] | L1-L2 | 100% | 67% |
| t[2] | L2 | 100% | 82% |
| t[3] | L3 | 100% | 80% |
| t[4] | L4-L5 | 100% | 46% |
| t[5] | L6-L7 | 100% | 17% (chance) |
| t[6] | L7 | 100% | 100% |

The early intermediate states survive FSL training (67-82%), but later ones are erased (t[5] drops to chance). Only the final answer t[6] remains perfectly readable.

### 4. Full evar heatmaps

**WS 8iter AO** (clean staircase, highest diagonal evar):
```
       t[1]  t[2]  t[3]  t[4]  t[5]  t[6]
  L0    23%   30%   21%   34%   19%   31%
  L1    29%   14%    9%   16%   10%   20%
  L2    37%*  29%*  16%   12%   10%   10%
  L3    41%   33%   27%*  13%   12%   10%
  L4    42%   36%   31%   24%   16%   10%
  L5    41%   35%   33%   30%*  25%   16%
  L6    41%   36%   33%   31%   30%*  24%
  L7    41%   36%   33%   30%   29%   28%*
```

Pattern: evar starts low at L0, ramps up as the trajectory state is computed, then plateaus. The *plateau* after computation (~30-41%) is where the model maintains the state for attention by later positions. The ramp-up corresponds to the staircase.

**Std 8L AO** (compressed into late layers):
```
       t[1]  t[2]  t[3]  t[4]  t[5]  t[6]
  L0    39%   47%   55%   47%   40%   51%
  L1    27%   34%   43%   37%   34%   45%
  L2    19%   22%   29%   27%   27%   35%
  L3    17%   21%   25%   24%   20%   28%
  L4    14%   16%   21%   22%   16%   29%
  L5    13%   17%   24%*  26%   15%   30%
  L6    13%*  16%   22%   24%   14%   27%
  L7    18%   18%*  20%   26%*  21%*  30%*
```

Pattern: evar is *highest in early layers* (39-55% at L0) and monotonically decreases. The standard model's computation happens in the late layers where evar is lowest. This is the opposite of WS.

## 5. k-power=2 matched models: the WS diagonal evar advantage disappears

The analysis above used uniform-curriculum models, where WS and Std learn different algorithms. To control for this confound, we repeated the diagonal evar analysis on k-power=2 models (Phase 16), where both architectures learn the same sequential one-hop-per-layer algorithm.

### AO phase comparison (matched algorithm)

| Model | Overall | Diagonal | Off-diag | Before | At+After |
|-------|---------|----------|----------|--------|----------|
| WS AO (uniform) | 25.8% | **30.1%** | 25.2% | 18.2% | **34.1%** |
| WS AO (kp=2) | 16.5% | 17.8% | 16.3% | 16.2% | 17.8% |
| Std AO (uniform) | 26.8% | 22.1% | 27.5% | 28.1% | 21.4% |
| Std AO (kp=2) | **28.3%** | **25.8%** | **28.6%** | **29.2%** | **25.1%** |

The WS kp=2 model has dramatically lower evar than the uniform WS model (17.8% vs 30.1% on the diagonal). The before/after split that was so dramatic for WS uniform (18.2% → 34.1%) is essentially flat for WS kp=2 (16.2% → 17.8%). The WS model's distinctive "ramp-up at computation" pattern is gone.

Meanwhile the standard kp=2 model — which now also learns sequential composition — has the highest overall evar of all AO models (28.3%) and a diagonal evar of 25.8%, higher than the WS kp=2 model's 17.8%.

Per-position diagonal for WS kp=2 AO (note: lower accuracy than uniform WS — the staircase is less clean):

| State | Critical Layer | Top-1 | Evar | Entropy |
|-------|---------------|-------|------|---------|
| t[1] | L6 | 100% | 18.3% | 1.4% |
| t[2] | L5 | 87% | 16.3% | 8.5% |
| t[3] | L7 | 83% | 16.9% | 5.0% |
| t[4] | L7 | 80% | 17.0% | 3.7% |
| t[5] | L7 | 66% | 16.0% | 5.0% |
| t[6] | L7 | 100% | 22.3% | 0.0% |

Per-position diagonal for Std kp=2 AO:

| State | Critical Layer | Top-1 | Evar | Entropy |
|-------|---------------|-------|------|---------|
| t[1] | L5 | 100% | 19.1% | 2.0% |
| t[2] | L7 | 92% | 19.5% | 3.0% |
| t[3] | L5 | 96% | 25.8% | 1.2% |
| t[4] | L7 | 99% | 27.4% | 0.5% |
| t[5] | L6 | 100% | 31.4% | 0.2% |
| t[6] | L7 | 100% | 31.9% | 0.0% |

The standard kp=2 model has both higher logit lens accuracy (96-100% vs 66-100%) AND higher diagonal evar (19-32% vs 16-22%) at every position. The standard model's later trajectory states (t[5], t[6]) reach 31-32% evar — the WS model peaks at 22%.

### FSL phase comparison (matched algorithm)

| Model | Overall | Diagonal | Off-diag | Before | At+After |
|-------|---------|----------|----------|--------|----------|
| WS AO→FSL (uniform) | 18.0% | 21.5% | 17.5% | 15.4% | 20.7% |
| WS AO→FSL (kp=2) | **11.8%** | **10.7%** | 12.0% | 12.3% | **11.1%** |
| Std AO→FSL (uniform) | 21.4% | 17.3% | 22.0% | 23.4% | 14.8% |
| Std AO→FSL (kp=2) | 20.9% | 20.5% | 21.0% | 30.1% | 13.8% |

WS kp=2 FSL is the worst across the board at 10.7% diagonal evar — intermediates are nearly at chance with high entropy (77.8% for t[3]-t[5]). The standard kp=2 FSL model retains 20.5% diagonal evar and shows a strong before/after split (30.1% → 13.8%), meaning its early layers (close to input) still carry significant embedding-aligned signal even after FSL training.

The Std kp=2 FSL model also shows something interesting: its early diagonal entries (t[1] at L1: 32.6%, t[2] at L2: 30.4%) have high evar because the sequential staircase makes these critical layers overlap with the high-evar early layers. Later entries (t[3]-t[5]) drop to 11% as critical layers move deeper and evar falls with depth.

### Full evar heatmaps (kp=2 models)

**WS 8iter AO (kp=2):**
```
       t[1]  t[2]  t[3]  t[4]  t[5]  t[6]
  L0    23%   23%   29%   28%   30%   30%
  L1    14%   11%   20%   18%   18%   17%
  L2    17%   11%   16%   13%   12%   16%
  L3    16%   13%   14%   11%   12%   10%
  L4    17%   15%   14%    9%    8%    9%
  L5    18%   16%*  17%   16%   11%   12%
  L6    18%*  17%   16%   17%   14%   13%
  L7    19%   17%   17%*  17%*  16%*  22%*
```

**Std 8L AO (kp=2):**
```
       t[1]  t[2]  t[3]  t[4]  t[5]  t[6]
  L0    32%   49%   51%   41%   45%   43%
  L1    31%   39%   36%   30%   30%   30%
  L2    23%   33%   33%   23%   21%   23%
  L3    20%   28%   29%   21%   17%   20%
  L4    19%   24%   28%   31%   23%   24%
  L5    19%*  20%   26%*  28%   27%   26%
  L6    19%   20%   27%   30%   31%*  33%
  L7    22%   19%*  26%   27%*  28%   32%*
```

The standard kp=2 model shows the familiar "high early, declining with depth" pattern, but critically, the decline is much more gradual than the uniform standard model — evar stabilizes around 19-33% in the late layers instead of dropping to 13-21%. This is because the sequential algorithm (forced by kp=2) distributes computation across layers, so the model doesn't rotate as aggressively away from the embedding basis.

## Key Findings

1. **The WS diagonal evar advantage was confounded by algorithm, not architecture.** With uniform data, WS diagonal evar (30.1%) was much higher than Std (22.1%). With kp=2 (matched algorithm), it inverts: Std diagonal evar (25.8%) exceeds WS (17.8%).

2. **kp=2 weighting sharply reduces WS evar overall.** WS kp=2 has only 16.5% overall evar vs 25.8% uniform. The training distribution change (fewer k=1 examples, more k=6) appears to make it harder for the WS shared MLP to maintain embedding-aligned representations. This may be related to the seed sensitivity issue (seed 43 failed entirely under kp=2).

3. **Standard models maintain higher evar under sequential composition.** When forced to learn the same one-hop-per-layer algorithm, the standard model achieves 28.3% overall evar and 25.8% diagonal evar — both higher than WS. Unique per-layer weights apparently allow the model to maintain more of the residual in the embedding subspace.

4. **The standard model's "high early, declining" pattern persists but is gentler with kp=2.** Even with sequential composition, the standard model's evar decreases with depth (49% at L0 → 19% at L7 for t[2]). But the late-layer floor is higher than uniform (19-33% vs 13-21%), suggesting the sequential algorithm limits how far representations rotate away from embeddings.

5. **FSL training degrades WS kp=2 evar catastrophically.** WS kp=2 FSL has 10.7% diagonal evar — the lowest of any model. Combined with near-chance logit lens accuracy on t[3]-t[5] (18-20%), this model has essentially lost intermediate visibility entirely.

6. **FSL training degrades standard kp=2 evar less.** Std kp=2 FSL retains 20.5% diagonal evar and maintains some intermediate staircase (t[1]=55%, t[3]=72%). The before/after split (30.1% → 13.8%) shows the early-layer embedding alignment is preserved.

7. **Averaging evar across all layers understates the WS uniform model's embedding alignment.** (Original finding, still true for uniform.) The WS uniform model's diagonal evar (30.1%) is ~5pp higher than its overall mean (25.8%). But this advantage is specific to the uniform training regime.

8. **Even at peak, evar is only ~25-32% for the best matched models.** The standard kp=2 model encodes trajectory states with ~25-32% of the residual norm in the embedding subspace. The logit lens succeeds (92-100% top-1) despite reading only a quarter of the residual. Most of the residual carries other information (attention patterns, positional signals, etc.) in orthogonal directions.
