# Phase 15: Minimum-Layer Curriculum AO->FSL Training (6L & 7L)

## Goal

Find the minimum number of layers needed for the curriculum AO->FSL experiment while preserving interpretable staircase structure. Phase 14 used 8 layers; here we test 6 and 7 layers to understand how spare capacity affects both accuracy and interpretability.

## Setup

Same two-phase training as Phase 14, with `--n-layers 6` and `--n-layers 7`. Both weight-shared (WS) and standard architectures tested at 96d/6h.

### 6-Layer Commands

```bash
# Weight-shared model
./train.sh remote --module experiments.lego.curriculum_ao_to_fsl lego -- \
    --dim 96 --n-heads 6 --n-layers 6 --weight-shared \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42

# Standard model
./train.sh remote --module experiments.lego.curriculum_ao_to_fsl lego -- \
    --dim 96 --n-heads 6 --n-layers 6 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42
```

### 7-Layer Commands

```bash
# Weight-shared model
./train.sh remote --module experiments.lego.curriculum_ao_to_fsl lego -- \
    --dim 96 --n-heads 6 --n-layers 7 --weight-shared \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42

# Standard model
./train.sh remote --module experiments.lego.curriculum_ao_to_fsl lego -- \
    --dim 96 --n-heads 6 --n-layers 7 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42
```

### Runs

| Model | Params | AO wandb | FSL wandb |
|-------|--------|----------|-----------|
| WS 96d/6h/6iter  | 125K unique (668K eff) | `nkyssb7v` | `0yxilhgz` |
| Std 96d/6h/6L    | 676K                   | `ysihzh1p` | `8fmz74bv` |
| WS 96d/6h/7iter  | 125K unique (779K eff) | `tib6mqpd` | `vohbhnmp` |
| Std 96d/6h/7L    | 788K                   | `0aekhcac` | `c8f02rl8` |

- **GPU**: RTX 4090 (RunPod)
- **Steps**: 58,593 AO + 97,656 FSL per model
- **Checkpoints**: In S3 (canonical store); backed up under `s3://brendanlong-experiments/wandb_migrated/lego-reasoning/`

---

## 6-Layer Results

### WS 96d/6h/6iter

**AO Phase** -- Converged at step 19,000. No staircase visible:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L3    | 0%   | 0%   | 0%   | 1%   | 0%   | 3%   |
| L4    | 0%   | 4%   | 0%   | 19%  | 0%   | 42%  |
| L5    | 36%  | 31%  | 0%   | 27%  | 6%   | 100% |

**FSL Phase** -- 100% accuracy maintained. All intermediates at chance:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 16%  | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 17%  | 15%  | 18%  | 15%  | 16%  | 0%   |
| L2    | 16%  | 17%  | 17%  | 18%  | 15%  | 16%  |
| L3    | 16%  | 17%  | 17%  | 18%  | 15%  | 20%  |
| L4    | 14%  | 18%  | 16%  | 17%  | 17%  | 37%  |
| L5    | 11%  | 15%  | 16%  | 14%  | 18%  | 100% |

### Std 96d/6h/6L

**AO Phase** -- Converged at step 22,000. Compressed staircase in L3-L5:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 0%   | 3%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 9%   | 5%   | 1%   | 0%   | 0%   | 0%   |
| L3    | 65%  | 60%  | 44%  | 30%  | 14%  | 3%   |
| L4    | 83%  | 81%  | 72%  | 60%  | 42%  | 19%  |
| L5    | 87%  | 99%  | 90%  | 68%  | 53%  | 100% |

**FSL Phase** -- 100% accuracy. Transient wavefronts visible:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 3%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 9%   | 8%   | 4%   | 0%   | 0%   | 0%   |
| L3    | 22%  | 45%  | 37%  | 29%  | 4%   | 0%   |
| L4    | 16%  | 21%  | 40%  | 41%  | 26%  | 35%  |
| L5    | 13%  | 29%  | 16%  | 14%  | 15%  | 100% |

---

## 7-Layer Results

### WS 96d/6h/7iter

**AO Phase** -- Converged at step 25,000. **Clear staircase returns:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 88%  | 66%  | 0%   | 0%   | 0%   | 0%   |
| L2    | 70%  | 94%  | 25%  | 0%   | 0%   | 0%   |
| L3    | 62%  | 75%  | 98%  | 12%  | 0%   | 1%   |
| L4    | 55%  | 63%  | 97%  | 60%  | 1%   | 8%   |
| L5    | 46%  | 55%  | 84%  | 97%  | 78%  | 44%  |
| L6    | 36%  | 49%  | 75%  | 90%  | 98%  | 100% |

t[1] peaks at L1 (88%), t[2] at L2 (94%), t[3] at L3 (98%), t[4] at L5 (97%), t[5] at L6 (98%). States are **persistent** -- once computed, they remain. Very similar to the 8L pattern.

**FSL Phase** -- 100% accuracy. **Transient wavefronts clearly visible:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 50%  | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 21%  | 55%  | 0%   | 0%   | 0%   | 0%   |
| L2    | 19%  | 24%  | 74%  | 0%   | 0%   | 0%   |
| L3    | 32%  | 27%  | 16%  | 32%  | 0%   | 0%   |
| L4    | 17%  | 18%  | 16%  | 17%  | 17%  | 1%   |
| L5    | 23%  | 20%  | 26%  | 24%  | 17%  | 66%  |
| L6    | 17%  | 29%  | 22%  | 12%  | 31%  | 100% |

t[1] peaks at L0 (50%), t[2] at L1 (55%), t[3] at L2 (74%), then each decays to chance. This is the traveling erasure mechanism -- just like 8L but with one less spare layer.

### Std 96d/6h/7L

**AO Phase** -- Converged at step 9,000. Computation concentrated in L5-L6:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L3    | 6%   | 12%  | 11%  | 2%   | 1%   | 0%   |
| L4    | 24%  | 32%  | 47%  | 18%  | 10%  | 10%  |
| L5    | 69%  | 93%  | 96%  | 65%  | 56%  | 42%  |
| L6    | 95%  | 95%  | 97%  | 94%  | 82%  | 100% |

**FSL Phase** -- 100% accuracy. Moderate transient structure:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 15%  | 5%   | 0%   | 2%   | 6%   | 0%   |
| L3    | 20%  | 17%  | 48%  | 6%   | 17%  | 0%   |
| L4    | 11%  | 15%  | 34%  | 17%  | 19%  | 5%   |
| L5    | 15%  | 19%  | 33%  | 20%  | 19%  | 54%  |
| L6    | 18%  | 17%  | 11%  | 13%  | 23%  | 100% |

---

## Cross-Layer Comparison (k=6 AO logit lens, WS models)

| Property | 6L WS | 7L WS | 8L WS (Phase 14) |
|----------|-------|-------|-------------------|
| AO convergence | Step 19,000 | Step 25,000 | Step 11,000 |
| Staircase visible? | No | **Yes** | **Yes** |
| Peak t[1] | 36% at L5 | 88% at L1 | 92% at L1 |
| Peak t[3] | 0% | 98% at L3 | 100% at L4 |
| Peak t[5] | 6% | 98% at L6 | 100% at L7 |
| FSL wavefronts? | No (chance) | **Yes** (50-74%) | **Yes** (46-82%) |
| FSL loss | ~0.986 | ~0.986 | ~0.985 |

| Property | 6L Std | 7L Std | 8L Std (Phase 14) |
|----------|--------|--------|-------------------|
| AO convergence | Step 22,000 | Step 9,000 | Step 10,000 |
| Staircase style | All states in L3-L5 | All states in L5-L6 | Compressed L4-L7 |
| FSL wavefronts? | Moderate (29-45%) | Moderate (33-48%) | Weak (33-36%) |

## Sequential Ablation Test

We tested whether the models read input clauses in causal order using the `ablate_sequential` tool. This zeros out clause token positions from the residual stream after a delay, checking whether the model has already extracted that information via attention.

```bash
uv run python -m experiments.lego.ablate_sequential \
    --checkpoint data/lego/checkpoints/6l_ws/step_97656.pt --k 6
# (repeated for 6l_std, 7l_ws, 7l_std)
```

Progressive ablation at offset=-3 (best alignment):

| Model | Progressive -3 | Reverse -3 | Random | Causal reading? |
|-------|---------------|------------|--------|-----------------|
| 6L WS FSL  | **90.0%** | 17.4% | 17-18% | Yes |
| 6L Std FSL | **100.0%** | 13.2% | 6-26% | Yes |
| 7L WS FSL  | **92.9%** | 16.7% | 17-19% | Yes |
| 7L Std FSL | **58.1%** | 16.3% | 0-17% | Yes (moderate) |
| 8L WS FSL  | **84.2%** | 17.3% | 17-20% | Yes |
| 8L Std FSL | **73.7%** | 10.7% | 14-21% | Yes |

All models read clauses in causal order (progressive >> reverse). However, **this test proves causal reading, not one-hop-per-layer composition.** It's compatible with a model that reads clauses early, then computes multi-step lookups in later layers.

### Counting argument: 6L models likely use multi-step computation

For k=6 in a 6L model, a true one-hop-per-layer algorithm would need t[j] computed at layer L_{j-1}, leaving no slack. But the 6L Std AO logit lens shows t[1] through t[3] all appearing simultaneously at L3 (65%, 60%, 44%). This means L0-L2 don't compute any hops, and L3-L5 must compute all 6 -- requiring at least some layers to do multi-step composed lookups.

The standard model's unique per-layer MLPs can memorize multi-step group composition tables (e.g., 2-hop or 3-hop). With 676K parameters across 6 unique layers, each MLP has enough capacity for this. So while the **reading** is causal, the **computation** likely involves memorized composed functions rather than pure single-step composition.

For the WS model, iteration embeddings could make the logit lens misleading about timing, but the same counting argument applies: with 6 layers and 6 hops, if intermediates aren't visible until L4-L5, multi-step computation per iteration is likely.

## OOD Generalization (k > 6)

Tested whether models trained on k=1-6 can generalize to longer chains:

| Model | k=6 | k=7 | k=8 | k=9 | k=10 |
|-------|-----|-----|-----|-----|------|
| 6L WS FSL  | 100% | 0% | 0% | 0% | 0% |
| 6L Std FSL | 100% | 1% | 20% | 17% | 12% |
| 7L WS FSL  | 100% | 7% | 1% | 17% | 0% |
| 7L Std FSL | 100% | 1% | 0% | 0% | 4% |

No model generalizes to k>6. Results at k>6 are at or below chance (1/6 ~ 16.7%).

## Key Findings

1. **6L models achieve 100% accuracy but likely use multi-step computation.** The counting argument (6 hops, 6 layers, no visible intermediates until L3+) implies some layers compute composed lookup tables rather than single-step operations. The ablation test confirms causal clause reading but cannot distinguish one-hop-per-layer from multi-hop-per-layer.

2. **7 layers restores the visible staircase (WS).** Just one extra layer (k+1 total) is enough for the WS model to develop the clean one-hop-per-layer pattern: t[1] peaks at L1 (88%), t[2] at L2 (94%), t[3] at L3 (98%). With 7 layers, the counting argument is satisfied -- each hop gets its own layer with one spare.

3. **The minimum for verifiable one-hop-per-layer WS composition is k+1 layers.** For k=6 hops, 7 layers suffice. The spare layer enables the traveling erasure mechanism.

4. **Standard models never show clean one-hop-per-layer staircases**, even at 7L or 8L. Their unique per-layer MLPs can specialize in multi-step operations, and the computation is always compressed into the final 2-3 layers.

5. **Convergence speed is non-monotonic for Std.** 6L Std takes 22K steps, 7L takes 9K, 8L takes 10K. The 6L model struggles at minimum capacity.

## Implications

- **For interpretability experiments: use k+1 layers minimum** (7L for k=6). This is the minimum where the WS model demonstrably uses one-hop-per-layer composition.
- **6 layers works for accuracy but the algorithm is opaque.** The model likely uses multi-step shortcuts, making mechanistic analysis unreliable.
- **The ablation test (`ablate_sequential`) proves causal reading, not composition depth.** To verify one-hop-per-layer, the logit lens counting argument (or supervised probes) is needed alongside the ablation.
- **Weight sharing remains critical for interpretability.** Standard models use multi-step MLPs regardless of depth; WS models are constrained to the single-step table and forced into clean sequential composition (given enough layers).
