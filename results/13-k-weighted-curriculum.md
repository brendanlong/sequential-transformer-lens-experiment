# Phase 16: k-Weighted Curriculum Training

## Goal

Standard transformers trained with uniform k-distribution learn multi-hop shortcuts (compressing all computation into the final 1-2 layers), making logit lens analysis unreliable. We test whether weighting the training distribution toward longer chains (`weight(k) = k^2`) can push standard models toward one-hop-per-layer sequential composition, enabling a fair WS vs Std comparison.

## Background

In Phase 15, we found that standard models always prefer late-layer bulk computation regardless of model size (tested 24d-96d) or depth (6-8L). The logit lens shows no staircase pattern in standard models -- intermediates are invisible or compressed into the final layers. This makes it impossible to fairly compare logit lens readability between WS and Std architectures, because the difference reflects algorithmic choice (sequential vs multi-hop) rather than probe sensitivity.

## Setup

Two changes from Phase 14:
1. **k-power weighting**: `--k-power 2` gives `weight(k) = k^2`, so k=6 gets 39.6% of data (36x more than k=1's 1.1%). k=1 still gets 330K examples out of 30M, enough to learn the single-step table.
2. Same curriculum: 30M AO examples, then 50M FSL examples.

```bash
# Standard model with k-power=2
./train.sh remote --module experiments.lego.curriculum_ao_to_fsl lego -- \
    --dim 96 --n-heads 6 --n-layers 8 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 --save-checkpoint --no-compile \
    --k-power 2 --seed 42

# Weight-shared model with k-power=2
./train.sh remote --module experiments.lego.curriculum_ao_to_fsl lego -- \
    --dim 96 --n-heads 6 --n-layers 8 --weight-shared \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 --save-checkpoint --no-compile \
    --k-power 2 --seed 42

# FSL-only control (no AO curriculum, uses train.py directly)
./train.sh remote lego -- \
    --dim 96 --n-heads 6 --n-layers 8 \
    --loss-mode full-sequence --generate-n 80000000 \
    --batch-size 512 --lr 3e-4 --save-checkpoint --no-compile \
    --k-power 2 --seed 42
```

### Runs

| Model | Seed | AO wandb | FSL wandb | GPU |
|-------|------|----------|-----------|-----|
| Std 96d/6h/8L kp=2 | 42 | `7gwt74kt` | `clzbke38` | RTX 3060 Ti (local) |
| Std 96d/6h/8L kp=2 | 43 | `8y1w1rld` | (same run) | RTX 4090 (RunPod) |
| WS 96d/6h/8iter kp=2 | 42 | `kc28xqvs` | `x000s412` | RTX 4090 (RunPod) |
| WS 96d/6h/8iter kp=2 | 43 | `xy73cefs` | (failed) | RTX 4090 (RunPod) |
| Std 96d/6h/8L FSL-only kp=2 | 42 | -- | `b5mcg9s9` | RTX 4090 (RunPod) |
| Std 96d/6h/8L FSL-only kp=2 | 43 | -- | `gqpwl3wa` | RTX 4090 (RunPod) |
| Std 96d/6h/8L FSL-only kp=2 | 44 | -- | `8oneyaas` | A40 (RunPod) |

- **Steps**: 58,593 AO + 97,656 FSL per model (FSL-only: 156,250 FSL)
- **Checkpoints**: In S3 (canonical store); backed up under `s3://brendanlong-experiments/wandb_migrated/lego-reasoning/`

## Standard Model (Std 96d/6h/8L, k_power=2)

### AO Phase (seed=42)

**Clean staircase -- the first in a standard model.** t[1] appears at L1 (79%), each subsequent hop follows at the next layer:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | **79%** | 47%  | 7%   | 2%   | 1%   | 0%   |
| L2    | 93%  | **92%** | 41%  | 12%  | 7%   | 2%   |
| L3    | 73%  | 82%  | **74%** | 21%  | 9%   | 4%   |
| L4    | 89%  | 76%  | 95%  | **79%** | 55%  | 39%  |
| L5    | 100% | 76%  | 96%  | 99%  | **99%** | 73%  |
| L6    | 93%  | 89%  | 93%  | 99%  | 100% | **100%** |
| L7    | 97%  | 92%  | 94%  | 99%  | 100% | 100% |

Compare to Phase 14's uniform-distribution Std 96d/6h/8L, where t[1] didn't appear until L4 (39%) and t[1]-t[3] appeared simultaneously at L5. The k-power=2 weighting fundamentally changes the algorithm the standard model learns.

### AO Phase (seed=43)

Also shows a staircase, slightly less clean but still sequential:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | **32%** | 3%   | 1%   | 1%   | 0%   | 1%   |
| L2    | 48%  | **34%** | 6%   | 2%   | 5%   | 10%  |
| L3    | 81%  | 89%  | 26%  | 29%  | 26%  | 25%  |
| L4    | 96%  | 100% | **84%** | **95%** | 84%  | 54%  |
| L5    | 96%  | 100% | 89%  | 99%  | **99%** | 85%  |
| L6    | 89%  | 99%  | 91%  | 100% | 99%  | **100%** |
| L7    | 89%  | 99%  | 90%  | 100% | 99%  | 100% |

### FSL Phase (seed=42)

100% accuracy maintained. Clear transient wavefronts -- each intermediate appears then decays:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | **55%** | 26%  | 0%   | 0%   | 0%   | 0%   |
| L2    | 28%  | **56%** | 0%   | 3%   | 0%   | 1%   |
| L3    | 30%  | 55%  | **72%** | 23%  | 6%   | 2%   |
| L4    | 12%  | 15%  | 31%  | **64%** | 39%  | 22%  |
| L5    | 15%  | 21%  | 21%  | 21%  | **48%** | 73%  |
| L6    | 16%  | 17%  | 17%  | 18%  | 15%  | **100%** |
| L7    | 19%  | 15%  | 12%  | 23%  | 12%  | 100% |

## Weight-Shared Model (WS 96d/6h/8iter, k_power=2)

### AO Phase (seed=42)

Staircase with persistent states, similar to Phase 14 uniform-distribution WS:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | **64%** | 0%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 81%  | **22%** | 0%   | 0%   | 0%   | 0%   |
| L3    | 83%  | 44%  | **16%** | 2%   | 1%   | 0%   |
| L4    | 91%  | 64%  | 46%  | **21%** | 11%  | 5%   |
| L5    | 99%  | 87%  | 68%  | 44%  | **27%** | 30%  |
| L6    | 100% | 84%  | 80%  | 72%  | 45%  | **66%** |
| L7    | 100% | 80%  | 83%  | 80%  | 66%  | **100%** |

### FSL Phase (seed=42)

100% accuracy. Transient wavefront at early layers, only t[6] visible at output:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | **60%** | 10%  | 0%   | 0%   | 0%   | 0%   |
| L2    | 16%  | **28%** | 0%   | 0%   | 0%   | 0%   |
| L3    | 17%  | 15%  | 7%   | 1%   | 0%   | 2%   |
| L4    | 16%  | 15%  | 17%  | 17%  | 6%   | 21%  |
| L5    | 16%  | 15%  | 17%  | 17%  | 18%  | 35%  |
| L6    | 19%  | 20%  | 17%  | 20%  | 17%  | **82%** |
| L7    | 10%  | 21%  | 18%  | 12%  | 20%  | **100%** |

### Seed sensitivity (WS)

WS seed=43 failed completely (35% accuracy in both AO and FSL). With k_power=2, the WS model sees fewer k=1 examples (1.1% vs 16.7%), which may make it harder for the shared MLP to learn the single-step table reliably. The uniform-distribution WS runs (Phase 14) never failed.

## Sequential Ablation Test

| Model | Progressive -3 | Progressive -2 | Reverse -3 | Random | Causal reading? |
|-------|---------------|----------------|------------|--------|-----------------|
| WS kp=2 s42 FSL | **96.7%** | 87.3% | 16.7% | 14-17% | Yes |
| Std kp=2 s42 FSL | **100.0%** | **100.0%** | 15.0% | 3-5% | Yes |
| Std kp=2 s43 FSL | **100.0%** | **100.0%** | 16.3% | 15-20% | Yes |

All models show strong causal reading (progressive >> reverse >> random). The standard model retains 100% accuracy even at offset=-2, showing very robust sequential reading order.

## FSL-Only Control (no AO curriculum)

All 3 seeds reached 100% accuracy but showed **no staircase** -- only t[6] visible:

| Seed | k=6 acc | Best intermediate visibility |
|------|---------|---------------------------|
| 42 | 100% | t[6] builds L3→L7, all others at chance |
| 43 | 100% | t[6] builds L3→L7, all others at chance |
| 44 | 100% | t[6] builds L3→L7, all others at chance |

**The AO curriculum is essential for the staircase.** k-power=2 weighting alone (without AO→FSL curriculum) does not produce visible intermediate computation.

## Side-by-Side WS vs Std Comparison

With k_power=2, both architectures learn the same sequential algorithm. Here's the direct comparison (seed=42):

### AO Phase -- Both show staircases

| Layer | WS t[1] | WS t[3] | WS t[6] | | Std t[1] | Std t[3] | Std t[6] |
|-------|---------|---------|---------|---|----------|----------|----------|
| L0 | 0% | 0% | 0% | | 0% | 0% | 0% |
| L1 | **64%** | 0% | 0% | | **79%** | 7% | 0% |
| L2 | 81% | 0% | 0% | | 93% | 41% | 2% |
| L3 | 83% | **16%** | 0% | | 73% | **74%** | 4% |
| L5 | 99% | 68% | 30% | | 100% | 96% | 73% |
| L7 | 100% | 83% | **100%** | | 97% | 94% | **100%** |

The standard model's logit lens values are **higher** than the WS model's (e.g., t[3]=74% at L3 vs 16%). This suggests that unique per-layer weights produce representations that are *more* aligned with the unembedding, not less. The standard model's logit lens advantage with uniform data was an artifact of it using a different algorithm, not better probe compatibility.

### FSL Phase -- Both show wavefronts, Std is sharper

| Layer | WS t[1] | WS t[3] | WS t[6] | | Std t[1] | Std t[3] | Std t[6] |
|-------|---------|---------|---------|---|----------|----------|----------|
| L1 | **60%** | 0% | 0% | | **55%** | 0% | 0% |
| L2 | 16% | 0% | 0% | | 28% | 0% | 1% |
| L3 | 17% | 7% | 2% | | 30% | **72%** | 2% |
| L4 | 16% | 17% | 21% | | 12% | 31% | 22% |
| L5 | 16% | 17% | 35% | | 15% | 21% | **73%** |
| L7 | 10% | 18% | **100%** | | 19% | 12% | **100%** |

The WS model erases intermediates faster (t[1] drops from 60% to 16% by L2). The standard model retains intermediates longer (t[3] stays at 72% at L3, 31% at L4). This is the traveling erasure mechanism (WS) vs deferred erasure (Std) -- but now we can confirm it's a genuine architectural difference, not an artifact of different algorithms.

## Key Findings

1. **k-power=2 weighting produces the first clean staircase in a standard transformer.** With 39.6% of training data at k=6 (vs 16.7% uniform), the standard model learns one-hop-per-layer sequential composition with t[1] visible at L1, matching the WS model's algorithm.

2. **The AO curriculum is still essential.** FSL-only training with k_power=2 produces 100% accuracy but no visible intermediates (3/3 seeds). The two-phase approach (AO then FSL) is necessary for the staircase.

3. **WS models are more fragile with k_power=2.** Seed 43 failed completely (35%). The reduced k=1 data (1.1% vs 16.7%) may starve the shared MLP's ability to learn the single-step table.

4. **Standard model logit lens values are higher than WS** when both use the same algorithm. This overturns the Phase 14 finding that "weight sharing makes the logit lens work better" -- it was confounded by algorithmic differences.

5. **The WS traveling erasure is a genuine architectural effect.** In the WS FSL model, t[1] peaks at L1 (60%) and drops to 16% by L2. In the Std FSL model, t[1] peaks at L1 (55%) and remains at 28-30% through L3. This difference persists when both models use the same sequential algorithm, confirming it reflects the shared-weight iteration structure rather than algorithmic choice.

## Implications

- **For fair WS vs Std comparisons**: use k_power=2 to ensure both architectures learn the same one-hop-per-layer algorithm. Without this, the standard model uses multi-hop shortcuts, making logit lens comparisons apples-to-oranges.
- **For robustness**: WS models may need more k=1 data to reliably learn the single-step table. Consider k_power=1 (linear weighting: k=6 gets 6x k=1) as a compromise.
- **The curriculum remains critical**: neither architecture shows visible intermediate computation under FSL-from-scratch, regardless of data weighting.
