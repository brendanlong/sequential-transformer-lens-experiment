# Phase 14: Curriculum AO->FSL Training

## Goal

Test whether the compositional (staircase) algorithm learned under answer-only (AO) loss is a stable attractor that survives switching to full-sequence loss (FSL). If composition is more efficient than memorization for a model that already has the algorithm, it should persist.

## Setup

Two-phase training: train AO to convergence on k=1-6, then switch to FSL for 50M examples. Both weight-shared (WS) and standard architectures tested at 96d/6h/8L.

```bash
# Weight-shared model
uv run python -m experiments.lego.curriculum_ao_to_fsl \
    --dim 96 --n-heads 6 --n-layers 8 --weight-shared \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42

# Standard model
uv run python -m experiments.lego.curriculum_ao_to_fsl \
    --dim 96 --n-heads 6 --n-layers 8 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42
```

| Model | Params | AO wandb | FSL wandb |
|-------|--------|----------|-----------|
| WS 96d/6h/8iter  | 125K unique (890K eff) | `84c0iqcs` | `ac6tyb31` |
| Std 96d/6h/8L    | 901K                   | `lyy1ex3n` | `9sl0fov0` |

- **GPU**: RTX 3060 (local)
- **Steps**: 58,593 AO + 97,656 FSL per model
- **Checkpoints**: In S3 (canonical store); backed up under `s3://brendanlong-experiments/wandb_migrated/lego-reasoning/`

## Weight-Shared Model (WS 96d/6h/8iter)

### AO Phase

Converged at step 11,000 -- 100% on all k=1-6. Clear staircase:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 7%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 92%  | 33%  | 0%   | 0%   | 0%   | 0%   |
| L2    | 100% | 100% | 47%  | 2%   | 0%   | 0%   |
| L3    | 90%  | 100% | 100% | 30%  | 1%   | 0%   |
| L4    | 79%  | 99%  | 100% | 97%  | 16%  | 13%  |
| L5    | 84%  | 100% | 100% | 100% | 90%  | 55%  |
| L6    | 87%  | 100% | 100% | 100% | 100% | 93%  |
| L7    | 84%  | 99%  | 100% | 100% | 100% | 100% |

States are **persistent** -- once t[1] reaches 92% at L1, it remains >=79% through all later layers.

### FSL Phase

100% accuracy maintained throughout all 97,656 steps. FSL loss settled at ~0.985.

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 6%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L1    | 67%  | 7%   | 0%   | 0%   | 0%   | 0%   |
| L2    | 30%  | 82%  | 0%   | 0%   | 0%   | 0%   |
| L3    | 17%  | 15%  | 80%  | 0%   | 0%   | 0%   |
| L4    | 17%  | 19%  | 18%  | 46%  | 0%   | 0%   |
| L5    | 17%  | 15%  | 18%  | 15%  | 16%  | 22%  |
| L6    | 17%  | 15%  | 18%  | 15%  | 16%  | 76%  |
| L7    | 25%  | 38%  | 17%  | 18%  | 17%  | 100% |

Intermediates become **transient wavefronts**: t[1] peaks at L1 (67%) then decays to chance (17%) by L3.

## Standard Model (Std 96d/6h/8L)

### AO Phase

Converged at step 10,000. Staircase compressed into L4-L7:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L5    | 93%  | 95%  | 64%  | 48%  | 31%  | 16%  |
| L7    | 97%  | 99%  | 58%  | 55%  | 48%  | 100% |

### FSL Phase

100% accuracy maintained. Weaker transient pattern:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L5    | 17%  | 20%  | 36%  | 16%  | 17%  | 17%  |
| L6    | 15%  | 21%  | 16%  | 33%  | 23%  | 41%  |
| L7    | 24%  | 20%  | 11%  | 16%  | 15%  | 100% |

## Minimum-Capacity Standard Model (48d/3h/8L)

```bash
uv run python -m experiments.lego.curriculum_ao_to_fsl \
    --dim 48 --n-heads 3 --n-layers 8 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 \
    --save-checkpoint --no-compile --seed 42
```

- **wandb**: AO `qo2f2myr`, FSL `2wy8h80u`

Surprising result: FSL *sharpened* the transient structure (t[2] peaks at L4 55%) and **improved ablation robustness** (63.5% -> 80.3%). The 48d model's small MLPs can't memorize multi-step shortcuts, so FSL pressure regularized toward cleaner per-layer composition.

## Sequential Ablation Test

Confirmed sequential computation in both curriculum-FSL models:

```bash
uv run python -m experiments.lego.ablate_sequential \
    --checkpoint data/lego/checkpoints/ws_curriculum_fsl/step_97656.pt --k 6
uv run python -m experiments.lego.ablate_sequential \
    --checkpoint data/lego/checkpoints/curriculum_fsl.pt --k 6
```

| Model | Progressive -3 | Reverse -3 | Random |
|-------|---------------|------------|--------|
| WS 96d FSL  | **84.2%** | 17.3% | 17-20% |
| Std 96d AO  | **78.6%** | 14.4% | 15-23% |
| Std 96d FSL | **73.7%** | 10.7% | 14-21% |
| Std 48d AO  | **63.5%** | 16.6% | 15-18% |
| Std 48d FSL | **80.3%** | 16.1% | 3-15%  |

## Supervised Linear Probes vs Logit Lens

**WS 96d/6h/8iter (curriculum-FSL) -- supervised probe:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 42%  | 14%  | 17%  | 17%  | 18%  | 18%  |
| L1    | 53%  | 29%  | 19%  | 18%  | 16%  | 17%  |
| L2    | 26%  | 49%  | 36%  | 22%  | 14%  | 17%  |
| L4    | 15%  | 19%  | 5%   | 51%  | 34%  | 25%  |
| L7    | 17%  | 14%  | 16%  | 4%   | 16%  | 88%  |

Supervised probes reveal a **more persistent staircase** than the logit lens. The "transient wavefront" from the logit lens is partly an artifact of probing with the wrong linear map.

## Per-Head Mechanistic Analysis

### WS 96d/6h/8iter -- Traveling erasure mechanism

**H1 -- Writer/Eraser (dual role)**: Writes large positive logit (+4 to +6) for t[j] at layer L_j (computation), then writes large negative logit (-5 to -8) for t[j] at layer L_{j+1} (erasure). Self-attention reaches 90-99% at the erase layer.

| Traj | Write layer | Write logit | Erase layer | Erase logit | Self-attn |
|------|-------------|-------------|-------------|-------------|-----------|
| t[1] | L0 | +5.88 | L2 | -8.02 | 90% |
| t[2] | L1 | +4.28 | L3 | -7.44 | 93% |
| t[3] | L2 | +2.87 | L4 | -6.95 | 96% |
| t[4] | L3 | +2.58 | L5 | -4.98 | 99% |
| t[5] | L4 | +2.33 | L6 | -7.28 | 97% |

**H0 -- Reader**: Attends 55-65% to the new element position and ~25-35% to the previous `<op>` position.

### Std 48d/3h/8L -- Deferred erasure mechanism

**H2 (all layers) -- Universal final eraser**: At L7, writes -3.8 to -4.2 for ALL trajectory states. Not selective -- erases everything at the end.

### Comparison

| Property | WS 96d (6 shared heads) | Std 48d (3x8 unique heads) |
|----------|-------------------------|---------------------------|
| Erasure | **Traveling** -- erase at L_{j+1} | **Fixed** -- erase all at L7 |
| Timing | Immediate (1 layer after write) | Delayed (L6-L7 for all) |
| Logit lens | Sharper peaks (clean per-layer) | Weaker (states superimposed) |

## Key Findings

1. **The compositional algorithm IS a stable attractor.** After 50M FSL examples, both models still compute via sequential composition. Accuracy never dropped below 99.9%.

2. **Intermediates become transient wavefronts.** Under AO, states are persistent. Under FSL, states appear transiently -- computed, used, then erased.

3. **Weight sharing produces cleaner transient staircases.** WS: clear 46-82% peaks. Standard: weaker 33-36% peaks with early intermediates erased.

4. **This is fundamentally different from FSL-from-scratch.** FSL-from-scratch shows NO staircase whatsoever. The curriculum model retains structure because composition was already learned.

5. **Weight sharing forces genuinely compositional algorithms** by preventing multi-step memorization. The shared MLP can only learn the 36-entry single-step group table, while standard unique-per-layer MLPs can specialize in multi-step shortcuts.

## Interpretability Implications

The probing tools form a hierarchy:
1. **Logit lens** -- assumption-free. Works well on WS models due to traveling erasure.
2. **Supervised probes** -- require labeled states. Found more than logit lens but require ground truth.
3. **Tuned lens** -- **fundamentally wrong for FSL models**: learns to predict next random element rather than trajectory state.
4. **Causal ablation** -- requires structural hypothesis. Confirmed sequential computation without decoding intermediates.

**Weight sharing makes the logit lens -- the only truly assumption-free tool -- work dramatically better.** This has safety implications: weight sharing (or similar constraints) could make hidden computation structurally harder to conceal from assumption-free probing.
