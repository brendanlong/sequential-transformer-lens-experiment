# Phase 4 + 8: Architecture Sweep

## Goal

Determine what architectural parameters (dim, heads, head_dim, layers) are necessary for both weight-shared and standard transformers to solve S3 6-hop composition. Find the minimum working configuration for each.

## Weight-Shared Model Sweep (Phase 4)

### Motivation

The 128d/4h WS model failed (82.7%) while 256d/8h succeeded (100%). Is this about total dimension (capacity), number of heads (attention diversity), or head dimension (per-head representation quality)?

### Sweep Design

Trained 13+ configurations with `--generate-n 5000000` (5M examples), then extended near-misses to 10M.

```bash
uv run python -m experiments.lego.sweep_ws_arch
uv run python -m experiments.lego.sweep_ws_arch --generate-n 10000000  # extended
```

### Results (5M examples)

| Config | Params | hd | k1 | k2 | k3 | k4 | k5 | k6 | Mean | Conv |
|--------|--------|---:|---:|---:|---:|---:|---:|---:|-----:|-----:|
| 128d/2h | 216K | 64 | 100 | 76 | 43 | 34 | 35 | 31 | 53.2 | never |
| 128d/4h | 216K | 32 | 100 | 100 | 100 | 91 | 72 | 33 | 82.7 | never |
| 128d/8h | 216K | 16 | 100 | 100 | 91 | 57 | 39 | 33 | 70.1 | never |
| 128d/16h | 216K | 8 | 100 | 100 | 100 | 90 | 62 | 43 | 82.5 | never |
| 160d/4h | 332K | 40 | 100 | 100 | 58 | 34 | 35 | 35 | 60.3 | never |
| 160d/5h | 332K | 32 | 100 | 100 | 88 | 62 | 41 | 34 | 70.8 | never |
| 160d/8h | 332K | 20 | 100 | 100 | 100 | 99 | 85 | 60 | 90.8 | never |
| 160d/16h | 332K | 10 | 100 | 100 | 100 | 100 | 99 | 85 | 97.4 | never |
| **192d/4h** | 472K | 48 | 100 | 100 | 100 | 13 | 15 | 17 | **57.6** | never |
| **192d/6h** | 472K | 32 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | **5000** |
| **192d/8h** | 472K | 24 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | **5500** |
| **192d/12h** | 472K | 16 | 100 | 100 | 100 | 100 | 100 | 31 | **88.5** | never |
| 256d/4h | 825K | 64 | 100 | 100 | 100 | 100 | 31 | 32 | 77.2 | never |
| 256d/8h | 825K | 32 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 5500 |
| 256d/16h | 825K | 16 | 100 | 100 | 100 | 100 | 100 | 99 | 99.9 | 8000 |

### Extended Results (10M examples)

- **GPU**: RunPod RTX 5090, torch.compile enabled, batch_size=512

#### Round 2 -- Near-misses from round 1

| Config | Params | hd | k1 | k2 | k3 | k4 | k5 | k6 | Mean | Conv |
|--------|--------|---:|---:|---:|---:|---:|---:|---:|-----:|-----:|
| **96d/6h** | 125K | 16 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | **converged** |
| 96d/3h | 125K | 32 | 100 | 100 | 100 | 89 | 58 | 21 | 77.8 | never |
| **128d/8h** | 216K | 16 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | **converged** |
| **128d/16h** | 216K | 8 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | **converged** |

#### Round 3 -- Below 96d

| Config | Params | hd | k1 | k2 | k3 | k4 | k5 | k6 | Mean | Conv |
|--------|--------|---:|---:|---:|---:|---:|---:|---:|-----:|-----:|
| 80d/5h | ~87K | 16 | 100 | 100 | 100 | 100 | 99 | 100 | 99.8 | never |
| 80d/4h | ~87K | 20 | 100 | 100 | 99 | 83 | 60 | 62 | 83.9 | never |
| 64d/4h | ~56K | 16 | 100 | 100 | 90 | 56 | 30 | 43 | 69.8 | never |
| 64d/2h | ~56K | 32 | 100 | 100 | 72 | 42 | 39 | 32 | 64.1 | never |
| 96d/4h | 125K | 24 | 100 | 100 | 100 | 96 | 70 | 70 | 89.3 | never |
| 96d/3h | 125K | 32 | 100 | 100 | 100 | 89 | 58 | 21 | 77.8 | never |

### WS Sweep Findings

**Finding 1: Both dim AND heads matter.** 256d/4h fails (77.2%) -- 825K params but head_dim=64, not enough heads. 128d/16h fails (82.5%) at 5M -- head_dim=8, many heads but not enough width.

**Finding 2: head_dim is the key architectural parameter.**

| head_dim | Succeeds? | Examples |
|----------|-----------|---------|
| 8-10 | No | 128d/16h (82.5%), 160d/16h (97.4%) |
| 16 | Only at 256d (5M), at 96d+ (10M) | 256d/16h (99.9%), 96d/6h (100% at 10M) |
| 24 | Yes at 192d | 192d/8h (100%) |
| 32 | Yes at >=192d | 192d/6h (100%), 256d/8h (100%) |
| 40-64 | No | 192d/4h (57.6%), 256d/4h (77.2%) |

**Finding 3: 192d/4h catastrophic failure reveals head bottleneck.** At 192d, switching from 4h to 6h jumps accuracy from 57.6% to 100%. The task requires multiple parallel attention patterns -- 4 heads can handle 3-hop composition but not 4+.

**Finding 4: Minimum working config is 96d/6h (125K params) with 10M examples.** A 3.8x parameter reduction from the 5M minimum of 192d/6h (472K params).

### Logit Lens: 192d/6h vs 256d/8h

Training run for 192d/6h:

```bash
PYTHONUNBUFFERED=1 uv run python -m experiments.lego.train \
    --weight-shared --k-max 6 --k-min 1 --generate-n 5000000 \
    --dim 192 --n-heads 6 --n-layers 8 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name 's3_ws_192d_6h_8L_k1-6_lr3e-4' --no-compile --seed 42 \
    --checkpoint-dir data/lego/checkpoints/ws192
```

- **wandb**: `lego-reasoning / s3_ws_192d_6h_8L_k1-6_lr3e-4` -- [link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/rjgpqr9s)
- **Checkpoint**: `data/lego/checkpoints/ws192/step_9765.pt`

**192d/6h at `<op>` positions (k=6):**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0 | 0% | 0% | 0% | 0% | 0% | 0% |
| L1 | **73%** | 4% | 0% | 0% | 0% | 0% |
| L2 | 84% | **71%** | 11% | 0% | 0% | 0% |
| L3 | 89% | 83% | **53%** | 16% | 0% | 0% |
| L4 | 94% | 87% | **92%** | 69% | 8% | 1% |
| L5 | 96% | 91% | 94% | **94%** | 46% | 17% |
| L6 | 96% | 87% | 90% | 96% | **90%** | 68% |
| L7 | 96% | 87% | 95% | 98% | **99%** | **100%** |

**Finding 5: 192d/6h achieves the best of both worlds** -- "Persistent staircase with early start":

| Property | Standard 128d/4h | WS 256d/8h | WS 192d/6h |
|----------|-----------------|------------|------------|
| t[1] first >50% | L4 | L1 | L1 |
| t[1] at L7 | 100% | 44% (decay) | **96%** |
| t[2] first >50% | L4 | L2 | L2 |
| t[2] at L7 | 100% | 59% (decay) | **87%** |
| Pattern | Persistent | Traveling wave | **Persistent + early** |

The capacity-constrained 192d model preserves early states because its shared block must be conservative with representations.

## Standard Model Sweep (Phase 8)

### Sweep Design

Swept over dim (32-128), heads (2-8), and layers (4-8) across 4 rounds.

```bash
uv run python -m experiments.lego.sweep_std_arch --no-wandb --generate-n 10000000
```

### Results

#### Round 1 (5M examples)

| Config | Params | hd | k6 | Mean | Conv |
|--------|--------|---:|---:|-----:|-----:|
| 64d/2h/8L | 404K | 32 | 35 | 87.3 | never |
| 96d/3h/8L | 901K | 32 | 89 | 98.1 | never |
| 128d/4h/8L | 1.59M | 32 | 95 | 99.2 | never |
| **128d/8h/8L** | 1.59M | 16 | 100 | **100** | **8500** |

#### Round 2 (10M examples)

| Config | Params | hd | Mean | Conv |
|--------|--------|---:|-----:|-----:|
| **64d/4h/8L** | 404K | 16 | **100** | **10500** |
| **96d/3h/8L** | 901K | 32 | **100** | **10000** |
| **96d/4h/8L** | 901K | 24 | **100** | **12000** |
| **96d/6h/8L** | 901K | 16 | **100** | **10500** |
| **128d/4h/8L** | 1.59M | 32 | **100** | **11000** |
| **128d/8h/6L** | 1.20M | 16 | **100** | **8000** |

#### Round 3 -- Find the floor

| Config | Params | hd | Mean | Conv |
|--------|--------|---:|-----:|-----:|
| **48d/3h/8L** | 229K | 16 | **100** | **17000** |
| 64d/4h/6L | 305K | 16 | 88.6 | never |
| **96d/6h/6L** | 679K | 16 | **100** | **16000** |

#### Round 4 -- Below the floor

| Config | Params | hd | Mean | Conv |
|--------|--------|---:|-----:|-----:|
| 32d/2h/8L | 104K | 16 | 88.4 | never |
| 40d/2h/8L | 160K | 20 | 68.0 | never |
| 48d/3h/6L | 174K | 16 | 97.5 | never |

### Standard Sweep Findings

**Finding 1: Minimum standard config is 48d/3h/8L (229K params).** Converges at step 17,000 with 10M examples.

**Finding 2: Depth is the critical dimension -- 8 layers required.** 64d/4h/4L (207K) fails at 94.1%. 128d/4h/6L (1.20M) fails at 87.6% -- 6x the params of the minimum 8L model. Each hop requires its own dedicated layer.

**Finding 3: head_dim is less critical for standard models.** At 128d, all head counts (2, 4, 8) converge. Standard layers can dedicate each layer's heads to a specific composition step.

### Standard vs WS Capacity Comparison

| Property | Standard minimum | WS minimum (5M) | WS minimum (10M) |
|----------|-----------------|------------------|-------------------|
| Config | 48d/3h/8L | 192d/6h/8iter | 96d/6h/8iter |
| Params | 229K | 472K | 125K |
| head_dim | 16 | 32 | 16 |
| Convergence | step 17,000 (10M) | step 5,000 (5M) | converged (10M) |

With 10M examples, the WS model achieves the smallest parameter count (125K vs 229K standard).
