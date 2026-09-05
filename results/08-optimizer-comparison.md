# Phase 10: AdamW vs Adam Comparison

## Goal

Determine whether switching from Adam (weight_decay=0) to AdamW with proper weight decay (0.01) improves convergence or final accuracy on the LEGO task.

## Setup

Trained all 4 headline models with AdamW at two weight decay values (0.0 and 0.01), using identical settings to the Phase 9 runs (10M examples, batch_size=512, lr=3e-4, cosine schedule).

```bash
uv run python -m experiments.lego.compare_adamw --weight-decay 0.0
uv run python -m experiments.lego.compare_adamw --weight-decay 0.01
```

- **GPU**: RTX 3060 Ti (local)
- **wandb runs**: `adamw_{large_std,small_std,large_ws,small_ws}_wd{0.0,0.01}`

## Results

| Model | Params | AdamW (wd=0) | AdamW (wd=0.01) | Adam ref (wd=0) |
|-------|--------|-------------|-----------------|-----------------|
| Large Standard (128d/4h/8L) | 1.59M | 100% @ step 12,500 | 100% @ step 13,500 | 100% @ step ~13,000 |
| Small Standard (48d/3h/8L) | 229K | 100% @ step 15,000 | 99.3% (**never**) | 100% @ step ~17,000 |
| Large WS (256d/8h/8L) | 825K | 100% @ step 6,000 | 100% @ step 5,500 | 100% @ step ~9,500 |
| Small WS (96d/6h/8L) | 125K | 99.8% (never, needs 15M) | 95.2% (**never**) | 100% @ step ~12,000 (15M) |

### Per-k accuracy for non-converging models (wd=0.01)

**Small Standard (48d/3h/8L) -- AdamW wd=0.01:**

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 100% | 100% | 100% | 100% | 96% |

**Small WS (96d/6h/8L) -- AdamW wd=0.01:**

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 100% | 100% | 100% | 95% | 76% |

## Findings

### Finding 1: The optimizer switch itself is a no-op at wd=0

AdamW with wd=0 produces nearly identical results to Adam with wd=0. Expected: Adam and AdamW are mathematically equivalent when weight_decay=0.

### Finding 2: Weight decay (wd=0.01) helps large WS, hurts small models

The Large WS model (825K params) converged faster with wd=0.01 (step 5,500) than with wd=0 (step 6,000). But both small models were harmed -- the Small WS (125K params) dropped from 99.8% to 95.2%. Weight decay effectively reduces effective model capacity, which is fatal for models already at their minimum viable size.

### Finding 3: Longer chains are most affected by weight decay

In both failing small models, k=1-4 still reach 100%. Failures concentrate on k=5 and k=6, consistent with weight decay constraining the model's ability to maintain precise intermediate representations across many iterations.

### Finding 4: Weight decay is not uniformly beneficial for LEGO

Unlike typical NLP tasks, the LEGO task operates near the capacity floor. Weight decay acts as implicit regularization that reduces effective model capacity. Harmless or helpful for large models, but prevents convergence for capacity-constrained models.

## Recommendation

Keep AdamW as the default optimizer but leave `weight_decay=0.0` as the default for LEGO training. Users can opt into weight decay for large models where it may accelerate convergence.
