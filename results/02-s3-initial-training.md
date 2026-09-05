# Phase 2-3: S3 Initial Training

## Goal

Establish the S3 group composition task and train both standard and weight-shared transformers. Determine whether weight-shared models show more interpretable intermediate states via the logit lens.

## Task Design

Replaced the binary group with S3, the smallest non-abelian group (6 elements: e, r, r2, s, rs, r2s). Non-commutativity forces genuine sequential composition -- no parity shortcut possible.

```
<start> e <op> r <op> s <op> r2 <op> rs <op> r2s <predict> ANSWER
```

- **Group**: S3 with left-multiplication convention: g_k . ... . g_1 . start
- **Vocabulary**: 10 tokens (PAD, 6 elements, START, OP, PREDICT)
- **Loss**: Cross-entropy at answer position only (not full sequence)
- **Chain lengths**: k in [1, 6], mixed uniformly during training

## Standard Transformer Training

### 128d/4h/8L standard, k=1-6

```bash
PYTHONUNBUFFERED=1 uv run python -m experiments.lego.train \
    --k-max 6 --k-min 1 --generate-n 10000000 \
    --dim 128 --n-heads 4 --n-layers 8 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name 's3_128d_4h_8L_k1-6_lr3e-4_v2' --no-compile --seed 42
```

- **Config**: streaming mode (10M examples, 1 epoch), cosine LR with 200-step warmup, weight_decay=0, learned positional encoding, LayerNorm, GELU, seed=42
- **GPU**: RTX 3060 (local), torch.compile disabled (hung on small model)
- **wandb**: `lego-reasoning / s3_128d_4h_8L_k1-6_lr3e-4_v2` -- [link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/9vdjpflm)
- **Checkpoint**: `data/lego/checkpoints/step_19531.pt`
- **Result**: 100% accuracy on all k=1 through k=6

Convergence order (shorter chains first, as expected):

| Step  | k=1  | k=2  | k=3  | k=4  | k=5  | k=6  | Mean |
|-------|------|------|------|------|------|------|------|
| 1000  | 100% | 99%  | 83%  | 52%  | 29%  | 19%  | 64%  |
| 2000  | 100% | 100% | 100% | 100% | 97%  | 82%  | 97%  |
| 3000  | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Loss Mode Comparison: answer-only vs full-sequence

| | answer-only | full-sequence |
|---|---|---|
| **k=6 final** | 100% | 95.6% |
| **Mean final** | 100% | 99.2% |
| **Steps to 100% mean** | ~3,000 | never (19.5k steps) |
| **Steps to 90% mean** | ~2,000 | ~12,000 |
| **wandb** | `s3_..._v2` ([link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/9vdjpflm)) | `s3_..._fullseq` ([link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/bisxwwzw)) |

**Answer-only is ~5x faster and reaches 100%.** Full-sequence loss dilutes the composition signal with trivial structural predictions (`<op>`, `<predict>`) and unpredictable random operands.

## Logit Lens on Standard Model

### Finding 1: No intermediate states at `<predict>` position

At the `<predict>` token, the model jumps directly to the final answer in layers 5-7. No evidence of successive trajectory states appearing across layers.

### Finding 2: Intermediate states stored at `<op>` positions (staircase pattern)

The model stores intermediate composition results at `<op>` token positions. The `<op>` at position 2*(j+1) encodes trajectory[j].

| Layer | p4->t[1] | p6->t[2] | p8->t[3] | p10->t[4] | p12->t[5] | p14->t[6] |
|-------|---------|---------|---------|----------|----------|----------|
| L0    | 0%      | 0%      | 0%      | 0%       | 0%       | 0%       |
| L1    | 0%      | 0%      | 0%      | 0%       | 0%       | 0%       |
| L2    | 2%      | 3%      | 0%      | 0%       | 0%       | 0%       |
| L3    | 25%     | 27%     | 5%      | 1%       | 0%       | 3%       |
| L4    | 76%     | 94%     | 72%     | 30%      | 16%      | 9%       |
| L5    | 94%     | 100%    | 97%     | 73%      | 55%      | 51%      |
| L6    | 100%    | 100%    | 99%     | 88%      | 88%      | 74%      |
| L7    | 100%    | 100%    | 99%     | 91%      | 87%      | 100%     |

Clear staircase: earlier positions converge first (t[1] at L4, t[2] at L4, t[3] at L5, ..., t[6] at L7).

### Finding 3: Computation happens in non-logit-lens-visible subspace

When position P computes trajectory[j] at layer L, it must attend to the previous position's state at layer L-1. But at L-1, the dependency often has only ~25-30% logit-lens decodability. The model encodes intermediate states in a subspace not aligned with the unembedding matrix. The logit lens only captures the component that happens to project onto token space.

## Weight-Shared Transformer Training

### Hypothesis

The standard transformer hides intermediate composition states in a learned subspace not aligned with the unembedding matrix. Weight-shared models reuse the same block at every iteration, so intermediate states must live in the same representational space as outputs. This could make intermediate states logit-lens-readable.

### 128d/4h/8L weight-shared, k=1-6

```bash
PYTHONUNBUFFERED=1 uv run python -m experiments.lego.train \
    --weight-shared --k-max 6 --k-min 1 --generate-n 10000000 \
    --dim 128 --n-heads 4 --n-layers 8 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name 's3_ws_128d_4h_8L_k1-6_lr3e-4' --no-compile --seed 42 \
    --checkpoint-dir data/lego/checkpoints/ws
```

- **Config**: streaming mode (10M examples, 1 epoch), cosine LR with 200-step warmup, weight_decay=0, learned positional encoding, LayerNorm, GELU, seed=42, `use_iteration_embed=True`
- **GPU**: RTX 3060 Ti (local), torch.compile disabled
- **wandb**: `lego-reasoning / s3_ws_128d_4h_8L_k1-6_lr3e-4` -- [link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/70yhzmr4)
- **Checkpoint**: `data/lego/checkpoints/ws/step_19531.pt`
- **Parameters**: ~400K (vs ~4.6M standard -- ~12x reduction)
- **Result**: 82.7% mean accuracy (k=1-3: 100%, k=4: 90.9%, k=5: 72.4%, k=6: 33.1%)

The 128d weight-shared model plateaus around step 10,000 and doesn't improve. With only ~216K unique params, it lacks capacity for longer chains.

### 256d/8h/8L weight-shared, k=1-6 (capacity-matched)

```bash
PYTHONUNBUFFERED=1 uv run python -m experiments.lego.train \
    --weight-shared --k-max 6 --k-min 1 --generate-n 10000000 \
    --dim 256 --n-heads 8 --n-layers 8 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name 's3_ws_256d_8h_8L_k1-6_lr3e-4' --no-compile --seed 42 \
    --checkpoint-dir data/lego/checkpoints/ws256
```

- **wandb**: `lego-reasoning / s3_ws_256d_8h_8L_k1-6_lr3e-4` -- [link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/rbor0r7s)
- **Checkpoint**: `data/lego/checkpoints/ws256/step_19531.pt`
- **Parameters**: 825K unique (6.3M effective)
- **Result**: 100% accuracy on all k=1 through k=6 (converges by ~step 5,500)

### 3-Way Accuracy Comparison

| k   | Standard 128d | WS 128d | WS 256d |
|-----|---------------|---------|---------|
| 1   | 100%          | 100%    | 100%    |
| 2   | 100%          | 100%    | 100%    |
| 3   | 100%          | 100%    | 100%    |
| 4   | 100%          | 90.9%   | 100%    |
| 5   | 100%          | 72.4%   | 100%    |
| 6   | 100%          | 33.1%   | 100%    |

## 3-Way Logit Lens Comparison at k=6

**Standard 128d/4h** -- Persistent staircase:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L4    | 76%  | 94%  | 72%  | 30%  | 16%  | 9%   |
| L5    | 94%  | 100% | 97%  | 73%  | 55%  | 51%  |
| L7    | 100% | 100% | 99%  | 91%  | 87%  | 100% |

**WS 128d/4h** -- Only t[1] visible (capacity-limited):

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L4    | 95%  | 19%  | 10%  | 2%   | 2%   | 31%  |
| L7    | 94%  | 23%  | 12%  | 6%   | 6%   | 31%  |

**WS 256d/8h** -- Traveling wave:

| Layer | t[1]   | t[2]   | t[3]   | t[4]   | t[5]   | t[6]   |
|-------|--------|--------|--------|--------|--------|--------|
| L1    | **73%**| 6%     | 0%     | 0%     | 0%     | 0%     |
| L3    | 64%    | **86%**| 48%    | 3%     | 0%     | 0%     |
| L5    | 58%    | 86%    | 83%    | **89%**| 28%    | 18%    |
| L7    | 44%    | 59%    | 77%    | 75%    | **90%**| **100%**|

## Key Findings

### Finding 1: Capacity-limited WS hides intermediates; capacity-matched WS reveals them

The 128d WS model hides all intermediate states. The wider 256d WS model shows **all** intermediate states as logit-lens-decodable. Weight-shared models CAN show more interpretable intermediate computations, but only with sufficient capacity.

### Finding 2: Weight-shared models show a "traveling wave"

The standard model's staircase is **persistent**: once t[1] becomes decodable at L4, it stays at 100% through L7. The WS 256d model's staircase is a **traveling wave**: each state rises, peaks, then **decays** as later iterations continue transforming the residual stream.

| State | First >50% | Peak (layer, val) | Final (L7) |
|-------|------------|-------------------|------------|
| t[1]  | L1         | L1-L2 (73%)       | 44% (decay)|
| t[2]  | L2         | L3 (86%)          | 59% (decay)|
| t[3]  | L4         | L4-L5 (82-83%)    | 77%        |
| t[4]  | L5         | L5 (89%)          | 75% (decay)|
| t[5]  | L6         | L7 (90%)          | 90%        |
| t[6]  | L6         | L7 (100%)         | 100%       |

### Finding 3: WS model starts computing 3 layers earlier

Both WS models produce decodable states at L1, while the standard model doesn't exceed 25% until L3.

### Finding 4: The unembedding alignment tells a clear story

| Architecture   | t[2] at peak | t[2] at final | Behavior |
|---------------|-------------|---------------|----------|
| Standard 128d | 100% (L5)   | 100% (L7)     | Persistent |
| WS 128d       | 15% (L3)    | 23% (L7)      | Hidden |
| WS 256d       | 86% (L3)    | 59% (L7)      | Wave (peak then decay) |

## Interpretation

Weight sharing **does** change the computation pattern, but not in the way originally hypothesized:

1. Shared weights create a "traveling wave" pattern: each iteration computes one step in a logit-lens-visible representation, but subsequent iterations partially overwrite earlier results.
2. The standard model avoids this because independent layer weights can selectively preserve earlier computations.
3. Capacity matters critically: with insufficient parameters (128d), the WS model can't afford to use the logit-aligned subspace for intermediates at all.
4. The traveling wave is arguably more interpretable than the standard model's persistent staircase: it directly reveals the sequential nature of the computation.
