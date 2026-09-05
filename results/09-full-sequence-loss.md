# Phase 12: Full-Sequence Loss Experiments

## Goal

Test whether full-sequence loss (FSL) -- training every position to predict its next token -- improves logit lens visibility by pushing all positions' representations toward the unembedding basis.

## Hypothesis

With answer-only loss, `<op>` positions receive no gradient and encode intermediate states in whatever subspace is convenient. Full-sequence loss should push ALL positions toward token space, making the logit lens more informative.

## Phase 12a: Standard Model with Full-Sequence Loss

### Training -- 256d/8h/12L standard, full-sequence loss

```bash
PYTHONUNBUFFERED=1 uv run python -m experiments.lego.train \
    --k-max 6 --k-min 1 --generate-n 30000000 \
    --dim 256 --n-heads 8 --n-layers 12 \
    --loss-mode full-sequence \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 10000 \
    --wandb-run-name "std_256d_8h_12L_fullseq_k1-6" \
    --save-checkpoint --no-compile --seed 42
```

- **Config**: 256d/8h/12L standard (9.49M params), streaming mode (30M examples), cosine LR with 200-step warmup, weight_decay=0, seed=42
- **GPU**: RTX 3060 (local)
- **wandb**: `lego-reasoning / std_256d_8h_12L_fullseq_k1-6` -- [link](https://wandb.ai/brendanlong-com/lego-reasoning/runs/wacuqea9)
- **Checkpoint**: S3 (canonical store), backed up under `s3://brendanlong-experiments/wandb_migrated/lego-reasoning/lego-std_256d_8h_12L_fullseq_k1-6/`
- **Result**: Converged at step 22,500 -- 100% on all k=1-6

### Logit Lens at `<op>` positions -- THE KEY COMPARISON

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L3    | 21%  | 16%  | 15%  | 17%  | 19%  | 59%  |
| L6    | 17%  | 15%  | 18%  | 15%  | 16%  | 98%  |
| L8    | 16%  | 15%  | 17%  | 17%  | 18%  | 100% |
| L11   | 19%  | 17%  | 16%  | 12%  | 18%  | 100% |

**The staircase is GONE.** Intermediate states t[1]-t[5] are at chance (~17%) across ALL layers. Only t[6] visible.

| State | Answer-only (128d/8L) | Full-sequence (256d/12L) |
|-------|----------------------|-------------------------|
| t[1]  | 100% at L7           | **17% (chance)**         |
| t[2]  | 100% at L7           | **17% (chance)**         |
| t[5]  | 87% at L7            | **18% (chance)**         |
| t[6]  | 100% at L7           | 100% at L8               |

### Finding 1: Full-sequence loss DESTROYS logit lens visibility (hypothesis rejected)

The opposite of the hypothesis occurred: FSL makes intermediate states completely invisible.

### Finding 2: Why full-sequence loss hides intermediate states

With FSL, each `<op>` position is trained to predict its next token -- a random S3 operand (uniform distribution). This actively pushes trajectory information **orthogonal** to the unembedding matrix. With answer-only loss, `<op>` positions have no gradient, so trajectory information naturally ends up partially aligned with the embedding basis.

## Phase 12b: Tuned Lens on Full-Sequence Standard Model

Tuned lens does NOT recover intermediate states.

| Layer | t[1] (LL/TL) | t[2] (LL/TL) | t[5] (LL/TL) | t[6] (LL/TL) |
|-------|---------------|---------------|---------------|---------------|
| L0    | 0%/17%        | 0%/15%        | 0%/16%        | 0%/17%        |
| L3    | 17%/18%       | 16%/15%       | 16%/17%       | 8%/32%        |
| L6    | 15%/16%       | 16%/16%       | 16%/17%       | 36%/65%       |
| L9    | 17%/14%       | 16%/17%       | 14%/16%       | 89%/94%       |
| L11   | 17%/15%       | 16%/18%       | 15%/19%       | 97%/97%       |

- **Logit lens avg (L0-L10)**: 17.2%
- **Tuned lens avg (L0-L10)**: 22.7%
- **Improvement**: +26.2% (vs +66-114% for answer-only models)

The information is genuinely inaccessible via any affine transformation of the residual stream.

## Phase 12c: Weight-Shared Model with Full-Sequence Loss

### Training

```bash
uv run python -m experiments.lego.train \
    --weight-shared --n-layers 12 --dim 256 --n-heads 8 \
    --loss-mode full-sequence --generate-n 30000000 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 1000 --save-every-steps 10000 \
    --wandb-run-name ws_256d_8h_12iter_fullseq_k1-6 \
    --save-checkpoint
```

- **Config**: 256d/8h/12 iterations, weight-shared, full-sequence loss, 2.1M params
- **GPU**: RTX 3060 (local)
- **wandb**: `lego-reasoning / ws_256d_8h_12iter_fullseq_k1-6` (6vyrn8iw)
- **Note**: 8 iterations stalled at 30% -- failed run: `ws_256d_8h_8L_fullseq_k1-6` (pi00y0k3)
- **Result**: 99.4% mean accuracy (k=6 plateaued at 96.2%)

Logit lens: same pattern as standard -- only t[6] emerges, t[1]-t[5] at chance.

Tuned lens avg (L0-L10): 23.6%, improvement +29.9%. All improvement concentrated on t[6].

**Weight sharing does NOT preserve intermediate state visibility under full-sequence loss.**

## Phase 12d: Grokking Experiment -- Same Architecture as Answer-Only

Does the same architecture (128d/4h/8L) eventually develop a staircase under FSL with 6x more training?

### Standard 128d/4h/8L, full-sequence loss, 100M examples

```bash
uv run python -m experiments.lego.train \
    --n-layers 8 --dim 128 --n-heads 4 \
    --loss-mode full-sequence --generate-n 100000000 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 1000 --log-every-steps 500 --save-every-steps 50000 \
    --wandb-run-name "std_128d_4h_8L_fullseq_grok_100M" \
    --save-checkpoint --no-compile --seed 42
```

- **wandb**: `lego-reasoning / std_128d_4h_8L_fullseq_grok_100M` (pbhu5kv6)
- **Checkpoint**: S3 backup `s3://brendanlong-experiments/wandb_migrated/lego-reasoning/lego-std_128d_4h_8L_fullseq_grok_100M/`
- **Converged at step 74,000** -- 100% on all k=1-6 (6x slower than AO: 12,500 steps)

**The staircase is STILL absent** -- even with the exact same architecture:

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0    | 0%   | 0%   | 0%   | 0%   | 0%   | 0%   |
| L4    | 19%  | 16%  | 16%  | 13%  | 17%  | 71%  |
| L7    | 19%  | 16%  | 16%  | 14%  | 18%  | 100% |

This eliminates the model-size confound. **The loss function ALONE determines whether intermediate states are visible.**

### WS 128d/4h/8iter, full-sequence loss, 100M examples

```bash
uv run python -m experiments.lego.train \
    --weight-shared --n-layers 8 --dim 128 --n-heads 4 \
    --loss-mode full-sequence --generate-n 100000000 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 1000 --log-every-steps 500 --save-every-steps 50000 \
    --wandb-run-name "ws_128d_4h_8iter_fullseq_grok_100M" \
    --save-checkpoint --no-compile --seed 42
```

- **wandb**: `lego-reasoning / ws_128d_4h_8iter_fullseq_grok_100M` (yx67758o)
- **Checkpoint**: S3 backup `s3://brendanlong-experiments/wandb_migrated/lego-reasoning/lego-ws_128d_4h_8iter_fullseq_grok_100M/`
- **Result**: k=1-4: 100%, k=5: 99.5%, k=6: 40.5% (did not converge)
- Same pattern: no staircase for t[1]-t[5]

## Phase 12e: MLP Probing -- Where Are Intermediate States?

**Single-position MLP probe (3-layer, 256->128->6) for t[1] at pos 4:** Every layer returns chance accuracy (17.7%). Information is genuinely not stored at any single position.

**Multi-position MLP probe results:**

| Positions | L0 | L3 | L6 | L9 | L11 |
|-----------|-----|-----|-----|-----|-----|
| pos 4 only (expected `<op>`) | 18% | 18% | 18% | 18% | 18% |
| pos 1+3 (e1 + e2 inputs) | 100% | 100% | 100% | 100% | 97% |
| all `<op>` positions | 19% | 19% | 18% | 18% | 18% |
| all element positions | 100% | 100% | 95% | 82% | 80% |

**Critical control**: The probe achieves 100% from pos 1+3 even at L0 (before any transformer processing). It's simply learning the S3 multiplication table from the two input embeddings (36 possible pairs -- trivially memorizable).

The model **never explicitly stores intermediate trajectory states** -- the original inputs are preserved at element positions in early layers and consumed by the computation, but no intermediate results are materialized anywhere.

## Phase 12 Summary

| Model | Logit Lens (t[1]-t[5]) | Tuned Lens (t[1]-t[5]) | t[6] Emergence |
|-------|------------------------|------------------------|----------------|
| Standard 256d/8h/12L | chance (17%) | chance (17%) | L3 (59%)->L8 (100%) |
| WS 256d/8h/12iter    | chance (17%) | chance (17%) | L5 (24%)->L11 (98%) |

**Key conclusions:**

1. **Full-sequence loss universally destroys intermediate state visibility** -- both architectures
2. **The mechanism is gradient-driven**: `<op>` positions must predict random next tokens, pushing trajectory information orthogonal to the unembedding basis
3. **Tuned lens cannot recover the hidden states** -- genuinely inaccessible via any affine probe
4. **Weight sharing doesn't help** under FSL pressure
5. **The information still exists** (models achieve near-100%) but requires nonlinear or multi-position analysis to decode
6. **Not a capacity issue**: same architecture (128d/4h/8L) shows clear staircase under AO but nothing under FSL
7. **Not a patience issue**: 100M examples doesn't recover the staircase

## Phase 13: Tiny Model Sweep -- AO vs FSL at Extreme Scale

**Hypothesis**: FSL models memorize input->output lookup tables rather than composing step-by-step. If so, FSL should be dramatically less efficient at tiny scales.

```bash
for dim in 8 12 16; do
  for loss in answer-only full-sequence; do
    uv run python -m experiments.lego.train \
      --weight-shared --n-layers 8 --dim $dim --n-heads 2 \
      --loss-mode $loss --generate-n 50000000 \
      --batch-size 512 --lr 3e-4 --lr-schedule cosine \
      --eval-every-steps 1000 --log-every-steps 5000 \
      --no-wandb --no-compile --seed 42
  done
done
```

- **GPU**: RTX 3060 (local)

| dim | params | Loss  | k=1    | k=2    | k=3  | k=4  | k=5  | k=6  |
|-----|--------|-------|--------|--------|------|------|------|------|
| 8   | 1,984  | AO    | **91%** | 46%   | 33%  | 31%  | 33%  | 20%  |
| 8   | 1,984  | FSL   | 17%    | 18%    | 17%  | 16%  | 18%  | 17%  |
| 12  | 3,552  | AO    | **100%** | **97%** | 19% | 17% | 16%  | 18%  |
| 12  | 3,552  | FSL   | 67%    | 15%    | 13%  | 18%  | 17%  | 19%  |
| 16  | 5,504  | AO    | **100%** | **100%** | 17% | 18% | 13% | 15%  |
| 16  | 5,504  | FSL   | 12%    | 18%    | 17%  | 17%  | 16%  | 16%  |

**AO dramatically outperforms FSL at all sizes.** The ~13x parameter efficiency gap (5.5K->72K for FSL to solve k=1-5) supports the hypothesis that AO learns a compositional algorithm (needing only the 36-entry group table) while FSL memorizes lookup tables.
