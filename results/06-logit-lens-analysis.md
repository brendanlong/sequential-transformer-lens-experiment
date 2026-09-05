# Phase 7 + 11: Logit Lens Analysis

## Goal

Track logit lens accuracy at `<op>` positions over training to understand whether intermediate states sharpen or decay, and comprehensively compare all four headline models with both logit lens and tuned lens across all position types.

## Phase 7: Sharpening Over Training

### 256d/8h: Peak logit lens accuracy at `<op>` positions over training

| State | Step 5k | Step 10k | Step 15k | Step 19.5k | Trend |
|-------|---------|----------|----------|------------|-------|
| t[1]  | 78% (L1) | 78% (L1) | 73% (L1) | 73% (L1) | declining |
| t[2]  | 87% (L6) | 92% (L5) | 86% (L3) | 86% (L3) | stable |
| t[3]  | 86% (L5) | 89% (L5) | 82% (L4) | 83% (L5) | declining |
| t[4]  | 85% (L5) | 89% (L5) | 90% (L5) | 89% (L5) | stable |
| t[5]  | 84% (L7) | 86% (L6) | 87% (L7) | 90% (L7) | slight increase |
| t[6]  | 97% (L7) | 100% (L7) | 100% (L7) | 100% (L7) | saturated |

**The 256d model does NOT get sharper.** Early states are flat or declining. The traveling wave intensifies with training.

### 192d/6h: Peak logit lens accuracy over training

| State | Step 5k | Step 9.7k | Trend |
|-------|---------|-----------|-------|
| t[1]  | 96% (L7) | 96% (L5-L7) | stable |
| t[2]  | 89% (L7) | 91% (L5) | improving |
| t[3]  | 91% (L7) | 95% (L7) | improving |
| t[4]  | 96% (L7) | 98% (L7) | improving |
| t[5]  | 98% (L7) | 99% (L7) | improving |
| t[6]  | 99% (L7) | 100% (L7) | saturated |

**The 192d model IS getting sharper.** All states are improving or stable. Near capacity limits, the embedding subspace is the most parameter-efficient way to store intermediate states.

### Interpretation

- **256d/8h**: pulls early intermediate states away from the embedding basis (t[1]: 78%->73%). Treats early states as expendable once consumed.
- **192d/6h**: actively converges toward the embedding space. The capacity constraint forces reuse of the unembedding matrix's representation space.

## Phase 11: Comprehensive Lens Comparison (Post-Cleanup Replication)

### Setup

Retrained all 4 headline models with 15M streaming examples and ran comprehensive analysis:

```bash
uv run python -m experiments.lego.compare_tuned_lens \
    --generate-n 15000000 --train-only

uv run python -m experiments.lego.full_lens_comparison \
    --eval-examples 1000 --tuned-lens-examples 10000
```

- **GPU**: RTX 3060 (local)
- **Training data**: 15M streaming examples per model, batch_size=512, lr=3e-4, cosine schedule
- **Tuned lens**: 10,000 examples, 1 pass, lr=1e-3, AdamW, per-layer KL divergence
- **Eval**: 1,000 examples at k=6

### Convergence

| Model | Params | Converged Step | Early Stop Step |
|-------|--------|---------------|-----------------|
| Large Standard (128d/4h/8L) | 1.59M | 12,500 | 14,500 |
| Small Standard (48d/3h/8L) | 229K | 12,000 | 14,000 |
| Large WS (256d/8h/8L) | 825K unique | 5,500 | 7,500 |
| Small WS (96d/6h/8L) | 125K unique | 12,500 | 14,500 |

- **wandb runs**: `tuned_lens_std_128d`, `tuned_lens_std_48d`, `tuned_lens_ws_256d`, `tuned_lens_ws_96d`

### Results: Op Positions (Staircase Pattern)

**Large Standard (128d/4h/8L) -- LL/TL:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L0 | 0%/17% | 0%/15% | 0%/20% | 0%/17% | 0%/19% | 0%/18% |
| L3 | 14%/89% | 23%/62% | 4%/42% | 1%/26% | 0%/23% | 4%/19% |
| L4 | 81%/100% | 98%/100% | 71%/96% | 25%/71% | 10%/47% | 10%/39% |
| L5 | 100%/100% | 100%/100% | 96%/100% | 73%/94% | 54%/78% | 50%/57% |
| L7 | 100%/100% | 100%/100% | 100%/100% | 96%/96% | 95%/95% | 100%/100% |

**Small Standard (48d/3h/8L) -- LL/TL:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L4 | 47%/54% | 42%/62% | 17%/29% | 24%/24% | 13%/24% | 10%/21% |
| L5 | 54%/54% | 47%/57% | 35%/37% | 28%/34% | 19%/24% | 11%/24% |
| L7 | 56%/56% | 70%/70% | 42%/42% | 51%/51% | 56%/56% | 100%/100% |

**Large WS (256d/8h/8L) -- LL/TL:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L1 | 90%/85% | 1%/70% | 0%/25% | 0%/16% | 0%/15% | 0%/17% |
| L3 | 71%/85% | 83%/81% | 41%/71% | 4%/49% | 0%/27% | 0%/23% |
| L5 | 68%/82% | 76%/86% | 78%/70% | 90%/77% | 24%/81% | 23%/56% |
| L7 | 79%/79% | 87%/86% | 68%/68% | 75%/76% | 85%/85% | 100%/100% |

**Small WS (96d/6h/8L) -- LL/TL:**

| Layer | t[1] | t[2] | t[3] | t[4] | t[5] | t[6] |
|-------|------|------|------|------|------|------|
| L1 | 66%/67% | 12%/37% | 4%/23% | 0%/17% | 0%/16% | 0%/17% |
| L3 | 97%/61% | 100%/83% | 81%/96% | 8%/46% | 0%/18% | 0%/20% |
| L5 | 67%/56% | 95%/78% | 99%/96% | 100%/100% | 59%/92% | 42%/55% |
| L7 | 59%/59% | 79%/79% | 96%/96% | 99%/99% | 98%/98% | 100%/100% |

### Results: Element Positions

**Small WS shows the most interpretable element encoding:**

| Layer | start | op1 | op2 | op3 | op4 | op5 | op6 |
|-------|-------|-----|-----|-----|-----|-----|-----|
| L0 | 100% | 17% | 18% | 15% | 15% | 14% | 18% |
| L3 | 33% | 47% | 46% | 31% | 17% | 13% | 15% |
| L5 | 33% | 45% | 47% | 55% | 38% | 23% | 17% |
| L7 | 33% | 42% | 46% | 59% | 52% | 45% | 32% |

Standard models show almost no element position structure (14-36% for the large, mostly noise for the small).

### Summary Table

| Model | Position | Logit Lens | Tuned Lens | Delta pp | Delta % |
|-------|----------|-----------|-----------|---------|---------|
| Large Std (128d/4h/8L) | Op | 32.7% | 51.2% | +18.5pp | +56.7% |
| Large Std (128d/4h/8L) | Predict | 7.6% | 19.9% | +12.3pp | +161.3% |
| Large Std (128d/4h/8L) | Element | 29.0% | 32.2% | +3.2pp | +11.1% |
| Small Std (48d/3h/8L) | Op | 16.2% | 28.2% | +12.0pp | +74.4% |
| Small Std (48d/3h/8L) | Predict | 8.7% | 17.8% | +9.1pp | +105.1% |
| Small Std (48d/3h/8L) | Element | 26.9% | 26.1% | -0.7pp | -2.7% |
| Large WS (256d/8h/8L) | Op | 36.5% | 56.0% | +19.6pp | +53.6% |
| Large WS (256d/8h/8L) | Predict | 5.1% | 19.7% | +14.6pp | +287.5% |
| Large WS (256d/8h/8L) | Element | 26.7% | 19.4% | -7.2pp | -27.0% |
| Small WS (96d/6h/8L) | Op | 44.8% | 53.8% | +9.0pp | +20.1% |
| Small WS (96d/6h/8L) | Predict | 7.4% | 19.7% | +12.3pp | +164.8% |
| Small WS (96d/6h/8L) | Element | 31.0% | 31.6% | +0.5pp | +1.7% |

**Architecture averages (op positions, layers 0-6):**
- Standard models: logit=24.5%, tuned=39.7%, gap=+15.3pp (+62.6%)
- Weight-shared models: logit=40.6%, tuned=54.9%, gap=+14.3pp (+35.1%)

### Data Flow Analysis

| Layer | Std 128d LL/TL | Std 48d LL/TL | WS 256d LL/TL | WS 96d LL/TL |
|-------|----------------|---------------|---------------|--------------|
| L0->t[1] | 0%/17% | 0%/19% | **0%/74%** | **0%/42%** |
| L1->t[2] | 0%/21% | 0%/17% | **1%/70%** | **12%/37%** |
| L2->t[3] | 0%/21% | 0%/10% | **18%/48%** | **14%/28%** |
| L3->t[4] | 1%/26% | 22%/26% | **4%/49%** | **8%/46%** |
| L4->t[5] | 10%/47% | 13%/24% | **3%/54%** | **5%/48%** |
| L5->t[6] | 50%/57% | 11%/24% | **23%/56%** | **42%/55%** |

WS models consistently show earlier computation via the tuned lens.

### Key Findings

**Finding 1: Results replicate after code cleanups.** All key findings from Phase 9 hold.

**Finding 2: The computation staircase is fundamentally sequential.** Small WS shows the clearest staircase: L1: t[1] at 66%, L2: t[2] at 79%, L3: t[3] at 81%, etc.

**Finding 3: Tuned lens reveals hidden computation 1-2 layers earlier.** For WS models, computation is consistently present 1-2 layers before the logit lens can detect it.

**Finding 4: Predict position shows minimal traveling wave.** All models jump to the final answer in the last 1-2 layers with no visible intermediate structure at `<predict>`.

**Finding 5: WS models encode running state at element positions.** The Small WS model shows trajectory state encoding at element positions with a clear diagonal pattern. Standard models show no such pattern.

### Figures

- `data/lego/full_lens_comparison_op.png` -- Staircase at op positions
- `data/lego/full_lens_comparison_predict.png` -- Traveling wave at predict position
- `data/lego/full_lens_comparison_element.png` -- Running state at element positions
- `data/lego/full_lens_comparison.png` -- Combined overview
- `data/lego/full_lens_comparison.json` -- All heatmap data in JSON format
