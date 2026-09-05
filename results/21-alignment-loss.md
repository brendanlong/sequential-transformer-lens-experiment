# Phase 21: Residual-Embedding Alignment Loss

## Hypothesis

Two auxiliary losses — soft alignment (pull residuals toward nearest embedding) and embedding repulsion (push element embeddings apart) — will make the residual stream more token-aligned, improving logit lens interpretability without hurting task accuracy.

## Conditions

All use Large Standard (128d/4h/8L, 1.59M params), S3, answer-only loss, 5M streaming examples.

| Condition | Flags | WandB ID |
|-----------|-------|----------|
| baseline | (none) | `hxxtj9e2` |
| align_only | `--align-loss --align-weight 0.1 --align-temp 1.0` | `31eqwo18` |
| align+repel | `--align-loss --repel-loss` (both weight 0.1) | `axcuf4n2` |
| repel_only | `--repel-loss --repel-weight 0.1` | `w34udtsf` |

### Exact commands

```bash
# baseline
./train.sh vast lego -- --generate-n 5000000 --save-checkpoint --wandb-run-name S3_align_baseline

# align_only
sky launch sky-align_only.yaml --yes --infra runpod --down \
  --env "TRAIN_ARGS=--generate-n 5000000 --align-loss --save-checkpoint --wandb-run-name S3_align_only" \
  --secret WANDB_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY

# align+repel
./train.sh remote lego -- --generate-n 5000000 --align-loss --repel-loss --save-checkpoint --wandb-run-name S3_align_repel

# repel_only
sky launch sky-repel_only.yaml --yes --infra runpod --down \
  --env "TRAIN_ARGS=--generate-n 5000000 --repel-loss --save-checkpoint --wandb-run-name S3_repel_only" \
  --secret WANDB_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY
```

## Results

### Task accuracy — no meaningful difference

| Condition | Mean Acc | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|-----------|----------|-----|-----|-----|-----|-----|-----|-----|
| baseline | **99.2%** | 100% | 100% | 100% | 100% | 100% | 99.9% | 94.3% |
| align_only | 98.7% | 100% | 100% | 100% | 100% | 100% | 99.9% | 90.8% |
| align+repel | 99.1% | 100% | 100% | 100% | 100% | 100% | 99.8% | 93.7% |
| repel_only | 99.0% | 100% | 100% | 100% | 100% | 100% | 99.9% | 93.0% |

All conditions converge to near-perfect accuracy on k=0-5. The k=6 variation (90.8-94.3%) is within normal run-to-run noise for 5M examples — these models haven't fully grokked k=6 yet (need ~10M). **No evidence of capability cost from alignment or repulsion losses.**

### Logit lens B top-1 — alignment HURTS, doesn't help

Average top-1 accuracy at op positions (fraction of positions where logit lens argmax matches true intermediate B):

| Layer | baseline | align_only | align+repel | repel_only |
|-------|----------|------------|-------------|------------|
| 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 2.3% | 0.4% | 0.6% | 2.6% |
| 3 | 11.4% | 2.1% | 2.0% | 10.1% |
| 4 | 27.8% | 15.4% | 12.9% | 27.2% |
| 5 | **45.0%** | 24.8% | 23.8% | 43.5% |
| 6 | **59.6%** | 30.7% | 25.8% | 55.1% |
| 7 | **72.4%** | 54.2% | 52.3% | 71.3% |
| **Mean** | **27.3%** | **16.0%** | **14.7%** | **26.2%** |

**The alignment loss substantially reduces logit lens visibility**, dropping mean B top-1 from 27.3% (baseline) to 16.0% (align_only) and 14.7% (align+repel). Repulsion alone has negligible effect (26.2%, within noise of baseline).

This is the opposite of what was hypothesized. The alignment loss pushes residuals closer to embeddings (see nearest cosine below), but the embeddings they align to are NOT the intermediate trajectory tokens.

### Nearest embedding cosine similarity — alignment works mechanically

| Layer | baseline | align_only | align+repel | repel_only |
|-------|----------|------------|-------------|------------|
| 0 | 0.658 | **0.686** | **0.686** | 0.658 |
| 1 | 0.528 | **0.623** | **0.621** | 0.529 |
| 2 | 0.426 | **0.570** | **0.570** | 0.430 |
| 3 | 0.349 | **0.501** | **0.511** | 0.355 |
| 4 | 0.300 | **0.475** | **0.483** | 0.305 |
| 5 | 0.256 | **0.488** | **0.486** | 0.261 |
| 6 | 0.226 | **0.502** | **0.497** | 0.233 |
| 7 | 0.219 | **0.481** | **0.477** | 0.225 |

The alignment loss *does* push residuals closer to embeddings — nearest cosine similarity roughly doubles (0.22→0.48 at layer 7). But residuals align to the **wrong** embeddings (PAD, `<op>`, etc. rather than the intermediate trajectory token B). The loss is unsupervised — it has no reason to prefer B over any other nearby embedding.

### What does it predict instead?

When the logit lens argmax doesn't match B, the top predictions reveal the failure mode:

**Baseline** misses predict `<op>` (the structural token) or `elem_0` (identity element `e`). This is the known pattern from Phase 7 — the residual stream stores trajectory information in directions that don't align with any particular token embedding.

**Alignment conditions** misses overwhelmingly predict **PAD** (token 0). The alignment loss pushes residuals toward PAD because PAD is included in the embedding matrix and PAD positions are masked out of the loss. Residuals at non-PAD positions can still align to PAD without being penalized. This is a design flaw: the loss pushes residuals toward the nearest embedding in the full vocab, but PAD is the "path of least resistance" since aligning to PAD doesn't require encoding any specific group-theoretic information.

### Embedding geometry — repulsion has negligible effect

| Metric | baseline | align_only | align+repel | repel_only |
|--------|----------|------------|-------------|------------|
| mean_cos | -0.015 | -0.013 | -0.014 | -0.015 |
| min_cos | -0.260 | -0.258 | -0.258 | -0.260 |
| max_cos | 0.121 | 0.124 | 0.121 | 0.119 |
| std_cos | 0.095 | 0.095 | 0.095 | 0.095 |

Element embeddings are already near-orthogonal (mean pairwise cosine -0.01 to -0.02) in all conditions, including baseline. The repulsion loss has essentially nothing to push apart. With only 6 elements in 128 dimensions, random initialization already gives near-orthogonal embeddings.

### Assignment entropy — no collapse

| Layer | baseline | align_only | align+repel | repel_only |
|-------|----------|------------|-------------|------------|
| 0 | 2.268 | 2.271 | 2.271 | 2.268 |
| 7 | 2.295 | 2.292 | 2.292 | 2.295 |

Max entropy for 10 tokens = ln(10) ≈ 2.303. All conditions are near-maximum entropy — no collapse. The soft assignment distributions are nearly uniform, meaning no residual is strongly aligned to any single token.

### Effective rank — no change

| Layer | baseline | align_only | align+repel | repel_only |
|-------|----------|------------|-------------|------------|
| 0 | 32.9 | 33.0 | 33.0 | 32.9 |
| 4 | 67.1 | 65.3 | 65.1 | 66.9 |
| 7 | 70.3 | 69.0 | 69.0 | 70.1 |

Effective rank is nearly identical across all conditions (~33 at layer 0, ~70 at layer 7, out of dim=128). The alignment loss doesn't meaningfully change the dimensionality of the residual stream.

## Interpretation

**The alignment loss is a negative result, but an informative one.**

1. **Mechanical success, semantic failure.** The alignment loss achieves its optimization objective — residuals are measurably closer to embeddings (cosine 0.48 vs 0.22). But it aligns to *structurally convenient* embeddings (PAD, `<op>`) rather than *semantically meaningful* ones (intermediate trajectory tokens). The loss has no supervision signal directing it toward B.

2. **The logit lens gets worse because alignment competes with computation.** The baseline model stores trajectory information at `<op>` positions in directions that happen to partially overlap with token embeddings (hence 27% B top-1). The alignment loss pressures the model to use the embedding subspace for *something*, and it chooses to encode PAD/`<op>` alignment rather than trajectory information. The result is that trajectory information gets pushed into the residual orthogonal complement, making the logit lens *less* readable.

3. **Repulsion is irrelevant for S3.** With 6 elements in 128 dimensions, embeddings are already nearly orthogonal. The repulsion loss has nothing to do.

4. **The finding contradicts the original hypothesis but aligns with existing RESULTS.md finding #4**: "Neither architecture keeps residuals 'in' the embedding space." The residual stream's ~70% non-embedding component isn't waste — it's where computation happens. Forcing alignment doesn't make the computation more interpretable; it just forces the model to waste capacity satisfying the alignment objective.

## Implications

- **Unsupervised alignment is not a viable path to logit lens readability.** Without telling the model *which* token to align to at each layer, it aligns to whatever is cheapest (PAD/structural tokens). This is analogous to how full-sequence loss destroys intermediate visibility (Phase 12) — gradient pressure pushes information away from the embedding basis when the training signal doesn't specifically reward embedding-aligned intermediates.

- **Supervised alignment (staircase loss) remains the only known way to force logit lens readability.** Phase 16 showed that staircase loss produces perfect 100% diagonal in the logit lens by directly supervising which token should appear at each (layer, position) pair. The cost is that you need to know the target tokens, which limits it to tasks with known intermediate states.

- **The result strengthens the "intermediate computation is naturally non-token-aligned" thesis.** Models don't keep residuals on the embedding manifold because it's not useful for them. The 16-30% evar from Phase 17 is not an accident or a missed optimization opportunity — it reflects the model's preference for using the full 128-dimensional space rather than the 10-dimensional token subspace.

## Follow-up ideas (deprioritized given negative result)

- ~~Sweep align_weight and align_temp~~ — unlikely to change the qualitative result
- ~~align_last_half~~ — same fundamental issue (unsupervised = wrong target)
- **align_supervised (LayerSkip-style)** — still worth testing as a comparison point, but Phase 16 (staircase loss) already covers this ground with more precision
- **Exclude PAD from alignment loss** — would prevent the PAD-alignment failure mode, but the model would likely just align to `<op>` or `<start>` instead. The core problem is unsupervised = no semantic target.
