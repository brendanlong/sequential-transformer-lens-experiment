# Phase 17: Enhanced Logit Lens Metrics

## Goal

Test whether weight-shared models keep residuals closer to the embedding space using direct measurements, beyond the indirect tuned lens gap. Introduce new metrics: softmax probability mass, embedding explained variance (evar), U-dark fraction, normalized logit entropy, and principal angles between layer subspaces.

## New Metrics

1. **Softmax probability mass**: Average softmax probability on the correct trajectory token (vs top-1 accuracy). Smoother metric that reveals partial signal even when the correct token isn't the argmax.

2. **Embedding explained variance (evar)**: Fraction of the normed residual's squared norm in the span of token embedding vectors (computed via SVD). With vocab_size=10 and d_model=96, the embedding subspace is at most 10-dimensional. High evar = residual is "in" the embedding space; low evar = logit lens reads noise.

3. **U-dark fraction**: 1 - evar. Fraction of the residual orthogonal to all token embeddings.

4. **Normalized logit entropy**: Entropy of the logit lens softmax distribution, normalized by log(vocab_size). 0 = maximally peaked, 1 = uniform. Distinguishes "on-manifold ambiguous" from "off-manifold noise."

5. **Principal angles**: Mean cosine of principal angles between each layer's PCA subspace (at op positions) and the final layer. Measures subspace rotation across depth.

## Key Findings

### 1. Embedding explained variance is low for ALL models (16-30%)

Neither WS nor standard models keep residuals close to the embedding space:

| Model | Evar mean | Dark mean |
|-------|-----------|-----------|
| WS 96d/6h/8iter AO (uniform) | 25.8% | 74.2% |
| Std 96d/6h/8L AO (uniform) | 26.8% | 73.2% |
| Std 96d/6h/8L AO (kp=2) | 28.3% | 71.7% |
| WS 96d/6h/8iter AO (kp=2) | 16.5% | 83.5% |

The logit lens projects a ~96-dimensional residual onto a ~10-dimensional subspace. ~70-84% of what the residual carries is invisible. WS models are NOT more embedding-aligned — in fact, the kp=2 WS model has the *lowest* evar.

### 2. Probability mass tracks top-1 closely

For these models, probability mass and top-1 accuracy differ by only 1-3%. When the model encodes an intermediate state, it does so decisively.

### 3. Logit entropy reveals active erasure in FSL models

WS 8iter uniform FSL — entropy at each trajectory state's computation layer vs subsequent layers:

| State | Computed at | Entropy there | Entropy +1 layer | Entropy L7 |
|-------|-----------|---------------|-------------------|------------|
| t[1] | L1 (67%) | 23% | 63% | 78% |
| t[2] | L2 (82%) | 18% | 63% | 78% |
| t[3] | L3 (80%) | 30% | 64% | 78% |
| t[4] | L4 (46%) | 32% | 38% | 78% |
| t[5] | L5 (16%) | 38% | 69% | 78% |

The model computes, uses, then scrubs intermediates. The 78% terminal entropy matches log(6)/log(10) = 77.8% — uniform over the 6 S3 elements, which is exactly what FSL trains the `<op>` positions to predict.

### 4. FSL erasure converges to a fixed attractor token (`r2s`)

Probing *what* the logit lens predicts (not just whether it's correct) reveals that after erasure, 99-100% of examples converge to predicting `r2s` (token 6, the last element). This is not meaningful computation — it's the optimization path toward the uniform target. The low entropy at erased positions (5%) reflects this peaked-but-wrong prediction before it eventually spreads to uniform.

### 5. Position-level probing: `<op>` tokens are compute registers

Probing all sequence positions for each trajectory state:

- **AO models**: Intermediate states appear exclusively at `<op>` positions (e.g., t[1] at pos 4 = 92%), never at element positions (g₁ at pos 3 = 33% at best, and only because the element itself has partial correlation).
- **FSL models**: Element positions carry exactly 0% trajectory signal. All computation at `<op>` positions.
- The model uses element positions to preserve raw input for attention, and `<op>` positions as computation accumulators.

### 6. Principal angles show gradual subspace rotation

| Model | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|-------|----|----|----|----|----|----|-----|
| WS 8iter AO | 0.51 | 0.67 | 0.78 | 0.83 | 0.89 | 0.95 | 0.99 |
| WS 8iter FSL | 0.53 | 0.59 | 0.63 | 0.73 | 0.79 | 0.90 | 0.97 |
| WS 7iter AO | 0.63 | 0.74 | 0.81 | 0.88 | 0.93 | 0.97 | — |
| WS 7iter FSL | 0.67 | 0.72 | 0.79 | 0.83 | 0.88 | 0.98 | — |

Representations rotate gradually toward the final layer's subspace. AO and FSL show similar rotation profiles. Weight sharing does not produce more stable subspaces.

## Models Analyzed

All models from Phase 14 (uniform curriculum) and Phase 16 (k-power=2 curriculum), both AO and FSL checkpoints. Analysis code: `compare_enhanced_lens.py`.
