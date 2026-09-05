# Experiment Plan: Weight-Shared Transformer Interpretability via LEGO

## Research Question

Do weight-shared (Universal Transformer) models produce more interpretable intermediate representations than standard transformers when performing multi-hop compositional reasoning?

Standard transformers use independent weights per layer, allowing each layer to develop its own internal representation space. Weight-shared transformers reuse the same block at every iteration, which should constrain intermediate states to live in a consistent representational space -- potentially one aligned with the model's output (unembedding) basis. If so, simple probing tools like the logit lens should reveal more about the model's step-by-step computation in weight-shared models.

## Task: S3 Group Composition (LEGO)

We use a synthetic compositional reasoning task based on ["Unveiling Transformers with LEGO"](https://arxiv.org/abs/2206.04301) (Zhang et al., 2022), adapted for decoder-only transformers.

**Why S3?** The symmetric group S3 (6 elements: e, r, r2, s, rs, r2s) is the smallest non-abelian group. Non-commutativity means the order of operations matters, forcing the model to compose each step sequentially rather than taking shortcuts. We initially tried binary group composition ({id, neg} on {0, 1}) but found the model could exploit a parity shortcut -- the abelian structure allowed reducing any chain to a single XOR, bypassing genuine multi-hop reasoning.

**Task format:** Each sequence is a chain of S3 group elements followed by a prediction token:

```
<start> e <op> r <op> s <op> r2 <predict> ANSWER
```

The model must compute the left-fold composition g_k . ... . g_1 . start and predict the final group element. Chain lengths k range from 1 to 6 (or higher for scaling experiments).

## Prior Work: FineWeb-Edu

Before settling on the LEGO task, we explored weight-shared transformers on FineWeb-Edu (a filtered web text corpus). This provided early evidence that weight-shared models could match standard models on language modeling, and we also experimented with Mixture-of-Experts (MoE) variants. However, natural language lacks ground-truth intermediate states, making it impossible to directly measure whether intermediate representations are more interpretable. We switched to S3 LEGO because it provides a fully controlled setting where the correct intermediate computation at each step is known exactly, enabling precise measurement of logit lens and tuned lens accuracy.

## Architectures

- **Standard Transformer**: Independent weights per layer. Baseline for comparison.
- **Weight-Shared Transformer** (Universal Transformer): A single transformer block reused N times with learned iteration embeddings added to the residual stream. Same effective depth as the standard model but with dramatically fewer unique parameters.

## Loss Modes

- **Answer-Only (AO)**: Cross-entropy loss computed only at the final prediction position. Intermediate positions receive no gradient signal, leaving their representations unconstrained.
- **Full-Sequence Loss (FSL)**: Standard autoregressive loss on all positions. Pushes every position's representation toward predicting its next token.
- **AO->FSL Curriculum**: Train to convergence under AO, then switch to FSL. Tests whether the compositional algorithm learned under AO survives the FSL transition.

## Analysis Tools

- **Logit lens**: Apply the model's final LayerNorm + unembedding to intermediate residual stream states. No training required -- the simplest possible probe.
- **Tuned lens**: Per-layer learned affine probes (initialized to identity) trained to translate intermediate states before applying norm + unembedding. Reveals information present but not aligned with the unembedding basis.
- **Supervised linear probes**: Probes trained directly on labeled trajectory states. Reveals the maximum linearly-decodable information at each layer/position.
- **Residual stream ablation**: Freeze or zero residual streams at specific positions/layers to test functional dependencies.
- **Sequential ablation**: Zero out consumed vs. unconsumed clause positions to confirm sequential vs. parallel computation.
- **Per-head mechanistic analysis**: Attribute logit contributions to individual attention heads to identify writer, reader, and eraser roles.

## Key Metrics

For each model architecture and loss mode, we measure:

1. **Task accuracy** per chain length k (must reach 100% for fair comparison)
2. **Logit lens accuracy** at `<op>` positions: can we decode the running trajectory state t[j] at the position where it should be computed?
3. **Staircase pattern**: does each layer compute one composition step, visible as a diagonal in the layer x position accuracy heatmap?
4. **Traveling wave vs. persistent staircase**: do early intermediate states persist through later layers (persistent) or get overwritten (traveling wave)?
5. **Logit lens vs. tuned lens gap**: smaller gap means the model stays closer to the embedding basis (more inherently interpretable)

## Code

The code lives in `lego/` (this public release ships the subset behind the writeup; see the note at the top of `RESULTS.md`). See `RESULTS.md` for a summary of findings and links to detailed per-experiment results in `results/`.
