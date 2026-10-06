# Two-hop summary tables

Printed by `uv run python -m looped_lm.report results/looped-lm`. Columns give the first layer at which the linear probe / the logit lens reaches 90% top-1 for each quantity (L0 = embeddings).

### train

| Model | answer acc. | bridge@r1: probe ≥ 90% / lens ≥ 90% | bridge@r1 lens: peak / last layer | target@r2: probe ≥ 90% / lens ≥ 90% | bridge@r2: best probe / best lens |
|---|---|---|---|---|---|
| vanilla 8L, train stage (L8) | 100% | never / never | 75% at L8 / 75% | never / L7 | 8% / 0% |
| vanilla 8L, ID stage (L8) | 100% | L3 / L7 | 98% at L7 / 97% | L7 / L6 | 97% / 0% |
| vanilla 4L, train stage (L4) | 99% | never / never | 1% at L4 / 1% | never / L3 | 4% / 0% |
| vanilla 4L, ID stage (L4) | 100% | L2 / L3 | 100% at L4 / 100% | L3 / L3 | 93% / 0% |
| looped 4L×2, train stage (L8) | 98% | never / never | 19% at L8 / 19% | never / L6 | 8% / 0% |
| looped 4L×2, ID stage (L8) | 100% | L3 / L4 | 98% at L5 / 59% | L5 / L4 | 92% / 0% |
| looped 4L×2, OOD stage (L8) | 100% | L2 / L1 | 100% at L2 / 100% | L6 / L5 | 99% / 0% |
| looped 4L×4, train stage (L16) | 99% | never / never | 61% at L16 / 61% | never / L11 | 7% / 0% |
| looped 4L×4, ID stage (L16) | 100% | L4 / L4 | 100% at L5 / 100% | L9 / L9 | 98% / 0% |
| looped 4L×4, OOD stage (L16) | 100% | L4 / L3 | 100% at L4 / 100% | L8 / L7 | 97% / 0% |
| looped 4L×8, train stage (L32) | 100% | never / never | 46% at L32 / 46% | never / L17 | 7% / 0% |
| looped 4L×8, ID stage (L32) | 100% | L5 / L4 | 100% at L6 / 100% | L19 / L16 | 100% / 0% |
| looped 4L×8, OOD stage (L32) | 100% | L4 / L5 | 100% at L9 / 75% | L20 / L19 | 100% / 0% |

### test ID

| Model | answer acc. | bridge@r1: probe ≥ 90% / lens ≥ 90% | bridge@r1 lens: peak / last layer | target@r2: probe ≥ 90% / lens ≥ 90% | bridge@r2: best probe / best lens |
|---|---|---|---|---|---|
| vanilla 8L, train stage (L8) | 1% | never / never | 75% at L8 / 75% | never / never | 4% / 2% |
| vanilla 8L, ID stage (L8) | 98% | L3 / L7 | 96% at L7 / 96% | L7 / L6 | 96% / 0% |
| vanilla 4L, train stage (L4) | 1% | never / never | 1% at L4 / 1% | never / never | 2% / 1% |
| vanilla 4L, ID stage (L4) | 99% | L2 / L3 | 100% at L4 / 100% | L3 / L3 | 91% / 0% |
| looped 4L×2, train stage (L8) | 1% | never / never | 17% at L8 / 17% | never / never | 3% / 0% |
| looped 4L×2, ID stage (L8) | 99% | L3 / L5 | 99% at L5 / 56% | L5 / L5 | 93% / 0% |
| looped 4L×2, OOD stage (L8) | 100% | L2 / L1 | 100% at L2 / 100% | L6 / L5 | 100% / 0% |
| looped 4L×4, train stage (L16) | 1% | never / never | 57% at L16 / 57% | never / never | 4% / 0% |
| looped 4L×4, ID stage (L16) | 99% | L4 / L4 | 100% at L5 / 100% | L9 / L9 | 98% / 0% |
| looped 4L×4, OOD stage (L16) | 100% | L4 / L3 | 100% at L4 / 100% | L8 / L7 | 97% / 0% |
| looped 4L×8, train stage (L32) | 2% | never / never | 47% at L32 / 47% | never / never | 4% / 0% |
| looped 4L×8, ID stage (L32) | 99% | L5 / L4 | 100% at L8 / 100% | L21 / L18 | 99% / 0% |
| looped 4L×8, OOD stage (L32) | 100% | L4 / L5 | 100% at L9 / 74% | L20 / L19 | 100% / 0% |

### test OOD

| Model | answer acc. | bridge@r1: probe ≥ 90% / lens ≥ 90% | bridge@r1 lens: peak / last layer | target@r2: probe ≥ 90% / lens ≥ 90% | bridge@r2: best probe / best lens |
|---|---|---|---|---|---|
| vanilla 8L, train stage (L8) | 0% | never / never | 27% at L8 / 27% | never / never | 0% / 1% |
| vanilla 8L, ID stage (L8) | 1% | never / never | 69% at L7 / 67% | never / never | 2% / 0% |
| vanilla 4L, train stage (L4) | 0% | never / never | 0% at L4 / 0% | never / never | 0% / 1% |
| vanilla 4L, ID stage (L4) | 1% | L3 / L3 | 100% at L3 / 100% | never / never | 3% / 0% |
| looped 4L×2, train stage (L8) | 0% | never / never | 2% at L8 / 2% | never / never | 0% / 0% |
| looped 4L×2, ID stage (L8) | 0% | never / L5 | 92% at L5 / 20% | never / never | 35% / 0% |
| looped 4L×2, OOD stage (L8) | 83% | L2 / L2 | 100% at L3 / 99% | never / never | 97% / 0% |
| looped 4L×4, train stage (L16) | 0% | never / never | 12% at L15 / 12% | never / never | 0% / 0% |
| looped 4L×4, ID stage (L16) | 20% | never / never | 86% at L6 / 83% | never / never | 72% / 0% |
| looped 4L×4, OOD stage (L16) | 82% | L4 / L4 | 99% at L5 / 98% | never / never | 83% / 0% |
| looped 4L×8, train stage (L32) | 0% | never / never | 7% at L29 / 7% | never / never | 0% / 0% |
| looped 4L×8, ID stage (L32) | 5% | never / never | 68% at L14 / 63% | never / never | 46% / 0% |
| looped 4L×8, OOD stage (L32) | 86% | L8 / L5 | 99% at L13 / 60% | never / never | 99% / 0% |

### Two-hop OMP, middle half of depth (layers at 25 to 75% of depth)

| Model | r1: R²(1) − null | r1: first atom = bridge | r2: R²(1) − null | r2: first atom = answer |
|---|---|---|---|---|
| vanilla 8L, ID stage | 0.03 | 43% | 0.17 | 20% |
| vanilla 4L, ID stage | 0.18 | 42% | 0.26 | 41% |
| looped 4L×2, ID stage | 0.18 | 72% | 0.12 | 57% |
| looped 4L×2, OOD stage | 0.77 | 100% | 0.21 | 33% |
| looped 4L×4, ID stage | 0.43 | 100% | 0.24 | 52% |
| looped 4L×4, OOD stage | 0.49 | 100% | 0.37 | 64% |
| looped 4L×8, ID stage | 0.43 | 100% | 0.08 | 53% |
| looped 4L×8, OOD stage | 0.59 | 100% | 0.28 | 32% |

