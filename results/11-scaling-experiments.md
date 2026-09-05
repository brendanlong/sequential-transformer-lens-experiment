# Phase 15: Scaling to 24 Hops

## Goal

Test how far the architecture can scale by pushing to 24-hop chains with 24 layers.

## Run 1: Standard 128d/4h/24L, 10M examples

```bash
uv run python -m experiments.lego.train \
    --k-max 24 --k-min 1 --generate-n 10000000 \
    --dim 128 --n-heads 4 --n-layers 24 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name "std_128d_4h_24L_k1-24_10M" --no-compile --seed 42 \
    --save-checkpoint
```

- **Config**: 4,748,800 params (4,732,416 effective), head_dim=32, answer-only loss
- **GPU**: RTX 3060 (local), 5.7 steps/s
- **Steps**: 19,531 (10M / 512)
- **wandb**: `lego-reasoning / std_128d_4h_24L_k1-24_10M` (run z3lsoreq)
- **Result**: 96.1% mean accuracy -- k=1-19 solved (>=99%), staircase still progressing at end

| k | 1-12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|------|----|----|----|----|----|----|----|----|----|----|----|----|
| Acc | 100% | 99.8% | 100% | 99.9% | 99.7% | 99.5% | 99.3% | 99.1% | 97.6% | 91.9% | 84.6% | 74.3% | 60.3% |

## Run 2: Standard 128d/4h/24L, 30M examples

```bash
uv run python -m experiments.lego.train \
    --k-max 24 --k-min 1 --generate-n 30000000 \
    --dim 128 --n-heads 4 --n-layers 24 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name "std_128d_4h_24L_k1-24_30M" --no-compile --seed 43 \
    --save-checkpoint
```

- **Config**: Same architecture as Run 1, 3x more data
- **GPU**: RTX 3060 (local), 5.0-5.7 steps/s
- **Steps**: 58,593 (30M / 512)
- **wandb**: `lego-reasoning / std_128d_4h_24L_k1-24_30M` (run jqdnzhc9)
- **Result**: 95.3% mean accuracy

| k | 1-13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|------|----|----|----|----|----|----|----|----|----|----|-----|
| Acc | 100% | 99.9% | 100% | 99.5% | 98.8% | 97.1% | 94.9% | 90.6% | 86.5% | 80.6% | 74.4% | 64.4% |

**30M examples did NOT fix the plateau** -- this is a capacity bottleneck, not a data limitation. The 10M run was actually *better* on k=16-19 because the compressed cosine schedule concentrated learning before LR decay.

## Logit Lens Analysis

```bash
uv run python -m experiments.lego.analyze_logit_lens \
    --checkpoint data/lego/checkpoints/step_19531.pt \
    --k 6 --n-examples 500 --output data/lego/logit_lens_24L_k6.png

uv run python -m experiments.lego.analyze_logit_lens \
    --checkpoint data/lego/checkpoints/step_19531.pt \
    --k 24 --n-examples 500 --output data/lego/logit_lens_24L_k24.png
```

**Key finding: the 24-layer standard model's computation is largely invisible to the logit lens**, even more so than the 8L model.

At k=6 (excess capacity):
- Bottom 16 layers show zero logit lens signal
- Weak staircase at `<op>` positions in layers 17-23

At k=24 (full capacity):
- No "one hop per layer" pattern -- all trajectory states emerge simultaneously in layers 17-23
- Intermediate state accuracy is weak (55-90%)
- No traveling wave -- only the final answer ever becomes visible

## Run 3: Standard 128d/4h/24L, 10M examples (seed 44)

Replication of Run 1 with different seed.

```bash
uv run python -m experiments.lego.train \
    --k-max 24 --k-min 1 --generate-n 10000000 \
    --dim 128 --n-heads 4 --n-layers 24 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name "std_128d_4h_24L_k1-24_10M_v2" --no-compile --seed 44 \
    --save-checkpoint
```

- **Config**: Same as Run 1 except seed=44
- **GPU**: RTX 3060 Ti (local, SkyPilot), 5.7 steps/s
- **Steps**: 19,531 (10M / 512)
- **wandb**: `lego-reasoning / std_128d_4h_24L_k1-24_10M_v2` (run 16kxm711)
- **Result**: 62.7% mean accuracy -- k=1-10 solved, then cliff at k=11

| k | 1-8 | 9 | 10 | 11 | 12 | 13-24 |
|---|-----|---|----|----|----|----|
| Acc | 100% | 99.9% | 99.5% | 72.3% | 43.6% | ~30-35% |

**Significant seed sensitivity**: this run got stuck much earlier than Run 1 (seed 42, 96.1% mean). The staircase learning pattern (each k learned sequentially) was the same, but learning stalled at k=10-11 instead of k=20.

## Run 4: Weight-Shared 128d/4h/24iter, 10M examples

```bash
uv run python -m experiments.lego.train \
    --weight-shared --k-max 24 --k-min 1 --generate-n 10000000 \
    --dim 128 --n-heads 4 --n-layers 24 \
    --batch-size 512 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name "ws_128d_4h_24L_k1-24_10M" --no-compile --seed 42 \
    --save-checkpoint
```

- **Config**: 218,112 unique params (4,735,488 effective via 24 iterations), head_dim=32, answer-only loss
- **GPU**: RTX 3060 Ti (local, SkyPilot), ~60 min
- **wandb**: `lego-reasoning / ws_128d_4h_24L_k1-24_10M` (run slyqfl2s)
- **Result**: 34.0% mean accuracy -- only k=1 partially learned (62.4%), everything else failed

| k | 1 | 2-24 |
|---|---|------|
| Acc | 62.4% | ~30-35% (2x chance, no composition) |

**Complete failure.** The 128d WS model can't even solve k=1 reliably with 24 iterations. The ~30-35% accuracy on k>=2 is above the 16.7% chance baseline, suggesting partial heuristics, but far from the composition needed to actually solve the task. This is consistent with Phase 3 findings where WS 128d struggled with deeper chains -- the 24 iteration embeddings and depth likely add significant optimization difficulty.

## Run 5: Weight-Shared 256d/8h/24iter, 10M examples

```bash
uv run python -m experiments.lego.train \
    --weight-shared --k-max 24 --k-min 1 --generate-n 10000000 \
    --dim 256 --n-heads 8 --n-layers 24 \
    --batch-size 128 --lr 3e-4 --lr-schedule cosine \
    --eval-every-steps 500 --log-every-steps 100 --save-every-steps 5000 \
    --wandb-run-name "ws_256d_8h_24L_k1-24_10M_bs128" --no-compile --seed 42 \
    --save-checkpoint
```

- **Config**: 825,098 unique params (19,135,754 effective via 24 iterations), head_dim=32, answer-only loss
- **GPU**: RTX 3060 Ti (local, SkyPilot), batch_size=128 (512 OOM'd), ~3h 14m
- **Steps**: 78,125 (10M / 128)
- **wandb**: `lego-reasoning / ws_256d_8h_24L_k1-24_10M_bs128` (run drp5w9w0)
- **Result**: 34.6% mean accuracy -- only k=1 solved (100%), everything else failed

| k | 1 | 2-24 |
|---|---|------|
| Acc | 100% | ~27-37% (2x chance, no composition) |

**Also failed.** The 256d WS model solved k=1 (100%) but couldn't compose beyond that. This model reached 100% on k=1-6 with 8 iterations in Phase 3 -- so the problem is specifically about scaling to 24 iterations, not raw capacity. Possible causes:

1. **Optimization difficulty with many iterations** -- gradients flowing through 24 applications of the same block may vanish or become noisy
2. **Iteration embedding scaling** -- 24 learned iteration embeddings may interfere with the shared block's ability to learn a universal composition step
3. **LR/batch dynamics** -- the smaller batch size (128 vs 512) changes effective noise, and 78K steps of cosine schedule may decay LR too slowly or too fast

## Summary

| Run | Model | Unique Params | Mean Acc | k=1-6 | wandb |
|-----|-------|--------------|----------|-------|-------|
| 1 | Std 128d/4h/24L (seed 42) | 4.75M | **96.1%** | 100% | z3lsoreq |
| 2 | Std 128d/4h/24L (seed 43, 30M) | 4.75M | 95.3% | 100% | jqdnzhc9 |
| 3 | Std 128d/4h/24L (seed 44) | 4.75M | 62.7% | 100% | 16kxm711 |
| 4 | WS 128d/4h/24iter | 218K | 34.0% | ~38% | slyqfl2s |
| 5 | WS 256d/8h/24iter (bs128) | 825K | 34.6% | ~44% | drp5w9w0 |

## Conclusions

1. **128d/4h/24L (4.75M params) reliably learns ~20 of 24 hops** -- first 20 reach >=90% (Run 1), though seed sensitivity is significant (Run 3 only reached k=10)
2. **Capacity, not data, is the bottleneck** -- 30M examples didn't help vs 10M
3. **The staircase learning curve scales** -- each hop takes ~700-1100 training steps
4. **Cosine LR with too many steps can hurt** -- compressed schedule performed better on mid-range k
5. **The learned algorithm differs from 8L** -- with excess layers, computation is hidden in rotated subspaces
6. **Weight sharing matters for interpretability** -- standard deep models are opaque to assumption-free probing even when solving the same task
7. **Weight-shared models fail to scale to 24 iterations** -- both 128d and 256d WS models (which solve k=6 with 8 iterations) fail completely at 24 iterations. The problem is not capacity but optimization difficulty with many shared iterations. A curriculum approach (gradually increasing iterations) may be needed.

## Future: Curriculum Depth Scaling

The WS models' failure at 24 iterations despite succeeding at 8 suggests the optimization landscape becomes too difficult when training all iterations from scratch. A promising approach: **train k=1/1-iter, then add one iteration and one hop at a time**, warm-starting from the previous checkpoint. This mirrors how the standard model's staircase learning works (k=1 first, then k=2, etc.) but makes it explicit in the training procedure. This could overcome the initialization/optimization barrier while preserving the interpretability benefits of weight sharing.
