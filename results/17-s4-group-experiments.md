# Phase 20: S4 Group Experiments

## Goal

Test whether scaling from S3 (6 elements) to S4 (24 elements) is feasible for both weight-shared and standard transformers. S4's 576-entry Cayley table (vs S3's 36-entry table) makes per-step parallel lookup shortcuts ~16x more expensive, potentially pushing models toward genuinely sequential composition algorithms. S5 (120 elements) was previously found to be too hard (see wiki journal 2026-05-09).

## Architecture Sweep: WS models with 5M examples

Tested three WS configurations with 5M streaming examples to find the capacity threshold for S4.

```bash
# 128d/4h
uv run python -m experiments.lego.train \
    --group S4 --weight-shared --dim 128 --n-heads 4 --n-layers 8 \
    --generate-n 5000000 --batch-size 512

# 192d/6h
uv run python -m experiments.lego.train \
    --group S4 --weight-shared --dim 192 --n-heads 6 --n-layers 8 \
    --generate-n 5000000 --batch-size 512

# 256d/8h
uv run python -m experiments.lego.train \
    --group S4 --weight-shared --dim 256 --n-heads 8 --n-layers 8 \
    --generate-n 5000000 --batch-size 512
```

| Config | Params (unique) | wandb ID | k=0 | k=1 | k=2-6 | Mean |
|--------|----------------|----------|-----|-----|-------|------|
| 128d/4h/8iter | ~218K | kj79wu45 | 100% | 3.6% | ~4% | 17.7% |
| 192d/6h/8iter | ~468K | uijwvjgw | 100% | 3.8% | ~4% | 17.9% |
| 256d/8h/8iter | ~825K | 7yzch426 | 100% | 4.8% | ~4% | 18.0% |

**Result**: 5M examples is far too few for S4. All models learned only k=0 (trivial identity). Random chance for S4 is ~4.2% (1/24), and k≥1 accuracies are at chance.

## Architecture Sweep: WS models with 50M examples

Repeated the sweep with 10x more data.

```bash
# Same three configs with --generate-n 50000000
```

| Config | Params (unique) | wandb ID | k=0 | k=1 | k=2-6 | Mean |
|--------|----------------|----------|-----|-----|-------|------|
| 128d/4h/8iter | ~218K | xz94rexb | 100% | 17.1% | ~4% | 20.0% |
| 192d/6h/8iter | ~468K | xrn2ik09 | 100% | 21.2% | ~4% | 20.3% |
| 256d/8h/8iter | ~825K | ey1x0g9v | 100% | **100%** | ~4% | 31.6% |

**Result**: Staircase grokking pattern confirmed — models learn k=0 first, then k=1. The 256d model fully grokked k=1 within 50M examples; smaller models were mid-grok when training ended. k=2+ still at chance for all, indicating much more data needed.

## Full Training: WS 256d/8h with 500M examples

```bash
uv run python -m experiments.lego.train \
    --group S4 --weight-shared --dim 256 --n-heads 8 --n-layers 8 \
    --generate-n 500000000 --batch-size 512
```

- **Config**: 825K unique params, head_dim=32, answer-only loss, 8 iterations
- **GPU**: RTX 3060 Ti (local), ~16.8 steps/s
- **Steps**: 677,500 / 976,562 (manually stopped after convergence)
- **Runtime**: ~683 minutes (~11.4 hours)
- **wandb**: `lego-reasoning / S4_ws_256d_8h_8L_k0-6` (run qlmbvng4)
- **Defaults used**: lr=3e-4, lr_schedule=cosine, seed=42, torch.compile=on

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Acc | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

**Result**: Full convergence on all k=0 through k=6. S4 is learnable for WS models at 256d — it needs ~50x more data than S3 (which converged in ~10M at 96d), though the comparison is confounded by the larger model size (256d vs 96d). The grokking curve in wandb shows sequential hop-by-hop convergence similar to S3.

## Full Training: Standard 256d/8h/8L with 500M examples

```bash
uv run python -m experiments.lego.train \
    --group S4 --dim 256 --n-heads 8 --n-layers 8 \
    --generate-n 500000000 --batch-size 512
```

- **Config**: ~6.5M params, head_dim=32, answer-only loss, 8 layers
- **GPU**: RTX 3060 Ti (local), ~15 steps/s
- **Steps**: 758,400 / 976,562 (manually stopped after convergence)
- **Runtime**: ~768 minutes (~12.8 hours)
- **wandb**: `lego-reasoning / S4_std_256d_8h_8L_k0-6` (run bvvkx36a)
- **Defaults used**: lr=3e-4, lr_schedule=cosine, seed=42, torch.compile=on

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Acc | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

**Result**: Standard model also achieves full convergence. Both architectures solve S4 k=0-6 with 500M examples at 256d.

## Key Takeaways

1. **S4 is the right difficulty level between S3 and S5.** S3 (6 elements) is too easy — parallel shortcuts are cheap. S5 (120 elements) is too hard — even k=1 requires dim≥384. S4 (24 elements) is learnable at dim=256 but requires ~50-100x more data than S3, confirming that per-step shortcuts are genuinely more expensive.

2. **Data requirements scale with group size.** S3 WS converged in ~10M examples at 96d. S4 WS needed ~500M at 256d. The 576-entry Cayley table requires both more model capacity and more data to grok.

3. **Staircase grokking pattern preserved.** S4 shows the same sequential hop-by-hop grokking as S3 (k=0 → k=1 → k=2 → ...), just slower. This is consistent with the model learning genuine sequential composition.

## Next Steps

- **Logit lens analysis**: Compare internal representations of S4 WS vs standard models. Does S4's larger group force cleaner sequential composition visible to the logit lens?
- **Staircase loss on S4**: Test whether auxiliary supervision on intermediate states accelerates learning or changes the algorithm.
- **Smaller WS models**: Determine minimum viable WS capacity for S4 (128d and 192d may work with more data).

## Note on Checkpoints

The WS run did not have `--save-checkpoint` enabled and its local checkpoints were overwritten by the subsequent standard run. The standard model checkpoints are available in `data/lego/checkpoints/`. The WS model would need to be retrained with `--save-checkpoint` for logit lens analysis.
