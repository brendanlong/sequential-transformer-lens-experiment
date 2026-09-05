# Phase 19: S5 Group Experiments

## Goal

Test whether larger, non-solvable groups force sequential composition. S3 (6 elements, solvable with composition factors Z/3Z and Z/2Z) allows structural shortcuts — models can track the sign and rotation class independently. S5 (120 elements, non-solvable) has a 14,400-entry Cayley table and no such decomposition, which should prevent both memorization shortcuts and structural shortcuts.

## Setup

Added parameterized group support (PR [#71](https://github.com/brendanlong/experiments/pull/71)):
- `--group` CLI flag: S3 (default), S4 (24 elts), A5 (60 elts), S5 (120 elts)
- `Tokenizer` class with group-dependent vocabulary (N+4 tokens)
- Programmatic Cayley table generation from permutation composition

All experiments use standard 8L architecture, answer-only loss, k=0-6 (unless noted), streaming data generation.

## Results

### Initial group sweep (5-10M examples)

| Group | dim | Heads | k=0 | k=1 | k=2 | k=3-6 | Wandb |
|-------|-----|-------|-----|-----|-----|--------|-------|
| S3 | 128 | 4 | 100% | 100% | 100% | 100/100/100/94% | S3_std_128d_baseline |
| S4 | 128 | 4 | 100% | 100% | 4% | ~4% | S4_std_128d |
| A5 | 256 | 8 | 100% | 100% | 1.5% | ~1.3% | A5_std_256d |
| S5 | 256 | 8 | 100% | 2% | 1.5% | ~1% | S5_std_256d |
| S5 | 128 | 4 | 100% | 3% | 0.6% | ~0.6% | S5_std_128d |

S4 and A5 learned single-step composition (k=1) but couldn't chain. S5 couldn't even learn k=1 at dim=256.

### Standard model dim sweep for S5 (k=1 only, 10M examples)

Isolated the single-step composition bottleneck.

| dim | MLP hidden | Params | k=1 final | Converge step | Wandb |
|-----|-----------|--------|-----------|---------------|-------|
| 128 | 512 | 1.6M | 1% (never) | - | S5_std_128d |
| 256 | 1024 | 6.3M | 1% (never) | - | S5_std_256d |
| **384** | **1536** | **14M** | **100%** | **13,000** | S5_std_384d_k1only |
| 512 | 2048 | 25M | 100% | 13,000 | S5_std_512d_k1only |
| 768 | 3072 | 57M | 100% | 12,000 | S5_std_768d_k1only |
| 1024 | 4096 | 101M | 100% | 6,000 | S5_std_1024d_k1only |

Sharp phase transition between dim=256 (total failure) and dim=384 (perfect convergence via grokking). The 384d learning curve: random for ~6K steps, then 1% -> 5% -> 47% -> 97% -> 100% in ~6K steps. Classic grokking.

### S5 full k=0-6 with 512d (20M and 50M examples, 3 seeds)

| Seed | Data | k=0 | k=1 | k=2 | k=3 | k=4-6 | Wandb |
|------|------|-----|-----|-----|-----|--------|-------|
| 42 | 20M | 100% | 41% | 1% | 1% | ~1% | S5_std_512d_k0-6_seed42 |
| 123 | 20M | 100% | 17% | 1% | 0% | ~1% | S5_std_512d_k0-6_seed123 |
| 7 | 20M | 100% | 100% | 1% | 2% | ~1% | S5_std_512d_k0-6_seed7 |
| 42 | 50M | 100% | 100% | 100% | 1% | ~1% | S5_std_512d_k0-6_50M_seed42 |
| 123 | 50M | 100% | 100% | 100% | 3% | ~1% | S5_std_512d_k0-6_50M_seed123 |
| 7 | 50M | 100% | 100% | 9% | 1% | ~1% | S5_std_512d_k0-6_50M_seed7 |

Highly seed-sensitive. With 50M examples, 2/3 seeds crack k=2. Each hop requires significantly more data than the previous.

### Long standard run (200M examples = 390K steps, seed 42)

**Command:**
```
uv run python -m experiments.lego.train \
    --group S5 --dim 512 --n-heads 8 --n-layers 8 \
    --generate-n 200000000 --batch-size 512 --seed 42 \
    --eval-every-steps 2000 --log-every-steps 500 \
    --wandb-run-name S5_std_512d_k0-6_200M_seed42
```

**GPU:** RTX 3090 (RunPod), ~19 hours

**Grokking staircase:**

| Step | k=1 | k=2 | k=3 | k=4 | Event |
|------|-----|-----|-----|-----|-------|
| 28K | 2% | 1% | 1% | 1% | Pre-grokking |
| 38K | **99%** | 1% | 1% | 1% | k=1 groks |
| 50K | 100% | 3% | 1% | 1% | k=2 starting |
| 62K | 100% | **75%** | 1% | 1% | k=2 mid-grok |
| 70K | 100% | 96% | 1% | 1% | k=2 converging |
| 82K | 100% | 97% | **39%** | 1% | k=3 starting |
| 84K | 100% | 99% | **90%** | 0% | k=3 mid-grok |
| 90K | 100% | 100% | **99%** | 1% | k=3 converged |
| 118K | 100% | 100% | 100% | 1% | k=3 stable |
| 200K | 100% | 100% | 100% | 0% | k=4 no progress |
| 300K | 100% | 100% | 100% | 1% | k=4 no progress |
| 390K | 100% | 100% | 100% | **1%** | k=4 never learned |

The model groks each hop sequentially: k=1 at ~38K, k=2 at ~62K, k=3 at ~82K. Each takes ~20-25K steps after the previous hop converges. **k=4 shows zero progress** for 300K steps (steps 90K-390K).

### Weight-shared model dim sweep (k=1 only, 20M examples)

| dim | MLP hidden | Unique params | k=1 final | Notes | Wandb |
|-----|-----------|---------------|-----------|-------|-------|
| 512 | 2048 | 3.2M | 1% | Never | S5_ws_512d_k0-6_50M_seed42 |
| **768** | **3072** | **7.2M** | **100%** | Grok at step 22K | S5_ws_768d_k1only |
| 1024 | 4096 | 12.7M | 2% | Failed! | S5_ws_1024d_k1only |
| 1536 | 6144 | 28.5M | 2% | Failed | S5_ws_1536d_k1only |
| 2048 | 8192 | 50.6M | 2% | Failed | S5_ws_2048d_k1only |

**Non-monotonic relationship**: only dim=768 works. All larger dims fail. The 768d run shows the same grokking pattern as standard models (random for 17K steps, then 1% -> 26% -> 73% -> 100% in ~4K steps). Larger models find suboptimal solutions and plateau.

### WS 768d on full k=0-6 (50M examples)

| k=0 | k=1 | k=2 | k=3 | k=4-6 | Wandb |
|-----|-----|-----|-----|--------|-------|
| 100% | 1% | 1% | 1% | ~1% | S5_ws_768d_k0-6_50M_seed42 |

WS 768d can learn k=1 in the k=1-only setting but **fails completely on the mixed k=0-6 task**. The diluted gradient (only ~1/7 of examples are k=1) pushes the grokking threshold beyond the training budget.

### WS 512d on full k=0-6 (50M, 3 seeds)

| Seed | k=0 | k=1 | k=2-6 | Wandb |
|------|-----|-----|-------|-------|
| 42 | 100% | 1% | ~1% | S5_ws_512d_k0-6_50M_seed42 |
| 123 | 100% | 1% | ~1% | S5_ws_512d_k0-6_50M_seed123 |
| 7 | 100% | 1% | ~1% | S5_ws_512d_k0-6_50M_seed7 |

Total failure across all seeds. WS at 512d can't learn S5 composition at all.

## Key findings

1. **S5 forces sequential composition** — the standard model learns each hop via a discrete grokking event, producing the same staircase pattern as S3 but 10-100x slower.

2. **Sharp capacity threshold** — dim >= 384 needed for S5 single-step (vs 48 for S3). Below threshold: random chance. Above: sudden grokking.

3. **Grokking staircase stalls at k=4** — after learning k=1,2,3 sequentially, 300K steps produce zero progress on k=4. Possible causes:
   - Layer budget: 8L model uses 3 layers for k=1-3, remaining 5 can't figure out k=4
   - Cosine LR decayed too far by step 90K
   - Fundamental optimization difficulty at this model size

4. **WS is broken for S5** — narrow 768d sweet spot for k=1 only, non-monotonic failure at larger dims, total failure on multi-step. The shared MLP can't reliably learn the 14,400-entry Cayley table.

5. **The solvability hypothesis is partially confirmed** — S5's non-solvability prevents abelian decomposition shortcuts, but the dominant effect is the much larger Cayley table creating a hard capacity/optimization barrier.

## S3 vs S5 comparison

| Property | S3 | S5 |
|----------|----|----|
| Group order | 6 | 120 |
| Cayley table | 36 entries | 14,400 entries |
| Solvable | Yes (composition factors Z/3Z, Z/2Z) | No (A5 simple) |
| Min dim (std, k=1) | 48 | 384 |
| Std convergence | k=6 in ~9K steps, 5M data | k=3 at 90K steps, k=4 wall (200M data) |
| WS viability | Primary architecture | Broken |
| Grokking per hop | ~1-2K steps | ~20-25K steps |
