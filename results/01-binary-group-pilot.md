# Phase 1: Binary Group Pilot

## Goal

Validate the LEGO task setup using the simplest possible group: binary {id, neg} on {0, 1}. Check whether a small transformer can learn multi-hop variable composition.

## Task Format

```
a id 1 ; b neg a ; c id b ; ? c 0
```

- **Group**: Binary {id, neg} on values {0, 1}
- **Format**: 4 tokens per clause `[var, op, ref, ;]` + 3-token query `[?, var, value]`
- **Tokenizer**: 33-token vocabulary (26 vars + 6 special + PAD)

## Training Runs

### Sanity Check -- 3 vars, 128d/4h/6L standard

```bash
uv run python -m experiments.lego.train \
    --n-vars 3 --n-layers 6 --dim 128 --n-heads 4 \
    --n-epochs 100 --batch-size 256 --lr 1e-3 \
    --wandb-run-name "3var_128d_4h_6L_sanity"
```

- **Config**: n_train=30000, n_test=3000, cosine LR, weight_decay=0, init_std=None (Kaiming), torch.compile=yes, seed=42
- **GPU**: RTX 3060 (local)
- **Result**: 100% accuracy on all depths by epoch 40
- **wandb**: `lego-reasoning / 3var_128d_4h_6L_sanity`

| Epoch | Depth 0 | Depth 1 | Depth 2 | Mean |
|-------|---------|---------|---------|------|
| 10    | 99%     | 51%     | 52%     | 67%  |
| 20    | 76%     | 74%     | 53%     | 68%  |
| 30    | 94%     | 99%     | 100%    | 98%  |
| 40    | 100%    | 100%    | 100%    | 100% |

### 6 vars, 128d/4h/6L standard

```bash
uv run python -m experiments.lego.train \
    --n-vars 6 --n-layers 6 --dim 128 --n-heads 4 \
    --n-epochs 200 --batch-size 256 --lr 1e-3 \
    --wandb-run-name "6var_128d_4h_6L"
```

- **Config**: n_train=60000, n_test=6000, cosine LR, weight_decay=0, init_std=None (Kaiming), torch.compile=yes, seed=42
- **GPU**: RTX 3060 (local)
- **Result**: 100% accuracy on all depths
- **wandb**: `lego-reasoning / 6var_128d_4h_6L`

## Logit Lens Analysis (6-var model)

Applied logit lens to the 6-var checkpoint. For one example chain:

```
z id 1 ; v id z ; c neg v ; s neg c ; w neg s ; r id w ;
Values: z=1, v=1, c=0, s=1, w=0, r=0
```

| Query    | Depth | L0  | L1     | L2     | L3  | L4     | L5     |
|----------|-------|-----|--------|--------|-----|--------|--------|
| ? z 1    | 0     | z   | z      | **1**  | 1   | 1      | 1      |
| ? v 1    | 1     | v   | **1**  | 1      | 1   | 1      | 1      |
| ? c 0    | 2     | c   | c      | **0**  | 0   | 0      | 0      |
| ? s 1    | 3     | s   | **1**  | 1      | 1   | 1      | 1      |
| ? w 0    | 4     | w   | w      | w      | 1   | 1(48%) | **0**  |
| ? r 0    | 5     | r   | 1      | 1      | 1   | 0(81%) | **0**  |

## Key Finding: Parity Shortcut

**The model learned a parity shortcut, not genuine composition.**

Evidence:
- Depth 1 (0 negs on path) solves in L1, but depth 2 (1 neg) takes until L2
- Depth 3 (2 negs, even parity) also solves in L1 -- same as depth 1
- If the model were doing step-by-step composition, depth 3 should require *more* layers than depth 2, not fewer

The binary group {id, neg} on {0, 1} is isomorphic to Z2, which is abelian. This means the chain can be reduced to a single parity computation: count negations mod 2 and XOR with the root value. The model doesn't need to resolve each link sequentially.

## Conclusion: Binary Group Is Too Simple

The binary {id, neg} group allows a parity shortcut that bypasses genuine multi-hop composition. To study how transformers compose operations across layers, we need a **non-abelian group** where:

1. Operations don't commute (order matters)
2. The chain can't be reduced to a simple aggregate
3. Each composition step must be resolved before the next

This motivated the switch to S3 (symmetric group on 3 elements), the smallest non-abelian group (6 elements: e, r, r2, s, rs, r2s).
