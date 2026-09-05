# Phase 6: Residual Stream Ablation

## Goal

Determine whether later-layer states at `<op>` positions are functionally important for the model's final output, or merely side effects of the shared block transforming all positions at each iteration.

## Setup

Two ablation modes:
- **Freeze**: keep the residual at its cutoff-layer value for all subsequent layers (preserves information for downstream attention)
- **Zero**: set the residual to zero for all subsequent layers (destroys information)

Models tested: Standard 128d/4h/8L, WS 256d/8h/8L, WS 192d/6h/8L

## Progressive Ablation: Freeze ALL `<op>` Positions

Freeze all `<op>` positions simultaneously at the same cutoff layer.

| Cutoff | Std 128d | WS 256d | WS 192d |
|--------|----------|---------|---------|
| >=L0   | 0.0%     | 0.0%    | 0.0%    |
| >=L3   | 2.8%     | 0.0%    | 0.0%    |
| >=L5   | 51.4%    | 18.4%   | 17.2%   |
| >=L6   | 74.4%    | 82.6%   | 67.6%   |
| >=L7   | 100.0%   | 100.0%  | 100.0%  |

**All models need L7 (the final layer) to be unfrozen.** The `<predict>` position at L7 attends to `<op>` positions and reads their current state.

## Per-Position Ablation: Freeze One `<op>` Position

Freeze only **one** position at a time, right after its functional layer (first layer where logit lens accuracy exceeds 50%).

Note: pos 14 is the `<predict>` token for k=6. Ablating it is especially destructive because the answer is read from this position.

**Freeze mode (keep functional layer's value):**

| Position | Std 128d | WS 256d | WS 192d |
|----------|----------|---------|---------|
| pos 4 (t[1]) | 100% | 100% | 100% |
| pos 6 (t[2]) | 100% | 99.6% | 99.6% |
| pos 8 (t[3]) | 100% | 100% | 99.2% |
| pos 10 (t[4]) | 100% | 100% | 96.6% |
| pos 12 (t[5]) | 100% | 100% | 100% |
| pos 14* (t[6]) | 74.4% | 100% | 100% |

*pos 14 = `<predict>` token

**Zero mode (destroy information):**

| Position | Std 128d | WS 256d | WS 192d |
|----------|----------|---------|---------|
| pos 4 (t[1]) | 100% | 100% | 99.0% |
| pos 6 (t[2]) | 100% | 98.4% | 98.8% |
| pos 8 (t[3]) | 97.6% | 100% | 96.0% |
| pos 10 (t[4]) | 98.0% | 100% | 87.2% |
| pos 12 (t[5]) | 80.0% | 100% | 100% |
| pos 14* (t[6]) | 16.2% | 16.2% | 16.0% |

*pos 14 = `<predict>` token

## Key Findings

### Finding 1: Later-layer states don't matter individually, but matter collectively

Freezing **one** `<op>` position barely affects accuracy (all >=96.6%). But freezing **all** simultaneously is catastrophic (67-83% at best). Later-layer computations at `<op>` positions serve as **communication channels** between positions.

### Finding 2: Standard model is MORE dependent on final-position later states

Standard model's pos 14 drops to 74.4% when frozen and 16.2% when zeroed. WS 256d stays at 100% when frozen (only drops when zeroed). The standard model's later layers do more active computation at the last `<op>` position.

### Finding 3: WS 256d is most robust to individual ablation

In zero mode, WS 256d maintains 100% accuracy when zeroing any single `<op>` position except pos 14. The extra capacity provides redundancy -- information about intermediate states is distributed across multiple positions.
