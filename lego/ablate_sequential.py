"""Ablation experiment: test if the model uses a sequential algorithm.

If the model computes t[j] at layer L_j by attending to clause j's positions,
then zeroing clause j's positions at layers > L_j should not affect accuracy
(that information was already consumed). But zeroing positions that are still
needed should break the computation.

We test a "progressive causal zeroing" schedule:
- At layer L, zero out the residual at all clause positions ≤ L
  (i.e., assume one clause is consumed per layer)
- If the model computes sequentially, this should preserve accuracy
- If it memorizes or uses a non-sequential algorithm, this should break it

We also test:
- Random zeroing (same number of positions, randomly chosen) as a control
- Reverse zeroing (zero clauses that HAVEN'T been consumed yet) — should break
- "All positions except answer" zeroing at the final layer

Usage:
    uv run python -m lego.ablate_sequential \
        --checkpoint data/lego/checkpoints/curriculum_fsl.pt
"""

import argparse

import torch
from torch import Tensor

from lego.analyze_logit_lens import make_logit_lens_fn, probe_op_positions
from lego.generator import generate_fixed_dataset
from lego.model import AnyModel
from lego.tokenizer import answer_position, encode
from lego.training import load_model


def clause_positions(k: int) -> list[list[int]]:
    """Get token positions for each clause (0-indexed).

    Sequence format: <start> e1 <op> g1 <op> g2 ... <predict> answer
    - Clause 0 (start element): positions [0, 1] = <start>, e1
    - Clause j (j≥1): positions [2j, 2j+1] = <op>, g_j

    Returns list of k+1 lists (clause 0 = start element, clauses 1..k = operations).
    """
    clauses: list[list[int]] = []
    # Clause 0: <start> e1
    clauses.append([0, 1])
    # Clauses 1..k: <op> g_j at positions 2j, 2j+1
    for j in range(1, k + 1):
        clauses.append([2 * j, 2 * j + 1])
    return clauses


def forward_with_ablation(
    model: AnyModel,
    input_ids: Tensor,
    zero_mask: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    """Forward pass with position zeroing at specified (layer, position) pairs.

    Args:
        model: The transformer model
        input_ids: (batch, seq_len) token IDs
        zero_mask: (n_layers, seq_len) boolean tensor.
            True = zero out this (layer, position) in the residual stream.

    Returns:
        logits, residuals (same as forward_with_residuals)
    """

    def hook(x: Tensor, layer_idx: int) -> Tensor:
        mask = zero_mask[layer_idx]  # (seq_len,)
        return x * (~mask).unsqueeze(0).unsqueeze(-1).float()

    return model.forward_with_layer_hooks(input_ids, hook)


def evaluate_with_ablation(
    model: AnyModel,
    device: torch.device,
    k: int,
    zero_mask: Tensor,
    n_examples: int = 1000,
    seed: int = 999,
) -> float:
    """Evaluate accuracy at the answer position with ablation applied."""
    examples = generate_fixed_dataset(k, n_examples, seed=seed)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        logits, _residuals = forward_with_ablation(model, input_ids, zero_mask)

    ans_pos = answer_position(k)
    preds = logits[:, ans_pos - 1, :].argmax(dim=-1)
    targets = input_ids[:, ans_pos]
    return (preds == targets).float().mean().item()


def make_progressive_mask(
    n_layers: int,
    seq_len: int,
    k: int,
    offset: int = 0,
) -> Tensor:
    """Create progressive causal zeroing mask.

    At layer L, zero out clause positions 0..min(L+offset, k).
    This assumes the model consumes one clause per layer.

    offset controls the alignment:
    - offset=0: zero clause 0 after L0, clause 1 after L1, etc.
    - offset=-2: delay zeroing by 2 layers (more conservative)
    """
    clauses = clause_positions(k)
    mask = torch.zeros(n_layers, seq_len, dtype=torch.bool)
    for layer_idx in range(n_layers):
        # At this layer, zero out clauses that should already be consumed
        n_consumed = min(layer_idx + 1 + offset, len(clauses))
        for clause_idx in range(max(0, n_consumed)):
            for pos in clauses[clause_idx]:
                mask[layer_idx, pos] = True
    return mask


def make_reverse_mask(
    n_layers: int,
    seq_len: int,
    k: int,
    offset: int = 0,
) -> Tensor:
    """Reverse of progressive: zero the clauses that HAVEN'T been consumed.

    This should break a sequential model because it removes info that's still needed.
    """
    clauses = clause_positions(k)
    mask = torch.zeros(n_layers, seq_len, dtype=torch.bool)
    for layer_idx in range(n_layers):
        n_consumed = min(layer_idx + 1 + offset, len(clauses))
        # Zero clauses that are NOT yet consumed (the future ones)
        for clause_idx in range(max(0, n_consumed), len(clauses)):
            for pos in clauses[clause_idx]:
                mask[layer_idx, pos] = True
    return mask


def make_random_mask(
    n_layers: int,
    seq_len: int,
    k: int,
    progressive_mask: Tensor,
    seed: int = 42,
) -> Tensor:
    """Random zeroing with same number of positions per layer as progressive."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    mask = torch.zeros(n_layers, seq_len, dtype=torch.bool)
    # Don't include the answer position or the <predict> position in candidates
    ans_pos = answer_position(k)
    predict_pos = ans_pos - 1
    candidate_positions = [p for p in range(seq_len) if p not in (predict_pos, ans_pos)]
    for layer_idx in range(n_layers):
        n_zeros = progressive_mask[layer_idx].sum().item()
        if n_zeros > 0 and len(candidate_positions) > 0:
            n_to_zero = min(int(n_zeros), len(candidate_positions))
            perm = torch.randperm(len(candidate_positions), generator=rng)
            for i in range(n_to_zero):
                mask[layer_idx, candidate_positions[perm[i]]] = True
    return mask


def run_logit_lens_ablated(
    model: AnyModel,
    device: torch.device,
    k: int,
    zero_mask: Tensor,
    n_examples: int = 500,
    label: str = "",
) -> None:
    """Run logit lens with ablation and print results."""
    examples = generate_fixed_dataset(k, n_examples, seed=999)
    input_ids = torch.tensor(
        [encode(ex) for ex in examples],
        dtype=torch.long,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        logits, residuals = forward_with_ablation(model, input_ids, zero_mask)

    ans_pos = answer_position(k)
    preds = logits[:, ans_pos - 1, :].argmax(dim=-1)
    targets = input_ids[:, ans_pos]
    acc = (preds == targets).float().mean().item()

    ll_fn = make_logit_lens_fn(model)
    heatmap = probe_op_positions(residuals, examples, ll_fn, k, device)
    n_layers = len(residuals)

    print(f"\n{label} k={k} (acc={acc:.0%}):")
    col_labels = [f"t[{j + 1}]" for j in range(k)]
    header = "  ".join(f"{c:>6s}" for c in col_labels)
    print(f"  Layer  {header}")
    for i in range(n_layers):
        vals = "  ".join(f"{heatmap[i, j]:>6.0%}" for j in range(k))
        print(f"  L{i:<3d}  {vals}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential ablation experiment",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (curriculum_fsl.pt)",
    )
    parser.add_argument("--k", type=int, default=6, help="Chain length to test")
    parser.add_argument("--n-examples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    model, model_config = load_model(args.checkpoint, device)

    k = args.k
    n_layers = model_config.n_layers
    sl = 2 * k + 4  # sequence length

    print(
        f"Model: {'WS' if model_config.weight_shared else 'Std'} "
        f"{model_config.dim}d/{model_config.n_heads}h/{n_layers}L"
    )
    print(f"Testing k={k}, seq_len={sl}, n_layers={n_layers}")
    print(f"Clause positions: {clause_positions(k)}")
    print()

    # === Baseline (no ablation) ===
    no_mask = torch.zeros(n_layers, sl, dtype=torch.bool, device=device)
    baseline_acc = evaluate_with_ablation(
        model,
        device,
        k,
        no_mask,
        n_examples=args.n_examples,
    )
    print(f"Baseline (no ablation):  {baseline_acc:.1%}")

    # === Progressive zeroing with different offsets ===
    print("\n--- Progressive zeroing (zero consumed clauses) ---")
    for offset in [-3, -2, -1, 0, 1, 2]:
        mask = make_progressive_mask(n_layers, sl, k, offset=offset).to(device)
        acc = evaluate_with_ablation(
            model,
            device,
            k,
            mask,
            n_examples=args.n_examples,
        )
        n_zeroed_per_layer = [int(mask[li].sum().item()) for li in range(n_layers)]
        print(
            f"  offset={offset:+d}: {acc:>6.1%}  (zeroed/layer: {n_zeroed_per_layer})"
        )

    # === Reverse zeroing ===
    print("\n--- Reverse zeroing (zero NOT-YET-consumed clauses) ---")
    for offset in [-3, -2, -1, 0, 1, 2]:
        mask = make_reverse_mask(n_layers, sl, k, offset=offset).to(device)
        acc = evaluate_with_ablation(
            model,
            device,
            k,
            mask,
            n_examples=args.n_examples,
        )
        n_zeroed_per_layer = [int(mask[li].sum().item()) for li in range(n_layers)]
        print(
            f"  offset={offset:+d}: {acc:>6.1%}  (zeroed/layer: {n_zeroed_per_layer})"
        )

    # === Random zeroing (control) ===
    print("\n--- Random zeroing (same count as progressive offset=0) ---")
    prog_mask = make_progressive_mask(n_layers, sl, k, offset=0).to(device)
    for seed in range(5):
        mask = make_random_mask(n_layers, sl, k, prog_mask, seed=seed).to(device)
        acc = evaluate_with_ablation(
            model,
            device,
            k,
            mask,
            n_examples=args.n_examples,
        )
        print(f"  seed={seed}: {acc:>6.1%}")

    # === Best progressive: logit lens to see what happens inside ===
    print("\n--- Logit lens with progressive zeroing (offset=0) ---")
    mask = make_progressive_mask(n_layers, sl, k, offset=0).to(device)
    run_logit_lens_ablated(model, device, k, mask, label="[Progressive]")

    print("\n--- Logit lens with reverse zeroing (offset=0) ---")
    mask = make_reverse_mask(n_layers, sl, k, offset=0).to(device)
    run_logit_lens_ablated(model, device, k, mask, label="[Reverse]")


if __name__ == "__main__":
    main()
