"""Tests for the ablation hooks."""

import torch

from lego.ablate_critical_layer import (
    accuracy_with_ablation,
    critical_layers_from_lens,
    op_positions,
)
from lego.ablate_sequential import clause_positions, make_progressive_mask
from lego.config import lego_model_config
from lego.generator import generate_fixed_dataset
from lego.model import create_model
from lego.tokenizer import answer_position, encode


def _tiny_model_and_inputs(k: int = 4) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = create_model(lego_model_config(dim=16, n_heads=2, n_layers=3)).eval()
    examples = generate_fixed_dataset(k, 32, seed=1)
    input_ids = torch.tensor([encode(ex) for ex in examples], dtype=torch.long)
    return model, input_ids


def test_op_positions_exclude_predict() -> None:
    k = 6
    assert op_positions(k) == [4, 6, 8, 10, 12]
    assert answer_position(k) - 1 not in op_positions(k)


def test_no_ablation_matches_plain_forward() -> None:
    model, input_ids = _tiny_model_and_inputs()
    k = 4
    with torch.no_grad():
        logits = model(input_ids)
    ans_pos = answer_position(k)
    plain = (logits[:, ans_pos - 1].argmax(-1) == input_ids[:, ans_pos]).float().mean()
    assert accuracy_with_ablation(model, input_ids, k, {}) == plain.item()  # type: ignore[arg-type]


def test_zeroing_past_last_layer_is_a_no_op() -> None:
    model, input_ids = _tiny_model_and_inputs()
    k = 4
    plain = accuracy_with_ablation(model, input_ids, k, {})  # type: ignore[arg-type]
    late = accuracy_with_ablation(model, input_ids, k, {4: 3})  # type: ignore[arg-type]
    assert late == plain


def test_hook_zeroes_only_requested_positions() -> None:
    model, input_ids = _tiny_model_and_inputs()
    seen: dict[int, torch.Tensor] = {}

    def hook(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx >= 1:
            x = x.clone()
            x[:, 4, :] = 0.0
        seen[layer_idx] = x
        return x

    with torch.no_grad():
        model.forward_with_layer_hooks(input_ids, hook)  # type: ignore[attr-defined]
    assert seen[0][:, 4, :].abs().sum() > 0
    assert seen[1][:, 4, :].abs().sum() == 0
    assert seen[1][:, 3, :].abs().sum() > 0


def test_critical_layers_from_lens() -> None:
    k = 4
    heatmap = torch.full((3, k), 0.17)  # chance everywhere
    heatmap[1, 0] = 0.9  # t[1] peaks at L1
    heatmap[2, 1] = 0.6  # t[2] peaks at L2
    heatmap[0, 1] = 0.4  # earlier but lower — the peak wins
    # t[3] never rises above chance
    assert critical_layers_from_lens(heatmap, k) == {4: 1, 6: 2, 8: None}


def test_progressive_mask_consumes_one_clause_per_layer() -> None:
    k = 3
    n_layers, sl = 4, 2 * k + 4
    mask = make_progressive_mask(n_layers, sl, k, offset=0)
    clauses = clause_positions(k)
    # after layer 0, clause 0 is zeroed; after layer 1, clauses 0-1; ...
    for layer in range(n_layers):
        expected = {p for c in clauses[: min(layer + 1, len(clauses))] for p in c}
        assert set(mask[layer].nonzero().flatten().tolist()) == expected
