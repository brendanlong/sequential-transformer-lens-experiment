"""Tests for the probe-vs-lens analysis."""

import torch

from lego.config import lego_model_config
from lego.generator import generate_fixed_dataset
from lego.model import create_model
from lego.probe_lens_gap import (
    TARGETS,
    centered_element_unembedding,
    probe_accuracy,
    relative_visibility,
    run_label,
    run_product,
    state_position,
    visible_basis,
)
from lego.tokenizer import answer_position, encode


def test_state_positions_are_ops_then_predict() -> None:
    k = 6
    ex = generate_fixed_dataset(k, 1, seed=0)[0]
    tokens = encode(ex)
    ops = [state_position(j) for j in range(1, k)]
    assert len({tokens[p] for p in ops}) == 1
    assert state_position(k) == answer_position(k) - 1


def test_run_products_match_trajectory() -> None:
    for ex in generate_fixed_dataset(6, 50, seed=3):
        for j in range(7):
            assert run_product(ex, 0, j) == ex.trajectory[j]
        for j in range(1, 7):
            assert run_product(ex, j, j) == TARGETS["operand g_j"](ex, j)
            assert run_product(ex, 1, j) == TARGETS["operand product g_j…g_1"](ex, j)
    assert run_label(0, 3) == "g3…e0"
    assert run_label(2, 2) == "g2"
    assert run_label(1, 2) == "g2·g1"


def test_visible_basis_is_orthonormal_and_spans_element_differences() -> None:
    torch.manual_seed(0)
    model = create_model(lego_model_config(dim=16, n_heads=2, n_layers=2))
    q = visible_basis(model)
    assert q.shape == (16, 5)
    torch.testing.assert_close(q.T @ q, torch.eye(5), atol=1e-5, rtol=0)
    rows = model.tok_emb.weight[1:7].detach()
    diffs = rows - rows.mean(0)
    torch.testing.assert_close(diffs @ q @ q.T, diffs, atol=1e-5, rtol=0)


def test_probe_separates_separable_classes_and_not_noise() -> None:
    gen = torch.Generator().manual_seed(0)
    y = torch.randint(0, 6, (1200,), generator=gen)
    centers = torch.randn(6, 8, generator=gen) * 5
    x = centers[y] + torch.randn(1200, 8, generator=gen)
    assert probe_accuracy(x[:1000], y[:1000], x[1000:], y[1000:]) > 0.95
    noise = torch.randn(1200, 8, generator=gen)
    assert probe_accuracy(noise[:1000], y[:1000], noise[1000:], y[1000:]) < 0.35


def test_relative_visibility_is_one_for_isotropic_and_zero_for_dark() -> None:
    torch.manual_seed(0)
    model = create_model(lego_model_config(dim=16, n_heads=2, n_layers=2))
    w = centered_element_unembedding(model)
    iso = torch.randn(200_000, 16)
    assert abs(relative_visibility(iso, w) - 1) < 0.02
    q = visible_basis(model)
    dark = iso - iso @ q @ q.T
    assert relative_visibility(dark, w) < 1e-6
