import torch

from looped_lm import lens_metrics as lm
from looped_lm.twohop import RecurrentGPT2


def test_folded_readout_removes_shared_directions() -> None:
    w = lm.folded_readout(torch.randn(50, 8) + 3, torch.rand(8) + 0.5, centers=True)
    assert torch.allclose(w.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(w.sum(1), torch.zeros(50), atol=1e-4)


def test_singular_basis_matches_svd() -> None:
    w = torch.randn(300, 12)
    s, vh = lm.singular_basis(w, chunk=64)
    ref = torch.linalg.svdvals(w)
    assert torch.allclose(s, ref, rtol=1e-4)
    assert torch.allclose((w @ vh.T).pow(2).sum(0).sqrt(), s, rtol=1e-4)


def test_visibility_and_bands() -> None:
    w = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5]))
    s, vh = lm.singular_basis(w)
    iso = torch.randn(20000, 4)
    assert abs(lm.relative_visibility(iso, s, vh) - 1) < 0.05
    bands = lm.energy_by_singular_band(iso, vh)
    assert torch.allclose(bands, torch.full((4,), 0.25), atol=0.02)
    top = torch.tensor([[1.0, 0, 0, 0]])
    expected = 16 / (s.pow(2).mean().item())
    assert abs(lm.relative_visibility(top, s, vh) - expected) < 1e-4
    assert lm.energy_by_singular_band(top, vh)[0] > 0.999


def test_omp_recovers_two_atoms() -> None:
    gen = torch.Generator().manual_seed(0)
    atoms = torch.nn.functional.normalize(torch.randn(100, 32, generator=gen), dim=-1)
    y = 3 * atoms[7] + 2 * atoms[42]
    r2, chosen = lm.omp_r2(y[None], atoms, k_max=3)
    assert set(chosen[0, :2].tolist()) == {7, 42}
    assert r2[1] > 0.999


def test_outlier_counts() -> None:
    logits = torch.randn(4, 10000, generator=torch.Generator().manual_seed(1))
    logits[:, :3] += 20
    assert lm.outlier_counts(logits).tolist() == [3, 3, 3, 3]


def test_recurrent_model_records_every_execution() -> None:
    model = RecurrentGPT2(vocab=11, dim=24, n_heads=4, n_layers=2, n_iterations=3)
    residuals = model.residuals(torch.randint(0, 11, (2, 3)))
    assert len(residuals) == 1 + 2 * 3
