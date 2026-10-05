"""Model-agnostic measures of how much of the residual stream the logit lens sees.

Every function takes the *normed* residual ``z`` (final norm applied without its
gain) and a readout ``W`` (vocab, dim) with the final norm's gain folded in, so
the lens logits are ``z @ W.T`` up to a per-token constant that softmax and
argmax ignore.

- **Visible / dark subspace.** ``W``'s right singular vectors with the largest
  singular values are the directions the lens reads most strongly; the bottom
  ones barely move any logit. ``energy_by_singular_band`` reports what share of
  ``z``'s energy falls in each band, against the isotropic share.
- **Sparse reconstruction** (``omp_r2``): variance of ``z`` explained by the
  best k token directions, against a Gaussian null with the same covariance.
  One token → high R²(1); "considering n tokens" → a rise to k ≈ n then a
  plateau; no token content → a curve that tracks the null.
- **Logit outliers** (``outlier_counts``): lens logits more than ``c`` robust
  standard deviations above the median, counting how many tokens stand out
  from the bulk regardless of the logit scale.
- **Probes** (``train_probe``): the reference for "linearly present".
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def folded_readout(unembed: Tensor, gain: Tensor | None, centers: bool) -> Tensor:
    """(vocab, dim) readout with the norm gain folded in and shared shifts removed.

    Subtracting the mean row drops the direction that moves every logit equally
    (softmax ignores it). For LayerNorm (``centers``), ``z`` has no component
    along the all-ones vector, so that direction is removed from the rows too.
    """
    w = unembed.float() * (1 if gain is None else gain.float())
    w = w - w.mean(0)
    if centers:
        ones = torch.full((w.shape[1],), w.shape[1] ** -0.5, device=w.device)
        w = w - (w @ ones)[:, None] * ones
    return w


def singular_basis(readout: Tensor, chunk: int = 65536) -> tuple[Tensor, Tensor]:
    """Singular values (descending) and the matching right singular vectors as rows.

    Computed from the (dim, dim) Gram matrix in float64, so a 250k-row readout
    never needs a full SVD.
    """
    gram = sum(
        (c.double().T @ c.double() for c in readout.split(chunk)),
        torch.zeros(
            readout.shape[1],
            readout.shape[1],
            dtype=torch.float64,
            device=readout.device,
        ),
    )
    evals, evecs = torch.linalg.eigh(gram)
    s = evals.clamp_min(0).sqrt().flip(0).float()
    return s, evecs.flip(1).T.float()


def energy_by_singular_band(z: Tensor, vh: Tensor, n_bands: int = 4) -> Tensor:
    """Share of Σ‖z‖² in each band of singular directions, top band first.

    Bands split ``vh``'s rows into equal-sized groups by singular value; an
    isotropic ``z`` puts 1/n_bands of its energy in each (directions outside
    the readout's row space, e.g. all-ones for LayerNorm, count in none).
    """
    coords = z.float() @ vh.T
    per_dir = coords.pow(2).sum(0)
    total = z.float().pow(2).sum()
    return torch.stack([b.sum() / total for b in per_dir.chunk(n_bands)])


def relative_visibility(z: Tensor, s: Tensor, vh: Tensor) -> float:
    """‖W z‖²/‖z‖² over its expectation for isotropic z (‖W‖²_F / dim).

    ``s``, ``vh`` are W's singular values and right singular vectors. 1 means
    the lens stretches ``z`` as much as a random direction; above 1, ``z`` sits
    in strongly read directions; near 0, in dark ones.
    """
    z = z.float()
    stretch = ((z @ vh.T).pow(2) * s.pow(2)).sum() / z.pow(2).sum()
    return (stretch / (s.pow(2).sum() / vh.shape[1])).item()


def gaussian_null(z: Tensor, generator: torch.Generator) -> Tensor:
    """Gaussian samples with z's covariance: random mixtures of the centered rows."""
    n = z.shape[0]
    mix = torch.randn(n, n, generator=generator).to(z.device) / math.sqrt(n)
    return mix @ (z - z.mean(0))


@torch.no_grad()
def omp_r2(
    y: Tensor,
    atoms: Tensor,
    k_max: int,
    chunk: int = 256,
) -> tuple[Tensor, Tensor]:
    """Orthogonal matching pursuit of each row of ``y`` with unit-norm ``atoms``.

    Atoms are picked by largest *positive* correlation with the remaining
    residual (a token direction, not its negation), then all chosen
    coefficients are refit by least squares. Returns the pooled variance
    explained after 1..k_max atoms, and the (rows, k_max) chosen atom ids.
    ``y`` should already be centered (R² of variance).
    """
    y = y.float()
    explained = torch.zeros(k_max, device=y.device)
    chosen = torch.zeros(y.shape[0], k_max, dtype=torch.long, device=y.device)
    eye = torch.eye(k_max, device=y.device) * 1e-6
    for start in range(0, y.shape[0], chunk):
        target = y[start : start + chunk]
        resid = target
        for k in range(k_max):
            chosen[start : start + chunk, k] = (resid @ atoms.T).argmax(-1)
            a = atoms[chosen[start : start + chunk, : k + 1]]  # (b, k+1, dim)
            gram = a @ a.transpose(1, 2) + eye[: k + 1, : k + 1]
            coef = torch.linalg.solve(gram, a @ target[:, :, None])
            resid = target - (coef.transpose(1, 2) @ a).squeeze(1)
            explained[k] += resid.pow(2).sum()
    total = y.pow(2).sum()
    return 1 - explained / total, chosen


@torch.no_grad()
def outlier_counts(logits: Tensor, c: float = 6.0) -> Tensor:
    """Per row, how many logits exceed median + c · 1.4826 · MAD."""
    logits = logits.float()
    med = logits.median(-1, keepdim=True).values
    mad = (logits - med).abs().median(-1, keepdim=True).values * 1.4826
    return (logits > med + c * mad.clamp_min(1e-6)).sum(-1)


def train_probe(
    x_train: Tensor,
    y_train: Tensor,
    n_classes: int,
    readout: Tensor | None = None,
    steps: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> torch.nn.Module:
    """Full-batch Adam on a linear probe; returns the trained module.

    With ``readout`` (n_classes, dim), the probe is an affine map *into the
    lens's own readout* — logits = readout (A z + b), A initialised to the
    identity — i.e. a tuned lens trained on this target. Without it, an
    unconstrained (n_classes, dim) multinomial probe.
    """
    dim = x_train.shape[1]
    x_train = x_train.float()
    probe: torch.nn.Module
    if readout is None:
        probe = torch.nn.Linear(dim, n_classes).to(x_train.device)
    else:
        probe = _ReadoutProbe(readout.float()).to(x_train.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe(x_train), y_train).backward()
        opt.step()
    return probe.eval()


class _ReadoutProbe(torch.nn.Module):
    def __init__(self, readout: Tensor) -> None:
        super().__init__()
        dim = readout.shape[1]
        self.register_buffer("readout", readout)
        self.translate = torch.nn.Linear(dim, dim)
        torch.nn.init.eye_(self.translate.weight)
        torch.nn.init.zeros_(self.translate.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.translate(x) @ self.readout.T


@torch.no_grad()
def probe_accuracy(probe: torch.nn.Module, x: Tensor, y: Tensor) -> float:
    return (probe(x.float()).argmax(-1) == y).float().mean().item()
