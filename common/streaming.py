"""Streaming synthetic dataset with correct worker/epoch seeding.

Two validity bugs recur in hand-rolled ``IterableDataset`` streams: every
DataLoader worker iterating its own copy of the same RNG (duplicated data),
and re-iterating with an epoch-independent seed (repeated data). This base
class handles both once.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")


def stream_rng(seed: int, *, epoch: int = 0, worker_id: int = 0) -> random.Random:
    """RNG seeded distinctly per (seed, epoch, worker).

    Seeding with a string avoids the additive collisions of
    ``seed + epoch + worker_id``; ``random.Random`` hashes str seeds with a
    deterministic, salt-independent scheme.
    """
    return random.Random(f"{seed}:{epoch}:{worker_id}")


def worker_share(n: int, num_workers: int, worker_id: int) -> int:
    """How many of ``n`` examples this worker should yield (shares sum to ``n``)."""
    return n // num_workers + (1 if worker_id < n % num_workers else 0)


class SyntheticStream(IterableDataset[T], ABC):
    """Base class for streaming datasets that generate examples from an RNG.

    - mixes the DataLoader worker id into the seed so workers don't duplicate
    - shards ``n_examples`` across workers so one pass yields exactly
      ``n_examples`` in total
    - mixes the epoch counter into the seed so re-iterating yields fresh data

    The automatic epoch counter only advances with ``num_workers=0`` or
    ``persistent_workers=True``; call :meth:`set_epoch` otherwise. The epoch
    advances at the *start* of every iteration, so don't use an instance as a
    repeatedly-iterated eval set expecting determinism — materialize a fixed
    eval batch instead.
    """

    def __init__(self, n_examples: int, seed: int = 42) -> None:
        self.n_examples = n_examples
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @abstractmethod
    def generate(self, rng: random.Random) -> T:
        """Generate a single example using ``rng`` as the only randomness."""

    def __len__(self) -> int:
        return self.n_examples

    def __iter__(self) -> Iterator[T]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = stream_rng(self.seed, epoch=self._epoch, worker_id=worker_id)
        self._epoch += 1
        for _ in range(worker_share(self.n_examples, num_workers, worker_id)):
            yield self.generate(rng)
