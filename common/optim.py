"""Shared optimizer + LR-scheduler construction.

AdamW with linear warmup into cosine (or constant) decay via ``SequentialLR``
was copy-pasted across nine experiments, and the warmup clamp drifted into
three different formulas along the way. This is the one blessed copy; an
experiment that needs a different optimizer (SGD, Shampoo, per-param groups)
should build that optimizer itself and only reuse :func:`warmup_scheduler`.
"""

from collections.abc import Iterable
from typing import Literal

import torch


def warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    lr_schedule: Literal["cosine", "constant"] = "cosine",
    warmup_steps: int = 200,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Linear warmup into cosine (or constant) decay.

    The warmup is clamped to half the run (``min(warmup_steps, max(1,
    total_steps // 2))``) so short runs aren't all warmup.
    """
    effective_warmup = min(warmup_steps, max(1, total_steps // 2))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / max(1, effective_warmup),
        total_iters=effective_warmup,
    )
    decay: torch.optim.lr_scheduler.LRScheduler
    if lr_schedule == "cosine":
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - effective_warmup),
        )
    else:
        decay = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps,
        )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[effective_warmup],
    )


def create_optimizer_and_scheduler(
    model: torch.nn.Module | Iterable[torch.nn.Parameter],
    *,
    lr: float,
    total_steps: int,
    weight_decay: float = 0.0,
    lr_schedule: Literal["cosine", "constant"] = "cosine",
    warmup_steps: int = 200,
    betas: tuple[float, float] = (0.9, 0.999),
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.SequentialLR]:
    """AdamW with linear warmup into cosine (or constant) decay."""
    params = model.parameters() if isinstance(model, torch.nn.Module) else model
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    scheduler = warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
    )
    return optimizer, scheduler
