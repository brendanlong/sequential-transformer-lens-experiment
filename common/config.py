"""Base training configuration."""

from typing import Literal

from pydantic import BaseModel


class BaseTrainingConfig(BaseModel):
    """Common training hyperparameters; experiments subclass and override defaults."""

    # Optimization
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = 100
    total_steps: int = 2000
    max_grad_norm: float = 1.0

    # Logging / eval cadence (see common.schedule for drift-free firing)
    log_every_steps: int = 100
    eval_every_steps: int = 500

    # Checkpointing
    checkpoint_dir: str = "data/checkpoints"

    # Wandb
    wandb_project: str = "experiments"
    wandb_run_name: str | None = None
    use_wandb: bool = True

    seed: int = 42
