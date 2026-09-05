"""Configuration for model architecture and LEGO training."""

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

from common.config import BaseTrainingConfig


class ModelConfig(BaseModel):
    """Model architecture configuration.

    No defaults — always construct via :func:`lego_model_config` or deserialize
    from a checkpoint dict. Checkpoints written by the original research code
    may carry extra keys (e.g. ``use_input_injection``); pydantic ignores them.
    """

    dim: int
    n_heads: int
    n_layers: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 100000.0
    norm_eps: float = 1e-5
    weight_shared: bool
    use_iteration_embed: bool
    dropout: float
    pos_encoding: Literal["rope", "learned"]
    norm_type: Literal["rmsnorm", "layernorm"]
    activation: Literal["swiglu", "gelu"]
    # None = PyTorch default (Kaiming) init.
    init_std: float | None

    @model_validator(mode="after")
    def _validate_architecture(self) -> "ModelConfig":
        if self.dim % self.n_heads != 0:
            msg = f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        if self.pos_encoding == "rope" and (self.dim // self.n_heads) % 2 != 0:
            msg = f"head_dim ({self.dim // self.n_heads}) must be even for rope"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def lego_model_config(
    weight_shared: bool = False,
    dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 8,
    pos_encoding: Literal["rope", "learned"] = "learned",
    norm_type: Literal["rmsnorm", "layernorm"] = "layernorm",
    activation: Literal["swiglu", "gelu"] = "gelu",
    init_std: float | None = None,
    dropout: float = 0.0,
    use_iteration_embed: bool = True,
    vocab_size: int | None = None,
) -> ModelConfig:
    """Create ModelConfig for LEGO experiments.

    Args:
        vocab_size: Override vocabulary size. If None, uses S3 default (10).
    """
    if vocab_size is None:
        from lego.tokenizer import VOCAB_SIZE

        vocab_size = VOCAB_SIZE

    return ModelConfig(
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        intermediate_dim=dim * 4,
        vocab_size=vocab_size,
        max_seq_len=128,  # plenty of room for any k_max
        weight_shared=weight_shared,
        use_iteration_embed=use_iteration_embed and weight_shared,
        dropout=dropout,
        pos_encoding=pos_encoding,
        norm_type=norm_type,
        activation=activation,
        init_std=init_std,
    )


class LegoTrainingConfig(BaseTrainingConfig):
    """Training hyperparameters for LEGO S3 experiments.

    ``total_steps``/``warmup_steps``/``max_grad_norm`` from the base are
    unused: lego derives steps from epochs × dataset size and uses the
    scheduler's warmup default.
    """

    # Overridden shared defaults
    batch_size: int = 512
    eval_every_steps: int = 1000
    checkpoint_dir: str = "data/lego/checkpoints"
    wandb_project: str = "lego-reasoning"

    # Data -- chain length range
    k_min: int = 0
    k_max: int = 6
    n_test: int = 1_000  # test examples per chain length k
    # weight(k) = max(1, k)^k_power; 0 = uniform, 2 = the writeup's k² weighting
    k_power: float = 0.0

    # Streaming mode
    generate_n: int | None = None

    # Fixed dataset mode (ignored when generate_n is set)
    n_train: int = 100_000

    # Training
    loss_mode: Literal["answer-only", "full-sequence"] = "answer-only"
    n_epochs: int = 200

    # Stop this many steps after convergence (all k >= 99.9%). None = train
    # the full schedule; several results depend on the post-convergence tail.
    early_stop_patience: int | None = None

    # Step-based checkpointing cadence
    save_every_steps: int = 5000
