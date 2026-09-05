"""Standard and weight-shared (Universal Transformer) decoder-only models.

Pre-norm blocks (LayerNorm + GELU by default for these tiny models; RMSNorm +
SwiGLU available), learned or RoPE positions, tied input/output embeddings.
Both expose ``forward_with_residuals`` / ``forward_with_layer_hooks`` so the
analyses can read or overwrite the residual stream after every layer.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lego.config import ModelConfig


def _create_norm(config: ModelConfig) -> nn.Module:
    """Create normalization layer based on config."""
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.dim, eps=config.norm_eps)
    return nn.RMSNorm(config.dim, eps=config.norm_eps)


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 100000.0
) -> Tensor:
    """Precompute RoPE complex frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    # Return as complex for easy rotation
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (seq_len, head_dim // 2) complex
    """
    # Reshape x into pairs for complex multiplication
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[: x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


class Attention(nn.Module):
    """Multi-head self-attention with ``rope`` or ``learned`` (external) positions."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.pos_encoding = config.pos_encoding

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim)

        if self.pos_encoding == "rope":
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)
        # "learned": positional info already added to embeddings

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.wo(out))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network (standard for LLM-scale models)."""

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GELUFeedForward(nn.Module):
    """GELU feed-forward network (for bAbI-scale models).

    Simple two-layer MLP with GELU activation. Used instead of SwiGLU
    for small models where only LayerNorm + GELU is confirmed working.
    """

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.gelu(self.w_up(x))))


def _create_ffn(config: ModelConfig) -> nn.Module:
    """Create feed-forward network based on config."""
    if config.activation == "gelu":
        return GELUFeedForward(config.dim, config.intermediate_dim, config.dropout)
    return SwiGLUFeedForward(config.dim, config.intermediate_dim, config.dropout)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = _create_norm(config)
        self.attn = Attention(config)
        self.ffn_norm = _create_norm(config)
        self.ffn = _create_ffn(config)

    def forward(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class _BaseTransformer(nn.Module):
    """Shared base for standard and weight-shared transformer LMs.

    Subclasses must implement ``get_hidden_states`` and
    ``forward_with_layer_hooks``.
    """

    config: ModelConfig
    tok_emb: nn.Embedding
    final_norm: nn.Module
    pos_emb: nn.Embedding | None
    rope_freqs: Tensor

    def _init_weights(self, std: float) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits using tied weights."""
        return F.linear(hidden_states, self.tok_emb.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits."""
        return self.get_logits(self.get_hidden_states(input_ids))

    def forward_with_residuals(self, input_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass that also returns residual stream at each layer.

        Returns:
            logits: (batch, seq_len, vocab_size)
            residuals: list of (batch, seq_len, dim) tensors, one per layer
        """
        return self.forward_with_layer_hooks(input_ids, hook=lambda x, _i: x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _embed(self, input_ids: Tensor) -> Tensor:
        """Token + positional embedding."""
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            ).unsqueeze(0)
            x = x + self.pos_emb(positions)
        return x

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning final hidden states before vocab projection."""
        raise NotImplementedError

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass that calls hook(residual, layer_idx) after each layer.

        The hook can modify the residual stream (e.g., zero out positions)
        and the modified value is used for subsequent layers.

        Returns:
            logits: (batch, seq_len, vocab_size)
            residuals: list of post-hook (batch, seq_len, dim) tensors
        """
        raise NotImplementedError


class StandardTransformer(_BaseTransformer):
    """Standard transformer LM with distinct weights per layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = _create_norm(config)

        # Positional encoding
        if config.pos_encoding == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(
                config.max_seq_len,
                config.dim,
            )
        else:
            self.pos_emb = None

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                config.head_dim, config.max_seq_len, config.rope_theta
            ),
            persistent=False,
        )

        if config.init_std is not None:
            self._init_weights(config.init_std)

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        x = self._embed(input_ids)
        assert isinstance(self.rope_freqs, Tensor)
        for layer in self.layers:
            x = layer(x, self.rope_freqs)
        return self.final_norm(x)

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        residuals: list[Tensor] = []
        x = self._embed(input_ids)
        assert isinstance(self.rope_freqs, Tensor)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, self.rope_freqs)
            x = hook(x, layer_idx)
            residuals.append(x)
        x = self.final_norm(x)
        return self.get_logits(x), residuals


class WeightSharedTransformer(_BaseTransformer):
    """Weight-shared (Universal Transformer) LM.

    Uses a single transformer block repeated n_layers times,
    with optional learned iteration embeddings.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.block = TransformerBlock(config)
        self.final_norm = _create_norm(config)

        # Learned iteration embeddings
        self.use_iteration_embed = config.use_iteration_embed
        self.iteration_embed: nn.Embedding | None = None
        if config.use_iteration_embed:
            self.iteration_embed = nn.Embedding(config.n_layers, config.dim)

        # Positional encoding
        if config.pos_encoding == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(
                config.max_seq_len,
                config.dim,
            )
        else:
            self.pos_emb = None

        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                config.head_dim, config.max_seq_len, config.rope_theta
            ),
            persistent=False,
        )

        if config.init_std is not None:
            self._init_weights(config.init_std)

    def _apply_iteration(self, x: Tensor, rope_freqs: Tensor, iter_idx: int) -> Tensor:
        """Apply one iteration: add iteration embedding, then the shared block."""
        if self.iteration_embed is not None:
            x = x + self.iteration_embed.weight[iter_idx]
        return self.block(x, rope_freqs)

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        x = self._embed(input_ids)
        assert isinstance(self.rope_freqs, Tensor)
        for i in range(self.config.n_layers):
            x = self._apply_iteration(x, self.rope_freqs, i)
        return self.final_norm(x)

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        residuals: list[Tensor] = []
        x = self._embed(input_ids)
        assert isinstance(self.rope_freqs, Tensor)
        for i in range(self.config.n_layers):
            x = self._apply_iteration(x, self.rope_freqs, i)
            x = hook(x, i)
            residuals.append(x)
        x = self.final_norm(x)
        return self.get_logits(x), residuals


AnyModel = StandardTransformer | WeightSharedTransformer


def create_model(config: ModelConfig) -> AnyModel:
    """Create a model from config."""
    if config.weight_shared:
        return WeightSharedTransformer(config)
    return StandardTransformer(config)


def print_model_summary(model: AnyModel) -> None:
    """Print a summary of model architecture and parameter counts."""
    config = model.config

    model_type = "Weight-Shared" if config.weight_shared else "Standard"

    emb_params = model.tok_emb.weight.numel()
    if config.weight_shared:
        assert isinstance(model, WeightSharedTransformer)
        block_params = sum(p.numel() for p in model.block.parameters())
        iter_params = (
            model.iteration_embed.weight.numel()
            if model.iteration_embed is not None
            else 0
        )
    else:
        assert isinstance(model, StandardTransformer)
        block_params = sum(p.numel() for p in model.layers.parameters())
        iter_params = 0

    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    total = model.count_parameters()

    print(f"\n{'=' * 50}")
    print(f"{model_type} Transformer")
    print(f"{'=' * 50}")
    print(f"  Hidden dim:     {config.dim}")
    print(f"  Heads:          {config.n_heads}")
    print(f"  Layers/Iters:   {config.n_layers}")
    print(f"  MLP dim:        {config.intermediate_dim}")
    print(f"  Vocab size:     {config.vocab_size}")
    print(f"  Context len:    {config.max_seq_len}")
    print("\nParameters:")
    print(f"  Embeddings:     {emb_params:>12,}")
    print(f"  Blocks:         {block_params:>12,}")
    if iter_params > 0:
        print(f"  Iter embeds:    {iter_params:>12,}")
    print(f"  Final norm:     {norm_params:>12,}")
    print(f"  Total:          {total:>12,}")
    effective = config.n_layers * block_params if config.weight_shared else block_params
    effective_total = effective + emb_params + norm_params + iter_params
    print(f"  Effective:      {effective_total:>12,}")
    print(f"{'=' * 50}\n")
