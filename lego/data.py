"""Dataset and loss utilities for the group composition task.

Each training example is a chain of group operations:
    <start> elem <op> elem … <predict> answer

The model predicts the answer (final group element) from the logits at the
<predict> position. Training samples chain lengths k from k_min to k_max,
either uniformly or with power-law weighting (weight(k) = k^k_power).
"""

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.streaming import SyntheticStream
from lego.generator import S3, Group, S3Example, generate_example
from lego.tokenizer import Tokenizer, answer_position, seq_len

_S3_TOKENIZER = Tokenizer(S3)


def encode_trajectory(
    example: S3Example,
    k_max: int,
    tokenizer: Tokenizer | None = None,
) -> list[int]:
    """Encode trajectory as element tokens, padded to k_max + 1."""
    tok = tokenizer or _S3_TOKENIZER
    tokens = [tok.element_token(t) for t in example.trajectory]
    tokens.extend([tok.pad_id] * (k_max + 1 - len(tokens)))
    return tokens


class S3FixedDataset(Dataset[dict[str, Tensor]]):
    """Fixed dataset of group composition chains, pre-encoded with padding."""

    def __init__(
        self,
        examples: list[S3Example],
        k_max: int,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.max_seq_len = seq_len(k_max)
        tok = tokenizer or _S3_TOKENIZER

        input_ids_list: list[Tensor] = []
        answer_pos_list: list[int] = []
        chain_len_list: list[int] = []
        trajectory_list: list[Tensor] = []

        for ex in examples:
            tokens = tok.encode_padded(ex, k_max)
            k = len(ex.ops)
            input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
            answer_pos_list.append(answer_position(k))
            chain_len_list.append(k)
            trajectory_list.append(
                torch.tensor(encode_trajectory(ex, k_max, tok), dtype=torch.long),
            )

        self.input_ids = torch.stack(input_ids_list)
        self.answer_positions = torch.tensor(answer_pos_list, dtype=torch.long)
        self.chain_lengths = torch.tensor(chain_len_list, dtype=torch.long)
        self.trajectories = torch.stack(trajectory_list)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "answer_position": self.answer_positions[idx],
            "chain_length": self.chain_lengths[idx],
            "trajectory": self.trajectories[idx],
        }


class S3StreamingDataset(SyntheticStream[dict[str, Tensor]]):
    """Streaming dataset generating fresh composition examples on the fly.

    Worker sharding/seeding and per-epoch seed mixing come from
    :class:`common.streaming.SyntheticStream`.
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        n_examples: int,
        seed: int = 42,
        k_power: float = 0.0,
        group: Group = S3,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(n_examples=n_examples, seed=seed)
        self.k_min = k_min
        self.k_max = k_max
        self.k_power = k_power
        self.group = group
        self.tokenizer = tokenizer or _S3_TOKENIZER

        # weight(k) = max(1, k)^power, so k=0 gets nonzero weight when k_power > 0
        self._k_values = list(range(k_min, k_max + 1))
        self._k_weights = [max(1, k) ** k_power for k in self._k_values]

    def _sample_k(self, rng: random.Random) -> int:
        # randint and choices consume the rng differently, so keep the
        # historical randint path for the unweighted case — existing runs
        # stay reproducible bit-for-bit.
        if self.k_power == 0.0:
            return rng.randint(self.k_min, self.k_max)
        return rng.choices(self._k_values, weights=self._k_weights)[0]

    def generate(self, rng: random.Random) -> dict[str, Tensor]:
        k = self._sample_k(rng)
        ex = generate_example(k, rng, self.group)
        tokens = self.tokenizer.encode_padded(ex, self.k_max)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "answer_position": torch.tensor(answer_position(k), dtype=torch.long),
            "chain_length": torch.tensor(k, dtype=torch.long),
            "trajectory": torch.tensor(
                encode_trajectory(ex, self.k_max, self.tokenizer),
                dtype=torch.long,
            ),
        }


def collate_s3(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a batch of group composition examples."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "answer_position": torch.stack([b["answer_position"] for b in batch]),
        "chain_length": torch.stack([b["chain_length"] for b in batch]),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
    }


def compute_answer_only_loss(
    logits: Tensor,
    input_ids: Tensor,
    answer_positions: Tensor,
) -> Tensor:
    """Cross-entropy at the answer position only.

    The model predicts the answer element from logits at the <predict>
    token position (= answer_position - 1).
    """
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    predict_logits = logits[batch_idx, answer_positions - 1]  # (batch, vocab)
    targets = input_ids[batch_idx, answer_positions]  # (batch,)
    return torch.nn.functional.cross_entropy(predict_logits, targets)


def compute_full_sequence_loss(
    logits: Tensor,
    input_ids: Tensor,
) -> Tensor:
    """Standard autoregressive loss on all non-pad positions.

    At each position i, the model predicts token i+1; positions whose target
    is PAD are ignored. Operand elements are uniformly random, so the optimal
    prediction there is a uniform distribution over elements.
    """
    shift_logits = logits[:, :-1].contiguous()  # (B, T-1, V)
    shift_targets = input_ids[:, 1:].contiguous()  # (B, T-1)
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=0,  # PAD_ID = 0 for all tokenizers
    )


def compute_loss(
    logits: Tensor,
    input_ids: Tensor,
    answer_positions: Tensor,
    *,
    full_sequence: bool = False,
) -> Tensor:
    """Answer-only loss by default; ``full_sequence=True`` for the LM objective."""
    if full_sequence:
        return compute_full_sequence_loss(logits, input_ids)
    return compute_answer_only_loss(logits, input_ids, answer_positions)


@torch.no_grad()
def compute_answer_accuracy(
    logits: Tensor,
    input_ids: Tensor,
    answer_positions: Tensor,
) -> Tensor:
    """Accuracy on the answer token, as a scalar tensor (no GPU→CPU sync)."""
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    predict_logits = logits[batch_idx, answer_positions - 1]
    targets = input_ids[batch_idx, answer_positions]
    predictions = predict_logits.argmax(dim=-1)
    return (predictions == targets).float().mean()


def make_eval_batch(
    examples: list[S3Example],
    k_max: int,
    tokenizer: Tokenizer | None = None,
) -> dict[str, Tensor]:
    """Create an evaluation batch, padded to k_max sequence length.

    All examples should have the same chain length k for clean per-k eval.
    """
    tok = tokenizer or _S3_TOKENIZER
    k = len(examples[0].ops)
    input_ids = torch.stack(
        [
            torch.tensor(tok.encode_padded(ex, k_max), dtype=torch.long)
            for ex in examples
        ]
    )
    return {
        "input_ids": input_ids,
        "answer_position": torch.full(
            (len(examples),), answer_position(k), dtype=torch.long
        ),
        "chain_length": torch.full((len(examples),), k, dtype=torch.long),
    }
