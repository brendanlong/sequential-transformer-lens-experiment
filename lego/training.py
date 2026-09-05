"""Training loop and checkpoint helpers for LEGO models.

Used by ``train.py`` (single-phase) and ``curriculum_ao_to_fsl.py``
(answer-only → full-sequence curriculum).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from common.checkpoint import load_model_checkpoint
from common.checkpoint import save_model_checkpoint as _save_checkpoint
from common.optim import create_optimizer_and_scheduler
from common.schedule import should_log_and_eval
from common.wandb_utils import finish_wandb, init_wandb, log_metrics
from lego.config import ModelConfig
from lego.data import compute_answer_accuracy, compute_loss, make_eval_batch
from lego.generator import S3Example
from lego.model import AnyModel, create_model, print_model_summary
from lego.tokenizer import Tokenizer


@dataclass
class TrainingResult:
    """Result from train_lego_model."""

    converged_step: int | None
    final_eval: dict[str, float]
    checkpoint_path: Path | None
    total_steps: int


def should_early_stop(
    global_step: int,
    converged_step: int | None,
    patience: int | None,
) -> bool:
    """Whether to stop ``patience`` steps after convergence (``None`` = never)."""
    if patience is None or converged_step is None:
        return False
    return global_step - converged_step >= patience


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_examples_per_k: dict[int, list[S3Example]],
    k_max: int,
    batch_size: int,
    device: torch.device,
    tokenizer: Tokenizer | None = None,
) -> dict[str, float]:
    """Evaluate per-chain-length accuracy."""
    model.eval()
    results: dict[str, float] = {}
    total_correct = torch.tensor(0.0, device=device)
    total_count = 0
    for k in sorted(test_examples_per_k):
        examples = test_examples_per_k[k]
        correct = torch.tensor(0.0, device=device)
        count = 0
        for i in range(0, len(examples), batch_size):
            chunk = examples[i : i + batch_size]
            batch = make_eval_batch(chunk, k_max, tokenizer)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            answer_positions = batch["answer_position"].to(device, non_blocking=True)
            logits = model(input_ids)
            acc = compute_answer_accuracy(logits, input_ids, answer_positions)
            correct += acc * len(chunk)
            count += len(chunk)
        results[f"test_acc/k_{k}"] = (correct / max(1, count)).item()
        total_correct += correct
        total_count += count
    results["test_acc/mean"] = (total_correct / max(1, total_count)).item()
    return results


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[AnyModel, ModelConfig]:
    """Load model and config from a local checkpoint or an ``hf:<relpath>``."""
    return load_model_checkpoint(checkpoint_path, ModelConfig, create_model, device)


def save_model_checkpoint(
    model: torch.nn.Module,
    step: int,
    model_config: ModelConfig,
    checkpoint_dir: Path,
) -> Path:
    """Save checkpoint, stripping torch.compile prefix from keys."""
    return _save_checkpoint(model, step, model_config, checkpoint_dir)


def train_lego_model(
    model: AnyModel,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    test_examples_per_k: dict[int, list[S3Example]],
    model_config: ModelConfig,
    device: torch.device,
    *,
    total_steps: int,
    k_min: int = 1,
    k_max: int = 6,
    n_epochs: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    lr_schedule: Literal["cosine", "constant"] = "cosine",
    use_compile: bool = True,
    full_sequence_loss: bool = False,
    early_stop_patience: int | None = 2000,
    log_every_steps: int = 100,
    eval_every_steps: int = 500,
    eval_batch_size: int = 512,
    run_name: str = "",
    checkpoint_dir: Path | None = None,
    save_every_steps: int | None = None,
    use_wandb: bool = True,
    wandb_project: str = "lego-reasoning",
    wandb_config: dict[str, object] | None = None,
    tokenizer: Tokenizer | None = None,
) -> TrainingResult:
    """Unified training loop for LEGO models.

    Trains with periodic evaluation, optional early stopping, wandb logging,
    and checkpoint saving. Convergence is detected when all chain lengths
    reach >= 99.9% accuracy.
    """
    print_model_summary(model)

    if use_compile and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=lr,
        total_steps=total_steps,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
    )

    init_wandb(
        enabled=use_wandb,
        project=wandb_project,
        run_name=run_name,
        config=wandb_config or {},
    )

    global_step = 0
    running_loss = torch.tensor(0.0, device=device)
    running_acc = torch.tensor(0.0, device=device)
    running_count = 0
    t0 = time.time()
    converged_step: int | None = None
    last_saved_step = -1
    last_checkpoint_path: Path | None = None
    stopped_early = False

    for _epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            answer_positions = batch["answer_position"].to(device, non_blocking=True)

            logits = model(input_ids)
            loss = compute_loss(
                logits,
                input_ids,
                answer_positions,
                full_sequence=full_sequence_loss,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += loss.detach()
            running_acc += compute_answer_accuracy(logits, input_ids, answer_positions)
            running_count += 1
            global_step += 1

            if should_early_stop(global_step, converged_step, early_stop_patience):
                print(
                    f"  [{run_name}] Early stop at step {global_step} "
                    f"({early_stop_patience} steps after convergence)",
                )
                stopped_early = True
                break

            # Logging also fires on eval steps so evals are never skipped
            # when eval_every_steps isn't a multiple of log_every_steps.
            do_log, do_eval = should_log_and_eval(
                global_step,
                log_every_steps=log_every_steps,
                eval_every_steps=eval_every_steps,
            )
            if do_log:
                avg_loss = (running_loss / running_count).item()
                avg_acc = (running_acc / running_count).item()
                elapsed = time.time() - t0
                steps_per_sec = running_count / elapsed

                log_dict: dict[str, float] = {
                    "train/loss": avg_loss,
                    "train/answer_acc": avg_acc,
                    "perf/steps_per_sec": steps_per_sec,
                    "lr": optimizer.param_groups[0]["lr"],
                }

                if do_eval:
                    eval_results = evaluate(
                        model,
                        test_examples_per_k,
                        k_max,
                        eval_batch_size,
                        device,
                        tokenizer,
                    )
                    log_dict.update(eval_results)
                    model.train()

                    mean_acc = eval_results["test_acc/mean"]
                    k_accs = [
                        eval_results.get(f"test_acc/k_{k}", 0.0)
                        for k in range(k_min, k_max + 1)
                    ]
                    k_str = " ".join(f"{a:.0%}" for a in k_accs)
                    print(
                        f"  [{run_name}] Step {global_step:6d}/{total_steps}"
                        f" | loss={avg_loss:.4f} | train={avg_acc:.1%}"
                        f" | test={mean_acc:.1%} | k=[{k_str}]"
                        f" | {steps_per_sec:.1f} steps/s",
                        flush=True,
                    )

                    if converged_step is None and all(a >= 0.999 for a in k_accs):
                        converged_step = global_step
                        print(
                            f"  [{run_name}] *** CONVERGED at step {global_step} ***",
                        )
                        remaining = total_steps - global_step
                        if early_stop_patience is None and remaining > 0:
                            hours = remaining / max(steps_per_sec, 1e-9) / 3600
                            print(
                                f"  [{run_name}] {remaining} steps (~{hours:.1f} h) "
                                "remain after convergence and --early-stop-patience "
                                "is unset, so training continues.",
                                flush=True,
                            )
                        if (
                            checkpoint_dir is not None
                            and last_saved_step != global_step
                        ):
                            last_checkpoint_path = save_model_checkpoint(
                                model, global_step, model_config, checkpoint_dir
                            )
                            last_saved_step = global_step
                else:
                    print(
                        f"  [{run_name}] Step {global_step:6d}/{total_steps}"
                        f" | loss={avg_loss:.4f} | train={avg_acc:.1%}"
                        f" | {steps_per_sec:.1f} steps/s",
                        flush=True,
                    )

                log_metrics(log_dict, global_step, enabled=use_wandb)

                running_loss = torch.tensor(0.0, device=device)
                running_acc = torch.tensor(0.0, device=device)
                running_count = 0
                t0 = time.time()

            if (
                save_every_steps is not None
                and checkpoint_dir is not None
                and global_step % save_every_steps == 0
                and last_saved_step != global_step
            ):
                last_checkpoint_path = save_model_checkpoint(
                    model, global_step, model_config, checkpoint_dir
                )
                last_saved_step = global_step

        if stopped_early:
            break

    if checkpoint_dir is not None and last_saved_step != global_step:
        last_checkpoint_path = save_model_checkpoint(
            model, global_step, model_config, checkpoint_dir
        )
        if converged_step is None:
            print(f"  [{run_name}] WARNING: Did not converge!")

    final_eval = evaluate(
        model, test_examples_per_k, k_max, eval_batch_size, device, tokenizer
    )
    log_metrics(final_eval, global_step, enabled=use_wandb)
    finish_wandb(enabled=use_wandb)

    return TrainingResult(
        converged_step=converged_step,
        final_eval=final_eval,
        checkpoint_path=last_checkpoint_path,
        total_steps=global_step,
    )
