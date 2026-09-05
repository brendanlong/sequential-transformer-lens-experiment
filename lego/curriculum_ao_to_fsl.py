"""Curriculum training: answer-only loss, then full-sequence loss.

Phase 1 trains under answer-only (AO) loss until the schedule ends (the model
typically converges to 100% within the first ~10k steps); phase 2 continues
from those weights under full-sequence loss (FSL) with a fresh optimizer.
The logit-lens staircase is printed after each phase.

This is the recipe behind the writeup's Model A:

    uv run python -m lego.curriculum_ao_to_fsl \
        --dim 96 --n-heads 6 --n-layers 8 \
        --ao-examples 30000000 --fsl-examples 50000000 \
        --batch-size 512 --lr 3e-4 --no-compile \
        --k-power 2 --seed 42

Checkpoints land in ``--checkpoint-dir`` as ``curriculum_ao.pt`` and
``curriculum_fsl.pt``; the training loop's periodic ``step_N.pt`` files go to
``<checkpoint-dir>/ao/`` and ``<checkpoint-dir>/fsl/`` so phase 2 never
overwrites phase 1's.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lego.analyze_logit_lens import make_logit_lens_fn, probe_op_positions
from lego.config import lego_model_config
from lego.data import S3StreamingDataset, collate_s3
from lego.generator import generate_fixed_dataset
from lego.model import AnyModel, create_model
from lego.tokenizer import answer_position, encode
from lego.training import train_lego_model


def run_logit_lens(
    model: AnyModel,
    device: torch.device,
    k_values: list[int],
    n_examples: int = 500,
    label: str = "",
) -> None:
    """Print the <op>-position logit-lens staircase for each k."""
    model.eval()
    ll_fn = make_logit_lens_fn(model)

    for k_test in k_values:
        examples = generate_fixed_dataset(k_test, n_examples, seed=999)
        input_ids = torch.tensor(
            [encode(ex) for ex in examples], dtype=torch.long, device=device
        )

        with torch.no_grad():
            logits, residuals = model.forward_with_residuals(input_ids)

        ans_pos = answer_position(k_test)
        preds = logits[:, ans_pos - 1, :].argmax(dim=-1)
        targets = input_ids[:, ans_pos]
        acc = (preds == targets).float().mean().item()

        heatmap = probe_op_positions(residuals, examples, ll_fn, k_test, device)
        n_layers = len(residuals)

        print(f"\n{label} k={k_test} (acc={acc:.0%}):")
        col_labels = [f"t[{j + 1}]" for j in range(k_test)]
        header = "  ".join(f"{c:>6s}" for c in col_labels)
        print(f"  Layer  {header}")
        for i in range(n_layers):
            vals = "  ".join(f"{heatmap[i, j]:>6.0%}" for j in range(k_test))
            print(f"  L{i:<3d}  {vals}")


def _unwrap(model: AnyModel) -> AnyModel:
    """Return the underlying module if ``model`` was wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum: AO → FSL transition experiment",
    )
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--weight-shared", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Defaults are the writeup's budgets (the research code defaulted to 5M AO).
    parser.add_argument("--ao-examples", type=int, default=30_000_000)
    parser.add_argument("--fsl-examples", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lego-reasoning")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--checkpoint-dir", default="data/lego/checkpoints")
    parser.add_argument(
        "--k-power",
        type=float,
        default=0.0,
        help="Power for k-weighting: weight(k)=k^power. "
        "0=uniform (default), 2=quadratic (k=6 gets 36x k=1)",
    )
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    k_min, k_max = args.k_min, args.k_max

    model_config = lego_model_config(
        weight_shared=args.weight_shared,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )

    test_examples_per_k = {
        k: generate_fixed_dataset(k, 1000, seed=args.seed + 1 + k)
        for k in range(k_min, k_max + 1)
    }

    ws_tag = "ws" if args.weight_shared else "std"
    arch = f"{ws_tag}_{args.dim}d_{args.n_heads}h_{args.n_layers}L"

    model = create_model(model_config).to(device)

    if args.k_power != 0.0:
        print(f"  k-weighting: weight(k) = k^{args.k_power}")
        weights = [max(1, k) ** args.k_power for k in range(k_min, k_max + 1)]
        total = sum(weights)
        for k, w in zip(range(k_min, k_max + 1), weights, strict=True):
            print(f"    k={k}: {w / total:.1%}")

    common_wandb_config: dict[str, object] = {
        "model": model_config.model_dump(),
        "ao_examples": args.ao_examples,
        "fsl_examples": args.fsl_examples,
        "experiment": "curriculum_ao_to_fsl",
        "k_power": args.k_power,
        "seed": args.seed,
    }

    # === Phase 1: answer-only ===
    print("=" * 60)
    print(f"PHASE 1: Answer-Only training ({args.ao_examples:,} examples)")
    print("=" * 60)

    ao_dataset = S3StreamingDataset(
        k_min, k_max, args.ao_examples, seed=args.seed + 100, k_power=args.k_power
    )
    ao_loader = DataLoader(
        ao_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_s3,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    ao_result = train_lego_model(
        model,
        ao_loader,
        test_examples_per_k,
        model_config,
        device,
        total_steps=args.ao_examples // args.batch_size,
        k_min=k_min,
        k_max=k_max,
        lr=args.lr,
        lr_schedule="cosine",
        use_compile=not args.no_compile,
        full_sequence_loss=False,
        early_stop_patience=None,
        log_every_steps=1000,
        eval_every_steps=1000,
        eval_batch_size=args.batch_size,
        run_name=f"{arch}_curriculum_AO",
        checkpoint_dir=Path(args.checkpoint_dir) / "ao",
        save_every_steps=5000,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_config={**common_wandb_config, "phase": "AO"},
    )

    raw_model = _unwrap(model)

    print("\n=== Post-AO Logit Lens ===")
    run_logit_lens(raw_model, device, [2, 4, 6], label="[AO]")

    ao_ckpt_path = Path(args.checkpoint_dir) / "curriculum_ao.pt"
    torch.save(
        {
            "step": ao_result.total_steps,
            "model_state_dict": raw_model.state_dict(),
            "model_config": model_config.model_dump(),
        },
        ao_ckpt_path,
    )
    print(f"\nSaved AO checkpoint: {ao_ckpt_path}")

    print("\n=== AO Final Evaluation ===")
    for k in range(k_min, k_max + 1):
        print(f"  k={k}: {ao_result.final_eval.get(f'test_acc/k_{k}', 0.0):.1%}")

    # === Phase 2: full-sequence, same weights, fresh optimizer ===
    print("\n" + "=" * 60)
    print(f"PHASE 2: Full-Sequence Loss fine-tuning ({args.fsl_examples:,} examples)")
    print("=" * 60)

    fsl_dataset = S3StreamingDataset(
        k_min, k_max, args.fsl_examples, seed=args.seed + 200, k_power=args.k_power
    )
    fsl_loader = DataLoader(
        fsl_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_s3,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    fsl_result = train_lego_model(
        model,
        fsl_loader,
        test_examples_per_k,
        model_config,
        device,
        total_steps=args.fsl_examples // args.batch_size,
        k_min=k_min,
        k_max=k_max,
        lr=args.lr,
        lr_schedule="cosine",
        use_compile=not args.no_compile,
        full_sequence_loss=True,
        early_stop_patience=None,
        log_every_steps=1000,
        eval_every_steps=1000,
        eval_batch_size=args.batch_size,
        run_name=f"{arch}_curriculum_FSL",
        checkpoint_dir=Path(args.checkpoint_dir) / "fsl",
        save_every_steps=5000,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_config={**common_wandb_config, "phase": "FSL"},
    )

    raw_model = _unwrap(model)

    print("\n=== Post-FSL Logit Lens ===")
    run_logit_lens(raw_model, device, [2, 4, 6], label="[FSL]")

    fsl_ckpt_path = Path(args.checkpoint_dir) / "curriculum_fsl.pt"
    torch.save(
        {
            "step": fsl_result.total_steps,
            "model_state_dict": raw_model.state_dict(),
            "model_config": model_config.model_dump(),
        },
        fsl_ckpt_path,
    )
    print(f"\nSaved FSL checkpoint: {fsl_ckpt_path}")

    print("\n=== FSL Final Evaluation ===")
    for k in range(k_min, k_max + 1):
        print(f"  k={k}: {fsl_result.final_eval.get(f'test_acc/k_{k}', 0.0):.1%}")


if __name__ == "__main__":
    main()
