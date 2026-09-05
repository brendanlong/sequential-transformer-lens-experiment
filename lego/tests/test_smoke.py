"""CI smoke test: a tiny end-to-end CPU training run.

Invokes the real CLI entrypoint with a tiny model on CPU and asserts the run
completes, loss is finite and decreases, and the saved checkpoint round-trips
back into a working model. ~15 s.
"""

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lego.training import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_lego_train_smoke(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    cmd = [
        sys.executable,
        "-m",
        "lego.train",
        "--group",
        "S3",
        "--k-min",
        "1",
        "--k-max",
        "3",
        "--dim",
        "32",
        "--n-heads",
        "2",
        "--n-layers",
        "2",
        "--generate-n",
        "1920",  # 1920 / batch 32 = 60 steps
        "--batch-size",
        "32",
        "--n-test",
        "50",
        "--lr",
        "1e-3",
        "--log-every-steps",
        "10",
        "--eval-every-steps",
        "30",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--no-wandb",
        "--no-compile",
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"training CLI failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    # Match the whole token so a diverged run logging "nan" is caught below.
    losses = [float(m) for m in re.findall(r"loss=(\S+)", result.stdout)]
    assert len(losses) >= 2, f"expected multiple logged losses, got {losses}"
    assert all(math.isfinite(x) for x in losses), f"non-finite loss: {losses}"
    assert losses[-1] < losses[0], (
        f"loss did not decrease: first={losses[0]} last={losses[-1]}"
    )

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert checkpoints, f"no checkpoint saved in {checkpoint_dir}"

    model, model_config = load_model(checkpoints[0], torch.device("cpu"))
    assert model_config.n_layers == 2
    assert model_config.dim == 32

    input_ids = torch.zeros((1, 8), dtype=torch.long)
    input_ids[0, :5] = torch.arange(1, 6)
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 8, model_config.vocab_size)
    assert torch.isfinite(logits).all()
