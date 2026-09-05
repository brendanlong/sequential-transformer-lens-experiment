"""Checkpoint save/load and public-artifact resolution.

Training saves checkpoints locally. The checkpoints behind the writeup are
hosted publicly on the Hugging Face Hub
(https://huggingface.co/datasets/brendanlong/sequential-transformer-lens-experiment)
and download on demand into the local HF cache; pass ``hf:<relpath>`` wherever
a script takes a checkpoint path.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

HF_DATASET = "brendanlong/sequential-transformer-lens-experiment"


def save_model_checkpoint(
    model: torch.nn.Module,
    step: int,
    model_config: dict[str, object] | BaseModel,
    checkpoint_dir: Path,
    filename: str | None = None,
) -> Path:
    """Save a checkpoint, stripping any torch.compile ``_orig_mod.`` prefix.

    The payload is ``{"step", "model_state_dict", "model_config"}`` so loaders
    can reconstruct the model from its config dict. Loadable with
    ``torch.load(..., weights_only=True)`` / :func:`load_model_checkpoint`.
    """
    if isinstance(model_config, BaseModel):
        model_config = model_config.model_dump()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
    }
    ckpt_path = checkpoint_dir / (filename or f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "model_config": model_config,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")
    return ckpt_path


def load_model_checkpoint[C: BaseModel, M: torch.nn.Module](
    checkpoint_path: str | Path,
    config_cls: type[C],
    factory: Callable[[C], M],
    device: torch.device,
) -> tuple[M, C]:
    """Load a :func:`save_model_checkpoint` checkpoint back into a model.

    Accepts a local path or an ``hf:<relpath>`` reference into the public
    dataset (see :func:`resolve_checkpoint`).
    """
    ckpt = torch.load(
        resolve_checkpoint(str(checkpoint_path)),
        weights_only=True,
        map_location=device,
    )
    config = config_cls(**ckpt["model_config"])
    model = factory(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


def artifact_path(relpath: str) -> Path:
    """Resolve a published artifact, preferring a local mirror over the HF dataset.

    ``relpath`` is e.g. ``lego/<run_name>/step_97656.pt``. If
    ``<repo>/data/artifacts/<relpath>`` exists (the escape hatch for analyzing
    fresh local runs before they are uploaded) it is used directly — announced
    on stdout since it shadows the published bytes; otherwise the file is
    downloaded from the public HF dataset and cached by huggingface_hub.
    """
    local = Path(__file__).resolve().parents[1] / "data" / "artifacts" / relpath
    if local.exists():
        print(f"[artifact_path] using local mirror: {local}")
        return local
    return Path(
        hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=relpath)
    )


def resolve_checkpoint(source: str) -> Path:
    """Resolve a checkpoint source to a local path.

    - A local path is returned as-is.
    - ``hf:<relpath>`` downloads from the public HF dataset.
    """
    if source.startswith("hf:"):
        return artifact_path(source.removeprefix("hf:"))
    return Path(source)
