"""Upload a local checkpoint tree to the HF dataset (maintainer utility).

Expects a directory shaped like ``lego/<run_name>/step_N.pt`` (+ optional
``.json`` sidecars), i.e. the layout of ``data/artifacts/``.

Usage: HF_TOKEN=... uv run python scripts/upload_artifacts.py data/artifacts
"""

import os
import sys

from huggingface_hub import HfApi

from common.checkpoint import HF_DATASET

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id=HF_DATASET,
    repo_type="dataset",
    folder_path=sys.argv[1],
    commit_message="Update checkpoints",
)
