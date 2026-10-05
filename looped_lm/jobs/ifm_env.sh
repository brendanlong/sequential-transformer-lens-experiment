#!/bin/bash
# Build (once per host) the environment for looped_lm.ifm: torch 2.11 cu128,
# xLLM-Loop's Python dependencies, and its checkout at a pinned commit.
# Prints the env's root; IFM_HOME defaults to $GPUC_DATA_DIR/looped-lm-ifm.
set -euo pipefail
IFM_HOME=${IFM_HOME:-${GPUC_DATA_DIR:?}/looped-lm-ifm}
XLLM_COMMIT=c4418a1e2293bd01efe1fa815c36395311713727
mkdir -p "$IFM_HOME"
if [ ! -d "$IFM_HOME/xllm-loop" ]; then
    git clone -q https://github.com/ifm-ai/xllm-loop.git "$IFM_HOME/xllm-loop.tmp"
    git -C "$IFM_HOME/xllm-loop.tmp" checkout -q "$XLLM_COMMIT"
    mv "$IFM_HOME/xllm-loop.tmp" "$IFM_HOME/xllm-loop"
fi
if [ ! -x "$IFM_HOME/env/bin/snakemake" ]; then
    rm -rf "$IFM_HOME/env"
    uv venv -q --python 3.12 "$IFM_HOME/env"
    export VIRTUAL_ENV="$IFM_HOME/env"
    uv pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
    uv pip install -q -r "$IFM_HOME/xllm-loop/requirements.txt" transformers pyarrow \
        huggingface_hub datasets snakemake
fi
echo "$IFM_HOME" >&2
