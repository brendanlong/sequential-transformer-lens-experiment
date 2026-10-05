#!/bin/bash
# gpuc entry point: build the env, then run the IFM part of the Snakefile.
# Extra arguments are more key=value config pairs, e.g. ifm_models=[dense-ouro-336b].
set -euo pipefail
export IFM_HOME=${IFM_HOME:-${GPUC_DATA_DIR:?}/looped-lm-ifm}
bash looped_lm/jobs/ifm_env.sh  # no-op after the setup phase built it
export PYTHONPATH="$PWD:$IFM_HOME/xllm-loop"
export PATH="$IFM_HOME/env/bin:$PATH"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0  # checkpoints and text stream from the Hub
snakemake -s looped_lm/Snakefile --cores 1 --resources gpu=1 \
    --config ifm_root="$IFM_HOME/checkpoints" ifm_python=python "$@" -- ifm_all
