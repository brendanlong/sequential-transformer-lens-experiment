#!/bin/bash
# Retrain the writeup's two models from scratch. Model A is 58,593 answer-only
# steps + 97,656 full-sequence steps and Model B is 156,250 full-sequence
# steps, all at batch 512. Wall-clock was not recorded for these runs;
# comparable 8-layer runs in RESULTS.md did ~16 steps/s on an RTX 3060 Ti, so
# expect a few hours each (Model A was trained on a 3060 Ti, Model B on an
# RTX 4090 — see results/13-k-weighted-curriculum.md). Checkpoints land under
# data/lego/checkpoints/<run>/.
#
# wandb logging is on only if WANDB_API_KEY is set; pass WANDB=1 to force it.
#
# The result is seed-dependent (see the writeup): seed 42 gave the clean
# staircase; seed 43 lost the staircase for the first three hops. Exact
# commands and the original results are in results/13-k-weighted-curriculum.md.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${WANDB_API_KEY:-}" ] || [ -n "${WANDB:-}" ]; then WANDB_FLAG=""; else WANDB_FLAG="--no-wandb"; fi

# --- Model A: answer-only -> full-sequence curriculum, k^2 data weighting ---
uv run python -m lego.curriculum_ao_to_fsl \
    --dim 96 --n-heads 6 --n-layers 8 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 --no-compile $WANDB_FLAG \
    --k-power 2 --seed 42 \
    --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s42_curriculum

# --- Model B: full-sequence loss from scratch, same data weighting ---
uv run python -m lego.train \
    --dim 96 --n-heads 6 --n-layers 8 \
    --loss-mode full-sequence --generate-n 80000000 \
    --batch-size 512 --lr 3e-4 --no-compile $WANDB_FLAG \
    --k-power 2 --seed 42 \
    --wandb-run-name std_96d_6h_8L_kp2_s42_fsl_only \
    --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s42_fsl_only

# --- Extra seeds / arms referenced in RESULTS.md (uncomment to run) ---
# for s in 43; do
#   uv run python -m lego.curriculum_ao_to_fsl --dim 96 --n-heads 6 --n-layers 8 \
#       --ao-examples 30000000 --fsl-examples 50000000 --batch-size 512 --lr 3e-4 \
#       --no-compile $WANDB_FLAG --k-power 2 --seed $s \
#       --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s${s}_curriculum
# done
# for s in 43 44; do
#   uv run python -m lego.train --dim 96 --n-heads 6 --n-layers 8 \
#       --loss-mode full-sequence --generate-n 80000000 --batch-size 512 --lr 3e-4 \
#       --no-compile $WANDB_FLAG --k-power 2 --seed $s \
#       --wandb-run-name std_96d_6h_8L_kp2_s${s}_fsl_only \
#       --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s${s}_fsl_only
# done
#
# --- Weight-shared (universal) transformer, uniform data (RESULTS.md Phase 14) ---
# uv run python -m lego.curriculum_ao_to_fsl --dim 96 --n-heads 6 --n-layers 8 \
#     --weight-shared --ao-examples 30000000 --fsl-examples 50000000 \
#     --batch-size 512 --lr 3e-4 --no-compile $WANDB_FLAG --seed 42 \
#     --checkpoint-dir data/lego/checkpoints/ws_96d_6h_8iter_uniform_s42_curriculum
#
# --- Standard transformer, uniform data (Phase 14: staircase compressed into L5-L7) ---
# uv run python -m lego.curriculum_ao_to_fsl --dim 96 --n-heads 6 --n-layers 8 \
#     --ao-examples 30000000 --fsl-examples 50000000 \
#     --batch-size 512 --lr 3e-4 --no-compile $WANDB_FLAG --seed 42 \
#     --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_uniform_s42_curriculum
