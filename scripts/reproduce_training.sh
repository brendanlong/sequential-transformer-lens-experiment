#!/bin/bash
# Retrain the writeup's two models from scratch. Each is a few hours on a
# consumer GPU (Model A: 58,593 AO steps + 97,656 FSL steps; Model B: 156,250
# FSL steps; ~16-20 steps/s on an RTX 3060 Ti at batch 512). Runs log to
# wandb by default (pass --no-wandb to disable). Checkpoints land under
# data/lego/checkpoints/<run>/.
#
# The result is seed-dependent (see the writeup): seed 42 gave the clean
# staircase; seed 43 gave a noisier one. Exact commands and the original
# results are in RESULTS.md / results/13-k-weighted-curriculum.md.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- Model A: answer-only -> full-sequence curriculum, k^2 data weighting ---
uv run python -m lego.curriculum_ao_to_fsl \
    --dim 96 --n-heads 6 --n-layers 8 \
    --ao-examples 30000000 --fsl-examples 50000000 \
    --batch-size 512 --lr 3e-4 --no-compile \
    --k-power 2 --seed 42 \
    --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s42_curriculum

# --- Model B: full-sequence loss from scratch, same data weighting ---
uv run python -m lego.train \
    --dim 96 --n-heads 6 --n-layers 8 \
    --loss-mode full-sequence --generate-n 80000000 \
    --batch-size 512 --lr 3e-4 --no-compile \
    --k-power 2 --seed 42 \
    --wandb-run-name std_96d_6h_8L_kp2_s42_fsl_only \
    --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s42_fsl_only

# --- Extra seeds / arms referenced in RESULTS.md (uncomment to run) ---
# for s in 43; do
#   uv run python -m lego.curriculum_ao_to_fsl --dim 96 --n-heads 6 --n-layers 8 \
#       --ao-examples 30000000 --fsl-examples 50000000 --batch-size 512 --lr 3e-4 \
#       --no-compile --k-power 2 --seed $s \
#       --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s${s}_curriculum
# done
# for s in 43 44; do
#   uv run python -m lego.train --dim 96 --n-heads 6 --n-layers 8 \
#       --loss-mode full-sequence --generate-n 80000000 --batch-size 512 --lr 3e-4 \
#       --no-compile --k-power 2 --seed $s \
#       --wandb-run-name std_96d_6h_8L_kp2_s${s}_fsl_only \
#       --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_kp2_s${s}_fsl_only
# done
#
# --- Weight-shared (universal) transformer, uniform data (RESULTS.md Phase 14) ---
# uv run python -m lego.curriculum_ao_to_fsl --dim 96 --n-heads 6 --n-layers 8 \
#     --weight-shared --ao-examples 30000000 --fsl-examples 50000000 \
#     --batch-size 512 --lr 3e-4 --no-compile --seed 42 \
#     --checkpoint-dir data/lego/checkpoints/ws_96d_6h_8iter_uniform_s42_curriculum
#
# --- Standard transformer, uniform data (Phase 14: staircase compressed into L5-L7) ---
# uv run python -m lego.curriculum_ao_to_fsl --dim 96 --n-heads 6 --n-layers 8 \
#     --ao-examples 30000000 --fsl-examples 50000000 \
#     --batch-size 512 --lr 3e-4 --no-compile --seed 42 \
#     --checkpoint-dir data/lego/checkpoints/std_96d_6h_8L_uniform_s42_curriculum
