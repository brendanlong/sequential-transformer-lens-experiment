#!/bin/bash
# Regenerate the writeup's tables from the hosted checkpoints
# (https://huggingface.co/datasets/brendanlong/sequential-transformer-lens-experiment).
# No GPU, no accounts; three ~3.6 MB checkpoints download on first use; ~30 s.
set -euo pipefail
cd "$(dirname "$0")/.."

A_AO="hf:lego/std_96d_6h_8L_kp2_s42_curriculum_AO/step_58593.pt"
A="hf:lego/std_96d_6h_8L_kp2_s42_curriculum_FSL/step_97656.pt"
B="hf:lego/std_96d_6h_8L_kp2_s42_fsl_only/step_156250.pt"

mkdir -p data/figures

echo "=== Model A after phase 1 (answer-only): persistent staircase ==="
uv run python -m lego.analyze_logit_lens --checkpoint "$A_AO" --output data/figures/model_a_ao.png

echo "=== Model A (AO -> FSL curriculum): the writeup's staircase table ==="
uv run python -m lego.analyze_logit_lens --checkpoint "$A" --output data/figures/model_a.png

echo "=== Model B (FSL only): no intermediates above chance ==="
uv run python -m lego.analyze_logit_lens --checkpoint "$B" --output data/figures/model_b.png

echo "=== Critical-layer ablation, Model A (critical layers read off the lens) ==="
uv run python -m lego.ablate_critical_layer --checkpoint "$A"

echo "=== Critical-layer ablation, Model B (Model A's critical layers) ==="
uv run python -m lego.ablate_critical_layer --checkpoint "$B" --critical-layers 1 2 3 4 5

echo "=== Layer sweep: every element and <op> position zeroed from each layer ==="
uv run python -m lego.ablate_critical_layer --checkpoint "$A" --sweep
uv run python -m lego.ablate_critical_layer --checkpoint "$B" --sweep

echo "=== Progressive / reverse / random clause zeroing (RESULTS.md Phase 16 table) ==="
uv run python -m lego.ablate_sequential --checkpoint "$A"
uv run python -m lego.ablate_sequential --checkpoint "$B"
