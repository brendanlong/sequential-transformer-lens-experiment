#!/bin/bash
# MLP-neuron analysis on the main Kohli two-hop checkpoints. KOHLI_ROOT holds
# their data and checkpoints (looped_lm.download_kohli).
set -euo pipefail
mkdir -p results/looped-lm/twohop-neurons
for c in r1_l8/checkpoint_epoch_2001 r1_l4/checkpoint_epoch_13501 r2_l4/checkpoint_epoch_1201 r2_l4/checkpoint_epoch_7101 \
         r4_l4/checkpoint_epoch_401 r4_l4/checkpoint_epoch_3501 r8_l4/checkpoint_epoch_301 r8_l4/checkpoint_epoch_2901; do
  name=$(echo "$c" | sed 's|/checkpoint_epoch_|_|')
  uv run --no-sync python -m looped_lm.twohop_neurons --root "${KOHLI_ROOT:?}" \
    --checkpoint "checkpoints/systematicity/$c.pt" --out "results/looped-lm/twohop-neurons/$name.json" \
    > "results/looped-lm/twohop-neurons/$name.log" 2>&1
  echo "done $name"
done
