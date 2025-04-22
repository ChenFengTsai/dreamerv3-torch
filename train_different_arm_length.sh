#!/bin/bash
# Array of arm lengths to test
ARM_LENGTHS=(0.6 0.8 1.0 1.2 1.4)
for length in "${ARM_LENGTHS[@]}"; do
  echo "Training with arm length scale: $length"
  python dreamer.py --configs dmc_vision --task dmc_reacher_easy \
    --logdir "./logdir/dmc_reacher_easy/arm_length_${length}" \
    --device cuda:0 --modify_env True --arm_length_scale $length
done