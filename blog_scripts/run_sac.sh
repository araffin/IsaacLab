#!/bin/bash

export XDG_CACHE_HOME=/home_local/$USER/.cache/
source env_isaaclab/bin/activate

# seeds=(3831217417 1756129809 4075310593 2904435568 4115729820 2726071262 1865059290 1408779145 3716099507 411100252)
seeds=(3831217417 1756129809 4075310593)
envs=(Isaac-Velocity-Flat-Unitree-A1-v0)

for env_id in ${envs[*]}; do
  for seed in ${seeds[*]}; do
      python scripts/reinforcement_learning/sb3/train.py --task $env_id \
      --headless --algo sac --num_envs 2048 --seed $seed
  done
done
