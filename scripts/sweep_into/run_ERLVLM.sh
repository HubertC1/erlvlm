#!/bin/bash

GPU_ID=0

ENV_NAME="metaworld_sweep-into-v2"

CACHED_QUERY=""

WANDB_GROUP="erlvlm_${ENV_NAME}"

CUDA_VISIBLE_DEVICES=$GPU_ID python train_PEBBLE_VLM.py \
  env=$ENV_NAME \
  num_unsup_steps=9000 num_train_steps=1000000 \
  num_ratings=2 \
  image_reward=True \
  vlm_feedback=True \
  reward_loss="mae" \
  weighting_loss=True \
  batch_stratify=True \
  run_group=$WANDB_GROUP \
  n_processes_query=5 \
  use_cached=False \
  query_cached=$CACHED_QUERY \
  seed=0 \
  debug=False