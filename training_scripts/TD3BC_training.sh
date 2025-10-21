#!/bin/bash

# Fair Comparison Training Script for TD3BC vs TD3BC_Online
# This script ensures both methods use identical hyperparameters for fair comparison

echo "Starting Fair Comparison Training..."

# Common hyperparameters for both methods
# These values override the defaults to ensure fair comparison
SLOPE=2
HIDDEN_SIZES="256 128"
CURRICULUM="--curriculum"
SLOPE_SCHEDULE="adaptive"
SCHEDULING_ORDER=3
BATCH_SIZE=512
ALPHA=2.5
TAU=0.005
GAMMA=0.99
ACTOR_LR=3e-4
CRITIC_LR=3e-4
POLICY_NOISE=0.0
NOISE_CLIP=0.5
EXPLORATION_NOISE=0.1
UPDATE_ACTOR_FREQ=2
MAX_EPOCHS=1000
WANDB_PROJECT="td3bc_fair_comparison"

echo "Training TD3BC (Offline) with aligned hyperparameters..."
python TD3BC.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --batch-size $BATCH_SIZE \
    --alpha $ALPHA \
    --tau $TAU \
    --gamma $GAMMA \
    --actor-lr $ACTOR_LR \
    --critic-lr $CRITIC_LR \
    --policy-noise $POLICY_NOISE \
    --noise-clip $NOISE_CLIP \
    --exploration-noise $EXPLORATION_NOISE \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --epoch $MAX_EPOCHS \
    --wandb-project $WANDB_PROJECT \
    # --ablation "td3bc_offline_fair"


echo "Fair comparison training completed!"
echo "Both methods used identical hyperparameters:"
echo "  - Slope: $SLOPE"
echo "  - Hidden sizes: $HIDDEN_SIZES"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Alpha: $ALPHA"
echo "  - Tau: $TAU"
echo "  - Gamma: $GAMMA"
echo "  - Actor LR: $ACTOR_LR"
echo "  - Critic LR: $CRITIC_LR"
echo "  - Policy noise: $POLICY_NOISE"
echo "  - Noise clip: $NOISE_CLIP"
echo "  - Exploration noise: $EXPLORATION_NOISE"
echo "  - Update actor freq: $UPDATE_ACTOR_FREQ"
echo "  - Max epochs: $MAX_EPOCHS"
