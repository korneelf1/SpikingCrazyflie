#!/bin/bash

# TD3BC_Online Training Script with Ablation Study
# This script runs multiple configurations to study the impact of different hyperparameters

echo "Starting TD3BC_Online Training with Ablation Study..."

# Common hyperparameters for all configurations
SLOPE=2
HIDDEN_SIZES="256 128"
CURRICULUM="--curriculum"
SLOPE_SCHEDULE="adaptive"
SCHEDULING_ORDER=3
BATCH_SIZE=1024
BUFFER_SIZE=25000
N_ROLLOUTS_PER_GATHER=3000
EPOCHS_PER_GATHER=20
MAX_EPOCHS=1000
UPDATE_ACTOR_FREQ=2
TAU=0.01
GAMMA=0.999
ALPHA=2.0
WANDB_PROJECT="td3bc_online_ablation"

echo "=========================================="
echo "Configuration 1: Normal TD3BC_Online"
echo "bc_val=1.0, jumpstart=True"
echo "=========================================="
python TD3BC_Online.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --bc-val 1.0 \
    --jumpstart \
    --wandb-project $WANDB_PROJECT \
    --ablation "normal_td3bc_online" \
    --batch-size $BATCH_SIZE \
    --buffer-size $BUFFER_SIZE \
    --n_rollouts_per_gather $N_ROLLOUTS_PER_GATHER \
    --epochs_per_gather $EPOCHS_PER_GATHER \
    --max_epochs $MAX_EPOCHS \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --tau $TAU \
    --gamma $GAMMA \
    --alpha $ALPHA

echo "=========================================="
echo "Configuration 2: No Behavioral Cloning"
echo "bc_val=0.0, jumpstart=True"
echo "=========================================="
python TD3BC_Online.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --bc-val 0.0 \
    --jumpstart \
    --wandb-project $WANDB_PROJECT \
    --ablation "td3bc_online_bc_val_0_jumpstart_true" \
    --batch-size $BATCH_SIZE \
    --buffer-size $BUFFER_SIZE \
    --n_rollouts_per_gather $N_ROLLOUTS_PER_GATHER \
    --epochs_per_gather $EPOCHS_PER_GATHER \
    --max_epochs $MAX_EPOCHS \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --tau $TAU \
    --gamma $GAMMA \
    --alpha $ALPHA

echo "=========================================="
echo "Configuration 3: No Jumpstart"
echo "bc_val=1.0, jumpstart=False"
echo "=========================================="
python TD3BC_Online.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --bc-val 1.0 \
    --wandb-project $WANDB_PROJECT \
    --ablation "td3bc_online_bc_val_1_jumpstart_false" \
    --batch-size $BATCH_SIZE \
    --buffer-size $BUFFER_SIZE \
    --n_rollouts_per_gather $N_ROLLOUTS_PER_GATHER \
    --epochs_per_gather $EPOCHS_PER_GATHER \
    --max_epochs $MAX_EPOCHS \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --tau $TAU \
    --gamma $GAMMA \
    --alpha $ALPHA

echo "=========================================="
echo "Configuration 4: Pure Online RL"
echo "bc_val=0.0, jumpstart=False"
echo "=========================================="
python TD3BC_Online.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --bc-val 0.0 \
    --wandb-project $WANDB_PROJECT \
    --ablation "td3bc_online_bc_val_0_jumpstart_false" \
    --batch-size $BATCH_SIZE \
    --buffer-size $BUFFER_SIZE \
    --n_rollouts_per_gather $N_ROLLOUTS_PER_GATHER \
    --epochs_per_gather $EPOCHS_PER_GATHER \
    --max_epochs $MAX_EPOCHS \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --tau $TAU \
    --gamma $GAMMA \
    --alpha $ALPHA

echo "=========================================="
echo "Configuration 5: Jumpstart Only for Warmup"
echo "bc_val=1.0, jumpstart_only_for_warmup=True"
echo "=========================================="
python TD3BC_Online.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --bc-val 1.0 \
    --jumpstart_only_for_warmup \
    --wandb-project $WANDB_PROJECT \
    --ablation "td3bc_online_jumpstart_warmup_only" \
    --batch-size $BATCH_SIZE \
    --buffer-size $BUFFER_SIZE \
    --n_rollouts_per_gather $N_ROLLOUTS_PER_GATHER \
    --epochs_per_gather $EPOCHS_PER_GATHER \
    --max_epochs $MAX_EPOCHS \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --tau $TAU \
    --gamma $GAMMA \
    --alpha $ALPHA

echo "=========================================="
echo "Configuration 6: Conservative BC"
echo "bc_val=0.5, jumpstart=True"
echo "=========================================="
python TD3BC_Online.py \
    --slope $SLOPE \
    --hidden-sizes $HIDDEN_SIZES \
    $CURRICULUM \
    --slope_schedule $SLOPE_SCHEDULE \
    --scheduling_order $SCHEDULING_ORDER \
    --bc-val 0.5 \
    --jumpstart \
    --wandb-project $WANDB_PROJECT \
    --ablation "td3bc_online_bc_val_0.5_jumpstart_true" \
    --batch-size $BATCH_SIZE \
    --buffer-size $BUFFER_SIZE \
    --n_rollouts_per_gather $N_ROLLOUTS_PER_GATHER \
    --epochs_per_gather $EPOCHS_PER_GATHER \
    --max_epochs $MAX_EPOCHS \
    --update-actor-freq $UPDATE_ACTOR_FREQ \
    --tau $TAU \
    --gamma $GAMMA \
    --alpha $ALPHA

echo "=========================================="
echo "TD3BC_Online Ablation Study Completed!"
echo "=========================================="
echo "Summary of configurations tested:"
echo "1. Normal TD3BC_Online (bc_val=1.0, jumpstart=True)"
echo "2. No Behavioral Cloning (bc_val=0.0, jumpstart=True)"
echo "3. No Jumpstart (bc_val=1.0, jumpstart=False)"
echo "4. Pure Online RL (bc_val=0.0, jumpstart=False)"
echo "5. Jumpstart Only for Warmup (bc_val=1.0, jumpstart_only_for_warmup=True)"
echo "6. Conservative BC (bc_val=0.5, jumpstart=True)"
echo ""
echo "Common hyperparameters used:"
echo "  - Slope: $SLOPE"
echo "  - Hidden sizes: $HIDDEN_SIZES"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Buffer size: $BUFFER_SIZE"
echo "  - N rollouts per gather: $N_ROLLOUTS_PER_GATHER"
echo "  - Epochs per gather: $EPOCHS_PER_GATHER"
echo "  - Max epochs: $MAX_EPOCHS"
echo "  - Update actor freq: $UPDATE_ACTOR_FREQ"
echo "  - Tau: $TAU"
echo "  - Gamma: $GAMMA"
echo "  - Alpha: $ALPHA"
echo "  - WandB project: $WANDB_PROJECT"
