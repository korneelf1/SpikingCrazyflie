#!/bin/bash

# Ablation Study for TD3BC_Online
# 6 combinations: normal, drop bc_val to 0, drop jumpstart to False, and all combinations

echo "Starting TD3BC_Online Ablation Study..."

# Configuration 1: Normal TD3BC_Online (bc_val=0.2, jumpstart=True)
# echo "Running Configuration 1: Normal TD3BC_Online (bc_val=0.2, jumpstart=True)"
# /Users/korneel/code/personal/SpikingCrazyflie/venv_py311/bin/python TD3BC_Online.py --slope 2 --hidden-sizes 256 128 --curriculum --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 1 --jumpstart --wandb-project neurips_ablation --ablation "normal_td3bc_online_no_preload" --batch-size 1024 --buffer-size 25000 --n_rollouts_per_gather 3000 --epochs_per_gather 20 --max_epochs 1000 --update-actor-freq 2 --tau 0.01 --gamma 0.999 --alpha 2.0

# # Configuration 2: Drop bc_val to 0, keep jumpstart=True
# echo "Running Configuration 2: bc_val=0, jumpstart=True"
# /Users/korneel/code/personal/SpikingCrazyflie/venv_py311/bin/python TD3BC_Online.py --slope 2 --hidden-sizes 256 128 --curriculum --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 0 --jumpstart --wandb-project neurips_ablation --ablation "td3bc_online_bc_val_0_jumpstart_true" --batch-size 1024 --buffer-size 25000 --n_rollouts_per_gather 3000 --epochs_per_gather 20 --max_epochs 1000 --update-actor-freq 2 --tau 0.01 --gamma 0.999 --alpha 2.0

# Configuration 3: Keep bc_val=0.2, drop jumpstart to False
echo "Running Configuration 3: bc_val=0.2, jumpstart=False"
python TD3BC_Online.py --slope 2 --hidden-sizes 256 128 --curriculum --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 1 --wandb-project neurips_ablation --ablation "td3bc_online_bc_val_0.2_jumpstart_false" --batch-size 1024 --buffer-size 25000 --n_rollouts_per_gather 3000 --epochs_per_gather 20 --max_epochs 1000 --update-actor-freq 2 --tau 0.01 --gamma 0.999 --alpha 2.0
echo "Running Configuration 3: bc_val=0.2, jumpstart=False"
python TD3BC_Online.py --slope 2 --hidden-sizes 256 128 --curriculum --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 1 --wandb-project neurips_ablation --ablation "td3bc_online_bc_val_0.2_jumpstart_false" --batch-size 1024 --buffer-size 25000 --n_rollouts_per_gather 3000 --epochs_per_gather 20 --max_epochs 1000 --update-actor-freq 2 --tau 0.01 --gamma 0.999 --alpha 2.0

# Configuration 4: Drop both bc_val to 0 and jumpstart to False
# echo "Running Configuration 4: bc_val=0, jumpstart=False"
# /Users/korneel/code/personal/SpikingCrazyflie/venv_py311/bin/python TD3BC_Online.py --slope 2 --hidden-sizes 256 128 --curriculum --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 0 --wandb-project neurips_ablation --ablation "td3bc_online_bc_val_0_jumpstart_false" --batch-size 1024 --buffer-size 25000 --n_rollouts_per_gather 3000 --epochs_per_gather 20 --max_epochs 1000 --update-actor-freq 2 --tau 0.01 --gamma 0.999 --alpha 2.0

echo "Ablation study completed!"


