python TD3BC.py --slope 10 --hidden-sizes  256 128 --slope_schedule 'adaptive' --scheduling_order 3 --wandb-project neurips_ablation
python TD3BC.py --slope 10 --hidden-sizes  256 128 --slope_schedule 'adaptive' --scheduling_order 3 --wandb-project neurips_ablation
python TD3BC.py --slope 10 --hidden-sizes  256 128 --slope_schedule 'adaptive' --scheduling_order 3 --wandb-project neurips_ablation

python TD3BC_Online.py --slope 10 --hidden-sizes  256 128 --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 0. --wandb-project neurips_ablation
python TD3BC_Online.py --slope 10 --hidden-sizes  256 128 --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 0. --wandb-project neurips_ablation
python TD3BC_Online.py --slope 10 --hidden-sizes  256 128 --slope_schedule 'adaptive' --scheduling_order 3 --bc-val 0. --wandb-project neurips_ablation


