python TD3BC_Online.py --slope=100 --device='cuda:1' --hidden-sizes  256 128 --bc-factor 0.999  --jumpstart --surrogate-scheduling
python TD3BC_Online.py --slope=2 --device='cuda:1' --hidden-sizes  256 128 --bc-factor 0.999 --jumpstart ---surrogate-scheduling
python TD3BC_Online.py --slope=2 --device='cuda:1' --hidden-sizes  256 128 --bc-factor 0.999 --jumpstart --surrogate-scheduling



python TD3BC_Online.py --slope=100 --device='cuda:1' --hidden-sizes  256 128 --bc-factor 0.999 --bc-val 0.4 --jumpstart --surrogate-scheduling
python TD3BC_Online.py --slope=2 --device='cuda:1' --hidden-sizes  256 128 --bc-factor 0.999 --bc-val 0.4 --jumpstart ---surrogate-scheduling
python TD3BC_Online.py --slope=2 --device='cuda:1' --hidden-sizes  256 128 --bc-factor 0.999 --bc-val 0.4 --jumpstart --surrogate-scheduling



