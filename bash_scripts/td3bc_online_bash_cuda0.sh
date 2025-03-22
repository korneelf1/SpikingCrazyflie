python TD3BC_Online.py --device='cuda' --hidden-sizes  256 128 --bc-factor 0.99 --bc-val 0.2 --jumpstart --curriculum --order 1
python TD3BC_Online.py --device='cuda' --hidden-sizes  256 128 --bc-factor 0.99 --bc-val 0.2 --jumpstart --curriculum --order 0
python TD3BC.py --device='cuda' --hidden-sizes  256 128 --bc-factor 0.99 --bc-val 0.2 --jumpstart --curriculum --order 1
python TD3BC.py --device='cuda' --hidden-sizes  256 128 --bc-factor 0.99 --bc-val 0.2 --jumpstart --curriculum --order 1

