python TD3BC_Online.py --slope=2 --device='cuda:2' --hidden-sizes  256 128 --bc-factor .99 --jumpstart --curriculum
python TD3BC_Online.py --slope=100 --device='cuda:2' --hidden-sizes  256 128 --bc-factor .99  --jumpstart --curriculum
python TD3BC_Online.py --slope=2 --device='cuda:2' --hidden-sizes  256 128 --bc-factor .99 
python TD3BC_Online.py --slope=100 --device='cuda:2' --hidden-sizes  256 128 --bc-factor .99 
