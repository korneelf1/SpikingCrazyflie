python tianshou_l2f_ddpg.py --spiking True --slope=2 --device='cuda:1' --exploration-noise 'None' --slope-schedule True --reset-interval 5000
python tianshou_l2f_ddpg.py --spiking True --slope=50 --device='cuda:1' 
python tianshou_l2f_ddpg.py --spiking True --slope=2 --device='cuda:1' --exploration-noise 'None' --slope-schedule True --reset-interval 7000
python tianshou_l2f_ddpg.py --spiking True --slope=50 --device='cuda:1'
bash td3bc_bash_cuda1.sh