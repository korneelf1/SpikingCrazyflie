from evotorch.algorithms import PGPE, CEM
from evotorch.logging import StdOutLogger, WandbLogger
from evotorch.neuroevolution import GymNE
from gym_sim import Drone_Sim
from spikingActorProb import SpikingNet
import wandb
import torch

# for seeds
import numpy as np
import random

simulator = Drone_Sim()
env_config = {
    "N_drones": 1,
    "gpu": False,
    }
# Specialized Problem class for RL
problem = GymNE(
    env=Drone_Sim,
    # Linear policy
    # network="Linear(obs_length, act_length)",
    network=SpikingNet(state_shape=simulator.observation_space.shape, 
                       action_shape=simulator.action_space.shape,
                       device="cpu",
                       hidden_sizes=[64, 64]),
    # network_args=
    env_config=env_config,
    observation_normalization=True,
    decrease_rewards_by=.0,
    # Use all available CPU cores
    num_actors="max",
)


# searcher = CEM(
#     problem,
#     popsize=500,
#     stdev_init=.5,
#     parenthood_ratio=.5,
    
# )

config = {'Algorithm':'PGPE',
        'Spiking':True,
        'popsize':200,
        'iterations': 500,
        'seed': int(3),
        'center_learning_rate':0.01125,
        'stdev_learning_rate':0.1,
        'max_speed' : 0.015,
        'radius_init' :0.27,
        'num_interactions':150000,
        'popsize_max':3200,
        }
searcher = PGPE(
    problem,
    popsize=config['popsize'],
    center_learning_rate=config['center_learning_rate'],
    stdev_learning_rate=config['stdev_learning_rate'],
    optimizer_config={"max_speed": config['max_speed']},
    radius_init=config['radius_init'],
    num_interactions=config['num_interactions'],
    popsize_max=config['popsize_max'],
)

np.random.seed(config['iterations'])
random.seed(config['iterations'])
torch.random.manual_seed(config['iterations'])

# wandb.init(project="evotorch drone sim",config=config)
# logger = StdOutLogger(searcher)
logger = WandbLogger(searcher, project="evotorch drone sim", config=config)
searcher.run(config['iterations'])

population_center = searcher.status["center"]
policy = problem.to_policy(population_center)
torch.save(policy.state_dict(),f'spiking_evo_{config["Algorithm"]}_{config["iterations"]}.pth')
# problem.visualize(policy)