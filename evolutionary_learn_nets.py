from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger, WandbLogger
from evotorch.neuroevolution import GymNE
from gym_sim import Drone_Sim
from spikingActorProb import SpikingNet
import torch
import wandb

wandb.init(mode='disabled')
simulator = Drone_Sim()
env_config = {
    "N_drones": 1,
    "gpu": False,
    }
actor = SpikingNet(state_shape=simulator.observation_space.shape, 
                       action_shape=simulator.action_space.shape,
                       device="cpu",
                       hidden_sizes=[64, 64])
# torch.save(actor.state_dict(), "spiking_actor.pth")

# actor.load_state_dict(torch.load("spiking_actor.pth"))
# Specialized Problem class for RL
problem = GymNE(
    env=Drone_Sim,
    # Linear policy
    # network="Linear(obs_length, act_length)",
    network=actor,
    # network_args=
    env_config=env_config,
    observation_normalization=True,
    decrease_rewards_by=5.0,
    # Use all available CPU cores
    num_actors="max",
)

searcher = PGPE(
    problem,
    popsize=200,
    center_learning_rate=0.01125,
    stdev_learning_rate=0.1,
    optimizer_config={"max_speed": 0.015},
    radius_init=0.27,
    num_interactions=150000,
    popsize_max=3200,
)
# logger = StdOutLogger(searcher)
logger = WandbLogger(searcher, project="evotorch drone sim")
searcher.run(1)
# torch.save(actor.state_dict(), "spiking_actor.pth")


population_center = searcher.status["center"]
policy = problem.to_policy(population_center)
torch.save(policy.state_dict(), "spiking_actor.pth")
problem.visualize(policy)