from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger, WandbLogger
from evotorch.neuroevolution import GymNE
from gym_sim import Drone_Sim
from spikingActorProb import SpikingNet
import torch
import wandb
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb

simulator = Drone_Sim()


# torch.save(actor.state_dict(), "spiking_actor.pth")

# actor.load_state_dict(torch.load("spiking_actor.pth"))
# Specialized Problem class for RL
wandb_config = {
    "N_drones": 1,
    "gpu": False,
    "hidden_sizes": [64, 64],
    "actor": "SpikingNet",
    "popsize": 200,
    "center_learning_rate": 0.01125,
    "stdev_learning_rate": 0.1,
    "max_speed": 0.015,
    "radius_init": 0.27,
    "num_interactions": 150000,
    "popsize_max": 3200,
    "drone": "stock drone",
    "num_runs": 500,
    "forward_per_sample": 1,
}
env_config = {
    "N_drones": 1,
    "gpu": False,
    "drone": "stock drone",
    }
# actor = SpikingNet(state_shape=simulator.observation_space.shape, 
#                        action_shape=simulator.action_space.shape,
#                        device="cpu",
#                        hidden_sizes=[64, 64], repeat=wandb_config["forward_per_sample"])

net_a = Net(state_shape=simulator.observation_space.shape,
                    hidden_sizes=[64,64])
        
# create actors and critics
actor = ActorProb(
        net_a,
        simulator.action_space.shape,
        unbounded=True,
        conditioned_sigma=True,
    )
problem = GymNE(
    env=Drone_Sim,
    # Linear policy
    # network="Linear(obs_length, act_length)",
    network=actor,
    # network_args=
    env_config=env_config,
    observation_normalization=True,
    decrease_rewards_by=2.0,
    # Use all available CPU cores
    num_actors="max",
)

searcher = PGPE(
    problem,
    popsize=wandb_config["popsize"],
    center_learning_rate=wandb_config["center_learning_rate"],
    stdev_learning_rate=wandb_config["stdev_learning_rate"],
    optimizer_config={"max_speed":wandb_config["max_speed"]},
    radius_init=wandb_config["radius_init"],
    num_interactions=wandb_config["num_interactions"],
    popsize_max=wandb_config["popsize_max"],
)
# logger = StdOutLogger(searcher)
logger = WandbLogger(searcher, project="evotorch drone sim", config=wandb_config)
searcher.run(wandb_config["num_runs"])
# torch.save(actor.state_dict(), "spiking_actor.pth")


population_center = searcher.status["center"]
policy = problem.to_policy(population_center)
torch.save(policy.state_dict(), "spiking_actor.pth")
problem.visualize(policy)