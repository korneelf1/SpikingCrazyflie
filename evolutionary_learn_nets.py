from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger, WandbLogger
from evotorch.neuroevolution import GymNE
from gym_sim import Drone_Sim
# from spikingActorProb import SpikingNet
import torch
import wandb
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic, RecurrentActorProb, RecurrentCritic
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.data import Batch
import numpy as np
from typing import Any, cast
from torch.distributions import Independent, Normal
from torch import nn

sim = Drone_Sim()


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
device = 'cpu'
def create_policy():
    # create the networks behind actors and critics
    
    net_a = Net(state_shape=sim.observation_space.shape,
                    hidden_sizes=[64,64], device=device)
        
    net_c1 = Net(state_shape=sim.observation_space.shape,action_shape=sim.action_space.shape,
                    hidden_sizes=[64,64],
                    concat=True,device=device)
    net_c2 = Net(state_shape=sim.observation_space.shape,action_shape=sim.action_space.shape,
                    hidden_sizes=[64,64],
                    concat=True,device=device)
    
    # model_logger.watch(net_a)
    # model_logger.watch(net_c1)
    # model_logger.watch(net_c2)

    # create actors and critics
    actor = ActorProb(
        net_a,
        sim.action_space.shape,
        unbounded=True,
        conditioned_sigma=True,
        device=device
    )
    critic1 = Critic(net_c1, device=device)
    critic2 = Critic(net_c2, device=device)

    # create the optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    # create the policy
    policy = SACPolicy(actor=actor, actor_optim=actor_optim, \
                        critic=critic1, critic_optim=critic_optim,\
                        critic2=critic2, critic2_optim=critic2_optim,\
                        action_space=sim.action_space,\
                        observation_space=sim.observation_space, \
                        action_scaling=True, action_bound_method=None) # make sure actions are scaled properly
    return policy
# actor = SpikingNet(state_shape=simulator.observation_space.shape, 
#                        action_shape=simulator.action_space.shape,
#                        device="cpu",
#                        hidden_sizes=[64, 64], repeat=wandb_config["forward_per_sample"])

# net_a = Net(state_shape=sim.observation_space.shape,
#                     hidden_sizes=[64,64])
        
# # create actors and critics
# actor = ActorProb(
#         net_a,
#         sim.action_space.shape,
#         unbounded=True,
#         conditioned_sigma=True,
#     )

actor = create_policy()

class PolicyWrapper(nn.Module):
    '''Policy wrapper to extract action from Tianshou policy'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def __call__(self,
            obs,
            state: dict | Batch | np.ndarray | None = None,
            **kwargs: Any,
        ):
        (loc_B, scale_B), hidden_BH = self.policy.actor(obs, state=state)
        dist = Independent(Normal(loc=loc_B, scale=scale_B), 1)
        if self.policy.deterministic_eval and not self.policy.is_within_training_step:
            act_B = dist.mode
        else:
            act_B = dist.rsample()
        log_prob = dist.log_prob(act_B).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act_B)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) + 1e-10).sum(
            -1,
            keepdim=True,
        )

        return squashed_action
actor = PolicyWrapper(actor)
problem = GymNE(
    env=Drone_Sim,
    # Linear policy
    # network="Linear(obs_length, act_length)",
    network=actor,
    # network_args=
    env_config=env_config,
    observation_normalization=True,
    decrease_rewards_by=.0,
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
torch.save(policy.state_dict(), "ann_actor.pth")
# problem.visualize(policy)