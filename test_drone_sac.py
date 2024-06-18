import argparse
import datetime
import os
import pprint

import numpy as np
import torch

from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from custom_collector import ParallelCollector
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
# from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import PPOPolicy, SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import WandbLogger

from gym_sim import Drone_Sim
import wandb


def create_policy():
    # create the networks behind actors and critics
    net_a = Net(state_shape=observation_space,
                hidden_sizes=[64,64], device='cpu')
    net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[64,64],
                    concat=True,)
    net_c2 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[64,64],
                    concat=True,)

    # create actors and critics
    actor = ActorProb(
        net_a,
        action_space,
        unbounded=True,
        conditioned_sigma=True,
    )
    critic1 = Critic(net_c1, device='cpu')
    critic2 = Critic(net_c2, device='cpu')

    # create the optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    # create the policy
    policy = SACPolicy(actor=actor, actor_optim=actor_optim, \
                        critic=critic1, critic_optim=critic_optim,\
                        critic2=critic2, critic2_optim=critic2_optim,\
                        action_space=env.action_space,\
                        observation_space=env.observation_space, \
                        action_scaling=True) # make sure actions are scaled properly
    return policy

policy_path = 'policy.pth'
env = Drone_Sim()
observation_space = env.observation_space.shape or env.observation_space.n
action_space = env.action_space.shape or env.action_space.n


policy = create_policy()

policy.load_state_dict(torch.load("policy.pth"))

# actor = SACPolicy.actor

state,_ = env.reset()
env.step_rollout(policy, tianshou_policy=True, test = True)