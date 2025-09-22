#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.env import DummyVectorEnv
from l2f_gym import Learning2Fly

# spiking neural network specific:
from spiking_gym_wrapper import SpikingEnv
from spikingActorProb import SpikingNet

# Import training class
from training.td3bc import TD3BC

# wandb
import wandb
import torch.nn as nn

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="l2f")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-data-task", type=str, default="halfcheetah-expert-v2")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--norm-obs", type=int, default=0)

    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_l2f.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    # Use 'store_true' or 'store_false' for boolean flags
    parser.add_argument("--slope", type=int, default=2, help="Slope value")
    parser.add_argument("--slope_schedule", type=str, default='adaptive')
    parser.add_argument("--scheduling_order", type=int, default=3)
    parser.add_argument("--curriculum", action='store_true', help="Enable reward curriculum scheduling")
    
    # Use 'store_true' for interval if you want it as a flag, or use 'type=int' if it's an integer
    parser.add_argument("--interval", type=int, default=1, help="Interval flag")
    return parser.parse_args()

SIGMA_MIN = 1e-3
SIGMA_MAX = .2
from torch.distributions import Independent, Normal
def dist_fion(mu,sigma):
    return Independent(Normal(loc=mu, scale=torch.clamp(sigma, min=SIGMA_MIN, max=SIGMA_MAX).exp()), 1)

class Wrapper(nn.Module):
    def __init__(self,model , size=128,stoch=False):
        super().__init__()
        self.preprocess = model
        self.mu = nn.Linear(size,4)
        self.sigma = nn.Linear(size,4)
        if stoch:
            self.dist = dist_fion
    def forward(self, x):
        x = self.preprocess(x)
        if isinstance(x, tuple):
            x = x[0]    #spiking
        if hasattr(self, 'dist'):
            return self.dist(nn.Tanh()(self.mu(x)), self.sigma(x)).rsample()
        return nn.Tanh()(self.mu(x))
    def reset(self, current_epoch=None):
        if hasattr(self.preprocess, 'reset'):
            self.preprocess.reset(current_epoch=current_epoch)
    def to_cuda(self):
        return self.preprocess.to(device)

if __name__ == "__main__":
    # prepare the data
    # buffer = ReplayBuffer.load_hdf5('l2f_controller_buffer.hdf5')
    buffer = ReplayBuffer.load_hdf5('buffers/l2f_buffer_1996.hdf5')
    # buffer = ReplayBuffer.load_hdf5('real_data_buffer_no_zeros_full.hdf5')
    # buffer2 = ReplayBuffer.load_hdf5('real_data_buffer_no_zeros_2.hdf5')
    # buffer.update(buffer2)
    print(len(buffer))
    env = Learning2Fly(fast_learning=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    wandb_args = {"spiking":True, 'Slope': args.slope,'Schedule': args.slope_schedule, 'Algo':'TD3BC', 'fast_learning':False, 'scheduling_order':args.scheduling_order, 'curriculum':args.curriculum}
    wandb.init(project="l2f_bc", config=wandb_args)
    # wandb.init(mode="disabled")

    wandb.define_metric("*", step_metric="epoch")
    print('Device in use:',device)
    print("Initial slope:",args.slope)
    print("Slope schedule:",args.slope_schedule)
    print("Scheduling order:",args.scheduling_order)
    print("Hidden sizes:",args.hidden_sizes)
    print("Policy Noise:",args.policy_noise)
    print("Curriculum:",args.curriculum)
    wandb.config.update({'slope':args.slope, 'slope_schedule':args.slope_schedule,'scheduling_order':args.scheduling_order,'hidden_sizes':args.hidden_sizes, 'policy_noise':args.policy_noise, 'curriculum':args.curriculum})
        # Initialize the spiking module
    spiking_module = SpikingNet(state_shape=18, 
                                action_shape=args.hidden_sizes[-1], 
                                hidden_sizes=args.hidden_sizes[:-1], 
                                device=device,
                                reset_in_call=False,
                                repeat=1,
                                slope=args.slope,
                                schedule=args.slope_schedule,
                                order=args.scheduling_order,
                                reward_range=(0,400),
                                max_slope=50,
                                verbose=True).to(device)
    
    model = Wrapper(spiking_module, size=args.hidden_sizes[-1]).to(device)
    # model.load_state_dict(torch.load("TD3BC_TEMP_original.pth",map_location=device))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize the critic networks
    critic1 = Critic(
        preprocess_net=Net(146, hidden_sizes=args.hidden_sizes, device=device),
        action_shape=4,
        device=device,
    ).to(device)
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    
    critic2 = Critic(
        preprocess_net=Net(146, hidden_sizes=args.hidden_sizes, device=device),
        action_shape=4,
        device=device,
    ).to(device)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=1e-3)
    
    # prepare the TD3BC  
    td3bc = TD3BC(env,model, optimizer, critic1, critic1_optimizer, critic2, critic2_optimizer, buffer, batch_size=50, device=device, curriculum=args.curriculum)
    # learn the model
    loss = td3bc.learn(epoch=300)
    print(loss)
    wandb.run.finish()
