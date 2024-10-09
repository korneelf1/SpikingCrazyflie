# behaverioral cloning
import numpy as np
import torch
from tianshou.data import ReplayBuffer

class BC:
    def __init__(self, model, optimizer, buffer, batch_size=64, warmup=50):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.batch_size = batch_size

    def learn(self, epoch=1):
        for _ in range(epoch):
            batch = self.buffer.sample(self.batch_size)
            self.optimizer.zero_grad()
            output = self.model(batch.obs)
            loss = torch.nn.functional.mse_loss(output, batch.act)
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
if __name__ == "__main__":
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
    from l2f_gym import Learning2Fly, SubprocVectorizedL2F, ShmemVectorizedL2F

    # spiking neural network specific:
    from spiking_gym_wrapper import SpikingEnv
    from spikingActorProb import SpikingNet

    # wandb
    import wandb

    # prepare the data
    buffer = ReplayBuffer.load_hdf5('l2f_controller_buffer.hdf5')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spiking_module = SpikingNet(state_shape=18, action_shape=128, hidden_sizes=[128], device=device).to(device)
    model = ActorProb(spiking_module,
                      action_shape=4,)
    post_processor = 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # prepare the BC
    bc = BC(model, optimizer, buffer)
    # learn the model
    loss = bc.learn()
    print(loss)