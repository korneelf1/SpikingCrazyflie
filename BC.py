# behaverioral cloning
import numpy as np
import torch
from tianshou.data import ReplayBuffer
from tqdm import tqdm
class BC:
    def __init__(self, model, optimizer, buffer, batch_size=256, warmup=50):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.batch_size = batch_size
        self.warmup = warmup
        self.best_epoch_loss = np.inf
        self.loss_fn = torch.nn.MSELoss()

    def learn(self, epoch=50):
        loss = np.inf
        for _ in tqdm(range(epoch)):
            losses=[]
            for _ in range(int(len(self.buffer)//self.batch_size*.7)):
                self.model.preprocess.reset()
                batch = self.buffer.sample(self.batch_size)[0]
                observations = batch.obs[:,:, :18]
                actions = torch.tensor(batch.obs[:,:, 146:150]).to(torch.float32)

                self.optimizer.zero_grad()
                outputs = []
                for t in range(observations.shape[1]):

                    output = self.model(observations[:, t])[0][0] # actor
                    outputs.append(output)

                outputs = torch.stack(outputs, dim=1).to(torch.float32)
                loss = self.loss_fn(outputs[:,self.warmup:], actions[:,self.warmup:]).to(torch.float32)
                loss.backward()
                losses.append(loss.item())
                wandb.log({"loss":loss.item()})
                self.optimizer.step()
            wandb.log({"epoch_loss":np.mean(losses)})
            if np.mean(losses)<self.best_epoch_loss:
                self.best_epoch_loss = np.mean(losses)
                torch.save(self.model.state_dict(), "model_bc.pth")
                
        
    
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
    from l2f_gym import Learning2Fly

    # spiking neural network specific:
    from spiking_gym_wrapper import SpikingEnv
    from spikingActorProb import SpikingNet
    

    # wandb
    import wandb

    # wandb.init(project="l2f_bc")
    wandb.init(mode="disabled")

    # prepare the data
    buffer = ReplayBuffer.load_hdf5('l2f_controller_buffer_short.hdf5')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spiking_module = SpikingNet(state_shape=18, action_shape=128, hidden_sizes=[128], device=device, reset_in_call=False, repeat=0).to(device)
    model = ActorProb(spiking_module, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # prepare the BC
    bc = BC(model, optimizer, buffer, batch_size=1)
    # learn the model
    loss = bc.learn()
    print(loss)