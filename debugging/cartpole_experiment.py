from TD3BC_Online import TD3BC_Online, Wrapper
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

import gymnasium as gym
# spiking neural network specific:
from spiking_gym_wrapper import SpikingEnv
from spikingActorProb import SpikingNet


# wandb
import wandb
import torch.nn as nn

env = gym.make("InvertedPendulum-v5")



# train decent controller with SB3 first
import stable_baselines3 as sb3
# SAC trained policy
current_dir = os.path.dirname(os.path.abspath(__file__))
policy = sb3.PPO.load(os.path.join(current_dir, "ppo-InvertedPendulum-v5.zip")).policy
# checkpoint = sb3.PPO.load(os.path.join(current_dir, "ppo-InvertedPendulum-v2.zip"))
# controller = sb3.PPO('MlpPolicy', env=env, verbose=1)
# controller.learn(total_timesteps=100000)
# controller.save(os.path.join(current_dir, "ppo-InvertedPendulum-v5.zip"))

# interact with env with the policy and print rewards
obs,info = env.reset()
for i in range(1000):
    action, _ = policy.predict(obs)
    obs, reward, done, terminated,info = env.step(action)
    print(reward)
    if done or terminated:
        obs, info = env.reset()
# controller.save("pendulum_controller.pth")
# wrap controller such that policy() calls policy.predict()
class Controller(nn.Module):
    def __init__(self, policy, device):
        super().__init__()
        self.policy = policy.to(device)
        self.device = device
    def forward(self, obs):
        return torch.tensor(self.policy.predict(obs)[0], device=self.device, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

controller = Controller(policy, device)

spiking_module = SpikingNet(state_shape=env.observation_space.shape, 
                            action_shape=64, 
                            hidden_sizes=[64,64],
                            repeat=1,
                            slope=5,
                            max_slope=5,
                            schedule = 'fixed',
                            device=device)

non_spiking = Net(
    state_shape=env.observation_space.shape[0],
    action_shape=64,
    hidden_sizes=[64],
    concat=False,
    device=device,
)

use_spiking = True
if use_spiking:
    model = Wrapper(spiking_module, size=64, action_size=env.action_space.shape[0]).to(device)
else:   
    model = Wrapper(non_spiking, size=64, action_size=env.action_space.shape[0]).to(device)
net_c1 = Net(
    state_shape=env.observation_space.shape[0],
    action_shape=env.action_space.shape[0],
    hidden_sizes=[64,64],
    concat=True,
    device=device,
)
net_c2 = Net(
    state_shape=env.observation_space.shape[0],
    action_shape=env.action_space.shape[0],
    hidden_sizes=[64,64],
    concat=True,
    device=device,
)
critic = Critic(net_c1, device=device, flatten_input=False).to(device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
critic2 = Critic(net_c2, device=device, flatten_input=False).to(device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)


actor_optim = torch.optim.Adam(model.parameters(), lr=3e-4)
# policy = SACPolicy(actor=model, actor_optim=actor_optim, critic=critic1, critic_optim=critic_optim, critic2=critic2, critic_optim2=critic_optim2, action_space=env.action_space, observation_space=env.observation_space)
wandb.init(project="cartpole_experiment")
wandb.define_metric("*", step_metric="epoch")

# Watch models for gradient tracking
wandb.watch(model, log="gradients", log_freq=100, log_graph=True)
wandb.watch(critic, log="gradients", log_freq=100, log_graph=True)
wandb.watch(critic2, log="gradients", log_freq=100, log_graph=True)

n_samples_per_gather = 5000
rollout_len = 1000
batch_size = 256
epochs_per_gather = 10
alpha = 5
jumpstart = False

wandb.config.update({
    "n_samples_per_gather": n_samples_per_gather,
    "rollout_len": rollout_len,
    "batch_size": batch_size,
    "epochs_per_gather": epochs_per_gather,
    "alpha": alpha,
    "gradient_clipping": True,
    "jumpstart": jumpstart,
    "use_spiking": use_spiking,
})


buffer = ReplayBuffer(size=20000)
trainer = TD3BC_Online(env=env, 
                        model=model,
                        optimizer=actor_optim,
                        critic1=critic,
                        critic1_optimizer=critic_optim,
                        critic2=critic2,
                        critic2_optimizer=critic2_optim,
                        buffer=buffer,
                        actor_obs_size=env.observation_space.shape[0],
                        priveliged_obs_size=env.observation_space.shape[0],
                        action_size=env.action_space.shape[0],
                        batch_size=batch_size, 
                        bc_val=0.2 ,
                        controller=controller.to(device),
                        wandb_run=wandb.run,
                        jumpstart_only_for_warmup=False,
                        warmup=10,
                        slicing_interval=5,
                        sequence_length=20,
                        alpha=alpha)

trainer.run(jumpstart=jumpstart,
            n_samples_per_gather=n_samples_per_gather,
            max_epochs=30000,
            epochs_per_gather=epochs_per_gather)
wandb.run.finish()
