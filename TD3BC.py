# behaverioral cloning
import numpy as np
import torch
from copy import deepcopy

from tianshou.data import ReplayBuffer
import matplotlib.pyplot as plt
from tianshou.data import Batch,to_torch_as
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn

class TD3BC:
    def __init__(self, env, model, optimizer,critic1, critic1_optimizer, critic2, critic2_optimizer, buffer, batch_size=256, warmup=50, device='cpu'):
        self.env = env
        self.model = model
        self.critic1 = critic1 
        self.critic2 = critic2
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.batch_size = batch_size
        self.warmup = warmup
        self.best_epoch_loss = np.inf
        self.loss_fn = torch.nn.MSELoss()
        self.device= device
        self.alpha = 2.5
        self._alpha = 2.5
        self.gamma = 0.99
        self.tau = 0.005
        self._freq = 1
        self._cnt = 0

        # create deep copies of the critic networks
        self.critic1_old = deepcopy(critic1)
        self.critic2_old = deepcopy(critic2)

        self.best_test = 0
    def test(self, n_episodes=20,viz=True):
        avg_rew = 0
        avg_len = 0
        for episode in range(n_episodes):
            obs = self.env.reset()[0]
            done= False
            total_rew = 0
            t = 0
            actions = []
            while not done:
                obs = torch.tensor(obs[:18],device=self.device)
                action = self.model(obs)
                actions.append(action.detach().cpu())
                obs, rew, done, done, info = self.env.step(np.array(action.detach().cpu()))
                t+=1
                total_rew+= rew
            avg_rew+= total_rew
            avg_len+= t
            # print("Flying for: ",t)
            # plot the actions
        if viz:
            actions = np.vstack(actions)
            fig, axs = plt.subplots(4,1,figsize=(10,10))
            for i in range(4):
                plt.subplot(4,1,i+1)
                plt.plot(actions[:,i])
                plt.ylabel(f"Action {i}")
            # plt.show()
            wandb.log({"img": [wandb.Image(fig, caption=f"BC Learning")]})
            batch = self.buffer.sample(1)[0]
            observations = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
            actions = torch.tensor(batch.obs[:,:, 146:150]).to(torch.float32).to(self.device)
            
            outputs = []
            # hidden = None
            # print(self.model.device)
            for t in range(observations.shape[1]):
                mu = self.model(observations[:, t])
                output = mu # actor
                outputs.append(output)

            outputs = torch.stack(outputs, dim=1).to(torch.float32)
            t = np.linspace(0,502,501)
            fig, ax = plt.subplots(4, 1)
            for i in range(4):
                ax[i].plot(t,actions.cpu().detach().numpy()[0,:,i], c='g')
                ax[i].plot(t,outputs.cpu().detach().numpy()[0,:,i], c='r')
            
            # plt.show()
            wandb.log({"img": [wandb.Image(fig, caption=f"Compared to true")]})

        wandb.log({'test reward': avg_rew/n_episodes,'test len': avg_len/n_episodes})
        if avg_rew/n_episodes > self.best_test:
            self.best_test = avg_rew/n_episodes
            torch.save(self.model.state_dict(), 'TD3BC_TEMP.pth')
            wandb.run.log_artifact("TD3BC_TEMP.pth", name='policy_streaming', type='model')

    def compute_returns(self,batch, nstep=1):
        '''Compute returns from rewards using discounted rewards:
        R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * r_T
        '''
        gamma = self.gamma
        rewards = batch.rew
        dones = batch.done
        returns = np.zeros_like(rewards)
        running_returns = 0
        actions = batch.act
        observations = batch.obs
        
        #torch.min(
        #     self.critic1_old(obs_next_batch.obs, act_),
        #     self.critic2_old(obs_next_batch.obs, act_),
        # )
        running_returns = 0
        # compute returns with Bellman equation and critics as value functin
        for t in reversed(range(len(rewards))):
            if t<len(rewards)-nstep:
                running_returns = rewards[t] + gamma*rewards[t+1]+\
                gamma**2 * torch.min(self.critic1_old(observations[:,t+2],actions[:,2]),
                              self.critic2_old(observations[:,t+2],actions[:,2])) * (1 - dones[t+2])
            else:
                running_returns = rewards[t] + gamma*running_returns * (1 - dones[t])
            returns[t] = running_returns

        # for t in reversed(range(len(rewards))):
        #     running_returns = rewards[t] + gamma * running_returns * (1 - dones[t])
        #     returns[t] = running_returns
        return returns
    def _mse_optimizer(
            self,
        batch,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs.reshape(-1,146), batch.act.reshape(-1,4)).flatten()
        target_q = batch.returns.flatten()
        td = current_q - torch.tensor(target_q).to(self.device)
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss
    
    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters(), strict=True):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        # self.soft_update(self.actor_old, self.actor, self.tau)

    def learn_batch(self, batch):
        # create batch from first observations
        batch_size = batch.obs.shape[0]
        observations = batch.obs[:,:, :146]
        actions = batch.obs[:,:, 146:150]
        rewards = batch.obs[:,:, 150]
        terminated = batch.obs[:,:, 151]
        observations_next = np.hstack((batch.obs[:,1:, :146], np.zeros((batch_size,1, 146))))

        # compute returns
        # modified batch
        batch = Batch(
            obs=observations,
            act=actions,
            rew=rewards,
            done=terminated,
            obs_next=observations_next,
            returns=None
        )
        compute_returns = self.compute_returns(batch)
        batch.returns = compute_returns

        # learn critics
        td1, critic_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optimizer
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optimizer
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer
        # actor
        if self._cnt % self._freq == 0:
            self.optimizer.zero_grad()
            outputs = []
            # hidden = None
            # print(self.model.device)
            priv_obs = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
            for t in range(observations.shape[1]):
                mu = self.model(priv_obs[:, t])
                output = mu # actor
                outputs.append(output)

            act = torch.stack(outputs, dim=1).to(torch.float32)
                
            q_value = self.critic1(batch.obs.reshape(-1,146), batch.act.reshape(-1,4)).reshape(-1,501)
            # after warmup
            q_value = q_value[:,self.warmup:]
            act = act[:,self.warmup:]
            lmbda = self._alpha / q_value.abs().mean().detach()
            actor_loss = -lmbda * q_value.mean() + F.mse_loss(
                act, to_torch_as(batch.act[:,self.warmup:], act)
            )
            actor_loss.backward()
            self._last = actor_loss.item()
            self.optimizer.step()
            self.sync_weight()
            wandb.log({"actor_loss": actor_loss.item()})
            wandb.log({"critic_loss": critic_loss.item()})
            wandb.log({"critic2_loss": critic2_loss.item()})
    def learn(self, epoch=50):
        loss = np.inf
        for n in tqdm(range(epoch)):
            losses=[]
            self.model.to(device)
            # print(self.model.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                self.model.preprocess.reset()
                batch = self.buffer.sample(self.batch_size)[0]
                
                self.learn_batch(batch)

            if n%10==0:
                self.test(viz=True)
                
        
    
if __name__ == "__main__":
    #!/usr/bin/env python3

    import argparse
    import datetime
    import os
    import pprint

    import numpy as np
    import torch
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
            default="cuda" if torch.cuda.is_available() else "cpu",
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
        parser.add_argument("--surrogate-scheduling", type=bool,default=False)
        parser.add_argument("--slope", type=int,default=2)
        parser.add_argument("--interval", type=bool,default=1)
        return parser.parse_args()


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
    import torch.nn as nn
    class Wrapper(nn.Module):
        def __init__(self,model , stoch=False):
            super().__init__()
            self.preprocess = model
            self.mu = nn.Linear(128,4)
            self.sigma = nn.Linear(128,4)
            if stoch:
                self.dist = torch.distributions.Normal
        def forward(self, x):
            x = self.preprocess(x)
            if isinstance(x, tuple):
                x = x[0]    #spiking
            if hasattr(self, 'dist'):
                return nn.Tanh()(self.dist(self.mu(x), self.sigma(x)))
            return nn.Tanh()(self.mu(x))
        def reset(self):
            if hasattr(self.preprocess, 'reset'):
                self.preprocess.reset()
        def to_cuda(self):
            return self.preprocess.to(device)
    
    wandb_args = {"spiking":True, 'Slope': 2,'Schedule': True, 'Algo':'BC'}
    # wandb.init(project="l2f_bc", config=wandb_args)
    wandb.init(mode="disabled")

    # prepare the data
    buffer = ReplayBuffer.load_hdf5('l2f_controller_buffer_short.hdf5')
    
    env = Learning2Fly()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    print('Device in use:',device)
    spiking_module = SpikingNet(state_shape=18, action_shape=128, hidden_sizes=[256], device=device,slope=args.slope,slope_schedule=args.surrogate_scheduling, reset_in_call=False, repeat=1).to(device)
    model = Wrapper(spiking_module).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # prepare the BC
        # critic network
    net_c1 = Net(
        state_shape=146,
        action_shape=4,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        state_shape=146,
        action_shape=4,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c1, device=args.device, flatten_input=False).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device, flatten_input=False).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)




    bc = TD3BC(env,model, optimizer,critic1=critic,
                critic1_optimizer=critic_optim,
                critic2=critic2,
                critic2_optimizer=critic2_optim,
                buffer=buffer,
                batch_size=1, device=device)
    # learn the model
    loss = bc.learn(epoch=300)
    print(loss)
    wandb.run.finish()