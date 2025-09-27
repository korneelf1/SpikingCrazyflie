# behaverioral cloning
import numpy as np
import torch
from copy import deepcopy
import datetime

from tianshou.data import ReplayBuffer
import matplotlib.pyplot as plt
from tianshou.data import Batch,to_torch_as
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from utils.directory_manager import save_checkpoint

class TD3BC_Online:
    def __init__(self, 
                 env, 
                 model, 
                 optimizer,
                 critic1, 
                 critic1_optimizer, 
                 critic2, 
                 critic2_optimizer, 
                 buffer, 
                 batch_size:int = 256, 
                 warmup:int = 50, 
                 device:str = 'cpu', 
                 controller:nn.Module|None = None, 
                 curriculum:bool = False,
                 bc_val:float = 0.2,
                 bc_factor:float = 0.95):
        self.env = env
        self.device = device
        self.model = model
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.curriculum = curriculum  
        self.timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        # create copy of first 30 percent of replay buffer with expert data
        len_buf = len(buffer)
        print("original buffer size:",len_buf)
        # self._buffer_initial = ReplayBuffer(size=int(len_buf*.3))
        # sub_buffer = buffer.get(list(np.random.randint(0, len_buf,size=(int(len_buf*.3),))))
        # self._buffer_initial.add(sub_buffer) # a buffer with expert data only

        self.batch_size = batch_size
        self.warmup = warmup
        self.best_epoch_loss = np.inf
        self.loss_fn = torch.nn.MSELoss().to(device)
        # if device is mps, no float64 can be used
        if device == 'mps':
            torch.set_default_dtype(torch.float32)

        self.alpha = 2.5
        self._alpha = 2.5
        self.gamma = 0.99
        self.tau = 0.005
        self._freq = 1
        self._cnt = 0
        self.controller = controller
        # create deep copies of the critic networks
        self.critic1_old = deepcopy(critic1).to(device)
        self.critic2_old = deepcopy(critic2).to(device)
        self.bc_coeff = bc_val
        self.bc_factor = bc_factor


        self.last_test_rew = 0 # used for adaptive scheduling of slopes
        self.best_test = 0
        self.envsteps = 0
        self.epoch = 0

        # TD3 adds noise to the target_q actions and evaluates using CURRENT policy! Seems to improve training
        self.policy_noise = 0.2 
        self.noise_clip = 0.5


    def test(self, 
             n_episodes:int = 30,
             viz:bool = True):
        avg_rew = 0
        avg_len = 0
        for episode in range(n_episodes):
            obs = self.env.reset()[0]
            done= False
            total_rew = 0
            t = 0
            actions = []
            while not done:
                if t< self.warmup:
                    obs = torch.tensor(obs,device=self.device)
                    action = self.controller(obs)
                    self.model(obs[:18]) # warmup the model
                else:
                    obs = torch.tensor(obs,device=self.device)
                    action = self.model(obs[:18])
                
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

        self.last_test_rew = avg_rew/n_episodes
        wandb.log({'test reward': self.last_test_rew,'test len': avg_len/n_episodes})
        if self.last_test_rew > self.best_test:
            self.best_test = self.last_test_rew
            filename = f"TD3BC_Online_TEMP_{self.timestamp}.pth"
            checkpoint_path = save_checkpoint(self.model.state_dict(), filename)
            wandb.run.log_artifact(checkpoint_path, name='policy_streaming', type='model')

    def _target_q(self, 
                  buffer: ReplayBuffer, 
                  indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        act_ = self(obs_next_batch, model="actor_old").act
        noise = torch.randn(size=act_.shape, device=act_.device) * self.policy_noise
        if self.noise_clip > 0.0:
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
        act_ += noise
        return torch.min(
            self.critic_old(obs_next_batch.obs, act_),
            self.critic2_old(obs_next_batch.obs, act_),
        )
    
    def compute_returns(self,
                        batch, 
                        nstep:int = 1, 
                        recompute_with_current_policy:bool = False):
        '''Compute returns from rewards using discounted rewards:
        R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * r_T
        where T is the last timestep of the episode
        recompute_with_current_policy: bool, if True, recompute the actions using the current policy, which makes te return estimate more accurate
        '''
        gamma = self.gamma
        rewards = batch.rew
        dones = batch.done
        returns = torch.zeros_like(rewards)
        running_returns = 0
        actions = batch.act
        observations = batch.obs
        
        #torch.min(
        #     self.critic1_old(obs_next_batch.obs, act_),
        #     self.critic2_old(obs_next_batch.obs, act_),
        # )
        running_returns = 0
        # compute returns with Bellman equation and critics as value functin
        for t in reversed(range(rewards.shape[-1])):
            if t<rewards.shape[-1]-nstep-1:
                act_  = actions[:,t+2]
                noise = torch.randn(size=act_.shape, device=act_.device) * self.policy_noise
                if self.noise_clip > 0.0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                act_ += noise
                running_returns = rewards[:,t] + gamma*rewards[:,t+1]+\
                gamma**2 * torch.min(self.critic1_old(observations[:,t+2],act_),
                              self.critic2_old(observations[:,t+2],act_)).squeeze(-1) * (1 - dones[:,t+2])
            else:
                running_returns = rewards[:,t] + gamma*running_returns * (1 - dones[:,t])
            returns[:,t] = running_returns.detach()

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
        td = current_q - target_q.clone().detach().requires_grad_(True).to(self.device)
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

    def learn_batch(self, 
                    batch, 
                    length:int = 100):
        if batch.obs.shape[1]%length!=0:
            # print("Batch length not divisible by length, cutting original batch")
            batch.obs = batch.obs[:,:-(batch.obs.shape[1]%length)]
        
            
        # create batch from first observations
        batch_size = batch.obs.shape[0]*batch.obs.shape[1]//length
        observations = batch.obs[:,:, :146]
        actions = batch.obs[:,:, 146:150]
        rewards = batch.obs[:,:, 150]
        terminated = batch.obs[:,:, 151]
        observations_next = np.hstack((batch.obs[:,1:, :146], np.zeros((batch.obs.shape[0],1, 146))),dtype=np.float32)

        

        # compute returns
        # modified batch
        batch = Batch(
            obs=observations,
            act=actions,
            rew=rewards,
            done=terminated,
            obs_next=observations_next,
            returns=np.empty((1,),dtype=np.float32)
        )
        batch = to_torch_as(batch,torch.zeros((1,),device=self.device,dtype=torch.float32))
        compute_returns = self.compute_returns(batch)
        batch.returns = compute_returns

        # reshape them where now (batch, t, features) -> (batch*500/length, length, features)
        observations = batch.obs.reshape(-1,length,146)
        actions = batch.act.reshape(-1,length,4)
        rewards = batch.rew.reshape(-1,length)
        terminated = batch.done.reshape(-1,length)
        observations_next = batch.obs_next.reshape(-1,length,146)
        returns = batch.returns.reshape(-1,length)

        batch = Batch(
            obs=observations,
            act=actions,
            rew=rewards,
            done=terminated,
            obs_next=observations_next,
            returns=returns
        )

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
            self._cnt+=1
            self.optimizer.zero_grad()
            outputs = []
            # hidden = None
            # print(self.model.device)
            priv_obs = batch.obs[:,:, :18].clone().detach().requires_grad_(True).to(self.device).to(torch.float32)
            for t in range(observations.shape[1]):
                mu = self.model(priv_obs[:, t])
                output = mu # actor
                outputs.append(output)

            act = torch.stack(outputs, dim=1)
                
            q_value = self.critic1(batch.obs.reshape(-1,146), batch.act.reshape(-1,4)).reshape(-1,length)
            # after warmup
            q_value = q_value[:,self.warmup:]
            act = act[:,self.warmup:]
            lmbda = self._alpha / q_value.abs().mean().detach()
            actor_loss = -lmbda * q_value.mean() + self.bc_coeff*F.mse_loss(
                act, to_torch_as(batch.act[:,self.warmup:], act)
            )
            actor_loss.backward()
            self._last = actor_loss.item()
            self.optimizer.step()
            self.sync_weight()
            wandb.log({"actor_loss": actor_loss.item()})
            wandb.log({"critic_loss": critic_loss.item()})
            wandb.log({"critic2_loss": critic2_loss.item()})

    def learn(self, 
              epoch:int = 0, 
              end_epoch:int = 10):
        loss = np.inf
        self.epoch += end_epoch
        for n in tqdm(range(epoch, end_epoch)):
            wandb.log({"epoch":n})
            losses=[]
            self.model.to(self.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                self.model.preprocess.reset(current_epoch = n,
                                            last_test_rew = self.last_test_rew) # pass last test reward and current epoch to reset (used for adaptive scheduling and interval based scheduling  of slopes, respectively)
                batch = self.buffer.sample(self.batch_size)[0]
                
                self.learn_batch(batch)


            if n%3==0:
                self.test(viz=False)

    def gather_buffer(self, 
                      name:str ='l2f_controller_buffer_in_ru ', 
                      size:int = 200, 
                      rollout_len:int = 501, 
                      jump_start_len:int|None = None,
                      keep_og:bool = False):
        print("Gathering buffer...")
        # gather new buffer - size should be large enough to hold all rollout data
        buffer = ReplayBuffer(size=size * (rollout_len/100)*2) # each sample is 100 in length
        n_rollouts = size  # number of rollouts to collect
        js = jump_start_len if jump_start_len is not None else 0
        for _ in tqdm(range(n_rollouts), desc="Gathering buffer"):
            # assure that we get usable sequences, by discarding rollouts that crash too fast or all the way at the end
            partial_rollout = True
            
            # make sure to gather full rollouts
            while partial_rollout:
                # creates lists
                obs_lst = []
                action_lst = []
                rewards_lst = []
                dones_lst = []
                # obs_next_lst = []

                returns = []
                obs = env.reset()[0]
                partial_rollout = False # assume full rollout unless dones before end of range
                t_warmup = 0 # timestep in rollout (used after crash)
                i = 0
                dones = False
                while i < rollout_len and not dones:
                    
                    obs = torch.tensor(obs,device=self.device)
                    if t_warmup<max(self.warmup,js):
                        t_warmup+=1
                        action = self.controller(obs)
                        _ = self.model(obs[:18])
                    else:
                        action = self.model(obs[:18])

                    obs_lst.append(obs.cpu().numpy())
                    obs, rewards, dones,_, info = env.step(action.cpu().detach().numpy()) 

                    # obs_next_lst.append(obs)
                    action_lst.append(action.cpu().detach().numpy().reshape(4,))
                    rewards_lst.append(rewards)
                    dones_lst.append(dones)
                    i += 1


                   
                obs = env.reset()[0]

                t_warmup = 0
                # if our rollout crashses before 2x warmup, we discard it, it is not worth to warmup the model for only a few timesteps
                # if our rollout crashes after rolloutlen - 2x warmup, we also wouldnt have at least the warmup length to trian on
                if (self.warmup*2<i):
                    obs_stack = np.hstack((np.array(obs_lst), np.array(action_lst), np.array(rewards_lst).reshape(-1,1), np.array(dones_lst).reshape(-1,1)))
                    partial_rollout = False
                    # once stacked, remove original lists to save memory
                    obs_lst.clear()
                    action_lst.clear()
                    rewards_lst.clear()
                    dones_lst.clear()
                    # obs_next_lst.clear()
                    
                    # add the rollout to the buffer
                    # chop up in 100 step sequences with step size of 50 (0-100,50-150,100-200,150-250,...)
                    for j in range(0,obs_stack.shape[0]-100,50):
                        # Extract data from obs_stack: obs (0:146), actions (146:150), rewards (150:151), dones (151:152)
                        # NOTE only obs_stack is used, rest is saved for compatibility purposes
                        buffer.add(Batch({
                            'obs': obs_stack[j:j+100],
                            'act': obs_stack[j:j+100, 146:150][-1],
                            'rew': float(obs_stack[j:j+100, 150:151][-1].item()),
                            'terminated': bool(obs_stack[j:j+100, 151:152][-1].item()),
                            'truncated': bool(obs_stack[j:j+100, 151:152][-1].item())
                        }))

                    # now add the last bit obs_stack.shape[0]%100 to the buffer
                    if obs_stack.shape[0]%100>0:
                        # Extract data from the last 100 elements of obs_stack
                        buffer.add(Batch({
                            'obs': obs_stack[-100:],
                            'act': obs_stack[-100:, 146:150][-1],
                            'rew': float(obs_stack[-100:, 150:151][-1].item()),
                            'terminated': bool(obs_stack[-100:, 151:152][-1].item()),
                            'truncated': bool(obs_stack[-100:, 151:152][-1].item())
                        }))
                else:
                    partial_rollout = True    
                    obs_lst.clear()
                    action_lst.clear()
                    rewards_lst.clear()
                    dones_lst.clear()
                    # obs_next_lst.clear()

                    # creates lists
                    obs_lst = []
                    action_lst = []
                    rewards_lst = []
                    dones_lst = []
                    # obs_next_lst = []

                    returns = []
                    obs = env.reset()[0]
                    t_warmup = 0 # timestep in rollout (used after crash)
                            
                    

        # create buffer with old and new data
        self.buffer.update(buffer)
        self.envsteps+=size*rollout_len
        wandb.log({'environment interactions': self.envsteps})
        return buffer
    
    def run(self, 
            jumpstart:bool = False,
            n_rollouts_per_gather:int = 200,
            max_epochs:int = 300,
            epochs_per_gather:int = 10
            ):
        """
        This function is used to run the training.
        It will gather data from the environment and train the model.
        It will also log the training progress to WandB.
        It will also save the model periodically.
        It will also update the curriculum if it is enabled.
        
        Args:
            jumpstart (bool): If True, the training will start with a jumpstart phase, meaning we initially use the pre-trained controller to collect data.

        """
        cur_epoch = 0
        # self.gather_buffer(jump_start_len=490, size=1000)
        n_epochs_tot = 0
        max_epochs = 300
        epochs_per_gather = 10
        iterator = range(0,max_epochs,epochs_per_gather)
        factor_i = 500/0.8/max_epochs # we want to be fully relying on the model by 80 percent of the end of the training
        # Update curriculum every 6 training cycles (more intuitive than len(iterator)//6)
        curriculum_interval = 6
        curriculum_update_count = 0
        for i in iterator:
            if jumpstart:
                self.gather_buffer(jump_start_len=500-i*factor_i, size=n_rollouts_per_gather)
                wandb.log({"jump start steps (500 - n)": i})
            else:
                self.gather_buffer(size=n_rollouts_per_gather)
            wandb.log({"behavorial cloning coefficient": self.bc_coeff})
            self.learn(epoch=cur_epoch, end_epoch=cur_epoch+epochs_per_gather)
            n_epochs_tot+=10
            self.bc_coeff *= self.bc_factor
            cur_epoch+=epochs_per_gather
            
            # Update curriculum every curriculum_interval training cycles
            if self.curriculum and curriculum_update_count % curriculum_interval == 0 and curriculum_update_count > 0:
                self.env.update_curriculum()
                print(f"Curriculum updated at training cycle {curriculum_update_count}")
            curriculum_update_count += 1
        while n_epochs_tot<max_epochs:
            self.learn(epoch=cur_epoch, end_epoch=cur_epoch+epochs_per_gather)
            n_epochs_tot+=epochs_per_gather
            cur_epoch+=epochs_per_gather
            filename = f"TD3BC_Online_TEMP_{self.timestamp}_epoch_{cur_epoch}.pth"
            save_checkpoint(self.model.state_dict(), filename)
            wandb.run.log_artifact(filename, name='policy_streaming', type='model')
            # Update curriculum every 6 epochs in the final training phase
            if self.curriculum and (cur_epoch // epochs_per_gather) % curriculum_interval == 0 and cur_epoch > 0:
                self.env.update_curriculum()
                print(f"Curriculum updated at epoch {cur_epoch}")
            self.gather_buffer()
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
        return self.preprocess.to(self.device)

  
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
    import torch.nn as nn

    # set working directory as file directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")
    print(f"Script directory: {script_dir}")

    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, default="l2f")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--expert-data-task", type=str, default="halfcheetah-expert-v2")
        parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256,128])
        parser.add_argument("--actor-lr", type=float, default=3e-4)
        parser.add_argument("--critic-lr", type=float, default=3e-4)
        parser.add_argument("--epoch", type=int, default=150)
        parser.add_argument("--step-per-epoch", type=int, default=5000)
        parser.add_argument("--n-step", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--buffer-size", type=int, default=20000, help="Buffer size")
        parser.add_argument("--buffer-preload", type=bool, default=True, help="Buffer preload")

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
        parser.add_argument("--wandb-project", type=str, default="l2f_bc")
        parser.add_argument(
            "--watch",
            default=False,
            action="store_true",
            help="watch the play of pre-trained policy only",
        )
        # Use 'store_true' or 'store_false' for boolean flags
        parser.add_argument("--surrogate-scheduling", type=str, default='adaptive', help="Enable surrogate scheduling, options: fixed, interval, adaptive")
        parser.add_argument("--curriculum", action='store_true', help="Enable reward curriculum scheduling")
        parser.add_argument("--jumpstart", action='store_true', help="JumpStartScheduling")
        parser.add_argument("--max_epochs", type=int, default=300, help="Max epochs")
        parser.add_argument("--epochs_per_gather", type=int, default=10, help="Epochs per gather")
        parser.add_argument("--slope", type=int, default=2, help="Slope value")
        parser.add_argument("--slope_schedule", type=str, default='adaptive')
        parser.add_argument("--scheduling_order", type=int, default=3)
        parser.add_argument("--bc-factor", type=float, default=0.99, help="Behavioral cloning factor")
        
        parser.add_argument("--bc-val", type=float, default=0.2, help="Behavioral cloning factor")
        # Use 'store_true' for interval if you want it as a flag, or use 'type=int' if it's an integer
        parser.add_argument("--interval", type=int, default=1, help="Interval flag")
        parser.add_argument("--ablation", type=str, default=None)
        parser.add_argument("--n_rollouts_per_gather", type=int, default=500, help="Number of rollouts per gather")
        parser.add_argument("--stable_flight", action='store_true', help="Enable stable flight")
        return parser.parse_args()

    args = get_args()

    from l2f_agent import ConvertedModel
    controller = ConvertedModel()
    controller.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))


    
    env = Learning2Fly(fast_learning=False, stable_flight=args.stable_flight)
    # list all availabel devices
    print("Available devices:",torch.cuda.device_count())
    # for macos
    
    # print(torch.device("cuda"))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    buffer_path = 'buffers/l2f_buffer_1996.hdf5'
    print(f"Looking for buffer file at: {buffer_path}")
    print(f"Absolute path: {os.path.abspath(buffer_path)}")
    print(f"File exists: {os.path.exists(buffer_path)}")

    buffer = ReplayBuffer(size=args.buffer_size)
    if args.buffer_preload:
        buffer_pre = ReplayBuffer.load_hdf5(buffer_path)
        buffer.update(buffer_pre)
    print(f"Successfully loaded buffer with {len(buffer)} samples")
    # prepare the data
    
    
    if not os.path.exists(buffer_path):
        print("Buffer file not found! Available files in buffers/ directory:")
        if os.path.exists('buffers/'):
            print(os.listdir('buffers/'))
        else:
            print("buffers/ directory does not exist!")
        raise FileNotFoundError(f"Buffer file not found: {buffer_path}")
    
    
    
    # Debug: Check the structure of the original buffer
    if len(buffer) > 0:
        sample = buffer[0]
        print(f"Original buffer sample shapes:")
        print(f"  obs shape: {sample.obs.shape}")
        print(f"  act shape: {sample.act.shape}")
        print(f"  rew shape: {sample.rew.shape}")
        print(f"  done shape: {sample.done.shape}")
        print(f"  terminated shape: {sample.terminated.shape}")
        print(f"  truncated shape: {sample.truncated.shape}")
        
        # Check if there are any extra dimensions
        print(f"  rew dtype: {type(sample.rew)}")
        print(f"  terminated dtype: {type(sample.terminated)}")
        print(f"  truncated dtype: {type(sample.truncated)}")
    # buffer = ReplayBuffer(size=20000)
    # buffer.update(bufferog)
    device = args.device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")
    # Initialize WandB
    wandb_args = {"spiking":True, 'Slope': args.slope,'Schedule': args.surrogate_scheduling, 'Algo':'TD3BC_JS_Online', 'fast_learning':False, 'curriculum':args.curriculum}
    wandb.init(project=args.wandb_project, config=wandb_args)

    wandb.define_metric("*", step_metric="epoch")

    print('Device in use:',device)
    print("Initial slope:",args.slope)
    print("Surrogate scheduling:",args.surrogate_scheduling)
    print("Hidden sizes:",args.hidden_sizes)
    print("Curriculum:",args.curriculum)
    print("Jumpstart:",args.jumpstart)
    args.jumpstart = True
    wandb.config.update({"device":device})
    wandb.config.update({"slope":args.slope})
    wandb.config.update({"surrogate_scheduling":args.surrogate_scheduling})
    wandb.config.update({"hidden_sizes":args.hidden_sizes})
    wandb.config.update({"curriculum":args.curriculum})
    wandb.config.update({"bc_factor":args.bc_factor})
    wandb.config.update({"jumpstart":args.jumpstart})
    wandb.config.update({"ablation":args.ablation})
    # args.surrogate_scheduling = True


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
                                reward_range=(-400,400),
                                max_slope=50,
                                verbose=True).to(device)
    
    # Initialize the wrapper
    model = Wrapper(spiking_module, size=args.hidden_sizes[-1]).to(device)
    # model.load_state_dict(torch.load("TD3BC_Online_TEMP.pth", map_location="cpu"))

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    controller.to(device)
    trainer = TD3BC_Online(env,model, optimizer,
               controller=controller,
               critic1=critic,
                critic1_optimizer=critic_optim,
                critic2=critic2,
                critic2_optimizer=critic2_optim,
                buffer=buffer,
                batch_size=1024 , device=device,
                curriculum=args.curriculum,
                bc_val=args.bc_val,
                bc_factor=args.bc_factor,)

    # learn the model
    trainer.run(jumpstart=args.jumpstart, 
    n_rollouts_per_gather=args.n_rollouts_per_gather,
    max_epochs=args.max_epochs,
    epochs_per_gather=args.epochs_per_gather)
    
    wandb.run.finish()
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    save_checkpoint(model.state_dict(), f'TD3BC_ONLINE_STABLE_{timestamp}.pth')