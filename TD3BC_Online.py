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

class TD3BC_Online:
    def __init__(self, env, model, optimizer,
                 critic1, critic1_optimizer, 
                 critic2, critic2_optimizer, 
                 buffer, batch_size=256, warmup=50, 
                 device='cpu', 
                 controller=None, curriculum=False,
                 bc_factor=0.95):
        self.env = env
        self.model = model
        self.critic1 = critic1 
        self.critic2 = critic2
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.curriculum = curriculum    
        # create copy of first 30 percent of replay buffer with expert data
        len_buf = len(buffer)
        print("original buffer size:",len_buf)
        # self._buffer_initial = ReplayBuffer(size=int(len_buf*.3))
        # sub_buffer = buffer.get(list(np.random.randint(0, len_buf,size=(int(len_buf*.3),))))
        # self._buffer_initial.add(sub_buffer) # a buffer with expert data only

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
        self.controller = controller
        # create deep copies of the critic networks
        self.critic1_old = deepcopy(critic1)
        self.critic2_old = deepcopy(critic2)
        self.bc_coeff = .2
        self.bc_factor = bc_factor

        self.best_test = 0
        self.envsteps = 0

        # TD3 adds noise to the target_q actions and evaluates using CURRENT policy! Seems to improve training
        self.policy_noise = 0.2 
        self.noise_clip = 0.5

    def test(self, n_episodes=30,viz=True):
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

        wandb.log({'test reward': avg_rew/n_episodes,'test len': avg_len/n_episodes})
        if avg_rew/n_episodes > self.best_test:
            self.best_test = avg_rew/n_episodes
            torch.save(self.model.state_dict(), 'TD3BC_Online_TEMP.pth')
            wandb.run.log_artifact("TD3BC_Online_TEMP.pth", name='policy_streaming', type='model')

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
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
    
    def compute_returns(self,batch, nstep=1, recompute_with_current_policy=False):
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
        for t in reversed(range(len(rewards))):
            if t<len(rewards)-nstep-1:
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

    def learn_batch(self, batch, length=100):
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
            priv_obs = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
            for t in range(observations.shape[1]):
                mu = self.model(priv_obs[:, t])
                output = mu # actor
                outputs.append(output)

            act = torch.stack(outputs, dim=1).to(torch.float32)
                
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

    def learn(self, epoch=0, end_epoch = 50):
        loss = np.inf
        for n in tqdm(range(epoch, end_epoch)):
            wandb.log({"epoch":n})
            losses=[]
            self.model.to(device)
            # print(self.model.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                self.model.preprocess.reset(current_epoch = n)
                batch = self.buffer.sample(self.batch_size)[0]
                
                self.learn_batch(batch)


            if n%10==0:
                self.test(viz=False)

    def gather_buffer(self, name='l2f_controller_buffer_in_ru ', size = 200, rollout_len = 501, jump_start_len=None,keep_og=False):
        print("Gathering buffer...")
        # gather new buffer
        buffer = ReplayBuffer(size=size)
        n_rollouts = size 
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
                obs_next_lst = []

                returns = []
                obs = env.reset()[0]
                partial_rollout = False # assume full rollout unless dones before end of range
                t_warmup = 0 # timestep in rollout (used after crash)
                for i in range(rollout_len):
                    
                    obs = torch.tensor(obs,device=self.device)
                    if t_warmup<max(self.warmup,js):
                        t_warmup+=1
                        action = self.controller(obs)
                        _ = self.model(obs[:18])
                    else:
                        action = self.model(obs[:18])
                    # action = model(obs)
                    obs_lst.append(obs.cpu().numpy())
                    obs, rewards, dones,_, info = env.step(action.cpu().detach().numpy()) 

                    obs_next_lst.append(obs)
                    action_lst.append(action.cpu().detach().numpy().reshape(4,))
                    rewards_lst.append(rewards)
                    dones_lst.append(dones)


                    if dones:
                        obs = env.reset()[0]
                        # print("Crashed at step", i)
                        t_warmup = 0
                        # if our rollout crashses before 2x warmup, we discard it, it is not worth to warmup the model for only a few timesteps
                        # if our rollout crashes after rolloutlen - 2x warmup, we also wouldnt have at least the warmup length to trian on
                        if not (self.warmup*2<i<rollout_len-self.warmup*2):
                            partial_rollout = True
                        # partial_rollout = True
            
        
            obs_stack = np.hstack((np.array(obs_lst), np.array(action_lst), np.array(rewards_lst).reshape(-1,1), np.array(dones_lst).reshape(-1,1)))
            # add the rollout to the buffer
            buffer.add(Batch({'obs':obs_stack,'act':np.array(action_lst[-1]),'rew':np.array(rewards_lst[-1]),'terminated': np.array(dones_lst[-1]).reshape(-1,1),'truncated': np.array(dones_lst)[-1].reshape(-1,1)}))
            # buffer.add(Batch({'obs':obs_stack}))
        # create buffer with old and new data
        self.buffer.update(buffer)
        self.envsteps+=size*rollout_len
        wandb.log({'environment interactions': self.envsteps})
        return buffer
    
    def run(self, jumpstart=True):
        cur_epoch = 0
        # self.gather_buffer(jump_start_len=490, size=1000)
        n_epochs_tot = 0
        for i in range(50,450,25):
            self.learn(epoch=cur_epoch, end_epoch=cur_epoch+50)
            n_epochs_tot+=50
            self.bc_coeff *= self.bc_factor
            cur_epoch+=50
            
            if self.curriculum and i%100==0: # 
                self.env.update_curriculum()
                self.env.update_curriculum()
            if jumpstart:
                self.gather_buffer(jump_start_len=500-i)
                wandb.log({"jump start steps (500 - n)": i})
            else:
                self.gather_buffer()
            wandb.log({"behavorial cloning coefficient": self.bc_coeff})
        while n_epochs_tot<1000:
            self.learn(epoch=cur_epoch, end_epoch=cur_epoch+50)
            n_epochs_tot+=50
            cur_epoch+=50
            if self.curriculum:
                self.env.update_curriculum()
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
        return self.preprocess.to(device)

  
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

    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, default="l2f")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--expert-data-task", type=str, default="halfcheetah-expert-v2")
        parser.add_argument("--buffer-size", type=int, default=1000000)
        parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256,128])
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
        parser.add_argument("--surrogate-scheduling", action='store_true', help="Enable surrogate scheduling")
        parser.add_argument("--curriculum", action='store_true', help="Enable reward curriculum scheduling")
        parser.add_argument("--jumpstart", action='store_true', help="JumpStartScheduling")

        parser.add_argument("--slope", type=int, default=2, help="Slope value")
        parser.add_argument("--bc-factor", type=float, default=0.99, help="Behavioral cloning factor")
        
        # Use 'store_true' for interval if you want it as a flag, or use 'type=int' if it's an integer
        parser.add_argument("--interval", type=int, default=1, help="Interval flag")
        return parser.parse_args()

    from l2f_agent import ConvertedModel
    controller = ConvertedModel()
    controller.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))


    # prepare the data
    bufferog = ReplayBuffer.load_hdf5('l2f_controller_buffer.hdf5')
    buffer = ReplayBuffer(size=1000)
    # buffer.update(bufferog)
    env = Learning2Fly(fast_learning=False)
    # list all availabel devices
    print("Available devices:",torch.cuda.device_count())
    # print(torch.device("cuda:1"))
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    args = get_args()
    device = args.device
    wandb_args = {"spiking":True, 'Slope': args.slope,'Schedule': args.surrogate_scheduling, 'Algo':'TD3BC_JS_Online', 'fast_learning':False, 'curriculum':args.curriculum}
    wandb.init(project="l2f_bc", config=wandb_args)
    # wandb.init(mode="disabled")
    wandb.define_metric("*", step_metric="epoch")

    print('Device in use:',device)
    print("Initial slope:",args.slope)
    print("Surrogate scheduling:",args.surrogate_scheduling)
    print("Hidden sizes:",args.hidden_sizes)
    print("Curriculum:",args.curriculum)
    wandb.config.update({"device":device})
    wandb.config.update({"slope":args.slope})
    wandb.config.update({"surrogate_scheduling":args.surrogate_scheduling})
    wandb.config.update({"hidden_sizes":args.hidden_sizes})
    wandb.config.update({"curriculum":args.curriculum})
    wandb.config.update({"bc_factor":args.bc_factor})
    wandb.config.update({"jumpstart":args.jumpstart})
    # args.surrogate_scheduling = True
    spiking_module = SpikingNet(state_shape=18, action_shape=args.hidden_sizes[-1], hidden_sizes=args.hidden_sizes[:-1], device=device,slope=args.slope,slope_schedule=args.surrogate_scheduling,reset_interval=5, reset_in_call=False, repeat=1).to(device)
    model = Wrapper(spiking_module, size=args.hidden_sizes[-1]).to(device)
    # model = ActorProb(spiking_module,4,device=args.device).to(device)
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

    controller.to(device)
    bc = TD3BC_Online(env,model, optimizer,
               controller=controller,
               critic1=critic,
                critic1_optimizer=critic_optim,
                critic2=critic2,
                critic2_optimizer=critic2_optim,
                buffer=buffer,
                batch_size=125, device=device,
                curriculum=args.curriculum,
                bc_factor=args.bc_factor,)
    # learn the model
    bc.run()
    # loss = bc.learn(epoch=250)
    # print(loss)
    
    wandb.run.finish()
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    torch.save(model.state_dict(),f'td3bconline_{timestamp}.pth')