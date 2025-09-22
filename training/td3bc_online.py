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
        self.model = model
        self.critic1 = critic1 
        self.critic2 = critic2
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
        self.loss_fn = torch.nn.MSELoss()
        self.device= device
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
        self.critic1_old = deepcopy(critic1)
        self.critic2_old = deepcopy(critic2)
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
            torch.save(self.model.state_dict(), filename)
            wandb.run.log_artifact(filename, name='policy_streaming', type='model')

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

    def learn(self, 
              epoch:int = 0, 
              end_epoch:int = 50):
        loss = np.inf
        self.epoch += end_epoch
        for n in tqdm(range(epoch, end_epoch)):
            wandb.log({"epoch":n})
            losses=[]
            self.model.to(device)
            for _ in range(5): # perform 5 updates for each epoch
                # print(self.model.device)
                for _ in range(int(len(self.buffer)//self.batch_size)):
                    self.model.preprocess.reset(current_epoch = n,
                                                last_test_rew = self.last_test_rew) # pass last test reward and current epoch to reset (used for adaptive scheduling and interval based scheduling  of slopes, respectively)
                    batch = self.buffer.sample(self.batch_size)[0]
                    
                    self.learn_batch(batch)


            if n%10==0:
                self.test(viz=False)

    def gather_buffer(self, 
                      name:str ='l2f_controller_buffer_in_ru ', 
                      size:int = 200, 
                      rollout_len:int = 501, 
                      jump_start_len:int|None = None,
                      keep_og:bool = False):
        print("Gathering buffer...")
        # gather new buffer
        buffer = ReplayBuffer(size=size)
        n_rollouts = int(size//(rollout_len/100) )
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

                        t_warmup = 0
                        # if our rollout crashses before 2x warmup, we discard it, it is not worth to warmup the model for only a few timesteps
                        # if our rollout crashes after rolloutlen - 2x warmup, we also wouldnt have at least the warmup length to trian on
                        if (self.warmup*2<i):
                            obs_stack = np.hstack((np.array(obs_lst), np.array(action_lst), np.array(rewards_lst).reshape(-1,1), np.array(dones_lst).reshape(-1,1)))
                            # add the rollout to the buffer
                            # chop up in 100 step sequences with step size of 50 (0-100,50-150,100-200,150-250,...)
                            for j in range(0,obs_stack.shape[0]-100,50):
                                buffer.add(Batch({'obs':obs_stack[j:j+100],'act':np.array(action_lst[j+99]),'rew':np.array(rewards_lst[j+99]),'terminated': np.array(dones_lst[j+99]).reshape(-1,1),'truncated': np.array(dones_lst)[j+99].reshape(-1,1)}))

                            # now add the last bit obs_stack.shape[0]%100 to the buffer
                            if obs_stack.shape[0]%100>0:
                                buffer.add(Batch({'obs':obs_stack[-100:],'act':np.array(action_lst[-1]),'rew':np.array(rewards_lst[-1]),'terminated': np.array(dones_lst[-1]).reshape(-1,1),'truncated': np.array(dones_lst)[-1].reshape(-1,1)}))
                            # buffer.add(Batch({'obs':obs_stack,'act':np.array(action_lst[-1]),'rew':np.array(rewards_lst[-1]),'terminated': np.array(dones_lst[-1]).reshape(-1,1),'truncated': np.array(dones_lst)[-1].reshape(-1,1)}))
                            # buffer.add(Batch({'obs':obs_stack}))
                        else:
                            partial_rollout = True
                        # partial_rollout = True
            
        
            # obs_stack = np.hstack((np.array(obs_lst), np.array(action_lst), np.array(rewards_lst).reshape(-1,1), np.array(dones_lst).reshape(-1,1)))
            # # add the rollout to the buffer
            # buffer.add(Batch({'obs':obs_stack,'act':np.array(action_lst[-1]),'rew':np.array(rewards_lst[-1]),'terminated': np.array(dones_lst[-1]).reshape(-1,1),'truncated': np.array(dones_lst)[-1].reshape(-1,1)}))
            # buffer.add(Batch({'obs':obs_stack}))
        # create buffer with old and new data
        self.buffer.update(buffer)
        self.envsteps+=size*rollout_len
        wandb.log({'environment interactions': self.envsteps})
        return buffer
    
    def run(self, 
            jumpstart:bool = False,):
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
        iterator = range(50,1000,25)
        n_curriculum_epochs = len(iterator)//6
        last_curr_update = 0
        for i in iterator:
            if jumpstart:
                self.gather_buffer(jump_start_len=500-i, size=50)
                wandb.log({"jump start steps (500 - n)": i})
            else:
                self.gather_buffer(size=50)
            wandb.log({"behavorial cloning coefficient": self.bc_coeff})
            self.learn(epoch=cur_epoch, end_epoch=cur_epoch+50)
            n_epochs_tot+=50
            self.bc_coeff *= self.bc_factor
            cur_epoch+=50
            
            if self.curriculum and i-last_curr_update >n_curriculum_epochs: # 
                self.env.update_curriculum()
                last_curr_update = i
        while n_epochs_tot<1000:
            self.learn(epoch=cur_epoch, end_epoch=cur_epoch+50)
            n_epochs_tot+=50
            cur_epoch+=50
            filename = f"TD3BC_Online_TEMP_{self.timestamp}_epoch_{cur_epoch}.pth"
            torch.save(self.model.state_dict(), filename)
            wandb.run.log_artifact(filename, name='policy_streaming', type='model')
            if self.curriculum:
                self.env.update_curriculum()
            self.gather_buffer()
