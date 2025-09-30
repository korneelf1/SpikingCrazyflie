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
                 actor_obs_size:int = 18, 
                 priveliged_obs_size:int = 146,
                 action_size:int = 4,
                 batch_size:int = 256, 
                 warmup:int = 50, 
                 slicing_interval:int = 25,
                 sequence_length:int = 100,
                 device:str = 'cpu', 
                 controller:nn.Module|None = None, 
                 curriculum:bool = False,
                 bc_val:float = 0.2,
                 bc_factor:float = 0.95,
                 jumpstart_only_for_warmup:bool = False,
                 wandb_run=None, 
                 policy_noise:float = 0.2,
                 noise_clip:float = 0.5,
                 alpha:float = 5):
        self.env = env
        self.device = device
        self.actor = model
        self.actor_obs_size = actor_obs_size
        self.priveliged_obs_size = priveliged_obs_size
        self.action_size = action_size
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        self.optimizer = optimizer
        self.buffer = buffer
        self.wandb_run = wandb_run
        self.slicing_interval = slicing_interval
        self.sequence_length = sequence_length
        if "terminated" in buffer._meta:
            self.term_size = buffer.terminated.shape
            # if terminated, truncated, dones and actions are 3 dims, squash to 2
            if len(buffer._meta.terminated.shape) == 3:
                buffer._meta.terminated = buffer.terminated[:,0,0]
            if len(buffer._meta.truncated.shape) == 3:
                buffer._meta.truncated = buffer.truncated[:,0,0]
            if len(buffer._meta.done.shape) == 3:
                buffer._meta.done = buffer.done[:,0,0]
            if len(buffer._meta.act.shape) == 3:
                buffer._meta.act = buffer.act[:,0]

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

        self.alpha = alpha
        self._alpha = alpha
        self.gamma = 0.99
        self.tau = 0.005
        self._freq = 1
        self._cnt = 0
        self.controller = controller
        # create deep copies of the critic networks
        self.critic1_old = deepcopy(critic1).to(device)
        self.critic2_old = deepcopy(critic2).to(device)
        self.actor_old = deepcopy(model).to(device)
        self.bc_coeff = bc_val
        self.bc_factor = bc_factor
        self.jumpstart_only_for_warmup = jumpstart_only_for_warmup

        self.last_test_rew = 0 # used for adaptive scheduling of slopes
        self.best_test = 0
        self.envsteps = 0
        self.epoch = 0

        # TD3 adds noise to the target_q actions and evaluates using CURRENT policy! Seems to improve training
        self.policy_noise = policy_noise 
        self.noise_clip = noise_clip


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
                    obs = torch.tensor(obs,device=self.device, dtype=torch.float32)
                    action = self.controller(obs)
                    self.actor(obs[:self.actor_obs_size]) # warmup the model
                else:
                    obs = torch.tensor(obs,device=self.device, dtype=torch.float32)
                    action = self.actor(obs[:self.actor_obs_size])
                    action = action.reshape(self.action_size,)
                
                actions.append(action.detach().cpu())
                obs, rew, done, term, info = self.env.step(np.array(action.detach().cpu()))
                done = done or term
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
            self.wandb_run.log({"img": [wandb.Image(fig, caption=f"BC Learning")]})
            batch = self.buffer.sample(1)[0]
            observations = torch.tensor(batch.obs[:,:, :self.actor_obs_size],dtype=torch.float32).to(self.device)
            actions = torch.tensor(batch.obs[:,:, self.actor_obs_size:self.actor_obs_size+self.action_size]).to(torch.float32).to(self.device)
            
            outputs = []
            # hidden = None
            # print(self.actor.device)
            for t in range(observations.shape[1]):
                mu = self.actor(observations[:, t])
                output = mu # actor
                outputs.append(output)

            outputs = torch.stack(outputs, dim=1).to(torch.float32)
            t = np.linspace(0,502,501)
            fig, ax = plt.subplots(4, 1)
            for i in range(4):
                ax[i].plot(t,actions.cpu().detach().numpy()[0,:,i], c='g')
                ax[i].plot(t,outputs.cpu().detach().numpy()[0,:,i], c='r')
            
            # plt.show()
            self.wandb_run.log({"img": [wandb.Image(fig, caption=f"Compared to true")]})

        self.last_test_rew = avg_rew/n_episodes
        self.wandb_run.log({'test reward': self.last_test_rew,'test len': avg_len/n_episodes})
        if self.last_test_rew > self.best_test:
            self.best_test = self.last_test_rew
            filename = f"TD3BC_Online_TEMP_{self.timestamp}.pth"
            checkpoint_path = save_checkpoint(self.actor.state_dict(), filename)
            self.wandb_run.log_artifact(checkpoint_path, name='policy_streaming', type='model')

    def compute_returns(self, batch, actions_current, use_next_actions=False):
        '''Compute returns from rewards using discounted rewards with bootstrapping.
        
        Args:
            batch: Batch containing obs, obs_next, rew, done
            actions_current: Actions for current states (buffer actions)
            use_next_actions: If False, compute next actions from current policy
        '''
        gamma = self.gamma
        obs_next = batch.obs_next     # [B, T, obs_dim]
        rewards = batch.rew           # [B, T]
        dones = batch.done            # [B, T]

        B, T, obs_dim = obs_next.shape
        obs_next_flat = obs_next.reshape(B * T, obs_dim).to(self.device)

        with torch.no_grad():
            # Compute NEXT actions using current policy (for bootstrapping)
            next_actions = []
            for t in range(T):
                action = self.actor_old(obs_next[:, t, :self.actor_obs_size])
                next_actions.append(action)
            next_actions = torch.stack(next_actions, dim=1)
            next_actions_flat = next_actions.reshape(B * T, -1)
            
            # Policy smoothing
            noise = (
                torch.randn_like(next_actions_flat) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions_flat = (next_actions_flat + noise).clamp(-1, 1)

            # Target Q using NEXT state and NEXT actions
            q1_target = self.critic1_old(obs_next_flat, next_actions_flat).view(B, T)
            q2_target = self.critic2_old(obs_next_flat, next_actions_flat).view(B, T)
            q_target = torch.min(q1_target, q2_target)

            # Bellman backup: r_t + γ * Q(s_{t+1}, π(s_{t+1}))
            returns = rewards.to(self.device) + gamma * (1 - dones.to(self.device)) * q_target

        return returns
    
    def _mse_optimizer(
            self,
        batch,
        critic: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs.reshape(-1,self.priveliged_obs_size), batch.act.reshape(-1,self.action_size)).flatten()
        target_q = batch.returns.flatten()
        td = (current_q - target_q.detach()).detach()  # TD error for logging/prioritization only

        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (current_q - target_q.detach()).pow(2).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        return td, critic_loss
    
    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters(), strict=True):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        self.soft_update(self.actor_old, self.actor, self.tau)

    # def learn_batch(self, 
    #                 batch,):
    #     if batch.obs.shape[1]%self.sequence_length!=0:
    #         # print("Batch length not divisible by length, cutting original batch")
    #         batch.obs = batch.obs[:,:-(batch.obs.shape[1]%self.sequence_length)]
        
            
    #     # create batch from first observations
    #     batch_size = batch.obs.shape[0]*batch.obs.shape[1]//self.sequence_length
    #     observations = batch.obs[:,:, :self.priveliged_obs_size]
    #     actions = batch.obs[:,:, self.priveliged_obs_size:self.priveliged_obs_size+self.action_size]
    #     rewards = batch.obs[:,:, self.priveliged_obs_size+self.action_size]
    #     terminated = batch.obs[:,:, self.priveliged_obs_size+self.action_size+1]
    #     observations_next = np.hstack((batch.obs[:,1:, :self.priveliged_obs_size], np.zeros((batch.obs.shape[0],1, self.priveliged_obs_size))),dtype=np.float32)

        

    #     # compute returns
    #     # modified batch
    #     batch = Batch(
    #         obs=observations,
    #         act=actions,
    #         rew=rewards,
    #         done=terminated,
    #         obs_next=observations_next,
    #         returns=np.empty((1,),dtype=np.float32)
    #     )
    #     batch = to_torch_as(batch,torch.zeros((1,),device=self.device,dtype=torch.float32))
    #     # recompute actions
    #     actions_recomputed = []
    #     for t in range(observations.shape[1]):
    #         action = self.actor(observations[:, t, :self.actor_obs_size])
    #         actions_recomputed.append(action)
    #     actions_recomputed = torch.stack(actions_recomputed, dim=1)

    #     compute_returns = self.compute_returns(batch, actions_recomputed)
    #     batch.returns = compute_returns


    #     # learn critics
    #     td1, critic_loss = self._mse_optimizer(
    #         batch, self.critic1, self.critic1_optimizer
    #     )
    #     td2, critic2_loss = self._mse_optimizer(
    #         batch, self.critic2, self.critic2_optimizer
    #     )
    #     batch.weight = (td1 + td2) / 2.0  # prio-buffer
    #     # actor
    #     if self._cnt % self._freq == 0:
    #         self._cnt+=1
    #         self.optimizer.zero_grad()
    #         # hidden = None
    #         # print(self.actor.device)
            
    #         # NOTE moved recomputing outputs to compute_returns
    #         # priv_obs = batch.obs[:,:, :self.actor_obs_size].clone().detach().requires_grad_(True).to(self.device).to(torch.float32)
    #         # for t in range(observations.shape[1]):
    #         #     mu = self.actor(priv_obs[:, t])
    #         #     output = mu # actor
    #         #     outputs.append(output)

    #         # act = torch.stack(outputs, dim=1)
                
    #         q_value = self.critic1(batch.obs.reshape(-1,self.priveliged_obs_size), actions_recomputed.reshape(-1,self.action_size)).reshape(-1,self.sequence_length)
    #         # after warmup
    #         q_value = q_value[:,self.warmup:]
    #         act = batch.act[:,self.warmup:]
    #         lmbda = self._alpha / q_value.abs().mean().detach()
    #         actor_loss = -lmbda * q_value.mean() + self.bc_coeff*F.mse_loss(
    #             act, to_torch_as(actions_recomputed[:,self.warmup:], actions_recomputed)
    #         )
    #         actor_loss.backward()
    #         self._last = actor_loss.item()
    #         self.optimizer.step()

    #         self.sync_weight()
            
    #         self.wandb_run.log({"actor_loss": actor_loss.item()})
    #         self.wandb_run.log({"critic_loss": critic_loss.item()})
    #         self.wandb_run.log({"critic2_loss": critic2_loss.item()})
    def learn_batch(self, batch):
        if batch.obs.shape[1] % self.sequence_length != 0:
            batch.obs = batch.obs[:, :-(batch.obs.shape[1] % self.sequence_length)]
        
        observations = torch.tensor(batch.obs[:, :, :self.priveliged_obs_size], device=self.device, dtype=torch.float32)
        batch.act = actions = torch.tensor(batch.obs[:, :, self.priveliged_obs_size:self.priveliged_obs_size+self.action_size], device=self.device, dtype=torch.float32)
        batch.rew = rewards = torch.tensor(batch.obs[:, :, self.priveliged_obs_size+self.action_size], device=self.device, dtype=torch.float32)
        batch.done = terminated = torch.tensor(batch.obs[:, :, self.priveliged_obs_size+self.action_size+1], device=self.device, dtype=torch.float32)
        batch.obs_next = observations_next = torch.hstack((observations[:, 1:, :self.priveliged_obs_size], 
                                    torch.zeros((observations.shape[0], 1, self.priveliged_obs_size), dtype=torch.float32)))
        batch.obs = observations
        batch.act = actions
        batch.rew = rewards
        batch.done = terminated
        batch.obs_next = observations_next
        # batch = Batch(
        #     obs=observations,
        #     act=actions,  # Buffer actions
        #     rew=rewards,
        #     done=terminated,
        #     obs_next=observations_next,
        #     returns=np.empty((1,), dtype=np.float32)
        # )
        # batch = to_torch_as(batch, torch.zeros((1,), device=self.device, dtype=torch.float32))
        
        # target actions with clipped noise (policy smoothing)
        with torch.no_grad():
            target_actions_next = []
            for t in range(batch.obs.shape[1]):
                action_next = self.actor_old(observations_next[:, t, :self.actor_obs_size])
                noise = torch.randn_like(action_next) * self.policy_noise
                if self.noise_clip > 0.0:
                    noise = noise.clamp(-self.noise_clip, self.noise_clip)
                action_next += noise
                # action_next = action_next
                target_actions_next.append(action_next)
            target_actions_next = torch.stack(target_actions_next, dim=1)
            # Compute returns - this will compute next actions internally
            # compute_returns = self.compute_returns(batch, batch.act)
            # batch.returns = compute_returns
            target_q1 = self.critic1_old(observations_next.reshape(-1, self.priveliged_obs_size), target_actions_next.reshape(-1, self.action_size))
            target_q2 = self.critic2_old(observations_next.reshape(-1, self.priveliged_obs_size), target_actions_next.reshape(-1, self.action_size))
            target_q = torch.min(target_q1, target_q2).reshape(-1, self.sequence_length)
        batch.returns = rewards + self.gamma * (1 - terminated) * target_q
        
        # === CRITIC UPDATE ===
        td1, critic_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optimizer
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optimizer
        )
        batch.weight = (td1 + td2) / 2.0
        
        # === ACTOR UPDATE ===
        if self._cnt % self._freq == 0:
            self.optimizer.zero_grad()
            
            # Compute current actions with gradients
            current_actions = []
            for t in range(batch.obs.shape[1]):
                action = self.actor(batch.obs[:, t, :self.actor_obs_size])
                current_actions.append(action)
            current_actions = torch.stack(current_actions, dim=1)
            
            q_value = self.critic1(
                batch.obs.reshape(-1, self.priveliged_obs_size), 
                current_actions.reshape(-1, self.action_size)
            ).reshape(-1, self.sequence_length)
            
            q_value = q_value[:, self.warmup:]
            act = batch.act[:, self.warmup:]
            current_actions_train = current_actions[:, self.warmup:]
            
            lmbda = self._alpha / q_value.abs().mean().detach()
            bc_loss = F.mse_loss(current_actions_train, act)
            actor_loss = lmbda * bc_loss - q_value.mean()
            # actor_loss = -lmbda * q_value.mean() + self.bc_coeff * F.mse_loss(
            #     current_actions_train, act
            # )
            actor_loss.backward()
            self._last = actor_loss.item()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()
            
            self.sync_weight()
            
            self.wandb_run.log({"actor_loss": actor_loss.item()})
            self.wandb_run.log({"critic_loss": critic_loss.item()})
            self.wandb_run.log({"critic2_loss": critic2_loss.item()})
        
        self._cnt += 1
    
    def learn(self, 
              epoch:int = 0, 
              end_epoch:int = 10):
        loss = np.inf
        self.epoch += end_epoch
        for n in tqdm(range(epoch, end_epoch)):
            self.wandb_run.log({"epoch":n})
            losses=[]
            self.actor.to(self.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                self.actor.reset(current_epoch = n,
                                            last_test_rew = self.last_test_rew) # pass last test reward and current epoch to reset (used for adaptive scheduling and interval based scheduling  of slopes, respectively)
                batch = self.buffer.sample(self.batch_size)[0]
                
                self.learn_batch(batch)
 
            if n%3==0:
                # self.sync_weight()
                self.test(viz=False)

    def gather_buffer(self, 
                      name:str ='l2f_controller_buffer_in_ru ', 
                      size:int = 2500, 
                      rollout_len:int = 501, 
                      jump_start_len:int|None = None,
                      keep_og:bool = False,):
        # print("Gathering buffer...")
        # at least batch_size or size * (rollout_len/100)*2
        # gather new buffer - size should be large enough to hold all rollout data
        samples_per_rollout = int(rollout_len/self.sequence_length)*self.sequence_length/self.slicing_interval # best case scenario
        samples_per_rollout_worst = self.sequence_length/self.slicing_interval # worst case scenario
        n_rollouts = max(np.ceil(self.batch_size/samples_per_rollout_worst), np.ceil(size / samples_per_rollout_worst))

        buffer_size = self.batch_size if self.batch_size > size + len(self.buffer) else size
        buffer = ReplayBuffer(size=buffer_size) # each sample is 100 in length
        js = jump_start_len if jump_start_len is not None else 0
        # for _ in tqdm(range(int(n_rollouts)), desc="Gathering buffer"):
        # pbar = tqdm(total=buffer_size, desc="Gathering buffer")
        safety_counter = 0
        while len(buffer) <= buffer.maxsize-1 and safety_counter<size*self.batch_size:
            safety_counter+=1
            # print(f"Gathering buffer... {len(buffer)}/{buffer.maxsize}")
            # assure that we get usable sequences, by discarding rollouts that crash too fast or all the way at the end
            partial_rollout = True
            
            # make sure to gather full rollouts
            # if len(buffer) >= buffer_size:
            #     print("Buffer is full, stopping rollout")
            #     break
            while partial_rollout:
                # creates lists
                obs_lst = []
                action_lst = []
                rewards_lst = []
                dones_lst = []
                # obs_next_lst = []

                returns = []
                obs = self.env.reset()[0]
                partial_rollout = False # assume full rollout unless dones before end of range
                t_warmup = 0 # timestep in rollout (used after crash)
                i = 0
                dones = False
                while i < rollout_len and not dones:
                    
                    obs = torch.tensor(obs,device=self.device, dtype=torch.float32)
                    if t_warmup<max(self.warmup,js):
                        t_warmup+=1
                        action = self.controller(obs)
                        _ = self.actor(obs[:self.actor_obs_size])
                    else:
                        action = self.actor(obs[:self.actor_obs_size])
                        # reshape in place
                        action = action.reshape(self.action_size,)
                        # add noise
                        noise = torch.randn_like(action) * self.policy_noise
                        if self.noise_clip > 0.0:
                            noise = noise.clamp(-self.noise_clip, self.noise_clip)
                        action += noise
                        action = action.clamp(-2, 2)

                    obs_lst.append(obs.cpu().numpy())
                    obs, rewards, dones,_, info = self.env.step(action.cpu().detach().numpy()) 

                    # obs_next_lst.append(obs)
                    action_lst.append(action.cpu().detach().numpy().reshape(self.action_size,))
                    rewards_lst.append(rewards)
                    dones_lst.append(dones)
                    i += 1
     
                obs = self.env.reset()[0]

                t_warmup = 0
                # if our rollout crashses before 2x warmup, we discard it, it is not worth to warmup the model for only a few timesteps
                # if our rollout crashes after rolloutlen - 2x warmup, we also wouldnt have at least the warmup length to trian on
                if (self.sequence_length<i):
                    obs_stack = np.hstack((np.array(obs_lst), np.array(action_lst), np.array(rewards_lst).reshape(-1,1), np.array(dones_lst).reshape(-1,1)))
                    partial_rollout = False
                    # once stacked, remove original lists to save memory
                    obs_lst.clear()
                    action_lst.clear()
                    rewards_lst.clear()
                    dones_lst.clear()
                    # obs_next_lst.clear()
                    
                    # add the rollout to the buffer
                    # chop up in sequence_length step sequences with step size of 50 (0-sequence_length,50-150,sequence_length-200,150-250,...)
                    for j in range(0,obs_stack.shape[0]-self.sequence_length,self.slicing_interval):
                        # Extract data from obs_stack: obs (0:146), actions (146:150), rewards (150:151), dones (151:152)
                        # NOTE only obs_stack is used, rest is saved for compatibility purposes
                        buffer.add(Batch({
                            'obs': obs_stack[j:j+self.sequence_length],
                            'act': obs_stack[j:j+self.sequence_length, self.priveliged_obs_size:self.priveliged_obs_size+self.action_size][-1],
                            'rew': float(obs_stack[j:j+self.sequence_length, self.priveliged_obs_size+self.action_size:self.priveliged_obs_size+self.action_size+1][-1].item()),
                            'terminated': bool(obs_stack[j:j+self.sequence_length, self.priveliged_obs_size+self.action_size+1:self.priveliged_obs_size+self.action_size+2][-1].item()),
                            'truncated': bool(obs_stack[j:j+self.sequence_length, self.priveliged_obs_size+self.action_size+1:self.priveliged_obs_size+self.action_size+2][-1].item())
                        }))
                        # pbar.update(1)

                    # now add the last bit obs_stack.shape[0]%self.sequence_length to the buffer
                    if obs_stack.shape[0]%self.sequence_length>0:
                        # Extract data from the last self.sequence_length elements of obs_stack
                        buffer.add(Batch({
                            'obs': obs_stack[-self.sequence_length:],
                            'act': obs_stack[-self.sequence_length:, self.priveliged_obs_size:self.priveliged_obs_size+self.action_size][-1],
                            'rew': float(obs_stack[-self.sequence_length:, self.priveliged_obs_size+self.action_size:self.priveliged_obs_size+self.action_size+1][-1].item()),
                            'terminated': bool(obs_stack[-self.sequence_length:, self.priveliged_obs_size+self.action_size+1:self.priveliged_obs_size+self.action_size+2][-1].item()),
                            'truncated': bool(obs_stack[-self.sequence_length:, self.priveliged_obs_size+self.action_size+1:self.priveliged_obs_size+self.action_size+2][-1].item())
                        }))
                        # pbar.update(1)
                    self.envsteps+=obs_stack.shape[0]
                    self.wandb_run.log({'environment interactions': self.envsteps})
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
                    obs = self.env.reset()[0]
                    t_warmup = 0 # timestep in rollout (used after crash)
                            
                    

        # create buffer with old and new data
        self.buffer.update(buffer)
        # pbar.close()
        
        return buffer
    
    def run(self, 
            jumpstart:bool = False,
            n_samples_per_gather:int = 200,
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
        filename = f"TD3BC_Online_TEMP_{self.timestamp}_epoch_{cur_epoch}.pth"
        checkpoint_path = save_checkpoint(self.actor.state_dict(), filename)
        # self.gather_buffer(jump_start_len=490, size=1000)
        n_epochs_tot = 0
        iterator = range(0,max_epochs,epochs_per_gather)
        factor_i = 500/0.8/max_epochs # we want to be fully relying on the model by 80 percent of the end of the training
        # Update curriculum every 6 training cycles (more intuitive than len(iterator)//6)
        curriculum_interval = 15
        curriculum_update_count = 0
        for i in iterator:
            if jumpstart and not self.jumpstart_only_for_warmup:
                self.gather_buffer(jump_start_len=500-i*factor_i, size=n_samples_per_gather)
                self.wandb_run.log({"jump start steps (500 - n)": i})
            elif self.jumpstart_only_for_warmup:
                self.gather_buffer(jump_start_len=100, size=n_samples_per_gather)
                self.wandb_run.log({"jump start steps (100)": 100})
            else:
                self.gather_buffer(size=n_samples_per_gather)
            self.wandb_run.log({"behavorial cloning coefficient": self.bc_coeff})
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
            checkpoint_path = save_checkpoint(self.actor.state_dict(), filename)
            # self.wandb_run.log_artifact(checkpoint_path, name='policy_streaming', type='model')
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
    def __init__(self,model , size=128, action_size=4,stoch=False):
        super().__init__()
        self.preprocess = model
        self.mu = nn.Linear(size,action_size)
        self.sigma = nn.Linear(size,action_size)
        if stoch:
            self.dist = dist_fion
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.preprocess(x)
        if isinstance(x, tuple):
            x = x[0]    #spiking
        if hasattr(self, 'dist'):
            return self.dist(nn.Tanh()(self.mu(x)), self.sigma(x)).rsample()
        return nn.Tanh()(self.mu(x))
    def reset(self, **kwargs):
        if hasattr(self.preprocess, 'reset'):
            self.preprocess.reset(**kwargs)
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
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--buffer-size", type=int, default=20000, help="Buffer size")
        parser.add_argument("--buffer-preload", action='store_true', help="Buffer preload")

        parser.add_argument("--alpha", type=float, default=2.5)
        parser.add_argument("--exploration-noise", type=float, default=0.1)
        parser.add_argument("--policy-noise", type=float, default=0.1)
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
        parser.add_argument("--epochs_per_gather", type=int, default=20, help="Epochs per gather")
        parser.add_argument("--slope", type=int, default=2, help="Slope value")
        parser.add_argument("--slope_schedule", type=str, default='adaptive')
        parser.add_argument("--scheduling_order", type=int, default=3)
        parser.add_argument("--bc-factor", type=float, default=0.99, help="Behavioral cloning factor")
        
        parser.add_argument("--bc-val", type=float, default=0.2, help="Behavioral cloning factor")
        parser.add_argument("--jumpstart_only_for_warmup", action='store_true', help="JumpStartScheduling only for warmup")
        # Use 'store_true' for interval if you want it as a flag, or use 'type=int' if it's an integer
        parser.add_argument("--interval", type=int, default=1, help="Interval flag")
        parser.add_argument("--ablation", type=str, default=None)
        parser.add_argument("--n_rollouts_per_gather", type=int, default=10, help="Number of rollouts per gather")
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
    print("Jumpstart only for warmup:",args.jumpstart_only_for_warmup)
    args.jumpstart = True
    wandb.config.update({"device":device})
    wandb.config.update({"slope":args.slope})
    wandb.config.update({"surrogate_scheduling":args.surrogate_scheduling})
    wandb.config.update({"hidden_sizes":args.hidden_sizes})
    wandb.config.update({"curriculum":args.curriculum})
    wandb.config.update({"bc_factor":args.bc_factor})
    wandb.config.update({"jumpstart":args.jumpstart})
    wandb.config.update({"ablation":args.ablation})
    wandb.config.update({"jumpstart_only_for_warmup":args.jumpstart_only_for_warmup})
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
                                max_slope=20,
                                verbose=True).to(device)
    
    # Initialize the wrapper
    model = Wrapper(spiking_module, size=args.hidden_sizes[-1]).to(device)
    # model.load_state_dict(torch.load("TD3BC_Online_TEMP.pth", map_location="cpu"))

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

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
                batch_size=args.batch_size , device=device,
                curriculum=args.curriculum,
                bc_val=args.bc_val,
                bc_factor=args.bc_factor,
                jumpstart_only_for_warmup=args.jumpstart_only_for_warmup,
                wandb_run=wandb.run,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip)

    # learn the model
    trainer.run(jumpstart=args.jumpstart, 
    n_samples_per_gather=args.n_rollouts_per_gather,
    max_epochs=args.max_epochs,
    epochs_per_gather=args.epochs_per_gather)
    
    wandb.run.finish()
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    save_checkpoint(model.state_dict(), f'TD3BC_ONLINE_STABLE_{timestamp}.pth')