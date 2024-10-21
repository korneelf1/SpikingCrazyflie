# behaverioral cloning
import numpy as np
import torch
from tianshou.data import ReplayBuffer
import matplotlib.pyplot as plt

from tqdm import tqdm
class BC:
    def __init__(self, env, model, optimizer, buffer, batch_size=256, warmup=50, device='cpu', noise=0.):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.batch_size = batch_size
        self.warmup = warmup
        self.best_epoch_loss = np.inf
        self.loss_fn = torch.nn.MSELoss()
        self.device= device
        self.noise = noise

    def test(self, n_episodes=20,viz=False):
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
                # add noise
                noise = torch.randn(size=obs.shape, device=obs.device) * self.noise
                obs = obs + noise
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
            # wandb.log({"img": [wandb.Image(fig, caption=f"Compared to true")]})

        wandb.log({'test reward': avg_rew/n_episodes,'test len': avg_len/n_episodes})


    def learn(self, epoch=50):
        loss = np.inf
        for n in tqdm(range(epoch)):
            if n%20==0:
                self.env.update_curriculum()
            wandb.log({"epoch":n})
            losses=[]
            self.model.to(device)
            # print(self.model.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                self.model.preprocess.reset(current_epoch=n)
                batch = self.buffer.sample(self.batch_size)[0]
                observations = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
                # add noise
                noise = torch.randn(size=observations.shape, device=observations.device) * self.noise
                observations = observations + noise
                actions = torch.tensor(batch.obs[:,:, 146:150]).to(torch.float32).to(self.device)

                self.optimizer.zero_grad()
                outputs = []
                # hidden = None
                # print(self.model.device)
                for t in range(observations.shape[1]):
                    mu = self.model(observations[:, t])
                    output = mu # actor
                    outputs.append(output)

                outputs = torch.stack(outputs, dim=1).to(torch.float32)
                loss = self.loss_fn(outputs[:,self.warmup:].flatten().to(self.device), actions[:,self.warmup:].flatten().to(self.device)).to(torch.float32)
                loss.backward()
                losses.append(loss.item())
                wandb.log({"loss":loss.item()})
                self.optimizer.step()
            wandb.log({"epoch_loss":np.mean(losses)})

            # print(np.mean(losses))
            if np.mean(losses)<self.best_epoch_loss:
                self.best_epoch_loss = np.mean(losses)
                torch.save(self.model.state_dict(), "model_bc.pth")
                wandb.run.log_artifact("model_bc.pth", name='policy_streaming', type='model')
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
        parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[32,32])
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
        # Use 'store_true' or 'store_false' for boolean flags
        parser.add_argument("--surrogate-scheduling", action='store_true', help="Enable surrogate scheduling")
        parser.add_argument("--slope", type=int, default=2, help="Slope value")
        
        # Use 'store_true' for interval if you want it as a flag, or use 'type=int' if it's an integer
        parser.add_argument("--interval", type=int, default=1, help="Interval flag")
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
        def __init__(self, model, size=128):
            super().__init__()
            self.preprocess = model
            self.mu = nn.Linear(size,4)
            self.sigma = nn.Linear(size,4)
        def forward(self, x):
            x = self.preprocess(x)
            if isinstance(x, tuple):
                x = x[0]    #spiking
            return nn.Tanh()(self.mu(x))
        def reset(self):
            if hasattr(self.preprocess, 'reset'):
                self.preprocess.reset()
        def to_cuda(self):
            return self.preprocess.to(device)
    
    



    # prepare the data
    buffer = ReplayBuffer.load_hdf5('l2f_controller_buffer.hdf5')
    
    env = Learning2Fly(fast_learning=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    wandb_args = {"spiking":True, 'Slope': args.slope,'Schedule': args.surrogate_scheduling, 'Algo':'BC', 'fast_learning':False}
    wandb.init(project="l2f_bc", config=wandb_args)
    # wandb.init(mode="disabled")

    wandb.define_metric("*", step_metric="epoch")
    print('Device in use:',device)
    print("Initial slope:",args.slope)
    print("Surrogate scheduling:",args.surrogate_scheduling)
    print("Hidden sizes:",args.hidden_sizes)
    print("Policy Noise:",args.policy_noise)
    wandb.config.update({'slope':args.slope, 'surrogate_scheduling':args.surrogate_scheduling,'hidden_sizes':args.hidden_sizes, 'policy_noise':args.policy_noise})
    spiking_module = SpikingNet(state_shape=18, action_shape=args.hidden_sizes[-1], hidden_sizes=args.hidden_sizes[:-1], device=device,slope=args.slope,slope_schedule=args.surrogate_scheduling,reset_interval=args.interval, reset_in_call=False, repeat=1).to(device)
    model = Wrapper(spiking_module, size=args.hidden_sizes[-1]).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # prepare the BC
    bc = BC(env,model, optimizer, buffer, batch_size=250, device=device, noise=args.policy_noise)
    # learn the model
    loss = bc.learn(epoch=300)
    print(loss)
    wandb.run.finish()