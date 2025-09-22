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
        self.test_reward = 0

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
            if batch.obs.shape[-1] <(146+4+1+1):    
                observations = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
                actions = torch.tensor(batch.obs[:,:, 18:22]).to(torch.float32).to(self.device)
            else:
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
            t = np.linspace(0,101,100)
            fig, ax = plt.subplots(4, 1)
            for i in range(4):
                ax[i].plot(t,actions.cpu().detach().numpy()[0,:,i], c='g')
                ax[i].plot(t,outputs.cpu().detach().numpy()[0,:,i], c='r')
            
            # plt.show()
            # wandb.log({"img": [wandb.Image(fig, caption=f"Compared to true")]})

        wandb.log({'test reward': avg_rew/n_episodes,'test len': avg_len/n_episodes})
        self.test_reward = avg_rew/n_episodes


    def learn(self, epoch=50):
        loss = np.inf
        for n in tqdm(range(epoch)):
            # if n%20==0:
            #     self.env.update_curriculum()
            wandb.log({"epoch":n})
            losses=[]
            self.model.to(device)
            # print(self.model.device)
            for _ in range(int(len(self.buffer)//self.batch_size)):
                self.model.preprocess.reset(current_epoch=n, last_test_rew=self.test_reward)
                batch = self.buffer.sample(self.batch_size)[0]
                if batch.obs.shape[-1] <(146+4+1+1):
                    observations = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
                    actions = torch.tensor(batch.obs[:,:, 18:22]).to(torch.float32).to(self.device)
                    # clip all actions between -1 and 1
                    actions = torch.clamp(actions, -1, 1)
                else:
                    observations = torch.tensor(batch.obs[:,:, :18],dtype=torch.float32).to(self.device)
                    actions = torch.tensor(batch.obs[:,:, 146:150]).to(torch.float32).to(self.device)
                # add noise
                noise = torch.randn(size=observations.shape, device=observations.device) * self.noise
                observations = observations + noise
                

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
                # wandb.log({"loss":loss.item()})
                self.optimizer.step()
            # wandb.log({"epoch_loss":np.mean(losses)})

            # print(np.mean(losses))
            if np.mean(losses)<self.best_epoch_loss:
                self.best_epoch_loss = np.mean(losses)
                torch.save(self.model.state_dict(), "model_bc.pth")
                wandb.run.log_artifact("model_bc.pth", name='policy_streaming', type='model')
            if n%10==0:
                self.test(viz=True)
                
        
    
