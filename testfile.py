from l2f_agent import ConvertedModel
import torch
from l2f_gym import Learning2Fly
import numpy as np

model = ConvertedModel()


env = Learning2Fly(action_history=True, quaternions_to_obs_matrices=True)
print(env.action_space)
print(env.observation_space)

model.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))


obs = env.reset()[0]

obs_lst = []
action_lst = []

for i in range(1000):
    obs = torch.tensor(obs)
    action = model(obs)
    obs, rewards, dones,_, info = env.step(action.detach().numpy()) 
    
    if dones:
        
        obs = env.reset()[0]
        print("Crashed at step", i)
    obs_lst.append(obs)
    action_lst.append(action)
# plot the actions
import matplotlib.pyplot as plt
import numpy as np
actions = np.array(action_lst)
plt.subplots(4,1,figsize=(10,10))
for i in range(4):
    plt.subplot(4,1,i+1)
    plt.plot(actions[:,i])
    plt.ylabel(f"Action {i}")
plt.show()
