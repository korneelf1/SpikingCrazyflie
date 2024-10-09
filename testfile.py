from l2f_agent import ConvertedModel
import torch
from l2f_gym import Learning2Fly
import numpy as np

model = ConvertedModel()


env = Learning2Fly()
print(env.action_space)
print(env.observation_space)

model.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))


obs = env.reset()[0]

obs_lst = []
action_lst = []

for i in range(1000):
    obs = torch.tensor(obs)
    action = model(obs)
    # print(action)
    obs, rewards, dones,_, info = env.step(action.detach().numpy()) 
    
    if dones:
        
        obs = env.reset()[0]
        print("Crashed at step", i)
    obs_lst.append(obs)
    action_lst.append(action.detach().numpy())
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

# plot the observations
def plot_pos(observations):
    '''Plot the position of the drone'''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig, axs = plt.subplots(3,1)
    for i in range(3):
        axs[i].plot(observations[:,i])
        axs[i].set_ylabel(f"Pos {i}")
    plt.show()
    plt.show()
plot_pos(np.array(obs_lst))
def plot_vel(observations):
    '''Plot the velocity of the drone'''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1)
    for i in range(3):
        axs[i].plot(observations[:,3+i])
        axs[i].set_ylabel(f"Velocity {i}")
    plt.show()
plot_vel(np.array(obs_lst))
def plot_rot(observations):
    '''Plot the rotation of the drone'''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4,1)
    for i in range(4):
        axs[i].plot(observations[:,6+i])
        axs[i].set_ylabel(f"Rotation {i}")
    plt.show()
    