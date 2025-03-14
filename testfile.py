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
action_range_01 = False
rew_avg = 0
total_crashes = 0
for _ in range(25):
    rew_total = 0


    for i in range(1000):
        obs = torch.tensor(obs)
        action = model(obs)
        if action_range_01:
            action = (action + 1) / 2
        # print(action)
        obs, rewards, dones,_, info = env.step(action.detach().numpy()) 
        rew_total += rewards
        # print(rewards)
        if dones:
            total_crashes += 1
            obs = env.reset()[0]
            print("Crashed at step", i)
        obs_lst.append(obs)
        action_lst.append(action.detach().numpy())
    print("Total reward:", rew_total)
    rew_avg += rew_total
print("Final avg reward:", rew_avg/25)
print("Total crashes:", total_crashes)
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


def plot_vel(observations):
    '''Plot the velocity of the drone'''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1)
    for i in range(3):
        axs[i].plot(observations[:,12+i])
        axs[i].set_ylabel(f"Velocity {i}")
    plt.show()


def plot_rot(observations):
    '''Plot the rotation of the drone'''
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4,1)
    for i in range(4):
        axs[i].plot(observations[:,6+i])
        axs[i].set_ylabel(f"Rotation {i}")
    plt.show()
    
plot_pos(np.array(obs_lst))
plot_vel(np.array(obs_lst))
