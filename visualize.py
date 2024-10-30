from l2f_gym import Learning2Fly
# tianshou code
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.utils.net.continuous import ActorProb, Critic, RecurrentActorProb, RecurrentCritic
from tianshou.utils.net.common import Net
from tianshou.data import VectorReplayBuffer,HERVectorReplayBuffer,PrioritizedVectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import WandbLogger, MultipleLRSchedulers
from tianshou.data.collector import Collector
from tianshou.env import SubprocVectorEnv, DummyVectorEnv

# spiking specific code
# from spiking_gym_wrapper import SpikingEnv
from spikingActorProb import SpikingNet
# from masked_actors import MaskedNet
# 
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from l2f_gym import observe_rotation_matrix
def rotation_to_euler(R):
    '''Convert rotation matrix (flattened) to euler angles'''
    R = R.reshape(3,3)
    if R[2,0] != 1 and R[2,0] != -1:
        theta = -np.arcsin(R[2,0])*180/np.pi
        psi = np.arctan2(R[2,1]/np.cos(theta), R[2,2]/np.cos(theta))*180/np.pi
        if psi<-90:
            psi+=180
        if psi>90:
            psi-=180

        phi = np.arctan2(R[1,0]/np.cos(theta), R[0,0]/np.cos(theta))*180/np.pi
        if phi<-90:
            phi+=180
        if phi>90:
            phi-=180
        # assure everything between -90 and 90
        
    else:
        print('Gimbal lock')
    return np.array([phi,theta,psi])

def quaternion_to_euler(q):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw) in radians.
    
    Args:
    q -- A tuple or list of four elements representing the quaternion (q0, q1, q2, q3)
    
    Returns:
    roll, pitch, yaw -- The Euler angles in radians
    """

    q0, q1, q2, q3 = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # convert to degrees
    roll = roll*180/np.pi
    pitch = pitch*180/np.pi
    yaw = yaw*180/np.pi
    return roll, pitch, yaw


env = Learning2Fly(ez_reset=False, terminal_conditions='testing')

observation_space = env.observation_space.shape
action_space = env.action_space.shape

def create_policy():
    # create the networks behind actors and critics
    net_a = SpikingNet(state_shape=observation_space,
                    hidden_sizes=[128,128],action_shape=128, )
        
    net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[128,128],
                    concat=True,)
    net_c2 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[128,128],
                    concat=True,)
    
    # model_logger.watch(net_a)
    # model_logger.watch(net_c1)
    # model_logger.watch(net_c2)

    # create actors and critics
    actor = ActorProb(
        net_a,
        action_space,
        unbounded=True,
        conditioned_sigma=True,
        
    )
    critic1 = Critic(net_c1, )
    critic2 = Critic(net_c2, )

    # create the optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-4)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-4)

    # create one learning rate scheduler for the 3 optimizers
    lr_scheduler_a = torch.optim.lr_scheduler.StepLR(actor_optim, step_size=1000, gamma=0.5)
    lr_scheduler_c1 = torch.optim.lr_scheduler.StepLR(critic_optim,step_size=1e3, gamma=0.5)
    lr_scheduler_c2 = torch.optim.lr_scheduler.StepLR(critic2_optim,step_size=1e3, gamma=0.5)

    lr_scheduler = MultipleLRSchedulers(lr_scheduler_a,lr_scheduler_c1,lr_scheduler_c2)


    # create the policy
    policy = SACPolicy(actor=actor, actor_optim=actor_optim, \
                        critic=critic1, critic_optim=critic_optim,\
                        critic2=critic2, critic2_optim=critic2_optim,lr_scheduler=lr_scheduler,\
                        action_space=env.action_space,\
                        observation_space=env.observation_space, \
                        action_scaling=True, action_bound_method=None) # make sure actions are scaled properly
    return policy

from torch import nn
def create_policy_td3():
    hidden_sizes = [128, 128]
    net_a = SpikingNet(state_shape=18, hidden_sizes=hidden_sizes[:-1], action_shape=hidden_sizes[-1], repeat=1, reset_in_call=False)
    class Wrapper(nn.Module):
        def __init__(self,model , stoch=False, warmup=50):
            super().__init__()
            self.preprocess = model
            self.mu = nn.Linear(128,4)
            self.sigma = nn.Linear(128,4)
            self.t = 0
            self.warmup = warmup
            if stoch:
                self.dist = torch.distributions.Normal
        def forward(self, x):
            # print(x)
            assert len(x.shape)==2
            if x.shape[1]>18:
                x = x[:,:18]
            while self.t<self.warmup:
                self.t+=1
                _ = self.preprocess(x)
            x = self.preprocess(x)
            if isinstance(x, tuple):
                x = x[0]    #spiking
            if hasattr(self, 'dist'):
                return nn.Tanh()(self.dist(self.mu(x), self.sigma(x)))
            return np.array(nn.Tanh()(self.mu(x)).detach().clone().cpu())
            # return np.array((self.mu(x)).detach().clone().cpu())
        def reset(self):
            self.t = 0
            if hasattr(self.preprocess, 'reset'):
                self.preprocess.reset()
        # def to_cuda(self):
        #     return self.preprocess.to(device)

    actor = Wrapper(net_a, stoch=False, warmup=0)
    return actor
    


policy = create_policy_td3()
# state_dict_full_precision = torch.load('TD3BC_TEMP_original.pth', map_location='cpu')
state_dict_full_precision = torch.load('model_bc_sweet_sunset.pth', map_location='cpu')

# Round all values to 6 decimal places
for key, param in state_dict_full_precision.items():
    state_dict_full_precision[key] = torch.round(param * 10**6) / 10**6

policy.load_state_dict(state_dict=state_dict_full_precision)

# policy.load_state_dict(torch.load('TD3BC_Online_TEMP_strict.pth', map_location='cpu'))
policy.eval()
# print(policy.preprocess.model.lif_in.threshold)
# just pass 0.5 as input to all 18 observations, compare to c code
obs = np.ones((18,))*-.5
obs = torch.tensor(obs).float().reshape(1,-1)
policy.reset()
out_array = np.zeros((250,4))
for i in range(250):
    action = policy(obs)
    out_array[i] = np.array(action)
np.savetxt('hiddenspikes.txt',out_array, delimiter=',')

# np.savetxt('hiddenspikes.txt',policy.preprocess.model.s1_arr, delimiter=',')
print("SAVED")
# print(out_array)
def smooth(x,window_len=11):
    return np.convolve(x, np.ones(window_len)/window_len, mode='same')


def test(target=None, euler=False, length=1000):
    '''Test the policy on the environment
    Target is an array of size 18, which is substraced from the observation before passing to the policy
    Allows for position commands,
    velocity commands, etc.
    defaults to zeros
    if target is shape (1,18) then it is repeated for the entire episode
    if target is shape (1000,18), then it is used for each time step'''
    if target is None:
        target = np.zeros((1,18))
    if target.shape[0] == 1:
        target = np.repeat(target,length,axis=0)
    # target to tensor
    target = torch.tensor(target).float().reshape(length,18)
    # add zeros to make it (100,146)
    target = torch.cat((target,torch.zeros((length,128))),dim=1)
    # run the policy on the environment
    out = []
    obs_out = []
    actions = []
    obs = env.reset()[0]
    quaternions = np.zeros((length,4))
    eulers = np.zeros((length,3))
    policy.reset()
    for i in range(length):
        obs = torch.tensor(obs).float().reshape(1,-1)
        # man_obs_mat = observe_rotation_matrix(env.state.orientation)
        # assert np.allclose(man_obs_mat,obs[:,3:12].detach().numpy(),atol=1e-3)
        # obs[:,3:12] = torch.tensor(man_obs_mat).reshape(1,-1)
        
        obs = obs - target[i].reshape(1,-1)
        # obs[:,:3] = 0
        obs_out.append(obs[:,:18])
        # obs = np.zeros_like(obs)
        # obs[:,3:15] = 0
        # obs[:,3] = 1
        # obs[:,7] = 1
        # obs[:,11] = 1
        quaternions[i] = env.state.orientation
        
        eulers[i] = quaternion_to_euler(quaternions[i])
        action = policy(obs)
        # print(obs[:,:18])
        actions.append(action.reshape(4,))
        # take low pass filter of last 5 actions
        # action_low = actions[-5:]
        # action = np.mean(np.stack(action_low),axis=0)
        # action = torch.tanh(action/50)
        obs, rewards,dones, _, info = env.step(action)
        out.append((obs, rewards, dones, info))
        
        if dones:
            print(i)
            obs = env.reset()[0]
    # save obs and actions to lists and save in file to be used in C code as inputs to function
    obs_out = torch.vstack(obs_out)
    actions = np.array(actions)
    print(obs_out.shape)

    # plot quaternions
    plt.figure()
    plt.suptitle("Quaternions")
    plt.subplot(411)
    plt.plot(quaternions[:,0],label='qw')
    plt.plot(df['stateEstimate.qw'],label='Real life', linestyle='--')
    plt.legend()
    plt.subplot(412)
    plt.plot(quaternions[:,1],label='qx')
    plt.plot(df['stateEstimate.qx'],label='Real life', linestyle='--')
    plt.legend()
    plt.subplot(413)
    plt.plot(quaternions[:,2],label='qy')
    plt.plot(df['stateEstimate.qy'],label='Real life', linestyle='--')
    plt.legend()
    plt.subplot(414)
    plt.plot(quaternions[:,3],label='qz')
    plt.plot(df['stateEstimate.qz'],label='Real life', linestyle='--')
    plt.legend()
    plt.show()
    plt.figure()
    plt.suptitle("Euler Angles")
    plt.subplot(311)
    plt.plot(eulers[:,0],label='phi')
    plt.legend()
    plt.subplot(312)
    plt.plot(eulers[:,1],label='theta')
    plt.legend()
    plt.subplot(313)
    plt.plot(eulers[:,2],label='psi')
    plt.legend()
    plt.show()
    
    # # Save obs and actions to csv files
    np.savetxt('obs_out.csv', obs_out, delimiter=',')
    np.savetxt('actions.csv', actions, delimiter=',')



    obs_means_std = torch.zeros((18,2))
    pos = obs_out[:,:3]
    vel = obs_out[:,12:15]
    pos_mag = torch.norm(pos,dim=1)
    vel_mag = torch.norm(vel,dim=1)
    for i in range(18):
        obs_means_std[i,0] = torch.mean(obs_out[:,i])
        obs_means_std[i,1] = torch.std(obs_out[:,i])
    print(obs_means_std)
    
    print('Mean pos: ', torch.mean(pos_mag))
    print('Mean vel: ', torch.mean(vel_mag))
    print('Std pos: ', torch.std(pos_mag))
    print('Std vel: ', torch.std(vel_mag))
    actions = np.array(actions).reshape(-1,4)
    # plot actions
    # rolling window actions
    actions = (np.array(actions)+1)/2
    window = 25
    # actions = np.convolve(actions, np.ones(window)/window, mode='same') 
    MAX_UINT16 = 65535
    plt.figure()
    plt.subplot(411)
    plt.plot(np.convolve(actions[:,0], np.ones(window)/window, mode='same') ,label='m1')
    plt.plot(df['og_ctrl.m1']/MAX_UINT16,label='Real life', linestyle='--')
    plt.ylim(-0,1)
    plt.legend()
    plt.subplot(412)
    plt.plot(np.convolve(actions[:,1], np.ones(window)/window, mode='same'),label='m2')
    plt.plot(df['og_ctrl.m2']/MAX_UINT16,label='Real life', linestyle='--')
    plt.ylim(-0,1)
    plt.legend()
    plt.subplot(413)
    plt.plot(np.convolve(actions[:,2], np.ones(window)/window, mode='same'),label='m3')
    plt.plot(df['og_ctrl.m3']/MAX_UINT16,label='Real life', linestyle='--')
    plt.ylim(-0,1)
    plt.legend()
    plt.subplot(414)
    plt.plot(np.convolve(actions[:,3], np.ones(window)/window, mode='same'),label='m4')
    plt.plot(df['og_ctrl.m4']/MAX_UINT16,label='Real life', linestyle='--')
    plt.ylim(-0,1)
    plt.legend()
    plt.show()



    xs = np.array(obs_out)
    pos = xs[:,:3]
    quat = xs[:,3:12]
    vel = xs[:,12:15]
    omega = xs[:,15:18]

    # pos_norm = np.linalg.norm(pos,axis=1)
    # quat_norm = np.linalg.norm(quat,axis=1)
    # vel_norm = np.linalg.norm(vel,axis=1)
    # omega_norm = np.linalg.norm(omega,axis=1)

    # use matplotlib to visualize the norm of the 4 vectors of the 3 drones
    plt.figure()
    plt.suptitle("Positions")
    # plot x y z of positions
    plt.subplot(311)
    plt.plot(smooth(pos[:,0],window_len=1),label='x')
    plt.plot(target[:,0],label='target', linestyle='--')
    plt.plot(df['snn_control.posx'],label='Real life', linestyle='--')    
    plt.ylim(-1,1)
    plt.legend()
    plt.subplot(312)
    plt.plot(smooth(pos[:,1]),label='y')
    plt.plot(target[:,1],label='target', linestyle='--')
    plt.plot(df['snn_control.posy'],label='Real life', linestyle='--')
    plt.ylim(-1,1)
    plt.legend()
    plt.subplot(313)
    plt.plot(smooth(pos[:,2]),label='z')
    plt.plot(target[:,2],label='target', linestyle='--')
    plt.plot(df['snn_control.posz'],label='Real life', linestyle='--')
    plt.legend()
    # plt.ylim(-1,1)
    plt.show()

    # plot velocities
    plt.figure()
    plt.suptitle("Velocities")
    plt.subplot(311)
    plt.plot(smooth(vel[:,0]),label='vx')
    plt.plot(target[:,12],label='target', linestyle='--')
    # plt.plot(df['snn_control.VelBodyX'],label='Real life', linestyle='--')
    plt.ylim(-1,1)
    plt.legend()
    plt.subplot(312)
    plt.plot(smooth(vel[:,1]),label='vy')
    plt.plot(target[:,13],label='target', linestyle='--')
    plt.ylim(-1,1)
    plt.legend()
    plt.subplot(313)
    plt.plot(smooth(vel[:,2]),label='vz')
    plt.plot(target[:,14],label='target', linestyle='--')
    plt.ylim(-1,1)
    plt.legend()
    plt.show()
    if euler:
        # convert quaternions to euler angles
        orient1 = df['snn_control.orient_1'].to_numpy()
        orient2 = df['snn_control.orient_2'].to_numpy()
        orient3 = df['snn_control.orient_3'].to_numpy()
        orient4 = df['snn_control.orient_4'].to_numpy()
        orient5 = df['snn_control.orient_5'].to_numpy()
        orient6 = df['snn_control.orient_6'].to_numpy()
        orient7 = df['snn_control.orient_7'].to_numpy()
        orient8 = df['snn_control.orient_8'].to_numpy()
        orient9 = df['snn_control.orient_9'].to_numpy()
        quat_real = np.vstack((orient1,orient2,orient3,orient4,orient5,orient6,orient7,orient8,orient9)).T
        euler = np.zeros((length,3))
        euler_real = np.zeros((length,3))
        for i in range(length):
            euler[i] = rotation_to_euler(quat[i])
            euler_real[i] = rotation_to_euler(quat_real[i])

        # plot euler angles
        plt.figure()
        plt.suptitle("Euler Angles")
        plt.subplot(311)
        plt.plot(euler[:,0],label='phi')
        plt.plot(euler_real[:,0],label='Real life', linestyle='--')
        plt.legend()
        plt.subplot(312)
        plt.plot(euler[:,1],label='theta')
        plt.plot(euler_real[:,1],label='Real life', linestyle='--')
        plt.legend()
        plt.subplot(313)
        plt.plot(euler[:,2],label='psi')
        plt.plot(euler_real[:,2],label='Real life', linestyle='--')
        plt.legend()
        plt.show()
    # plot quaternions
    else:
        orient1 = df['snn_control.orient_1'].to_list()
        orient2 = df['snn_control.orient_2'].to_list()
        orient3 = df['snn_control.orient_3'].to_list()
        orient4 = df['snn_control.orient_4'].to_list()
        orient5 = df['snn_control.orient_5'].to_list()
        orient6 = df['snn_control.orient_6'].to_list()
        orient7 = df['snn_control.orient_7'].to_list()
        orient8 = df['snn_control.orient_8'].to_list()
        orient9 = df['snn_control.orient_9'].to_list()
        plt.figure()
        plt.suptitle("Orientations")
        plt.subplot(911)
        plt.plot(quat[:,0],label='11')
        plt.plot(orient1,label='Real life', linestyle='--')
        # plt.plot(target[:,3],label='target', linestyle='--')
        # plt.ylim(-1,1)
        plt.legend()
        plt.subplot(912)
        plt.plot(quat[:,1],label='12')
        plt.plot(orient2,label='Real life', linestyle='--')
        # plt.plot(target[:,4],label='target', linestyle='--')
        plt.ylim(-1,1)
        plt.legend()
        plt.subplot(913)
        plt.plot(quat[:,2],label='13')
        plt.plot(orient3,label='Real life', linestyle='--')
        # plt.plot(target[:,5],label='target', linestyle='--')
        plt.ylim(-1,1)
        plt.legend()
        plt.subplot(914)
        plt.plot(quat[:,3],label='21')
        plt.plot(orient4,label='Real life', linestyle='--')
        # plt.plot(target[:,6],label='target', linestyle='--')
        plt.ylim(-1,1)
        plt.legend()
        plt.subplot(915)
        plt.plot(quat[:,4],label='22')
        plt.plot(orient5,label='Real life', linestyle='--')
        # plt.plot(target[:,7],label='target', linestyle='--')
        # plt.ylim(-1,1)
        plt.legend()
        plt.subplot(916)
        plt.plot(quat[:,5],label='23')
        plt.plot(orient6,label='Real life', linestyle='--')
        # plt.plot(target[:,8],label='target', linestyle='--')
        plt.legend()
        plt.ylim(-1,1)
        plt.subplot(917)
        plt.plot(quat[:,6],label='31')      
        plt.plot(orient7,label='Real life', linestyle='--')
        # plt.plot(target[:,9],label='target', linestyle='--')
        plt.ylim(-1,1)
        plt.legend()
        plt.subplot(918)
        plt.plot(quat[:,7],label='32')
        plt.plot(orient8,label='Real life', linestyle='--')
        # plt.plot(target[:,10],label='target', linestyle='--')
        plt.ylim(-1,1)
        plt.legend()
        plt.subplot(919)
        plt.plot(quat[:,8],label='33')
        plt.plot(orient9,label='Real life', linestyle='--')
        # plt.plot(target[:,11],label='target', linestyle='--')
        # plt.ylim(-1,1)
        plt.legend()
        plt.show()


    # plot angular velocities
    plt.figure()
    plt.suptitle("Angular Velocities")
    plt.subplot(311)
    plt.plot(smooth(omega[:,0]),label='wx')
    plt.plot(df['snn_control.gyroX']/90*np.pi,label='Real life', linestyle='--')
    # plot the cumulative of omega
    # plt.plot(np.cumsum(smooth(omega[:,0])), label='cumulative wx', linestyle='--')
    plt.plot(target[:,15],label='target', linestyle='--')
    
    # plt.ylim(-2,2)
    plt.legend()
    plt.subplot(312)
    plt.plot(smooth(omega[:,1]),label='wy')
    # plt.plot(np.cumsum(smooth(omega[:,1])), label='cumulative wy', linestyle='--')
    plt.plot(target[:,16],label='target', linestyle='--')
    plt.plot(df['snn_control.gyroY']/90*np.pi,label='Real life', linestyle='--')
    # plt.ylim(-2,2)
    plt.legend()
    plt.subplot(313)
    plt.plot(smooth(omega[:,2]),label='wz')
    # plt.plot(np.cumsum(smooth(omega[:,2])), label='cumulative wz', linestyle='--')
    plt.plot(target[:,17],label='target', linestyle='--')
    plt.plot(df['snn_control.gyroZ']/90*np.pi,label='Real life', linestyle='--')
    # plt.ylim(-2,2)
    plt.legend()
    plt.show()

    # plot quaternions on subplots and compare to real quaternions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(quat[:,0],quat[:,1],quat[:,2],label='quaternion')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
import pandas as pd
def test_animated(target=None, euler=False, length=1000):
    '''Test the policy on the environment
    Target is an array of size 18, which is substraced from the observation before passing to the policy
    Allows for position commands,
    velocity commands, etc.
    defaults to zeros
    if target is shape (1,18) then it is repeated for the entire episode
    if target is shape (1000,18), then it is used for each time step'''
    if target is None:
        target = np.zeros((1,18))
    if target.shape[0] == 1:
        target = np.repeat(target,length,axis=0)
    # target to tensor
    target = torch.tensor(target).float().reshape(length,18)
    # add zeros to make it (100,146)
    target = torch.cat((target,torch.zeros((length,128))),dim=1)
    # run the p`olicy on the environment
    out = []
    obs_out = []
    actions = []
    obs = env.reset()[0]
    # create low pass filter for actions
    action = np.zeros((4,))
    for i in range(5):
        actions.append(action)

    for i in range(length):
        obs = torch.tensor(obs).float().reshape(1,-1)
        obs = obs - target[i]
        action = policy(obs)
        actions.append(action.reshape(4,))
        # take low pass filter of last 5 actions
        # action_low = actions[-5:]
        # action = np.mean(np.stack(action_low),axis=0)
        # action = torch.tanh(action/50)
        obs, rewards,dones, _, info = env.step(action)
        out.append((obs, rewards, dones, info))
        obs_out.append(obs[:18])

    # draw the drone in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    # plot the target with dotted line
    ax.plot(target[:,0],target[:,1],target[:,2],label='target', linestyle='dashed')
    # plot the drone
    xs = np.array(obs_out)
    pos = xs[:,:3]
    quat = xs[:,3:12]
    vel = xs[:,12:15]
    omega = xs[:,15:18]
    # plot the drone
    posx = x_data = smooth(pos[:,0], window_len=50)
    posy = y_data = smooth(pos[:,1], window_len=50)
    posz = z_data = smooth(pos[:,2], window_len=50)
    abs_vel = smooth(np.linalg.norm(vel,axis=1), window_len=50)

    # Normalize colors based on index
    # colors = np.linspace(0, 1, len(posx))
    sc = ax.scatter(posx, posy, posz, c=abs_vel, cmap='viridis', label='drone', s=2)
    plt.colorbar(sc, ax=ax, label='Velocity')  # Color bar for reference    

    ax.legend()
    # set limits
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)
    plt.show()
    # animate the drone
    # creaete each frame with step of 5 (such that we have 20fps)
    fig, ax = plt.subplots()
    ims = []
    for i in range(0,length,5):
        im = ax.imshow(posx[i],posy[i])

        ims.append([im])

    ani = ArtistAnimation(fig, ims, blit=True, interval=50, repeat_delay=1000)
    ani.save('drone.mp4')
    plt.show()
    
file = '/home/korneel/Desktop/snn_logs/latest/logkorneel91.csv'

df = pd.read_csv(file, sep=',')
df = df[500:-400]
# reset the indices
df = df.reset_index(drop=True)
length = len(df)-1
target = np.zeros((length,18))

target_x = df['snn_control.posx_target'].to_numpy()
target_y = df['snn_control.posy_target'].to_numpy()
target_z = df['snn_control.posz_target'].to_numpy()
# orient1 = df['snn_control.orient_1'].to_list()
# orient2 = df['snn_control.orient_2'].to_list()
# orient3 = df['snn_control.orient_3'].to_list()
# orient4 = df['snn_control.orient_4'].to_list()
# orient5 = df['snn_control.orient_5'].to_list()
# orient6 = df['snn_control.orient_6'].to_list()
# orient7 = df['snn_control.orient_7'].to_list()
# orient8 = df['snn_control.orient_8'].to_list()
# orient9 = df['snn_control.orient_9'].to_list()
# target[:,0] = target_x
# target[:,1] = target_y
# target[:,2] = target_z
# 
# remove first 200 rows
target[1000:1500,0] = 0
# target[600:900,1] = 1
test(target=target, euler=True, length=length)
# test(target=None, euler=False, length=100)

# create path of an eight figure:
T = 10 # period in sec
t = np.arange(0,2*np.pi,step=.01/T)
length = len(t)
target = np.zeros((length,18))

a = 2
b = 2
x = a*np.sin(t)
y = b*np.sin(t)*np.cos(t)
target[:,0] = x
target[:,1] = y
target[:,2] = 0
# test_animated(target=target, euler=True, length=length)
