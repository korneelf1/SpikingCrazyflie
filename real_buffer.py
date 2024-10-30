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

# open all csv files in a directory and create a buffer of observations
import numpy as np
def df_entry_to_obs_arr(df):
    '''Given a dataframe row with posx, posy, posz, orient1-9, velBodyX, velBodyY, velBodyZ, GyroX, GyroY, GyroZ,
    Create an array with those entries'''
    df['snn_control.posx'] = df['snn_control.posx']-df['snn_control.posx_target']
    df['snn_control.posy'] = df['snn_control.posy']-df['snn_control.posy_target']
    df['snn_control.posz'] = df['snn_control.posz']-df['snn_control.posz_target']
    df['snn_control.gyroX'] = df['snn_control.gyroX']*(np.pi/180)
    df['snn_control.gyroY'] = df['snn_control.gyroY']*(np.pi/180)
    df['snn_control.gyroZ'] = df['snn_control.gyroZ']*(np.pi/180)
    if 'stateEstimate.qw' in df.keys():
        qw = df['stateEstimate.qw']
        qz = df['stateEstimate.qx']
        qy = df['stateEstimate.qy']
        qx = df['stateEstimate.qz']
        df['snn_control.orient_1'] = 1 - 2 * qy * qy - 2 * qz * qz
        df['snn_control.orient_2'] = 2 * qx * qy - 2 * qw * qz
        df['snn_control.orient_3'] = 2 * qx * qz + 2 * qw * qy
        df['snn_control.orient_4'] = 2 * qx * qy + 2 * qw * qz
        df['snn_control.orient_5'] = 1 - 2 * qx * qx - 2 * qz * qz
        df['snn_control.orient_6'] = 2 * qy * qz - 2 * qw * qx
        df['snn_control.orient_7'] = 2 * qx * qz - 2 * qw * qy
        df['snn_control.orient_8'] =  2 * qy * qz + 2 * qw * qx
        df['snn_control.orient_9'] = 1 - 2 * qx * qx - 2 * qy * qy
    obs =  np.array([
        df['snn_control.posx'], df['snn_control.posy'], df['snn_control.posz'],
        df['snn_control.orient_1'], df['snn_control.orient_2'], df['snn_control.orient_3'],
        df['snn_control.orient_4'], df['snn_control.orient_5'], df['snn_control.orient_6'],
        df['snn_control.orient_7'], df['snn_control.orient_8'], df['snn_control.orient_9'],
        df['snn_control.velBodyX'], df['snn_control.velBodyY'], df['snn_control.velBodyZ'],
        df['snn_control.gyroX'], df['snn_control.gyroY'], df['snn_control.gyroZ']
    ])
    # horizontally stack 128 zeros to make obs shape 146
    MAX_UINT16 = 65535
    act = ((np.array([
        df['og_ctrl.m1'], df['og_ctrl.m2'], df['og_ctrl.m3'], df['og_ctrl.m4']
    ]))/MAX_UINT16)*2-1
    try:
        if np.max(np.abs(act))>1:
            print('act out of bounds')
            return None
    except:
        print('act out of bounds')
        return None
    return obs, act
def calculate_reward(obs):
    '''Given an observation, calculate the reward'''
    pos = obs[:3]
    vel = obs[12:15]
    q = obs[3:12]
    qd = obs[15:]

    r = - 2*np.sum((vel)**2) \
                -2.5*2*(np.arccos(1-q[0]**2)+np.arccos(1-q[4]**2)+np.arccos(1-q[8]**2))/3\
                - 0.25*np.sum((qd)**2) \
                    + 2 \
                        -2*np.sum((pos)**2) 
    return r
def create_buffer(directory, action_history=False):
    '''Given a directory of csv files, create a buffer of observations'''
    import os
    import pandas as pd
    buffer =ReplayBuffer(5000)
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            df = df[300:] # remove the first 300 entries which is takeoff
            arrays = df_entry_to_obs_arr(df)
            if arrays is None:
                print('\n\nskipping: ',filename)
                continue
            obs, act = arrays

            obs = obs.T
            # stack zeros to make shape n,18 n,146
            # obs = np.hstack((obs,np.zeros((obs.shape[0],128))))
            act = act.T
            rewards = [calculate_reward(obs[i]) for i in range(obs.shape[0])]
            # convert rewards to array
            rewards = np.array(rewards).reshape(-1,1)
            dones = np.zeros_like(rewards)
            obs_stack = np.hstack((obs, act, rewards.reshape(-1,1), dones.reshape(-1,1)))
            # add the rollout to the buffer
            # chop up in 100 step sequences with step size of 50 (0-100,50-150,100-200,150-250,...)
            for j in range(0,obs_stack.shape[0]-100,50):
                buffer.add(Batch({'obs':obs_stack[j:j+100],'act':act[j+99],'rew':rewards[j+99],'terminated': dones[j+99].reshape(-1,1),'truncated': dones[j+99].reshape(-1,1)}))

            # now add the last bit obs_stack.shape[0]%100 to the buffer
            if obs_stack.shape[0]%100>0:
                buffer.add(Batch({'obs':obs_stack[-100:],'act':act[-1],'rew':rewards[-1],'terminated': dones[-1].reshape(-1,1),'truncated': dones[-1].reshape(-1,1)}))
            # buffer.add(Batch({'obs':obs_stack,'act':np.array(action_lst[-1]),'rew':np.array(rewards_lst[-1]),'terminated': np.array(dones_lst[-1]).reshape(-1,1),'truncated': np.array(dones_lst)[-1].reshape(-1,1)}))
                            
    return buffer

def convert_to_action_history(buffer):
    len_buffer = len(buffer)
    for i in range(len_buffer):
        batch = buffer[i]
        obs = batch.obs
        action_history = np.zeros((132,4))
        action_history[32:] = obs[:,18:22]
        new_obs = np.zeros((100,152))
        for j in range(obs.shape[1]):
            # action history is updated with the previous action (found at obs[:,:,146:150]) at the end of the history, everything else is moved forward
            actions = action_history[j:j+32].flatten()
            new_obs[j] = np.hstack((obs[j,:18],actions, obs[j,18:]))
        batch.obs = new_obs
# create a buffer from the csv files in the home Desktop snn_logs/latest directory
# buffer = create_buffer('/home/korneel/Desktop/snn_logs/latest')
# buffer2 = create_buffer('/home/korneel/Desktop/snn_logs/latest/totrain')
# buffer.update(buffer2)
# buffer.save_hdf5("real_data_buffer_no_zeros_full.hdf5")
# # buffer2.save_hdf5("real_data_buffer_no_zeros_2.hdf5")
# # test load
buffer = ReplayBuffer.load_hdf5("real_data_buffer_no_zeros_full.hdf5")
# print(len(buffer))
convert_to_action_history(buffer)
buffer.save_hdf5("real_data_buffer_no_zeros_full_action_history.hdf5")