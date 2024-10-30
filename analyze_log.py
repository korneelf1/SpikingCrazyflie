import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# spiking specific code
# from spiking_gym_wrapper import SpikingEnv
from spikingActorProb import SpikingNet
# from masked_actors import MaskedNet
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from l2f_gym import Learning2Fly
env = Learning2Fly()

observation_space = env.observation_space.shape
action_space = env.action_space.shape

from torch import nn
def create_policy_td3(hidden=[256,128]):
    hidden_sizes = hidden
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
            return np.array((self.mu(x)).detach().clone().cpu())
        def reset(self):
            self.t = 0
            if hasattr(self.preprocess, 'reset'):
                self.preprocess.reset()
        # def to_cuda(self):
        #     return self.preprocess.to(device)

    actor = Wrapper(net_a, stoch=False, warmup=0)
    return actor
    
def rotation_to_euler(R):
    '''Convert rotation matrix (flattened) to euler angles'''
    R = R.reshape(3,3)
    if R[2,0] != 1 and R[2,0] != -1:
        theta = -np.arcsin(R[2,0])
        psi = np.arctan2(R[2,1]/np.cos(theta), R[2,2]/np.cos(theta))
        if psi<-90:
            psi+=180
        if psi>90:
            psi-=180

        phi = np.arctan2(R[1,0]/np.cos(theta), R[0,0]/np.cos(theta))
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
    import math
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
    
    return roll, pitch, yaw

def euler_to_rotation(phi, theta, psi):
    '''Convert euler angles to rotation matrix'''
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    psi = np.deg2rad(psi)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])
    
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R.flatten()
def smooth(x,window_len=30):
    return np.convolve(x, np.ones(window_len)/window_len, mode='same')


def extract_quaternion(orient):
    R11, R12, R13 = orient[0], orient[1], orient[2]
    R21, R22, R23 = orient[3], orient[4], orient[5]
    R31, R32, R33 = orient[6], orient[7], orient[8]

    # Calculate the quaternion components
    q_w = np.sqrt(1 + R11 + R22 + R33) / 2
    q_x = np.sqrt(1 + R11 - R22 - R33) / 2
    q_y = np.sqrt(1 - R11 + R22 - R33) / 2
    q_z = np.sqrt(1 - R11 - R22 + R33) / 2

    # Determine the signs of qx, qy, qz based on matrix elements
    q_x = np.copysign(q_x, R32 - R23)
    q_y = np.copysign(q_y, R13 - R31)
    q_z = np.copysign(q_z, R21 - R12)

    return q_w, q_x, q_y, q_z

policy = create_policy_td3(hidden=[128,128])
# policy.load_state_dict(torch.load('TD3BC_Online_TEMP.pth', map_location=torch.device('cpu')))
policy.load_state_dict(torch.load('model_bc_sweet_sunset.pth', map_location=torch.device('cpu')))

# file = '/home/korneel/Desktop/snn_logs/latest/logkorneel74.csv' # forward file
# file = '/home/korneel/Desktop/snn_logs/latest/logkorneel02.csv'
file = '/home/korneel/Desktop/snn_logs/latest/logkorneel50.csv' # yawing file

df = pd.read_csv(file, sep=',')
# df = df[500:-200]# remove first 200 rows
df = df[500:-400]
print(df.keys())
print(df.head())
# print(df['stateEstimate.qw'])
# print(df.head())
def plot(outs=None, euler=True, quaternions=False):
    MAX_UINT16_Correct = 1
    MAX_UINT16 = 65535
    # compare og_ctrl.m1 and snn_control.motor1 and m2 motor2 ...
    og_m1 = (df['og_ctrl.m1']/MAX_UINT16_Correct).to_list() 
    og_m2 = (df['og_ctrl.m2']/MAX_UINT16_Correct).to_list()
    og_m3 = (df['og_ctrl.m3']/MAX_UINT16_Correct).to_list()
    og_m4 = (df['og_ctrl.m4']/MAX_UINT16_Correct).to_list()

    if outs is None:
        snn_m1 = ((df['snn_control.motor1']+1)/2*MAX_UINT16).to_list()
        snn_m2 = ((df['snn_control.motor2']+1)/2*MAX_UINT16).to_list()
        snn_m3 = ((df['snn_control.motor3']+1)/2*MAX_UINT16).to_list()
        snn_m4 = ((df['snn_control.motor4']+1)/2*MAX_UINT16).to_list()
        # window = 10
        # snn_m1 = np.convolve(snn_m1, np.ones(window)/window, mode='same')
        # snn_m2 = np.convolve(snn_m2, np.ones(window)/window, mode='same')
        # snn_m3 = np.convolve(snn_m3, np.ones(window)/window, mode='same')
        # snn_m4 = np.convolve(snn_m4, np.ones(window)/window, mode='same')
    else:
        print(outs.shape)
        snn_m1 = ((outs[:,0,0]+1)/2*MAX_UINT16)
        snn_m2 = ((outs[:,0,1]+1)/2*MAX_UINT16)
        snn_m3 = ((outs[:,0,2]+1)/2*MAX_UINT16)
        snn_m4 = ((outs[:,0,3]+1)/2*MAX_UINT16)
        snn_m1_real = ((df['snn_control.motor1']+1)/2*MAX_UINT16).to_list()
        snn_m2_real = ((df['snn_control.motor2']+1)/2*MAX_UINT16).to_list()
        snn_m3_real = ((df['snn_control.motor3']+1)/2*MAX_UINT16).to_list()
        snn_m4_real = ((df['snn_control.motor4']+1)/2*MAX_UINT16).to_list()
        # use rolling window for ing
        window = 10
        # snn_m1 = np.convolve(snn_m1, np.ones(window)/window, mode='same')
        # snn_m2 = np.convolve(snn_m2, np.ones(window)/window, mode='same')
        # snn_m3 = np.convolve(snn_m3, np.ones(window)/window, mode='same')
        # snn_m4 = np.convolve(snn_m4, np.ones(window)/window, mode='same')
        # snn_m1_real = np.convolve(snn_m1_real, np.ones(window)/window, mode='same')
        # snn_m2_real = np.convolve(snn_m2_real, np.ones(window)/window, mode='same')
        # snn_m3_real = np.convolve(snn_m3_real, np.ones(window)/window, mode='same')
        # snn_m4_real = np.convolve(snn_m4_real, np.ones(window)/window, mode='same')

    simulated_c = np.loadtxt('outputs_c.csv', delimiter=',')
    x_vals = np.arange(0, len(og_m1), 1)/100
    plt.subplot(411)
    plt.plot(x_vals, og_m1)
    plt.plot(x_vals, (snn_m1))
    # plt.plot(x_vals, (snn_m1_real))
    # plt.plot(x_vals[:1000], (simulated_c[:,0]))
    plt.legend(['og_m1', 'snn_m1', 'snn_m1_real','simulated_c'])
    # plt.ylim(0, MAX_UINT16)
    plt.subplot(412)
    plt.plot(x_vals, og_m2)
    plt.plot(x_vals, snn_m2)
    # plt.plot(x_vals, snn_m2_real)
    # plt.plot(x_vals[:1000], (simulated_c[:,1]))
    plt.legend(['og_m2', 'snn_m2', 'snn_m2_real','simulated_c'])
    # plt.ylim(0, MAX_UINT16)
    plt.subplot(413)
    plt.plot(x_vals, og_m3)
    plt.plot(x_vals, snn_m3)
    # plt.plot(x_vals, snn_m3_real)
    # plt.plot(x_vals[:1000], (simulated_c[:,2]))
    plt.legend(['og_m3', 'snn_m3', 'snn_m3_real','simulated_c'])
    plt.ylim(0, MAX_UINT16)
    plt.subplot(414)
    plt.plot(x_vals, og_m4)
    plt.plot(x_vals, snn_m4)
    # plt.plot(x_vals, snn_m4_real)
    # plt.plot(x_vals[:1000], (simulated_c[:,3]))
    plt.legend(['og_m4', 'snn_m4', 'snn_m4_real','simulated_c'])
    plt.xlabel('Time (s)')
    plt.ylabel('Motor output')
    # plt.ylim(0, MAX_UINT16)
    plt.suptitle('Motor output over time')
    plt.show()
    # using_snn = df['og_ctrl.snn_in_use'].to_list()
    velx = df['snn_control.velBodyX'].to_list()
    vely = df['snn_control.velBodyY'].to_list()
    velz = df['snn_control.velBodyZ'].to_list()
    x_vals = np.arange(0, len(velx), 1)/100
    plt.plot(x_vals, velx)
    plt.plot(x_vals, vely)
    plt.plot(x_vals, velz)
    # plt.plot(x_vals, using_snn)
    plt.legend(['velx', 'vely', 'velz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity over time')
    plt.show()

    posx = df['snn_control.posx'].to_list()
    posy = df['snn_control.posy'].to_list()
    posz = df['snn_control.posz'].to_list()
    posx_target = df['snn_control.posx_target'].to_list()
    posy_target = df['snn_control.posy_target'].to_list()
    posz_target = df['snn_control.posz_target'].to_list()
    x_vals = np.arange(0, len(posx), 1)/100
    plt.plot(x_vals, posx)
    # plt.plot(x_vals, posx_target, linestyle='--')
    plt.plot(x_vals, posy)
    # plt.plot(x_vals, posy_target, linestyle='--')
    plt.plot(x_vals, posz)
    # plt.plot(x_vals, posz_target, linestyle='--')
    # plt.plot(x_vals, using_snn
    plt.legend(['posx error', 'posy error', 'posz error'])
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position error over time')
    plt.show()

    # plot angular velocities from gyroX, gyroY, gyroZ
    gyrox = df['snn_control.gyroX'].to_list()
    gyroy = df['snn_control.gyroY'].to_list()
    gyroz = df['snn_control.gyroZ'].to_list()
    plt.plot(x_vals, gyrox)
    plt.plot(x_vals, gyroy)
    plt.plot(x_vals, gyroz)
    plt.legend(['gyrox', 'gyroy', 'gyroz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity over time')
    plt.show()
    # qw = df['stateEstimate.qw'].to_numpy()
    # qx = df['stateEstimate.qx'].to_numpy()
    # qy = df['stateEstimate.qy'].to_numpy()
    # qz = df['stateEstimate.qz'].to_numpy()
    # # plot quaternions
    # plt.figure()
    # plt.suptitle("Quaternions")
    # plt.subplot(411)
    # plt.plot(qw,label='q_w')
    # plt.subplot(412)
    # plt.plot(qx, label='q_x')
    # plt.subplot(413)
    # plt.plot(qy, label='q_y')
    # plt.subplot(414)
    # plt.plot(qz, label='q_z')
    # plt.legend()
    # plt.show()
    # plot orients:
    orient1 = df['snn_control.orient_1'].to_list()
    orient2 = df['snn_control.orient_2'].to_list()
    orient3 = df['snn_control.orient_3'].to_list()
    orient4 = df['snn_control.orient_4'].to_list()
    orient5 = df['snn_control.orient_5'].to_list()
    orient6 = df['snn_control.orient_6'].to_list()
    orient7 = df['snn_control.orient_7'].to_list()
    orient8 = df['snn_control.orient_8'].to_list()
    orient9 = df['snn_control.orient_9'].to_list()
    eulers = np.zeros((len(orient1),3))
    # rotation_to_euler(np.array([orient1,orient2,orient3,orient4,orient5,orient6,orient7,orient8,orient9]).T)
    orients = np.zeros((len(orient1),9))
    # for i in range(len(orient1)):
    #     orient = np.array([orient1[i],orient2[i],orient3[i],orient4[i],orient5[i],orient6[i],orient7[i],orient8[i],orient9[i]])
    #     # print(orient)
    #     eulers[i] = np.array(rotation_to_euler(orient))
    #     orients[i] = euler_to_rotation(eulers[i][0], eulers[i][1], eulers[i][2])
    # orient1 = orients[:,0]
    # orient2 = orients[:,1]
    # orient3 = orients[:,2]
    # orient4 = orients[:,3]
    # orient5 = orients[:,4]
    # orient6 = orients[:,5]
    # orient7 = orients[:,6]
    # orient8 = orients[:,7]
    # orient9 = orients[:,8]

    plt.figure()
    plt.suptitle("Orientations")
    plt.subplot(911)
    plt.plot(orient1,label='11')
    plt.legend()
    plt.subplot(912)
    plt.plot(orient2,label='12')
    plt.legend()
    plt.subplot(913)
    plt.plot(orient3,label='13')
    plt.legend()
    plt.subplot(914)
    plt.plot(orient4,label='21')
    plt.legend()
    plt.subplot(915)
    plt.plot(orient5,label='22')
    plt.legend()
    plt.subplot(916)
    plt.plot(orient6,label='23')
    plt.legend()
    plt.subplot(917)
    plt.plot(orient7,label='31')  
    plt.legend()
    plt.subplot(918)
    plt.plot(orient8,label='32')
    plt.legend()
    plt.subplot(919)
    plt.plot(orient9,label='33')
    plt.legend()
    plt.show()
    
    orient1 = df['snn_control.orient_1'].to_numpy()
    orient2 = df['snn_control.orient_2'].to_numpy()
    orient3 = df['snn_control.orient_3'].to_numpy()
    orient4 = df['snn_control.orient_4'].to_numpy()
    orient5 = df['snn_control.orient_5'].to_numpy()
    orient6 = df['snn_control.orient_6'].to_numpy()
    orient7 = df['snn_control.orient_7'].to_numpy()
    orient8 = df['snn_control.orient_8'].to_numpy()
    orient9 = df['snn_control.orient_9'].to_numpy()
    length = len(orient1)
    quat_real = np.vstack((orient1,orient2,orient3,orient4,orient5,orient6,orient7,orient8,orient9)).T
    if quaternions:
        quats = np.zeros((length,4))
        for i in range(length):
            quats[i] = extract_quaternion(quat_real[i])

        # plot quaternions
        plt.figure()
        plt.suptitle("Quaternions")
        plt.subplot(411)
        plt.plot(quats[:,0],label='q_w')
        plt.subplot(412)
        plt.plot(quats[:,1], label='q_x')
        plt.subplot(413)
        plt.plot(quats[:,2], label='q_y')
        plt.subplot(414)
        plt.plot(quats[:,3], label='q_z')
        plt.legend()       
        plt.show()

    if euler:
        euler = np.zeros((length,3))
        euler_real = np.zeros((length,3))
        for i in range(length):
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


def df_entry_to_obs_arr(df):
    '''Given a dataframe row with posx, posy, posz, orient1-9, velBodyX, velBodyY, velBodyZ, GyroX, GyroY, GyroZ,
    Create an array with those entries'''
    df['snn_control.posx'] = df['snn_control.posx']-df['snn_control.posx_target']
    df['snn_control.posy'] = df['snn_control.posy']-df['snn_control.posy_target']
    df['snn_control.posz'] = df['snn_control.posz']-df['snn_control.posz_target']
    return np.array([
        df['snn_control.posx'], df['snn_control.posy'], df['snn_control.posz'],
        df['snn_control.orient_1'], df['snn_control.orient_2'], df['snn_control.orient_3'],
        df['snn_control.orient_4'], df['snn_control.orient_5'], df['snn_control.orient_6'],
        df['snn_control.orient_7'], df['snn_control.orient_8'], df['snn_control.orient_9'],
        df['snn_control.velBodyX'], df['snn_control.velBodyY'], df['snn_control.velBodyZ'],
        df['snn_control.gyroX'], df['snn_control.gyroY'], df['snn_control.gyroZ']
    ])

def test(df, yaw_zero = False):
    # create observation from df
    # observations = df.apply(df_entry_to_obs_arr, axis=1)
    # print(observations.head())
    # print(df.keys())

    # df['snn_control.posx'] = np.zeros_like(df['snn_control.posx'].to_numpy())
    # df['snn_control.posy'] = np.zeros_like(df['snn_control.posy'].to_numpy())
    # df['snn_control.posz'] = np.zeros_like(df['snn_control.posz'].to_numpy())
    observations = df_entry_to_obs_arr(df)
    outs = []
    policy.reset()
    obs_to_test = np.zeros((observations.shape[1],18))
    for i in range(observations.shape[1]):
        obs = observations[:,i].reshape(-1,18)
        # set yaw to zero
        if yaw_zero:
            orients = obs[:,3:12]
            # obs[:,3:12] = np.zeros_like(orients)
            euler = rotation_to_euler(orients)
            # # euler[2] +=np.pi/4
            # # euler[1] *= -1
            # # euler[0] -= np.pi/16
            # yaw = euler[2]*np.pi/180
            # pitch = euler[1] *np.pi/180
            # roll = euler[0] *np.pi/180
            # orients = euler_to_rotation(roll,pitch, yaw)
            # obs = np.zeros_like(obs)
            obs[:,2] = 0
            # obs[:,1] -= 0.1
            obs_to_test[i] = obs
            # obs[:,3:12] = orients
            # obs[:,12:15] *= 0.01
            gyrox = obs[:,15]
            gyroy = obs[:,16]
            gyroz = obs[:,17]
            # obs[:,15] = gyrox
            # obs[:,16] = gyroy+0.075/2
            # obs[:,17] = gyroz+0.075
            print(obs)
            
        # obs[:,2]-=
        
        outs.append(policy(obs))
    np.savetxt('obs_out_from_real.csv', obs_to_test, delimiter=',')
    return np.array(outs)

# compare sim to true observations ranges
def compare_sim_real(obs_sim, obs_true):
    print("shape of sim obs: ", obs_sim.shape)
    print("shape of real obs: ", obs_true.shape)
    
    means_sim = np.mean(obs_sim, axis=1)[0:18]
    std_sim = np.std(obs_sim,axis=1)[0:18]

    means_true = np.mean(obs_true, axis=1)[0:]
    std_true = np.std(obs_true,axis=1)[0:]
    print("Simulated Observations - Means: ", means_sim)
    print("Simulated Observations - Standard Deviations: ", std_sim)
    print("Real Observations - Means: ", means_true)
    print("Real Observations - Standard Deviations: ", std_true)
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(means_sim)), means_sim, yerr=std_sim, fmt='-o', label='Simulated')
    plt.errorbar(range(len(means_true)), means_true, yerr=std_true, fmt='-o', label='Real')
    plt.xlabel('Observation Index')
    plt.ylabel('Value')
    plt.title('Comparison of Simulated and Real Observations')
    plt.legend()
    plt.show()



# plot()
df['snn_control.posx'] = df['snn_control.posx']-df['snn_control.posx_target']
# df['snn_control.posx'] = 0
df['snn_control.posy'] = df['snn_control.posy']-df['snn_control.posy_target']
# df['snn_control.posy'] = 0
df['snn_control.posz'] = df['snn_control.posz']-df['snn_control.posz_target']
# df['snn_control.posz'] = 0
df['snn_control.gyroX'] = -df['snn_control.gyroX']/(180/np.pi)
df['snn_control.gyroY'] = df['snn_control.gyroY']/(180/np.pi)
df['snn_control.gyroZ'] = df['snn_control.gyroZ']/(180/np.pi)
# same for orientations
# if stateEstimate.qw is present, use that to calculate the orientations

# net_outs = test(df, yaw_zero=False)
plot(outs=None, quaternions=False, euler=True)
real_observations = df_entry_to_obs_arr(df)
sim_observations = 'obs_out.csv'
# load sim observation in array (was created with np.savetxt)
sim_observations = np.loadtxt('obs_out.csv', delimiter=',').transpose()

compare_sim_real(sim_observations, real_observations)