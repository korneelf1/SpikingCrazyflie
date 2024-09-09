import numpy as np

# tianshou code
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import Net
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import WandbLogger


import torch
class NumpyDeque(object):
    def __init__(self, shape:tuple, device='cpu') -> None:
        self.shape_arr = shape
        print(len(self.shape_arr))
        print(self.shape_arr)
        self.array = np.zeros((self.shape_arr), dtype=np.float32)

    def __len__(self):
        return self.shape_arr[1]
    
    def append(self, els):
        if len(self.shape_arr) == 1: # no batch but individual
            self.array = np.roll(self.array, els.shape[0], axis=0)
            self.array[0:els.shape[0]] = els.astype(np.float32)
        else:
            assert els.shape[0] == self.shape_arr[0] 
            self.array = np.roll(self.array, els.shape[1], axis=1)
            self.array[:,0:els.shape[1]] = els.astype(np.float32)

    def reset(self, vecs=None):
        if vecs is None:
            self.array = np.zeros((self.shape_arr), dtype=np.float32)
        elif isinstance(vecs,np.ndarray):
            self.array[vecs==1] = 0.
            
    def __call__(self):
        return self.array
    def __repr__(self):
        return str(self.array)
    def __array__(self, dtype=None):
        if dtype:
            return self.array.astype(dtype)
        return self.array
    def __getitem__(self, idx):
        return self.array[idx]
    @property
    def shape(self):
        return self.shape_arr

def quaternion_to_euler(q):
    '''
    q: np.array of shape (4,)
    '''
    qw, qx, qy, qz = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def forcetorque_to_rpm(ft):
    '''
    ft: np.array of shape (4,)
    NOTE: if SAC then ft is [-1,1]
    https://github.com/arplaboratory/learning_to_fly_controller/blob/d52b11a03caa1dadaf6b6bf02f328719b5bdd215/rl_tools_controller.c#L265
    and power distribution quadrotor
    '''
    # Motor Mixing Matrix
    A = 0.046*0.707106781/4
    B = 0.046*0.707106781/4
    C = 0.005964552/4
    MM= np.array([[1/4, -A, -B, -C],
                    [1/4, -A, B, C],
                    [1/4, A, B, -C],
                    [1/4, A, -B, C]]) # why so different than legacy code?
    
    Farr = MM@ft

    # thrust = a * pwm^2 + b * pwm from power distribution (different than the thrust = c1 + c2*rpm + c3*rpm^2)
    pwmToThrustA = 0.091492681 
    pwmToThrustB = 0.067673604
    
    # pwm = (-pwmToThrustB + np.sqrt(pwmToThrustB**2 - 4*pwmToThrustA*(-Farr)))/(2*pwmToThrustA)
    # clip between 0 and 1 and scale to -1 to 1
    # pwm = np.clip(pwm, 0, 1) * 2 - 1


    # Calculate RPM
    # # F = c1 + c2*rpm + c3*rpm^2
    # # F = MM*ft
    # # c1 + c2*rpm + c3*rpm^2 = MM*ft
    c1 = 0.0213
    c2 = -0.0112
    c3 = 0.1201
    # F = MM@ft
    rpm = max(np.roots([c3, c2, c1 - Farr[0]])) # only take the positive root
    # # reschale from 0 to 1 to -1 to 1
    pwm = 2*rpm - 1

    # # clip between -1 and 1
    # rpm = np.clip(rpm, -1, 1)
    print(pwm)
    return pwm


if __name__=='__main__':
    def create_policy(env):
        observation_space = env.observation_space.shape or env.observation_space.n
        action_space = env.action_space.shape or env.action_space.n
        # create the networks behind actors and critics
        net_a = Net(state_shape=observation_space,
                    hidden_sizes=[64,64], device='cpu')
        net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                        hidden_sizes=[64,64],
                        concat=True,)
        net_c2 = Net(state_shape=observation_space,action_shape=action_space,
                        hidden_sizes=[64,64],
                        concat=True,)

        # create actors and critics
        actor = ActorProb(
            net_a,
            action_space,
            unbounded=True,
            conditioned_sigma=True,
        )
        critic1 = Critic(net_c1, device='cpu')
        critic2 = Critic(net_c2, device='cpu')

        # create the optimizers
        actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

        # create the policy
        policy = SACPolicy(actor=actor, actor_optim=actor_optim, \
                            critic=critic1, critic_optim=critic_optim,\
                            critic2=critic2, critic2_optim=critic2_optim,\
                            action_space=env.action_space,\
                            observation_space=env.observation_space, \
                            action_scaling=True) # make sure actions are scaled properly
        return policy

    def forward_policy(policy, state):
        return policy(state)


    test_qeue = NumpyDeque((3,5))
    print(type(test_qeue.array))
    print(test_qeue)
    ones_1 = np.ones((3,1))
    twos_2 = np.ones((3,2))*2
    test_qeue.append(ones_1)
    print(test_qeue)
    test_qeue.append(twos_2)
    print(test_qeue)
    test_qeue.reset(np.array([0,1,0]))
    print(test_qeue)

    # test forcetorque_to_rpm
    ft = np.array([0.027*9.81*0.,.0,.00,.00])
    print(forcetorque_to_rpm(ft))

