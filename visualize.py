from gym_sim import Drone_Sim

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
# from spikingActorProb import SpikingNet
# from masked_actors import MaskedNet
# 
import torch
import numpy as np
import os


env = Drone_Sim(drone='og',N_drones=3)

observation_space = env.observation_space.shape
action_space = env.action_space.shape

def create_policy():
    # create the networks behind actors and critics
    net_a = Net(state_shape=observation_space,
                    hidden_sizes=[64,64], )
        
    net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[64,64],
                    concat=True,)
    net_c2 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[64,64],
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



policy = create_policy()
policy.load_state_dict(torch.load('policy.pth'))
policy.eval()

out = env.step_rollout(policy,n_step=1e3, tianshou_policy=True)

xs = out[0]
dones = out[3]

pos = xs[:,:,:3]
quat = xs[:,:,6:10]
vel = xs[:,:,3:6]
omega = xs[:,:,10:13]

pos_norm = np.linalg.norm(pos,axis=2)
quat_norm = np.linalg.norm(quat,axis=2)
vel_norm = np.linalg.norm(vel,axis=2)
omega_norm = np.linalg.norm(omega,axis=2)

# use matplotlib to visualize the norm of the 4 vectors of the 3 drones
import matplotlib.pyplot as plt
plt.figure()
# plot x y z of positions
plt.subplot(311)
plt.plot(pos[:,:,0],label='x')
plt.legend()
plt.subplot(312)
plt.plot(pos[:,:,1],label='y')
plt.legend()
plt.subplot(313)
plt.plot(pos[:,:,2],label='z')
plt.legend()
plt.show()


plt.figure()
plt.subplot(221)
print(pos_norm.shape)
plt.plot(pos_norm,label='pos')
plt.legend()

plt.subplot(222)
plt.plot(quat_norm,label='quat')
plt.legend()

plt.subplot(223)
plt.plot(vel_norm,label='vel')
plt.legend()

plt.subplot(224)
plt.plot(omega_norm,label='omega')
plt.legend()
plt.show()
