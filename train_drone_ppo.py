# custom code
from custom_collector import ParallelCollector, FastPyDroneSimCollector
from gym_sim import Drone_Sim

# tianshou code
from tianshou.policy import SACPolicy, BasePolicy, PPOPolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import WandbLogger

import torch
import os
from torch.distributions import Distribution, Independent, Normal

# torch.set_anomaly_enabled(True) # check for detecting anomalies

def dist(loc_scale) -> Distribution:
        loc, scale = loc_scale[0], loc_scale[1]
        return Independent(Normal(loc, scale), 1)

def create_policy():
    print('Observation space: ',observation_space)
    # create the networks behind actors and critics
    net_a = Net(state_shape=observation_space,
                hidden_sizes=[64,64], device='cpu')
    net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                    hidden_sizes=[64,64],
                    concat=False,)

    # create actors and critics
    actor = ActorProb(
        net_a,
        action_space,
        unbounded=True,
        conditioned_sigma=True,
    )
    critic1 = Critic(net_c1, device='cpu')
    actor_critic = ActorCritic(actor=actor, critic=critic1)
    

    # create the optimizer
    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)


    
    policy = PPOPolicy(actor=actor, critic = critic1, optim=optim, action_space=env.action_space,action_scaling=True, dist_fn= dist)
    return policy

# define training args
args = {
      'epoch': 1e2,
      'step_per_epoch': 5e3,
      'step_per_collect': 5e2, # 5 s
      'test_num': 10,
      'repeat_per_collect': 10,
      'update_per_step': 2,
      'batch_size': 1,
      'wandb_project': 'FastPyDroneGym',
      'resume_id':1,
      'logger':'wandb',
      'algo_name': 'ppo',
      'task': 'stabilize',
      'seed': int(4),
      'logdir':'/',
      }

# define number of drones to be simulated
N_envs = 1

# define action buffer True to encapsulate action history in observation space
env = Drone_Sim(N_drones=N_envs, action_buffer=True,test=False)
test_env = Drone_Sim(N_drones=1, action_buffer=True, test=True)

observation_space = env.observation_space.shape or env.observation_space.n
action_space = env.action_space.shape or env.action_space.n

policy = create_policy()

# create buffer (stack_num defines the number of sequenctial samples)
buffer=VectorReplayBuffer(total_size=200000,buffer_num=N_envs, stack_num=1)
# create the parallel train_collector, which is optimized to gather custom vectorized envs
train_collector = FastPyDroneSimCollector(policy=policy, env=env, buffer=buffer)
train_collector.reset()
test_collector = FastPyDroneSimCollector(policy=policy,env=test_env)
# define a number of start timesteps to fill buffer (now one sec of data *100 drones )
train_collector.collect(n_step=1e2)

# log
import datetime
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
# args['algo_name = "sac"
log_name = os.path.join(args['task'], "sac", str(args['seed']), now)
log_path = os.path.join(args['logdir'], log_name)

# logger
# logger = WandbLogger()
# logger_factory = LoggerFactoryDefault()
# if args['logger'] == "wandb":
#     logger_factory.logger_type = "wandb"
#     logger_factory.wandb_project = 'test'
# else:
#     logger_factory.logger_type = "tensorboard"
# logger = logger_factory.create_logger(
#     log_dir=log_path,
#     experiment_name=log_name,
#     run_id=args['resume_id'],
#     config_dict=args,
# )

def save_best_fn(policy: BasePolicy, log_path='') -> None:
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_ppo.pth"))


print("Start training")
# trainer
result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,
            max_epoch=args['epoch'],
            step_per_epoch=args['step_per_epoch'],
            repeat_per_collect=args['repeat_per_collect'],
            episode_per_test=args['test_num'],
            batch_size=args['batch_size'],
            step_per_collect=args['step_per_collect'],
            save_best_fn=save_best_fn,
            test_in_train=False,
        ).run()
# print with nice formatting
import pprint
pprint.pprint(result)
