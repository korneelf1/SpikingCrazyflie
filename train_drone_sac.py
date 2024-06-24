# custom code
from custom_collector import ParallelCollector, FastPyDroneSimCollector
from gym_sim import Drone_Sim

# tianshou code
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import Net
from tianshou.data import VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils import WandbLogger


import torch
import os

def create_policy():
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

# define training args
args = {
      'epoch': 1e2,
      'step_per_epoch': 1e4,
      'step_per_collect': 1e3, # 5 s
      'test_num': 10,
      'update_per_step': 2,
      'batch_size': 128,
      'wandb_project': 'FastPyDroneGym',
      'resume_id':1,
      'logger':'wandb',
      'algo_name': 'sac',
      'task': 'stabilize',
      'seed': int(3),
      'logdir':'/',
      }

# define number of drones to be simulated
N_envs = 100

# define action buffer True to encapsulate action history in observation space
env = Drone_Sim(N_cpu=N_envs, action_buffer=True,test=False)
test_env = Drone_Sim(N_cpu=1, action_buffer=True, test=True)

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
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))


print("Start training")
# trainer
result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector, # no testing performed 
    max_epoch=args['epoch'],
    step_per_epoch=args['step_per_epoch'],
    step_per_collect=args['step_per_collect'],
    episode_per_test=args['test_num'],
    batch_size=args['batch_size'],
    save_best_fn=save_best_fn,
    # logger=logger,
    update_per_step=args['update_per_step'],
    test_in_train=False,
    buffer=buffer,).run()

# print with nice formatting
import pprint
pprint.pprint(result)