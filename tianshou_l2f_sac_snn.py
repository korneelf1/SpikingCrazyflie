#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch

from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.env import DummyVectorEnv
from l2f_gym import Learning2Fly

# spiking neural network specific:
from spiking_gym_wrapper import SpikingEnv
from spikingActorProb import SpikingNet

# wandb
import wandb
# wandb.init(mode='disabled')
args_wandb = {
      'epoch': 1,
      'step_per_epoch': 5e3,
      'step_per_collect': 1, # 2.5 s
      'update_per_step': 2,
      'test_num': 50,
      'batch_size': 256,
      'Environment': 'L2F',
      'resume_id':1,
      'logger':'wandb',
      'algo_name': 'sac',
      'task': 'stabilize',
      'seed': int(3),
      'logdir':'',
      'spiking':False,
      'recurrent':False,
      'masked':False,
      'logger': 'wandb',
      'drone': 'stock drone',
      'buffer_size': 1000000,
      'collector_type': 'Collector',
      'reinit': True,
      'reward_function': 'surrogate slope scheduling, alpha=0.0 symmetric observations with action history',
      'slope': 2,
      'slope_schedule': False,
        'alpha': 0.0,
        'action_history': True,
        'stack_number': 1,
      }

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=args_wandb['alpha'])
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=args_wandb['step_per_epoch'])
    parser.add_argument("--step-per-collect", type=int, default=args_wandb['step_per_collect'])
    parser.add_argument("--update-per-step", type=int, default=args_wandb['update_per_step'])
    parser.add_argument("--repeat-per-forward", type=int, default=4)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=12)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--slope", type=float, default=args_wandb['slope'])
    parser.add_argument("--slope_schedule", type=bool, default=args_wandb['slope_schedule'])
    parser.add_argument("--stack-number", type=bool, default=args_wandb['stack_number'])
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="l2f")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()

# log
import datetime
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
# args['algo_name = "sac"
current_path = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(current_path,args_wandb['logdir'], args_wandb['task'], "sac")
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter

logger = WandbLogger(project="thesis_graphs_fast_learning",config=args_wandb)
writer = SummaryWriter(log_path)
writer.add_text("args", str(args_wandb))
logger.load(writer)
import wandb
# wandb.init(mode='disabled')
import gymnasium as gym
def test_sac(args: argparse.Namespace = get_args(),logger=None) -> None:
    wandb.define_metric('*', step_metric='global_step')
    # env = gym.make("MountainCarContinuous-v0")
    env = Learning2Fly()
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    pprint.pprint(args_wandb)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    if args_wandb['stack_number'] == 1:
        net_a = SpikingNet(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device, action_shape=128, repeat=args.repeat_per_forward, slope=args.slope, slope_schedule=args.slope_schedule, reset_in_call=True)
    else:
        net_a = SpikingNet(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device, action_shape=128, repeat=1, slope=args.slope, slope_schedule=args.slope_schedule, reset_in_call=True)
    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)

    train_envs = DummyVectorEnv([lambda: Learning2Fly() for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: Learning2Fly() for _ in range(args.test_num)])
    
    logger.wandb_run.watch(actor)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    print("Does spiking network have attribute epoch?")
    print(hasattr(actor.preprocess, "epoch"))
    policy: SACPolicy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs), stack_num=args.stack_number)
    else:
        buffer = ReplayBuffer(args.buffer_size, stack_num=args.stack_number)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=False)
    test_collector = Collector(policy, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # args.algo_name = "sac"
    # log_name = os.path.join(args.algo_name, str(args.seed), now)
    # log_path = os.path.join(args.logdir, log_name)

    # # logger
    # logger_factory = LoggerFactoryDefault()
    # if args.logger == "wandb":
    #     logger_factory.logger_type = "wandb"
    #     logger_factory.wandb_project = args.wandb_project
    # else:
    #     logger_factory.logger_type = "tensorboard"

    # logger = logger_factory.create_logger(
    #     log_dir=log_path,
    #     experiment_name=log_name,
    #     run_id=args.resume_id,
    #     config_dict=vars(args),
    # )

    start_time = datetime.datetime.now()
    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path,f"policy_snn_actor_Full_State_{str(start_time)}.pth"))
        wandb.log_artifact(os.path.join(log_path,f"policy_snn_actor_Full_State_{str(start_time)}.pth"), name='SNN_128', type='model')
    print("Does policy have epoch attribute?")
    print(hasattr(policy.actor.preprocess, "epoch"))
    if not args.watch:
        # trainer
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    test_sac(logger=logger)
    
    wandb.finish()