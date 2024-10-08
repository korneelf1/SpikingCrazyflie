#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch

from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.env import DummyVectorEnv
from l2f_gym import Learning2Fly, SubprocVectorizedL2F, ShmemVectorizedL2F
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from spikingActorProb import SpikingNet

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=1.5e4)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=12)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
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
    parser.add_argument("--exploration-noise", type=str, default="default")
    parser.add_argument("--spiking", type=bool, default=False)
    parser.add_argument("--slope", type=float, default=2)
    parser.add_argument("--slope-schedule", type=bool, default=True)
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def test_sac(args: argparse.Namespace = get_args()) -> None:
    env = Learning2Fly()
    train_envs = DummyVectorEnv([lambda: Learning2Fly() for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: Learning2Fly() for _ in range(args.test_num)])

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.spiking:
        args.hidden_sizes = [128]
        net_a = SpikingNet(state_shape=args.state_shape, hidden_sizes=[args.hidden_sizes], action_shape=256, repeat=args.repeat_per_forward, slope=args.slope, slope_schedule=args.slope_schedule, reset_in_call=True)
    else: # model
        net_a = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net_a, args.action_shape, device=args.device,).to(args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )

    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    args_wandb = {
      'epoch': args.epoch,
      'step_per_epoch': args.step_per_epoch,
      'step_per_collect': args.step_per_collect, # 2.5 s
      'test_num': args.test_num,
      'update_per_step': args.update_per_step,
      'batch_size': args.batch_size,
      'wandb_project': 'FastPyDroneGym',
      'resume_id':1,
      'logger':'wandb',
      'algo_name': 'sac',
      'task': 'stabilize',
      'seed': int(3),
      'logdir':'',
      'spiking':args.spiking,
      'recurrent':False,
      'masked':False,
      'logger': 'wandb',
      'drone': 'stock drone',
      'buffer_size': 300000,
      'collector_type': 'Collector',
      'reinit': True,
      'reward_function': 'reward_squared_fast_learning',
      'exploration_noise': args.exploration_noise,
      'surrogate sigmoid': args.exploration_noise,
      'slope': args.slope,
      'slope_schedule': args.slope_schedule
      }
    
    if args.exploration_noise == 'None':
        args.exploration_noise = None
    policy: DDPGPolicy = DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        action_space=env.action_space,
        exploration_noise=args.exploration_noise,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)



    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ddpg"
    log_name = os.path.join(args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
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

    logger = WandbLogger(project="l2f_ddpg",config=args_wandb)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args_wandb))
    logger.load(writer)


    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    exploration_noise = str(args.exploration_noise)
    if args.spiking:
        exploration_noise = "spiking_" + exploration_noise
        
    print("Exploration noise:", exploration_noise)
    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, f"policy_ddpg_{exploration_noise}_{timestamp}.pth"))
        logger.wandb_run.log_artifact(os.path.join(log_path, f"policy_ddpg_{exploration_noise}_{timestamp}.pth"), name='policy', type='model')

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
            show_progress=False,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    import wandb
    test_sac()
    wandb.run.finish()