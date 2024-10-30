from l2f_agent import ConvertedModel
import torch
from l2f_gym import Learning2Fly
import numpy as np
import tianshou
from tianshou.data import ReplayBuffer, Batch
import pickle, numpy
from tqdm import tqdm
from tianshou.data.types import RolloutBatchProtocol
from tianshou.data import Collector

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar

import gymnasium as gym
import numpy as np
import torch

import argparse
import datetime
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.exploration import GaussianNoise
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils.space_info import SpaceInfo
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.exploration import BaseNoise
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    BatchWithReturnsProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils.net.continuous import Actor, Critic

model = ConvertedModel()
model.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))

env = Learning2Fly()
print(env.action_space)
print(env.observation_space)

model.load_state_dict(torch.load("l2f_agent.pth", map_location="cpu"))
class CustomBuffer():
    def __init__(self, size):
        self.size = size
        self.data = []
        self.idx = 0

    def add(self, data: dict):
        if len(self.data) < self.size:
            self.data.append(data)
        else:
            self.data[self.idx] = data
            self.idx = (self.idx + 1) % self.size
    
    def save_hdf5(self, name='CustomBuffer.pkl'):
        with open(name, 'wb') as f:
            pickle.dump(self.data, f)
    
    def __len__(self):
        return len(self.data)
    
    def sample(self, idx):
        return self.data[idx]

def gather_buffer(model, name='l2f_controller_buffer_short', size = 10, step_len = 501):
    buffer = ReplayBuffer(size=size)
    n_rollouts = size 
    for _ in tqdm(range(n_rollouts), desc="Gathering buffer"):
        partial_rollout = True
        
        # make sure to gather full rollouts
        while partial_rollout:
            # creates lists
            obs_lst = []
            action_lst = []
            rewards_lst = []
            dones_lst = []
            obs_next_lst = []

            returns = []
            obs = env.reset()[0]
            partial_rollout = False # assume full rollout unless dones before end of range
            for i in range(step_len):

                obs = torch.tensor(obs)
                action = model(obs)
                obs_lst.append(obs.numpy())
                obs, rewards, dones,_, info = env.step(action) 

                obs_next_lst.append(obs)
                action_lst.append(action)
                rewards_lst.append(rewards)
                dones_lst.append(dones)


                if dones:
                    obs = env.reset()[0]
                    print("Crashed at step", i)
                    partial_rollout = True
        
    
        obs_stack = np.hstack((np.array(obs_lst), np.array(action_lst), np.array(rewards_lst).reshape(-1,1), np.array(dones_lst).reshape(-1,1)))
        # add the rollout to the buffer
        buffer.add(Batch({'obs':obs_stack,'act':np.array(action_lst[-1]),'rew':np.array(rewards_lst[-1]),'terminated': np.array(dones_lst[-1]).reshape(-1,1),'truncated': np.array(dones_lst)[-1].reshape(-1,1)}))
        # buffer.add(Batch({'obs':obs_stack}))
    buffer.save_hdf5(f"{name}.hdf5")
    return buffer

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch_as
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import TD3Policy


class TD3BCPolicy(TD3Policy):
    """Implementation of TD3+BC. arXiv:2106.06860.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param float exploration_noise: the exploration noise, add to the action.
        Default to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network.
        Default to 0.2.
    :param int update_actor_freq: the update frequency of actor network.
        Default to 2.
    :param float noise_clip: the clipping range used in updating policy network.
        Default to 0.5.
    :param float alpha: the value of alpha, which controls the weight for TD3 learning
        relative to behavior cloning.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        alpha: float = 2.5,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor = actor,
            actor_optim = actor_optim,
            critic = critic,
            critic_optim = critic_optim,
            critic2 = critic2,
            critic2_optim = critic2_optim,
            tau = tau,
            gamma = gamma,
            exploration_noise = exploration_noise,
            policy_noise = policy_noise,
            update_actor_freq = update_actor_freq,
            noise_clip = noise_clip,
            estimation_step = estimation_step,
            **kwargs

        )
        self._alpha = alpha


    def compute_returns(self,batch):
        '''Compute returns from rewards using discounted rewards:
        R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * r_T
        '''
        gamma = self.gamma
        rewards = batch.rew
        dones = batch.done
        returns = np.zeros_like(rewards)
        running_returns = 0
        for t in reversed(range(len(rewards))):
            running_returns = rewards[t] + gamma * running_returns * (1 - dones[t])
            returns[t] = running_returns
        return returns

    def learn(self, batch, independent=True, **kwargs: Any) -> Dict[str, float]:
        # create batch from first observations
        batch_size = batch.obs.shape[0]
        observations = batch.obs[:,:, :146]
        actions = batch.obs[:,:, 146:150]
        rewards = batch.obs[:,:, 150]
        terminated = batch.obs[:,:, 151]
        observations_next = np.hstack((batch.obs[:,1:, :146], np.zeros((batch_size,1, 146))))

        # compute returns
        # modified batch
        batch = Batch(
            obs=observations,
            act=actions,
            rew=rewards,
            done=terminated,
            obs_next=observations_next,
            returns=None
        )
        compute_returns = self.compute_returns(batch)
        batch.returns = compute_returns
        # critic 1&2
        # batch = RolloutBatchProtocol(**batch)
        td1, critic_loss = self._mse_optimizer(
            batch, self.critic, self.critic_optim
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self._freq == 0:
            act = self(batch, eps=0.0).act
            q_value = self.critic(batch.obs, act)
            lmbda = self._alpha / q_value.abs().mean().detach()
            actor_loss = -lmbda * q_value.mean() + F.mse_loss(
                act, to_torch_as(batch.act, act)
            )
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1
        return {
            "loss/actor": self._last,
            "loss/critic": critic_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="l2f")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-data-task", type=str, default="halfcheetah-expert-v2")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--norm-obs", type=int, default=0)

    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)
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
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_l2f.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()



def test_td3_bc(buffer) -> None:
    args = get_args()
    env = Learning2Fly()
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    args.min_action = space_info.action_info.min_action
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", args.min_action, args.max_action)

    args.state_dim = space_info.observation_info.obs_dim
    args.action_dim = space_info.action_info.action_dim
    print("Max_action", args.max_action)

    test_envs = Learning2Fly()
    if args.norm_obs:
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # test_envs.seed(args.seed)

    # model
    # actor network
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Actor(
        net_a,
        action_shape=args.action_shape,
        max_action=args.max_action,
        device=args.device,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # critic network
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
    critic = Critic(net_c1, device=args.device, flatten_input=False).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device, flatten_input=False).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy: TD3BCPolicy = TD3BCPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    test_collector = Collector[CollectStats](policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "td3_bc"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger: WandbLogger | TensorboardLogger
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
        logger.load(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def watch() -> None:
        if args.resume_path is None:
            args.resume_path = os.path.join(log_path, "policy.pth")

            policy.load_state_dict(torch.load(args.resume_path, map_location=torch.device("cpu")))
            collector = Collector[CollectStats](policy, env)
            collector.collect(n_episode=1, render=1 / 35)

    if not args.watch:
        replay_buffer = buffer
        # trainer
        result = OfflineTrainer(
            policy=policy,
            buffer=replay_buffer,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
        )

        for i in range(1, 1 + args.epoch):
            batch = replay_buffer.sample(args.batch_size)[0]
            policy.learn(batch, )
        pprint.pprint(result)
    else:
        watch()

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    replay_buffer = gather_buffer(model)
    # print("Buffer size: ", len(buffer))


    # import h5py
    
    # replay_buffer = ReplayBuffer.load_hdf5('l2f_controller_buffer_short.hdf5')
    print("Buffer size: ", len(replay_buffer))
    test_td3_bc(replay_buffer)



