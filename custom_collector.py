import gymnasium as gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
# from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import (
    Batch,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    CachedReplayBuffer,
    to_numpy,
)
from tianshou.data import Collector

class StackPreprocessor:
    def __init__(self, stack_num: int, data_key: str = "obs") -> None:
        self.stack_num = stack_num
        self.data_key = data_key

def add_rolout(buffer: ReplayBuffer, rollout: Batch) -> None:
        """Add a rollout into replay buffer.

        :param Batch rollout: the input rollout. "obs", "act", "rew",
            "terminated", "truncated", "obs_next" are required keys.
        """
        assert set(["obs", "act", "rew", "terminated", "truncated", "obs_next"]
                   ).issubset(rollout.keys())

        for i in range(len(rollout.obs)):
            buffer.add(Batch(
                obs=rollout.obs[i],
                act=rollout.act[i],
                rew=rollout.rew[i],
                terminated=rollout.terminated[i],
                truncated=rollout.truncated[i],
                obs_next=rollout.obs_next[i],
            ))

class ParallelCollector(Collector):
    """Parallel Collector enables the policy to interact with envs that inherintly simulate multiple objects at the same time with \
    exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param n_parallel: number of parallel simulated objects (eg FastPyDroneSim)
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive with only "obs" when the collector resets
    the environment, and will receive four keys "obs_next", "rew", "done", "info" in a
    normal env step. It returns either a dict or a :class:`~tianshou.data.Batch` with
    the modified keys and values. Examples are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        # super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.env = env
        self.env_num = len(env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = env.action_space
        # avoid creating attribute outside __init__
        self.reset()

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param bool reset_buffer: if true, reset the replay buffer that is attached
            to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={},
            act={},
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
            self.reset_stat()

    def reset_stat(self) -> None:
            """Reset the statistic variables."""
            super().reset_stat()

    def reset_buffer(self, keep_statistics:bool=False) -> None:
            """Reset the data buffer."""
            super().reset_buffer(keep_statistics)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Reset all of the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs, info = self.env.reset(**gym_reset_kwargs)
        if self.preprocess_fn:
            processed_data = self.preprocess_fn(
                obs=obs, info=info, env_id=np.arange(self.env_num)
            )
            obs = processed_data.get("obs", obs)
            info = processed_data.get("info", info)
        self.data.info = info
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == np.object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def collect(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            render: Optional[float] = None,
            no_grad: bool = True,
        ) -> Dict[str, Any]:
            """Collect a specified number of step or episode.

            To ensure unbiased sampling result with n_episode option, this function will
            first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
            episodes, they will be collected evenly from each env.

            :param int n_step: how many steps you want to collect.
            :param int n_episode: how many episodes you want to collect.
            :param bool random: whether to use random policy for collecting data. Default
                to False.
            :param float render: the sleep time between rendering consecutive frames.
                Default to None (no rendering).
            :param bool no_grad: whether to retain gradient in policy.forward(). Default to
                True (no gradient retaining).

            .. note::

                One and only one collection number specification is permitted, either
                ``n_step`` or ``n_episode``.

            :return: A dict including the following keys

                * ``n/ep`` collected number of episodes.
                * ``n/st`` collected number of steps.
                * ``rews`` array of episode reward over collected episodes.
                * ``lens`` array of episode length over collected episodes.
                * ``idxs`` array of episode start index in buffer over collected episodes.
            """
            # assert not self.env.is_async, "Please use AsyncCollector if using async venv."
            
            if n_step is not None:
                assert n_episode is None, (
                    f"Only one of n_step or n_episode is allowed in Collector."
                    f"collect, got n_step={n_step}, n_episode={n_episode}."
                )
                assert n_step > 0
                if not n_step % self.env_num == 0:
                    warnings.warn(
                        f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                        "which may cause extra transitions collected into the buffer."
                    )
                ready_env_ids = np.arange(self.env_num)
            elif n_episode is not None:
                assert n_episode > 0
                assert n_episode % self.env_num == 0, (
                    f"n_episode={n_episode} is not a multiple of #env ({self.env_num})."
                )
                ready_env_ids = np.arange(min(self.env_num, n_episode))
                self.data = self.data[:min(self.env_num, n_episode)]
            else:
                raise TypeError("Please specify at least one (either n_step or n_episode) "
                                "in AsyncCollector.collect().")

            start_time = time.time()

            step_count = 0
            episode_count = 0
            episode_rews = []
            episode_lens = []
            episode_start_indices = []

            while True:
                assert len(self.data) == len(ready_env_ids)
                # restore the state: if the last state is None, it won't store
                last_state = self.data.policy.pop("hidden_state", None)

                # get the next action
                if random:
                    self.data.update(
                        act=[self._action_space[i].sample() for i in ready_env_ids])
                else:
                    if no_grad:
                        with torch.no_grad():  # faster than retain_grad version
                            # self.data.obs will be used by agent to get result
                            result = self.policy(self.data, last_state)
                    else:
                        result = self.policy(self.data, last_state)
                    # update state / act / policy into self.data
                    policy = result.get("policy", Batch())
                    assert isinstance(policy, Batch)
                    state = result.get("state", None)
                    if state is not None:
                        policy.hidden_state = state  # save state into buffer
                    act = to_numpy(result.act)
                    if self.exploration_noise:
                        act = self.policy.exploration_noise(act, self.data)
                    
                    
                    self.data.update(policy=policy, act=act)

                # step in env
                obs_next, rew, done,_, info = self.env.step(np.array(self.data.act,dtype=np.float32))

                self.data.update(obs_next=obs_next, rew=rew, terminated=done, truncated=done, info=info)
                if self.preprocess_fn:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            terminated=self.data.done, 
                            truncated=self.data.done,
                            info=self.data.info,
                            policy=self.data.policy,
                            env_id=ready_env_ids,
                            act=self.data.act,
                        ))

                if render:
                    self.env.render()
                    if render > 0 and not np.isclose(render, 0):
                        time.sleep(render)

                # add data into the buffer
                ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                    self.data, buffer_ids=ready_env_ids)

                # collect statistics
                step_count += len(ready_env_ids)

                if np.any(done):
                    env_ind_local = np.where(done)[0]
                    env_ind_global = ready_env_ids[env_ind_local]
                    episode_count += len(env_ind_local)
                    episode_lens.append(ep_len[env_ind_local])
                    episode_rews.append(ep_rew[env_ind_local])
                    episode_start_indices.append(ep_idx[env_ind_local])
                    # now we copy obs_next to obs, but since there might be
                    # finished episodes, we have to reset finished envs first.
                    obs_reset = self.env.reset(int(np.mean(env_ind_global)))
                    if self.preprocess_fn:
                        obs_reset = self.preprocess_fn(obs=obs_reset).get("obs", obs_reset)
                    self.data.obs_next[env_ind_local] = obs_reset[0]
                    for i in env_ind_local:
                        self._reset_state(i)

                    # remove surplus env id from ready_env_ids
                    # to avoid bias in selecting environments
                    if n_episode:
                        surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                        if surplus_env_num > 0:
                            mask = np.ones_like(ready_env_ids, np.bool)
                            mask[env_ind_local[:surplus_env_num]] = False
                            ready_env_ids = ready_env_ids[mask]
                            self.data = self.data[mask]

                self.data.obs = self.data.obs_next

                if (n_step and step_count >= n_step) or \
                        (n_episode and episode_count >= n_episode):
                    break

            # generate statistics
            self.collect_step += step_count
            self.collect_episode += episode_count
            self.collect_time += max(time.time() - start_time, 1e-9)

            if n_episode:
                self.data = Batch(obs={}, act={}, rew={}, done={},
                                obs_next={}, info={}, policy={})
                self.reset_env()

            if episode_count > 0:
                rews, lens, idxs = list(map(
                    np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
            else:
                rews, lens, idxs = np.array([]), np.array([], np.int16), np.array([], np.int16)

            return {
                "n/ep": episode_count,
                "n/st": step_count,
                "rews": rews,
                "lens": lens,
                "idxs": idxs,
            }

    def collect_rollout(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            render: Optional[float] = None,
            no_grad: bool = True,
        ) -> Dict[str, Any]:
            """Collect a specified number of step or episode.

            To ensure unbiased sampling result with n_episode option, this function will
            first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
            episodes, they will be collected evenly from each env.

            :param int n_step: how many steps you want to collect.
            :param int n_episode: how many episodes you want to collect.
            :param bool random: whether to use random policy for collecting data. Default
                to False.
            :param float render: the sleep time between rendering consecutive frames.
                Default to None (no rendering).
            :param bool no_grad: whether to retain gradient in policy.forward(). Default to
                True (no gradient retaining).

            .. note::

                One and only one collection number specification is permitted, either
                ``n_step`` or ``n_episode``.

            :return: A dict including the following keys

                * ``n/ep`` collected number of episodes.
                * ``n/st`` collected number of steps.
                * ``rews`` array of episode reward over collected episodes.
                * ``lens`` array of episode length over collected episodes.
                * ``idxs`` array of episode start index in buffer over collected episodes.
            """
            if not hasattr(self.env, 'step_rollout'):
                raise NotImplementedError("This environment does not support rollout collection.")
            # assert not self.env.is_async, "Please use AsyncCollector if using async venv."
            if n_step is not None:
                assert n_episode is None, (
                    f"Only one of n_step or n_episode is allowed in Collector."
                    f"collect, got n_step={n_step}, n_episode={n_episode}."
                )
                assert n_step > 0
                if not n_step % self.env_num == 0:
                    warnings.warn(
                        f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                        "which may cause extra transitions collected into the buffer."
                    )
                ready_env_ids = np.arange(self.env_num)
            elif n_episode is not None:
                raise NotImplementedError("n_episode is not supported for rollout collection. Please pass a number of episodes.")

            start_time = time.time()

            # interact with environment
            obs, act, rew, dones, obs_next, info = self.env.step_rollout(self.policy, n_step=n_step, tianshou_policy=True)
            print("Collection itself: ", time.time()-start_time)

            # add rollout into the replay buffer, cut it up into single transitions
            add_rolout(self.buffer, Batch(obs=obs, act=act, rew=rew, terminated=dones, truncated=dones, obs_next=obs_next, info=info))
            
            step_count = n_step
            episode_count = self.env_num
            # episode_rews = []
            # episode_lens = []
            # episode_start_indices = []

            if self.exploration_noise:
                raise warnings.warn("Exploration noise is not supported for rollout collection.")


            # collect statistics
            step_count += len(ready_env_ids)

            # all envs are done
            episode_count += self.env_num
            # episode_lens.append(n_step)
            # episode_rews.append(rew)



            # remove surplus env id from ready_env_ids
            # to avoid bias in selecting environments
            if n_episode:
                raise NotImplementedError("n_episode is not supported for rollout collection.")
 

            # generate statistics
            self.collect_step += step_count
            self.collect_episode += episode_count
            self.collect_time += max(time.time() - start_time, 1e-9)

            if n_episode:
                self.data = Batch(obs={}, act={}, rew={}, done={},
                                obs_next={}, info={}, policy={})
                self.reset_env()

            # if episode_count > 0:
            #     rews, lens, idxs = list(map(
            #         np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
            # else:
            #     rews, lens, idxs = np.array([]), np.array([], np.int16), np.array([], np.int16)

            return {
                "n/ep": episode_count,
                "n/st": step_count,
                # "rews": rews,
                # "lens": lens,
                # "idxs": idxs,
            }


if __name__ == "__main__":
    from gym_sim import Drone_Sim
    import torch.nn as nn
    from tianshou.policy import SACPolicy
    from tianshou.utils.net.continuous import ActorProb, Critic
    from tianshou.utils.net.common import Net

    N_envs = 100
    env = Drone_Sim(N_cpu=N_envs, action_buffer=False)
    import networks as n
    policy = n.Actor_ANN(17,4,1)
    
    observation_space = env.observation_space.shape or env.observation_space.n
    action_space = env.action_space.shape or env.action_space.n

    net_a = Net(state_shape=observation_space,
                hidden_sizes=[64,64], device='cpu')
    actor = ActorProb(
        net_a,
        action_space,
        unbounded=True,
        conditioned_sigma=True,
    )
    net_c1 = Net(state_shape=observation_space,action_shape=action_space,
                 hidden_sizes=[64,64],
                 concat=True,)
    net_c2 = Net(state_shape=observation_space,action_shape=action_space,
                 hidden_sizes=[64,64],
                 concat=True,)
    critic1 = Critic(net_c1, device='cpu')
    critic2 = Critic(net_c2, device='cpu')

    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)

    policy = SACPolicy(actor=actor, actor_optim=actor_optim, \
                       critic=critic1, critic_optim=critic_optim,\
                        critic2=critic2, critic2_optim=critic2_optim\
                            ,action_space=env.action_space,observation_space=env.observation_space, action_scaling=True)

    buffer=VectorReplayBuffer(total_size=200000,buffer_num=N_envs, stack_num=1)
    collector = ParallelCollector(policy=policy, env=env, buffer=buffer)
    import time

    print("Starting rollout collection (1 000 000 steps)...") 
    collector.reset_buffer()
    start = time.time()
    collector.collect_rollout(n_step=1e3, random=False)
    t_roll = time.time()-start
    print("Done in: ",t_roll)
    print(len(buffer))

    t_ind = 92.5
    print("Speedup: ", t_ind/t_roll)

    sample = buffer.sample(2)
    # print(sample, len(sample))