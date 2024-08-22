from l2f import Rng, Device, Environment, Parameters, State, Observation, Action,initialize_environment,step,initialize_rng,parameters_to_json,sample_initial_parameters,initial_state
import gymnasium as gym
import numpy as np

# for vectorization
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv
from typing import List, Callable, Optional, Union
import multiprocessing as mp
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Learning2Fly(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        # L2F initialization
        self.device = Device()
        self.rng = Rng()
        self.env = Environment()
        self.params = Parameters()
        self.state = State()
        self.next_state = State()
        self.observation = Observation()
        self.next_observation = Observation()
        self.action = Action()
        initialize_environment(self.device, self.env, self.params)
        initialize_rng(self.device, self.rng, 0)

        # Gym initialization
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,))
        self.global_step_counter = 0   
        self.t = 0

        self.reset()

    def step(self, action):
        self.action.motor_command = action
        step(self.device, self.env, self.params, self.state, self.action, self.next_state, self.rng)
        self.state = self.next_state

        self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)
        self.t += 1

        done = self._check_done()
        reward = self._reward()

        return self.obs, reward, done,done, {}
    
    def reset(self,seed=None):
        sample_initial_parameters(self.device, self.env, self.params, self.rng)
        initial_state(self.device, self.env, self.params, self.state)
        self.global_step_counter += self.t
        self.t = 0
        return np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32), {}
    
    def _reward(self):
        # intial parameters
        Cp = 1 # position weight
        Cv = .0 # velocity weight
        Cq = 0 # orientation weight
        Ca = .0 # action weight og .334, but just learns to fly out of frame
        Cw = .0 # angular velocity weight 
        Crs = .2 # reward for survival
        Cab = 0.0 # action baseline

        # curriculum parameters
        Nc = 1e5 # interval of application of curriculum

        CpC = 1.2 # position factor
        Cplim = 20 # position limit

        CvC = 1.4 # velocity factor
        Cvlim = 1.5 # velocity limit

        CaC = 1.4 # orientation factor
        Calim = .5 # orientation limit

        pos   = self.obs[0:3]
        vel   = self.obs[3:6]
        q     = self.obs[6:10]
        qd    = self.obs[10:13]

        # curriculum
        if self.global_step_counter % Nc == 0:
            Cp = min(Cp*CpC, Cplim)
            Cv = min(Cv*CvC, Cvlim)
            Ca = min(Ca*CaC, Calim)

        # r[0] = max(-1e3,-Cp*np.sum((pos-pset)**2) \
        #         - Cv*np.sum((vel)**2) \
        #             - Ca*np.sum((motor_commands-Cab)**2) \
        #                 - Cw*np.sum((qd)**2) \
        #                     + Crs)
        # print("pos penalty: ", -Cp*np.sum((pos)**2))
        # print("vel penalty: ", - Cv*np.sum((vel)**2))
        # print("action penalty: ", - Ca*np.sum((motor_commands-Cab)**2))
        # print("orientation penalty: ", -Cq*(1-q[0]**2)) aims to keep upright!
        # print("angular velocity penalty: ", - Cw*np.sum((qd)**2))
        # print("reward for survival: ", Crs)

        # in theory pos error max sqrt( .6)*2.5 = 1.94
        # vel error max sqrt(1000)*.005 = 0.158
        # qd error max sqrt(1000)*.00 = 0.
        # should roughly be between -2 and 2
        r = - Cv*np.sum((vel)**2) \
                - Ca*np.sum((np.array(self.action.motor_command)-Cab)**2) \
                    -Cq*(1-q[0]**2)\
                    - Cw*np.sum((qd)**2) \
                        + Crs \
                            -Cp*np.sum((pos)**2) 
        return r
    
    def _check_done(self):
        done = False
        pos_threshold = np.sum((np.abs(self.obs[0:3])>1.5))
        velocity_threshold = np.sum((np.abs(self.obs[3:6]) > 1000))
        angular_threshold  = np.sum((np.abs(self.obs[10:13]) > 1000))
        time_threshold = self.t>500

        # print("pos_threshold: ", pos_threshold)
        # print("velocity_threshold: ", velocity_threshold)
        # print("angular_threshold: ", angular_threshold)
        # print("time_threshold: ", time_threshold)

        if any(np.isnan(self.obs)):
            done = True
        elif pos_threshold or velocity_threshold or angular_threshold or time_threshold:
            done = True
            

        return done


def create_learning2fly_env():
    try:
        return Learning2Fly()
    except Exception as e:
        print(f"Error creating Learning2Fly environment: {e}")
        traceback.print_exc()
        return None

class SubprocVectorizedL2F(SubprocVectorEnv):
    def __init__(self, env_num: int = 4):
        env_fns = [create_learning2fly_env for _ in range(env_num)]
        super().__init__(env_fns)
        logger.info(f"Initialized TianshouSubprocVectorizedL2F with {env_num} environments")

    def reset(self, 
              mask: Optional[Union[np.ndarray, List[int]]] = None,
              *,
              seed: Optional[Union[int, List[int]]] = None,
              options: Optional[dict] = None):
        try:
            if mask is None:
                mask = list(range(self.env_num))
            elif isinstance(mask, int):
                mask = [mask]
            
            reset_kwargs = {"seed": seed, "options": options}
            reset_kwargs = {k: v for k, v in reset_kwargs.items() if v is not None}

            for env_id in mask:
                self.send(env_id, ("reset", reset_kwargs))

            results = []
            for env_id in mask:
                try:
                    results.append(self.recv(env_id))
                except EOFError:
                    logger.error(f"EOFError encountered for environment {env_id}. Attempting to recreate...")
                    self._reset_env(env_id, **reset_kwargs)
                    results.append(self.recv(env_id))

            obs_list, info_list = zip(*results)
            return np.stack(obs_list), info_list
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            traceback.print_exc()
            raise

    def step(self, action):
        try:
            return super().step(action)
        except EOFError as e:
            logger.error(f"EOFError in step: {e}")
            self._reset_all_envs()
            obs, info = self.reset()
            return obs, np.zeros(self.env_num), np.ones(self.env_num, dtype=bool), np.ones(self.env_num, dtype=bool), info
        except Exception as e:
            logger.error(f"Error in step: {e}")
            traceback.print_exc()
            raise

    def _reset_env(self, env_id, **kwargs):
        self.close_env(env_id)
        self.workers[env_id] = mp.Process(target=self.worker_fn, args=(self.env_fns[env_id], self.pipes[env_id][1]))
        self.workers[env_id].start()
        self.send(env_id, ("reset", kwargs))

    def _reset_all_envs(self, **kwargs):
        logger.info("Resetting all environments")
        for env_id in range(self.env_num):
            self._reset_env(env_id, **kwargs)

    def close(self):
        try:
            super().close()
        except ConnectionResetError:
            logger.error("ConnectionResetError during close. Some environments may not have closed properly.")
        except Exception as e:
            logger.error(f"Error during close: {e}")
            traceback.print_exc()

class ShmemVectorizedL2F(ShmemVectorEnv):
    '''
    Optimized for CPU usage with sub processes, but uses a shared buffer to share experiences.'''
    def __init__(self, env_num: int = 4):
        env_fns = [create_learning2fly_env for _ in range(env_num)]
        super().__init__(env_fns)

    def reset(self, 
              mask: Optional[Union[np.ndarray, List[int]]] = None,
              *,
              seed: Optional[Union[int, List[int]]] = None,
              options: Optional[dict] = None,
              env_id: Optional[Union[int, List[int]]] = None):
        if env_id is not None:
            if isinstance(env_id, int):
                env_id = [env_id]
            mask = env_id

        return super().reset(mask=mask, seed=seed, options=options)

        
if __name__=='__main__':
    from stable_baselines3.common.env_checker import check_env
    env = Learning2Fly()
    check_env(env)
    # register the env
    gym.register('L2F',Learning2Fly())