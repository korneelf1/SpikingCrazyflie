from l2f import Rng, Device, Environment, Parameters, State, Observation, Action,initialize_environment,step,initialize_rng,parameters_to_json,sample_initial_parameters,initial_state, sample_initial_state
import gymnasium as gym
import numpy as np

# for vectorization
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv
from typing import List, Callable, Optional, Union
import multiprocessing as mp
import traceback
import logging
from imu import IMU, quaternion_rotation_matrix
from helpers import NumpyDeque, forcetorque_to_rpm, quaternion_to_euler
import torch
from collections import deque


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Learning2Fly(gym.Env):
    def __init__(self, curriculum_terminal=False,
                 seed=None, 
                 euler=False, 
                 imu=False,
                 imu_only=False, 
                 t_history=1, 
                 out_forces=False) -> None:
        '''
        Initializes the Learning2Fly environment.
        Args:
            curriculum_terminal (bool): If True, the environment will use soft terminal conditions initially.
            seed (int): The seed to initialize the environment with.
            euler (bool): If True, the environment will return euler angles rather than quaternions, and return velocity in body frame.
            imu (bool): If True, the environment will return the IMU data, pos AND initial velocity and orientation.
            imu_only (bool): If True, the environment will only return the IMU data and positions.
            t_history (int): The number of timesteps to include in the observation. 1 is only last
            out_forces (bool): If True, the actions are the forces and moments that are tehn converted to rpms
        '''
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
        self.IMU = None
        if imu:
            self.IMU = IMU(noise=np.zeros(6), bias=np.zeros(6), dt=0.01, offset=np.zeros(3))

        initialize_environment(self.device, self.env, self.params)
        if seed is None:
            seed = np.random.randint(0, 2**32-1)
        print("Environment initialized with seed: ", seed)
        initialize_rng(self.device, self.rng, seed)

        self.t_history = t_history
        # Gym initialization
        if out_forces:
            self.action_space = gym.spaces.Box(low=np.array([0,-.1,-.1,-.1]), high=np.array([0.7,.1,.1,.1]), shape=(4,))
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))

        self.euler = euler
        if imu:
            self.imu_history = NumpyDeque(shape=(9*t_history,),device='cpu')
            # state history is needed to allow single transitions with the IMU, you need initial velocity and orientation
            self.states_history = NumpyDeque(shape=(6*t_history,),device='cpu') # holds velocity and orientation in */euler angels

            self.imu_only = imu_only
            if not imu_only:
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9*t_history+6,)) # IMU and pos history, and initial velocity and rotation
            else:
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9*t_history,))
        else:
            if self.euler:
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16*t_history,))
            else:
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17*t_history,))

        # self.obs = NumpyDeque((self.observation_space.shape[0],),device='cpu')
        self.out_forces = out_forces

        self.global_step_counter = 0   
        self.t = 0
        self.curriculum_terminal = curriculum_terminal # if True, the environment will use soft terminal conditions initially

        self.reset()

    def step(self, action):
        if self.out_forces:
            action = forcetorque_to_rpm(action)
        self.action.motor_command = action.reshape((4,))
        dt = step(self.device, self.env, self.params, self.state, self.action, self.next_state, self.rng)

        self.state = self.next_state

        if self.IMU is not None:
            R = quaternion_rotation_matrix(self.state.orientation)

            # transfrom velocity to body frame
            vel_body = np.dot(R.T, self.state.linear_velocity)
        
            self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    

            # simulate the imu
            imu_sim = self.IMU.simulate(self.obs)
            self.imu_history.append(imu_sim)

            # you need a history of info to let network estimate velocities etc
            self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    
            
            # append history of states
            self.states_history.append(np.concatenate([vel_body, quaternion_to_euler(self.state.orientation)]))
            if self.imu_only:
                self.obs = self.imu_history.array.astype(np.float32)

            else:
                self.obs = np.concatenate([self.imu_history.array, self.states_history.array[-6:]]).astype(np.float32) # take oldest velocity and orientation

        else:
            if self.euler:
                R = quaternion_rotation_matrix(self.state.orientation)

                # transfrom velocity to body frame
                vel_body = np.dot(R.T, self.state.linear_velocity)
                self.obs = np.concatenate([self.state.position, quaternion_to_euler(self.state.orientation), vel_body, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    
            else:
                self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    

        self.t += 1

        done = self._check_done()
        
        reward = self._reward()
        
        # self.obs.array[13:] = np.zeros((4,))
        # self.obs[12:] = np.zeros((4,))
        return self.obs, reward, done,done, {}
    
    def reset(self,seed=None):
        sample_initial_parameters(self.device, self.env, self.params, self.rng)

        self.params.parameters.dynamics.mass *= 0.1

        sample_initial_state(self.device, self.env, self.params, self.state, self.rng)
        # print("Initial state: ", self.state.position)
        self.global_step_counter += self.t
        self.t = 0

        if self.IMU is not None:
            self.imu_history.reset()
            self.states_history.reset()
        
        # fill observations with t_history steps
        # for _ in range(self.t_history):
        #     self.step(np.ones((4,))*0.6670265023020774*2-1) # pass hover action

        

        if self.IMU is not None:
            if self.euler:
                R = quaternion_rotation_matrix(self.state.orientation)

                # transfrom velocity to body frame
                vel_body = np.dot(R.T, self.state.linear_velocity)

            self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    

            # simulate the imu
            imu_sim = self.IMU.simulate(self.obs)
            self.imu_history.append(imu_sim)

            # you need a history of info to let network estimate velocities etc
            self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    
            
            if self.imu_only:
                self.obs = self.imu_history.array.astype(np.float32)

            else:
                # append history of states
                # originally it was vel_body and self.state.angular_velocity????
                self.states_history.append(np.concatenate([vel_body, quaternion_to_euler(self.state.orientation)]))
                self.obs = np.concatenate([self.imu_history.array, self.states_history.array[-6:]]).astype(np.float32) # take oldest velocity and orientation

        else:
            if self.euler:
                R = quaternion_rotation_matrix(self.state.orientation)

                # transfrom velocity to body frame
                vel_body = np.dot(R.T, self.state.linear_velocity)
                self.obs = np.concatenate([self.state.position, quaternion_to_euler(self.state.orientation), vel_body, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    
            else:
                self.obs = np.concatenate([self.state.position, self.state.orientation, self.state.linear_velocity, self.state.angular_velocity, self.state.rpm]).astype(np.float32)    
                # print(self.obs)
        return self.obs, {}
    
    def _reward(self):
        # intial parameters
        Cp = 0.20 # position weight
        Cv = .0200 # velocity weight
        Cq = 0 # orientation weight
        Ca = .334 # action weight og .334, but just learns to fly out of frame
        Cw = .00 # angular velocity weight 
        Crs = 1 # reward for survival
        Cab = 0.0 # action baseline

        # curriculum parameters
        Nc = 1e5 # interval of application of curriculum

        CpC = 1.2 # position factor
        Cplim = 20 # position limit

        CvC = 1.4 # velocity factor
        Cvlim = 1.5 # velocity limit

        CaC = 1.4 # orientation factor
        Calim = .5 # orientation limit

        CrsC = .8 # reward for survival factor
        Crslim = .1 # reward for survival limit
        if self.IMU is None:
            pos   = self.obs[0:3]
            vel   = self.obs[3:6]
            q     = self.obs[6:10]
            qd    = self.obs[10:13]
        else:
            pos   = self.obs[0:3]
            vel   = self.obs[-6:-3]
            q     = self.obs[-3:]
            qd    = self.obs[6:9]


        # # curriculum
        # if self.global_step_counter % Nc == 0:
        #     print("Updating curriculum parameters")
        #     # updating the curriculum parameters
        #     Cp = min(Cp*CpC, Cplim)
        #     # Cv = min(Cv*CvC, Cvlim)
        #     # Ca = min(Ca*CaC, Calim)
        #     Crs = max(Crs*CrsC, Crslim)

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
        if self.IMU is None:
            pos   = self.obs[0:3]
            vel   = self.obs[3:6]
            q     = self.obs[6:10]
            qd    = self.obs[10:13]
        else:
            pos   = self.obs[0:3]
            vel   = self.obs[-6:-3]
            q     = self.obs[-3:]
            qd    = self.obs[6:9]
        
        if self.curriculum_terminal:
            pos_limit = 1.
            pos_min = 0.6
            factor = 1/1.5
            xy_softening = 1 # to first train hover
            if self.global_step_counter%2e4==0:
                pos_limit = max(pos_min, pos_limit*factor)
                xy_softening = max(1,xy_softening*factor)
            z_terminal = pos[3]>pos_limit
            xy_terminal = np.sum((np.abs(pos[0:2])>pos_limit*xy_softening))
            pos_threshold = z_terminal + xy_terminal
        else:
            pos_threshold = np.sum((np.abs(pos[0:3])>1.5))

        if self.IMU is None:
            velocity_threshold = np.sum((np.abs(vel[3:6]) > 1000))
            angular_threshold  = np.sum((np.abs(qd[10:13]) > 1000))
        else:
            velocity_threshold = np.sum((np.abs(vel[9:12]) > 1000))
            angular_threshold  = np.sum((np.abs(qd[6:9]) > 1000))
        time_threshold = self.t>500

        if np.any(np.isnan(self.obs)):
            return True
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
    # from stable_baselines3.common.env_checker import check_env
    env = Learning2Fly(t_history=3,imu=True, out_forces=False)
    env.reset()
    # env2 = Learning2Fly()
    # env2.reset()

    # check_env(env)
    # # register the env
    # gym.register('L2F',Learning2Fly())
    action = env.action_space.sample()
    

    hover = 0.6670265023020774*2-1
    action = np.ones((4,))*hover

    for i in range(100):
        env.step(action)
        print(env.obs.array[0:3])

    pass