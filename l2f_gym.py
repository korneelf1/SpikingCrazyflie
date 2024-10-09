from l2f import Rng, Device, Environment, Parameters, State, Observation, Action,initialize_environment,step,initialize_rng,parameters_to_json,sample_initial_parameters,initial_state, sample_initial_state, observe
import gymnasium as gym
import numpy as np

# for vectorization
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv
from typing import List, Callable, Optional, Union
import multiprocessing as mp
import traceback
import logging
import helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import wandb
import math
def observe_rotation_matrix(state):
    # Extract the quaternion components from the state matrix
    qw = state[0]
    qx = state[1]
    qy = state[2]
    qz = state[3]

    # Initialize the observation matrix (assuming 18 columns, and 1 row for now)
    observation = np.zeros((9))

    # Compute the 3x3 rotation matrix from the quaternion
    observation[0] = 1 - 2 * qy * qy - 2 * qz * qz
    observation[1] = 2 * qx * qy - 2 * qw * qz
    observation[2] = 2 * qx * qz + 2 * qw * qy
    observation[3] = 2 * qx * qy + 2 * qw * qz
    observation[4] = 1 - 2 * qx * qx - 2 * qz * qz
    observation[5] = 2 * qy * qz - 2 * qw * qx
    observation[6] = 2 * qx * qz - 2 * qw * qy
    observation[7] = 2 * qy * qz + 2 * qw * qx
    observation[8] = 1 - 2 * qx * qx - 2 * qy * qy

    return observation
def power_distribution_force_torque(control, arm_length=0.046, thrust_to_torque=0.005964552, pwm_to_thrust_a=0.091492681, pwm_to_thrust_b=0.067673604):
    # rescale control from -1 - 1 to  
   
    motor_forces = [0.0] * 4

    arm = 0.707106781 * arm_length
    roll_part = 0.25 / arm * control[0]/100
    pitch_part = 0.25 / arm * control[1]/100
    thrust_part = 0.25 * (control[2]+1)/2  # N (per rotor)
    yaw_part = 0.25 * (control[3] / thrust_to_torque)/100

    motor_forces[0] = thrust_part - roll_part - pitch_part - yaw_part
    motor_forces[1] = thrust_part - roll_part + pitch_part + yaw_part
    motor_forces[2] = thrust_part + roll_part + pitch_part - yaw_part
    motor_forces[3] = thrust_part + roll_part - pitch_part + yaw_part

    print(motor_forces)
    action = np.zeros(4)
    for motor_index in range(4):
        motor_force = motor_forces[motor_index]
        
        if motor_force < 0.0:
            motor_force = 0.0

        motor_pwm = (-pwm_to_thrust_b + math.sqrt(pwm_to_thrust_b**2 + 4.0 * pwm_to_thrust_a * motor_force)) / (2.0 * pwm_to_thrust_a)
        
        # rescale to -1 - 1
        # print(motor_pwm)
        motor_pwm = 2 * (motor_pwm - 0.5)
        # print(motor_pwm)
        motor_pwm = max(-1, min(1, motor_pwm))
        
        action[motor_index] = motor_pwm
    return action
        # motor_thrust_uncapped['list'][motor_index] = motor_pwm * 65535  # UINT16_MAX in C

# Example usage:
# control = {'torqueX': 0, 'torqueY': 0, 'torqueZ': 0, 'thrustSi': 0}
# motor_thrust_uncapped = {'list': [0] * 4}
# power_distribution_force_torque(control, motor_thrust_uncapped, arm_length=0.1, thrust_to_torque=0.05, pwm_to_thrust_a=0.01, pwm_to_thrust_b=0.02)

class Learning2Fly(gym.Env):
    def __init__(self, fast_learning=True,seed=None) -> None:

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
        if seed is None:
            seed = np.random.randint(0, 2**32-1)
        # print("Environment initialized with seed: ", seed)
        initialize_rng(self.device, self.rng, seed)

        # curriculum parameters
        self.Nc = 7e5 # interval of application of curriculum, roughly 10 epochs

        sample_initial_parameters(self.device, self.env, self.params, self.rng)

        self.params.parameters.dynamics.mass *= 0.1
        sample_initial_state(self.device, self.env, self.params, self.state, self.rng)

        observe(self.device, self.env, self.params, self.state, self.observation, self.rng)

        # Gym initialization
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.observation.observation),))
        
        self.global_step_counter = 0   
        self.t = 0
        self.fast_learning = fast_learning # if True, the environment will use soft terminal conditions initially

        if not self.fast_learning:
            # Reward parameters
                    # intial parameters
            self.Cp = .1 # position weight
            self.Cv = .01 # velocity weight
            self.Cq = .01 # orientation weight
            self.Ca = .1 # action weight og .334, but just learns to fly out of frame
            self.Cw = .00 # angular velocity weight 
            self.Crs = 1 # reward for survival
            self.Cab = 2*.334-1 # action baseline
        else:
            # Reward parameters
            self.Cp = 1.0
            self.Cv = .0005# velocity weight
            self.Cq = .25 # orientation weight
            self.Ca = .01 # action weight og .334, but just learns to fly out of frame
            self.Cw = .00 # angular velocity weight 
            self.Crs = 1 # reward for survival
            self.Cab = 2*.334-1 # action baseline

        # Curriculum parameters
        self.CpC = 1.2 # position factor
        self.Cplim = 20 # position limit

        self.CvC = 1.4 # velocity factor
        self.Cvlim = .5 # velocity limit

        self.Cac = 1.4  # action factor
        self.Calim = .5 # action limit


        # reset environmnet
        self.reset()

    @property
    def obs(self):
        return np.array(self.observation.observation,dtype=np.float32)

    def step(self, action):
        # self.action.motor_command = power_distribution_force_torque(action.reshape((4,)))
        self.action.motor_command = action.reshape((4,))
        # print(self.action.motor_command)
        step(self.device, self.env, self.params, self.state, self.action, self.next_state, self.rng)
        self.state = self.next_state

        observe(self.device, self.env, self.params, self.state, self.observation, self.rng)

        self.t += 1

        done = self._check_done()
        # print("Step: ", self.t, "Done: ", done)
        reward = self._reward()
        # print(self.obs)
        return self.obs, reward, done,done, {}
    
    def reset(self,seed=None):
        sample_initial_parameters(self.device, self.env, self.params, self.rng)

        self.params.parameters.dynamics.mass *= 0.1
        sample_initial_state(self.device, self.env, self.params, self.state, self.rng)

        observe(self.device, self.env, self.params, self.state, self.observation, self.rng)
        # self.global_step_counter += self.t
        self.t = 0


        return self.obs, {}
    
    def _reward(self):
        pos   = np.array(self.state.position)
        vel   = np.array(self.state.linear_velocity)
        q     = np.array(self.state.orientation)
        qd    = np.array(self.state.angular_velocity)

        self.global_step_counter += 1
        if self.global_step_counter > self.Nc:
            # updating the curriculum parameters
            self.Cp = min(self.Cp*self.CpC, self.Cplim)
            self.Cv = min(self.Cv*self.CvC, self.Cvlim)
            self.Ca = min(self.Ca*self.CaC, self.Calim)
            # self.Crs = max(self.Crs*self.CrsC, self.Crslim)
            print("\n\n\nCurriculum parameters updated:")
            print(f"Position Term: {self.Cp}, Survival Reward: {self.Crs}")
            print("\n\n\n")
            # print("Updating curriculum parameters")
            if wandb.run is not None:
                wandb.run.log({'Position Term':self.Cp,'Survival Reward':self.Crs})
            
            self.Nc += self.Nc

        # 2*math::acos(device.math, 1-math::abs(device.math, state.orientation[3])) in python:
        # orient_penalty = 2*np.acos(1-q[3]**2)

        r = - self.Cv*np.sum((vel)**2) \
                - self.Ca*np.sum((np.array(self.action.motor_command)-self.Cab)**2) \
                    -self.Cq*2*np.arccos(1-q[3]**2)\
                    - self.Cw*np.sum((qd)**2) \
                        + self.Crs \
                            -self.Cp*np.sum((pos)**2) 
        return r
    
    def _check_done(self):
        done = False

        pos   = np.array(self.state.position)
        vel   = np.array(self.state.linear_velocity)
        q     = np.array(self.state.orientation)
        qd    = np.array(self.state.angular_velocity)

        pos_lim = 1.5 if self.fast_learning else .6        
        # if self.curriculum_terminal:
        #     pos_limit = 1.5
        #     pos_min = 0.6
        #     factor = 1/1.5
        #     xy_softening = 10 # to first train hover
        #     if self.global_step_counter%2e4==0:
        #         pos_limit = max(pos_min, pos_limit*factor)
        #         xy_softening = max(1,xy_softening*factor)
        #     z_terminal = self.obs[3]>pos_limit
        #     xy_terminal = np.sum((np.abs(self.obs[0:2])>pos_limit*xy_softening))
        #     pos_threshold = z_terminal + xy_terminal
        # else:
        pos_threshold = np.sum((np.abs(pos)>pos_lim))

        velocity_threshold = np.sum((np.abs(vel) > 1000))
        angular_threshold  = np.sum((np.abs(qd) > 1000))
        time_threshold = self.t>500

        if np.any(np.isnan(self.obs)):
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
    # from stable_baselines3.common.env_checker import check_env
    env = Learning2Fly()
    env.reset()
    env2 = Learning2Fly()
    env2.reset()

    # check_env(env)
    # # register the env
    # gym.register('L2F',Learning2Fly())
    pass