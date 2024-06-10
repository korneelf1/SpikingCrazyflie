#!/usr/bin/env python3
"""
    Vectorized quadrotor simulation with websocket pose output

    
    maybe it would be better to provide step function which takes a model and performs a complete rollout where model is optimized


    Copyright (C) 2024 Till Blaha -- TU Delft

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import numba as nb
from collections import deque
from tqdm import tqdm
import asyncio
from time import time
from libs.wsInterface import wsInterface, dummyInterface
from crafts import QuadRotor, Rotor
from numba import cuda
import gymnasium as gym
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

GRAVITY = 9.80665

class Drone_Sim(gym.Env):
    metadata = {
        # 'runtime.vectorized': True,
    }

    def __init__(self, gpu=False, viz=True, log=True, realtime=False, action_buffer=True, dt=0.01, T=2, N_cpu=1, spiking_model=None):
        super(Drone_Sim, self).__init__()
        '''
        Vectorized quadrotor simulation with websocket pose output
        NOTE: CURRENTLY JITTER AND KERNELLER ARE GLOBAL VARIABLES!
        
        Args:
            gpu (bool): run on the gpu using cuda
            viz (bool): stream pose to a websocket connection
            log (bool): log states from gpu back to the CPU (setting False has no speedup on CPU)
            realtime (bool): wait every timestep to try and be real time
            dt (float): step time is dt seconds (forward Euler)
            T (float): run for T seconds
            N_cpu (int): number of simulations to run in parallel
            spiking_model (object): spiking model, which will be reset at environment reset (spiking_model.reset_hidden())
            '''

        ### sim config ###
        self.gpu = gpu              # run on the self.gpu using cuda
        self.viz = viz              # stream pose to a websocket connection
        self.log = log              # log states from self.gpu back to the CPU (setting False has no speedup on CPU)
        self.realtime = realtime    # wait every timestep to try and be real time

        # length / number of parallel sims
        self.action_buffer = action_buffer
        self.dt = dt                # step time is self.dt seconds (forward Euler)
        self.T = T                  # run for self.T seconds
        if self.gpu:                # number of simulations to run in parallel
            self.blocks = 128       # 128 or 256 seem best. Should be multiple of 32
            self.threads_per_block = 64 # depends on global memorty usage. 256 seems best without. Should be multiple of 64
            # self.dt 0.01, self.T 10, no viz, self.log_interval 0, no controller, self.blocks 256, threads 256, self.gpu = True --> 250M ticks/sec
            self.N = self.blocks * self.threads_per_block
        else:
            self.N = N_cpu # cpu
        N = self.N

        self.spiking_model = spiking_model
        # initial states: 0:3 pos, 3:6 vel, 6:10 quaternion, 10:13 body rates Omega, 13:17 motor speeds omega

        
        if action_buffer: # add last 25 inputs as observation
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17+25*4,), dtype=np.float32)
            self.action_history = deque(maxlen=25)
            for i in range(25):
                self.action_history.append(np.zeros((self.N,4),dtype=np.float32))

        else:
        # create gymnasium observation and action space
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0., high=1., shape=(4,), dtype=np.float32)

        self.reset()

        # other settings (vizualition and logging)
        self.viz_interval = 0.05 # visualize every self.viz_interval simulation-seconds
        self.Nviz = 512 # max number of quadrotors to visualize
        self.log_interval = 1    # log state every x iterations. Too low may cause out_of_memory on the self.gpu. False == 0

        # create the drones
        self._create_drones()

        # precompute stuff
        self.itaus = 1. / self.taus


        # create controller (legacy) and position setpoints
        # (FIXME: should be a weighted pseudoinverse!!)
        self.G1pinvs = np.linalg.pinv(self.G1s) / (self.omegaMaxs*self.omegaMaxs)[:, :, np.newaxis]

        # position setpoints --> uniform on rectangular grid
        grid_size = int(np.ceil(np.sqrt(N)))
        x_vals = np.linspace(-7, 7, grid_size)
        y_vals = np.linspace(-7, 7, grid_size)
        X, Y = np.meshgrid(x_vals, y_vals)
        vectors = np.column_stack((X.ravel(), Y.ravel(), -1.5*np.ones_like(X.ravel())))
        self.pSets = vectors[:N].astype(np.float32) # position setpoint

        # import compute kernels
        global kerneller; global jitter
        if self.gpu:
            jitter = lambda signature: nb.cuda.jit(signature, fastmath=False, device=True, inline=False)
            kerneller = lambda signature: nb.cuda.jit(signature, fastmath=False, device=False)
            from libs.gpuKernels import step as kernel_step
            self.kernel_step = kernel_step
        else:
            jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
            kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
            nb.set_num_threads(max(nb.config.NUMBA_DEFAULT_NUM_THREADS-4, 1))
            from libs.cpuKernels import step as kernel_step
            from libs.cpuKernels import reward_function 
            self.kernel_step = kernel_step
            self.reward_function = reward_function

        # allocate sim data
        self.log_interval = log*5
        self.iters = int(self.T / self.dt)
        self.Nlog = int(self.iters / self.log_interval) if self.log_interval > 0 else 0

        self.us = np.random.random((N, 4)).astype(np.float32) # inputs (motor speeds)

        # create logs
        self._create_logs()

        if viz:
            print("initializing websocket. Awaiting connection... ")
            self.wsI = wsInterface(8765)
        else:
            self.wsI = dummyInterface()

        # reward scheduling
        # intial parameters
        self.Cp = 2.5 # position weight
        self.Cv = .005 # velocity weight
        self.Cq = 2.5 # orientation weight
        self.Ca = .005 # action weight
        self.Cw = 0.0 # angular velocity weight
        self.Crs = 2 # reward for survival
        self.Cab = 0.334 # action baseline

        # target parameters
        self.CpT = 1.2 # position weight
        self.CvT = .5 # velocity weight
        self.CqT = 2.5 # orientation weight
        self.CaT = .5 # action weight
        self.CwT = 0.0 # angular velocity weight
        self.CrsT = 2 # reward for survival
        self.CabT = 0.334 # action baseline

        # curriculum parameters
        self.Nc = 1e5 # interval of application of curriculum

        self.CpC = 1.2 # position factor
        self.Cplim = 20 # position limit

        self.CvC = 1.4 # velocity factor
        self.Cvlim = .5 # velocity limit
        
        self.CaC = 1.4 # orientation factor
        self.Calim = .5 # orientation limit

        self.r = np.empty(self.N, dtype=np.float32)

        # gym specific stuff
        self.episode_counter = 0
        self.global_step_counter = 0 # used for reward curriculum
        self.iters = int(self.T / self.dt)

    # def return_kernels(self):
    #     '''Return kernels for the environment
    #     REQUIRED FOR THE CPUKERNELS TO WORK!!!'''
    #     return self.jitter, self.kerneller
    @property
    def n(self):
        return self.N
    @property
    def num_envs(self):
        return self.N
    
    def _create_drones(self):
        '''Creates drones, crazyflies
        parameters retrieved from:
        https://github.com/arplaboratory/learning_to_fly_media/blob/ae72456e879137b840b9dfde366253886c3ec131/parameters.pdf
        '''
        # mass of crazyfly
        self.m = 0.027
        self.Ixx = 3.85e-6
        self.Iyy = 3.85e-6
        self.Izz = 5.9675e-6
        self.I = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]])

        self.G1s = np.empty((self.N, 4, 4), dtype=np.float32)
        self.G2s = np.empty((self.N, 1, 4), dtype=np.float32)
        self.omegaMaxs = np.empty((self.N, 4), dtype=np.float32) # max rpm (in rads)? if so, max 21702 rpm -> 21702/60*2pi rad/sec
        self.taus = np.empty((self.N, 4), dtype=np.float32) # RPM time constant? if so, 0.15sec

        max_rads = 21702/60*2*3.1415
        for i in tqdm(range(self.N), desc="Building crafts"):
            q = QuadRotor()
            q.setInertia(self.m, self.I)
            q.rotors.append(Rotor([-0.028, 0.028, 0], dir='cw', wmax = max_rads, tau=0.15, Izz= 3.16e-10,k=0.005964552)) # rotor 3
            q.rotors.append(Rotor([0.028, 0.028, 0], dir='ccw', wmax = max_rads, tau=0.15, Izz= 3.16e-10,k=0.005964552)) # rotor 4
            q.rotors.append(Rotor([-0.028, -0.028, 0], dir='ccw', wmax = max_rads, tau=0.15, Izz= 3.16e-10,k=0.005964552)) # rotor 2	
            q.rotors.append(Rotor([0.028, -0.028, 0], dir='cw', wmax = max_rads, tau=0.15, Izz= 3.16e-10,k=0.005964552)) # rotor 1

            q.fillArrays(i, self.G1s, self.G2s, self.omegaMaxs, self.taus)

    def _create_logs(self,):
        '''Creates logs'''	
        self.xs_log = np.empty(
            (self.N, self.Nlog, 17), dtype=np.float32)
        self.xs_log[:] = np.nan

    def reset(self,seed=None):
        super().reset(seed=seed)
         # initial states: 0:3 pos, 3:6 vel, 6:10 quaternion, 10:13 body rates Omega, 13:17 motor speeds omega
        x0 = np.random.random((self.N, 17)).astype(np.float32) - 0.5
        x0[:, 6:10] /= np.linalg.norm(x0[:, 6:10], axis=1)[:, np.newaxis] # quaternion needs to be normalized

        self.xs = x0.copy() # states
        self.t = 0
        self.episode_counter = 0
        if self.spiking_model:
            self.spiking_model.reset_hidden()

        if self.action_buffer:
            for i in range(25):
                self.action_history.append(np.zeros((self.N,4),dtype=np.float32))
            return np.concatenate((x0, np.array(self.action_history).reshape(self.N,100)), axis=1), {}
        # YOU CAN RESET YOUR MODEL IN THE ENVIRONMENT RESET FUNCTION!!!!!!!
        return self.xs,{} # state, info

    def _simulate_step(self):
        self.log_idx =0
        if self.gpu:
            self.kernel_step[self.blocks,self.threads_per_block](self.d_xs, self.d_us, self.d_itaus, self.d_omegaMaxs, self.d_G1s, self.d_G2s, self.dt, self.log_idx, self.d_xs_log)
        else:
            self.kernel_step(self.xs, self.us, self.itaus, self.omegaMaxs, self.G1s, self.G2s, self.dt, self.log_idx, self.xs_log)

    def _compute_reward(self):
        '''Compute reward, reward function from learning to fly in 18sec paper
        TODO: optimize with cpuKernels and gpuKernels'''	
        orientation = self.xs[:, 6:10]
        position = self.xs[:, 0:3]
        velocity = self.xs[:, 3:6]
        angular_velocity = self.xs[:, 10:13]
        motor_speeds = self.xs[:, 13:16]

        # curriculum
        if self.global_step_counter % self.Nc == 0:
            self.Cp = min(self.Cp*self.CpC, self.Cplim)
            self.Cv = min(self.Cv*self.CvC, self.Cvlim)
            self.Ca = min(self.Ca*self.CaC, self.Calim)
        
        # sum over axis 1, along position and NOT allong nr of drones
        r = -self.Cp*np.abs(position - self.pSets**2).sum(axis=1) \
            - self.Cv*np.abs(velocity**2).sum(axis=1) \
                - self.Cq*(1-orientation**2).sum(axis=1) \
                    - self.Ca*np.abs((motor_speeds-self.Cab)).sum(axis=1) \
                        - self.Cw*np.abs(angular_velocity).sum(axis=1) \
                            + self.Crs
        
        return r
    

    def render(self, mode='human'):
        pass
    def _move_to_cuda(self):
        '''Move data to cuda'''
        self.d_us = cuda.to_device(self.us)
        self.d_xs = cuda.to_device(self.xs)
        self.d_xs_log = cuda.to_device(self.xs_log)
        self.d_itaus = cuda.to_device(self.itaus)
        self.d_omegaMaxs = cuda.to_device(self.omegaMaxs)
        self.d_G1s = cuda.to_device(self.G1s)
        self.d_G2s = cuda.to_device(self.G2s)

        # position setpoints
        self.d_pSets = cuda.to_device(self.pSets)
        self.d_G1pinvs = cuda.to_device(self.G1pinvs)
        cuda.synchronize()

    async def _step(self):
        self.episode_counter += 1
        self.global_step_counter += 1

        if self.gpu:
            self._move_to_cuda()

            # log_idx = -1
            # if (self.log_interval > 0) and not (i % self.log_interval):
            #     log_idx = int(i / self.log_interval)

            self.kernel_step[self.blocks,self.threads_per_block](self.d_xs, self.d_us, self.d_itaus, self.d_omegaMaxs, self.d_G1s, self.d_G2s, self.dt, self.log_idx, self.d_xs_log)
        else:
            self.kernel_step(self.xs, self.us, self.itaus, self.omegaMaxs, self.G1s, self.G2s, self.dt, int(0), self.xs_log)

            self.reward_function(self.xs, self.pSets, self.us, self.global_step_counter,self.r)

        # make sure all threads complete before stopping the count
        if self.gpu:
            cuda.synchronize()

        if self.gpu and (self.log_interval > 0):
            self.xs_log[:] = self.d_xs_log.copy_to_host()

    def step(self, action):
        '''Step function for gym'''
        self.us = action
        
        # done = self._check_done()
        done=False
        if not done:
            asyncio.run(self._step())
            # self._compute_reward()
        
        if self.action_buffer:
            self.action_history.append(np.array(action).reshape(self.N,4))
            return np.concatenate((self.xs, np.array(self.action_history).reshape(self.N,100)), axis=1), self.r, done, done,{}
        return self.xs,self.r, done,done, {}
        # return self.xs, self._compute_reward(), False, {}
    
    def _check_done(self):

        '''Check if the episode is done'''
        if self.episode_counter > self.iters:
            self.episode_counter = 0
            # self.reset()
            return True # gotta change this
        
        else:
            return False
        

global jitter; global kerneller
# debug mode
jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
GRAVITY = 9.80665
if __name__ == "__main__":
    N_drones = 1

    sim = Drone_Sim(gpu=False, viz=False, log=True, realtime=False, dt=0.01, T=10, N_cpu=N_drones, spiking_model=None)
    from libs.cpuKernels import controller
    print(sim.pSets)
    # position controller gains (attitude/rate hardcoded for now, sorry)
    posPs = 2*np.ones((N_drones, 3), dtype=np.float32)
    velPs = 2*np.ones((N_drones, 3), dtype=np.float32)

    
    print("Environment created!")
    sim.reset()
    print("running 1e3 steps...")
    t0 = time()
    t_step = 0
    t_steps = []

    for i in range(int(1e3)):
        controller(sim.xs, sim.us, posPs, velPs, sim.pSets, sim.G1pinvs)
        t_step = time()
        reward = sim.step(sim.us)[1]
        # print(sim.xs)
        # print(reward)
        t_steps.append(time()-t_step)
      
        # sim._compute_reward()
    print("1e3 steps took: ", time()-t0, " seconds")
    print("Average step time: ", np.mean(t_steps))
    print("Total step time: ", np.sum(t_steps))
    print("Done")


    print("\n\n\nTest default gym parallelization")
    # sim = Drone_Sim(gpu=False, viz=False, log=True, realtime=False, dt=0.01, T=10, N_cpu=1, spiking_model=None)
    gym.register("Drone_Sim-v0", entry_point=Drone_Sim)

    # debug mode
    jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
    kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
    GRAVITY = 9.80665

    from libs.cpuKernels import controller
    print(sim.pSets)
    
    sim = gym.make_vec("Drone_Sim-v0", num_envs=N_drones)
    
    # position controller gains (attitude/rate hardcoded for now, sorry)
    posPs = 2*np.ones((N_drones, 3), dtype=np.float32)
    velPs = 2*np.ones((N_drones, 3), dtype=np.float32)
    
    print("Environment created!")
    sim.reset()
    print("running 1e3 steps...")
    t0 = time()
    t_step = 0
    t_steps = []

    for i in range(int(1e3)):
        controller(sim.xs, sim.us, posPs, velPs, sim.pSets, sim.G1pinvs)
        t_step = time()
        reward = sim.step(sim.us)[1]
        # print(sim.xs)
        # print(reward)
        t_steps.append(time()-t_step)
      
        # sim._compute_reward()
    print("1e3 steps took: ", time()-t0, " seconds")
    print("Average step time: ", np.mean(t_steps))
    print("Total step time: ", np.sum(t_steps))
    print("Done")
        
