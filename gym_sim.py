"""
    Vectorized quadrotor simulation using gymnasium API

    
    maybe it would be better to provide step function which takes a model and performs a complete rollout where model is optimized
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
import torch
from tianshou.data import to_numpy, Batch
from helpers import NumpyDeque

GRAVITY = 9.80665
# print('\nNOTE NOW REWARD IS MODIFIED TO JUST MAKE STABILIZING CONTROLLER (DISREGARDING ANY POSITIONS)\n')
class Drone_Sim(gym.Env):
    def __init__(self, gpu=False, drone='CrazyFlie', task='stabilization', action_buffer=True,action_buffer_len=32, dt=0.01, T=2, N_drones=1, spiking_model=None, test=False, device='cpu', disturbances=False):
        super(Drone_Sim, self).__init__()
        '''
        Vectorized quadrotor simulation with websocket pose output
        NOTE: CURRENTLY JITTER AND KERNELLER ARE GLOBAL VARIABLES!
        
        Args:
            gpu (bool): run on the gpu using cuda
            drone (str): 'CrazyFlie' or 'Default'
            dt (float): step time is dt seconds (forward Euler)
            T (float): run for T seconds NOT USED
            N_drones (int): number of simulations to run in parallel
            spiking_model (object): spiking model, which will be reset at environment reset (spiking_model.reset_hidden())
            test (bool): if True, reward is cumulative reward over environment steps
            '''

        ### sim config ###
        self.gpu = gpu              # run on the self.gpu using cuda
        self.device = device        # cpu or cuda, used if env is ran on CPU, but model on GPU 
        # length / number of parallel sims
        self.action_buffer = action_buffer
        self.dt = dt                # step time is self.dt seconds (forward Euler)
        self.T = T                  # run for self.T seconds
        if self.gpu:                # number of simulations to run in parallel
            self.blocks = 128      # 128 or 256 seem best. Should be multiple of 32
            self.threads_per_block = 64 # depends on global memorty usage. 256 seems best without. Should be multiple of 64
            # # self.dt 0.01, self.T 10, no viz, self.log_interval 0, no controller, self.blocks 256, threads 256, self.gpu = True --> 250M ticks/sec
            self.N = self.blocks * self.threads_per_block
            if self.N != N_drones:
                print("N_drones is set to ", N_drones, " but will be overwritten by the number of threads and blocks")
        else:
            self.N = N_drones # cpu
        N = self.N

        self.is_async = False

        self.spiking_model = spiking_model
        # initial states: 0:3 pos, 3:6 vel, 6:10 quaternion, 10:13 body rates Omega, 13:17 motor speeds omega

        self.stabilization = False
        self.n_states = 20
        if task == 'stabilization':
            self.stabilization = True
            self.n_states = 17
        if action_buffer: # add last 25 inputs as observation
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4*action_buffer_len+self.n_states,), dtype=np.float32)
            self.action_history = NumpyDeque((self.N,4*action_buffer_len)) # 25 timesteps, 4 actions

        else:
        # create gymnasium observation and action space 17 + 3 for position
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_states,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0., high=1., shape=(4,), dtype=np.float32)


        self.done = np.zeros((self.N),dtype=bool) # for resetting all vectorized envs
        
        self.test = test
        if self.test:
            self.r_means = [0]
            self.r_maxs  = [0]
            self.r_mins  = [0]
        
        

        # other settings (vizualition and logging)
        self.viz_interval = 0.05 # visualize every self.viz_interval simulation-seconds
        self.Nviz = 512 # max number of quadrotors to visualize
        self.log_interval = 1    # log state every x iterations. Too low may cause out_of_memory on the self.gpu. False == 0

        
        # create the drones
        if drone=='CrazyFlie':
            self._create_drones(og_drones=False)
        else:
            self._create_drones(og_drones=True)

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
        self.pSets = np.zeros_like(self.pSets)
        # import compute kernels
        global kerneller; global jitter
        if self.gpu:
            jitter = lambda signature: nb.cuda.jit(signature, fastmath=False, device=True, inline=False)
            kerneller = lambda signature: nb.cuda.jit(signature, fastmath=False, device=False)

            from libs.gpuKernels import step as kernel_step
            from libs.gpuKernels import reward_function, check_done

            self.kernel_step = kernel_step
            self.reward_function = reward_function
            self.check_done      = check_done

        else:
            jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
            kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
            nb.set_num_threads(max(nb.config.NUMBA_DEFAULT_NUM_THREADS-4, 1))

            if disturbances:
                from libs.cpuKernels import step_disturbance as kernel_step
            else:
                from libs.cpuKernels import step as kernel_step
            from libs.cpuKernels import reward_function, check_done

            self.kernel_step     = kernel_step
            self.reward_function = reward_function
            self.check_done      = check_done
            # self.reset_subenvs   = reset_subenvs
            # self.termination = termination

        # allocate sim data
        log=1
        self.log_interval = log*5
        self.iters = int(self.T / self.dt)
        self.Nlog = int(self.iters / self.log_interval) if self.log_interval > 0 else 0

        self.reset()
        self.us = np.random.random((N, 4)).astype(np.float32) # inputs (motor speeds)

        # create logs
        self._create_logs()

        self.r = np.zeros(self.N, dtype=np.float32)

        # gym specific stuff
        self.episode_counter = 0
        self.global_step_counter = 0 # used for reward curriculum
        self.iters = int(self.T / self.dt)

    @property
    def n(self):
        '''
        Number of drones that are simulated.'''
        return self.N
    
    @property
    def num_envs(self):
        '''
        Number of drones that are simulated.'''
        return self.N
    
    def __len__(self):
        '''
        Number of drones that are simulated.'''
        return self.N
    
    def _create_drones(self, og_drones=True):
        '''Creates drones, crazyflies
        parameters retrieved from:
        https://github.com/arplaboratory/learning_to_fly_media/blob/ae72456e879137b840b9dfde366253886c3ec131/parameters.pdf

        Args:   
            og_drones (bool): if True, simulates the drones from the original FastPyDroneSim, else CrazyFlie
        '''
        # OG drone data
        if og_drones:
            self.G1s = np.empty((self.N, 4, 4), dtype=np.float32)
            self.G2s = np.empty((self.N, 1, 4), dtype=np.float32)
            self.omegaMaxs = np.empty((self.N, 4), dtype=np.float32)
            self.taus = np.empty((self.N, 4), dtype=np.float32)

            for i in tqdm(range(self.N), desc="Building crafts"):
                q = QuadRotor()
                q.setInertia(0.42, 1e-3*np.eye(3))
                q.rotors.append(Rotor([-0.1, 0.1, 0], dir='cw'))
                q.rotors.append(Rotor([0.1, 0.1, 0], dir='ccw'))
                q.rotors.append(Rotor([-0.1, -0.1, 0], dir='ccw'))
                q.rotors.append(Rotor([0.1, -0.1, 0], dir='cw'))

                q.fillArrays(i, self.G1s, self.G2s, self.omegaMaxs, self.taus)
        else:
            # mass of crazyfly
            self.m = 0.027
            self.Ixx = 3.85e-6
            self.Iyy = 3.85e-6
            self.Izz = 5.9675e-6
            self.I = np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]])

            self.G1s = np.empty((self.N, 4, 4), dtype=np.float32)
            self.G2s = np.empty((self.N, 1, 4), dtype=np.float32)
            self.omegaMaxs = np.empty((self.N, 4), dtype=np.float32) # max rpm (in rads)? if so, max 21702 rpm -> 21702/60*2pi rad/sec
            self.taus = np.empty((self.N, 4), dtype=np.float32) # RPM time constant? if so, 0.15sec or 0.015sec?

            # max_rads = 21702/60*2*3.1415
            max_rads = 21702
            for i in tqdm(range(self.N), desc="Building crafts"):
                q = QuadRotor()
                q.setInertia(self.m, self.I)
                q.rotors.append(Rotor([-0.028, 0.028, 0], dir='cw', wmax = max_rads, tau=0.15, k= 3.16e-10,Izz=0.005964552)) # rotor 3
                q.rotors.append(Rotor([0.028, 0.028, 0], dir='ccw', wmax = max_rads, tau=0.15, k= 3.16e-10,Izz=0.005964552)) # rotor 4
                q.rotors.append(Rotor([-0.028, -0.028, 0], dir='ccw', wmax = max_rads, tau=0.15, k= 3.16e-10,Izz=0.005964552)) # rotor 2	
                q.rotors.append(Rotor([0.028, -0.028, 0], dir='cw', wmax = max_rads, tau=0.15, k= 3.16e-10,Izz=0.005964552)) # rotor 1

                q.fillArrays(i, self.G1s, self.G2s, self.omegaMaxs, self.taus)

    def _create_logs(self,):
        '''Creates logs
        NOTE: currently not used'''	
        if self.action_buffer:
            self.xs_log = np.empty(
            (self.N, self.Nlog, self.n_states +len(self.action_history)), dtype=np.float32)
        else:
            self.xs_log = np.empty(
                (self.N, self.Nlog, self.n_states), dtype=np.float32)
        self.xs_log[:] = np.nan

    def _simulate_step(self, disturbance=None):
        '''Simulates a single step using gpu or cpu kernels'''
        self.log_idx =0
        # make sure xs is float32
        # if np.min(self.us)<0.0 or np.max(self.us)>1.0:
        #     raise RuntimeWarning('Action is not in action space!')
        self.xs = self.xs.astype(np.float32)
        if self.gpu:
            self.kernel_step[self.blocks,self.threads_per_block](self.d_xs, self.d_us, self.d_itaus, self.d_omegaMaxs, self.d_G1s, self.d_G2s, self.dt, self.log_idx, self.d_xs_log)
        else:
            if disturbance is not None:
                disturbance = np.array(disturbance).astype(np.float32)
                self.kernel_step(self.xs, self.us, self.itaus, self.omegaMaxs, self.G1s, self.G2s, self.dt, disturbance, int(0), self.xs_log)
            else:
                self.kernel_step(self.xs, self.us, self.itaus, self.omegaMaxs, self.G1s, self.G2s, self.dt, int(0), self.xs_log)
            # self.xs = np.concatenate((self.xs, self.pSets),axis=1)
                        
    def _compute_reward(self):
        '''Compute reward, reward function from learning to fly in 18sec paper
        TODO: optimize with cpuKernels and gpuKernels'''	
        if self.test:
            r = self.r
        if self.gpu:
            self.reward_function[self.blocks,self.threads_per_block](self.d_xs, self.d_pSets, self.d_us, self.global_step_counter,self.d_r)
        else:
            self.reward_function(self.xs, self.pSets, self.us, self.global_step_counter,self.r)
        if self.test:
            self.r += r    
    def _check_done(self, numba_opt = True):
        '''Check if the episode is done, sets done array to True for respective environments.'''
        if numba_opt:
            if self.gpu:
                self.check_done[self.blocks,self.threads_per_block](self.d_xs, self.d_done)
            else:
                self.check_done(self.xs, self.done)
            # self.done = np.expand_dims(self.done, axis=1)
        else:
            # if any velocity in the abs(self.xs) array is greater than 10 m/s, then the episode is done
            # if any rotational velocity in the abs(self.xs) array is greater than 10 rad/s, then the episode is done
            if self.gpu:
                self.done = np.logical_or(np.any(np.abs(self.d_xs[:,3:6]) > 10, axis=1), \
                                    np.any(np.abs(self.d_xs[:,10:13]) > 20, axis=1))
            else:
                self.done = np.logical_or(np.any(np.abs(self.xs[:,3:6]) > 10, axis=1), \
                                        np.any(np.abs(self.xs[:,10:13]) > 20, axis=1))

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

        self.d_r = cuda.to_device(self.r)
        # self.d_global_step_counter = cuda.to_device(self.global_step_counter)
        self.d_done = cuda.to_device(self.done)

        cuda.synchronize()

    def _reset_subenvs(self, numba_opt = True, seed = None):
        '''Reset subenvs, uses the done array
        NOTE: First call _check_done()'''
        if numba_opt:
            raise NotImplementedError("This function is not implemented yet, see dones!")
            self.reset_subenvs(self.done, seed,self.xs)
        else:
            if self.gpu:
                if self.action_buffer:
                    self.action_history.reset(self.d_done)
                # create new states
                xs_new = np.random.random((self.N, 17)).astype(np.float32) - 0.5
                xs_new[:, 6:10] /= np.linalg.norm(xs_new[:, 6:10], axis=1)[:, np.newaxis]

                # mask with done array
                self.d_xs[:,0:17][self.d_done,:] = xs_new[self.d_done,:]
                # self.xs = np.concatenate((self.xs[:,0:17],self.pSets),axis=1) # pos should still be in xs!
                if self.action_buffer:
                    self.d_xs = np.concatenate((self.d_xs[:,0:self.n_states],self.action_history),axis=1,dtype=np.float32)
            else:
                if self.action_buffer:
                    self.action_history.reset(self.done)
                # create new states
                xs_new = np.random.random((self.N, 17)).astype(np.float32) - 0.5
                xs_new[:, 6:10] /= np.linalg.norm(xs_new[:, 6:10], axis=1)[:, np.newaxis]

                # mask with done array
                self.xs[:,0:17][self.done,:] = xs_new[self.done,:]
                # self.xs = np.concatenate((self.xs[:,0:17],self.pSets),axis=1) # pos should still be in xs!
                if self.action_buffer:
                    self.xs = np.concatenate((self.xs[:,0:self.n_states],self.action_history),axis=1,dtype=np.float32)
            
    async def _step(self, enable_reset = True, disturbance=None):
        '''
        Perform a step:
        Simulate step
        Compute reward
        Check if any env is done
        Reset respective envs'''
        # self.episode_counter += 1
        # self.global_step_counter += self.N 
        self.global_step_counter += 1 # for easier experimenting
        
        if self.gpu:
            self._move_to_cuda()

        self._simulate_step(disturbance=disturbance)
        # self.kernel_step(self.xs, self.us, self.itaus, self.omegaMaxs, self.G1s, self.G2s, self.dt, int(0), self.xs_log)
        # self.reward_function(self.xs, self.pSets, self.us, self.global_step_counter,self.r)
        self._compute_reward()
        
        self._check_done()
        # print(self.done)

        # self.termination(self.xs, self.done)
        # make sure all threads complete before stopping the count
        if self.gpu:
            cuda.synchronize()

        if self.gpu and (self.log_interval > 0):
            self.xs_log[:] = self.d_xs_log.copy_to_host()
        
        if enable_reset:
            self._reset_subenvs(numba_opt=False)

    async def _step_rollout(self, policy, nr_steps=None,nr_episodes=None,tianshou_policy=False):
        '''
        Collects a series of rollouts, 
        policy: is tianshou policy that uses act method for interaction
        nr_steps: number of steps to collect
        nr_episodes: number of episodes to collect
        tianshou_policy: if True, uses tianshou policy, else uses normal policy

        Stores info in arrays.

        To perform a step
        Simulate step
        Compute reward
        Check if any env is done
        Reset respective envs
        '''
        # if True:
        #     raise NotImplementedError("This function is not implemented yet, see dones!")
        if nr_steps:
            iters = int(nr_steps)
        elif nr_episodes:
            iters = int(nr_episodes*1e2) # huge nr of steps (we will exit with break statement)

        else:
            raise ValueError("Either nr_steps or nr_episodes should be provided!")
        # print("Running simulation for ", iters, " steps, with ", self.N, " drones")

        if self.action_buffer:
            obs_arr = np.zeros((iters, self.N, self.n_states+len(self.action_history)), dtype=np.float32)
            obs_next_arr = np.zeros((iters, self.N, self.n_states+len(self.action_history)), dtype=np.float32)
        else:
            obs_arr = np.zeros((iters, self.N, self.n_states), dtype=np.float32)
            obs_next_arr = np.zeros((iters, self.N, self.n_states), dtype=np.float32)
        act_arr = np.zeros((iters, self.N, 4), dtype=np.float32)
        done_arr = np.zeros((iters, self.N, ), dtype=bool)
        info_arr = np.zeros((iters, self.N, 1), dtype=bool)
        rew_arr = np.zeros((iters, self.N, ), dtype=np.float32)

        # for statistics
        episode_lens = []
        episode_rews = []
        episode_len_arr = np.zeros((self.N,),dtype=np.int32)

        if not self.test:
            self.reset()
        
        if self.gpu:
            self._move_to_cuda()

        ts = time()
        ei = 0
        # for i in range(iters):
        for i in tqdm(range(iters), desc="Running simulation"):

            # self.global_step_counter += int(self.N)
            self.global_step_counter += int(1)
            if self.gpu:
                obs_arr[i] = self.d_xs.copy_to_host()
            else:
                obs_arr[i] = self.xs

            self._simulate_step()
            

            self._compute_reward()
            

            if self.test:
                self.r_means.append(np.mean(self.r))
                self.r_maxs.append(np.max(self.r))
                self.r_mins.append(np.min(self.r))

            self._check_done()

            if self.gpu:
                obs_next_arr[i] = self.d_xs.copy_to_host()
                rew_arr[i] = self.d_r.copy_to_host()
                done_host = self.d_done.copy_to_host()
                done_arr[i] = done_host

                
                episode_len_arr += 1 - done_host # adds an increment for every step
                if np.any(done_host):
                    episode_lens.extend(episode_len_arr[done_host].tolist())
                    episode_rews.extend(rew_arr[i][done_host].tolist())
                    episode_len_arr[done_host] = 0
                    ei += np.sum(done_host)

            else:
                obs_next_arr[i] = self.xs
                rew_arr[i] = self.r
                done_arr[i] = self.done

                episode_len_arr += 1 - self.done # adds an increment for every step
                if np.any(self.done):
                    episode_lens.extend(episode_len_arr[self.done].tolist())
                    episode_rews.extend(rew_arr[i][self.done].tolist())
                    episode_len_arr[self.done] = 0
                    ei += np.sum(self.done)

            with torch.no_grad():
                # self.us = to_numpy(policy(Batch(obs=self.xs, info={})).act)
                if tianshou_policy:
                    if self.gpu:
                        # raise UserWarning('Position setpoints not passed on GPU yet.')
                        if self.action_buffer:
                            self.us = to_numpy(policy.map_action(policy(Batch({'obs':np.concatenate((self.d_xs[:,0:self.n_states],self.action_history.array),axis=1,dtype=np.float32), 'info':{}})).act))
                            self.action_history.append(self.us)
                        else:
                            self.us = to_numpy(policy.map_action(policy(Batch({'obs':self.d_xs, 'info':{}})).act))
                    else:
                        
                        # self.xs = np.concatenate((self.xs[:,0:17],self.pSets),axis=1)
                        if self.action_buffer:
                            self.xs = np.concatenate((self.xs[:,0:self.n_states],self.action_history.array),axis=1,dtype=np.float32)
                            xs_torch = torch.from_numpy(self.xs).to(self.device)
                            self.us = to_numpy(policy.map_action(policy(Batch({'obs':xs_torch, 'info':{}})).act))
                            self.action_history.append(self.us)
                            self.xs = np.concatenate((self.xs[:,0:self.n_states],self.action_history.array),axis=1,dtype=np.float32)
                        else:
                            xs_torch = torch.from_numpy(self.xs).to(self.device)
                            self.us = to_numpy(policy.map_action(policy(Batch({'obs':xs_torch, 'info':{}})).act))
                else:
                    if self.gpu:
                        raise UserWarning('GPU support not yet implemented for non tianshou policies.')
                    if self.action_buffer:
                        self.us = to_numpy(policy(np.concatenate((self.xs[:,0:self.n_states],self.action_history.array),axis=1,dtype=np.float32)))
                    else:
                        self.us = to_numpy(policy(self.xs))
                if self.action_buffer:
                    if self.gpu:
                        self.d_xs = np.concatenate((self.d_xs[:,0:self.n_states],self.action_history.array),axis=1,dtype=np.float32)
                    # else:
                    #     self.xs = np.concatenate((self.xs[:,0:20],self.action_history),axis=1,dtype=np.float32)
                act_arr[i] = self.us
            
            if nr_episodes and ei >= nr_episodes:
                self.episode_counter += ei
                # trim the arrays
                obs_arr = obs_arr[:i+1]
                act_arr = act_arr[:i+1]
                rew_arr = rew_arr[:i+1]
                done_arr = done_arr[:i+1]
                obs_next_arr = obs_next_arr[:i+1]
                info_arr = info_arr[:i+1]
                
                break

            self._reset_subenvs(numba_opt=False)

        # make sure all threads complete before stopping the count
        if self.gpu:
            cuda.synchronize()
        # done_arr[i] = np.ones((self.N, 1), dtype=bool)
        ei+= self.N - np.sum(done_arr[i])



        return obs_arr, act_arr, rew_arr, done_arr, obs_next_arr, info_arr, {'episode_lens': episode_lens, 'episode_rews': episode_rews, 'episode_ctr': ei, 'time': time()-ts}
    
    def reset(self,seed=None, dones = None):
        '''
        For the reset of specific envs, use the done array to reset the correct envs
        First multiply the relevant env states with zero-mask, then add the new states with inverse zero mask (zeros everywhere but the relevant states)
        '''
        super().reset(seed=seed)
         # initial states: 0:3 pos, 3:6 vel, 6:10 quaternion, 10:13 body rates Omega, 13:17 motor speeds omega
        x0 = np.random.random((self.N, 17)).astype(np.float32) - 0.5
        x0[:, 6:10] /= np.linalg.norm(x0[:, 6:10], axis=1)[:, np.newaxis] # quaternion needs to be normalized

        self.xs = x0.copy() # states
        self.t = 0
        self.episode_counter = 0
        if self.test:
            # print('Rewards \tmean: ', np.mean(self.r_means),'\tmax: ', np.max(self.r_maxs),'\tmin: ', np.min(self.r_mins))
            self.r_means = [0]
            self.r_maxs = [0]
            self.r_mins = [0]
        self.r = np.zeros(self.N, dtype=np.float32)
        if self.spiking_model:
            self.spiking_model.reset_hidden()
            print('Reward: \taverage: ',np.mean(self.r),\
                    '\tmax: ',np.max(self.r), '\tmin: ', np.min(self.r))
            
        if not self.stabilization:    
            self.xs = np.concatenate((self.xs[:,0:17], self.pSets),axis=1)

        if self.action_buffer:
            self.action_history.reset()
            self.xs = np.concatenate((self.xs,self.action_history),axis=1,dtype=np.float32)
            # return x0,{}
        # YOU CAN RESET YOUR MODEL IN THE ENVIRONMENT RESET FUNCTION!!!!!!!
        return self.xs,np.array([{} for _ in range(self.N)]) # state, info
                
    def step(self, action, enable_reset=True, disturbance=None):
        '''Step function for gym'''
        self.us = action
        # self.done =np.zeros((self.N,1),dtype=bool)
        # done = self._check_done()
        asyncio.run(self._step(enable_reset=enable_reset, disturbance=disturbance))

        return self.xs,self.r, self.done,self.done, {}
   
    def step_rollout(self, policy, n_step = None, n_episode=None, numba_policy=False, tianshou_policy=False, random=False):
        '''Step function for collecting and entire rollout, which can be faster in this vectorized environment
        policy: is tianshou policy that uses:
         action = policy.act(observation) method for interaction
        NOTE: you need random steps for exploration, in this case, the policy will be a stochastic SNN, so it is inherrent in the policy'''
        if numba_policy:
            print("gotta fix this!")
        return asyncio.run(self._step_rollout(policy,nr_steps=n_step,nr_episodes=n_episode,tianshou_policy=tianshou_policy))
    
    def render(self, mode='human', policy=None, n_step=1e3, tianshou_policy=False):
        # other settings
        viz_interval = 0.05 # visualize every viz_interval simulation-seconds
        Nviz = 512 # max number of quadrotors to visualize
        log_interval = 1    # log state every x iterations. Too low may cause out_of_memory on the GPU. False == 0
        viz = False
        if mode=='human':
            viz = True
        if viz:
            print("initializing websocket. Awaiting connection... ")
            wsI = wsInterface(8765)
        else:
            wsI = dummyInterface()
        with wsI as ws:
            tsAll = time()

            for i in range(n_step):
            # self.global_step_counter += int(self.N)
                self._simulate_step()

            # print('action: ',self.us[1])
            # print('motor speeds: ',self.xs[1,13:17])
                self._compute_reward()
            # print(self.done)

            with torch.no_grad():
                # self.us = to_numpy(policy(Batch(obs=self.xs, info={})).act)
                if tianshou_policy:
                    if self.action_buffer:
                        self.us = to_numpy(policy.map_action(policy(Batch({'obs':self.xs, 'info':{}})).act))
                        self.action_history.append(self.us)
                    else:
                        self.us = to_numpy(policy.map_action(policy(Batch({'obs':self.xs, 'info':{}})).act))
                else:
                    if self.action_buffer:
                        self.us = to_numpy(policy(np.concatenate((self.xs[:,0:17],self.action_history.array),axis=1,dtype=np.float32)))
                        self.action_history.append(self.us)
                    else:
                        self.us = to_numpy(policy(self.xs))

                
            if viz  and  ws.ws is not None  and  not i % int(viz_interval/self.dt):
                # visualize every 0.1 seconds
 
                ws.sendData(self.xs[::int(np.ceil(self.N/Nviz))].astype(np.float64))

    def mpl_render(self, observations):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation

        xyz = observations[:,0, :3]
        # Set up the figure and the 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize a point in the plot
        point, = ax.plot([], [], [], 'bo')
        # set point at (0,0,0)
        origin, = ax.plot([], [], [], 'ro')
        
        # Set the limits of the plot
        ax.set_xlim(np.min(xyz[:, 0]), np.max(xyz[:, 0]))
        ax.set_ylim(np.min(xyz[:, 1]), np.max(xyz[:, 1]))
        ax.set_zlim(np.min(xyz[:, 2]), np.max(xyz[:, 2]))

        # Update function for the animation
        def update(frame):
            # Update the point's position
            point.set_data(xyz[frame, 0], xyz[frame, 1])
            origin.set_data(0,0)

            point.set_3d_properties(xyz[frame, 2])
            origin.set_3d_properties(0.1)
            return point,origin,

        # Create the animation
        ani = FuncAnimation(fig, update, frames=xyz.shape[0], interval=self.dt * 10000, blit=True)

        # Show the plot
        plt.show()

global jitter; global kerneller
# NOTE if linux went into suspend mode, might not be able to communicate with GPU, run:
# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm
# in terminal 

# torch.cuda.init()
gpu = torch.cuda.is_available()
gpu=False
print(gpu)
# debug mode
if gpu:
    jitter = lambda signature: nb.cuda.jit(signature, fastmath=False, device=True, inline=False)
    kerneller = lambda signature: nb.cuda.jit(signature, fastmath=False, device=False)
else:
    jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
    kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)

if __name__ == "__main__":
    N_drones = 1

    sim = Drone_Sim(gpu=False, 
                    dt=0.01, 
                    T=2, 
                    N_drones=N_drones, 
                    spiking_model=None, 
                    action_buffer=False, 
                    drone='ogDrone', 
                    disturbances=True)
    
    # from libs.cpuKernels import controller_rl
    # position controller gains (attitude/rate hardcoded for now, sorry)
    posPs = 2*np.ones((N_drones, 3), dtype=np.float32)
    velPs = 2*np.ones((N_drones, 3), dtype=np.float32)

    
    print("Environment created!")

    iters = int(10)
    sim.reset()
    # create actor
    t0 = time()
    t_steps = []
    import networks as n

    print("\nTest individual steps")
    sim.reset()
    # from libs.cpuKernels import controller
    # position controller gains (attitude/rate hardcoded for now, sorry)
    G1pinvs = np.linalg.pinv(sim.G1s) / (sim.omegaMaxs*sim.omegaMaxs)[:, :, np.newaxis]
    t0 = time()
    obs = []
    
    # from learning to fly....
    import datastruct_dict as d
    log_file_path = "/home/korneel/learning_to_fly/learning_to_fly/learning-to-fly/include/learning_to_fly/simulator/log.txt"
    extracted_data = d.parse_log_file(log_file_path)
    obs = extracted_data[0]
    obs_next = extracted_data[1]
    disturbances = extracted_data[2]
    actions = extracted_data[3]
    rewards = extracted_data[4]

    

    iters = 9
    def rescale_actions(actions):
        return ((actions+1)/2).astype(np.float32)
    for i in tqdm(range(iters), desc="Running simulation steps"):
        sim.xs = obs[:,i].reshape(1, 17)
        # rpms = [10908.4873, 10854.8584,10902.9434,10830.2275]
        # sim.xs[:, 13:17] = np.array(rpms).reshape(1,4)
        sim.us = rescale_actions(actions[:,i])
        # sim.us = (np.array(rpms).reshape(1,4)/21702).astype(np.float32)
        # sim.step(policy(sim.xs).detach().numpy().astype(np.float32), enable_reset=False)
        xs_next, reward, _,_,_ = sim.step(sim.us,enable_reset=True, disturbance=disturbances[:,i])
        
        sim.xs = obs_next[:,i].reshape(1, 17).astype(np.float32)    
        sim._compute_reward() 
        print(sim.r)

    obs = np.array(obs).reshape(-1, sim.N, 17)
    sim.mpl_render(obs)
    # t_steps.append(time()-t_step)
      
        # sim._compute_reward()
    print("1e3 steps took: ", time()-t0, " seconds")
    print("Average step time: ",  (time()-t0)/iters)
    # print("Total step time: ", np.sum(t_steps))
    print("Done")
        
'''
Issue:
if parallel envs, how to organize dones? now it was running indefinetly, but values become infinity.
I should have a done array, which is updated in the step function, and then the done array is checked in the step function
Then, either reset correct envs and policies OR just silence appropriate envs till global reset is done...
'''