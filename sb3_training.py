import sb3_contrib as sb3
from gym_sim import Drone_Sim
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import stable_baselines3

import numba as nb
# debug mode
jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
GRAVITY = 9.80665

num_cpu = 1
# Define the environment
env = Drone_Sim()
# vec_env = SubprocVecEnv([Drone_Sim() for i in range(num_cpu)])
print(env.observation_space)
print(env.reset()[0].shape)
# gym.register("Drone_Sim-v0", entry_point=Drone_Sim)

# env = gym.make("Drone_Sim-v0")
print("Env created!")
# If needed, wrap it into a DummyVecEnv

# model = sb3.RecurrentPPO("MlpLstmPolicy", env, verbose=2)
model = stable_baselines3.SAC("MlpPolicy", env, verbose=1)
print(model.collect_rollouts)
print("Model created!")
model.learn(total_timesteps=3e6,)
model.save("ppo_drone_sim")
