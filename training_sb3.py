import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from l2f import *
from l2f_gym import Learning2Fly

env = Learning2Fly()
env = Monitor(env)
check_env(env)
env = make_vec_env(lambda: Learning2Fly(fast_learning=True), n_envs=12)

model = sb3.SAC("MlpPolicy", env, verbose=1, ent_coef=0.)

model.learn(total_timesteps=2e6)

model.save("TD3_l2f")
model = sb3.TD3.load("TD3_l2f")

obs = env.reset()
obs_lst = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    
    if dones:
        obs = env.reset()
    obs_lst.append(obs)
def mpl_render(observations, fps=30):
        '''Render function for gym: visualizes the simulation in a matplotlib animation window, not very flashy but reasonably useful for debugging'''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation

        xyz = observations[:,0, :3]
        # convert to fps (every fps frames)
        dt_anim = 1/fps
        print(dt_anim)
        xyz = xyz[::int(np.ceil(dt_anim/.01))]
        

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
        ani = FuncAnimation(fig, update, frames=xyz.shape[0], interval=dt_anim*1000, blit=True)

        # Show the plot
        plt.show()

import numpy as np
obs_lst = np.array(obs_lst)
mpl_render(obs_lst)