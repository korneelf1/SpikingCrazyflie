import numpy as np
from numpy.typing import NDArray
import torch
from libs.cpuKernels import step
class PID:
    def __init__(self, kp:NDArray | None = np.ones((3,)), 
                 ki:NDArray | None = np.ones((3,)), 
                 kd:NDArray | None = np.ones((3,)), 
                 G1pinv:NDArray | None = np.ones((4,4)), dt: float = 0.01):
        '''
        kp: proportional gain, array of shape (n,) where n is the number of states
        ki: integral gain, same shape as kp
        kd: derivative gain, same shape as kp
        G1pinv: inverse of G1 matrix, for motormixing, 4x4 matrix'''
        self.kp = np.eye(kp.shape[0]) * kp
        self.ki = np.eye(kp.shape[0]) * ki
        self.kd = np.eye(kp.shape[0]) * kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        # self.G1pinv = G1pinv

    def update(self, error: NDArray):

        P = self.kp @ error
        self.integral += error * self.dt
        I = self.ki @ self.integral
        derivative = (error - self.prev_error) / self.dt
        D = self.kd @ derivative
        self.prev_error = error
        u = P + I + D

        return u
    
    # def control(self, error: NDArray):
    #     u = self.update(error)

    #     return self.G1pinv @ u
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0

    # def tune_pytorch(self, kp: NDArray, ki: NDArray, kd: NDArray):
    #     self.kp = torch.eye(kp.shape[0]) * torch.from_numpy(kp)
    #     self.ki = torch.eye(kp.shape[0]) * torch.from_numpy(ki)
    #     self.kd = torch.eye(kp.shape[0]) * torch.from_numpy(kd)

    #     def errorfion()
    #     loss_fn = 
    #     optimizer = torch.optim.Adam([self.kp, self.ki, self.kd], lr=0.01)

    #     for i in range(100):
    #         optimizer.zero_grad()
    #         error = torch.randn(4)
    #         u = self.update(error)
    #         loss = loss_fn(u, torch.zeros(4))
    #         loss.backward()
    #         optimizer.step()
    #     def train_step():


class CascadePID:
    def __init__(self, G1pinvs, dt, pos_pid, att_pid, tuning=False):
        self.G1pinvs = G1pinvs
        self.dt = dt
        self.pos_pid = pos_pid
        self.att_pid = att_pid
        self.tuning = tuning

    def update(self, error: NDArray):
        '''Error is state vector, target is to make positions and angular velocities zero'''
        pos_error = error[:3]
        euler_angles = quaternion_to_euler(error[6:10])

        pos_u = self.pos_pid.update(pos_error) # contains pitch, roll and thrust
        
        att_error = np.array(euler_angles)
        att_error[:2] = pos_u[:2]
        
        thrust = pos_u[2]

        if self.tuning:
            thrust = 0.
            att_error = np.array(euler_angles)
            att_error = np.zeros_like(euler_angles)
        att_u = self.att_pid.update(att_error) # contains pitch moment and roll moment and yaw moment


        

        u = np.concatenate((np.array([thrust]), att_u)) # contains thrust, pitch moment, roll moment and yaw moment
        thrust_settings = np.clip(self.G1pinvs @ u,0,1).astype(np.float32)
        
        return thrust_settings

import math

def quaternion_to_euler(q):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw) in radians.
    
    Args:
    q -- A tuple or list of four elements representing the quaternion (q0, q1, q2, q3)
    
    Returns:
    roll, pitch, yaw -- The Euler angles in radians
    """

    q0, q1, q2, q3 = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

if __name__ == '__main__':
    from fastPyDroneSim_gym.fastPyDroneSim.gym_sim import Drone_Sim

    env = Drone_Sim(drone = 'og')
    G1pinvs = np.linalg.pinv(env.G1s) / (env.omegaMaxs*env.omegaMaxs)[:, :, np.newaxis]
    att_pid = PID(kp=np.array([0.1, 0.1, -0.1])*-20, ki=np.array([0.1, 0.1, 0.1]), kd=np.array([0.1, 0.1, 0.1]), G1pinv=G1pinvs)
    pid_controller = CascadePID(G1pinvs, 0.01, PID(), att_pid=att_pid, tuning=True)

    states = []
    state = env.reset()[0]
    for i in range(100):        
        error = state
        thrust_settings = pid_controller.update(error[0,:])
        state, _, _, _,_ = env.step(thrust_settings)
        states.append(state)

    states = np.array(states)
    # env.mpl_render(states)
    import matplotlib.pyplot as plt

    # make a figure with 3 subplots showing the angular velocities
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    axs[0, 0].plot(states[:,0, 10])
    axs[0, 0].set_title('Roll rate')
    axs[0, 1].plot(states[:,0, 11])
    axs[0, 1].set_title('Pitch rate')
    axs[0, 2].plot(states[:,0, 12])
    axs[0, 2].set_title('Yaw rate')
    axs[1, 0].plot(states[:,0, 0])
    axs[1, 0].set_title('x-position')
    axs[1, 1].plot(states[:,0, 1])
    axs[1, 1].set_title('y-position')
    axs[1, 2].plot(states[:,0, 2])
    axs[1, 2].set_title('z-position')
    # axs[1, 1].plot(states[:, 0, 2])
    # axs[1, 1].set_title('Thrust')
    plt.show()
