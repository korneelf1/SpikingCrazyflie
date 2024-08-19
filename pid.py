import numpy as np
from numpy.typing import NDArray
import torch
from libs.cpuKernels import step
class PID:
    def __init__(self, kp:NDArray | None = np.ones((17,)), 
                 ki:NDArray | None = np.ones((17,)), 
                 kd:NDArray | None = np.ones((17,)), 
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
        self.G1pinv = G1pinv

    def update(self, error: NDArray):

        P = self.kp @ error
        self.integral += error * self.dt
        I = self.ki @ self.integral
        derivative = (error - self.prev_error) / self.dt
        D = self.kd @ derivative
        self.prev_error = error
        u = P + I + D

        return u
    
    def control(self, error: NDArray):
        u = self.update(error)

        return self.G1pinv @ u
    
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
