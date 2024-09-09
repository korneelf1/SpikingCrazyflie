import numpy as np
from numpy.typing import NDArray
import torch
from libs.cpuKernels import step
class PID:
    def __init__(self, kp:NDArray | None = np.ones((3,)), 
                 ki:NDArray | None = np.ones((3,)), 
                 kd:NDArray | None = np.ones((3,)), 
                 dt: float = 0.01):
        '''
        kp: proportional gain, array of shape (n,) where n is the number of states
        ki: integral gain, same shape as kp
        kd: derivative gain, same shape as kp
        dt: time step'''

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
    
def create_crazyflie_pid_rate():
    # roll
    Kp_rollrate = 250
    Ki_rollrate = 500
    Kd_rollrate = 2.5

    # pitch
    Kp_pitchrate = 250  
    Ki_pitchrate = 500
    Kd_pitchrate = 2.5

    # yaw
    Kp_yawrate = 120
    Ki_yawrate = 16.7
    Kd_yawrate = 0

    kp = np.array([Kp_rollrate, Kp_pitchrate, Kp_yawrate])
    ki = np.array([Ki_rollrate, Ki_pitchrate, Ki_yawrate])
    kd = np.array([Kd_rollrate, Kd_pitchrate, Kd_yawrate])

    pid = PID(kp=kp, ki=ki, kd=kd)

    return pid

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

def thrust_to_rpm(thrust: NDArray):
    '''Convert thrust to rpm
    "thrust_curve": [0.021300, -0.011200, 0.120100]
    thrust = 0.0213 - 0.0112*rpm + 0.1201*rpm^2
    rpm = (-b +- sqrt(b^2 - 4*a*c))/(2*a)
    '''
    try:
        rpms = np.zeros(4)
        for i in range(4):
            
            a = 0.1201
            b = -0.0112
            c = 0.0213 - thrust
            rpm1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            rpm2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            rpm = np.max([rpm1, rpm2]) # should always be one neg one pos rpm unless very low thrust
            rpms[i] = rpm
        # rescale to [-1,1]
        rpm = rpms*2 - 1
        return rpm
    except:
        raise ValueError("Thrust too low")
    return 
if __name__ == '__main__':
    import l2f_gym as l2f
    import numpy as np
    env = l2f.Learning2Fly()
    pid = create_crazyflie_pid_rate()
    from l2f import parameters_to_json

    # print(parameters_to_json(env.device, env.env,env.env.DynamicsParameters))

    obs = env.reset()[0]
    done = False
    # pid.reset()

    rate_target = np.array([0,0,0])
    while not done:
        # action = pid.update(obs[10:13])
        thrust = np.array([0.05,0.05,0.05,0.05])
        action = thrust_to_rpm(thrust)
        obs, reward, done, info,_ = env.step(action)
        print(action)
        print(obs)