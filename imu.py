import numpy as np
from helpers import NumpyDeque
import matplotlib.pyplot as plt

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def euler_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """
    Convert euler angles to quaternion
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.zeros(4)
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q

def quaternion_to_euler(q):
    '''
    q: np.array of shape (4,)
    '''
    qw, qx, qy, qz = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    a = np.sqrt(1+2*(qw*qy-qz*qx))
    b = np.sqrt(1-2*(qw*qy-qz*qx))
    pitch = -np.pi/2 + 2*np.arctan2(a,b)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class IMU:
    def __init__(self, 
                 noise=np.array([0.0015497692442013257, 0.0018856712200106122, 0.0017949298128139245, 0.0033169208, 0.003474, 0.002747] ), 
                 bias=np.array([0, 0, 0, 0, 0, 0]), 
                 dt=0.01, 
                 offset=np.array([0, 0, 0]),
                 euler_in:bool=False):
        '''
        noise: standard deviation of the noise
        bias: initial value of the bias (modelled as brownian motion)
        offset: offset of the sensor in body frame
        euler_in: if True, the input orientation is in euler angles, otherwise in quaternion AND orientation has len 3 rather than 4 AND input velocity is in body frame
        
        From static measurements with crazyflie, we have:
        accelerometer noise:
        x: 0.0015497692442013257 m/s^2
        y: 0.0018856712200106122 m/s^2
        z: 0.0017949298128139245 m/s^2

        gyroscope noise:
        x: 0.18986995961776781 deg/s = 0.0033169208 rad/s
        y: 0.19904941377037574 deg/s = 0.003474 rad/s
        z: 0.15737015901676 deg/s = 0.002747 rad/s

        noise = [0.0015497692442013257, 0.0018856712200106122, 0.0017949298128139245, 0.0033169208, 0.003474, 0.002747] # m/s^2, rad/s
        '''
        assert len(noise) == 6
        assert len(bias) == 6
        assert len(offset) == 3

        self.dt = dt

        self.noise = noise/np.sqrt(self.dt)
        self.bias = bias

        self.accel = np.array([0, 0, 0])
        self.gyro = np.array([0, 0, 0])

        self.offset = offset

        self.vel_history = NumpyDeque((10, 3))
        self.accel_history = NumpyDeque((10, 3))
        self.gyro_history = NumpyDeque((10, 3))

        self.vel_body_prev = np.array([0, 0, 0])

        self.euler = euler_in

    def add_brownian_bias(self):
        self.bias += np.random.normal(0, 0.01, size=(6,)) * np.sqrt(self.dt)
    def add_noise(self):
        self.accel = np.random.normal(self.accel, self.noise[:3]) - self.bias[:3]
        self.gyro = np.random.normal(self.gyro, self.noise[3:]) - self.bias[3:]


    def simulate(self, state):
        '''
        state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        '''
        # update acceleration
        vel = state[3:6]
        # self.vel_history.append(vel.reshape(1,3))
        if not self.euler:
            q = state[6:10]
            R = quaternion_rotation_matrix(q)

            # transfrom velocity to body frame
            vel_body = np.dot(R.T, vel)

            self.gyro = state[10:13]
        else:
            vel_body = vel
            # update angular velocity
            self.gyro = state[9:12]

            R = quaternion_rotation_matrix(euler_to_quaternion(state[6:9]))

        # self.vel_history.append(vel_body.reshape(3,1))
        # transform gravity to body frame
        accel_grav = np.dot(R, np.array([0, 0, -9.81])) 

        accel_grav_body = np.dot(R.T, accel_grav)
        # compute acceleration from velocity history
        if self.vel_body_prev is None:
            self.vel_body_prev = vel_body
        accel_vel_change = (vel_body - self.vel_body_prev)/self.dt
        # accel_vel_change = (self.vel_history[0] - self.vel_history[1]) / self.dt

        self.accel = accel_grav_body + np.array(accel_vel_change)

        

        self.add_noise()

        # observation should be position, acceleration, angular velocity
        pos = state[:3]
        return np.concatenate([pos,self.accel, self.gyro])

    def reset(self):
        self.accel = np.array([0, 0, 0])
        self.gyro = np.array([0, 0, 0])
        self.bias = np.array([0, 0, 0, 0, 0, 0])

        self.vel_body_prev = None

    def render(self):
        pass

if __name__ == '__main__':
    # quaternion to euler
    q = np.array([0.5, 0.5, 0.5, 0.5])
    euler = quaternion_to_euler(q)
    print(euler)
    # euler to quaternion
    q = euler_to_quaternion(euler)
    print(q)

    # other quaternion
    q = np.array([0.5, 0.2, -0.5, 0.1])
    # q = np.norm(q)
    euler = quaternion_to_euler(q)
    print(euler)
    q = euler_to_quaternion(euler)
    print(q)
