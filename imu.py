import numpy

class IMU:
    def __init__(self, noise=0.0, bias=0.0):
        self.noise = noise
        self.bias = bias

        self.accel = numpy.array([0, 0, 0])
        self.gyro = numpy.array([0, 0, 0])
        self.mag = numpy.array([0, 0, 0])
    
    def update(self, dt):
        self.accel = numpy.random.normal(self.accel, self.noise) - self.bias
        self.gyro = numpy.random.normal(self.gyro, self.noise) - self.bias
        self.mag = numpy.random.normal(self.mag, self.noise) - self.bias

    def simulate(self, state, dt):
        '''
        state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        '''
        # update acceleration
        self.accel = state[3:6]
        # update angular velocity
        self.gyro = state[10:13]

    def reset(self):
        self.accel = numpy.array([0, 0, 0])
        self.gyro = numpy.array([0, 0, 0])
        self.mag = numpy.array([0, 0, 0])