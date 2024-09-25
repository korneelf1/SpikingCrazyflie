from l2f_gym import Learning2Fly
import serial
import time

teensy = serial.Serial(port='COM4', baudrate=115200, timeout=1.)


def write_read(x): 
    teensy.write(bytes(x, 'utf-8')) 
    time.sleep(0.05) 
    data = teensy.readline() 
    return data 

def main():
    env = Learning2Fly()
    obs = env.reset()[0]
    t_prev = time.now()
    for i in range(1000):
        time.wait(0.01-(time.now()-t_prev))
        action = write_read(obs)
        obs, reward, done, info = env.step(action)
        t_prev = time.now()
        print(obs)
        print(reward)
        print(done)
        print(info)
        print("------")
        t_prev = time.now()
      
            