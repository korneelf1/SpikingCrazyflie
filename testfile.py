import serial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as time
import re
import serial
import csv
import struct
# from drawnow import *


input_freq_start = '15500000'#input('Enter starting frequency: ')
input_freq_stop = '16500000'#input('Enter stopping frequency: ')
input_freq_step = '100'#input('Enter frequency stepsize: ')


print('Starting frequency: '+input_freq_start)
ser = serial.Serial('/dev/ttyACM0', 9600)
print("Connected to: " + ser.portstr)
ser.flushInput()
ser.write((input_freq_start+';'+input_freq_stop+';'+input_freq_step+'\n').encode())

data = ''
Sweep = True
Input = False
while Sweep == True:
    if ser.inWaiting()>0:
        received = ser.read(ser.inWaiting()).decode('ascii').strip()
        data=data+received
        if data[-1]=='s':
            Sweep = False
ser.close()

data = data.split(",")
array = list(map(int, data[1:-1]))
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_ylim(0,9000)
ax.plot(range(len(array)),array,color='red')
