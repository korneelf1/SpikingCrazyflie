import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = '/home/korneel/Desktop/logkorneel01.csv'

df = pd.read_csv(file, sep=',')
print(df.head())
z_pos = df['snn_control.velBodyX'].to_list()
x_vals = np.arange(0, len(z_pos), 1)/100
plt.plot(x_vals, z_pos)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position over time')
plt.show()