import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set a seaborn color palette globally
sns.set_palette('muted', desat=.75)  # Or you can use 'muted', 'bright', 'dark', etc.

def plot_avg_rew(filename):
    # file names have slopeSLOPE in them, extract the slope for the legend
    if 'slope' in filename:
        slope = filename.split('slope')[1].split('_')[0]
    else:
        slope ='scheduled'
    if '64' in filename:
        hidden_layer_size = 64
    elif '128' in filename:
        hidden_layer_size = 128

    # Load the data
    data = pd.read_csv(filename)
    # disregard cols with MIN or MAX in name
    data = data.loc[:, ~data.columns.str.contains('MIN|MAX')]
    data.interpolate(inplace=True, method='linear')
    # Calculate mean and std for each row in the DataFrame, ONLY for cols that have reward in name
    data_rews = data.filter(like='returns').fillna(0)
    data['mean'] = data_rews.mean(axis=1)
    data['std'] = data_rews.std(axis=1)

    # smoothen the mean values by interpolating for all Step values 1->3000
    # add cols in Step for all values 1->3000 for entries with no Step value fill in NaN
    
    # data['mean'] = data['mean'].mean()
    # rescale steps between 0 and 500
    data['Step'] = data['Step'] / data['Step'].max() * 500
    # round data['Step'] to nearest integer
    data['Step'] = data['Step'].round().astype(int)
    
    # smoothen mean and std data using exponentially weighted moving average
    data['mean'] = data['mean'].ewm(span=30).mean()
    data['std'] = data['std'].ewm(span=25).mean()
    # data['mean'] = data['mean'].rolling(window=3).mean()
    # data['std'] = data['std'].rolling(window=10).mean()

    # data = data.interpolate(method='linear')
    
    

    plt.plot(data['Step'],data['mean'], label=f'Slope {slope}')
    plt.fill_between(data['Step'], data['mean']-data['std'], data['mean']+data['std'], alpha=0.1)
# find all files in figures/bc/
import os
import re
files = os.listdir('figures/rl/')
for file in files:
    if 'slope2' in file or 'slope100' in file:
        plot_avg_rew(f'figures/rl/{file}')
plt.xlim(0, 500)
plt.ylim(0,500)
plt.legend(fontsize=13)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Average Return', fontsize=13)
plt.suptitle('Average Return for Surrogate Slopes, TD3', fontsize=15)
plt.show()