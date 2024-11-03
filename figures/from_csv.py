import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set a seaborn color palette globally
sns.set_palette('muted', desat=.75)  # Or you can use 'muted', 'bright', 'dark', etc.

def plot_avg_rew(filename, plot_type='time_to_100'):
    '''
    plot type:
    rewards - plot the rewards
    time_to_100 - plot the time to reach 100 reward
    final - plot the final best reward
    '''
    # file names have slopeSLOPE in them, extract the slope for the legend
    if 'slope2' in filename:
        slope = 'SNN, single transition training' 
    elif 'slope' in filename:
        slope = filename.split('slope')[1].split('_')[0]
    elif 'rnn' in filename:
        slope = 'SNN, sequence training'
    elif 'sched' in filename:
        slope = 'scheduled'
    elif '_2_' in filename:
        slope = '2'
    elif '_50_' in filename:
        slope = '50'
    elif '_100_' in filename:
        slope = '100'
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
    # remove columns with all NaN values or only 1 unique value
    data = data.dropna(axis=1, how='all')
    data = data.loc[:, data.nunique() > 5]

    data.interpolate(inplace=True, method='linear')
    # Calculate mean and std for each row in the DataFrame, ONLY for cols that have reward in name
    data_rews = data.filter(like='reward').fillna(0)
    data['mean'] = data_rews.mean(axis=1)
    data['std'] = data_rews.std(axis=1)
    # compute std of a point and its 3 neighbors
    # first add shifted means to the data
    data['mean_shifted'] = data['mean'].shift(1)
    data['mean_shifted_2'] = data['mean'].shift(2)
    data['mean_shifted_3'] = data['mean'].shift(3)
    data_rews_shifted = data.filter(like='mean').fillna(0)
    data['std_shifted'] = data_rews_shifted.std(axis=1)

    # smoothen the mean values by interpolating for all Step values 1->3000
    # add cols in Step for all values 1->3000 for entries with no Step value fill in NaN
    
    # data['mean'] = data['mean'].mean()
    # rescale steps between 0 and 500
    if slope == 'SNN, sequence training':
        max_step = 1.5e6

    elif slope == 'SNN, single transition training':
        max_step = 900000
    else:
        max_step = 500
    data['Step'] = data['Step'] / data['Step'].max() * max_step
    # round data['Step'] to nearest integer
    data['Step'] = data['Step'].round().astype(int)
    
    # smoothen mean and std data using exponentially weighted moving average
    data['mean'] = data['mean'].ewm(span=30).mean()
    
    # if std all 0, use std_shifted
    if data['std'].sum() == 0:
        data['std'] = data['std_shifted']

    data['std'] = data['std'].ewm(span=20).mean()
    # find the mean x value where the mean crosses 100 and the relatcro
    epoch_100 = data[data['mean'] > 100]['Step'].min()
    # given a horizontal line at y  = 100 find the std accross that line
    # max val mean + std, min val mean - std
    data['mean+std'] = data['mean'] + data['std']
    data['mean-std'] = data['mean'] - data['std']
    # where data mean+std > 100 and mean-std < 100
    data_till_100 = np.abs(data[data['mean+std'] > 100]['Step'].min()-epoch_100)
    data_before_100 = np.abs(data[data['mean-std'] < 100]['Step'].max()-epoch_100)
    std = (data_till_100+data_before_100)/2
    print(f'Epoch 100: {epoch_100} with std {(std)}')


    # data['mean'] = data['mean'].rolling(window=3).mean()
    # data['std'] = data['std'].rolling(window=10).mean()

    # data = data.interpolate(method='linear')
    if str(epoch_100) =='nan':
        epoch_100 = 10000
        std = 0
    if plot_type=='rewards':

        plt.plot(data['Step'],data['mean'], label=f'Slope {slope}')
        plt.fill_between(data['Step'], data['mean']-data['std'], data['mean']+data['std'], alpha=0.1)

        return epoch_100, std
    elif plot_type=='time_to_100':
       
        return epoch_100, std

    elif plot_type=='final':
        # return the mean max value and related std of the runs in the rewards
        max = data_rews.max(axis=1).mean()
        std_of_max = data_rews.max(axis=1).std()
        return max, std_of_max
import os
import re
# files = os.listdir('figures/rl/')
# for file in files:
#     if 'slope2' in file or 'slope100' in file:
#         plot_avg_rew(f'figures/rl/{file}')

# plot_avg_rew('figures/compare_algos/wandb_rnn.csv')
# plot_avg_rew('figures/rl/td3_slope2_.csv')
time_to_100_slope2 = {}
time_to_100_std_slope2 = {}
time_to_100_slope50 = {}
time_to_100_std_slope50 = {}
time_to_100_slope100 = {}
time_to_100_std_slope100 = {}
time_to_100_sched = {}
time_to_100_std_sched = {}

slopes = ['2', '50', '100', 'sched']
for slope in slopes:
    for hidden_size in ['1616', '3232', '6464', '128128']:
        try:
            time, std = plot_avg_rew(f'figures/bc/wandb_td3_{slope}_128128{hidden_size}.csv', plot_type='final')
            if slope == '2':
                time_to_100_slope2[hidden_size] = time
                time_to_100_std_slope2[hidden_size] = std
            elif slope == '50':
                time_to_100_slope50[hidden_size] = time
                time_to_100_std_slope50[hidden_size] = std
            elif slope == '100':
                time_to_100_slope100[hidden_size] = time
                time_to_100_std_slope100[hidden_size] = std
            else:
                time_to_100_sched[hidden_size] = time
                time_to_100_std_sched[hidden_size] = std

        except:
            print(f'No file for slope {slope} and hidden size {hidden_size}')
            if slope == '2':
                time_to_100_slope2[hidden_size] = 1e4
                time_to_100_std_slope2[hidden_size] = 0
            elif slope == '50':
                time_to_100_slope50[hidden_size] = 1e4
                time_to_100_std_slope50[hidden_size] = 0
            elif slope == '100':
                time_to_100_slope100[hidden_size] = 1e4
                time_to_100_std_slope100[hidden_size] = 0
            else:
                time_to_100_sched[hidden_size] = 0
                time_to_100_std_sched[hidden_size] = 0
            
# plot_avg_rew('figures/bc/wandb_bc_2_3232.csv')
# plot_avg_rew('figures/bc/wandb_bc_50_3232.csv')
# plot_avg_rew('figures/bc/wandb_bc_100_3232.csv')
# plot_avg_rew('figures/bc/wandb_bc_sched_3232.csv')
from pprint import pprint
print('Slope 2')
pprint(time_to_100_slope2)
pprint(time_to_100_std_slope2)
print('Slope 50')
pprint(time_to_100_slope50)
pprint(time_to_100_std_slope50)
print('Slope 100')
pprint(time_to_100_slope100)
pprint(time_to_100_std_slope100)
print('Scheduled')
pprint(time_to_100_sched)
pprint(time_to_100_std_sched)



# plt.xlim(0, 500)
# plt.ylim(0,500)
# plot the time to reach 100 reward with std
# Plot the time to reach 100 reward with std using plot and fill_between
plt.plot(time_to_100_slope2.keys(), time_to_100_slope2.values(), label='Slope 2')
plt.fill_between(time_to_100_slope2.keys(), 
                 [time - err for time, err in zip(time_to_100_slope2.values(), time_to_100_std_slope2.values())], 
                 [time + err for time, err in zip(time_to_100_slope2.values(), time_to_100_std_slope2.values())], 
                 alpha=0.1)

plt.plot(time_to_100_slope50.keys(), time_to_100_slope50.values(), label='Slope 50')
plt.fill_between(time_to_100_slope50.keys(), 
                 [time - err for time, err in zip(time_to_100_slope50.values(), time_to_100_std_slope50.values())], 
                 [time + err for time, err in zip(time_to_100_slope50.values(), time_to_100_std_slope50.values())], 
                 alpha=0.1)

plt.plot(time_to_100_slope100.keys(), time_to_100_slope100.values(), label='Slope 100')
plt.fill_between(time_to_100_slope100.keys(), 
                 [time - err for time, err in zip(time_to_100_slope100.values(), time_to_100_std_slope100.values())], 
                 [time + err for time, err in zip(time_to_100_slope100.values(), time_to_100_std_slope100.values())], 
                 alpha=0.1)

plt.plot(time_to_100_sched.keys(), time_to_100_sched.values(), label='Scheduled')
plt.fill_between(time_to_100_sched.keys(), 
                 [time - err for time, err in zip(time_to_100_sched.values(), time_to_100_std_sched.values())], 
                 [time + err for time, err in zip(time_to_100_sched.values(), time_to_100_std_sched.values())], 
                 alpha=0.1)

plt.legend(fontsize=13)
plt.xlabel('Hidden layer sizes', fontsize=13)
plt.ylabel('Epochs', fontsize=13)
plt.ylim(0, 500)
plt.suptitle('Epochs to surpass a return of 100 \nbehavorial cloning', fontsize=15)
plt.show()