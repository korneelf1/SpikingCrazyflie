import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# create a smoothening function for a list, exponential moving average
def smoothen_list(lst, alpha=0.36):
    smoothed = [lst[0]]
    for i in range(1, len(lst)):
        smoothed.append(alpha * lst[i] + (1 - alpha) * smoothed[-1])
    return smoothed

# open wandb_dfs/df_grouped_test_reward_('[128, 128]', 'adaptive').csv
df_grouped_test_reward = pd.read_csv("wandb_dfs/df_grouped_test_reward_('[64, 64]',).csv")

# convert str of list of floats to list of floats
df_grouped_test_reward["epoch"] = df_grouped_test_reward["epoch"].apply(lambda x: eval(x))

# same for test reward history
df_grouped_test_reward["test reward history"] = df_grouped_test_reward["test reward history"].apply(lambda x: eval(x))

# create new column with header 'slope setting' if config.Schedule in ['adaptive', 'interval', 'true'] else config.Slope
slope_setting = []
for i, row in df_grouped_test_reward.iterrows():
    if row["config.Schedule"] in ['adaptive', 'interval', 'True']:
        if row["config.Schedule"] == 'True':
            slope_setting.append('interval')
        else:
            slope_setting.append(row["config.Schedule"])
    else:
        slope_setting.append(row["config.Slope"])
df_grouped_test_reward["slope setting"] = slope_setting

epochs = df_grouped_test_reward["epoch"].tolist()[0]

# use epochs as index and create column for each row with the test reward history
df_grouped_test_reward = df_grouped_test_reward.reset_index(drop=True)
reward_histories = df_grouped_test_reward["test reward history"].tolist()
# make each reward history length of longest reward history
max_length = max(len(history) for history in reward_histories)

reward_histories_corrected = {}
j = 0
for i, history in enumerate(reward_histories):
    add = True
    # check if history has mostly negative values, if so ignore
    if sum(1 for x in history if x < 0) > len(history) * 0.5:
        print(f"History {i} has mostly negative values, ignoring")
        add = False
    # if max of history is less than 50, ignore
    if max(history) < 50:
        print(max(history))
        print(f"History {i} has max reward less than 50, ignoring")
        add = False
    if len(history) < max_length:
        # fill with average of 10 last values
        history.extend([np.mean(history[-10:])] * (max_length - len(history)))

    if add:
        
        reward_histories_corrected[f"reward_history_{j}__{df_grouped_test_reward['slope setting'].iloc[i]}"] = history
        j += 1

# print(list(reward_histories_corrected.values())[0])
epochs = list(range(0,list(reward_histories_corrected.values())[0].__len__(),1))
print(len(epochs))
# create df with entries of first row epochs as index and test reward histories as columns
df_grouped_test_reward = pd.DataFrame(reward_histories_corrected, index=epochs)

# for each column, plot the reward history
adaptive_cols = [col for col in df_grouped_test_reward.columns if "adaptive" in col]
interval_cols = [col for col in df_grouped_test_reward.columns if "interval" in col]
other_cols = [col for col in df_grouped_test_reward.columns if "interval" not in col and "adaptive" not in col]

# adaptive max for each column
adaptive_max = df_grouped_test_reward[adaptive_cols].max(axis=0)
adaptive_max_avg = np.mean(adaptive_max)
adaptive_max_std = np.std(adaptive_max)

# time to 100
cols_gt_100 = df_grouped_test_reward[adaptive_cols][df_grouped_test_reward[adaptive_cols]>100]
# get indeces
time_to_100 = time_to_100.index[0]

# plot adaptive max
plt.plot(adaptive_max, color='red', label='Adaptive max')
plt.plot(adaptive_max_avg, color='blue', label='Adaptive max avg')
plt.plot(adaptive_max_std, color='green', label='Adaptive max std')

# plot adaptive avg
plt.plot(df_grouped_test_reward[adaptive_cols[0]], color='red', label='Adaptive avg')
plt.plot(df_grouped_test_reward[interval_cols[0]], color='blue', label='Interval avg')
plt.plot(df_grouped_test_reward[other_cols[0]], color='green', label='Other avg')
plt.legend()
plt.show()