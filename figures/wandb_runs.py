import wandb
import pandas as pd

# Authenticate to wandb (replace with your API key if needed)
wandb.login()

# Define your project name and column to filter by
entity = ""  # Replace with your W&B username or team name
project = "l2f_bc"  # Replace with your project name
filter_column = "Algo"  # Replace with the column name you want to filter by
filter_value = "BC"  # Replace with the specific value you want to filter
cols_to_keep = ["test reward", "test len"]  # Replace with the columns you want to keep
# Initialize a list to store all filtered data
all_data = []

# Fetch all runs in the project
api = wandb.Api()
runs = api.runs(f"{project}")

data_slope_2 = []
data_slope_25 = []
data_slope_50 = []
data_slope_100 = []
data_slope_scheduled = []
rewards_slope_2 = []
rewards_slope_25 = []
rewards_slope_50 = []
rewards_slope_100 = []
rewards_slope_scheduled = []

# Iterate over runs to collect data based on the column filter
for run in runs:
    # print(run.name)
    if run.config.get(filter_column) != filter_value:
        continue
    else:
        # Fetch the history of the run (the logged data)
        run_df = run.history()[cols_to_keep]
        # add a column filled with "hidden_sizes" to the DataFrame, add it for each row

        run_df["hidden_sizes"] = str(run.config.get("hidden_sizes"))

        if run.config.get("Scheduled") == True:
            data_slope_scheduled.append(run_df)
            # rewards_slope_scheduled.append(run_df["test reward"].to_list())
        elif run.config.get("slope") == 2:
            data_slope_2.append(run_df)
            # rewards_slope_2.append(run_df["test reward"].to_list())
        elif run.config.get("slope") == 25:
            data_slope_25.append(run_df)
            # rewards_slope_25.append(run_df["test reward"].to_list())
        elif run.config.get("slope") == 50:
            data_slope_50.append(run_df)
            # rewards_slope_50.append(run_df["test reward"].to_list())
        elif run.config.get("slope") == 100:
            data_slope_100.append(run_df)
            # rewards_slope_100.append(run_df["test reward"].to_list())

# make sure all data have same amount of rows
max_len = max([len(data_slope_2), len(data_slope_25), len(data_slope_50), len(data_slope_100), len(data_slope_scheduled)])
# for each dataframe, interpolate the values in reward column, such that no Nan Values are present
for df_lst in [data_slope_2, data_slope_25, data_slope_50, data_slope_100, data_slope_scheduled]:
    for df in df_lst:
        df = df.interpolate()
        x = df["test reward"].to_list()
        y = np.linspace(0, len(x), max_len)
        plt.plot(df["test reward"])

data_slope_2 = pd.concat(data_slope_2, ignore_index=False).reset_index().rename(columns={'index': 'time'})
hidden_sizes = data_slope_2["hidden_sizes"].unique()
# create dataframe with reward columns for each hidden_sizes
# Get unique hidden_sizes
unique_hidden_sizes = data_slope_2['hidden_sizes'].unique()

# Loop through unique hidden_sizes and create new columns
for size in unique_hidden_sizes:
    # Create a new column for each hidden_size and fill with corresponding test_reward values
    data_slope_2[f'test_reward_hidden_size_{size}'] = data_slope_2.apply(
        lambda row: row['test reward'] if row['hidden_sizes'] == size else None, axis=1
    )

import matplotlib.pyplot as plt
import numpy as np
# plot each column, with y val the col value (if not None) and x val the index
for size in unique_hidden_sizes:
    if size !='None':
        # zip the x and y values and drop the pairs where y = 0
        data_zipped = list(zip(data_slope_2.time, data_slope_2[f'test_reward_hidden_size_{size}'].dropna()))
        data_zipped = [pair for pair in data_zipped if pair[1] != 0 or pair[1]<0]
        # sort data_zipped based on x values
        data_zipped = sorted(data_zipped, key=lambda x: x[0])
        x, y = zip(*data_zipped)
        # for each x compute mean and std of y values
        # first create x array with # of columns = number of 0 instances in y and rows = max(x)
        y_s = []
        prev_x = 0
        prev_lst = []
        for xy in data_zipped:
            if xy[0] == prev_x:
                prev_lst.append(xy[1])
            else:
                y_s.append(prev_lst)
                prev_lst = [xy[1]]
                prev_x = xy[0]
        means = [np.mean(lst) for lst in y_s]
        std = [np.std(lst) for lst in y_s]
        plt.plot(x, means, label=f'Hidden Size {size}')
        plt.fill_between(x, np.array(means)-np.array(std), np.array(means)+np.array(std), alpha=0.5)

plt.legend()
plt.show()