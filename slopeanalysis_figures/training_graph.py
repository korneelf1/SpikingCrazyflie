"""
Utility for interpolating a DataFrame to have more granular steps.

This module provides functions to:
1. Create a new DataFrame with more granular steps for a specified column
2. Interpolate the missing values in other columns
"""

from typing import List, Union, Optional
import os   
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Set the style to match NeurIPS aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 17
mpl.rcParams['axes.titlesize'] = 17
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['figure.titlesize'] = 20

# Set a professional color palette (colorblind-friendly)
colors = sns.color_palette("colorblind")


def interpolate_dataframe(
    df: pd.DataFrame,
    step_column: str,
    target_step_size: int = 1,
    columns_to_interpolate: Optional[List[str]] = None,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Interpolate a DataFrame to have more granular steps for a specified column.
    
    Args:
        df: The original DataFrame
        step_column: The column that contains the step values to be made more granular
        target_step_size: The desired step size (default: 1)
        columns_to_interpolate: List of columns to interpolate. If None, all columns except step_column will be interpolated.
        method: Interpolation method to use ('linear', 'cubic', 'polynomial', etc.) as supported by pandas
        
    Returns:
        A new DataFrame with more granular steps and interpolated values
    """
    if df.empty:
        return df.copy()
    
    # Sort the DataFrame by the step column
    df_sorted = df.sort_values(by=step_column).reset_index(drop=True)
    
    # Ensure step_column is numeric
    df_sorted[step_column] = pd.to_numeric(df_sorted[step_column], errors='coerce')
    
    # Get the original range of the step column
    min_step = df_sorted[step_column].min()
    max_step = df_sorted[step_column].max()
    
    # Create a new DataFrame with the more granular step values
    new_steps = np.arange(min_step, max_step + target_step_size, target_step_size)
    new_df = pd.DataFrame({step_column: new_steps})
    
    # Ensure new_df step_column is numeric
    new_df[step_column] = pd.to_numeric(new_df[step_column], errors='coerce')
    
    # Merge the original DataFrame with the new one
    merged_df = pd.merge(new_df, df_sorted, on=step_column, how='left')
    
    # Determine which columns to interpolate
    if columns_to_interpolate is None:
        columns_to_interpolate = [col for col in merged_df.columns if col != step_column]
    
    # Interpolate the missing values
    merged_df[columns_to_interpolate] = merged_df[columns_to_interpolate].interpolate(method=method)
    
    return merged_df


def interpolate_dataframe_with_groups(
    df: pd.DataFrame,
    step_column: str,
    group_columns: List[str],
    target_step_size: int = 1,
    columns_to_interpolate: Optional[List[str]] = None,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Interpolate a DataFrame to have more granular steps for a specified column,
    while respecting group boundaries.
    
    Args:
        df: The original DataFrame
        step_column: The column that contains the step values to be made more granular
        group_columns: Columns to group by before interpolating
        target_step_size: The desired step size (default: 1)
        columns_to_interpolate: List of columns to interpolate. If None, all columns except step_column and group_columns will be interpolated.
        method: Interpolation method to use ('linear', 'cubic', 'polynomial', etc.) as supported by pandas
        
    Returns:
        A new DataFrame with more granular steps and interpolated values
    """
    if df.empty:
        return df.copy()
    
    # Determine which columns to interpolate
    if columns_to_interpolate is None:
        columns_to_interpolate = [col for col in df.columns 
                                 if col != step_column and col not in group_columns]
    
    # Group the DataFrame and apply interpolation to each group
    grouped = df.groupby(group_columns)
    results = []
    
    for name, group in grouped:
        # Interpolate this group
        interpolated_group = interpolate_dataframe(
            group,
            step_column=step_column,
            target_step_size=target_step_size,
            columns_to_interpolate=columns_to_interpolate,
            method=method
        )
        
        # Add back the group column values
        if isinstance(name, tuple):
            for i, col in enumerate(group_columns):
                interpolated_group[col] = name[i]
        else:
            interpolated_group[group_columns[0]] = name
            
        results.append(interpolated_group)
    
    # Combine all the interpolated groups
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()
    
def extract_mean_std(df: pd.DataFrame, step_column: str, n_steps: int = 100000):
    # drop all columns that contain _MIN or _MAX
    df = df.drop(columns=[col for col in df.columns if '_MIN' in col or '_MAX' in col])
    
    # drop all rows where the step column is less than the last_n_steps
    df = df[df[step_column] <= n_steps]
    # drop the step column
    df = df.drop(columns=[step_column])
    # return the mean and std of the test reward column for each row
    return df.mean(axis=1), df.std(axis=1)

def extract_best_time_to_100(df: pd.DataFrame, step_column: str, score_column: str, correct_for_fast_reward=False):
    if not isinstance(score_column, list):
        score_column = [score_column]
    # find the epoch at which 100 is surpassed for the first time
    time_to_100 = df[df[score_column[0]] >= 100][step_column].values[0]
    sorted_vals = df[score_column[0]].dropna().sort_values(ascending=True) # filtering outliers
    best_score = sorted_vals[-20:].mean()

    if correct_for_fast_reward:
        best_score = best_score*.61
    if time_to_100<2:
        time_to_100=1
    return time_to_100, best_score

def create_bar_plot(results: dict):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib as mpl

    # === Set NeurIPS-style aesthetics ===
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['axes.labelsize'] = 17
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['legend.fontsize'] = 13
    mpl.rcParams['figure.titlesize'] = 20

    # === Settings ===
    method = ['BC', 'TD3']
    slopes = ['Slope: 2', 'Slope: 50', 'Slope: 100', 'Interval', 'Adaptive']
    colors = sns.color_palette("colorblind", len(slopes))

    # === Extract data ===
    time_to_100_data = {m: {s: results[m][s][0] if s in results[m] else 0 for s in slopes} for m in method}
    best_perf_data = {m: {s: results[m][s][1] if s in results[m] else 0 for s in slopes} for m in method}

    def plot_grouped_bar(data, title, ylabel, filename):
        bar_width = 0.15
        index = np.arange(len(method))
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        for i, slope in enumerate(slopes):
            values = [data[m][slope] for m in method]
            ax.bar(index + i * bar_width, values, width=bar_width, label=slope, color=colors[i])

        # X-axis and Labels
        ax.set_xticks(index + (len(slopes)/2 - 0.5) * bar_width)
        ax.set_xticklabels(method)
        ax.set_ylabel(ylabel, weight='bold')
        ax.set_title(title, pad=15)

        # Legend styling
        ax.legend(title="Slope", loc='best', frameon=True, framealpha=0.95, edgecolor='lightgray', fancybox=False)

        # Spine and grid styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, axis='y')

        # Y padding
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        plt.tight_layout()

        # Save in multiple formats
        for ext in ['pdf', 'png', 'svg']:
            plt.savefig(f'{filename}.{ext}', format=ext, bbox_inches='tight', dpi=300)

        plt.show()

    plot_grouped_bar(time_to_100_data, 'Time to Reach 100 Performance (lower better)', 'Time to 100', 'neurips_time_to_100_plot')
    plot_grouped_bar(best_perf_data, 'Best Performance Achieved (higher better)', 'Best Performance', 'neurips_best_performance_plot')


    
if __name__ == "__main__":

    # Load all csv files
    files = (pathlib.Path(__file__).parent.parent / "slopeanalysis_figures").glob("*.csv")
    results_bc = {}
    results_td3 = {}
    for file in files:
        df = pd.read_csv(file)
        
        # # Parse filenames for legend labels
        if "bc" in file.stem:
            if "slope" in file.stem:
                key = "BC"
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                # Interpolate the dataframe
                df = interpolate_dataframe(df, "epoch", target_step_size=1)
                # drop all cols with _step in the name
                
                # if slope: 50 in name
                columns = [col for col in df.columns if 'slope: 50' in col]
                columns += ['epoch']
                df_50 = df[columns]
                # score col contains returns 
                score_col = [col for col in df_50.columns if 'test reward' in col]
                # extract the best time to 100
                best_time_to_100, best_score = extract_best_time_to_100(df_50, step_column="epoch", score_column=score_col[0])
                results_bc['Slope: 50'] = (best_time_to_100, best_score)

                # if slope: 100 in name
                columns = [col for col in df.columns if 'slope: 100' in col]
                columns += ['epoch']
                df_100 = df[columns]
                # score col contains test rewards 
                score_col = [col for col in df_100.columns if 'test reward' in col]
                # extract the best time to 100
                best_time_to_100, best_score = extract_best_time_to_100(df_100, step_column="epoch", score_column=score_col[0])
                results_bc['Slope: 100'] = (best_time_to_100, best_score)

                # if Slope: 2 in name
                columns = [col for col in df.columns if 'slope: 2' in col]

                columns += ['epoch']
                df_2 = df[columns]
                # score col contains test rewards 
                score_col = [col for col in df_2.columns if 'test reward' in col]
                # extract the best time to 100
                best_time_to_100, best_score = extract_best_time_to_100(df_2, step_column="epoch", score_column=score_col[0])
                results_bc['Slope: 2'] = (best_time_to_100, best_score)

                # # Extract the mean and std of the test reward column
                # mean_50, std_50 = extract_mean_std(df_50, step_column="epoch")
                # mean_100, std_100 = extract_mean_std(df_100, step_column="epoch")
                # mean_2, std_2 = extract_mean_std(df_2, step_column="epoch")
                # results_bc['Slope: 50'] = (mean_50, std_50)
                # results_bc['Slope: 100'] = (mean_100, std_100)
                # results_bc['Slope: 2'] = (mean_2, std_2)
            elif "adaptive" in file.stem:
                key = "Adaptive"
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                # Interpolate the dataframe
                df = interpolate_dataframe(df, "epoch", target_step_size=1)

                # if "Schedule: adaptive" in name
                columns = [col for col in df.columns if 'Schedule: adaptive' in col]
                score_col = [col for col in df.columns if 'test reward' in col]
                columns += ['epoch']
                df_adaptive = df[columns]
                # score col contains test rewards 
                score_col = [col for col in df_adaptive.columns if 'test reward' in col]
                # extract the best time to 100
                best_time_to_100, best_score = extract_best_time_to_100(df_adaptive, step_column="epoch", score_column=score_col[0])
                results_bc['Adaptive'] = (best_time_to_100, best_score)

                # if "Schedule: interval" in name
                columns = [col for col in df.columns if 'Schedule: interval' in col]
                score_col = [col for col in df.columns if 'test reward' in col]
                columns += ['epoch']
                df_interval = df[columns]
                # score col contains test rewards 
                score_col = [col for col in df_interval.columns if 'test reward' in col]
                # extract the best time to 100
                best_time_to_100, best_score = extract_best_time_to_100(df_interval, step_column="epoch", score_column=score_col[0])
                results_bc['Interval'] = (best_time_to_100, best_score)

                # Extract the mean and std of the test reward column
                # mean_adaptive, std_adaptive = extract_mean_std(df_adaptive, step_column="epoch")
                # mean_interval, std_interval = extract_mean_std(df_interval, step_column="epoch")
                # results_bc['Schedule: adaptive'] = (mean_adaptive, std_adaptive)
                # results_bc['Schedule: interval'] = (mean_interval, std_interval)
                
        if "td3" in file.stem:
            if "slope" in file.stem:
                key = "TD3"

                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                # Interpolate the dataframe
                # df = interpolate_dataframe(df, "Relative Time (Process)", target_step_size=1)
                # rescale Relative Time (Process) to be between 0 and 300
                df['Relative Time (Process)'] = df['Relative Time (Process)'] * 300/df['Relative Time (Process)'].max()
                
                # remove all cols with _step in the name or _MIN or _MAX

                # if slope: 50 in name
                columns = [col for col in df.columns if 'slope: 50' in col]
                
                columns += ['Relative Time (Process)']
                df_50 = df[columns]
                # score col contains returns 
                score_col = [col for col in df_50.columns if 'returns' in col]
                # extract the best time to 100
                best_time_to_100, best_score = extract_best_time_to_100(df_50, step_column="Relative Time (Process)", score_column=score_col, correct_for_fast_reward=True)
                results_td3['Slope: 50'] = (best_time_to_100, best_score)

                # if slope: 100 in name
                columns = [col for col in df.columns if 'slope: 100' in col]
                columns += ['Relative Time (Process)']
                df_100 = df[columns]
                # score col contains returns 
                score_col = [col for col in df_100.columns if 'returns' in col]
                # extract the best time to 100  
                best_time_to_100, best_score = extract_best_time_to_100(df_100, step_column="Relative Time (Process)", score_column=score_col, correct_for_fast_reward=True)
                results_td3['Slope: 100'] = (best_time_to_100, best_score)

                # if Slope: 2 in name
                columns = [col for col in df.columns if 'slope: 2' in col]
                columns += ['Relative Time (Process)']
                df_2 = df[columns]  
                # score col contains returns 
                score_col = [col for col in df_2.columns if 'returns' in col]
                # extract the best time to 100  
                best_time_to_100, best_score = extract_best_time_to_100(df_2, step_column="Relative Time (Process)", score_column=score_col, correct_for_fast_reward=True)
                results_td3['Slope: 2'] = (best_time_to_100, best_score)

    
            elif "adaptive" in file.stem:
                key = "TD3"
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])

                # Interpolate the dataframe
                # df = interpolate_dataframe(df, "Relative Time (Process)", target_step_size=1)
                # drop all cols with _step in the name
                df['Relative Time (Process)'] = df['Relative Time (Process)'] * 300/df['Relative Time (Process)'].max()


                # if slope_schedule: adaptive in name
                columns = [col for col in df.columns if 'slope_schedule: adaptive' in col]
                columns += ['Relative Time (Process)']
                df_adaptive = df[columns]
                # score col contains returns 
                score_col = [col for col in df_adaptive.columns if 'returns' in col]
                # extract the best time to 100  
                best_time_to_100, best_score = extract_best_time_to_100(df_adaptive, step_column="Relative Time (Process)", score_column=score_col, correct_for_fast_reward=True)
                results_td3['Adaptive'] = (best_time_to_100, best_score)

                # if slope_schedule: interval in name   
                columns = [col for col in df.columns if 'slope_schedule: interval' in col]
                columns += ['Relative Time (Process)']
                df_interval = df[columns]
                # score col contains returns 
                score_col = [col for col in df_interval.columns if 'returns' in col]
                # extract the best time to 100  
                best_time_to_100, best_score = extract_best_time_to_100(df_interval, step_column="Relative Time (Process)", score_column=score_col, correct_for_fast_reward=True)
                results_td3['Interval'] = (best_time_to_100, best_score)


    # create bar plot
    # add to one dict
    results = {}
    results['BC'] = results_bc
    results['TD3'] = results_td3
    create_bar_plot(results)

    