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
    
    # Get the original range of the step column
    min_step = df_sorted[step_column].min()
    max_step = df_sorted[step_column].max()
    
    # Create a new DataFrame with the more granular step values
    new_steps = np.arange(min_step, max_step + target_step_size, target_step_size)
    new_df = pd.DataFrame({step_column: new_steps})
    
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

  
if __name__ == "__main__":

    # Load all csv files
    files = (pathlib.Path(__file__).parent.parent / "curriculum_Results").glob("*.csv")
    results = {}
    for file in files:
        if not "curriculum" in file.stem:
            continue
        df = pd.read_csv(file)
        
        # Parse filenames for legend labels
        if "TD3BCJSRL" in file.stem:
            key = "TD3BC+JSRL"
        elif "TD3BC" in file.stem:
            key = "TD3BC"
        elif "BC" in file.stem:
            key = "BC"
        elif "TD3" in file.stem:
            key = "TD3"
    
        
        # Interpolate the dataframe
        df = interpolate_dataframe(df, "Step", target_step_size=1)
        
        # Extract the mean and std of the test reward column
        mean, std = extract_mean_std(df, step_column="Step")
        results[key] = (mean, std)

    # sort the results by the key
    results = dict(sorted(results.items()))
    # Create the enhanced plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot each method with professional styling
    for i, (key, (mean, std)) in enumerate(results.items()):
        ax.plot(mean, label=key, color=colors[i], linewidth=2)
        ax.fill_between(range(len(mean)), mean - std, mean + std, 
                    alpha=0.15, color=colors[i], linewidth=0)

    # Format x-axis with scientific notation
    def format_x_ticks(x, pos):
        return f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'

    ax.xaxis.set_major_formatter(FuncFormatter(format_x_ticks))

    # Add labels and title
    ax.set_xlabel('Training Steps', weight='bold')
    ax.set_ylabel('Test Reward', weight='bold')
    ax.set_title('Comparison of Test Reward under Curriculum', pad=20)

    # Enhance the legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, 
            edgecolor='lightgray', fancybox=False)

    # Adjust spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Add a light grid for better readability
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust y-axis to add some padding
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Tight layout for optimal spacing
    plt.tight_layout()

    # Add annotations if needed
    # ax.annotate('Best performing method', xy=(60000, results['TD3BC+JSRL'][0][60000]), 
    #             xytext=(65000, results['TD3BC+JSRL'][0][60000] + 50),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #             fontsize=9)

    # To show or save the figure
    plt.savefig('neurips_comparison_plot_reward.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('neurips_comparison_plot_reward.png', format='png', bbox_inches='tight', dpi=300)
    # save an svg version
    plt.savefig('neurips_comparison_plot_reward.svg', format='svg', bbox_inches='tight')
    plt.show()