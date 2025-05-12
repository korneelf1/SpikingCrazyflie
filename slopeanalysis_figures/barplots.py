"""
Utility for interpolating a DataFrame to have more granular steps.

This module provides functions to:
1. Create a new DataFrame with more granular steps for a specified column
2. Interpolate the missing values in other columns
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional
import os   
import matplotlib.pyplot as plt
import pathlib


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

def extract_data(file_path: str) -> pd.DataFrame:
        # load BC_wand_exports/BC-16-16.CSV
    df_bc = pd.read_csv(file_path)


    # keep only columns that contain test reward and the epoch
    df_bc = df_bc.filter(regex='test reward|epoch')

    # remove cols with MIN or MAX in the name
    df_bc = df_bc.filter(regex='^(?!.*(MIN|MAX)$).*')

    # cut at 300 epochs
    df_bc = df_bc[df_bc["epoch"] <= 300]

    # interpolate such that epochs step size is 1
    df = interpolate_dataframe(df_bc, "epoch", target_step_size=1)

    results_time_to_100 = {}
    results_best_perf = {}
        # find the epoch at which 100 is surpassed for the first time
    for col in df.columns:
        if "Schedule" in col:
            time_to_100 = np.max(df[df[col] >= 100][col],1)
            best_perf = np.max(df[col].max(),1)
            

            if len(time_to_100) > 0:
                # get index of first non-nan value
                time_to_100 = time_to_100.index[0]
                if time_to_100 == 0:
                    time_to_100 = 1 # for plotting purposes
            else:
                time_to_100 = -10
            if 'adaptive' in col:
                results_time_to_100['adaptive'] =  time_to_100
                results_best_perf['adaptive'] = best_perf
            elif 'fixed' in col or 'false' in col:
                results_time_to_100['fixed'] = time_to_100
                results_best_perf['fixed'] = best_perf
            elif 'true' in col or 'interval' in col:
                results_time_to_100['interval'] = time_to_100
                results_best_perf['interval'] = best_perf

    return results_time_to_100, results_best_perf

def extract_data_from_all_files(folder_path: str) -> pd.DataFrame:
    results = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            results[file] = extract_data(os.path.join(folder_path, file))
    return results

def create_bar_plot(results: dict):
    # creates 2 bar plots of the results, one for time to 100 and one for best performance
    # bars are grouped by model size
    # Extract model sizes and methods
    model_sizes = list(results.keys())
    # sort model sizes by the number in the name
    # model_sizes.sort(key=lambda x: int(x.split('-')[0]))
    
    # remove the 32 32 model and 256 128 model
    # model_sizes = [size for size in model_sizes if '32' not in size and '256' not in size]
    model_sizes = [size for size in model_sizes if '32' not in size]
    methods = ['fixed', 'interval', 'adaptive']

    # Prepare data for plotting
    time_to_100_data = {method: [results[size][0][method] for size in model_sizes] for method in methods}
    best_perf_data = {method: [results[size][1][method] for size in model_sizes] for method in methods}
    
    # Set up the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set width of bars and positions of the bars
    bar_width = 0.25
    r1 = np.arange(len(model_sizes))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create bars for time to 100 - reordered to fixed, interval, adaptive
    ax1.bar(r1, time_to_100_data['fixed'], width=bar_width, label='Fixed', color='lightgreen')
    ax1.bar(r2, time_to_100_data['interval'], width=bar_width, label='Interval', color='salmon')
    ax1.bar(r3, time_to_100_data['adaptive'], width=bar_width, label='Adaptive', color='skyblue')

    # Create bars for best performance - reordered to fixed, interval, adaptive
    ax2.bar(r1, best_perf_data['fixed'], width=bar_width, label='Fixed', color='lightgreen')
    ax2.bar(r2, best_perf_data['interval'], width=bar_width, label='Interval', color='salmon')
    ax2.bar(r3, best_perf_data['adaptive'], width=bar_width, label='Adaptive', color='skyblue')

    # Customize the plots
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Epochs to Reach 100 (-10 if never reached)')
    ax1.set_title('Epochs to Reach Performance of 100')
    ax1.set_xticks([r + bar_width for r in range(len(model_sizes))])
    # remove the .csv from the model sizes
    model_sizes = [size.replace('.csv', '') for size in model_sizes]
    ax1.set_xticklabels(model_sizes)
    ax1.legend()

    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Best Performance')
    ax2.set_title('Best Performance Achieved')
    ax2.set_xticks([r + bar_width for r in range(len(model_sizes))])
    
    ax2.set_xticklabels(model_sizes)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
def create_comparison_plots(results_bc, results_td3):
    """
    Create two bar plots comparing BC and TD3 results:
    1. Time to reach 100 reward
    2. Best performance achieved
    """
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for plotting
    categories = ['Slope: 2', 'Slope: 50', 'Slope: 100', 'Adaptive', 'Interval']
    bc_times = [results_bc[cat][0] for cat in categories]
    td3_times = [results_td3[cat][0] for cat in categories]
    bc_perf = [results_bc[cat][1] for cat in categories]
    td3_perf = [results_td3[cat][1] for cat in categories]
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(categories))
    r2 = [x + barWidth for x in r1]
    
    # Create the first subplot (Time to 100)
    ax1.bar(r1, bc_times, width=barWidth, label='BC', color='#2ecc71')
    ax1.bar(r2, td3_times, width=barWidth, label='TD3', color='#3498db')
    
    # Add labels and title for first subplot
    ax1.set_xlabel('Method', fontweight='bold')
    ax1.set_ylabel('Time to Reach 100', fontweight='bold')
    ax1.set_title('Time to Reach Reward of 100', pad=20)
    ax1.set_xticks([r + barWidth/2 for r in range(len(categories))])
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    
    # Create the second subplot (Best Performance)
    ax2.bar(r1, bc_perf, width=barWidth, label='BC', color='#2ecc71')
    ax2.bar(r2, td3_perf, width=barWidth, label='TD3', color='#3498db')
    
    # Add labels and title for second subplot
    ax2.set_xlabel('Method', fontweight='bold')
    ax2.set_ylabel('Best Reward Achieved', fontweight='bold')
    ax2.set_title('Best Performance Achieved', pad=20)
    ax2.set_xticks([r + barWidth/2 for r in range(len(categories))])
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('comparison_plots.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('comparison_plots.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig('comparison_plots.svg', format='svg', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and process the data
    files = (pathlib.Path(__file__).parent.parent / "slopeanalysis_figures").glob("*.csv")
    results_bc = {}
    results_td3 = {}
    
    for file in files:
        df = pd.read_csv(file)
        
        if "bc" in file.stem:
            if "slope" in file.stem:
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                df = interpolate_dataframe(df, "epoch", target_step_size=1)
                
                # Process slope: 50
                columns = [col for col in df.columns if 'slope: 50' in col]
                columns += ['epoch']
                df_50 = df[columns]
                score_col = [col for col in df_50.columns if 'test reward' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_50, step_column="epoch", score_column=score_col)
                results_bc['Slope: 50'] = (best_time_to_100, best_score)
                
                # Process slope: 100
                columns = [col for col in df.columns if 'slope: 100' in col]
                columns += ['epoch']
                df_100 = df[columns]
                score_col = [col for col in df_100.columns if 'test reward' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_100, step_column="epoch", score_column=score_col)
                results_bc['Slope: 100'] = (best_time_to_100, best_score)
                
                # Process slope: 2
                columns = [col for col in df.columns if 'slope: 2' in col]
                columns += ['epoch']
                df_2 = df[columns]
                score_col = [col for col in df_2.columns if 'test reward' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_2, step_column="epoch", score_column=score_col)
                results_bc['Slope: 2'] = (best_time_to_100, best_score)
                
            elif "adaptive" in file.stem:
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                df = interpolate_dataframe(df, "epoch", target_step_size=1)
                
                # Process adaptive
                columns = [col for col in df.columns if 'Schedule: adaptive' in col]
                columns += ['epoch']
                df_adaptive = df[columns]
                score_col = [col for col in df_adaptive.columns if 'test reward' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_adaptive, step_column="epoch", score_column=score_col)
                results_bc['Adaptive'] = (best_time_to_100, best_score)
                
                # Process interval
                columns = [col for col in df.columns if 'Schedule: interval' in col]
                columns += ['epoch']
                df_interval = df[columns]
                score_col = [col for col in df_interval.columns if 'test reward' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_interval, step_column="epoch", score_column=score_col)
                results_bc['Interval'] = (best_time_to_100, best_score)
                
        if "td3" in file.stem:
            if "slope" in file.stem:
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                df['Relative Time (Process)'] = df['Relative Time (Process)'] * 300/df['Relative Time (Process)'].max()
                
                # Process slope: 50
                columns = [col for col in df.columns if 'slope: 50' in col]
                columns += ['Relative Time (Process)']
                df_50 = df[columns]
                score_col = [col for col in df_50.columns if 'returns' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_50, step_column="Relative Time (Process)", score_column=score_col)
                results_td3['Slope: 50'] = (best_time_to_100, best_score)
                
                # Process slope: 100
                columns = [col for col in df.columns if 'slope: 100' in col]
                columns += ['Relative Time (Process)']
                df_100 = df[columns]
                score_col = [col for col in df_100.columns if 'returns' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_100, step_column="Relative Time (Process)", score_column=score_col)
                results_td3['Slope: 100'] = (best_time_to_100, best_score)
                
                # Process slope: 2
                columns = [col for col in df.columns if 'slope: 2' in col]
                columns += ['Relative Time (Process)']
                df_2 = df[columns]
                score_col = [col for col in df_2.columns if 'returns' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_2, step_column="Relative Time (Process)", score_column=score_col)
                results_td3['Slope: 2'] = (best_time_to_100, best_score)
                
            elif "adaptive" in file.stem:
                df = df.drop(columns=[col for col in df.columns if '_step' in col or '_MIN' in col or '_MAX' in col])
                df['Relative Time (Process)'] = df['Relative Time (Process)'] * 300/df['Relative Time (Process)'].max()
                
                # Process adaptive
                columns = [col for col in df.columns if 'slope_schedule: adaptive' in col]
                columns += ['Relative Time (Process)']
                df_adaptive = df[columns]
                score_col = [col for col in df_adaptive.columns if 'returns' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_adaptive, step_column="Relative Time (Process)", score_column=score_col)
                results_td3['Adaptive'] = (best_time_to_100, best_score)
                
                # Process interval
                columns = [col for col in df.columns if 'slope_schedule: interval' in col]
                columns += ['Relative Time (Process)']
                df_interval = df[columns]
                score_col = [col for col in df_interval.columns if 'returns' in col][0]
                best_time_to_100, best_score = extract_best_time_to_100(df_interval, step_column="Relative Time (Process)", score_column=score_col)
                results_td3['Interval'] = (best_time_to_100, best_score)
    
    # Create the comparison plots
    create_comparison_plots(results_bc, results_td3)
   