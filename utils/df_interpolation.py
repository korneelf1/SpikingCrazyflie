"""
Utility for interpolating a DataFrame to have more granular steps.

This module provides functions to:
1. Create a new DataFrame with more granular steps for a specified column
2. Interpolate the missing values in other columns
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional


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


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame with steps of 10
    data = {
        'step': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'value1': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'value2': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    # Interpolate to step size of 1
    interpolated_df = interpolate_dataframe(df, 'step', target_step_size=1)
    
    print("\nInterpolated DataFrame (sample):")
    print(interpolated_df.head(15))  # Show first 15 rows
    
    # Example with groups
    group_data = {
        'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'step': [0, 10, 20, 30, 0, 10, 20, 30],
        'value': [0, 10, 20, 30, 100, 90, 80, 70]
    }
    group_df = pd.DataFrame(group_data)
    
    print("\nOriginal DataFrame with groups:")
    print(group_df)
    
    # Interpolate with groups
    interpolated_group_df = interpolate_dataframe_with_groups(
        group_df, 'step', ['group'], target_step_size=2)
    
    print("\nInterpolated DataFrame with groups:")
    print(interpolated_group_df) 