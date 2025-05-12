"""
Utility for scraping W&B logs and visualizing results.

This tool allows for:
1. Fetching runs from W&B based on filters
2. Extracting performance metrics
3. Creating visualizations to compare different configurations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime


class WandbAnalyzer:
    def __init__(self, project_names: Union[str, List[str]], entity: str = None):
        """
        Initialize the W&B analyzer.
        
        Args:
            project_names: Single project name or list of project names to analyze
            entity: W&B entity (username or team name)
        """
        self.project_names = [project_names] if isinstance(project_names, str) else project_names
        self.entity = entity
        self.api = wandb.Api()
        self.runs_df = None
        
    def fetch_runs(self, 
                   filters: Dict = None, 
                   min_step: int = None, 
                   include_metrics: List[str] = None,
                   include_configs: List[str] = None) -> pd.DataFrame:
        """
        Fetch runs from W&B based on filters.
        
        Args:
            filters: Dictionary of filters to apply (e.g., {"config.Algo": "TD3BC"})
            min_step: Minimum number of steps a run must have completed
            include_metrics: List of metrics to include in the dataframe
            include_configs: List of config parameters to include in the dataframe
            
        Returns:
            DataFrame containing the runs that match the filters
        """
        all_runs = []
        
        for project_name in self.project_names:
            # Construct the query string for the W&B API
            query = f"korneelvandenberghe/{project_name}"
            # if self.entity:
            #     query = f"entity={self.entity} AND {query}"
                
            # # Apply filters
            # if filters:
            #     filter_str = " AND ".join([f"{k}={v}" for k, v in filters.items()])
            #     query = f"{query} AND {filter_str}"
                
            # Fetch runs
            runs = self.api.runs(query)
            
            for run in runs:
                # check if filters are in the run config
                
                # Extract run information
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "algo": run.config.get("Algo", ""),
                    "project": project_name,
                    "state": run.state,
                    "url": run.url,
                    "created_at": run.created_at,
                }
                
                # Extract config parameters
                config = run.config
                if config:
                    if filters:
                        for k, v in filters.items():
                            if k not in run.config:
                                break
                            if run.config[k] != v:
                                break
                    if include_configs:
                        # Only include specified configs
                        for config_key in include_configs:
                            parts = config_key.split('.')
                            value = config
                            for part in parts:
                                if part in value:
                                    value = value[part]
                                else:
                                    value = None
                                    break
                            if value is not None:
                                run_data[f"config.{config_key}"] = value
                    else:
                        # Include all configs (flattened)
                        self._flatten_dict(config, run_data, prefix="config.")
                
                # Extract metrics
                summary = run.summary
                if summary:
                    if include_metrics:
                        # Only include specified metrics
                        for metric in include_metrics:
                            if metric in summary:
                                run_data[f"metric.{metric}"] = summary[metric]
                    else:
                        # Include all metrics
                        for k, v in summary.items():
                            if k.startswith('_'):  # Skip internal keys
                                continue
                            if isinstance(v, (int, float, str, bool)) or v is None:
                                run_data[f"metric.{k}"] = v
                
                # extract test reward and test len history
                history = run.history(keys=["test reward", "test len", "_runtime", "epoch", "_step"])
                if len(history) > 0:
                    run_data["test reward history"] = history["test reward"].tolist()
                    run_data["test len history"] = history["test len"].tolist()
                    run_data["runtime"] = history["_runtime"].tolist()
                    run_data["epoch"] = history["epoch"].tolist()
                    run_data["step"] = history["_step"].tolist()
                # Skip runs that don't have enough steps
                if min_step is not None:
                    if "metric._step" in run_data and run_data["metric._step"] < min_step:
                        continue
                # skip runs that have __runtime less than 10 mins (600 seconds)
                if "__runtime" in run_data and run_data["__runtime"] < 600:
                    continue
                all_runs.append(run_data)
        
        # Convert to DataFrame
        self.runs_df = pd.DataFrame(all_runs)
        # remove runs where test reward history is nan
        self.runs_df = self.runs_df[self.runs_df["test reward history"].notna()]

        return self.runs_df
    
    def _flatten_dict(self, d: Dict, result: Dict, prefix: str = "") -> None:
        """Helper method to flatten nested dictionaries."""
        for k, v in d.items():
            if isinstance(v, dict):
                self._flatten_dict(v, result, prefix + k + ".")
            elif isinstance(v, (int, float, str, bool)) or v is None:
                result[prefix + k] = v
    
    def filter_runs(self, **kwargs) -> pd.DataFrame:
        """
        Filter the already fetched runs based on column values.
        
        Args:
            **kwargs: Column-value pairs to filter on
            
        Returns:
            Filtered DataFrame
        """
        if self.runs_df is None:
            raise ValueError("No runs have been fetched yet. Call fetch_runs() first.")
        
        filtered_df = self.runs_df.copy()
        for column, value in kwargs.items():
            if column in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]
        
        return filtered_df
    
    def extract_best_to100(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the best test reward and test len from the history.
        Also extract the epoch to 100 reward and epoch to 100 len.
        """
      
        # extract the best test reward and test len from the history for each run and add column to df

        best_test_reward = []
        best_test_len = []
        epoch_to_100_reward = []
        epoch_to_100_len = []

        for index, row in data.iterrows():
            best_test_reward.append(row["test reward history"].max())
            best_test_len.append(row["test len history"].max())
            # find the index of the first value in the test reward history that is greater than or equal to 100
            for i, reward in enumerate(row["test reward history"]):
                if reward >= 100:
                    epoch_to_100_reward.append(i)
                    epoch_to_100_len.append(row["test len history"][i])
                    break
        # add to df
        data["best test reward"] = best_test_reward
        data["best test len"] = best_test_len
        data["epoch to 100 reward"] = epoch_to_100_reward
        data["epoch to 100 len"] = epoch_to_100_len
        return data
    
    def plot_reward_histories(self, df):
        # Convert lists to tuples for grouping
        df = df.copy()
        df['config.hidden_sizes'] = df['config.hidden_sizes'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        for hidden_size, group_df in df.groupby('config.hidden_sizes'):
            plt.figure(figsize=(10, 6))
            plt.title(f"Test Reward History — Hidden Size: {hidden_size}")
            plt.xlabel("Step")
            plt.ylabel("Test Reward")

            for schedule, schedule_df in group_df.groupby('config.Schedule'):
                # Convert all histories to arrays
                histories = [np.array(h) for h in schedule_df['test reward history']]

                # Truncate all histories to the same minimum length
                min_len = min(len(h) for h in histories)
                trimmed = np.array([h[:min_len] for h in histories])

                # Compute mean and std across runs
                mean = trimmed.mean(axis=0)
                std = trimmed.std(axis=0)
                steps = np.arange(min_len)

                # Plot with shaded std
                plt.plot(steps, mean, label=f"Schedule: {schedule}")
                plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

            plt.legend(title="Schedule")
            plt.grid(True)
            plt.tight_layout()
            # save to file
            plt.savefig(f"test_reward_history_hidden_size_{hidden_size}.png")
            plt.close()
            # plt.show()
    def plot_scalar_metric(self, 
                           metric: str, 
                           groupby: Union[str, List[str]], 
                           data: pd.DataFrame = None,
                           plot_type: str = 'bar',
                           title: str = None,
                           figsize: Tuple[int, int] = (12, 6),
                           **kwargs) -> plt.Figure:
        """
        Create a plot comparing a scalar metric across different groups.
        
        Args:
            metric: Metric to plot (column name in the dataframe)
            groupby: Column(s) to group by
            data: DataFrame to use (if None, uses the filtered runs_df)
            plot_type: Type of plot ('bar', 'box', or 'scatter')
            title: Plot title
            figsize: Figure size
            **kwargs: Additional arguments to pass to the plotting function
            
        Returns:
            Matplotlib figure
        """
        if data is None:
            if self.runs_df is None:
                raise ValueError("No runs have been fetched yet. Call fetch_runs() first.")
            data = self.runs_df
        
        if not title:
            title = f"{metric} by {groupby}"
            
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'bar':
            grouped = data.groupby(groupby)[metric].mean().reset_index()
            
            if isinstance(groupby, list) and len(groupby) > 1:
                # Create a pivot table for multi-level grouping
                pivot_cols = groupby[1:]
                pivot_df = grouped.pivot(index=groupby[0], columns=pivot_cols, values=metric)
                pivot_df.plot(kind='bar', ax=ax, **kwargs)
            else:
                # Simple bar chart
                sns.barplot(x=groupby if isinstance(groupby, str) else groupby[0], 
                          y=metric, data=grouped, ax=ax, **kwargs)
                
        elif plot_type == 'box':
            sns.boxplot(x=groupby if isinstance(groupby, str) else groupby[0], 
                       y=metric, data=data, ax=ax, **kwargs)
            
        elif plot_type == 'scatter':
            if isinstance(groupby, list) and len(groupby) > 1:
                # Color by the second groupby column
                sns.scatterplot(x=groupby[0], y=metric, 
                              hue=groupby[1], data=data, ax=ax, **kwargs)
            else:
                # Simple scatter plot
                sns.scatterplot(x=groupby, y=metric, data=data, ax=ax, **kwargs)
                
        elif plot_type == 'line':
            if isinstance(groupby, list) and len(groupby) > 1:
                # Multiple lines by group
                for name, group in data.groupby(groupby[1:]):
                    group.plot(x=groupby[0], y=metric, ax=ax, label=name, **kwargs)
            else:
                # Simple line plot
                sns.lineplot(x=groupby, y=metric, data=data, ax=ax, **kwargs)
        
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xlabel(groupby if isinstance(groupby, str) else groupby[0])
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_metric_history(self, 
                           run_ids: List[str],
                           metric_name: str,
                           smooth_factor: int = 1,
                           title: str = None,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the history of a metric for selected runs.
        
        Args:
            run_ids: List of run IDs to include
            metric_name: Name of the metric to plot
            smooth_factor: Window size for smoothing (1 = no smoothing)
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not title:
            title = f"{metric_name} History"
            
        fig, ax = plt.subplots(figsize=figsize)
        
        for run_id in run_ids:
            run = self.api.run(f"{self.entity}/{self.project_names[0]}/{run_id}" 
                              if self.entity else f"{self.project_names[0]}/{run_id}")
            history = run.history(keys=[metric_name, "_step"])
            
            if smooth_factor > 1:
                history[f"{metric_name}_smooth"] = history[metric_name].rolling(window=smooth_factor).mean()
                metric_to_plot = f"{metric_name}_smooth"
            else:
                metric_to_plot = metric_name
                
            ax.plot(history["_step"], history[metric_to_plot], label=run.name)
            
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def export_results(self, 
                      data: pd.DataFrame = None, 
                      filename: str = None) -> None:
        """
        Export results to a CSV file.
        
        Args:
            data: DataFrame to export (if None, uses the runs_df)
            filename: Filename to save to (if None, generates one based on timestamp)
        """
        if data is None:
            if self.runs_df is None:
                raise ValueError("No runs have been fetched yet. Call fetch_runs() first.")
            data = self.runs_df
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wandb_analysis_{timestamp}.csv"
            
        data.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = WandbAnalyzer(
        project_names=["l2f_bc"], 
        entity=None  # Set your W&B username or team name here
    )
    
    # Fetch runs with specific filters
    runs = analyzer.fetch_runs(
        filters={"Algo": "BC"},
        include_metrics=["test reward", "test len"],
        include_configs=["Slope", "Schedule", "hidden_sizes"]
    )
    # save runs to file
    runs.to_csv("runs_l2f_bc.csv", index=False) 
    print(f"Found {len(runs)} runs")
    analyzer.plot_reward_histories(runs)
    runs = analyzer.extract_best_to100(runs)
    # Filter runs further if needed
    # filtered_runs = analyzer.filter_runs(**{
    #     "config.Slope":,
    # })
    
    # Plot results
    fig = analyzer.plot_scalar_metric(
        metric="metric.test reward",
        groupby=["config.Schedule", "config.hidden_sizes"],
        data=runs,
        plot_type="bar"
    )
    
    # Save the figure
    fig.savefig("slope_schedule_comparison.png")
    
    # Export results
    analyzer.export_results(runs, "analysis_results.csv") 