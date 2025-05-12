#!/usr/bin/env python3
"""
Command-line interface for analyzing W&B runs.

This script allows you to quickly analyze runs from W&B with various filtering
and visualization options from the command line.

Examples:
    # List all runs for a project
    python wandb_analysis_cli.py list --project l2f_bc
    
    # Compare TD3BC and SAC algorithms' performance
    python wandb_analysis_cli.py compare --project l2f_bc --group-by "config.Algo" --metric "metric.test reward"
    
    # Plot performance by slope settings for TD3BC algorithm
    python wandb_analysis_cli.py compare --project l2f_bc --filter "config.Algo=TD3BC" --group-by "config.Slope" --metric "metric.test reward"
    
    # Export results to CSV
    python wandb_analysis_cli.py export --project l2f_bc --output results.csv
"""

import argparse
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb_analyzer import WandbAnalyzer


def parse_filter_string(filter_string: str) -> Dict[str, Any]:
    """Parse a filter string into a dictionary."""
    if not filter_string:
        return {}
    
    filters = {}
    for item in filter_string.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            # Try to convert value to proper type
            try:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            except:
                pass  # Keep as string
                
            filters[key.strip()] = value
            
    return filters


def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description="Analyze W&B runs")
    
    # Global options
    parser.add_argument("--entity", type=str, help="W&B entity (username or team name)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List runs")
    list_parser.add_argument("--project", type=str, required=True, help="W&B project name")
    list_parser.add_argument("--filter", type=str, help="Filter runs (key1=value1,key2=value2)")
    list_parser.add_argument("--min-step", type=int, help="Minimum number of steps")
    list_parser.add_argument("--output", type=str, help="Output file (CSV)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare runs")
    compare_parser.add_argument("--project", type=str, required=True, help="W&B project name or comma-separated list")
    compare_parser.add_argument("--filter", type=str, help="Filter runs (key1=value1,key2=value2)")
    compare_parser.add_argument("--group-by", type=str, required=True, help="Group by field(s), comma-separated")
    compare_parser.add_argument("--metric", type=str, required=True, help="Metric to compare")
    compare_parser.add_argument("--plot-type", type=str, default="bar", 
                              choices=["bar", "box", "scatter", "line"], 
                              help="Type of plot")
    compare_parser.add_argument("--output", type=str, help="Output file for plot (PNG)")
    compare_parser.add_argument("--title", type=str, help="Plot title")
    compare_parser.add_argument("--figsize", type=str, default="12,6", help="Figure size (width,height)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export run data")
    export_parser.add_argument("--project", type=str, required=True, help="W&B project name")
    export_parser.add_argument("--filter", type=str, help="Filter runs (key1=value1,key2=value2)")
    export_parser.add_argument("--include-metrics", type=str, help="Metrics to include (comma-separated)")
    export_parser.add_argument("--include-configs", type=str, help="Config parameters to include (comma-separated)")
    export_parser.add_argument("--output", type=str, required=True, help="Output file (CSV)")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Plot metric history")
    history_parser.add_argument("--project", type=str, required=True, help="W&B project name")
    history_parser.add_argument("--run-ids", type=str, required=True, help="Run IDs (comma-separated)")
    history_parser.add_argument("--metric", type=str, required=True, help="Metric to plot")
    history_parser.add_argument("--smooth", type=int, default=1, help="Smoothing factor")
    history_parser.add_argument("--output", type=str, help="Output file for plot (PNG)")
    history_parser.add_argument("--title", type=str, help="Plot title")
    history_parser.add_argument("--figsize", type=str, default="12,6", help="Figure size (width,height)")
    
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Parse the project names
    if "," in args.project:
        projects = [p.strip() for p in args.project.split(",")]
    else:
        projects = args.project
    
    # Initialize the analyzer
    analyzer = WandbAnalyzer(projects, entity=args.entity)
    
    # Handle the list command
    if args.command == "list":
        filters = parse_filter_string(args.filter)
        runs = analyzer.fetch_runs(filters=filters, min_step=args.min_step)
        
        # Print summary
        print(f"Found {len(runs)} runs")
        
        # Print basic information about each run
        if not runs.empty:
            # Select columns to display
            display_cols = ["id", "name", "state", "created_at"]
            
            # Add any config columns that exist
            config_cols = [col for col in runs.columns if col.startswith("config.")]
            display_cols.extend(config_cols[:5])  # Limit to first 5 to avoid overwhelming output
            
            # Add any metric columns that exist
            metric_cols = [col for col in runs.columns if col.startswith("metric.")]
            display_cols.extend(metric_cols[:5])  # Limit to first 5
            
            # Display the DataFrame
            pd_display_cols = [col for col in display_cols if col in runs.columns]
            print(runs[pd_display_cols].to_string())
            
            # Export if requested
            if args.output:
                runs.to_csv(args.output, index=False)
                print(f"Exported {len(runs)} runs to {args.output}")
    
    # Handle the compare command
    elif args.command == "compare":
        filters = parse_filter_string(args.filter)
        
        # Parse group by
        group_by = [g.strip() for g in args.group_by.split(",")]
        
        # Parse figure size
        figsize = tuple(map(int, args.figsize.split(",")))
        
        # Fetch runs
        runs = analyzer.fetch_runs(filters=filters)
        
        if runs.empty:
            print("No runs found matching the criteria")
            return
        
        # Create the plot
        fig = analyzer.plot_scalar_metric(
            metric=args.metric,
            groupby=group_by,
            plot_type=args.plot_type,
            title=args.title,
            figsize=figsize
        )
        
        # Save or show the plot
        if args.output:
            fig.savefig(args.output)
            print(f"Plot saved to {args.output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()
    
    # Handle the export command
    elif args.command == "export":
        filters = parse_filter_string(args.filter)
        
        # Parse include lists
        include_metrics = None
        if args.include_metrics:
            include_metrics = [m.strip() for m in args.include_metrics.split(",")]
            
        include_configs = None
        if args.include_configs:
            include_configs = [c.strip() for c in args.include_configs.split(",")]
        
        # Fetch runs
        runs = analyzer.fetch_runs(
            filters=filters,
            include_metrics=include_metrics,
            include_configs=include_configs
        )
        
        if runs.empty:
            print("No runs found matching the criteria")
            return
        
        # Export the data
        runs.to_csv(args.output, index=False)
        print(f"Exported {len(runs)} runs to {args.output}")
    
    # Handle the history command
    elif args.command == "history":
        # Parse run IDs
        run_ids = [r.strip() for r in args.run_ids.split(",")]
        
        # Parse figure size
        figsize = tuple(map(int, args.figsize.split(",")))
        
        # Create the plot
        fig = analyzer.plot_metric_history(
            run_ids=run_ids,
            metric_name=args.metric,
            smooth_factor=args.smooth,
            title=args.title,
            figsize=figsize
        )
        
        # Save or show the plot
        if args.output:
            fig.savefig(args.output)
            print(f"Plot saved to {args.output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main() 