# W&B Analysis Tools

This directory contains utilities for analyzing and visualizing experiments logged in Weights & Biases (W&B).

## Overview

These tools allow you to:

1. **Fetch runs** from W&B based on various filters (algorithm, model size, slope settings, etc.)
2. **Extract metrics** from completed runs
3. **Visualize results** using different plot types (bar charts, scatter plots, line plots)
4. **Export data** for further analysis

## Files

- `wandb_analyzer.py`: Main class for fetching and analyzing W&B runs
- `wandb_analysis_cli.py`: Command-line interface for quick analyses

## Usage Examples

### Python API

```python
from utils.wandb_analyzer import WandbAnalyzer

# Initialize the analyzer
analyzer = WandbAnalyzer(project_names=["thesis_graphs_fast_learning", "l2f"], entity="your-username")

# Fetch runs with specific configuration
runs = analyzer.fetch_runs(
    filters={"config.Algo": "TD3BC"},
    include_metrics=["test reward", "test len"],
    include_configs=["Slope", "Schedule", "hidden_sizes"]
)

# Filter runs further
filtered_runs = analyzer.filter_runs(
    **{"config.Slope": 25.0, "state": "finished"}
)

# Visualize results - compare performance across different slope schedules
fig = analyzer.plot_scalar_metric(
    metric="metric.test reward",
    groupby=["config.Schedule", "config.Slope"],
    data=filtered_runs,
    plot_type="bar"
)

# Save the visualization
fig.savefig("slope_schedule_comparison.png")

# Export the filtered data
analyzer.export_results(filtered_runs, "analysis_results.csv")
```

### Command Line Interface

List all runs in a project:
```bash
python utils/wandb_analysis_cli.py list --project thesis_graphs_fast_learning
```

Compare algorithm performance:
```bash
python utils/wandb_analysis_cli.py compare \
    --project thesis_graphs_fast_learning \
    --group-by "config.Algo" \
    --metric "metric.test reward" \
    --output algorithm_comparison.png
```

Compare performance across different slope settings for TD3BC:
```bash
python utils/wandb_analysis_cli.py compare \
    --project thesis_graphs_fast_learning \
    --filter "config.Algo=TD3BC" \
    --group-by "config.Slope" \
    --metric "metric.test reward" \
    --plot-type bar \
    --output slope_comparison.png
```

View learning curves for specific runs:
```bash
python utils/wandb_analysis_cli.py history \
    --project thesis_graphs_fast_learning \
    --run-ids "run1,run2,run3" \
    --metric "test reward" \
    --smooth 5 \
    --output learning_curves.png
```

Export filtered data:
```bash
python utils/wandb_analysis_cli.py export \
    --project thesis_graphs_fast_learning \
    --filter "config.Slope=25.0,config.Schedule=constant" \
    --include-metrics "test reward,test len" \
    --include-configs "Slope,Schedule,hidden_sizes,Algo" \
    --output td3bc_results.csv
```

## Common Analysis Scenarios

### Algorithm Comparison

Compare the performance of different algorithms (TD3BC, SAC, DDPG):

```bash
python utils/wandb_analysis_cli.py compare \
    --project thesis_graphs_fast_learning,l2f \
    --group-by "config.Algo" \
    --metric "metric.test reward" \
    --plot-type box \
    --output algorithm_comparison.png
```

### Slope Settings Analysis

Analyze how different slope settings affect performance:

```bash
python utils/wandb_analysis_cli.py compare \
    --project thesis_graphs_fast_learning \
    --filter "config.Algo=TD3BC" \
    --group-by "config.Slope" \
    --metric "metric.test reward" \
    --plot-type bar \
    --output slope_settings.png
```

### Schedule Comparison

Compare different slope scheduling strategies:

```bash
python utils/wandb_analysis_cli.py compare \
    --project thesis_graphs_fast_learning \
    --filter "config.Algo=TD3BC" \
    --group-by "config.Schedule" \
    --metric "metric.test reward" \
    --output schedule_comparison.png
```

### Model Size Impact

Analyze how model size affects performance:

```bash
python utils/wandb_analysis_cli.py compare \
    --project thesis_graphs_fast_learning \
    --group-by "config.hidden_sizes" \
    --metric "metric.test reward" \
    --output model_size_impact.png
```

## Dependencies

- wandb
- pandas
- numpy
- matplotlib
- seaborn

## Installation

Make sure you have the required dependencies installed:

```bash
pip install wandb pandas numpy matplotlib seaborn
``` 