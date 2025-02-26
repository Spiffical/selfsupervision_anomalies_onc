# Weights & Biases (wandb) Utilities

This document describes the `wandb_utils.py` module which centralizes all wandb-related functionality in the project. This module provides a consistent interface for initializing wandb, logging metrics, and managing runs.

## Key Functions

### `init_wandb(args, project_name="ssamba", entity=None, group=None, run_id=None)`

Initializes a wandb run with the given configuration.

- `args`: Argument object containing experiment configuration
- `project_name`: Name of the wandb project (default: "ssamba")
- `entity`: Optional wandb entity (team or username)
- `group`: Optional group name for organizing related runs
- `run_id`: Optional run ID for resuming a previous run

Returns the wandb run object.

**Note**: This function now tracks initialization status using `args.wandb_initialized` to prevent double initialization when called from multiple places in the codebase.

### `log_training_metrics(metrics_dict, step=None, use_wandb=True)`

Logs training metrics to wandb.

- `metrics_dict`: Dictionary of metrics to log
- `step`: Optional step number
- `use_wandb`: Whether to log to wandb (default: True)

**Enhanced**: Now properly extracts and logs hydrophone metrics with epoch information.

### `log_validation_metrics(metrics, task, epoch=None, prefix="", use_wandb=True)`

Logs validation metrics with task-specific handling.

- `metrics`: Dictionary of metrics to log
- `task`: Task name for task-specific metric handling
- `epoch`: Optional epoch number
- `prefix`: Optional prefix for metric names
- `use_wandb`: Whether to log to wandb (default: True)

**Enhanced**: Now ensures per-hydrophone metrics are properly tracked with epoch information.

### `log_hydrophone_metrics(hydrophone_metrics, epoch=None, prefix="")`

Logs hydrophone-specific metrics.

- `hydrophone_metrics`: Dictionary of hydrophone metrics
- `epoch`: Optional epoch number
- `prefix`: Optional prefix for metric names

**Enhanced**: Now includes epoch information in all hydrophone metrics and creates visualizations of metrics over time.

### `create_hydrophone_plots(hydrophone_metrics, epoch, prefix, metric_types)`

Creates interactive plots for hydrophone metrics over time.

- `hydrophone_metrics`: Dictionary of hydrophone metrics
- `epoch`: Current epoch number
- `prefix`: Prefix for metric names
- `metric_types`: Set of metric types to plot

**New**: Generates bar charts showing per-hydrophone performance for each metric type.

### `log_model_artifact(model, model_path, name, type="model", metadata=None)`

Logs a model as a wandb artifact.

- `model`: Model to log
- `model_path`: Path to save the model
- `name`: Name of the artifact
- `type`: Type of the artifact (default: "model")
- `metadata`: Optional metadata to attach to the artifact

### `finish_run()`

Finishes the current wandb run.

**Note**: This function checks if a wandb run is active before attempting to finish it, and in `traintest_mask.py` it only finishes runs that were initialized in that function.

## Integration with Existing Classes

The module is integrated with the following classes:

- `MetricsTracker`: Uses `log_training_metrics` to log training metrics
- `ValidationMetricsCollector`: Uses `log_validation_metrics` to log validation metrics

## Files Updated to Use wandb_utils

The following files have been updated to use the centralized wandb utilities:

- `run_amba_spectrogram.py`: Uses `init_wandb`, `log_training_metrics`, and `finish_run`
- `run_amba.py`: Uses `init_wandb`, `log_training_metrics`, and `finish_run`
- `run.py`: Uses `init_wandb`, `log_training_metrics`, and `finish_run`
- `traintest.py`: Uses `log_training_metrics` instead of direct wandb calls
- `test_training.py`: Uses `init_wandb` and `finish_run`
- `traintest_mask.py`: Updated to prevent double initialization and only finish runs it started

## Configuration

To enable wandb logging, set `args.use_wandb = True` in your script. If wandb is not enabled, the utility functions will gracefully handle this case without causing errors.

## Preventing Double Initialization

The codebase has been updated to prevent wandb from being initialized multiple times when functions are called in sequence. This is achieved by:

1. Tracking initialization status in `args.wandb_initialized`
2. Checking this flag before initializing wandb
3. Only finishing wandb runs in the same function that initialized them

This is particularly important when scripts like `run_amba_spectrogram.py` call functions in `traintest_mask.py` that would otherwise initialize wandb again.

## Hydrophone Metrics Visualization

The module now provides enhanced visualization of per-hydrophone metrics:

1. **Metrics Over Time**: Each hydrophone's metrics are tracked as a function of epoch, allowing you to see how performance evolves for each hydrophone.

2. **Interactive Tables**: Tables showing per-hydrophone metrics are created periodically (at epoch 1 and every 10 epochs thereafter).

3. **Bar Charts**: Bar charts comparing performance across hydrophones are generated for each metric type.

4. **Sample Distribution**: A table showing the distribution of samples across hydrophones is logged periodically.

These visualizations help identify:
- Which hydrophones perform better or worse
- How performance for each hydrophone changes over time
- Potential data imbalances across hydrophones 