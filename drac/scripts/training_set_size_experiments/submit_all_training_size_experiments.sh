#!/bin/bash

# Check if dataset path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_path>"
  exit 1
fi

DATASET_PATH=$1

# Submit jobs with different training ratios
sbatch submit_training_size_experiment.sh "$DATASET_PATH" 0.2
sbatch submit_training_size_experiment.sh "$DATASET_PATH" 0.4
sbatch submit_training_size_experiment.sh "$DATASET_PATH" 0.6
sbatch submit_training_size_experiment.sh "$DATASET_PATH" 0.8
sbatch submit_training_size_experiment.sh "$DATASET_PATH" 0.9
