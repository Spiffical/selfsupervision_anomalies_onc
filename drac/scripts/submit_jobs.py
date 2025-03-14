#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path
from typing import List, Optional

def submit_linked_jobs(
    dataset_path: str,
    wandb_project: str = "amba_spectrogram",
    wandb_group: str = "default_experiment",
    train_ratio: float = 0.8,
    num_jobs: int = 5,
    project_path: str = os.path.expanduser("~/ssamba"),
    resume: bool = True,
    task: str = "pretrain_joint",
    exp_dir: str = "/exp",
    time_limit: str = "08:00:00",
    job_name: str = "amba_spectrogram"
) -> None:
    """
    Submit a series of linked SLURM jobs for a single training ratio.
    Each job depends on the completion of the previous job.
    """
    # Create output and error directories
    os.makedirs("out", exist_ok=True)
    os.makedirs("err", exist_ok=True)
    
    prev_job_id = None
    
    for i in range(1, num_jobs + 1):
        # Prepare sbatch command
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}_{train_ratio}",
            f"--time={time_limit}",
            f"--output=out/{job_name}_ratio{train_ratio}_job{i}_%j.out",
            f"--error=err/{job_name}_ratio{train_ratio}_job{i}_%j.err",
        ]
        
        # Add dependency if not first job
        if prev_job_id is not None:
            cmd.append(f"--dependency=afterany:{prev_job_id}")
        
        # Add script and its arguments
        cmd.extend([
            "submit_amba_spectrogram.sh",
            dataset_path,
            wandb_project,
            wandb_group,
            str(train_ratio),
            project_path,
            str(resume).lower(),
            task,
            exp_dir
        ])
        
        # Submit job
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            prev_job_id = result.stdout.strip()
            print(f"Submitted job {i} for ratio {train_ratio} with Job ID: {prev_job_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {i}: {e}")
            print(f"Command output: {e.output}")
            break

def submit_training_size_experiments(
    dataset_path: str,
    wandb_project: str = "amba_spectrogram",
    wandb_group: str = "training_size_experiment",
    train_ratios: Optional[List[float]] = None,
    num_jobs: int = 5,
    project_path: str = os.path.expanduser("~/ssamba"),
    resume: bool = True,
    task: str = "pretrain_joint",
    exp_dir: str = "/exp",
    time_limit: str = "08:00:00",
    job_name: str = "amba_spectrogram"
) -> None:
    """
    Submit linked jobs for multiple training ratios.
    """
    if train_ratios is None:
        train_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    
    for ratio in train_ratios:
        print(f"\nSubmitting {num_jobs} linked jobs for training ratio {ratio}")
        submit_linked_jobs(
            dataset_path=dataset_path,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            train_ratio=ratio,
            num_jobs=num_jobs,
            project_path=project_path,
            resume=resume,
            task=task,
            exp_dir=exp_dir,
            time_limit=time_limit,
            job_name=job_name
        )

def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for AMBA training")
    
    # Required arguments
    parser.add_argument("dataset_path", help="Path to the training dataset")
    
    # Optional arguments with defaults
    parser.add_argument("--wandb-project", default="amba_spectrogram", help="W&B project name")
    parser.add_argument("--wandb-group", default="default_experiment", help="W&B group name")
    parser.add_argument("--num-jobs", type=int, default=5, help="Number of linked jobs per training ratio")
    parser.add_argument("--project-path", default=os.path.expanduser("~/ssamba"), help="Path to project directory")
    parser.add_argument("--resume", type=lambda x: x.lower() == "true", default=True, help="Whether to resume from checkpoint")
    parser.add_argument("--task", default="pretrain_joint", help="Training task")
    parser.add_argument("--exp-dir", default="/exp", help="Experiment directory")
    parser.add_argument("--time-limit", default="08:00:00", help="Job time limit in HH:MM:SS format")
    parser.add_argument("--job-name", default="amba_spectrogram", help="Base name for the SLURM jobs")
    
    # Mode selection
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                       help="'single' for one linked job, 'multi' for training size experiments")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training ratio for single mode (ignored in multi mode)")
    parser.add_argument("--train-ratios", type=float, nargs="+",
                       help="List of training ratios for multi mode (default: [0.1, 0.2, 0.4, 0.6, 0.8])")

    args = parser.parse_args()

    if args.mode == "single":
        submit_linked_jobs(
            dataset_path=args.dataset_path,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            train_ratio=args.train_ratio,
            num_jobs=args.num_jobs,
            project_path=args.project_path,
            resume=args.resume,
            task=args.task,
            exp_dir=args.exp_dir,
            time_limit=args.time_limit,
            job_name=args.job_name
        )
    else:  # multi mode
        submit_training_size_experiments(
            dataset_path=args.dataset_path,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            train_ratios=args.train_ratios,
            num_jobs=args.num_jobs,
            project_path=args.project_path,
            resume=args.resume,
            task=args.task,
            exp_dir=args.exp_dir,
            time_limit=args.time_limit,
            job_name=args.job_name
        )

if __name__ == "__main__":
    main() 