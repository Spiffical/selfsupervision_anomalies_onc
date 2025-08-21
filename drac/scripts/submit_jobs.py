#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import os
import shlex
from typing import List, Optional

def submit_linked_jobs(
    dataset_path: str,
    job_name: str,
    num_jobs: int,
    wandb_project: str,
    wandb_group: str,
    project_path: str,
    exp_dir: str,
    training_type: str = "selfsupervised",
    wandb_entity: Optional[str] = None,
    exclude_labels: Optional[List[str]] = None,
    pretrained_path: Optional[str] = None,
    resume: bool = True,
    task: str = "ft_cls",
    train_ratio: float = 0.8,
    time_limit: str = "0-12:00:00",
    dry_run: bool = False,
    multiclass: bool = False,
    num_classes: Optional[int] = None,
    venv_path: Optional[str] = None,
    gpus: Optional[str] = None,
    sbatch_flags: Optional[List[str]] = None,
) -> None:
    """Submit a series of linked jobs where each subsequent job depends on the previous one.

    Args:
        dataset_path: Path to the dataset file
        job_name: Base name for the jobs
        num_jobs: Number of jobs to submit
        wandb_project: W&B project name
        wandb_group: W&B group name
        project_path: Path to the project directory
        exp_dir: Directory for experiment outputs
        training_type: Type of training ('selfsupervised' or 'supervised')
        wandb_entity: W&B entity (team) name
        exclude_labels: List of labels to exclude from training
        pretrained_path: Path to pretrained model
        resume: Whether to resume training
        task: Training task type (only used for pretrain_finetune)
        train_ratio: Ratio of data to use for training
        time_limit: Time limit for each job
        dry_run: Whether to only print commands without executing
    """
    # Create output directories if they don't exist
    Path("out").mkdir(exist_ok=True)
    Path("err").mkdir(exist_ok=True)

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

        # Optional GPU flag and extra sbatch flags
        has_gpu_in_sbatch = False
        if sbatch_flags:
            # Check if user already passed a GPU-related flag
            for raw_flag in sbatch_flags:
                if "--gpus=" in raw_flag or "--gres=" in raw_flag:
                    has_gpu_in_sbatch = True
                    break
        if gpus and not has_gpu_in_sbatch:
            # If value includes a GPU type (letters), prefer --gres to avoid --gpus* conflicts
            if any(c.isalpha() for c in gpus):
                cmd.append(f"--gres=gpu:{gpus}")
            else:
                cmd.append(f"--gpus={gpus}")
        if sbatch_flags:
            for raw_flag in sbatch_flags:
                cmd.extend(shlex.split(raw_flag))

        # Add dependency if not first job
        if prev_job_id is not None:
            cmd.append(f"--dependency=afterany:{prev_job_id}")

        # Choose script based on training type (use absolute paths to avoid stale copies)
        scripts_dir = Path(__file__).resolve().parent
        supervised_script = str(scripts_dir / "submit_supervised.sh")
        spectrogram_script = str(scripts_dir / "submit_amba_spectrogram.sh")
        script_path = supervised_script if training_type == "supervised" else spectrogram_script
        cmd.append(script_path)

        # Add required arguments
        cmd.extend([
            "--dataset", dataset_path,
            "--wandb-project", wandb_project,
            "--wandb-group", wandb_group,
            "--train-ratio", str(train_ratio),
            "--project-path", project_path,
            "--exp-dir", exp_dir,
        ])

        # Add script-specific arguments
        if training_type == "selfsupervised":
            cmd.extend([
                "--resume", str(resume).lower(),
                "--task", task,
            ])
            if pretrained_path:
                cmd.extend(["--pretrained-path", pretrained_path])
            # Virtualenv is passed via environment variable VENV_PATH to avoid arg mismatches
        elif training_type == "supervised" and resume:
            # Supervised script expects a value for --resume
            cmd.extend(["--resume", str(resume).lower()])

        # Add optional arguments
        if wandb_entity:
            cmd.extend(["--wandb-entity", wandb_entity])

        # Add exclude labels as separate --exclude-label arguments
        if exclude_labels:
            for label in exclude_labels:
                cmd.extend(["--exclude-label", label])

        if multiclass:
            cmd.extend(["--multiclass"])
            if num_classes and int(num_classes) > 0:
                cmd.extend(["--num-classes", str(num_classes)])

        if dry_run:
            cmd.extend(["--dry-run"])

        # Print the command that would be executed
        print(f"\nCommand for job {i}:")
        print(" ".join(cmd))
        print()

        # Execute the command unless in dry-run mode
        if not dry_run:
            try:
                # Ensure VENV_PATH is available to the sbatch job environment
                env = os.environ.copy()
                if venv_path:
                    env["VENV_PATH"] = venv_path
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if result.returncode != 0:
                    print(f"Error executing command: {result.stderr}")
                else:
                    job_id = result.stdout.strip()
                    print(f"Job {i} submitted successfully. Job ID: {job_id}")
                    prev_job_id = job_id
            except Exception as e:
                print(f"Error submitting job {i}: {e}")
                break
        else:
            print("Dry run - command not executed")

def submit_training_size_experiments(
    dataset_path: str,
    job_name: str,
    num_jobs: int,
    wandb_project: str,
    wandb_group: str,
    project_path: str,
    exp_dir: str,
    train_ratios: Optional[List[float]] = None,
    training_type: str = "pretrain_finetune",
    wandb_entity: Optional[str] = None,
    exclude_labels: Optional[List[str]] = None,
    pretrained_path: Optional[str] = None,
    resume: bool = True,
    task: str = "ft_cls",
    time_limit: str = "0-12:00:00",
    dry_run: bool = False,
    multiclass: bool = False,
    num_classes: Optional[int] = None,
    venv_path: Optional[str] = None,
    gpus: Optional[str] = None,
    sbatch_flags: Optional[List[str]] = None,
) -> None:
    """Submit linked jobs for multiple training ratios.

    Args:
        Same as submit_linked_jobs, plus:
        train_ratios: List of training ratios to experiment with
    """
    if train_ratios is None:
        train_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]

    for ratio in train_ratios:
        print(f"\nSubmitting {num_jobs} linked jobs for training ratio {ratio}")
        submit_linked_jobs(
            dataset_path=dataset_path,
            job_name=job_name,
            num_jobs=num_jobs,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            project_path=project_path,
            exp_dir=exp_dir,
            training_type=training_type,
            wandb_entity=wandb_entity,
            exclude_labels=exclude_labels,
            pretrained_path=pretrained_path,
            resume=resume,
            task=task,
            train_ratio=ratio,
            time_limit=time_limit,
            dry_run=dry_run,
            multiclass=multiclass,
            num_classes=num_classes,
            venv_path=venv_path,
            gpus=gpus,
            sbatch_flags=sbatch_flags,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to the dataset file")
    parser.add_argument("--job-name", required=True, help="Base name for the jobs")
    parser.add_argument("--num-jobs", type=int, required=True, help="Number of jobs to submit")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")
    parser.add_argument("--wandb-group", required=True, help="W&B group name")
    parser.add_argument("--project-path", required=True, help="Path to the project directory")
    parser.add_argument("--exp-dir", required=True, help="Directory for experiment outputs")
    parser.add_argument("--wandb-entity", help="W&B entity (team) name")
    parser.add_argument("--exclude-labels", nargs="+", help="Labels to exclude from training")
    parser.add_argument("--pretrained-path", help="Path to pretrained model")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--task", default="ft_cls", help="Self-supervised/finetune task",
                        choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
    parser.add_argument("--time-limit", default="0-12:00:00", help="Time limit for each job")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--multiclass", action="store_true", help="Enable multiclass using all available classes")
    parser.add_argument("--num-classes", type=int, dest="num_classes", help="Override number of classes when using --multiclass")
    parser.add_argument("--venv-path", help="Path to virtualenv to use (forwarded to pretrain/finetune script)")
    parser.add_argument("--gpus", help="Value for sbatch --gpus=... (e.g., 'h100:1' or '1')")
    parser.add_argument("--sbatch", action="append", dest="sbatch_flags", help="Extra sbatch flags; can be repeated. Each entry may include multiple tokens, e.g., '--partition=gpu --qos=high'")

    # Mode selection
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                       help="'single' for one linked job, 'multi' for training size experiments")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training ratio for single mode (ignored in multi mode)")
    parser.add_argument("--train-ratios", type=float, nargs="+",
                       help="List of training ratios for multi mode (default: [0.1, 0.2, 0.4, 0.6, 0.8])")
    parser.add_argument("--training-type", choices=["selfsupervised", "supervised"], default="selfsupervised",
                        help="Training mode: selfsupervised (spectrogram pipeline) or supervised")

    args = parser.parse_args()

    # Create output directories if they don't exist
    Path("out").mkdir(exist_ok=True)
    Path("err").mkdir(exist_ok=True)

    if args.mode == "single":
        submit_linked_jobs(
            dataset_path=args.dataset_path,
            job_name=args.job_name,
            num_jobs=args.num_jobs,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            project_path=args.project_path,
            exp_dir=args.exp_dir,
            training_type=args.training_type,
            wandb_entity=args.wandb_entity,
            exclude_labels=args.exclude_labels,
            pretrained_path=args.pretrained_path,
            resume=args.resume,
            task=args.task,
            train_ratio=args.train_ratio,
            time_limit=args.time_limit,
            dry_run=args.dry_run,
            multiclass=bool(args.multiclass),
            num_classes=args.num_classes,
            venv_path=args.venv_path,
            gpus=args.gpus,
            sbatch_flags=args.sbatch_flags,
        )
    else:
        submit_training_size_experiments(
            dataset_path=args.dataset_path,
            job_name=args.job_name,
            num_jobs=args.num_jobs,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            project_path=args.project_path,
            exp_dir=args.exp_dir,
            train_ratios=args.train_ratios,
            training_type=args.training_type,
            wandb_entity=args.wandb_entity,
            exclude_labels=args.exclude_labels,
            pretrained_path=args.pretrained_path,
            resume=args.resume,
            task=args.task,
            time_limit=args.time_limit,
            dry_run=args.dry_run,
            multiclass=bool(args.multiclass),
            num_classes=args.num_classes,
            venv_path=args.venv_path,
            gpus=args.gpus,
            sbatch_flags=args.sbatch_flags,
        )

if __name__ == "__main__":
    main() 