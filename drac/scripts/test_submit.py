#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import shlex

def test_submit(
    dataset_path: str,
    wandb_project: str,
    wandb_group: str,
    project_path: str,
    exp_dir: str,
    wandb_entity: str = None,
    exclude_labels: list = None,
    pretrained_path: str = None,
    resume: bool = False,
    task: str = "ft_cls",
    train_ratio: float = 0.8,
):
    """Test submit_amba_spectrogram.sh without submitting to SLURM.
    
    This function directly calls submit_amba_spectrogram.sh to test argument handling.
    """
    # Build command with named arguments
    cmd = [
        "./submit_amba_spectrogram.sh",
        "--dataset", dataset_path,
        "--wandb-project", wandb_project,
        "--wandb-group", wandb_group,
        "--train-ratio", str(train_ratio),
        "--project-path", project_path,
        "--resume", str(resume).lower(),
        "--task", task,
        "--exp-dir", exp_dir,
    ]
    
    if wandb_entity:
        cmd.extend(["--wandb-entity", wandb_entity])
    
    # Add exclude labels as separate --exclude-label arguments
    if exclude_labels:
        for label in exclude_labels:
            cmd.extend(["--exclude-label", label])
    
    if pretrained_path:
        cmd.extend(["--pretrained-path", pretrained_path])

    cmd.extend(["--dry-run"])

    # Print the command that would be executed
    print("Command that would be executed:")
    print(" ".join(shlex.quote(arg) for arg in cmd))
    
    # Print the exclude labels for debugging
    if exclude_labels:
        print("\nExclude labels being passed:")
        print(exclude_labels)
    
    # Ask for confirmation
    response = input("\nDo you want to execute this command? [y/N] ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Execute the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\nCommand output:")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"\nExit code: {result.returncode}")
    except Exception as e:
        print(f"Error executing command: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test submit_amba_spectrogram.sh without SLURM submission")
    parser.add_argument("dataset_path", help="Path to the dataset file")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")
    parser.add_argument("--wandb-group", required=True, help="W&B group name")
    parser.add_argument("--project-path", required=True, help="Path to the project directory")
    parser.add_argument("--exp-dir", required=True, help="Directory for experiment outputs")
    parser.add_argument("--wandb-entity", help="W&B entity (team) name")
    parser.add_argument("--exclude-labels", nargs="+", help="Labels to exclude from training")
    parser.add_argument("--pretrained-path", help="Path to pretrained model")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training")
    parser.add_argument("--task", default="ft_cls", help="Training task type")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training")

    args = parser.parse_args()

    test_submit(
        dataset_path=args.dataset_path,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        project_path=args.project_path,
        exp_dir=args.exp_dir,
        wandb_entity=args.wandb_entity,
        exclude_labels=args.exclude_labels,
        pretrained_path=args.pretrained_path,
        resume=args.resume,
        task=args.task,
        train_ratio=args.train_ratio,
    )

if __name__ == "__main__":
    main() 