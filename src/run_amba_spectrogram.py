# Adopted from run_amba.py, modified for pre-computed spectrogram dataset
import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
from ssamba.utilities import *
from ssamba.utilities.training_utils import create_model
from ssamba.utilities.metrics.validation_metrics import validate
from ssamba.utilities.stats import calculate_dataset_stats
import numpy as np
from ssamba.traintest import train
from ssamba.traintest_mask import trainmask
import datetime
from ssamba.utilities.wandb_utils import init_wandb, finish_run, log_training_metrics
from ssamba.dataset import ONCSpectrogramDataset, get_onc_spectrogram_data
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data h5 file")
parser.add_argument("--data-eval", type=str, default=None, help="optional separate evaluation data h5 file")
parser.add_argument("--n_class", type=int, default=None, help="number of classes")

# Dataset split parameters
parser.add_argument("--train_ratio", type=float, default=0.8, help="ratio of data to use for training")
parser.add_argument("--val_ratio", type=float, default=0.1, help="ratio of data to use for validation")
parser.add_argument("--split_seed", type=int, default=42, help="random seed for dataset splitting")
parser.add_argument("--exclude_labels", nargs="+", type=str, default=None, help="list of labels to exclude from training")

parser.add_argument("--dataset", type=str, default="custom", help="dataset name")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=512, help="number of input frequency bins")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--head_lr', type=int, default=10, help='multiplier for learning rate of mlp head during finetuning')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument("--loss", type=str, default="BCE", help="loss function to use", choices=["BCE", "CE"])
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

# Patch and model parameters
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='model size', type=str, default='base')
parser.add_argument("--patch_size", type=int, default=16, help="patch size for the vision mamba model")
parser.add_argument("--embed_dim", type=int, default=768, help="embedding dimension")
parser.add_argument("--depth", type=int, default=24, help="number of transformer layers")

# Model architecture parameters
parser.add_argument('--rms_norm', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--residual_in_fp32', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--fused_add_norm', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--if_rope', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--if_rope_residual', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--bimamba_type', type=str, default='v2')
parser.add_argument('--drop_path_rate', type=float, default=0.1)
parser.add_argument('--stride', type=int, default=16)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--drop_rate', type=float, default=0.0)
parser.add_argument('--norm_epsilon', type=float, default=1e-5)
parser.add_argument('--if_bidirectional', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--final_pool_type', type=str, default='none')
parser.add_argument('--if_abs_pos_embed', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--if_bimamba', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--if_cls_token', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--if_devide_out', type=str, choices=['true', 'false'], default='true')
parser.add_argument('--use_double_cls_token', type=str, choices=['true', 'false'], default='false')
parser.add_argument('--use_middle_cls_token', type=str, choices=['true', 'false'], default='false')

# Training task
parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", 
                    choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
parser.add_argument("--mask_patch", type=int, default=300, help="number of patches to mask")
parser.add_argument("--epoch_iter", type=float, default=0.5, 
                    help="fraction of training set to process before saving (e.g., 0.25 = 25% of dataset)")
parser.add_argument("--main_metric", type=str, default='auc', help="metric to track for model selection",
                    choices=['auc', 'acc', 'f2', 'precision', 'recall'])

# Wandb logging
parser.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team) to use')
parser.add_argument('--wandb_group', type=str, default=None, help='WandB group name for organizing runs')
parser.add_argument('--wandb_project', type=str, default='amba_spectrogram', help='WandB project name')

# Resume training and model loading
parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint if available')
parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained model for finetuning')

args = parser.parse_args()

# Ensure experiment directory exists
os.makedirs(args.exp_dir, exist_ok=True)
os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)

# Update wandb project name based on task type
if args.use_wandb:
    if 'pretrain' in args.task:
        args.wandb_project = f"{args.wandb_project}_pretrain"
    else:
        args.wandb_project = f"{args.wandb_project}_finetune"

run_id_file = os.path.join(args.exp_dir, 'wandb_run_id.txt')

# Check if a previous run ID exists
if os.path.exists(run_id_file):
    with open(run_id_file, 'r') as f:
        run_id = f.read().strip()
    print(f"Resuming W&B run with ID: {run_id}")
else:
    run_id = None

# Initialize wandb using our centralized utility
if args.use_wandb:
    run = init_wandb(
        args,
        project_name=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        run_id=run_id
    )
    
    # Save the run ID for future resumption
    if not os.path.exists(run_id_file) and run is not None:
        with open(run_id_file, 'w') as f:
            f.write(run.id)

# Convert string arguments to boolean
args.rms_norm = args.rms_norm == 'true'
args.residual_in_fp32 = args.residual_in_fp32 == 'true'
args.fused_add_norm = args.fused_add_norm == 'true'
args.if_rope = args.if_rope == 'true'
args.if_rope_residual = args.if_rope_residual == 'true'
args.if_bidirectional = args.if_bidirectional == 'true'
args.if_abs_pos_embed = args.if_abs_pos_embed == 'true'
args.if_bimamba = args.if_bimamba == 'true'
args.if_cls_token = args.if_cls_token == 'true'
args.if_devide_out = args.if_devide_out == 'true'
args.use_double_cls_token = args.use_double_cls_token == 'true'
args.use_middle_cls_token = args.use_middle_cls_token == 'true'

# Verify split ratios sum to <= 1.0
test_ratio = 1.0 - args.train_ratio - args.val_ratio
if test_ratio < 0:
    raise ValueError(f"Train ratio ({args.train_ratio}) + val ratio ({args.val_ratio}) must sum to <= 1.0")

print(f"\nDataset split ratios:")
print(f"Train: {args.train_ratio:.1%}")
print(f"Val: {args.val_ratio:.1%}")
print(f"Test: {test_ratio:.1%}")
print(f"Using random seed: {args.split_seed}")

# Calculate dataset statistics if needed
if args.dataset_mean == "none" or args.dataset_std == "none":
    print("Calculating dataset statistics...")
    mean, std = calculate_dataset_stats(args.data_train)
    print(f"Calculated dataset mean: {mean:.6f}")
    print(f"Calculated dataset std: {std:.6f}")
    args.dataset_mean = mean
    args.dataset_std = std

# Audio configuration
audio_conf = {
    'num_mel_bins': args.num_mel_bins,
    'target_length': args.target_length,
    'freqm': args.freqm,
    'timem': args.timem,
    'mixup': args.mixup,
    'dataset': args.dataset,
    'mode': 'train',
    'mean': args.dataset_mean,
    'std': args.dataset_std,
    'noise': False
}

val_audio_conf = {
    'num_mel_bins': args.num_mel_bins,
    'target_length': args.target_length,
    'freqm': 0,
    'timem': 0,
    'mixup': 0,
    'dataset': args.dataset,
    'mode': 'evaluation',
    'mean': args.dataset_mean,
    'std': args.dataset_std,
    'noise': False
}

# Create data loaders using ONCSpectrogramDataset with splits
print('Creating train/val/test splits from ONC dataset')

# Get datasets using the helper function from onc_dataset.py
exclude_labels = args.exclude_labels if args.exclude_labels else None

print(f"Excluding labels: {exclude_labels}")

if exclude_labels:
    ssl_train_dataset, ssl_val_dataset, test_dataset, train_dataset, val_dataset, excluded_test_dataset = get_onc_spectrogram_data(
        data_path=args.data_train,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_length=args.target_length,
        num_mel_bins=args.num_mel_bins,
        freqm=args.freqm,
        timem=args.timem,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        mixup=args.mixup,
        ood=-1,  # No OOD filtering
        amount=1.0,
        subsample_test=True,
        exclude_labels=exclude_labels
    )
else:
    ssl_train_dataset, ssl_val_dataset, test_dataset, train_dataset, val_dataset = get_onc_spectrogram_data(
        data_path=args.data_train,
        seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_length=args.target_length,
        num_mel_bins=args.num_mel_bins,
        freqm=args.freqm,
        timem=args.timem,
        dataset_mean=args.dataset_mean,
        dataset_std=args.dataset_std,
        mixup=args.mixup,
        ood=-1,  # No OOD filtering
        amount=1.0,
        subsample_test=True,
        exclude_labels=exclude_labels
    )
    excluded_test_dataset = None

# Use the appropriate datasets based on the task
if 'pretrain' in args.task:
    # For pretraining, use the SSL datasets (normal samples only)
    train_loader = torch.utils.data.DataLoader(
        ssl_train_dataset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=False, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        ssl_val_dataset,
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=False
    )
    
    eval_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    if excluded_test_dataset is not None:
        excluded_eval_loader = torch.utils.data.DataLoader(
            excluded_test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        # Create combined test loader for full evaluation
        full_test_dataset = torch.utils.data.ConcatDataset([test_dataset, excluded_test_dataset])
        full_eval_loader = torch.utils.data.DataLoader(
            full_test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
else:
    # For fine-tuning, use the supervised datasets (balanced normal/anomalous)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=False, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=False
    )
    
    eval_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    if excluded_test_dataset is not None:
        excluded_eval_loader = torch.utils.data.DataLoader(
            excluded_test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        # Create combined test loader for full evaluation
        full_test_dataset = torch.utils.data.ConcatDataset([test_dataset, excluded_test_dataset])
        full_eval_loader = torch.utils.data.DataLoader(
            full_test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

print('Dataset splits:')
print(f'SSL Train (normal only): {len(ssl_train_dataset)} samples')
print(f'SSL Val (normal only): {len(ssl_val_dataset)} samples')
print(f'Supervised Train (balanced): {len(train_dataset)} samples')
print(f'Supervised Val (balanced): {len(val_dataset)} samples')
print(f'Test: {len(test_dataset)} samples')
if excluded_test_dataset is not None:
    print(f'Excluded Test: {len(excluded_test_dataset)} samples')
    print(f'Total Test (Combined): {len(test_dataset) + len(excluded_test_dataset)} samples')

# Create experiment directory if it doesn't exist
os.makedirs(args.exp_dir, exist_ok=True)
os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)

# Save arguments for future reference
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

# Calculate epoch_iter based on dataset size
steps_per_epoch = len(train_loader)  # number of batches per epoch
args.epoch_iter = int(steps_per_epoch * args.epoch_iter)
print(f"Saving model every {args.epoch_iter} steps ({args.epoch_iter/steps_per_epoch:.1%} of an epoch)")

# Start training
if 'pretrain' not in args.task:
    print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
    # Load pretrained model if path is provided
    pretrained_model = None
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f'Loading pretrained model from {args.pretrained_path}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_model = create_model(args)
        sd = torch.load(args.pretrained_path, map_location=device)
        pretrained_model.load_state_dict(sd, strict=False)
    train(pretrained_model, train_loader, val_loader, args)
else:
    print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
    trainmask(None, train_loader, val_loader, args)

# Evaluate on test set if provided
# if args.data_eval is not None:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     audio_model = create_model(args)
#     sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
#     audio_model.load_state_dict(sd, strict=False)

#     args.loss_fn = torch.nn.BCEWithLogitsLoss()
#     stats, _ = validate(audio_model, val_loader, args, 'valid_set')
#     val_acc = stats[0]['acc']
#     val_mAUC = np.mean([stat['auc'] for stat in stats])
#     print('---------------evaluate on the validation set---------------')
#     print("Accuracy: {:.6f}".format(val_acc))
#     print("AUC: {:.6f}".format(val_mAUC))

#     # Evaluate on original test set
#     test_metrics = evaluate(audio_model, eval_loader, device, args, "Original Test Set")
    
#     metrics_to_log = {
#         'val_accuracy': val_acc,
#         'val_mAUC': val_mAUC,
#         'test_auroc': test_metrics['auroc'],
#         'test_accuracy': test_metrics['accuracy'],
#         'test_optimal_threshold': test_metrics['optimal_threshold']
#     }
    
#     # If we have excluded data, evaluate that too
#     if excluded_test_dataset is not None:
#         # Evaluate excluded test set
#         excluded_metrics = evaluate(audio_model, excluded_eval_loader, device, args, "Excluded Test Set")
        
#         # Evaluate combined test set
#         full_metrics = evaluate(audio_model, full_eval_loader, device, args, "Combined Test Set")
        
#         # Add excluded and full metrics to logging
#         metrics_to_log.update({
#             'excluded_test_auroc': excluded_metrics['auroc'],
#             'excluded_test_accuracy': excluded_metrics['accuracy'],
#             'excluded_test_optimal_threshold': excluded_metrics['optimal_threshold'],
#             'full_test_auroc': full_metrics['auroc'],
#             'full_test_accuracy': full_metrics['accuracy'],
#             'full_test_optimal_threshold': full_metrics['optimal_threshold']
#         })
    
#     if args.use_wandb:
#         log_training_metrics(metrics_to_log, use_wandb=args.use_wandb)
    
#     # Save all metrics to CSV
#     metrics_list = [
#         val_acc, val_mAUC,
#         test_metrics['accuracy'], test_metrics['auroc']
#     ]
#     if excluded_test_dataset is not None:
#         metrics_list.extend([
#             excluded_metrics['accuracy'], excluded_metrics['auroc'],
#             full_metrics['accuracy'], full_metrics['auroc']
#         ])
#     np.savetxt(args.exp_dir + '/eval_result.csv', metrics_list)

# Finish wandb run if it was started
if args.use_wandb:
    finish_run()

def evaluate(model, test_loader, device, args, name=""):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_sources = []
    
    with torch.no_grad():
        for batch in test_loader:
            data, labels, sources = batch
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            predictions = torch.sigmoid(outputs)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sources.extend(sources)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auroc = roc_auc_score(all_labels, all_preds)
    
    # Get optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate accuracy at optimal threshold
    predictions_at_threshold = (all_preds >= optimal_threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions_at_threshold)
    
    # Print detailed results
    print(f"\n{name} Results:")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy at optimal threshold: {accuracy:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Count sample types
    normal_count = np.sum(all_labels == 0)
    anomalous_count = np.sum(all_labels == 1)
    
    print(f"\nDataset Composition:")
    print(f"Normal samples: {normal_count}")
    print(f"Anomalous samples: {anomalous_count}")
    print(f"Total samples: {len(all_labels)}")
    
    # Return metrics
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'predictions': all_preds,
        'labels': all_labels,
        'sources': all_sources
    } 