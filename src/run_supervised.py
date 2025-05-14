import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np
import datetime
from utilities.wandb_utils import init_wandb, finish_run, log_training_metrics
from onc_dataset import ONCSpectrogramDataset, get_onc_spectrogram_data
from models.supervised_model import SupervisedAMBAModel
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data h5 file")
parser.add_argument("--n_class", type=int, default=2, help="number of classes")

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
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading')
parser.add_argument("--n-epochs", type=int, default=200, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=3, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

# Model parameters
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='model size', type=str, default='base')
parser.add_argument("--embed_dim", type=int, default=768, help="embedding dimension")
parser.add_argument("--depth", type=int, default=24, help="number of transformer layers")
parser.add_argument("--in_chans", type=int, default=1, help="number of input channels")

# Wandb logging
parser.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team) to use')
parser.add_argument('--wandb_group', type=str, default=None, help='WandB group name for organizing runs')
parser.add_argument('--wandb_project', type=str, default='amba_spectrogram_supervised', help='WandB project name')

parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')

args = parser.parse_args()

# Ensure experiment directory exists
os.makedirs(args.exp_dir, exist_ok=True)
os.makedirs(os.path.join(args.exp_dir, 'models'), exist_ok=True)

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

# Save arguments for future reference
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

# Verify split ratios sum to <= 1.0
test_ratio = 1.0 - args.train_ratio - args.val_ratio
if test_ratio < 0:
    raise ValueError(f"Train ratio ({args.train_ratio}) + val ratio ({args.val_ratio}) must sum to <= 1.0")

print(f"\nDataset split ratios:")
print(f"Train: {args.train_ratio:.1%}")
print(f"Val: {args.val_ratio:.1%}")
print(f"Test: {test_ratio:.1%}")
print(f"Using random seed: {args.split_seed}")

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

# Create data loaders
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

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size * 2, 
    shuffle=False, 
    num_workers=args.num_workers, 
    pin_memory=True
)

if excluded_test_dataset is not None:
    excluded_test_loader = torch.utils.data.DataLoader(
        excluded_test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # Create combined test loader for full evaluation
    full_test_dataset = torch.utils.data.ConcatDataset([test_dataset, excluded_test_dataset])
    full_test_loader = torch.utils.data.DataLoader(
        full_test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

print('Dataset splits:')
print(f'Train: {len(train_dataset)} samples')
print(f'Validation: {len(val_dataset)} samples')
print(f'Test: {len(test_dataset)} samples')
if excluded_test_dataset is not None:
    print(f'Excluded Test: {len(excluded_test_dataset)} samples')
    print(f'Total Test (Combined): {len(test_dataset) + len(excluded_test_dataset)} samples')

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SupervisedAMBAModel(
    backbone_config={
        'img_size': args.target_length,
        'patch_size': args.fshape,  # Using frequency shape as patch size
        'stride': args.fstride,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_classes': 0,  # No classification head in backbone
        'if_cls_token': True,
        'channels': args.in_chans,
        'final_pool_type': 'none'
    }
).to(device)

# Debug: Print model architecture and parameter count
print("\n=== Model Architecture ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Debug: Print dataset statistics
print("\n=== Dataset Statistics ===")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Analyze label distribution
print("\nAnalyzing label distribution...")
# Get first label to determine shape
_, first_label, _ = train_dataset[0]
print(f"Label shape: {first_label.shape}")

if len(first_label.shape) > 0:  # Multi-label case
    train_label_dist = np.sum([label.numpy() for _, label, _ in train_dataset], axis=0)
    val_label_dist = np.sum([label.numpy() for _, label, _ in val_dataset], axis=0)
    print("\nLabel distribution in training set:")
    for i, count in enumerate(train_label_dist):
        print(f"Class {i}: {count} samples ({count/len(train_dataset)*100:.2f}%)")
    print("\nLabel distribution in validation set:")
    for i, count in enumerate(val_label_dist):
        print(f"Class {i}: {count} samples ({count/len(val_dataset)*100:.2f}%)")
else:  # Single-label case
    train_labels = [label.item() for _, label, _ in train_dataset]
    val_labels = [label.item() for _, label, _ in val_dataset]
    unique_labels = sorted(set(train_labels + val_labels))
    
    print("\nLabel distribution in training set:")
    for label in unique_labels:
        count = sum(1 for x in train_labels if x == label)
        print(f"Class {label}: {count} samples ({count/len(train_dataset)*100:.2f}%)")
    
    print("\nLabel distribution in validation set:")
    for label in unique_labels:
        count = sum(1 for x in val_labels if x == label)
        print(f"Class {label}: {count} samples ({count/len(val_dataset)*100:.2f}%)")

# Debug: Print input shape and range
sample_batch, sample_labels, _ = next(iter(train_loader))
print("\n=== Input Statistics ===")
print(f"Input shape: {sample_batch.shape}")
print(f"Label shape: {sample_labels.shape}")
print(f"Input range: [{sample_batch.min():.4f}, {sample_batch.max():.4f}]")
print(f"Input mean: {sample_batch.mean():.4f}")
print(f"Input std: {sample_batch.std():.4f}")

# Loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Add gradient clipping
max_grad_norm = 1.0

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True
)

# Initialize tracking variables
best_auroc = 0
best_epoch = 0
start_epoch = 0
patience = args.lr_patience * 2  # Early stopping patience
patience_counter = 0

# Debug: Print parameter initialization statistics
print("\n=== Parameter Initialization Statistics ===")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

# Resume from checkpoint if requested
if args.resume:
    checkpoint_path = os.path.join(args.exp_dir, 'models/latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint['best_auroc']
        best_epoch = checkpoint['best_epoch']
        patience_counter = checkpoint['patience_counter']
        print(f"Resuming from epoch {start_epoch} with best AUROC: {best_auroc:.4f}")
    else:
        print("No checkpoint found, starting from scratch")

print(f"Starting training for {args.n_epochs} epochs...")

for epoch in range(start_epoch, args.n_epochs):
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    for batch_idx, (data, labels, _) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        # Debug: Print intermediate activations for first batch
        if batch_idx == 0 and epoch == 0:
            print("\n=== First Batch Activation Statistics ===")
            with torch.no_grad():
                # Assuming outputs is the final layer
                print(f"Final layer output stats: mean={outputs.mean():.4f}, std={outputs.std():.4f}")
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
        # Debug: Print detailed batch information every n_print_steps
        if (batch_idx + 1) % args.n_print_steps == 0:
            batch_preds = torch.sigmoid(outputs).detach().cpu().numpy()
            print(f'\n=== Batch {batch_idx + 1} Debug Info ===')
            print(f'Loss: {loss.item():.4f}')
            print(f'Predictions distribution: min={batch_preds.min():.4f}, max={batch_preds.max():.4f}, mean={batch_preds.mean():.4f}')
            print(f'Labels distribution: {np.sum(labels.cpu().numpy(), axis=0)}')
            print(f'Sample predictions vs labels:')
            for i in range(min(5, len(batch_preds))):  # Show first 5 samples
                print(f'Sample {i}: pred={batch_preds[i]}, true={labels[i].cpu().numpy()}')
            
            # Debug: Print gradient statistics
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append((name, torch.norm(param.grad).item()))
            print("\nGradient norms:")
            for name, norm in grad_norms[:5]:  # Show first 5 gradients
                print(f'{name}: {norm:.4f}')
    
    # Calculate training metrics
    train_loss /= len(train_loader)
    train_preds = np.array(train_preds)
    train_labels = np.array(train_labels).astype(int)
    train_auroc = roc_auc_score(train_labels, train_preds)
    
    # Debug: Print detailed epoch statistics
    print(f"\n=== Epoch {epoch + 1} Detailed Statistics ===")
    print(f"Training predictions: min={train_preds.min():.4f}, max={train_preds.max():.4f}, mean={train_preds.mean():.4f}")
    print(f"Training labels distribution: {np.sum(train_labels, axis=0)}")
    print(f"Number of positive predictions (>0.5): {np.sum(train_preds > 0.5)}")
    print(f"Number of positive labels: {np.sum(train_labels)}")
    
    # Debug: Print confusion matrix-like statistics
    pred_binary = (train_preds > 0.5).astype(int)
    true_pos = np.sum((pred_binary == 1) & (train_labels == 1))
    false_pos = np.sum((pred_binary == 1) & (train_labels == 0))
    true_neg = np.sum((pred_binary == 0) & (train_labels == 0))
    false_neg = np.sum((pred_binary == 0) & (train_labels == 1))
    print("\nPrediction Statistics (threshold=0.5):")
    print(f"True Positives: {true_pos}")
    print(f"False Positives: {false_pos}")
    print(f"True Negatives: {true_neg}")
    print(f"False Negatives: {false_neg}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for data, labels, _ in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels).astype(int)
    val_auroc = roc_auc_score(val_labels, val_preds)
    
    # Update learning rate scheduler
    scheduler.step(val_auroc)
    
    # Log metrics
    metrics = {
        'train_loss': train_loss,
        'train_auroc': train_auroc,
        'val_loss': val_loss,
        'val_auroc': val_auroc,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    if args.use_wandb:
        log_training_metrics(metrics, use_wandb=True)
    
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")
    
    # Save latest checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auroc': best_auroc,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'val_metrics': metrics
    }
    torch.save(checkpoint, os.path.join(args.exp_dir, 'models/latest_checkpoint.pth'))
    print(f"Saved latest checkpoint for epoch {epoch + 1}")
    
    # Save best model
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        best_epoch = epoch
        patience_counter = 0
        if args.save_model:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auroc': best_auroc,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'val_metrics': metrics
            }
            torch.save(checkpoint, os.path.join(args.exp_dir, 'models/best_model.pth'))
            print(f"Saved new best model with validation AUROC: {val_auroc:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

print(f"\nTraining completed. Best validation AUROC: {best_auroc:.4f} at epoch {best_epoch + 1}")

# Final evaluation on test sets
def evaluate_dataset(model, loader, name=""):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for data, label, _ in loader:
            data = data.to(device)
            outputs = model(data)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    preds = np.array(preds)
    labels = np.array(labels).astype(int)
    auroc = roc_auc_score(labels, preds)
    
    # Get optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate accuracy at optimal threshold
    preds_binary = (preds >= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, preds_binary)
    
    print(f"\n{name} Results:")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy at optimal threshold: {accuracy:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'predictions': preds,
        'labels': labels
    }

print("\nStarting final evaluation...")

# Evaluate original test set
test_metrics = evaluate_dataset(model, test_loader, "Original Test Set")

# Log metrics
metrics_to_log = {
    'test_auroc': test_metrics['auroc'],
    'test_accuracy': test_metrics['accuracy'],
    'test_optimal_threshold': test_metrics['optimal_threshold']
}

# If we have excluded data, evaluate that too
if excluded_test_dataset is not None:
    # Evaluate excluded test set
    excluded_metrics = evaluate_dataset(model, excluded_test_loader, "Excluded Test Set")
    
    # Evaluate combined test set
    full_metrics = evaluate_dataset(model, full_test_loader, "Combined Test Set")
    
    # Add excluded and full metrics to logging
    metrics_to_log.update({
        'excluded_test_auroc': excluded_metrics['auroc'],
        'excluded_test_accuracy': excluded_metrics['accuracy'],
        'excluded_test_optimal_threshold': excluded_metrics['optimal_threshold'],
        'full_test_auroc': full_metrics['auroc'],
        'full_test_accuracy': full_metrics['accuracy'],
        'full_test_optimal_threshold': full_metrics['optimal_threshold']
    })

if args.use_wandb:
    log_training_metrics(metrics_to_log, use_wandb=True)
    finish_run()

print("\nExperiment completed successfully!") 