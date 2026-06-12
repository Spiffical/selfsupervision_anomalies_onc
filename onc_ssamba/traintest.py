#Adopted from traintest_mask by Yuan Gong, modified for ssamba

import sys
import os
import datetime
import csv
# sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # Removed this line
from .utilities import * # Changed to relative import
from .utilities.metrics.training_metrics import MetricsTracker, AverageMeterSet # Changed to relative import
from .utilities.metrics.validation_metrics import ValidationMetricsCollector # Changed to relative import
from .utilities.checkpoint_utils import save_checkpoint # Changed to relative import
from .utilities.training_utils import ( # Changed to relative import
    create_model, setup_training, training_loop, validation_loop
)
from .utilities.wandb_utils import init_wandb, finish_run # Changed to relative import
import time
import torch
from torch import nn
import numpy as np


def _metric_improved(current_value, best_value, mode, min_delta):
    if best_value is None:
        return True
    if mode == "min":
        return current_value < best_value - min_delta
    return current_value > best_value + min_delta


def _load_metric_history(exp_dir, metric_name):
    result_path = os.path.join(exp_dir, "result.csv")
    if not os.path.exists(result_path):
        return []
    history = []
    try:
        with open(result_path, newline="") as f:
            for row in csv.DictReader(f):
                if metric_name not in row or row[metric_name] in ("", None):
                    continue
                history.append((int(float(row["epoch"])), float(row[metric_name])))
    except Exception as exc:
        print(f"Could not load early stopping history from {result_path}: {exc}")
        return []
    return history


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)
    debug = getattr(args, 'debug', False)
    save_every_epoch = getattr(args, 'save_every_epoch', False)

    # Initialize wandb if enabled and not already initialized
    if args.use_wandb and not hasattr(args, 'wandb_initialized'):
        init_wandb(args)
        args.wandb_initialized = True

    # Initialize metrics tracking
    metrics_tracker = MetricsTracker(args.exp_dir, args, use_wandb=args.use_wandb)
    train_meters = AverageMeterSet()
    multiclass = getattr(args, 'multiclass', False)
    num_classes = getattr(args, 'num_classes', 2)
    val_collector = ValidationMetricsCollector(
        task=args.task,
        multiclass=multiclass,
        num_classes=num_classes,
        debug=debug,
    )
    
    # Create model if not provided
    if audio_model is None:
        audio_model = create_model(args)
        if audio_model is None:
            raise RuntimeError("Failed to create model")
        
        # Move model to device before wrapping in DataParallel
        audio_model = audio_model.to(device)
        
        # Wrap in DataParallel if multiple GPUs available and not already wrapped
        if torch.cuda.device_count() > 1 and not isinstance(audio_model, nn.DataParallel):
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            audio_model = nn.DataParallel(audio_model)
    else:
        # If model is provided, ensure it's on the right device
        audio_model = audio_model.to(device)
        
        # Wrap in DataParallel if multiple GPUs available and not already wrapped
        if torch.cuda.device_count() > 1 and not isinstance(audio_model, nn.DataParallel):
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            audio_model = nn.DataParallel(audio_model)
    
    # Set up model, optimizer, scheduler and get starting epoch
    audio_model, optimizer, scheduler, epoch = setup_training(audio_model, args)

    # Set up loss function
    if hasattr(args, 'multiclass') and args.multiclass:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss() if args.loss == 'BCE' else nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    # Initialize training state
    global_step = epoch * args.epoch_iter
    start_time = time.time()
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 0) or 0)
    early_stopping_metric = getattr(args, "early_stopping_metric", None) or getattr(args, "main_metric", "auc")
    early_stopping_min_delta = float(getattr(args, "early_stopping_min_delta", 0.0) or 0.0)
    early_stopping_mode = getattr(args, "early_stopping_mode", None)
    if early_stopping_mode is None:
        early_stopping_mode = "min" if early_stopping_metric.endswith("loss") else "max"
    best_metric_value = None
    best_metric_epoch = None

    if early_stopping_patience > 0:
        metric_history = _load_metric_history(args.exp_dir, early_stopping_metric)
        for metric_epoch, metric_value in metric_history:
            if _metric_improved(metric_value, best_metric_value, early_stopping_mode, early_stopping_min_delta):
                best_metric_value = metric_value
                best_metric_epoch = metric_epoch
        if best_metric_value is not None:
            metrics_tracker.best_metrics[early_stopping_metric] = best_metric_value
            last_metric_epoch = metric_history[-1][0]
            epochs_without_improvement = last_metric_epoch - best_metric_epoch
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "Early stopping already satisfied from history: "
                    f"best {early_stopping_metric}={best_metric_value:.6f} at epoch {best_metric_epoch}; "
                    f"last epoch {last_metric_epoch}; patience {early_stopping_patience}."
                )
                return
        else:
            epochs_without_improvement = 0
    else:
        epochs_without_improvement = 0
    
    # Note: Step counting for wandb will start from 0 automatically for new runs
    
    if debug:
        print("Current progress: steps=%s, epochs=%s" % (global_step, epoch))
        print("Starting training...")
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        
        # Training loop
        global_step, train_metrics = training_loop(
            model=audio_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_tracker=metrics_tracker,
            train_meters=train_meters,
            args=args,
            global_step=global_step,
            epoch=epoch
        )
        
        # Validation loop
        if debug:
            print('Starting validation...')
        val_metrics = validation_loop(
            model=audio_model,
            val_loader=test_loader,
            val_collector=val_collector,
            args=args
        )
        
        # Log metrics
        val_collector.log_metrics(val_metrics, epoch=epoch, prefix="ft_", use_wandb=args.use_wandb)
        if early_stopping_patience > 0:
            current_metric = val_metrics.get(early_stopping_metric)
            if current_metric is None:
                print(f"Early stopping metric {early_stopping_metric!r} missing; disabling early stopping.")
                early_stopping_patience = 0
            else:
                current_metric = float(current_metric)
                if _metric_improved(
                    current_metric,
                    best_metric_value,
                    early_stopping_mode,
                    early_stopping_min_delta,
                ):
                    best_metric_value = current_metric
                    best_metric_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement = epoch - (best_metric_epoch or epoch)
        
        # Save results to CSV
        result_dict = {
            'epoch': epoch,
            'accuracy': val_metrics['acc'],
            'auc': val_metrics['auc'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f2': val_metrics['f2'],
            'train_loss': train_metrics['loss'],
            'valid_loss': val_metrics['loss'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Add per-hydrophone metrics if available
        if val_metrics.get('hydrophone_metrics'):
            for hydrophone, metrics in val_metrics['hydrophone_metrics'].items():
                for metric_name, value in metrics.items():
                    result_dict[f'{hydrophone}_{metric_name}'] = value
        
        # Save results to CSV
        if epoch == 1:
            with open(args.exp_dir + '/result.csv', 'w') as f:
                header = ','.join(result_dict.keys())
                f.write(header + '\n')
        with open(args.exp_dir + '/result.csv', 'a') as f:
            values = ','.join(map(str, result_dict.values()))
            f.write(values + '\n')

        # Log metrics to wandb
        if args.use_wandb:
            metrics_dict = {
                "ft_epoch": epoch,
                "ft_train_loss": train_metrics['loss'],
                "ft_val_loss": val_metrics['loss'],
                "ft_val_accuracy": val_metrics['acc'],
                "ft_val_auc": val_metrics['auc'],
                "ft_val_precision": val_metrics['precision'],
                "ft_val_recall": val_metrics['recall'],
                "ft_val_f2": val_metrics['f2'],
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            if val_metrics.get('hydrophone_metrics'):
                metrics_dict["hydrophone_metrics"] = val_metrics['hydrophone_metrics']
            
            metrics_tracker.log_training_metrics(metrics_dict)

        # Save model if performance improved
        if metrics_tracker.should_save_best(val_metrics[args.main_metric], metric_name=args.main_metric):
            save_checkpoint(
                model=audio_model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics_tracker=metrics_tracker,
                args=args,
                exp_dir=args.exp_dir,
                epoch=epoch,
                global_step=global_step,
                val_metrics=val_metrics,
                is_best=True
            )

        # Save periodic checkpoint if enabled
        if save_every_epoch:
            save_checkpoint(
                model=audio_model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics_tracker=metrics_tracker,
                args=args,
                exp_dir=args.exp_dir,
                epoch=epoch,
                global_step=global_step,
                val_metrics=val_metrics,
                is_best=False
            )

        # Update learning rate
        if args.adaptschedule:
            scheduler.step(val_metrics[args.main_metric])
        else:
            scheduler.step()

        metrics_tracker.save_progress(epoch, global_step, epoch)

        finish_time = time.time()
        metric_name = args.main_metric if hasattr(args, 'main_metric') else None
        metric_val = val_metrics.get(metric_name) if metric_name else None
        metric_str = f"{metric_name}: {metric_val:.4f}" if metric_val is not None else "metric: n/a"
        print(
            "Epoch {}/{} - train loss: {:.4f} | val loss: {:.4f} | {} ({:.2f}s)".format(
                epoch,
                args.n_epochs,
                train_metrics.get('loss', float('nan')),
                val_metrics.get('loss', float('nan')),
                metric_str,
                finish_time - begin_time,
            )
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping at epoch {}: best {}={:.6f} at epoch {}; "
                "no improvement greater than {} for {} epochs.".format(
                    epoch,
                    early_stopping_metric,
                    best_metric_value,
                    best_metric_epoch,
                    early_stopping_min_delta,
                    epochs_without_improvement,
                )
            )
            break

        # Reset metrics for next epoch
        train_meters.reset()
        
        epoch += 1

    # Finish wandb run if enabled
    if args.use_wandb and hasattr(args, 'wandb_initialized') and args.wandb_initialized:
        finish_run()
        args.wandb_initialized = False
