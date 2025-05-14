#Adopted from traintest_mask by Yuan Gong, modified for ssamba

import sys
import os
import datetime
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

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize wandb if enabled and not already initialized
    if args.use_wandb and not hasattr(args, 'wandb_initialized'):
        init_wandb(args)
        args.wandb_initialized = True

    # Initialize metrics tracking
    metrics_tracker = MetricsTracker(args.exp_dir, args, use_wandb=args.use_wandb)
    train_meters = AverageMeterSet()
    val_collector = ValidationMetricsCollector(task=args.task)
    
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
    loss_fn = nn.BCEWithLogitsLoss() if args.loss == 'BCE' else nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    # Initialize training state
    global_step = epoch * args.epoch_iter
    start_time = time.time()
    
    # Note: Step counting for wandb will start from 0 automatically for new runs
    
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
        print('Starting validation...')
        val_metrics = validation_loop(
            model=audio_model,
            val_loader=test_loader,
            val_collector=val_collector,
            args=args
        )
        
        # Log metrics
        val_collector.log_metrics(val_metrics, epoch=epoch, prefix="ft_", use_wandb=args.use_wandb)
        
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

        # Save periodic checkpoint
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
        print('Epoch {} completed in {:.3f} seconds'.format(epoch, finish_time - begin_time))

        # Reset metrics for next epoch
        train_meters.reset()
        
        epoch += 1

    # Finish wandb run if enabled
    if args.use_wandb and hasattr(args, 'wandb_initialized') and args.wandb_initialized:
        finish_run()
        args.wandb_initialized = False