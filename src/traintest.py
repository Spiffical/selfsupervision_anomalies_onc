#Adopted from traintest_mask by Yuan Gong, modified for ssamba

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
from utilities.metrics.training_metrics import MetricsTracker, AverageMeterSet
from utilities.metrics.validation_metrics import (
    ValidationMetricsCollector, validate_ensemble, validate_wa, validate
)
from utilities.metrics.hydrophone_metrics import (
    calculate_hydrophone_metrics, print_hydrophone_metrics, 
    extract_hydrophone, calculate_binary_metrics
)
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from utilities.wandb_utils import log_training_metrics

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize metrics tracking
    metrics_tracker = MetricsTracker(args.exp_dir, args, use_wandb=args.use_wandb)
    train_meters = AverageMeterSet()
    val_collector = ValidationMetricsCollector(task=args.task)
    
    # Initialize training state
    global_step = args.start_epoch * len(train_loader) if hasattr(args, 'start_epoch') else 0
    epoch = args.start_epoch if hasattr(args, 'start_epoch') else 1
    start_time = time.time()

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    # diff lr optimizer
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    
    # Set up optimizer with different learning rates for mlp head and base
    print('The mlp header uses {:d} x larger lr'.format(args.head_lr))
    optimizer = torch.optim.Adam(
        [
            {'params': base_params, 'lr': args.lr}, 
            {'params': mlp_params, 'lr': args.lr * args.head_lr}
        ], 
        weight_decay=5e-7, betas=(0.95, 0.999)
    )
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]

    print('Total mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # Set up learning rate scheduler
    if args.adaptschedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True
        )
        print('Using adaptive learning rate scheduler')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
            gamma=args.lrscheduler_decay
        )
        # Update scheduler state if resuming
        if hasattr(args, 'start_epoch') and not args.adaptschedule:
            for _ in range(args.start_epoch):
                scheduler.step()
    
    # Set up loss function
    loss_fn = nn.BCEWithLogitsLoss() if args.loss == 'BCE' else nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    print('Training setup:')
    print(f'Dataset: {args.dataset}')
    print(f'Main metric: {args.metrics}')
    print(f'Loss function: {loss_fn}')
    print(f'Learning rate scheduler: {scheduler}')
    print('Learning rate schedule: start at epoch {}, decay rate {} every {} epochs'.format(
        args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step
    ))

    print("Current progress: steps=%s, epochs=%s" % (global_step, epoch))
    print("Starting training...")
    
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("Current epoch=%s, steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Update timing metrics
            train_meters.update('data_time', time.time() - end_time)
            train_meters.update('per_sample_data_time', (time.time() - end_time) / B)
            dnn_start_time = time.time()

            # Warm-up learning rate
            if global_step <= 1000 and global_step % 50 == 0 and args.warmup:
                for group_id, param_group in enumerate(optimizer.param_groups):
                    warm_lr = (global_step / 1000) * lr_list[group_id]
                    param_group['lr'] = warm_lr
                    print('Warm-up learning rate is {:f}'.format(param_group['lr']))

            # Forward pass
            audio_output = audio_model(audio_input, args.task)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = loss_fn(audio_output, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_meters.update('loss', loss.item(), B)
            train_meters.update('batch_time', time.time() - end_time)
            train_meters.update('per_sample_time', (time.time() - end_time) / B)
            train_meters.update('per_sample_dnn_time', (time.time() - dnn_start_time) / B)

            # Print progress
            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Per Sample Total Time {:.5f}\t'
                      'Per Sample Data Time {:.5f}\t'
                      'Per Sample DNN Time {:.5f}\t'
                      'Train Loss {:.4f}\t'.format(
                       epoch, i, len(train_loader),
                       train_meters.get_value('per_sample_time'),
                       train_meters.get_value('per_sample_data_time'),
                       train_meters.get_value('per_sample_dnn_time'),
                       train_meters.get_value('loss')), flush=True)

                if np.isnan(train_meters.get_value('loss')):
                    print("Training diverged...")
                    # Save debug information
                    torch.save(audio_model.state_dict(), f"{args.exp_dir}/models/nan_audio_model.pth")
                    torch.save(optimizer.state_dict(), f"{args.exp_dir}/models/nan_optim_state.pth")
                    with open(args.exp_dir + '/audio_input.npy', 'wb') as f:
                        np.save(f, audio_input.cpu().detach().numpy())
                    np.savetxt(args.exp_dir + '/audio_output.csv', audio_output.cpu().detach().numpy(), delimiter=',')
                    np.savetxt(args.exp_dir + '/labels.csv', labels.cpu().detach().numpy(), delimiter=',')
                    print('Debug information saved.')
                    return

            end_time = time.time()
            global_step += 1

        # Run validation at the end of each epoch
        print('Starting validation...')
        val_collector.reset()
        
        with torch.no_grad():
            for val_batch, (val_input, val_labels) in enumerate(test_loader):
                val_input = val_input.to(device)
                val_labels = val_labels.to(device)
                sources = None
                if hasattr(test_loader.dataset, 'sources'):
                    sources = test_loader.dataset.sources[test_loader.dataset.indices[val_batch*test_loader.batch_size:(val_batch+1)*test_loader.batch_size]]
                
                # Get model output
                val_output = audio_model(val_input, args.task)
                
                # Calculate loss
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    val_loss = loss_fn(val_output, torch.argmax(val_labels.long(), axis=1))
                else:
                    val_loss = loss_fn(val_output, val_labels)
                
                val_collector.update((val_output, val_loss), val_labels, sources)
        
        # Compute validation metrics
        val_metrics = val_collector.compute_metrics()
        val_collector.log_metrics(val_metrics, epoch=epoch, prefix="ft_", use_wandb=args.use_wandb)
        
        # Calculate ensemble metrics
        cum_metrics = validate_ensemble(metrics_tracker, val_metrics['predictions'], val_metrics['targets'], epoch)
        
        # Save results
        result_dict = {
            'epoch': epoch,
            'accuracy': val_metrics['acc'],
            'auc': val_metrics['auc'],
            'global_precision': val_metrics['global_precision'],
            'global_recall': val_metrics['global_recall'],
            'global_f2': val_metrics['global_f2'],
            'train_loss': train_meters.get_value('loss'),
            'valid_loss': val_metrics['loss'],
            'cum_acc': cum_metrics['acc'],
            'cum_auc': cum_metrics['auc'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Add per-hydrophone metrics
        if val_metrics['hydrophone_metrics']:
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

        # Log metrics to wandb at the end of each epoch
        if args.use_wandb:
            metrics_dict = {
                "ft_epoch": epoch,
                "ft_train_loss": train_meters.get_value('loss'),
                "ft_val_loss": val_metrics['loss'],
                "ft_val_accuracy": val_metrics['acc'],
                "ft_val_auc": val_metrics['auc'],
                "ft_val_precision": val_metrics['global_precision'],
                "ft_val_recall": val_metrics['global_recall'],
                "ft_val_f2": val_metrics['global_f2'],
                "ft_cum_accuracy": cum_metrics['acc'],
                "ft_cum_auc": cum_metrics['auc'],
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # Add per-hydrophone metrics with proper naming for WandB visualization
            if val_metrics['hydrophone_metrics']:
                for hydrophone, metrics in val_metrics['hydrophone_metrics'].items():
                    for metric_name, value in metrics.items():
                        # Format metric names for better visualization in WandB
                        if metric_name in ['accuracy', 'precision', 'recall', 'f2']:
                            metrics_dict[f"hydrophone_metrics/{hydrophone}/val_{metric_name}"] = value
                        elif metric_name == 'count':
                            metrics_dict[f"hydrophone_metrics/{hydrophone}/sample_count"] = value
            
            log_training_metrics(metrics_dict)

        # Save model if performance improved
        if metrics_tracker.should_save_best(val_metrics[args.main_metric], metric_name=args.main_metric):
            metrics_tracker.save_model(
                audio_model, optimizer, val_metrics[args.main_metric],
                metric_name=args.main_metric, is_best=True
            )

        # Save periodic checkpoint with all states
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': audio_model.module.state_dict() if isinstance(audio_model, nn.DataParallel) else audio_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metrics': metrics_tracker.best_metrics,
            # Save only necessary args as a dict
            'args': {
                'task': args.task,
                'lr': args.lr,
                'head_lr': args.head_lr,
                'n_epochs': args.n_epochs,
                'adaptschedule': args.adaptschedule,
                'main_metric': args.main_metric,
                'loss': args.loss
            }
        }
        torch.save(checkpoint, f"{args.exp_dir}/models/checkpoint.{epoch}.pth")

        # Save individual states for backward compatibility
        model_state = audio_model.module.state_dict() if isinstance(audio_model, nn.DataParallel) else audio_model.state_dict()
        torch.save(model_state, f"{args.exp_dir}/models/audio_model.{epoch}.pth")
        torch.save(optimizer.state_dict(), f"{args.exp_dir}/models/optim_state.{epoch}.pth")

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

    # Run weighted average validation if requested
    if args.wa:
        print('Running weighted average validation...')
        stats = validate_wa(audio_model, test_loader, args, args.wa_start, args.wa_end)
        print('Weighted average results:')
        print(f"mAP: {stats['mAP']:.6f}")
        print(f"AUC: {stats['auc']:.6f}")
        print(f"Accuracy: {stats['acc']:.6f}")
        np.savetxt(args.exp_dir + '/wa_result.csv', [stats['mAP'], stats['auc'], stats['acc']])