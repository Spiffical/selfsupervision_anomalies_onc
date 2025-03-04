#Adopted from traintest_mask by Yuan Gong, modified for ssamba

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
from utilities.metrics.training_metrics import MetricsTracker, AverageMeterSet
from utilities.metrics.validation_metrics import ValidationMetricsCollector
from utilities.metrics.hydrophone_metrics import (
    calculate_hydrophone_metrics, print_hydrophone_metrics, 
    extract_hydrophone, calculate_binary_metrics
)
from utilities.checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint, setup_model_from_checkpoint
from utilities.training_utils import (
    create_model, setup_training, training_loop, validation_loop
)
from utilities.wandb_utils import init_wandb, log_validation_metrics, finish_run
import time
import torch
from torch import nn
import numpy as np
import pickle

def trainmask(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Now running on : ' + str(device))

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
        
        # Wrap in DataParallel if not already wrapped
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
    else:
        # If model is provided, ensure it's on the right device
        audio_model = audio_model.to(device)
        
        # Wrap in DataParallel if not already wrapped
        if not isinstance(audio_model, nn.DataParallel):
            audio_model = nn.DataParallel(audio_model)
    
    # Set up model, optimizer, scheduler and get starting epoch
    audio_model, optimizer, scheduler, epoch = setup_training(audio_model, args)
    
    # Initialize training state
    global_step = epoch * args.epoch_iter
    start_time = time.time()
    
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
        
        # Run validation at the end of each epoch_iter steps
        if global_step % args.epoch_iter == 0:
            print('---------------- step '+ str(global_step) +' evaluation ----------------')
            equ_epoch = epoch
            
            # Validation loop
            print('Starting validation...')
            val_metrics = validation_loop(
                model=audio_model,
                val_loader=test_loader,
                val_collector=val_collector,
                args=args
            )
            
            # Log metrics
            val_collector.log_metrics(val_metrics, epoch=equ_epoch, prefix="pt_", use_wandb=args.use_wandb)
            
            # Print training metrics
            print("masked acc train: {:.6f}".format(train_metrics['acc']))
            print("nce loss train: {:.6f}".format(train_metrics['nce']))
            
            # Save results
            result = [
                train_metrics['acc'],
                train_metrics['nce'],
                val_metrics['acc'],
                val_metrics['nce'],
                optimizer.param_groups[0]['lr']
            ]
            # Include task in the result filename
            task_prefix = args.task.replace('_', '-')
            np.savetxt(f"{args.exp_dir}/{task_prefix}_result.csv", result, delimiter=',')

            # Log metrics to wandb
            if args.use_wandb:
                metrics_dict = {
                    "pt_epoch": equ_epoch,
                    "pt_train_loss": train_metrics['nce'],
                    "pt_train_accuracy": train_metrics['acc'],
                    "pt_val_loss": val_metrics['nce'],
                    "pt_val_accuracy": val_metrics['acc'],
                    "pt_learning_rate": optimizer.param_groups[0]['lr']
                }
                
                if val_metrics.get('hydrophone_metrics'):
                    metrics_dict["hydrophone_metrics"] = val_metrics['hydrophone_metrics']
                
                metrics_tracker.log_training_metrics(metrics_dict)

            # Save model if validation accuracy improved
            if metrics_tracker.should_save_best(val_metrics['acc']):
                save_checkpoint(
                    model=audio_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics_tracker=metrics_tracker,
                    args=args,
                    exp_dir=args.exp_dir,
                    epoch=equ_epoch,
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
                epoch=equ_epoch,
                global_step=global_step,
                val_metrics=val_metrics,
                is_best=False
            )

            # Update learning rate
            if args.task == 'pretrain_mpg':
                scheduler.step(-val_metrics['acc'])  # For MPG, lower MSE is better
            else:
                scheduler.step(val_metrics['acc'])

            print('# {:d}, step {:d}-{:d}, lr: {:e}'.format(
                equ_epoch, global_step-args.epoch_iter, global_step,
                optimizer.param_groups[0]['lr']))

            metrics_tracker.save_progress(epoch, global_step, equ_epoch)

            finish_time = time.time()
            print('# {:d}, step {:d}-{:d}, training time: {:.3f}'.format(
                equ_epoch, global_step-args.epoch_iter, global_step,
                finish_time-begin_time))
            begin_time = time.time()

            # Reset metrics
            train_meters.reset()
            print('---------------- evaluation finished ----------------')
            
            # Increment epoch after validation
            epoch += 1

    # Finish wandb run if enabled
    if args.use_wandb and hasattr(args, 'wandb_initialized') and args.wandb_initialized:
        finish_run()
        args.wandb_initialized = False

def validatemask(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    A_acc = []
    A_nce = []
    A_predictions = []
    A_targets = []
    A_sources = []

    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            # Handle different return formats from different dataset classes
            if len(batch_data) == 3:  # Dataset returns (input, labels, sources)
                audio_input, _, sources = batch_data
                audio_input = audio_input.to(device)
                
                # Process sources to extract hydrophone names
                processed_sources = []
                for source in sources:
                    if isinstance(source, bytes):
                        source = source.decode('utf-8')
                    
                    # Extract just the hydrophone name if not already processed
                    if '_' in source:
                        hydrophone_name = source.split('_')[0]
                        processed_sources.append(hydrophone_name)
                    else:
                        processed_sources.append(source)
                
                A_sources.extend(processed_sources)
                
            elif len(batch_data) == 2:  # Dataset returns (input, labels)
                audio_input, _ = batch_data
                audio_input = audio_input.to(device)

                # Get sources from the dataloader if available
                if hasattr(val_loader.dataset, 'sources'):
                    batch_indices = val_loader.dataset.indices[i*val_loader.batch_size:(i+1)*val_loader.batch_size] if hasattr(val_loader.dataset, 'indices') else list(range(i*val_loader.batch_size, min((i+1)*val_loader.batch_size, len(val_loader.dataset))))
                    batch_sources = [val_loader.dataset.sources[idx] for idx in batch_indices]
                    
                    # Process sources to extract hydrophone names
                    processed_sources = []
                    for source in batch_sources:
                        if isinstance(source, bytes):
                            source = source.decode('utf-8')
                        
                        # Extract just the hydrophone name if not already processed
                        if '_' in source:
                            hydrophone_name = source.split('_')[0]
                            processed_sources.append(hydrophone_name)
                        else:
                            processed_sources.append(source)
                    
                    A_sources.extend(processed_sources)

            # use cluster masking only when masking patches, not frames
            cluster = (args.num_mel_bins != args.fshape)
            # always use mask_patch=400 for evaluation, even the training mask patch number differs.
            if args.task == 'pretrain_mpc':
                acc, nce = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                A_acc.append(torch.mean(acc).cpu())
                A_nce.append(torch.mean(nce).cpu())
                # Store predictions and targets for hydrophone metrics
                predictions = torch.sigmoid(acc).cpu().detach()
                targets = torch.ones_like(predictions)  # In MPC, we're predicting masked tokens
                A_predictions.append(predictions)
                A_targets.append(targets)
            elif args.task == 'pretrain_mpg':
                mse = audio_model(audio_input, args.task, mask_patch=400, cluster=cluster)
                # this is dirty code to track mse loss, A_acc and A_nce now track mse, not the name suggests
                A_acc.append(torch.mean(mse).cpu())
                A_nce.append(torch.mean(mse).cpu())
            elif args.task == 'pretrain_joint':
                acc, _ = audio_model(audio_input, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                mse = audio_model(audio_input, 'pretrain_mpg', mask_patch=400, cluster=cluster)
                A_acc.append(torch.mean(acc).cpu())
                # A_nce then tracks the mse loss
                A_nce.append(torch.mean(mse).cpu())
                
                # Store predictions and targets for hydrophone metrics (using MPC part)
                predictions = torch.sigmoid(acc).cpu().detach()
                targets = torch.ones_like(predictions)  # In MPC, we're predicting masked tokens
                A_predictions.append(predictions)
                A_targets.append(targets)

        acc = np.mean(A_acc)
        nce = np.mean(A_nce)

        # Calculate hydrophone metrics if we have sources and predictions
        hydrophone_metrics = {}
        global_precision = global_recall = global_f2 = 0
        
        if A_sources and A_predictions and args.task == 'pretrain_mpc':
            predictions = torch.cat(A_predictions).numpy()
            targets = torch.cat(A_targets).numpy()
            global_precision, global_recall, global_f2, hydrophone_metrics = calculate_hydrophone_metrics(predictions, targets, A_sources)

            if args.use_wandb:
                # Create metrics dictionary
                metrics = {
                    "acc": acc,
                    "nce": nce,
                    "global_precision": global_precision,
                    "global_recall": global_recall,
                    "global_f2": global_f2,
                    "hydrophone_metrics": hydrophone_metrics
                }
                
                # Log validation metrics
                log_validation_metrics(metrics, args.task, epoch, prefix="pt_", use_wandb=args.use_wandb)

        # Print metrics if available
        if hydrophone_metrics:
            print("\nGlobal Metrics:")
            print(f"Precision: {global_precision:.4f}")
            print(f"Recall: {global_recall:.4f}")
            print(f"F2: {global_f2:.4f}")
            
            print("\nPer-Hydrophone Metrics:")
            for hydrophone, metrics in hydrophone_metrics.items():
                print(f"\n{hydrophone}:")
                print(f"  Samples: {metrics['count']}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F2: {metrics['f2']:.4f}")

    return acc, nce, global_precision, global_recall, global_f2, hydrophone_metrics
