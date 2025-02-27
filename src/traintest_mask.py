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
    
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    
    # Define functions to create optimizer and scheduler
    def create_optimizer(model):
        audio_trainables = [p for p in model.parameters() if p.requires_grad]
        print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
        return torch.optim.AdamW(audio_trainables, args.lr, weight_decay=5e-8, betas=(0.95, 0.999))
    
    def create_scheduler(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    
    # Check if we need to resume from a checkpoint
    if hasattr(args, 'resume') and args.resume:
        # Find the latest checkpoint for the current task
        last_epoch, checkpoint_path = find_latest_checkpoint(args.exp_dir, task=args.task)
        
        if checkpoint_path and os.path.isfile(checkpoint_path):
            # Load checkpoint
            checkpoint = load_checkpoint(checkpoint_path, device)
            
            # Setup model, optimizer and scheduler from checkpoint
            audio_model, optimizer, scheduler, start_epoch = setup_model_from_checkpoint(
                checkpoint, audio_model, create_optimizer, create_scheduler
            )
            
            # Restore best metrics if available
            if 'best_metrics' in checkpoint and checkpoint['best_metrics']:
                metrics_tracker.best_metrics = checkpoint['best_metrics']
                print(f"Restored best metrics: {metrics_tracker.best_metrics}")
            
            # Set initial training state
            global_step = start_epoch * args.epoch_iter
            epoch = start_epoch
            print(f"Resuming training from epoch {epoch}, global step {global_step}")
        else:
            # No checkpoint found, start from scratch
            print("No checkpoint found. Starting from scratch.")
            optimizer = create_optimizer(audio_model)
            scheduler = create_scheduler(optimizer)
            global_step = 0
            epoch = 1
    else:
        # Start from scratch
        optimizer = create_optimizer(audio_model)
        scheduler = create_scheduler(optimizer)
        global_step = 0
        epoch = 1
    
    start_time = time.time()
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()

    # training until break
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        for i, batch_data in enumerate(train_loader):
            # Handle different return formats from different dataset classes
            if len(batch_data) == 3:  # Dataset returns (input, labels, sources)
                audio_input, _, sources = batch_data
            elif len(batch_data) == 2:  # Dataset returns (input, labels)
                audio_input, _ = batch_data
                sources = None
            else:
                raise ValueError(f"Unexpected batch data format with {len(batch_data)} elements")
                
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)

            # Update timing metrics
            train_meters.update('data_time', time.time() - end_time)
            train_meters.update('per_sample_data_time', (time.time() - end_time) / B)
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # Forward pass
            cluster = (args.num_mel_bins != args.fshape)
            if args.task == 'pretrain_mpc':
                acc, loss = audio_model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                acc, loss = acc.mean(), loss.mean()
            elif args.task == 'pretrain_mpg':
                loss = audio_model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                loss = loss.mean()
                acc = loss  # For MPG, we track MSE as accuracy
            elif args.task == 'pretrain_joint':
                acc, loss1 = audio_model(audio_input, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=cluster)
                acc, loss1 = acc.mean(), loss1.mean()
                loss2 = audio_model(audio_input, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=cluster)
                loss2 = loss2.mean()
                loss = loss1 + 10 * loss2

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_meters.update('acc', acc.detach().cpu().item())
            train_meters.update('nce', loss.detach().cpu().item())
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
                      'Per Sample Total Time {3:.5f}\t'
                      'Per Sample Data Time {4:.5f}\t'
                      'Per Sample DNN Time {5:.5f}\t'
                      'Train Loss {6:.4f}\t'.format(
                       epoch, i, len(train_loader),
                       train_meters.get_value('per_sample_time'),
                       train_meters.get_value('per_sample_data_time'),
                       train_meters.get_value('per_sample_dnn_time'),
                       train_meters.get_value('loss')), flush=True)
                
                if np.isnan(train_meters.get_value('loss')):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

            # Validation and checkpointing
            if global_step % args.epoch_iter == 0:
                print('---------------- step '+ str(global_step) +' evaluation ----------------')
                equ_epoch = epoch
                
                # Run validation
                audio_model.eval()
                val_collector.reset()
                
                print("\n[DEBUG] Starting validation...")
                
                # Check if dataset returns sources directly or has sources attribute
                # We'll let the validation loop handle this instead of creating dummy sources
                has_sources = hasattr(test_loader.dataset, 'sources') or hasattr(test_loader.dataset, 'sample_info')
                if not has_sources:
                    print("[DEBUG] Dataset doesn't have sources attribute or sample_info. Hydrophone metrics may not be available.")
                    print("[DEBUG] Make sure your dataset's __getitem__ method returns sources or has a sources attribute.")
                
                with torch.no_grad():
                    for val_batch, batch_data in enumerate(test_loader):
                        # Handle different return formats from different dataset classes
                        if len(batch_data) == 3:  # Dataset returns (input, labels, sources)
                            val_input, _, sources = batch_data
                            val_input = val_input.to(device)
                            if val_batch == 0:  # Only print for first batch to avoid too much output
                                print(f"[DEBUG] Batch {val_batch}: Found sources with length {len(sources)}")
                                print(f"[DEBUG] First few sources: {sources[:5]}")
                        elif len(batch_data) == 2:  # Dataset returns (input, labels)
                            val_input, _ = batch_data
                            val_input = val_input.to(device)
                            sources = None
                            if val_batch == 0:
                                print(f"[DEBUG] Batch {val_batch}: No sources found in batch data")
                                
                                # Try to get sources from dataset if available
                                if hasattr(test_loader.dataset, 'sample_info'):
                                    # For ONCSpectrogramDataset
                                    batch_indices = list(range(val_batch*test_loader.batch_size, 
                                                              min((val_batch+1)*test_loader.batch_size, 
                                                                  len(test_loader.dataset))))
                                    sources = [test_loader.dataset.sample_info[i]['source'] for i in batch_indices]
                                    print(f"[DEBUG] Retrieved sources from ONCSpectrogramDataset sample_info: {len(sources)}")
                                elif hasattr(test_loader.dataset, 'sources'):
                                    # For other dataset types
                                    batch_indices = test_loader.dataset.indices[val_batch*test_loader.batch_size:(val_batch+1)*test_loader.batch_size] if hasattr(test_loader.dataset, 'indices') else list(range(val_batch*test_loader.batch_size, min((val_batch+1)*test_loader.batch_size, len(test_loader.dataset))))
                                    sources = [test_loader.dataset.sources[i] for i in batch_indices]
                                    print(f"[DEBUG] Retrieved sources from dataset.sources: {len(sources)}")
                        
                        # Get model output based on task
                        if args.task == 'pretrain_mpc':
                            output = audio_model(val_input, args.task, mask_patch=400, cluster=cluster)
                        elif args.task == 'pretrain_mpg':
                            output = audio_model(val_input, args.task, mask_patch=400, cluster=cluster)
                        elif args.task == 'pretrain_joint':
                            mpc_output = audio_model(val_input, 'pretrain_mpc', mask_patch=400, cluster=cluster)
                            mpg_output = audio_model(val_input, 'pretrain_mpg', mask_patch=400, cluster=cluster)
                            output = (mpc_output, mpg_output)
                        
                        val_collector.update(output, val_input, sources)
                
                # Compute and log validation metrics
                print("[DEBUG] Computing validation metrics...")
                val_metrics = val_collector.compute_metrics()
                print("[DEBUG] Logging validation metrics...")
                
                val_collector.log_metrics(val_metrics, epoch=equ_epoch, prefix="pt_", use_wandb=args.use_wandb)
                
                # Print training metrics
                print("masked acc train: {:.6f}".format(train_meters.get_value('acc')))
                print("nce loss train: {:.6f}".format(train_meters.get_value('nce')))
                
                # Save results
                result = [
                    train_meters.get_value('acc'),
                    train_meters.get_value('nce'),
                    val_metrics['acc'],
                    val_metrics['nce'],
                    optimizer.param_groups[0]['lr']
                ]
                # Include task in the result filename
                task_prefix = args.task.replace('_', '-')
                np.savetxt(f"{args.exp_dir}/{task_prefix}_result.csv", result, delimiter=',')

                # Log metrics to wandb at the end of each epoch
                if args.use_wandb:
                    # Create metrics dictionary with training and validation metrics
                    metrics_dict = {
                        "pt_epoch": equ_epoch,
                        "pt_train_loss": train_meters.get_value('nce'),
                        "pt_train_accuracy": train_meters.get_value('acc'),
                        "pt_val_loss": val_metrics['nce'],
                        "pt_val_accuracy": val_metrics['acc'],
                        "pt_learning_rate": optimizer.param_groups[0]['lr']
                    }
                    
                    # Include hydrophone metrics if available
                    print("\n[DEBUG] Checking for hydrophone_metrics in val_metrics...")
                    if 'hydrophone_metrics' in val_metrics:
                        print(f"[DEBUG] hydrophone_metrics found in val_metrics: {bool(val_metrics['hydrophone_metrics'])}")
                        if val_metrics['hydrophone_metrics']:
                            print(f"[DEBUG] Number of hydrophones: {len(val_metrics['hydrophone_metrics'])}")
                            print(f"[DEBUG] Hydrophones: {list(val_metrics['hydrophone_metrics'].keys())}")
                            for hydrophone, metrics in val_metrics['hydrophone_metrics'].items():
                                print(f"[DEBUG] {hydrophone} metrics: {metrics}")
                            metrics_dict["hydrophone_metrics"] = val_metrics['hydrophone_metrics']
                        else:
                            print("[DEBUG] hydrophone_metrics is empty")
                    else:
                        print("[DEBUG] No hydrophone_metrics key in val_metrics")
                        print(f"[DEBUG] val_metrics keys: {val_metrics.keys()}")
                    
                    # Log all metrics
                    print("[DEBUG] Calling metrics_tracker.log_training_metrics...")
                    metrics_tracker.log_training_metrics(metrics_dict)
                    print("[DEBUG] Finished calling metrics_tracker.log_training_metrics")

                # Save model if validation accuracy improved
                if metrics_tracker.should_save_best(val_metrics['acc']):
                    # Use checkpoint utility to save best checkpoint
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
                audio_model.train()
                print('---------------- evaluation finished ----------------')
                
                # Increment epoch after validation
                epoch += 1

        # We've already incremented the epoch after validation, so we don't need to do it here
        # epoch += 1

    # Finish wandb run if enabled and initialized in this function
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
