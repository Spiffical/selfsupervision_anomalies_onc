"""
Utilities for training setup, including optimizer creation and checkpoint handling.
"""

import os
import torch
from torch import nn
from .checkpoint_utils import load_checkpoint, find_latest_checkpoint, setup_model_from_checkpoint
from src.models import AMBAModel
import numpy as np

def create_model(args):
    """Create and initialize the AMBA model based on task type.
    
    Args:
        args: Training arguments containing model configuration
        
    Returns:
        model: Initialized model
    """
    # Vision Mamba configuration
    vision_mamba_config = {
        'img_size': (args.num_mel_bins, args.target_length),
        'patch_size': args.patch_size,
        'stride': args.stride,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'channels': args.channels,
        'num_classes': args.num_classes,
        'drop_rate': args.drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'norm_epsilon': args.norm_epsilon,
        'rms_norm': args.rms_norm,
        'residual_in_fp32': args.residual_in_fp32,
        'fused_add_norm': args.fused_add_norm,
        'if_rope': args.if_rope,
        'if_rope_residual': args.if_rope_residual,
        'bimamba_type': args.bimamba_type,
        'if_cls_token': args.if_cls_token,
        'if_devide_out': args.if_devide_out,
        'use_double_cls_token': args.use_double_cls_token,
        'use_middle_cls_token': args.use_middle_cls_token,
        'if_bidirectional': args.if_bidirectional,
        'final_pool_type': args.final_pool_type,
        'if_abs_pos_embed': args.if_abs_pos_embed,
        'if_bimamba': args.if_bimamba
    }
    
    # For binary classification tasks, ensure label_dim is 1
    label_dim = 1 if args.task == 'ft_cls' and args.n_class == 2 else args.n_class
    
    # For finetuning, try to find pretrained weights
    if 'pretrain' not in args.task:
        if hasattr(args, 'pretrained_mdl_path') and args.pretrained_mdl_path:
            pretrained_path = args.pretrained_mdl_path
        else:
            # Look for any pretrained checkpoint in models directory
            models_dir = os.path.join(args.exp_dir, 'models')
            print(f"Models directory: {models_dir}")
            if os.path.exists(models_dir):
                pretrain_checkpoints = [f for f in os.listdir(models_dir) if 'pretrain' in f and 'best_checkpoint.pth' in f]
                if pretrain_checkpoints:
                    # Use the first found pretrained checkpoint
                    pretrained_path = os.path.join(models_dir, pretrain_checkpoints[0])
                    print(f"Found pretrained checkpoint: {pretrain_checkpoints[0]}")
                else:
                    raise ValueError('No pretrained checkpoint found in models directory. Please run pretraining first or specify pretrained_mdl_path.')
            else:
                raise ValueError('Models directory not found. Please run pretraining first or specify pretrained_mdl_path.')

    if 'pretrain' in args.task:
        cluster = (args.num_mel_bins != args.fshape)
        if cluster:
            print('Using cluster masking (num_mel_bins != fshape)')
        else:
            print('Using frame masking (num_mel_bins == fshape)')
        
        model = AMBAModel(
            fshape=args.fshape, tshape=args.tshape,
            fstride=args.fshape, tstride=args.tshape,
            input_fdim=args.num_mel_bins,
            input_tdim=args.target_length,
            model_size=args.model_size,
            pretrain_stage=True,
            vision_mamba_config=vision_mamba_config
        )
    else:
        model = AMBAModel(
            label_dim=label_dim,
            fshape=args.fshape, tshape=args.tshape,
            fstride=args.fstride, tstride=args.tstride,
            input_fdim=args.num_mel_bins,
            input_tdim=args.target_length,
            model_size=args.model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path=pretrained_path,
            vision_mamba_config=vision_mamba_config
        )

    return model

def create_optimizer(model, args):
    """Create optimizer based on task type and model parameters.
    
    Args:
        model: The model to create optimizer for
        args: Training arguments containing lr, head_lr, etc.
        
    Returns:
        optimizer: The created optimizer
    """
    if 'pretrain' in args.task:
        # For pretraining, use single learning rate
        audio_trainables = [p for p in model.parameters() if p.requires_grad]
        print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
        return torch.optim.AdamW(audio_trainables, args.lr, weight_decay=5e-8, betas=(0.95, 0.999))
    else:
        # For finetuning, use different learning rates for mlp head and base
        mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
        
        # Get the actual model (handle both DataParallel and non-DataParallel cases)
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Filter parameters based on names
        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, actual_model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, actual_model.named_parameters()))
        
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]
        
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(
            sum(p.numel() for p in (mlp_params + base_params)) / 1e6))
        print('Total mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
        print('Total base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
        print('The mlp header uses {:d} x larger lr'.format(args.head_lr))
        
        return torch.optim.AdamW(
            [
                {'params': base_params, 'lr': args.lr}, 
                {'params': mlp_params, 'lr': args.lr * args.head_lr}
            ], 
            weight_decay=5e-7, betas=(0.95, 0.999)
        )

def create_scheduler(optimizer, args):
    """Create learning rate scheduler based on arguments.
    
    Args:
        optimizer: The optimizer to create scheduler for
        args: Training arguments containing scheduler settings
        
    Returns:
        scheduler: The created scheduler
    """
    if args.adaptschedule:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True
        )
    else:
        # Default values if not provided
        lrscheduler_start = getattr(args, 'lrscheduler_start', 20)
        lrscheduler_step = getattr(args, 'lrscheduler_step', 10)
        lrscheduler_decay = getattr(args, 'lrscheduler_decay', 0.5)
        
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            list(range(lrscheduler_start, 1000, lrscheduler_step)),
            gamma=lrscheduler_decay
        )

def setup_training(model, args):
    """Set up model, optimizer, scheduler and initial epoch for training.
    
    Args:
        model: The model to set up training for
        args: Training arguments
        
    Returns:
        tuple: (model, optimizer, scheduler, start_epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # First move model to device before DataParallel wrapping
    model = model.to(device)
    
    # Check if we have multiple GPUs and if model isn't already wrapped
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    # Define functions to create optimizer and scheduler
    def create_opt(m): 
        # Ensure we pass the actual model to optimizer, not DataParallel wrapper
        model_to_optimize = m.module if isinstance(m, nn.DataParallel) else m
        return create_optimizer(model_to_optimize, args)
    
    def create_sched(opt): return create_scheduler(opt, args)
    
    # Handle checkpoint loading based on task
    if hasattr(args, 'resume') and args.resume:
        if 'pretrain' not in args.task:
            # For finetuning, first try to load finetuning checkpoint
            print("Looking for existing finetuning checkpoint...")
            last_epoch, ft_checkpoint_path = find_latest_checkpoint(args.exp_dir, task='ft_cls')
            
            if ft_checkpoint_path and os.path.isfile(ft_checkpoint_path):
                print(f"Found finetuning checkpoint at epoch {last_epoch}")
                checkpoint = load_checkpoint(ft_checkpoint_path, device)
                model, optimizer, scheduler, start_epoch = setup_model_from_checkpoint(
                    checkpoint, model, create_opt, create_sched
                )
                print(f"Resuming finetuning from epoch {start_epoch}")
            else:
                # No finetuning checkpoint, try to load pretrained checkpoint
                print("No finetuning checkpoint found. Looking for pretrained checkpoint...")
                if hasattr(args, 'pretrained_mdl_path') and args.pretrained_mdl_path:
                    pt_checkpoint_path = args.pretrained_mdl_path
                else:
                    # Look for any pretrained checkpoint in models directory
                    models_dir = os.path.join(args.exp_dir, 'models')
                    if os.path.exists(models_dir):
                        pretrain_checkpoints = [f for f in os.listdir(models_dir) if 'pretrain' in f and 'best_checkpoint.pth' in f]
                        if pretrain_checkpoints:
                            # Use the first found pretrained checkpoint
                            pt_checkpoint_path = os.path.join(models_dir, pretrain_checkpoints[0])
                            print(f"Found pretrained checkpoint: {pretrain_checkpoints[0]}")
                        else:
                            pt_checkpoint_path = None
                    else:
                        pt_checkpoint_path = None
                
                if pt_checkpoint_path and os.path.isfile(pt_checkpoint_path):
                    print(f"Loading pretrained checkpoint from {pt_checkpoint_path}")
                    checkpoint = load_checkpoint(pt_checkpoint_path, device)
                    # For finetuning from pretrained, we only load the model state
                    # Handle both DataParallel and non-DataParallel state dicts
                    state_dict = checkpoint['model_state_dict']
                    if not isinstance(model, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
                        # Remove 'module.' prefix if model is not wrapped but state dict is
                        state_dict = {k[7:]: v for k, v in state_dict.items()}
                    elif isinstance(model, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
                        # Add 'module.' prefix if model is wrapped but state dict isn't
                        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                    
                    # Handle positional embedding resizing
                    pos_embed_key = 'module.v.pos_embed' if isinstance(model, nn.DataParallel) else 'v.pos_embed'
                    if pos_embed_key in state_dict:
                        pretrained_pos_embed = state_dict[pos_embed_key]
                        current_pos_embed = model.state_dict()[pos_embed_key]
                        
                        if pretrained_pos_embed.shape != current_pos_embed.shape:
                            print(f"Resizing positional embedding from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")
                            # Get the positional embeddings without the class token
                            pretrained_pos_embed_2d = pretrained_pos_embed[:, 1:, :]
                            current_pos_embed_2d = current_pos_embed[:, 1:, :]
                            
                            # Compute number of patches in each dimension
                            pretrained_num_patches = pretrained_pos_embed_2d.shape[1]
                            current_num_patches = current_pos_embed_2d.shape[1]
                            pretrained_size = int(np.sqrt(pretrained_num_patches))
                            current_size = int(np.sqrt(current_num_patches))
                            
                            # Reshape to 2D grid
                            pretrained_pos_embed_2d = pretrained_pos_embed_2d.reshape(1, pretrained_size, pretrained_size, -1).permute(0, 3, 1, 2)
                            
                            # Interpolate
                            pretrained_pos_embed_2d = torch.nn.functional.interpolate(
                                pretrained_pos_embed_2d,
                                size=(current_size, current_size),
                                mode='bicubic',
                                align_corners=False
                            )
                            
                            # Reshape back and combine with class token
                            pretrained_pos_embed_2d = pretrained_pos_embed_2d.permute(0, 2, 3, 1).reshape(1, current_num_patches, -1)
                            state_dict[pos_embed_key] = torch.cat([pretrained_pos_embed[:, :1, :], pretrained_pos_embed_2d], dim=1)
                    
                    model.load_state_dict(state_dict, strict=False)
                    optimizer = create_opt(model)
                    scheduler = create_sched(optimizer)
                    start_epoch = 1
                    print("Starting finetuning from pretrained checkpoint")
                else:
                    print("No pretrained checkpoint found. Starting from scratch.")
                    optimizer = create_opt(model)
                    scheduler = create_sched(optimizer)
                    start_epoch = 1
        else:
            # For pretraining, look for pretraining checkpoint
            print("Looking for existing pretraining checkpoint...")
            last_epoch, checkpoint_path = find_latest_checkpoint(args.exp_dir, task=args.task)
            
            if checkpoint_path and os.path.isfile(checkpoint_path):
                print(f"Found pretraining checkpoint at epoch {last_epoch}")
                checkpoint = load_checkpoint(checkpoint_path, device)
                model, optimizer, scheduler, start_epoch = setup_model_from_checkpoint(
                    checkpoint, model, create_opt, create_sched
                )
                print(f"Resuming pretraining from epoch {start_epoch}")
            else:
                print("No pretraining checkpoint found. Starting from scratch.")
                optimizer = create_opt(model)
                scheduler = create_sched(optimizer)
                start_epoch = 1
    else:
        # Not resuming, but check if we're finetuning from a pretrained model
        if 'pretrain' not in args.task and hasattr(args, 'pretrained_mdl_path') and args.pretrained_mdl_path:
            print(f"Loading pretrained model from {args.pretrained_mdl_path}")
            checkpoint = load_checkpoint(args.pretrained_mdl_path, device)
            # Handle both DataParallel and non-DataParallel state dicts
            state_dict = checkpoint['model_state_dict']
            if not isinstance(model, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            elif isinstance(model, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
            # Handle positional embedding resizing
            pos_embed_key = 'module.v.pos_embed' if isinstance(model, nn.DataParallel) else 'v.pos_embed'
            if pos_embed_key in state_dict:
                pretrained_pos_embed = state_dict[pos_embed_key]
                current_pos_embed = model.state_dict()[pos_embed_key]
                
                if pretrained_pos_embed.shape != current_pos_embed.shape:
                    print(f"Resizing positional embedding from {pretrained_pos_embed.shape} to {current_pos_embed.shape}")
                    # Get the positional embeddings without the class token
                    pretrained_pos_embed_2d = pretrained_pos_embed[:, 1:, :]
                    current_pos_embed_2d = current_pos_embed[:, 1:, :]
                    
                    # Compute number of patches in each dimension
                    pretrained_num_patches = pretrained_pos_embed_2d.shape[1]
                    current_num_patches = current_pos_embed_2d.shape[1]
                    pretrained_size = int(np.sqrt(pretrained_num_patches))
                    current_size = int(np.sqrt(current_num_patches))
                    
                    # Reshape to 2D grid
                    pretrained_pos_embed_2d = pretrained_pos_embed_2d.reshape(1, pretrained_size, pretrained_size, -1).permute(0, 3, 1, 2)
                    
                    # Interpolate
                    pretrained_pos_embed_2d = torch.nn.functional.interpolate(
                        pretrained_pos_embed_2d,
                        size=(current_size, current_size),
                        mode='bicubic',
                        align_corners=False
                    )
                    
                    # Reshape back and combine with class token
                    pretrained_pos_embed_2d = pretrained_pos_embed_2d.permute(0, 2, 3, 1).reshape(1, current_num_patches, -1)
                    state_dict[pos_embed_key] = torch.cat([pretrained_pos_embed[:, :1, :], pretrained_pos_embed_2d], dim=1)
            
            model.load_state_dict(state_dict, strict=False)
            optimizer = create_opt(model)
            scheduler = create_sched(optimizer)
            start_epoch = 1
        else:
            optimizer = create_opt(model)
            scheduler = create_sched(optimizer)
            start_epoch = 1
    
    return model, optimizer, scheduler, start_epoch

def training_loop(model, train_loader, optimizer, scheduler, metrics_tracker, train_meters, args, global_step, epoch):
    """Common training loop used by both pretraining and finetuning.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        metrics_tracker: Tracker for training metrics
        train_meters: AverageMeterSet for tracking metrics
        args: Training arguments
        global_step: Current global step
        epoch: Current epoch
        
    Returns:
        tuple: (global_step, train_metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    for i, batch_data in enumerate(train_loader):
        # Handle different return formats from different dataset classes
        if isinstance(batch_data, (list, tuple)):
            audio_input = batch_data[0]
            labels = batch_data[1] if len(batch_data) > 1 else None
            sources = batch_data[2] if len(batch_data) > 2 else None
        else:
            audio_input = batch_data
            labels = None
            sources = None
            
        audio_input = audio_input.to(device, non_blocking=True)
        if labels is not None and 'pretrain' not in args.task:
            labels = labels.to(device, non_blocking=True)
        
        B = audio_input.size(0)
        
        # Warmup learning rate
        if global_step <= 1000 and global_step % 50 == 0 and args.warmup:
            lr_list = [group['lr'] for group in optimizer.param_groups]
            for group_id, param_group in enumerate(optimizer.param_groups):
                warm_lr = (global_step / 1000) * lr_list[group_id]
                param_group['lr'] = warm_lr
                print('Warm-up learning rate is {:f}'.format(param_group['lr']))

        # Forward pass
        if 'pretrain' in args.task:
            cluster = (args.num_mel_bins != args.fshape)
            if args.task == 'pretrain_mpc':
                acc, loss = model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                acc, loss = acc.mean(), loss.mean()
            elif args.task == 'pretrain_mpg':
                loss = model(audio_input, args.task, mask_patch=args.mask_patch, cluster=cluster)
                loss = loss.mean()
                acc = loss  # For MPG, we track MSE as accuracy
            elif args.task == 'pretrain_joint':
                acc, loss1 = model(audio_input, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=cluster)
                acc, loss1 = acc.mean(), loss1.mean()
                loss2 = model(audio_input, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=cluster)
                loss2 = loss2.mean()
                loss = loss1 + 10 * loss2
        else:
            output = model(audio_input, args.task)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(output, torch.argmax(labels.long(), axis=1))
            else:
                # For binary classification (BCE loss), squeeze the output to match target shape
                if 'pretrain' not in args.task:
                    output = output.squeeze()
                loss = args.loss_fn(output, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        if 'pretrain' in args.task:
            train_meters.update('acc', acc.item())
            train_meters.update('nce', loss.item())
        train_meters.update('loss', loss.item(), B)
        
        # Print progress
        if global_step % args.n_print_steps == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                   epoch, i, len(train_loader),
                   loss=train_meters.get_value('loss')))

        global_step += 1

    return global_step, {
        'loss': train_meters.get_value('loss'),
        'acc': train_meters.get_value('acc') if 'pretrain' in args.task else None,
        'nce': train_meters.get_value('nce') if 'pretrain' in args.task else None
    }

def validation_loop(model, val_loader, val_collector, args):
    """Run validation loop and compute metrics.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        val_collector: Metrics collector for validation
        args: Training arguments
        
    Returns:
        val_metrics: Dictionary of validation metrics
    """
    print("[DEBUG] Starting validation loop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_collector.reset()
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle different return formats from dataloader
            print("[DEBUG] Processing batch with type:", type(batch_data))
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 3:
                    # Dataset returns (input, label, source)
                    val_input, labels, sources = batch_data
                    print("[DEBUG] Got sources from batch:", sources[:5])  # Print first 5 sources
                elif len(batch_data) == 2:
                    # Dataset returns (input, label)
                    val_input, labels = batch_data
                    sources = None
                    print("[DEBUG] No sources in batch data")
                else:
                    val_input = batch_data[0]
                    labels = None
                    sources = None
                    print("[DEBUG] Single item batch, no labels or sources")
            else:
                val_input = batch_data
                labels = None
                sources = None
                print("[DEBUG] Single tensor batch, no labels or sources")
            
            # Move inputs to device
            val_input = val_input.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            # Process sources if available
            if sources is not None:
                print("[DEBUG] Processing sources")
                # Convert bytes to strings if needed
                if isinstance(sources[0], bytes):
                    sources = [s.decode('utf-8') for s in sources]
                    print("[DEBUG] Decoded sources from bytes:", sources[:5])
                
                # Extract hydrophone names
                hydrophone_names = []
                for source in sources:
                    parts = source.split('_')
                    if len(parts) >= 1:
                        hydrophone_names.append(parts[0])
                    else:
                        hydrophone_names.append(source)
                print("[DEBUG] Extracted hydrophone names:", hydrophone_names[:5])
                sources = hydrophone_names
            
            # Get model output
            if args.task == 'pretrain_mpc':
                output = model(val_input, args.task, cluster=True, mask_patch=args.mask_patch)
                print("[DEBUG] Model output type for pretrain_mpc:", type(output))
                if isinstance(output, tuple):
                    print("[DEBUG] Output tuple length:", len(output))
            elif args.task == 'pretrain_mpg':
                output = model(val_input, args.task, cluster=True, mask_patch=args.mask_patch)
            elif args.task == 'pretrain_joint':
                mpc_output = model(val_input, 'pretrain_mpc', mask_patch=args.mask_patch, cluster=True)
                mpg_output = model(val_input, 'pretrain_mpg', mask_patch=args.mask_patch, cluster=True)
                output = (mpc_output, mpg_output)
                print("[DEBUG] Model output type for pretrain_joint:", type(output))
                if isinstance(output, tuple):
                    print("[DEBUG] Output tuple length:", len(output))
            else:
                output = model(val_input, args.task)
                if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                    loss = args.loss_fn(output, torch.argmax(labels.long(), axis=1))
                else:
                    # For binary classification (BCE loss), squeeze the output to match target shape
                    if 'pretrain' not in args.task:
                        output = output.squeeze()
                    loss = args.loss_fn(output, labels)
                output = (output, loss)  # Pack output and loss together
            
            # Update metrics
            print("[DEBUG] Updating metrics collector with sources:", sources[:5] if sources else None)
            val_collector.update(output, labels, sources)
    
    # Compute final metrics
    print("[DEBUG] Computing final metrics")
    val_metrics = val_collector.compute_metrics()
    print("[DEBUG] Validation metrics:", val_metrics)
    if 'hydrophone_metrics' in val_metrics:
        print("[DEBUG] Found hydrophone metrics:", list(val_metrics['hydrophone_metrics'].keys()))
    else:
        print("[DEBUG] No hydrophone metrics in validation metrics")
    
    return val_metrics 