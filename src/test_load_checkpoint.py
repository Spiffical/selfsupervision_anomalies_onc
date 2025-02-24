#!/usr/bin/env python3

import torch
import os
import argparse
from models import AMBAModel
from torch import nn
import json

def print_state_dict_info(state_dict, name="State Dict"):
    """Helper function to print information about a state dict."""
    print(f"\n{name} Information:")
    print(f"Number of keys: {len(state_dict)}")
    print("First few keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {key}")
    if 'module.' in next(iter(state_dict.keys()), ''):
        print("Contains 'module.' prefix")
    else:
        print("Does not contain 'module.' prefix")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory containing the checkpoints")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Define paths
    model_path = os.path.join(args.exp_dir, 'models', f'audio_model.{args.epoch}.pth')
    optim_path = os.path.join(args.exp_dir, 'models', f'optim_state.{args.epoch}.pth')
    checkpoint_path = os.path.join(args.exp_dir, 'models', f'checkpoint.{args.epoch}.pth')
    best_metric_path = os.path.join(args.exp_dir, 'models', 'best_metric.pth')

    try:
        # First try loading complete checkpoint if it exists
        if os.path.exists(checkpoint_path):
            print(f"\nAttempting to load complete checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print("\nCheckpoint contents:")
            print("Keys:", list(checkpoint.keys()))
            
            if 'model_state_dict' in checkpoint:
                print_state_dict_info(checkpoint['model_state_dict'], "Model State Dict from Checkpoint")
            
            if 'optimizer_state_dict' in checkpoint:
                print_state_dict_info(checkpoint['optimizer_state_dict'], "Optimizer State Dict from Checkpoint")
            
            if 'args' in checkpoint:
                print("\nSaved args:")
                print(json.dumps(checkpoint['args'], indent=2))

        # Try loading individual model state
        if os.path.exists(model_path):
            print(f"\nAttempting to load model state from: {model_path}")
            model_state = torch.load(model_path, map_location=device)
            print_state_dict_info(model_state, "Individual Model State Dict")

        # Try loading individual optimizer state
        if os.path.exists(optim_path):
            print(f"\nAttempting to load optimizer state from: {optim_path}")
            optim_state = torch.load(optim_path, map_location=device)
            print_state_dict_info(optim_state, "Individual Optimizer State Dict")
            
            if 'param_groups' in optim_state:
                print("\nOptimizer Parameter Groups:")
                for i, group in enumerate(optim_state['param_groups']):
                    print(f"\nGroup {i}:")
                    print("Learning rate:", group['lr'])
                    print("Number of parameters:", len(group['params']))
                    print("Group keys:", list(group.keys()))

        # Try loading best metric
        if os.path.exists(best_metric_path):
            print(f"\nAttempting to load best metric from: {best_metric_path}")
            try:
                # First try with weights_only=False
                torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                best_metric = torch.load(
                    best_metric_path,
                    map_location=device,
                    weights_only=False
                )
                print("Successfully loaded best metric with weights_only=False")
            except Exception as e1:
                print(f"Failed to load with weights_only=False: {str(e1)}")
                try:
                    # Try without weights_only
                    best_metric = torch.load(
                        best_metric_path,
                        map_location=device
                    )
                    print("Successfully loaded best metric without weights_only")
                except Exception as e2:
                    print(f"Failed to load without weights_only: {str(e2)}")
                    best_metric = None
            
            if best_metric is not None:
                print(f"Best metric value: {best_metric}")

        # Create vision mamba config
        vision_mamba_config = {
            'fshape': 16,
            'tshape': 16,
            'fstride': 16,
            'tstride': 16,
            'input_fdim': 512,
            'input_tdim': 512,
            'embed_dim': 768,
            'depth': 24,
            'channels': 1,
            'num_classes': 1000,
            'drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'norm_epsilon': 1e-5,
            'rms_norm': False,
            'residual_in_fp32': False,
            'fused_add_norm': False,
            'if_rope': False,
            'if_rope_residual': False,
            'if_bidirectional': True,
            'final_pool_type': 'none',
            'if_abs_pos_embed': True,
            'if_bimamba': False,
            'if_cls_token': True,
            'if_devide_out': True,
            'use_double_cls_token': False,
            'use_middle_cls_token': False
        }

        print("\nCreating model...")
        model = AMBAModel(
            fshape=16,
            tshape=16,
            fstride=16,
            tstride=16,
            input_fdim=512,
            input_tdim=512,
            model_size='base',
            pretrain_stage=True,
            vision_mamba_config=vision_mamba_config
        )
        model = model.to(device)

        print("\nAttempting to load model state...")
        # If we have a complete checkpoint, use that
        if 'model_state_dict' in locals():
            state_dict_to_load = checkpoint['model_state_dict']
        else:
            state_dict_to_load = model_state

        # Check if we need to handle DataParallel
        if any(k.startswith('module.') for k in state_dict_to_load.keys()):
            print("State dict has DataParallel format")
            if not isinstance(model, nn.DataParallel):
                print("Wrapping model in DataParallel")
                model = nn.DataParallel(model)
        else:
            print("State dict does not have DataParallel format")
            if isinstance(model, nn.DataParallel):
                print("Unwrapping model from DataParallel")
                model = model.module
            else:
                print("Adding 'module.' prefix to state dict")
                state_dict_to_load = {f'module.{k}': v for k, v in state_dict_to_load.items()}
                model = nn.DataParallel(model)

        # Try to load the state dict
        try:
            model.load_state_dict(state_dict_to_load)
            print("Successfully loaded model state!")
        except Exception as e:
            print(f"Failed to load model state: {str(e)}")
            print("\nModel's state_dict keys:")
            for k in model.state_dict().keys():
                print(f"  {k}")
            print("\nLoaded state_dict keys:")
            for k in state_dict_to_load.keys():
                print(f"  {k}")

        # Set up optimizer
        print("\nSetting up optimizer...")
        mlp_list = ['module.mlp_head.0.weight', 'module.mlp_head.0.bias', 
                    'module.mlp_head.1.weight', 'module.mlp_head.1.bias']
        
        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.named_parameters()))
        
        print("\nMLP parameters:", [name for name, _ in mlp_params])
        print("Number of MLP parameters:", len(mlp_params))
        print("Number of base parameters:", len(base_params))
        
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 1e-4},
            {'params': mlp_params, 'lr': 1e-4 * 10}
        ], weight_decay=5e-7, betas=(0.95, 0.999))

        print("\nAttempting to load optimizer state...")
        if 'optimizer_state_dict' in locals():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully loaded optimizer state from checkpoint!")
        elif 'optim_state' in locals():
            optimizer.load_state_dict(optim_state)
            print("Successfully loaded optimizer state from individual file!")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 