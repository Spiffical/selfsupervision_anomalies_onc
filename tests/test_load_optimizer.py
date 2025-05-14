#!/usr/bin/env python3

import torch
import os
import argparse
from models import AMBAModel
from torch import nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the optimizer state dict")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model state dict")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # First try loading the complete checkpoint
        print("\nAttempting to load checkpoint...")
        if os.path.exists(args.checkpoint_path.replace('optim_state', 'checkpoint')):
            checkpoint = torch.load(
                args.checkpoint_path.replace('optim_state', 'checkpoint'),
                map_location=device
            )
            print("\nFound complete checkpoint. Contents:")
            print("Keys:", list(checkpoint.keys()))
            optim_state = checkpoint['optimizer_state_dict']
        else:
            # Fall back to loading individual optimizer state
            print("\nNo complete checkpoint found, loading individual optimizer state...")
            optim_state = torch.load(args.checkpoint_path, map_location=device)
        
        print("\nOptimizer state contents:")
        print("State keys:", list(optim_state.keys()))
        if 'state' in optim_state:
            print("Number of state entries:", len(optim_state['state']))
        if 'param_groups' in optim_state:
            print("\nParameter groups:")
            for i, group in enumerate(optim_state['param_groups']):
                print(f"\nGroup {i}:")
                print("Learning rate:", group['lr'])
                print("Number of parameters:", len(group['params']))
                print("Group keys:", list(group.keys()))
        
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
        
        print("\nVision Mamba Config:", vision_mamba_config)
        
        # Create model
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
        
        # Move model to device
        model = model.to(device)
        
        # Load model state
        print("\nAttempting to load model state...")
        model_state = torch.load(args.model_path, map_location=device)
        
        # Handle DataParallel prefix
        if list(model_state.keys())[0].startswith('module.'):
            # If state dict has 'module' prefix but model isn't DataParallel, remove prefix
            if not isinstance(model, nn.DataParallel):
                model = nn.DataParallel(model)
        else:
            # If state dict doesn't have 'module' prefix but we need DataParallel
            if torch.cuda.is_available():
                model = nn.DataParallel(model)
                # Add 'module' prefix to state dict keys
                model_state = {f'module.{k}': v for k, v in model_state.items()}
        
        # Load the state dict
        model.load_state_dict(model_state)
        print("Successfully loaded model state!")
        
        # Set up optimizer with same structure as training
        print("\nSetting up optimizer...")
        mlp_list = ['module.mlp_head.0.weight', 'module.mlp_head.0.bias', 
                    'module.mlp_head.1.weight', 'module.mlp_head.1.bias']
        
        # Print all parameter names for debugging
        print("\nAll parameter names:")
        for name, _ in model.named_parameters():
            print(name)
            
        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.named_parameters()))
        
        print("\nMLP parameters:", [name for name, _ in mlp_params])
        print("Number of MLP parameters:", len(mlp_params))
        print("Number of base parameters:", len(base_params))
        
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]
        
        # Create optimizer with same structure as in training
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 1e-4},
            {'params': mlp_params, 'lr': 1e-4 * 10}
        ], weight_decay=5e-7, betas=(0.95, 0.999))
        
        print("\nNew optimizer setup:")
        print("Number of parameter groups:", len(optimizer.param_groups))
        for i, group in enumerate(optimizer.param_groups):
            print(f"\nGroup {i}:")
            print("Learning rate:", group['lr'])
            print("Number of parameters:", len(group['params']))
        
        # Try loading the optimizer state
        print("\nAttempting to load state into new optimizer...")
        optimizer.load_state_dict(optim_state)
        print("Successfully loaded optimizer state!")
        
        # Verify optimizer state after loading
        print("\nVerifying optimizer state after loading:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"\nGroup {i}:")
            print("Learning rate:", group['lr'])
            print("Number of parameters:", len(group['params']))
            print("Parameter group keys:", list(group.keys()))
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 