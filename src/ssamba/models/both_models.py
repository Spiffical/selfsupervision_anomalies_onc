"""
Author: Siavash Shams
Date: 4/10/2024

This script contains the implementation of the AMBAModel and ASTModel for audio processing tasks.
The ASTModel is adapted from Yuan Gong's code.

"""
import torch.nn as nn
import torch
import sys

from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random

try:
    from .models_mamba import VisionMamba
    print("VisionMamba imported successfully")
except ImportError:
    print("Failed to import VisionMamba")
    VisionMamba = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    print("RMSNorm imported successfully")
except ImportError:
    print("Failed to import RMSNorm")
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        
        t = torch.arange(self.ft_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[:, None, None, :])
        self.register_buffer("sin_cached", emb.sin()[:, None, None, :])

    def forward(self, x):
        return x * self.cos_cached + self.rotate_half(x) * self.sin_cached

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

class AMBAModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None, vision_mamba_config=None):
        
        print("Vision Mamba Config:", vision_mamba_config)
        super(AMBAModel, self).__init__()
        #assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AMBA models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # Update default config to match our spectrogram dimensions
            default_vision_mamba_config = {
                'img_size': (512, 512),  # Changed to match our input dimensions
                'patch_size': 16,
                'stride': 16,
                'embed_dim': 768,
                'depth': 24,
                'rms_norm': False,
                'residual_in_fp32': False,
                'fused_add_norm': False,
                'final_pool_type': 'none',
                'if_abs_pos_embed': True,
                'if_rope': False,
                'if_rope_residual': False,
                'if_cls_token': True,
                'if_devide_out': True,
                'use_middle_cls_token': False,
            }
            
            combined_vision_mamba_config = {**default_vision_mamba_config, **(vision_mamba_config or {})}
            # Remove 'bimamba_type' if present to avoid compatibility issues with newer mamba
            if 'bimamba_type' in combined_vision_mamba_config:
                del combined_vision_mamba_config['bimamba_type']
            # Replace self.v with MambaBlocksSequential
            print("combined_vision_mamba_config",combined_vision_mamba_config)
            self.v = VisionMamba(**combined_vision_mamba_config)

            self.cls_token_num = 1
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = nn.Unfold(kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride), padding=(self.fshape//2, self.tshape//2))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        # use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            print("Loading pretrained model from:", load_pretrained_mdl_path)

            sd = torch.load(load_pretrained_mdl_path, map_location=device, weights_only=False)
            print("Loaded state dict keys:", sd.keys())
            
            # Try to get shapes with and without DataParallel prefix
            try:
                if 'module.v.patch_embed.proj.weight' in sd:
                    p_fshape = sd['module.v.patch_embed.proj.weight'].shape[2]
                    p_tshape = sd['module.v.patch_embed.proj.weight'].shape[3]
                    p_input_fdim = sd['module.p_input_fdim'].item()
                    p_input_tdim = sd['module.p_input_tdim'].item()
                elif 'v.patch_embed.proj.weight' in sd:
                    p_fshape = sd['v.patch_embed.proj.weight'].shape[2]
                    p_tshape = sd['v.patch_embed.proj.weight'].shape[3]
                    p_input_fdim = sd['p_input_fdim'].item()
                    p_input_tdim = sd['p_input_tdim'].item()
                else:
                    # If loading from a checkpoint dictionary
                    if 'model_state_dict' in sd:
                        state_dict = sd['model_state_dict']
                        if 'module.v.patch_embed.proj.weight' in state_dict:
                            p_fshape = state_dict['module.v.patch_embed.proj.weight'].shape[2]
                            p_tshape = state_dict['module.v.patch_embed.proj.weight'].shape[3]
                            p_input_fdim = state_dict['module.p_input_fdim'].item()
                            p_input_tdim = state_dict['module.p_input_tdim'].item()
                        elif 'v.patch_embed.proj.weight' in state_dict:
                            p_fshape = state_dict['v.patch_embed.proj.weight'].shape[2]
                            p_tshape = state_dict['v.patch_embed.proj.weight'].shape[3]
                            p_input_fdim = state_dict['p_input_fdim'].item()
                            p_input_tdim = state_dict['p_input_tdim'].item()
                        else:
                            raise KeyError("Could not find patch_embed weights in checkpoint")
                    else:
                        raise KeyError("Could not find expected weights in checkpoint")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Available keys in checkpoint:", sd.keys())
                if 'model_state_dict' in sd:
                    print("Available keys in model_state_dict:", sd['model_state_dict'].keys())
                raise ValueError('Failed to load checkpoint weights. Please check the checkpoint structure.')
            
            print('Now loading SSL pretrained model from ' + load_pretrained_mdl_path)
            print(f'Patch shapes from checkpoint - fshape: {p_fshape}, tshape: {p_tshape}')
            print(f'Input dimensions from checkpoint - fdim: {p_input_fdim}, tdim: {p_input_tdim}')
            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
            default_vision_mamba_config = {
            'img_size': (128, 1024),
            'patch_size': 16,
            'stride': 8,
            'embed_dim': 768,
            'depth': 24,
            'rms_norm': True,
            'residual_in_fp32': True,
            'fused_add_norm': True,
            'final_pool_type': 'mean',
            'if_abs_pos_embed': True,
            'if_rope': False,
            'if_rope_residual': False,
            'bimamba_type': "v2",
            'if_cls_token': True,
            'if_devide_out': True,
            'use_middle_cls_token': True,
        }
            
            
            combined_vision_mamba_config = {**default_vision_mamba_config, **vision_mamba_config}
            # Remove 'bimamba_type' if present to avoid compatibility issues with newer mamba
            if 'bimamba_type' in combined_vision_mamba_config:
                del combined_vision_mamba_config['bimamba_type']
            # Replace self.v with MambaBlocksSequential
            print("combined_vision_mamba_config",combined_vision_mamba_config)

            audio_model = AMBAModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                   input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size, vision_mamba_config=combined_vision_mamba_config)
            
            # Handle both DataParallel and non-DataParallel state dicts
            if 'model_state_dict' in sd:
                state_dict = sd['model_state_dict']
            else:
                state_dict = sd
                
            # Convert state dict keys if needed
            if not isinstance(audio_model, nn.DataParallel) and any(k.startswith('module.') for k in state_dict.keys()):
                # Remove 'module.' prefix if model is not wrapped but state dict is
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif isinstance(audio_model, nn.DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
                # Add 'module.' prefix if model is wrapped but state dict isn't
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
            # Handle positional embedding resizing before loading state dict
            pos_embed_key = 'module.v.pos_embed' if isinstance(audio_model, nn.DataParallel) else 'v.pos_embed'
            if pos_embed_key in state_dict:
                pos_embed_checkpoint = state_dict[pos_embed_key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = (audio_model.module.v.patch_embed.num_patches if isinstance(audio_model, nn.DataParallel) 
                             else audio_model.v.patch_embed.num_patches)
                num_extra_tokens = 1  # cls token
                
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                
                if orig_size != new_size:
                    print(f'Position embedding grid-size from {orig_size}x{orig_size} to {new_size}x{new_size}')
                    # exclude cls token
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens.permute(0, 3, 1, 2),
                        size=(new_size, new_size),
                        mode='bicubic',
                        align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((pos_embed_checkpoint[:, :num_extra_tokens], pos_tokens), dim=1)
                    state_dict[pos_embed_key] = new_pos_embed
                    
                    print(f'Resized position embedding from {pos_embed_checkpoint.shape} to {new_pos_embed.shape}')
            
            # Wrap in DataParallel if not already wrapped
            if not isinstance(audio_model, nn.DataParallel):
                audio_model = torch.nn.DataParallel(audio_model)
            
            # Load the state dict
            try:
                audio_model.load_state_dict(state_dict, strict=False)
                print("Model state loaded successfully")
            except Exception as e:
                print(f"Error loading model state: {str(e)}")
                print("Available keys in state_dict:", state_dict.keys())
                print("Model's state_dict keys:", audio_model.state_dict().keys())
                raise ValueError('Failed to load model state. Keys may not match.')
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, 1)  # Output single value for binary classification
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)
    
    def finetuningavgtok_1sec(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # mamba impl
        residual = None
        hidden_states = x
        token_position = 0
        if not self.v.if_bidirectional:
            for layer in self.v.layers:
                hidden_states, residual = layer(hidden_states, residual)
        else:
            for i in range(len(self.v.layers) // 2):
                hidden_states_f, residual_f = self.v.layers[i * 2](hidden_states, residual)
                hidden_states_b, residual_b = self.v.layers[i * 2 + 1](hidden_states.flip([1]), None if residual is None else residual.flip([1]))
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.v.drop_path(hidden_states)
            hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.v.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )

        x = self.v.norm_f(hidden_states)

        # Average output of tokens within each 1-second segment
        tokens_per_second = x.shape[1] // 60  
        x_averaged = torch.stack([torch.mean(x[:, i * tokens_per_second:(i + 1) * tokens_per_second, :], dim=1) for i in range(60)], dim=1)
        x_averaged = self.mlp_head(x_averaged)
        return x_averaged

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        
        # mamba impl
        residual = None
        hidden_states = x
        token_position = 0
        if not self.v.if_bidirectional:
            for layer in self.v.layers:
                
                hidden_states, residual = layer(
                    hidden_states, residual
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.v.layers) // 2):

                hidden_states_f, residual_f = self.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.v.drop_path(hidden_states)
            hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.v.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )

        x = self.v.norm_f(hidden_states)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        
        # mamba impl
        residual = None
        hidden_states = x
        token_position = 0
        if not self.v.if_bidirectional:
            for layer in self.v.layers:
                
                hidden_states, residual = layer(
                    hidden_states, residual
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.v.layers) // 2):

                hidden_states_f, residual_f = self.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.v.drop_path(hidden_states)
            hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.v.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )

       
        x = self.v.norm_f(hidden_states)


        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    # masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
    
        
        # mamba impl
        residual = None
        hidden_states = x
        token_position = 0
        if not self.v.if_bidirectional:
            for layer in self.v.layers:
                
                hidden_states, residual = layer(
                    hidden_states, residual
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.v.layers) // 2):

                hidden_states_f, residual_f = self.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.v.drop_path(hidden_states)
            hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.v.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )

        # return only cls token if it exists
        
        x = self.v.norm_f(hidden_states)
        
        
   
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # 8/12/2022: has a difference with equation (1) in the ssast paper but (likely) performance-wise similar, see https://github.com/YuanGongND/ssast/issues/13
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # visualize the masked area, for probing test only, set show_mask = False for any training/inference.
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            # print(total)
            # print(self.softmax(total))
            # print(torch.argmax(self.softmax(total), dim=0))
            # print(self.mask_correct)
            # print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            # print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

        
    # # masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        # mamba impl
        residual = None
        hidden_states = x
        token_position = 0
        if not self.v.if_bidirectional:
            for layer in self.v.layers:
                
                hidden_states, residual = layer(
                    hidden_states, residual
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.v.layers) // 2):

                hidden_states_f, residual_f = self.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.v.drop_path(hidden_states)
            hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.v.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )

        
        x = self.v.norm_f(hidden_states)
        
        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse
        
     
    def forward(self, x, task, cluster=True, mask_patch=400):
        # Handle different input formats
        if x.dim() == 3:
            # Input is (B, time, freq); add channel dimension
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # Input is (B, H, W, C); permute to (B, C, H, W)
            if x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            # Input is (B, 1, time, freq, 1); remove the extra trailing dimension
            x = x.squeeze(-1)
        
        # For finetuning tasks, we need to transpose to (B, 1, freq, time).
        # For pretraining tasks, keep the original (B, 1, time, freq) orientation
        if task in ['ft_avgtok', 'ft_avgtok_1sec', 'ft_cls']:
            x = x.transpose(2, 3)
        
        # Ensure input dimensions match model's expected size
        B, C, T, F = x.shape
        if T != 512 or F != 512:
            raise ValueError(f'Input shape {x.shape} does not match expected shape (B, 1, 512, 512)')
        if C != 1:
            raise ValueError(f'Expected 1 channel but got {C} channels')
        
        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAMBA fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        elif task == 'ft_avgtok_1sec':
            return self.finetuningavgtok_1sec(x)
        # alternatively, use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

            
class ASTModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base_nokd',
                 pretrain_stage=True, load_pretrained_mdl_path=None):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride), padding=(fshape//2, tshape//2))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        # use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            sd = torch.load(load_pretrained_mdl_path, map_location=device, weights_only=False)
            print("Loaded state dict keys:", sd.keys())
            
            # Try to get shapes with and without DataParallel prefix
            try:
                if 'module.v.patch_embed.proj.weight' in sd:
                    p_fshape = sd['module.v.patch_embed.proj.weight'].shape[2]
                    p_tshape = sd['module.v.patch_embed.proj.weight'].shape[3]
                    p_input_fdim = sd['module.p_input_fdim'].item()
                    p_input_tdim = sd['module.p_input_tdim'].item()
                elif 'v.patch_embed.proj.weight' in sd:
                    p_fshape = sd['v.patch_embed.proj.weight'].shape[2]
                    p_tshape = sd['v.patch_embed.proj.weight'].shape[3]
                    p_input_fdim = sd['p_input_fdim'].item()
                    p_input_tdim = sd['p_input_tdim'].item()
                else:
                    # If loading from a checkpoint dictionary
                    if 'model_state_dict' in sd:
                        state_dict = sd['model_state_dict']
                        if 'module.v.patch_embed.proj.weight' in state_dict:
                            p_fshape = state_dict['module.v.patch_embed.proj.weight'].shape[2]
                            p_tshape = state_dict['module.v.patch_embed.proj.weight'].shape[3]
                            p_input_fdim = state_dict['module.p_input_fdim'].item()
                            p_input_tdim = state_dict['module.p_input_tdim'].item()
                        elif 'v.patch_embed.proj.weight' in state_dict:
                            p_fshape = state_dict['v.patch_embed.proj.weight'].shape[2]
                            p_tshape = state_dict['v.patch_embed.proj.weight'].shape[3]
                            p_input_fdim = state_dict['p_input_fdim'].item()
                            p_input_tdim = state_dict['p_input_tdim'].item()
                        else:
                            raise KeyError("Could not find patch_embed weights in checkpoint")
                    else:
                        raise KeyError("Could not find expected weights in checkpoint")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Available keys in checkpoint:", sd.keys())
                if 'model_state_dict' in sd:
                    print("Available keys in model_state_dict:", sd['model_state_dict'].keys())
                raise ValueError('Failed to load checkpoint weights. Please check the checkpoint structure.')
            
            print('Now loading SSL pretrained model from ' + load_pretrained_mdl_path)
            print(f'Patch shapes from checkpoint - fshape: {p_fshape}, tshape: {p_tshape}')
            print(f'Input dimensions from checkpoint - fdim: {p_input_fdim}, tdim: {p_input_tdim}')
            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
            audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                   input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, 1)  # Output single value for binary classification
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)
    
    def finetuningavgtok_1sec(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # Average output of tokens within each 1-second segment
        tokens_per_second = x.shape[1] // 60  
        x_averaged = torch.stack([torch.mean(x[:, i * tokens_per_second:(i + 1) * tokens_per_second, :], dim=1) for i in range(60)], dim=1)
        x_averaged = self.mlp_head(x_averaged)
        return x_averaged
        

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        
        # mamba impl
        residual = None
        hidden_states = x
        token_position = 0
        if not self.v.if_bidirectional:
            for layer in self.v.layers:
                
                hidden_states, residual = layer(
                    hidden_states, residual
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.v.layers) // 2):

                hidden_states_f, residual_f = self.v.layers[i * 2](
                    hidden_states, residual
                )
                hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1])
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.v.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.v.drop_path(hidden_states)
            hidden_states = self.v.norm_f(residual.to(dtype=self.v.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.v.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.v.drop_path(hidden_states),
                self.v.norm_f.weight,
                self.v.norm_f.bias,
                eps=self.v.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.v.residual_in_fp32,
            )

       
        x = self.v.norm_f(hidden_states)


        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    # masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # 8/12/2022: has a difference with equation (1) in the ssast paper but (likely) performance-wise similar, see https://github.com/YuanGongND/ssast/issues/13
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # visualize the masked area, for probing test only, set show_mask = False for any training/inference.
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            # print(total)
            # print(self.softmax(total))
            # print(torch.argmax(self.softmax(total), dim=0))
            # print(self.mask_correct)
            # print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            # print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

    # # masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def forward(self, x, task, cluster=True, mask_patch=400):
        # Handle different input formats
        if x.dim() == 3:
            # Input is (B, time, freq); add channel dimension
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # Input is (B, H, W, C); permute to (B, C, H, W)
            if x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            # Input is (B, 1, time, freq, 1); remove the extra trailing dimension
            x = x.squeeze(-1)
        
        # For finetuning tasks, we need to transpose to (B, 1, freq, time).
        # For pretraining tasks, keep the original (B, 1, time, freq) orientation
        if task in ['ft_avgtok', 'ft_avgtok_1sec', 'ft_cls']:
            x = x.transpose(2, 3)
        
        # Ensure input dimensions match model's expected size
        B, C, T, F = x.shape
        if T != 512 or F != 512:
            raise ValueError(f'Input shape {x.shape} does not match expected shape (B, 1, 512, 512)')
        if C != 1:
            raise ValueError(f'Expected 1 channel but got {C} channels')
        
        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAMBA fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        elif task == 'ft_avgtok_1sec':
            return self.finetuningavgtok_1sec(x)
        # alternatively, use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

            
            
            
