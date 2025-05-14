import torch.nn as nn
from models.models_mamba import VisionMamba
import torch

class SupervisedAMBAModel(nn.Module):
    """
    A simplified version of AMBAModel for direct supervised training.
    Removes all pretraining-specific components and focuses on classification.
    """
    def __init__(self, backbone_config):
        super().__init__()
        # Ensure we use CLS token in the backbone
        backbone_config['if_cls_token'] = True
        backbone_config['use_middle_cls_token'] = False
        backbone_config['final_pool_type'] = 'none'
        backbone_config['num_classes'] = 0
        
        self.backbone = VisionMamba(**backbone_config)
        
        # Initialize backbone weights
        self._init_backbone_weights()
        
        # Modified classifier with better normalization and initialization
        hidden_dim = backbone_config['embed_dim'] // 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_config['embed_dim'], eps=1e-6),
            nn.Linear(backbone_config['embed_dim'], hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2, eps=1e-6),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize classifier weights with smaller values
        self._init_classifier_weights()
        
        # Store config for debugging
        self.embed_dim = backbone_config['embed_dim']
        self.debug_mode = False
        
    def _init_backbone_weights(self):
        """Initialize backbone weights with better defaults"""
        for name, param in self.backbone.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'cls_token' in name:
                nn.init.normal_(param, std=0.02)
            elif 'pos_embed' in name:
                nn.init.normal_(param, std=0.02)
    
    def _init_classifier_weights(self):
        """Initialize classifier weights with smaller values for stability"""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # Use smaller initialization for better stability
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                layer.weight.data.mul_(0.1)  # Scale down weights
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        if self.debug_mode:
            # Input statistics
            print(f"\nInput shape: {x.shape}")
            print(f"Input stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        # Get backbone features
        features = self.backbone(x, return_features=True)
        
        if self.debug_mode:
            # Feature statistics
            print(f"Backbone output shape: {features.shape}")
            print(f"Feature stats - min: {features.min():.4f}, max: {features.max():.4f}, mean: {features.mean():.4f}, std: {features.std():.4f}")
            
            # Verify we're getting the expected CLS token output
            if features.shape[1] != self.embed_dim:
                raise ValueError(f"Expected feature dim {self.embed_dim}, got {features.shape[1]}")
        
        # Pass through classifier layers with debug info
        for i, layer in enumerate(self.classifier):
            features = layer(features)
            if self.debug_mode:
                print(f"After layer {i} ({type(layer).__name__}) - shape: {features.shape}")
                print(f"Stats - min: {features.min():.4f}, max: {features.max():.4f}, mean: {features.mean():.4f}, std: {features.std():.4f}")
        
        return features.squeeze(-1)  # Shape: [B]
    
    def get_layer_output(self, x, layer_name):
        """Helper method to get output of a specific layer for debugging"""
        if layer_name == 'backbone':
            return self.backbone(x, return_features=True)
        elif layer_name == 'classifier':
            features = self.backbone(x, return_features=True)
            return self.classifier(features)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")