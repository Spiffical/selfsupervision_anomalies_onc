import torch.nn as nn
from models.models_mamba import VisionMamba

class SupervisedAMBAModel(nn.Module):
    """
    A simplified version of AMBAModel for direct supervised training.
    Removes all pretraining-specific components and focuses on classification.
    """
    def __init__(self, backbone_config):
        super().__init__()
        # Ensure we use CLS token in the backbone
        backbone_config['if_cls_token'] = True
        backbone_config['use_middle_cls_token'] = False  # Use standard CLS token at start
        backbone_config['final_pool_type'] = 'none'  # Don't do any pooling, we'll use CLS token
        backbone_config['num_classes'] = 0  # Set to 0 to remove the head and get embeddings
        
        self.backbone = VisionMamba(**backbone_config)
        
        # For binary classification, output size is 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_config['embed_dim']),  # Add normalization
            nn.Linear(backbone_config['embed_dim'], 1)
            # No sigmoid since we're using BCEWithLogitsLoss
        )
        
    def forward(self, x):
        # Get backbone features - returns [B, D] when using CLS token
        x = self.backbone(x, return_features=True)  # Use return_features=True to get embeddings
        
        # Pass through classifier
        x = self.classifier(x)  # Shape: [B, 1]
        
        return x.squeeze(-1)  # Shape: [B]