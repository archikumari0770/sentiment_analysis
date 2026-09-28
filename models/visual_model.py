import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class VisualFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(VisualFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, num_frames, channels, height, width)
        Returns:
            features: tensor of shape (batch_size, feature_dim)
        '''
        batch_size, num_frames, c, h, w = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features
        features = self.features(x)  # (batch_size * num_frames, 2048, 1, 1)
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 2048)
        
        # Temporal aggregation (average pooling over frames)
        features = features.permute(0, 2, 1)  # (batch_size, 2048, num_frames)
        features = self.temporal_pool(features).squeeze(-1)  # (batch_size, 2048)
        
        return features