import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AudioFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(AudioFeatureExtractor, self).__init__()
        
        # Load pre-trained Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, audio_length)
        Returns:
            features: tensor of shape (batch_size, feature_dim)
        '''
        # Process audio through Wav2Vec2
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(x)
            features = outputs.last_hidden_state  # (batch_size, seq_len, 768)
        
        # Temporal aggregation
        features = features.permute(0, 2, 1)  # (batch_size, 768, seq_len)
        features = self.temporal_pool(features).squeeze(-1)  # (batch_size, 768)
        
        return features