import torch
import torch.nn as nn
from models.visual_model import VisualFeatureExtractor
from models.audio_model import AudioFeatureExtractor
from config import config

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=3, fusion_type='attention'):
        '''
        Flexible multimodal model that handles:
        - Video only (audio missing)
        - Audio only (video missing)
        - Both video and audio (multimodal)
        '''
        super(MultimodalFusionModel, self).__init__()
        
        self.fusion_type = fusion_type
        
        # Feature extractors
        self.visual_extractor = VisualFeatureExtractor(freeze_backbone=True)
        self.audio_extractor = AudioFeatureExtractor(freeze_backbone=True)
        
        # Individual modality classifiers (for unimodal samples)
        self.visual_classifier = nn.Sequential(
            nn.Linear(config.VISUAL_FEATURE_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, num_classes)
        )
        
        self.audio_classifier = nn.Sequential(
            nn.Linear(config.AUDIO_FEATURE_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, num_classes)
        )
        
        # Fusion layer for multimodal samples
        if fusion_type == 'concat':
            fusion_dim = config.VISUAL_FEATURE_DIM + config.AUDIO_FEATURE_DIM
        elif fusion_type == 'add':
            self.visual_proj = nn.Linear(config.VISUAL_FEATURE_DIM, config.HIDDEN_DIM)
            self.audio_proj = nn.Linear(config.AUDIO_FEATURE_DIM, config.HIDDEN_DIM)
            fusion_dim = config.HIDDEN_DIM
        elif fusion_type == 'attention':
            self.attention = CrossModalAttention(
                config.VISUAL_FEATURE_DIM,
                config.AUDIO_FEATURE_DIM,
                config.HIDDEN_DIM
            )
            fusion_dim = config.HIDDEN_DIM
        
        # Multimodal classifier
        self.multimodal_classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, num_classes)
        )
    
    def detect_missing_modality(self, visual_input, audio_input):
        '''
        Detect which modalities are present
        Returns: (has_visual, has_audio)
        '''
        # Check if visual is all zeros/silence
        has_visual = not torch.allclose(visual_input, torch.zeros_like(visual_input), atol=1e-6)
        
        # Check if audio is all zeros/silence
        has_audio = not torch.allclose(audio_input, torch.zeros_like(audio_input), atol=1e-6)
        
        return has_visual, has_audio
    
    def forward(self, visual_input, audio_input):
        '''
        Args:
            visual_input: tensor of shape (batch_size, num_frames, 3, 224, 224)
            audio_input: tensor of shape (batch_size, audio_length)
        Returns:
            logits: tensor of shape (batch_size, num_classes)
        '''
        batch_size = visual_input.shape[0]
        outputs = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            visual_sample = visual_input[i:i+1]
            audio_sample = audio_input[i:i+1]
            
            # Detect which modalities are present
            has_visual, has_audio = self.detect_missing_modality(visual_sample, audio_sample)
            
            if has_visual and has_audio:
                # Multimodal: Use both modalities
                visual_features = self.visual_extractor(visual_sample)
                audio_features = self.audio_extractor(audio_sample)
                
                # Fusion
                if self.fusion_type == 'concat':
                    fused_features = torch.cat([visual_features, audio_features], dim=1)
                elif self.fusion_type == 'add':
                    visual_proj = self.visual_proj(visual_features)
                    audio_proj = self.audio_proj(audio_features)
                    fused_features = visual_proj + audio_proj
                elif self.fusion_type == 'attention':
                    fused_features = self.attention(visual_features, audio_features)
                
                output = self.multimodal_classifier(fused_features)
                
            elif has_visual and not has_audio:
                # Video only
                visual_features = self.visual_extractor(visual_sample)
                output = self.visual_classifier(visual_features)
                
            elif not has_visual and has_audio:
                # Audio only
                audio_features = self.audio_extractor(audio_sample)
                output = self.audio_classifier(audio_features)
                
            else:
                # Neither modality present (shouldn't happen, but handle gracefully)
                output = torch.zeros(1, self.multimodal_classifier[-1].out_features).to(visual_input.device)
            
            outputs.append(output)
        
        # Stack all outputs
        return torch.cat(outputs, dim=0)

class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, audio_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, visual_features, audio_features):
        # Project features
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)  # (batch, 1, hidden)
        audio_proj = self.audio_proj(audio_features).unsqueeze(1)  # (batch, 1, hidden)
        
        # Concatenate for attention
        features = torch.cat([visual_proj, audio_proj], dim=1)  # (batch, 2, hidden)
        
        # Self-attention
        attended, _ = self.attention(features, features, features)
        
        # Average pool
        fused = attended.mean(dim=1)  # (batch, hidden)
        
        return fused