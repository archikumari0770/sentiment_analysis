
import torch
import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_VIDEO_DIR = os.path.join(DATA_DIR, 'raw_videos')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
    
    # Video Processing
    VIDEO_FPS = 1  # Extract 1 frame per second
    FRAME_SIZE = (224, 224)  # ResNet input size
    MAX_FRAMES = 30  # Maximum frames to process per video
    
    # Audio Processing
    SAMPLE_RATE = 16000  # Standard for speech
    AUDIO_DURATION = 10  # seconds
    N_MELS = 128
    HOP_LENGTH = 512
    
    # Model Parameters
    VISUAL_FEATURE_DIM = 2048  # ResNet50 output
    AUDIO_FEATURE_DIM = 768  # Wav2Vec2 output
    HIDDEN_DIM = 512
    NUM_CLASSES = 3  # Negative, Neutral, Positive
    DROPOUT = 0.3
    
    # Training Parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class Labels
    SENTIMENT_LABELS = {
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    }
    
    @staticmethod
    def create_directories():
        '''Create necessary directories'''
        os.makedirs(Config.RAW_VIDEO_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

config = Config()
