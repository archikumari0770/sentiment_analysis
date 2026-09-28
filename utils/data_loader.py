import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.video_processor import VideoProcessor
from utils.audio_processor import AudioProcessor
from config import config

class MultimodalDataset(Dataset):
    def __init__(self, video_dir, annotation_file=None, mode='train'):
        '''
        Args:
            video_dir: Directory containing video files
            annotation_file: CSV file with columns: filename, label
            mode: 'train', 'val', or 'test'
        '''
        self.video_dir = video_dir
        self.mode = mode
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
        # Load annotations if provided
        if annotation_file and os.path.exists(annotation_file):
            self.annotations = pd.read_csv(annotation_file)
            self.video_files = self.annotations['filename'].tolist()
            self.labels = self.annotations['label'].tolist()
        else:
            # If no annotations, just list all videos
            self.video_files = [f for f in os.listdir(video_dir) 
                               if f.endswith(('.mp4', '.avi', '.mov'))]
            self.labels = [-1] * len(self.video_files)  # Dummy labels
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        
        # Process visual data
        visual_features = self.video_processor.process_video(video_path)
        
        # Process audio data
        audio_features = self.audio_processor.process_audio(video_path)
        
        # Get label
        label = self.labels[idx]
        
        return {
            'visual': visual_features,
            'audio': audio_features,
            'label': torch.tensor(label, dtype=torch.long),
            'filename': self.video_files[idx]
        }

# Move collate_fn outside to make it picklable on Windows
def collate_fn(batch):
    '''Custom collate function to handle variable length sequences'''
    # Handle variable length sequences
    max_frames = max([item['visual'].shape[0] for item in batch])
    
    visual_padded = []
    audio_list = []
    labels = []
    filenames = []
    
    for item in batch:
        # Pad visual features
        visual = item['visual']
        if visual.shape[0] < max_frames:
            padding = torch.zeros(max_frames - visual.shape[0], *visual.shape[1:])
            visual = torch.cat([visual, padding], dim=0)
        visual_padded.append(visual)
        
        audio_list.append(item['audio'])
        labels.append(item['label'])
        filenames.append(item['filename'])
    
    return {
        'visual': torch.stack(visual_padded),
        'audio': torch.stack(audio_list),
        'label': torch.stack(labels),
        'filename': filenames
    }

def get_data_loader(video_dir, annotation_file=None, batch_size=None, mode='train', shuffle=True):
    '''Create data loader'''
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    dataset = MultimodalDataset(video_dir, annotation_file, mode)
    
    # Use num_workers=0 for Windows to avoid multiprocessing issues
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # Changed from 2 to 0 for Windows compatibility
        pin_memory=False  # Disabled for CPU
    )
    
    return loader