
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config import config

class VideoProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(config.FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_frames(self, video_path, fps=None):
        '''Extract frames from video at specified FPS'''
        if fps is None:
            fps = config.VIDEO_FPS
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if len(frames) >= config.MAX_FRAMES:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def preprocess_frames(self, frames):
        '''Preprocess frames for model input'''
        processed_frames = []
        
        for frame in frames:
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame)
            # Apply transformations
            tensor_frame = self.transform(pil_frame)
            processed_frames.append(tensor_frame)
        
        # Stack frames into tensor
        if processed_frames:
            return torch.stack(processed_frames)
        else:
            # Return empty tensor if no frames
            return torch.zeros((1, 3, *config.FRAME_SIZE))
    
    def process_video(self, video_path):
        '''Complete video processing pipeline'''
        frames = self.extract_frames(video_path)
        processed_frames = self.preprocess_frames(frames)
        return processed_frames
