import torch
import numpy as np
from config import config
from models.fusion_model import MultimodalFusionModel
from utils.video_processor import VideoProcessor
from utils.audio_processor import AudioProcessor
import os

class SentimentPredictor:
    def __init__(self, model_path):
        self.device = config.DEVICE
        
        # Load model
        self.model = MultimodalFusionModel(num_classes=config.NUM_CLASSES).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
    
    def predict(self, video_path):
        '''Predict sentiment from video'''
        with torch.no_grad():
            # Process video
            visual_features = self.video_processor.process_video(video_path)
            visual_features = visual_features.unsqueeze(0).to(self.device)
            
            # Process audio
            audio_features = self.audio_processor.process_audio(video_path)
            audio_features = audio_features.to(self.device)
            
            # Get prediction
            outputs = self.model(visual_features, audio_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return {
                'sentiment': config.SENTIMENT_LABELS[predicted_class],
                'class_id': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    config.SENTIMENT_LABELS[i]: probabilities[0, i].item()
                    for i in range(config.NUM_CLASSES)
                }
            }
    
    def predict_batch(self, video_paths):
        '''Predict sentiment for multiple videos'''
        results = []
        for video_path in video_paths:
            print(f'Processing: {os.path.basename(video_path)}')
            result = self.predict(video_path)
            result['filename'] = os.path.basename(video_path)
            results.append(result)
        return results

def main():
    # Example usage
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print('Error: Model not found. Please train the model first.')
        return
    
    # Create predictor
    predictor = SentimentPredictor(model_path)
    
    # Get videos to process
    video_dir = config.RAW_VIDEO_DIR
    video_files = [
        os.path.join(video_dir, f) 
        for f in os.listdir(video_dir)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    
    if not video_files:
        print('No video files found in', video_dir)
        return
    
    # Predict
    results = predictor.predict_batch(video_files)
    
    # Display results
    print('\\n' + '='*60)
    print('SENTIMENT ANALYSIS RESULTS')
    print('='*60)
    
    for result in results:
        print(f"\\nFile: {result['filename']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print('-'*60)

if __name__ == '__main__':
    main()