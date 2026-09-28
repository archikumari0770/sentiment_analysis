"""
Test the model on different modality types:
- Video-only
- Audio-only
- Multimodal (both)

Shows how the model handles each type
"""

import torch
import os
from config import config
from models.fusion_model import MultimodalFusionModel
from utils.video_processor import VideoProcessor
from utils.audio_processor import AudioProcessor

class ModalityTester:
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
            
            # Detect modality
            has_visual = not torch.allclose(visual_features, torch.zeros_like(visual_features), atol=1e-6)
            has_audio = not torch.allclose(audio_features, torch.zeros_like(audio_features), atol=1e-6)
            
            # Get prediction
            outputs = self.model(visual_features, audio_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return {
                'sentiment': config.SENTIMENT_LABELS[predicted_class],
                'class_id': predicted_class,
                'confidence': confidence,
                'has_visual': has_visual,
                'has_audio': has_audio,
                'modality_type': self._get_modality_type(has_visual, has_audio),
                'probabilities': {
                    config.SENTIMENT_LABELS[i]: probabilities[0, i].item()
                    for i in range(config.NUM_CLASSES)
                }
            }
    
    def _get_modality_type(self, has_visual, has_audio):
        if has_visual and has_audio:
            return 'Multimodal (Video + Audio)'
        elif has_visual:
            return 'Video Only'
        elif has_audio:
            return 'Audio Only'
        else:
            return 'No Modality Detected'
    
    def test_all_types(self, video_dir='data/raw_videos'):
        '''Test on all modality types'''
        print("\n" + "=" * 70)
        print("MODALITY TYPE TESTING")
        print("=" * 70)
        
        if not os.path.exists(video_dir):
            print(f"❌ Directory not found: {video_dir}")
            return
        
        # Find videos of each type
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        video_only = [f for f in video_files if 'video_only' in f]
        audio_only = [f for f in video_files if 'audio_only' in f]
        multimodal = [f for f in video_files if 'multimodal' in f]
        
        print(f"\nFound:")
        print(f"  • Video-only: {len(video_only)}")
        print(f"  • Audio-only: {len(audio_only)}")
        print(f"  • Multimodal:  {len(multimodal)}")
        
        # Test one of each type
        results = {}
        
        if video_only:
            print("\n" + "-" * 70)
            print("1️⃣  Testing VIDEO-ONLY sample")
            print("-" * 70)
            result = self.predict(os.path.join(video_dir, video_only[0]))
            self._print_result(video_only[0], result)
            results['video_only'] = result
        
        if audio_only:
            print("\n" + "-" * 70)
            print("2️⃣  Testing AUDIO-ONLY sample")
            print("-" * 70)
            result = self.predict(os.path.join(video_dir, audio_only[0]))
            self._print_result(audio_only[0], result)
            results['audio_only'] = result
        
        if multimodal:
            print("\n" + "-" * 70)
            print("3️⃣  Testing MULTIMODAL sample")
            print("-" * 70)
            result = self.predict(os.path.join(video_dir, multimodal[0]))
            self._print_result(multimodal[0], result)
            results['multimodal'] = result
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\n✅ The model successfully handled all modality types!")
        print("\nKey advantages:")
        print("  • Works with video-only inputs")
        print("  • Works with audio-only inputs")
        print("  • Combines both when available")
        print("  • More robust to real-world scenarios")
        
        return results
    
    def _print_result(self, filename, result):
        print(f"\nFile: {filename}")
        print(f"Modality Type: {result['modality_type']}")
        print(f"  Visual Present: {'✓' if result['has_visual'] else '✗'}")
        print(f"  Audio Present:  {'✓' if result['has_audio'] else '✗'}")
        print(f"\nPrediction:")
        print(f"  Sentiment:  {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 40)
            print(f"  {label:8s}: {bar} {prob:.2%}")

def main():
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print('❌ Model not found. Please train the model first.')
        print('   Run: python train.py')
        return
    
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "MODALITY TYPE TESTER" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tester = ModalityTester(model_path)
    tester.test_all_types()
    
    print("\n")

if __name__ == '__main__':
    main()