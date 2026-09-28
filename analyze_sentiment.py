"""
Interactive Sentiment Analyzer
User-friendly script to analyze video/audio sentiment
"""

import os
import sys
from inference import SentimentPredictor

def print_header():
    """Print a nice header"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "🎭 SENTIMENT ANALYZER 🎭" + " "*30 + "║")
    print("╚" + "="*68 + "╝")
    print("\n📹 Analyze emotions from videos and audio files")
    print("🎯 Supports: Video-only, Audio-only, or Both combined")

def get_video_path():
    """Ask user for video/audio file path"""
    print("\n" + "─"*70)
    print("STEP 1: Select your file")
    print("─"*70)
    print("\n💡 Tips:")
    print("  • Drag and drop the file into this window, OR")
    print("  • Type/paste the full file path")
    print("  • Supported: .mp4, .avi, .mov, .wav, .mp3")
    print("\n📁 Examples:")
    print("  Windows: C:\\Users\\YourName\\Videos\\my_video.mp4")
    print("  Drag-drop: Just drag your file here and press Enter")
    
    while True:
        print("\n" + "─"*70)
        file_path = input("📂 Enter file path (or 'q' to quit): ").strip().strip('"').strip("'")
        
        if file_path.lower() == 'q':
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        if not file_path:
            print("❌ Please enter a file path!")
            continue
        
        if not os.path.exists(file_path):
            print(f"\n❌ File not found: {file_path}")
            print("Please check the path and try again.")
            continue
        
        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.wav', '.mp3', '.m4a']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"\n⚠️  Warning: File type may not be supported")
            print(f"Supported: {', '.join(valid_extensions)}")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                continue
        
        return file_path

def analyze_file(file_path, predictor):
    """Analyze the file and show results"""
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
    
    print("\n" + "─"*70)
    print("STEP 2: Analyzing...")
    print("─"*70)
    print(f"\n📄 File: {filename}")
    print(f"💾 Size: {file_size:.2f} MB")
    print("\n⏳ Processing... Please wait...\n")
    
    try:
        # Get prediction
        result = predictor.predict(file_path)
        
        # Show results
        print("\n" + "═"*70)
        print("✨ ANALYSIS COMPLETE ✨")
        print("═"*70)
        
        # Main result
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        # Emoji for sentiment
        emoji_map = {
            'Positive': '😊',
            'Negative': '😢',
            'Neutral': '😐'
        }
        emoji = emoji_map.get(sentiment, '🎭')
        
        print(f"\n{emoji}  SENTIMENT: {sentiment.upper()}")
        print(f"📊 CONFIDENCE: {confidence:.2%}")
        
        # Confidence level description
        if confidence > 0.9:
            conf_desc = "Very High - Model is very confident"
        elif confidence > 0.75:
            conf_desc = "High - Strong prediction"
        elif confidence > 0.6:
            conf_desc = "Moderate - Reasonably confident"
        else:
            conf_desc = "Low - Uncertain prediction"
        
        print(f"   └─ {conf_desc}")
        
        # Probability distribution
        print(f"\n📈 DETAILED BREAKDOWN:")
        print("─"*70)
        
        probabilities = result['probabilities']
        
        # Sort by probability (highest first)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for label, prob in sorted_probs:
            # Visual bar
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            # Emoji
            label_emoji = emoji_map.get(label, '🎭')
            
            # Highlight the predicted sentiment
            if label == sentiment:
                print(f"  {label_emoji} {label:10s} │{bar}│ {prob:6.2%} ⭐")
            else:
                print(f"  {label_emoji} {label:10s} │{bar}│ {prob:6.2%}")
        
        # Modality information
        print(f"\n🎬 MODALITY DETECTION:")
        print("─"*70)
        
        has_visual = result.get('has_visual', True)
        has_audio = result.get('has_audio', True)
        
        visual_status = "✅ Detected" if has_visual else "❌ Not detected"
        audio_status = "✅ Detected" if has_audio else "❌ Not detected"
        
        print(f"  📹 Visual: {visual_status}")
        print(f"  🔊 Audio:  {audio_status}")
        
        # Determine analysis type
        if has_visual and has_audio:
            analysis_type = "Multimodal (Video + Audio combined)"
            note = "Best accuracy - using both visual and audio cues"
        elif has_visual:
            analysis_type = "Visual-only (Video frames)"
            note = "Analyzing facial expressions and visual content"
        elif has_audio:
            analysis_type = "Audio-only (Sound/voice)"
            note = "Analyzing tone, pitch, and audio patterns"
        else:
            analysis_type = "Unknown"
            note = "Unable to detect modalities"
        
        print(f"\n  🎯 Analysis Type: {analysis_type}")
        print(f"     💡 {note}")
        
        print("\n" + "═"*70)
        
        return True
        
    except Exception as e:
        print("\n" + "═"*70)
        print("❌ ERROR DURING ANALYSIS")
        print("═"*70)
        print(f"\n⚠️  {str(e)}")
        print("\nPossible issues:")
        print("  • File format not supported")
        print("  • File is corrupted")
        print("  • Model not found (run training first)")
        print("\n" + "═"*70)
        return False

def main():
    """Main program loop"""
    
    print_header()
    
    # Check if model exists
    model_path = 'saved_models/best_model.pth'
    
    if not os.path.exists(model_path):
        print("\n" + "="*70)
        print("❌ ERROR: Trained model not found!")
        print("="*70)
        print(f"\nExpected location: {model_path}")
        print("\n📝 To fix this:")
        print("  1. Train the model first: python train.py")
        print("  2. Wait for training to complete")
        print("  3. Run this script again")
        print("\n" + "="*70)
        sys.exit(1)
    
    # Load model
    print("\n⏳ Loading AI model...")
    try:
        predictor = SentimentPredictor(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Main loop
    while True:
        # Get file from user
        file_path = get_video_path()
        
        # Analyze it
        success = analyze_file(file_path, predictor)
        
        # Ask if user wants to analyze another
        print("\n" + "─"*70)
        another = input("\n🔄 Analyze another file? (y/n): ").strip().lower()
        
        if another != 'y':
            print("\n" + "╔" + "="*68 + "╗")
            print("║" + " "*20 + "THANK YOU FOR USING" + " "*29 + "║")
            print("║" + " "*18 + "SENTIMENT ANALYZER! 🎉" + " "*29 + "║")
            print("╚" + "="*68 + "╝")
            print("\n👋 Goodbye!\n")
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
        sys.exit(0)