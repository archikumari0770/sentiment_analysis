

import os
import pandas as pd
from pathlib import Path

def list_videos(video_dir):
    """List all video files in the directory"""
    video_extensions = ['.mp4', '.avi', '.mov']
    videos = []
    
    if not os.path.exists(video_dir):
        print(f"❌ Directory not found: {video_dir}")
        return videos
    
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            videos.append(file)
    
    return sorted(videos)

def interactive_annotation():
    """Interactive annotation creation"""
    
    video_dir = 'data/raw_videos'
    videos = list_videos(video_dir)
    
    if not videos:
        print("❌ No videos found in data/raw_videos/")
        print("Please place your video files there first!")
        return
    
    print("=" * 70)
    print("VIDEO ANNOTATION CREATOR")
    print("=" * 70)
    print(f"\nFound {len(videos)} videos:\n")
    
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video}")
    
    print("\n" + "=" * 70)
    print("SENTIMENT LABELS:")
    print("  0 = Negative (sad, angry, upset, frustrated)")
    print("  1 = Neutral (calm, expressionless, indifferent)")
    print("  2 = Positive (happy, excited, joyful, pleased)")
    print("=" * 70)
    
    annotations = []
    
    print("\nLet's annotate each video!")
    print("(Press Ctrl+C to stop and save what you've done)\n")
    
    try:
        for video in videos:
            while True:
                label = input(f"Label for '{video}' (0/1/2): ").strip()
                if label in ['0', '1', '2']:
                    annotations.append({
                        'filename': video,
                        'label': int(label)
                    })
                    label_name = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[int(label)]
                    print(f"  ✓ Saved as: {label_name}\n")
                    break
                else:
                    print("  ⚠️  Please enter 0, 1, or 2")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving current progress...")
    
    if not annotations:
        print("❌ No annotations created!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(annotations)
    
    # Split into train/val/test
    n = len(df)
    train_size = int(0.7 * n)  # 70% training
    val_size = int(0.15 * n)   # 15% validation
    # Rest (15%) for testing
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    # Ensure at least one sample in val and test if possible
    if len(train_df) == 0:
        train_df = df
        val_df = pd.DataFrame(columns=['filename', 'label'])
        test_df = pd.DataFrame(columns=['filename', 'label'])
    elif len(val_df) == 0 and len(df) > 1:
        val_df = train_df.iloc[-1:]
        train_df = train_df.iloc[:-1]
    
    # Save files
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_annotations.csv', index=False)
    val_df.to_csv('data/val_annotations.csv', index=False)
    test_df.to_csv('data/test_annotations.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ ANNOTATION FILES CREATED!")
    print("=" * 70)
    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples:       {len(test_df)}")
    print("\nFiles created:")
    print("  - data/train_annotations.csv")
    print("  - data/val_annotations.csv")
    print("  - data/test_annotations.csv")
    
    # Show distribution
    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION:")
    print("=" * 70)
    
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    if len(train_df) > 0:
        print("\nTraining set:")
        for label, count in train_df['label'].value_counts().sort_index().items():
            print(f"  {label_names[label]}: {count} videos")
    
    if len(val_df) > 0:
        print("\nValidation set:")
        for label, count in val_df['label'].value_counts().sort_index().items():
            print(f"  {label_names[label]}: {count} videos")
    
    if len(test_df) > 0:
        print("\nTest set:")
        for label, count in test_df['label'].value_counts().sort_index().items():
            print(f"  {label_names[label]}: {count} videos")

def quick_annotation():
    """Quick annotation for all videos with same label (for testing)"""
    
    video_dir = 'data/raw_videos'
    videos = list_videos(video_dir)
    
    if not videos:
        print("❌ No videos found!")
        return
    
    print(f"Found {len(videos)} videos")
    print("\nQuick annotation mode - assign same label to all videos")
    print("(Use this only for quick testing!)\n")
    
    label = input("Label for ALL videos (0=Negative, 1=Neutral, 2=Positive): ").strip()
    
    if label not in ['0', '1', '2']:
        print("❌ Invalid label!")
        return
    
    label = int(label)
    label_name = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[label]
    
    annotations = [{'filename': video, 'label': label} for video in videos]
    df = pd.DataFrame(annotations)
    
    # Split
    n = len(df)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_df = df.iloc[:train_size] if train_size > 0 else df
    val_df = df.iloc[train_size:train_size + val_size] if val_size > 0 else pd.DataFrame(columns=['filename', 'label'])
    test_df = df.iloc[train_size + val_size:] if len(df) > train_size + val_size else pd.DataFrame(columns=['filename', 'label'])
    
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train_annotations.csv', index=False)
    val_df.to_csv('data/val_annotations.csv', index=False)
    test_df.to_csv('data/test_annotations.csv', index=False)
    
    print(f"\n✅ Created annotations with all videos labeled as: {label_name}")
    print(f"   Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

def manual_csv_guide():
    """Show how to create CSV manually"""
    
    print("=" * 70)
    print("MANUAL CSV CREATION GUIDE")
    print("=" * 70)
    
    print("\nIf you prefer to create CSV files manually:")
    print("\n1. Create data/train_annotations.csv with this format:")
    print("-" * 70)
    print("filename,label")
    print("happy_video1.mp4,2")
    print("sad_video1.mp4,0")
    print("neutral_video1.mp4,1")
    print("happy_video2.mp4,2")
    print("-" * 70)
    
    print("\n2. Create data/val_annotations.csv:")
    print("-" * 70)
    print("filename,label")
    print("happy_test.mp4,2")
    print("sad_test.mp4,0")
    print("-" * 70)
    
    print("\n3. Create data/test_annotations.csv:")
    print("-" * 70)
    print("filename,label")
    print("neutral_test.mp4,1")
    print("-" * 70)
    
    print("\nLabel meanings:")
    print("  0 = Negative (sad, angry, upset)")
    print("  1 = Neutral (calm, expressionless)")
    print("  2 = Positive (happy, excited, joyful)")
    
    print("\nSave each file in the 'data/' folder")
    print("=" * 70)

def main():
    print("\n" + "=" * 70)
    print("ANNOTATION FILE CREATOR")
    print("=" * 70)
    print("\nChoose annotation mode:")
    print("  1. Interactive (recommended) - Label each video individually")
    print("  2. Quick (testing only) - Same label for all videos")
    print("  3. Manual CSV Guide - Create CSV files yourself")
    print("  4. Exit")
    
    choice = input("\nYour choice (1/2/3/4): ").strip()
    
    if choice == '1':
        interactive_annotation()
    elif choice == '2':
        quick_annotation()
    elif choice == '3':
        manual_csv_guide()
    else:
        print("Exiting...")

if __name__ == '__main__':
    main()