
import os
import urllib.request
import zipfile
from tqdm import tqdm
import pandas as pd

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_ravdess_sample():
    """
    Download RAVDESS dataset sample
    Full dataset: https://zenodo.org/record/1188976
    """
    print("=" * 70)
    print("RAVDESS DATASET DOWNLOADER")
    print("=" * 70)
    print("\nThe RAVDESS dataset contains:")
    print("  - 24 professional actors (12 male, 12 female)")
    print("  - 8 emotions: calm, happy, sad, angry, fearful, surprise, disgust, neutral")
    print("  - Speech and song")
    print("\nNote: Full dataset is ~12GB. This script helps you get started.")
    print("\nTo download manually:")
    print("1. Go to: https://zenodo.org/record/1188976")
    print("2. Download: Video_Speech_Actor_*.zip files")
    print("3. Extract to: data/raw_videos/")
    
    choice = input("\nOpen download page in browser? (y/n): ")
    if choice.lower() == 'y':
        import webbrowser
        webbrowser.open('https://zenodo.org/record/1188976')
    
    print("\n" + "=" * 70)

def download_youtube_samples():
    """
    Instructions to download videos from YouTube
    """
    print("=" * 70)
    print("YOUTUBE VIDEO DOWNLOADER")
    print("=" * 70)
    print("\nYou can download videos from YouTube for training.")
    print("\nSearch terms to try:")
    print("  - 'happy people compilation'")
    print("  - 'sad movie scenes'")
    print("  - 'angry reactions'")
    print("  - 'neutral expressions'")
    print("  - 'emotional interviews'")
    
    print("\nMethod 1: Using yt-dlp (Recommended)")
    print("-" * 70)
    print("1. Install yt-dlp:")
    print("   pip install yt-dlp")
    print("\n2. Download a video:")
    print("   yt-dlp -f 'best[height<=480]' -o 'data/raw_videos/%(title)s.%(ext)s' VIDEO_URL")
    print("\n3. Example:")
    print("   yt-dlp -f 'best[height<=480]' -o 'data/raw_videos/happy_video.%(ext)s' 'https://youtube.com/watch?v=...'")
    
    print("\nMethod 2: Online Tools")
    print("-" * 70)
    print("1. Go to: https://yt1s.com/ or https://y2mate.com/")
    print("2. Paste YouTube URL")
    print("3. Download MP4 format")
    print("4. Save to: data/raw_videos/")
    
    print("\n⚠️  Important: Only use videos you have permission to use!")
    print("=" * 70)

def create_sample_dataset():
    """
    Create a small sample dataset using webcam (for testing)
    """
    print("=" * 70)
    print("CREATE SAMPLE DATASET (WEBCAM)")
    print("=" * 70)
    print("\nThis will use your webcam to record sample videos.")
    print("You can record yourself expressing different emotions!")
    
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV not installed!")
        print("Install with: pip install opencv-python")
        return
    
    choice = input("\nProceed with webcam recording? (y/n): ")
    if choice.lower() != 'y':
        return
    
    os.makedirs('data/raw_videos', exist_ok=True)
    
    emotions = {
        '0': 'negative',
        '1': 'neutral', 
        '2': 'positive'
    }
    
    annotations = []
    
    print("\n" + "=" * 70)
    print("RECORDING INSTRUCTIONS")
    print("=" * 70)
    print("For each emotion:")
    print("  1. Get ready to express the emotion")
    print("  2. Press SPACE to start recording (5 seconds)")
    print("  3. Press Q to skip to next emotion")
    print("=" * 70)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    # Get video properties
    fps = 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for label, emotion in emotions.items():
        print(f"\n📹 Recording {emotion.upper()} emotion...")
        print("Press SPACE when ready, Q to skip")
        
        recording = False
        frames = []
        frame_count = 0
        max_frames = fps * 5  # 5 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Display instructions
            if recording:
                cv2.putText(frame, f"RECORDING {emotion.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}/{max_frames}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frames.append(frame)
                frame_count += 1
                
                if frame_count >= max_frames:
                    break
            else:
                cv2.putText(frame, f"Express {emotion.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to record", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press Q to skip", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Record Video', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not recording:
                recording = True
                frames = []
                frame_count = 0
            elif key == ord('q'):
                break
        
        # Save video
        if frames:
            filename = f"{emotion}_{len(annotations)+1}.mp4"
            filepath = os.path.join('data/raw_videos', filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            annotations.append({
                'filename': filename,
                'label': int(label)
            })
            
            print(f"✅ Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save annotations
    if annotations:
        df = pd.DataFrame(annotations)
        
        # Split into train/val/test
        n = len(df)
        train_size = max(1, int(0.7 * n))
        val_size = max(1, int(0.15 * n))
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        train_df.to_csv('data/train_annotations.csv', index=False)
        val_df.to_csv('data/val_annotations.csv', index=False)
        test_df.to_csv('data/test_annotations.csv', index=False)
        
        print("\n" + "=" * 70)
        print("✅ SAMPLE DATASET CREATED!")
        print("=" * 70)
        print(f"Total videos: {len(annotations)}")
        print(f"Training: {len(train_df)}")
        print(f"Validation: {len(val_df)}")
        print(f"Test: {len(test_df)}")

def download_sample_videos():
    """
    Download a few sample videos from public domain
    """
    print("=" * 70)
    print("SAMPLE VIDEO DOWNLOADER")
    print("=" * 70)
    print("\nDownloading sample videos from public domain sources...")
    
    os.makedirs('data/raw_videos', exist_ok=True)
    
    # Note: These are placeholder URLs. In practice, you'd use actual video URLs
    samples = [
        {
            'name': 'sample_positive.mp4',
            'emotion': 'positive',
            'info': 'Happy expression video'
        },
        {
            'name': 'sample_negative.mp4',
            'emotion': 'negative',
            'info': 'Sad expression video'
        },
        {
            'name': 'sample_neutral.mp4',
            'emotion': 'neutral',
            'info': 'Neutral expression video'
        }
    ]
    
    print("\nTo get sample videos:")
    print("\n1. Search on Pexels (free stock videos):")
    print("   https://www.pexels.com/search/videos/happy%20person/")
    print("   https://www.pexels.com/search/videos/sad%20person/")
    print("   https://www.pexels.com/search/videos/neutral%20face/")
    
    print("\n2. Or use Pixabay:")
    print("   https://pixabay.com/videos/search/happy/")
    print("   https://pixabay.com/videos/search/sad/")
    
    print("\n3. Download and save to: data/raw_videos/")
    
    choice = input("\nOpen Pexels in browser? (y/n): ")
    if choice.lower() == 'y':
        import webbrowser
        webbrowser.open('https://www.pexels.com/search/videos/emotions/')

def main():
    """Main menu"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "VIDEO DATASET SOURCES" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print("\nChoose a data source:")
    print("\n1. Download RAVDESS Dataset (Professional, Pre-labeled)")
    print("   - Best for: Serious projects")
    print("   - Size: 12GB")
    print("   - Quality: High")
    
    print("\n2. Download from YouTube (Custom)")
    print("   - Best for: Specific use cases")
    print("   - Size: Varies")
    print("   - Quality: Varies")
    
    print("\n3. Record with Webcam (Quick Testing)")
    print("   - Best for: Quick testing/demo")
    print("   - Size: Small")
    print("   - Quality: Depends on webcam")
    
    print("\n4. Download Sample Stock Videos (Free)")
    print("   - Best for: Getting started")
    print("   - Size: Small")
    print("   - Quality: Good")
    
    print("\n5. Exit")
    
    choice = input("\nYour choice (1-5): ").strip()
    
    if choice == '1':
        download_ravdess_sample()
    elif choice == '2':
        download_youtube_samples()
    elif choice == '3':
        create_sample_dataset()
    elif choice == '4':
        download_sample_videos()
    else:
        print("Exiting...")

if __name__ == '__main__':
    main()