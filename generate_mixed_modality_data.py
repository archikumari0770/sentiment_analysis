"""
Generate mixed modality dataset:
- Video-only samples (visual features, no audio)
- Audio-only samples (audio features, no video)
- Multimodal samples (both video and audio)

This trains the model to handle ANY combination of inputs!
"""

import cv2
import numpy as np
import os
from scipy.io import wavfile
import pandas as pd
from tqdm import tqdm
import random

def generate_audio_emotion(emotion, duration=5, sample_rate=16000):
    """Generate audio for emotion"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    if emotion == 'positive':
        # Happy: High pitch, varied
        base_freq = 440
        audio += 0.3 * np.sin(2 * np.pi * base_freq * t)
        audio += 0.15 * np.sin(2 * np.pi * base_freq * 1.5 * t)
        rhythm = np.sin(2 * np.pi * 4 * t)
        audio = audio * (0.7 + 0.3 * np.abs(rhythm))
        
    elif emotion == 'negative':
        # Sad: Low pitch, slow
        base_freq = 220
        audio += 0.25 * np.sin(2 * np.pi * base_freq * t)
        audio += 0.15 * np.sin(2 * np.pi * base_freq * 1.2 * t)
        tremolo = np.sin(2 * np.pi * 2 * t)
        audio = audio * (0.8 + 0.2 * tremolo)
        
    else:  # neutral
        base_freq = 330
        audio += 0.2 * np.sin(2 * np.pi * base_freq * t)
    
    # Add noise
    audio += np.random.normal(0, 0.02, audio.shape)
    audio = audio / np.max(np.abs(audio))
    return (audio * 0.7 * 32767).astype(np.int16)

def create_visual_frame(emotion, width, height, frame_num):
    """Create visual frame for emotion"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    phase = (frame_num % 60) / 60.0
    
    if emotion == 'positive':
        # Bright yellow/orange
        frame[:] = (100, 220, 255)
        
        # Happy face
        face_y = int(height/2 + 10 * np.sin(phase * 2 * np.pi * 2))
        cv2.circle(frame, (width//2, face_y), 80, (50, 200, 255), -1)
        cv2.circle(frame, (width//2 - 25, face_y - 20), 12, (0, 0, 0), -1)
        cv2.circle(frame, (width//2 + 25, face_y - 20), 12, (0, 0, 0), -1)
        cv2.ellipse(frame, (width//2, face_y + 20), (40, 25), 0, 0, 180, (0, 0, 0), 3)
        
        cv2.putText(frame, 'HAPPY', (width//2 - 60, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 180, 0), 3)
        
    elif emotion == 'negative':
        # Dark blue/grey
        frame[:] = (100, 80, 60)
        
        # Sad face
        face_y = int(height/2 - 10 * np.sin(phase * 2 * np.pi))
        cv2.circle(frame, (width//2, face_y), 80, (150, 120, 100), -1)
        cv2.circle(frame, (width//2 - 25, face_y - 20), 12, (0, 0, 0), -1)
        cv2.circle(frame, (width//2 + 25, face_y - 20), 12, (0, 0, 0), -1)
        cv2.ellipse(frame, (width//2, face_y + 30), (40, 25), 0, 180, 360, (0, 0, 0), 3)
        
        # Tears
        tear_y = int(face_y + phase * 60)
        cv2.line(frame, (width//2 - 25, face_y), (width//2 - 25, tear_y), (200, 200, 255), 3)
        
        cv2.putText(frame, 'SAD', (width//2 - 50, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 3)
        
    else:  # neutral
        # Grey
        frame[:] = (140, 140, 140)
        
        # Neutral face
        cv2.circle(frame, (width//2, height//2), 80, (160, 160, 160), -1)
        cv2.circle(frame, (width//2 - 25, height//2 - 20), 12, (0, 0, 0), -1)
        cv2.circle(frame, (width//2 + 25, height//2 - 20), 12, (0, 0, 0), -1)
        cv2.line(frame, (width//2 - 35, height//2 + 20), 
                (width//2 + 35, height//2 + 20), (0, 0, 0), 3)
        
        cv2.putText(frame, 'NEUTRAL', (width//2 - 90, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
    
    # Add noise
    noise = np.random.randint(-15, 15, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return frame

def create_video_only(filename, emotion, duration=5, fps=20):
    """Create video with NO AUDIO"""
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    for frame_num in range(total_frames):
        frame = create_visual_frame(emotion, width, height, frame_num)
        # Add "VIDEO ONLY" watermark
        cv2.putText(frame, 'VIDEO ONLY', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        video_writer.write(frame)
    
    video_writer.release()
    return True

def create_audio_only(filename, emotion, duration=5):
    """Create audio file with BLACK VIDEO (audio only)"""
    width, height = 640, 480
    fps = 20
    
    # Create black video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add "AUDIO ONLY" text
    cv2.putText(black_frame, 'AUDIO ONLY', (width//2 - 120, height//2),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    total_frames = duration * fps
    for _ in range(total_frames):
        video_writer.write(black_frame)
    
    video_writer.release()
    
    # Add audio
    audio = generate_audio_emotion(emotion, duration)
    audio_temp = filename.replace('.mp4', '_temp.wav')
    wavfile.write(audio_temp, 16000, audio)
    
    # Merge
    ffmpeg_path = os.path.join('ffmpeg_portable', 'bin', 'ffmpeg.exe')
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = 'ffmpeg'
    
    output_final = filename.replace('.mp4', '_final.mp4')
    
    import subprocess
    try:
        subprocess.run([
            ffmpeg_path, '-i', filename, '-i', audio_temp,
            '-c:v', 'copy', '-c:a', 'aac', '-y', output_final
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        if os.path.exists(output_final):
            os.remove(filename)
            os.rename(output_final, filename)
            os.remove(audio_temp)
            return True
    except:
        pass
    
    if os.path.exists(audio_temp):
        os.remove(audio_temp)
    return False

def create_multimodal(filename, emotion, duration=5, fps=20):
    """Create video WITH AUDIO (both modalities)"""
    width, height = 640, 480
    
    # Create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    for frame_num in range(total_frames):
        frame = create_visual_frame(emotion, width, height, frame_num)
        # Add "VIDEO + AUDIO" watermark
        cv2.putText(frame, 'MULTIMODAL', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        video_writer.write(frame)
    
    video_writer.release()
    
    # Add audio
    audio = generate_audio_emotion(emotion, duration)
    audio_temp = filename.replace('.mp4', '_temp.wav')
    wavfile.write(audio_temp, 16000, audio)
    
    # Merge
    ffmpeg_path = os.path.join('ffmpeg_portable', 'bin', 'ffmpeg.exe')
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = 'ffmpeg'
    
    output_final = filename.replace('.mp4', '_final.mp4')
    
    import subprocess
    try:
        subprocess.run([
            ffmpeg_path, '-i', filename, '-i', audio_temp,
            '-c:v', 'copy', '-c:a', 'aac', '-y', output_final
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        if os.path.exists(output_final):
            os.remove(filename)
            os.rename(output_final, filename)
            os.remove(audio_temp)
            return True
    except:
        pass
    
    if os.path.exists(audio_temp):
        os.remove(audio_temp)
    return False

def generate_mixed_dataset(num_per_type=5):
    """
    Generate dataset with three types:
    - Video-only samples
    - Audio-only samples
    - Multimodal samples
    
    Args:
        num_per_type: Number of samples per emotion per type
                     Total = num_per_type * 3 emotions * 3 types
    """
    print("\n" + "=" * 70)
    print("MIXED MODALITY DATASET GENERATOR")
    print("=" * 70)
    
    total = num_per_type * 3 * 3  # emotions * types
    print(f"\nGenerating {total} videos:")
    print(f"  • {num_per_type * 3} Video-only (visual features)")
    print(f"  • {num_per_type * 3} Audio-only (audio features)")
    print(f"  • {num_per_type * 3} Multimodal (both modalities)")
    print("\nThis may take several minutes...\n")
    
    os.makedirs('data/raw_videos', exist_ok=True)
    
    emotions = {'positive': 2, 'negative': 0, 'neutral': 1}
    all_videos = []
    
    # Generate video-only samples
    print("1️⃣  Creating VIDEO-ONLY samples...")
    for emotion, label in emotions.items():
        for i in tqdm(range(num_per_type), desc=f"    {emotion.capitalize()}"):
            filename = os.path.join('data/raw_videos', f'video_only_{emotion}_{i+1:03d}.mp4')
            create_video_only(filename, emotion)
            all_videos.append({
                'filename': os.path.basename(filename),
                'label': label,
                'type': 'video_only'
            })
    
    # Generate audio-only samples
    print("\n2️⃣  Creating AUDIO-ONLY samples...")
    for emotion, label in emotions.items():
        for i in tqdm(range(num_per_type), desc=f"    {emotion.capitalize()}"):
            filename = os.path.join('data/raw_videos', f'audio_only_{emotion}_{i+1:03d}.mp4')
            create_audio_only(filename, emotion)
            all_videos.append({
                'filename': os.path.basename(filename),
                'label': label,
                'type': 'audio_only'
            })
    
    # Generate multimodal samples
    print("\n3️⃣  Creating MULTIMODAL samples...")
    for emotion, label in emotions.items():
        for i in tqdm(range(num_per_type), desc=f"    {emotion.capitalize()}"):
            filename = os.path.join('data/raw_videos', f'multimodal_{emotion}_{i+1:03d}.mp4')
            create_multimodal(filename, emotion)
            all_videos.append({
                'filename': os.path.basename(filename),
                'label': label,
                'type': 'multimodal'
            })
    
    # Create annotations
    df = pd.DataFrame(all_videos)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_df = df.iloc[:train_size][['filename', 'label']]
    val_df = df.iloc[train_size:train_size + val_size][['filename', 'label']]
    test_df = df.iloc[train_size + val_size:][['filename', 'label']]
    
    train_df.to_csv('data/train_annotations.csv', index=False)
    val_df.to_csv('data/val_annotations.csv', index=False)
    test_df.to_csv('data/test_annotations.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ MIXED MODALITY DATASET CREATED!")
    print("=" * 70)
    print(f"\nTotal videos: {len(df)}")
    print(f"  Training:   {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test:       {len(test_df)}")
    
    print("\nBy modality type:")
    for mod_type in ['video_only', 'audio_only', 'multimodal']:
        count = len(df[df['type'] == mod_type])
        print(f"  {mod_type}: {count}")
    
    print("\nBy emotion:")
    for emotion, label in emotions.items():
        count = len(df[df['label'] == label])
        print(f"  {emotion.capitalize()}: {count}")
    
    print("\n✓ Files created:")
    print("  • data/train_annotations.csv")
    print("  • data/val_annotations.csv")
    print("  • data/test_annotations.csv")
    print(f"  • data/raw_videos/ ({len(df)} videos)")
    
    print("\n" + "=" * 70)
    print("🎓 HOW IT WORKS")
    print("=" * 70)
    print("\nThe model will learn to:")
    print("  ✓ Analyze emotion from VIDEO ONLY")
    print("  ✓ Analyze emotion from AUDIO ONLY")
    print("  ✓ Combine BOTH when available")
    print("\nThis makes it robust to missing modalities!")
    
    print("\n" + "=" * 70)
    print("🚀 READY TO TRAIN!")
    print("=" * 70)
    print("\nRun:")
    print("  1. python verify_setup.py")
    print("  2. python train.py")
    print("\n")

if __name__ == '__main__':
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "MIXED MODALITY DATA GENERATOR" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print("\nSelect dataset size:")
    print("\n  1. TINY   - 27 videos (3 per emotion per type)  ~3 min")
    print("  2. SMALL  - 45 videos (5 per emotion per type)  ~5 min")
    print("  3. MEDIUM - 90 videos (10 per emotion per type) ~10 min")
    print("  4. CUSTOM - Specify your own amount")
    
    choice = input("\nYour choice (1-4): ").strip()
    
    sizes = {'1': 3, '2': 5, '3': 10}
    
    if choice in sizes:
        generate_mixed_dataset(sizes[choice])
    elif choice == '4':
        try:
            num = int(input("Samples per emotion per type: "))
            if 1 <= num <= 30:
                generate_mixed_dataset(num)
            else:
                print("Please enter 1-30")
        except:
            print("Invalid input!")
    else:
        print("Invalid choice. Running default (5 per type)...")
        generate_mixed_dataset(5)
        