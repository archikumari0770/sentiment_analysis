

import os
import pandas as pd
import sys

def check_directories():
    """Check if all required directories exist"""
    print("=" * 70)
    print("CHECKING DIRECTORIES")
    print("=" * 70)
    
    required_dirs = [
        'data',
        'data/raw_videos',
        'data/processed',
        'models',
        'utils',
        'saved_models'
    ]
    
    all_good = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"❌ {directory} - MISSING!")
            all_good = False
            # Try to create it
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   → Created {directory}")
            except:
                pass
    
    return all_good

def check_videos():
    """Check if videos are present"""
    print("\n" + "=" * 70)
    print("CHECKING VIDEOS")
    print("=" * 70)
    
    video_dir = 'data/raw_videos'
    if not os.path.exists(video_dir):
        print(f"❌ Directory not found: {video_dir}")
        return False
    
    video_extensions = ['.mp4', '.avi', '.mov']
    videos = [f for f in os.listdir(video_dir) 
              if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not videos:
        print(f"❌ No videos found in {video_dir}")
        print("\nPlease add video files (.mp4, .avi, .mov)")
        print("\nWays to get videos:")
        print("  1. Download from: https://www.pexels.com/videos/")
        print("  2. Run: python download_datasets.py")
        return False
    
    print(f"✓ Found {len(videos)} videos:")
    for i, video in enumerate(videos[:5], 1):  # Show first 5
        file_size = os.path.getsize(os.path.join(video_dir, video)) / (1024*1024)
        print(f"  {i}. {video} ({file_size:.1f} MB)")
    
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")
    
    return True

def check_annotations():
    """Check if annotation files exist and are valid"""
    print("\n" + "=" * 70)
    print("CHECKING ANNOTATIONS")
    print("=" * 70)
    
    annotation_files = [
        'data/train_annotations.csv',
        'data/val_annotations.csv',
        'data/test_annotations.csv'
    ]
    
    all_good = True
    for file in annotation_files:
        if not os.path.exists(file):
            print(f"❌ {file} - MISSING!")
            all_good = False
            continue
        
        try:
            df = pd.read_csv(file)
            
            # Check required columns
            if 'filename' not in df.columns or 'label' not in df.columns:
                print(f"❌ {file} - Missing required columns (filename, label)")
                all_good = False
                continue
            
            # Check if videos exist
            video_dir = 'data/raw_videos'
            missing_videos = []
            for video in df['filename']:
                if not os.path.exists(os.path.join(video_dir, video)):
                    missing_videos.append(video)
            
            if missing_videos:
                print(f"⚠️  {file} - {len(missing_videos)} videos not found:")
                for video in missing_videos[:3]:
                    print(f"     • {video}")
                if len(missing_videos) > 3:
                    print(f"     ... and {len(missing_videos) - 3} more")
            
            # Check labels
            valid_labels = [0, 1, 2]
            invalid_labels = df[~df['label'].isin(valid_labels)]
            if len(invalid_labels) > 0:
                print(f"⚠️  {file} - {len(invalid_labels)} invalid labels found")
                all_good = False
                continue
            
            # Show distribution
            label_counts = df['label'].value_counts().sort_index()
            label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            
            print(f"✓ {file}")
            print(f"  Total: {len(df)} videos")
            for label, count in label_counts.items():
                print(f"  {label_names[label]}: {count}")
            
        except Exception as e:
            print(f"❌ {file} - Error: {e}")
            all_good = False
    
    if not all_good:
        print("\n💡 To create annotations, run: python create_annotations.py")
    
    return all_good

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'cv2': 'opencv-python',
        'librosa': 'librosa',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'moviepy': 'moviepy',
        'PIL': 'pillow',
        'tqdm': 'tqdm'
    }
    
    all_good = True
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED!")
            all_good = False
            missing.append(package_name)
    
    if missing:
        print("\n💡 To install missing packages:")
        print(f"   pip install {' '.join(missing)}")
    
    return all_good

def check_ffmpeg():
    """Check if FFmpeg is available"""
    print("\n" + "=" * 70)
    print("CHECKING FFMPEG")
    print("=" * 70)
    
    # Check portable FFmpeg
    ffmpeg_path = os.path.join('ffmpeg_portable', 'bin', 'ffmpeg.exe')
    
    if os.path.exists(ffmpeg_path):
        print(f"✓ Portable FFmpeg found: {ffmpeg_path}")
        return True
    else:
        print(f"❌ Portable FFmpeg not found!")
        print("\n💡 To install:")
        print("   python setup_ffmpeg.py")
        return False

def check_python_files():
    """Check if all required Python files exist"""
    print("\n" + "=" * 70)
    print("CHECKING PYTHON FILES")
    print("=" * 70)
    
    required_files = {
        'config.py': 'Configuration file',
        'train.py': 'Training script',
        'inference.py': 'Inference script',
        'models/visual_model.py': 'Visual model',
        'models/audio_model.py': 'Audio model',
        'models/fusion_model.py': 'Fusion model',
        'utils/video_processor.py': 'Video processor',
        'utils/audio_processor.py': 'Audio processor',
        'utils/data_loader.py': 'Data loader'
    }
    
    all_good = True
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} - MISSING!")
            all_good = False
    
    if not all_good:
        print("\n💡 Create missing files from the provided code artifacts")
    
    return all_good

def check_gpu():
    """Check GPU availability"""
    print("\n" + "=" * 70)
    print("CHECKING GPU")
    print("=" * 70)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU Available: {gpu_name}")
            print(f"  Memory: {gpu_memory:.2f} GB")
            
            # Test GPU
            try:
                test_tensor = torch.randn(100, 100).cuda()
                del test_tensor
                print("  GPU test: ✓ Working")
            except:
                print("  GPU test: ⚠️  Failed to allocate memory")
            
            return True
        else:
            print("⚠️  No GPU detected - will use CPU (slower)")
            print("  Training will take longer but will still work")
            print("\n💡 For faster training:")
            print("   - Use Google Colab (free GPU)")
            print("   - Or install CUDA if you have NVIDIA GPU")
            return True
    except:
        print("❌ Cannot check GPU - torch not installed")
        return False

def estimate_training_time():
    """Estimate training time based on data"""
    print("\n" + "=" * 70)
    print("ESTIMATED TRAINING TIME")
    print("=" * 70)
    
    try:
        import torch
        
        # Count training samples
        train_file = 'data/train_annotations.csv'
        if os.path.exists(train_file):
            df = pd.read_csv(train_file)
            n_samples = len(df)
            
            # Rough estimates
            if torch.cuda.is_available():
                time_per_epoch_min = n_samples * 0.5 / 60  # ~0.5 sec per sample on GPU
                device = "GPU"
            else:
                time_per_epoch_min = n_samples * 2 / 60  # ~2 sec per sample on CPU
                device = "CPU"
            
            total_time = time_per_epoch_min * 20  # Assume 20 epochs
            
            print(f"Training samples: {n_samples}")
            print(f"Device: {device}")
            print(f"Estimated time per epoch: ~{time_per_epoch_min:.1f} minutes")
            print(f"Estimated total time (20 epochs): ~{total_time:.1f} minutes ({total_time/60:.1f} hours)")
            
            if device == "CPU" and n_samples > 50:
                print("\n⚠️  Warning: Training on CPU with large dataset will be very slow")
                print("   Consider using Google Colab for free GPU access")
        else:
            print("Cannot estimate - no training annotations found")
    except:
        print("Cannot estimate - missing data or torch")

def main():
    """Run all checks"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "SETUP VERIFICATION" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    
    checks = [
        ("Directories", check_directories),
        ("Python Files", check_python_files),
        ("Videos", check_videos),
        ("Annotations", check_annotations),
        ("Dependencies", check_dependencies),
        ("FFmpeg", check_ffmpeg),
        ("GPU", check_gpu)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results[name] = False
    
    # Estimate training time
    estimate_training_time()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Critical checks (must pass)
    critical_checks = ["Python Files", "Videos", "Annotations", "Dependencies", "FFmpeg"]
    critical_passed = all(results.get(check, False) for check in critical_checks)
    
    if critical_passed:
        print("✅ ALL CRITICAL CHECKS PASSED!")
        print("\n🚀 You're ready to start training:")
        print("   python train.py")
    else:
        print("⚠️  SOME CRITICAL CHECKS FAILED!")
        print("\nPlease fix the issues above before training.")
        
        failed = [name for name, passed in results.items() if not passed and name in critical_checks]
        if failed:
            print("\nFailed checks:")
            for name in failed:
                print(f"  • {name}")
    
    print("\n")

if __name__ == '__main__':
    main()