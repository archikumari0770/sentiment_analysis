import os

# Check if FFmpeg is available
ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg_portable', 'bin', 'ffmpeg.exe')

if os.path.exists(ffmpeg_path):
    print(f"✓ FFmpeg found at: {ffmpeg_path}")
    
    # Test it
    os.system(f'"{ffmpeg_path}" -version')
else:
    print("✗ FFmpeg not found!")
    print("Please run: python setup_ffmpeg.py")