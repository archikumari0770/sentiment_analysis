import os
import urllib.request
import zipfile
import shutil

def setup_portable_ffmpeg():
    """Downloads and sets up FFmpeg in your project directory."""
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_dir = os.path.join(project_dir, 'ffmpeg_portable')
    
    print("Setting up portable FFmpeg...")
    
    # Check if already installed
    ffmpeg_exe = os.path.join(ffmpeg_dir, 'bin', 'ffmpeg.exe')
    if os.path.exists(ffmpeg_exe):
        print("✓ FFmpeg already installed!")
        os.environ['FFMPEG_BINARY'] = ffmpeg_exe
        return ffmpeg_dir
    
    os.makedirs(ffmpeg_dir, exist_ok=True)
    
    # Download
    download_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    zip_path = os.path.join(ffmpeg_dir, 'ffmpeg.zip')
    
    print("Downloading FFmpeg... (this may take 2-3 minutes)")
    try:
        urllib.request.urlretrieve(download_url, zip_path)
        print("✓ Download complete!")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ffmpeg_dir)
    
    # Find and move bin directory
    for root, dirs, files in os.walk(ffmpeg_dir):
        if 'ffmpeg.exe' in files:
            bin_dir = root
            target_bin = os.path.join(ffmpeg_dir, 'bin')
            if bin_dir != target_bin:
                os.makedirs(target_bin, exist_ok=True)
                for file in os.listdir(bin_dir):
                    src = os.path.join(bin_dir, file)
                    dst = os.path.join(target_bin, file)
                    if os.path.isfile(src):
                        shutil.move(src, dst)
            break
    
    os.remove(zip_path)
    os.environ['FFMPEG_BINARY'] = ffmpeg_exe
    print("✓ FFmpeg installed successfully!")
    return ffmpeg_dir

if __name__ == '__main__':
    setup_portable_ffmpeg()