import librosa
import numpy as np
import torch
import os
import subprocess
from config import config

class AudioProcessor:
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.duration = config.AUDIO_DURATION
        self._setup_ffmpeg()
    
    def _setup_ffmpeg(self):
        """Setup portable FFmpeg if available"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        
        self.ffmpeg_path = os.path.join(project_dir, 'ffmpeg_portable', 'bin', 'ffmpeg.exe')
        
        if os.path.exists(self.ffmpeg_path):
            print(f"✓ Using FFmpeg for audio: {self.ffmpeg_path}")
        else:
            print(f"⚠️  Portable FFmpeg not found")
            self.ffmpeg_path = 'ffmpeg'
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """Extract audio using FFmpeg directly (no MoviePy needed)"""
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = base + '_audio.wav'
        
        try:
            # Use FFmpeg command directly
            command = [
                self.ffmpeg_path,
                '-i', video_path,           # Input video
                '-vn',                      # No video
                '-acodec', 'pcm_s16le',    # Audio codec
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',                 # Mono (1 channel)
                '-y',                       # Overwrite output
                output_path
            ]
            
            # Run FFmpeg (suppress output)
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Check if extraction succeeded
            if result.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    return output_path
                else:
                    print(f"⚠️  Audio file too small: {os.path.basename(video_path)}")
                    return None
            else:
                print(f"✗ FFmpeg failed for: {os.path.basename(video_path)}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout extracting: {os.path.basename(video_path)}")
            return None
        except FileNotFoundError:
            print(f"✗ FFmpeg not found! Run: python setup_ffmpeg.py")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def load_audio(self, audio_path):
        """Load audio file using librosa"""
        try:
            audio, sr = librosa.load(
                audio_path, 
                sr=self.sample_rate, 
                duration=self.duration,
                mono=True
            )
            return audio
        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            return np.zeros(self.sample_rate * self.duration, dtype=np.float32)
    
    def pad_or_truncate(self, audio):
        """Pad or truncate audio to fixed length"""
        target_length = self.sample_rate * self.duration
        
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        return audio
    
    def process_audio(self, video_path):
        """Complete audio processing pipeline"""
        # Extract audio from video
        audio_path = self.extract_audio_from_video(video_path)
        
        if audio_path is None:
            # Return silence if extraction fails
            print(f"⚠️  Using silence for: {os.path.basename(video_path)}")
            return torch.zeros(1, self.sample_rate * self.duration, dtype=torch.float32)
        
        # Load and process audio
        audio = self.load_audio(audio_path)
        audio = self.pad_or_truncate(audio)
        
        # Clean up temporary audio file
        if os.path.exists(audio_path) and '_audio.wav' in audio_path:
            try:
                os.remove(audio_path)
            except:
                pass  # Ignore cleanup errors
        
        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)