"""
Audio Preprocessing Module
Handles audio cleaning, normalization, and feature extraction
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, Dict, Any, List
import os
from pathlib import Path
import torch
import torchaudio
from scipy.signal import butter, filtfilt


class AudioProcessor:
    """Audio preprocessing and feature extraction"""
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 1024, 
                 hop_length: int = 256, n_mels: int = 80, 
                 fmin: int = 0, fmax: Optional[int] = None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
    def load_audio(self, file_path: str, sr: Optional[int] = None, use_fast_loading: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file with fallback methods - optimized for speed"""
        target_sr = sr or self.sample_rate
        
        # Try soundfile first for faster loading (if enabled and file is wav/flac)
        if use_fast_loading:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.wav', '.flac', '.ogg']:
                try:
                    y, sr_read = sf.read(file_path)
                    # Handle stereo by converting to mono
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                    # Resample if needed
                    if sr_read != target_sr:
                        y = librosa.resample(y, orig_sr=sr_read, target_sr=target_sr)
                    return y, target_sr
                except Exception:
                    pass  # Fall through to librosa
        
        # Try librosa (handles most formats including MP3, M4A)
        try:
            y, sr = librosa.load(file_path, sr=target_sr)
            return y, sr
        except Exception as e:
            if not use_fast_loading:  # Only print if not in fast mode
                print(f"Librosa failed to load {file_path}: {e}")
        
        # Try torchaudio as last resort
        try:
            y, sr_read = torchaudio.load(file_path)
            y = y.numpy()
            if len(y.shape) > 1:
                y = y[0] if y.shape[0] == 1 else np.mean(y, axis=0)  # Handle mono/stereo
            else:
                y = y.squeeze()
            # Resample if needed
            if sr_read != target_sr:
                y = librosa.resample(y, orig_sr=sr_read, target_sr=target_sr)
            return y, target_sr
        except Exception as e:
            if not use_fast_loading:
                print(f"Torchaudio failed to load {file_path}: {e}")
        
        if not use_fast_loading:
            print(f"All audio loading methods failed for {file_path}")
        return None, None
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: Optional[int] = None):
        """Save audio to file"""
        try:
            sf.write(file_path, audio, sr or self.sample_rate)
        except Exception as e:
            print(f"Error saving audio {file_path}: {e}")
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
        
        # Calculate target RMS
        target_rms = 10**(target_db / 20.0)
        
        # Normalize
        normalized = audio * (target_rms / rms)
        
        # Clip to prevent overflow
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from audio"""
        return librosa.effects.trim(audio, top_db=top_db)[0]
    
    def apply_preemphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """Apply preemphasis filter"""
        return librosa.effects.preemphasis(audio, coef=coeff)
    
    def apply_bandpass_filter(self, audio: np.ndarray, lowcut: int = 80, 
                            highcut: int = 8000, order: int = 5) -> np.ndarray:
        """Apply bandpass filter to remove noise"""
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, audio)
        
        return filtered
    
    def extract_mel_spectrogram(self, audio: np.ndarray, 
                              sr: Optional[int] = None) -> np.ndarray:
        """Extract mel-spectrogram from audio"""
        sr = sr or self.sample_rate
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_mfcc(self, audio: np.ndarray, sr: Optional[int] = None, 
                    n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio"""
        sr = sr or self.sample_rate
        
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def extract_pitch(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Extract pitch (F0) from audio"""
        sr = sr or self.sample_rate
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=sr, threshold=0.1, fmin=50, fmax=400
        )
        
        # Get the most prominent pitch at each time step
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_values.append(pitch if pitch > 0 else 0)
        
        return np.array(pitch_values)
    
    def extract_energy(self, audio: np.ndarray, frame_length: int = 2048, 
                      hop_length: Optional[int] = None) -> np.ndarray:
        """Extract energy (RMS) from audio"""
        hop_length = hop_length or self.hop_length
        
        # Calculate RMS energy
        energy = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )
        
        return energy.squeeze()
    
    def preprocess_audio(self, audio_path: str, output_path: Optional[str] = None,
                        apply_filters: bool = True, normalize: bool = True,
                        trim_silence: bool = True) -> Dict[str, Any]:
        """Complete audio preprocessing pipeline"""
        # Load audio
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return {"error": "Failed to load audio"}
        
        # Apply preprocessing steps
        if trim_silence:
            audio = self.trim_silence(audio)
        
        if apply_filters:
            audio = self.apply_bandpass_filter(audio)
            audio = self.apply_preemphasis(audio)
        
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio, sr)
        mfcc = self.extract_mfcc(audio, sr)
        pitch = self.extract_pitch(audio, sr)
        energy = self.extract_energy(audio)
        
        # Save processed audio if output path provided
        if output_path:
            self.save_audio(audio, output_path, sr)
        
        return {
            "audio": audio,
            "sample_rate": sr,
            "mel_spectrogram": mel_spec,
            "mfcc": mfcc,
            "pitch": pitch,
            "energy": energy,
            "duration": len(audio) / sr
        }
    
    def batch_preprocess(self, input_dir: str, output_dir: str, 
                        file_extensions: List[str] = ['.wav', '.mp3', '.flac'],
                        **preprocess_kwargs) -> Dict[str, Any]:
        """Batch process audio files in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
        
        print(f"Found {len(audio_files)} audio files to process")
        
        for audio_file in audio_files:
            try:
                # Create output path
                rel_path = audio_file.relative_to(input_path)
                output_file = output_path / rel_path.with_suffix('.wav')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Process audio
                result = self.preprocess_audio(
                    str(audio_file), 
                    str(output_file),
                    **preprocess_kwargs
                )
                
                if "error" not in result:
                    results["processed"] += 1
                    print(f"Processed: {audio_file.name}")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{audio_file.name}: {result['error']}")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{audio_file.name}: {str(e)}")
                print(f"Error processing {audio_file.name}: {e}")
        
        return results


class AudioAugmentation:
    """Audio data augmentation techniques"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add random noise to audio"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def time_shift(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """Randomly shift audio in time"""
        shift = np.random.randint(-int(shift_max * len(audio)), 
                                 int(shift_max * len(audio)))
        return np.roll(audio, shift)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: int = 2) -> np.ndarray:
        """Randomly shift pitch of audio"""
        steps = np.random.randint(-n_steps, n_steps + 1)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
    
    def time_stretch(self, audio: np.ndarray, rate_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Randomly stretch time of audio"""
        rate = np.random.uniform(rate_range[0], rate_range[1])
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def volume_scale(self, audio: np.ndarray, scale_range: Tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
        """Randomly scale volume of audio"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return audio * scale
    
    def apply_augmentation(self, audio: np.ndarray, 
                          augmentation_types: List[str] = None) -> np.ndarray:
        """Apply random augmentations to audio"""
        if augmentation_types is None:
            augmentation_types = ['noise', 'time_shift', 'pitch_shift', 
                                'time_stretch', 'volume_scale']
        
        augmented = audio.copy()
        
        for aug_type in augmentation_types:
            if aug_type == 'noise' and np.random.random() < 0.3:
                augmented = self.add_noise(augmented)
            elif aug_type == 'time_shift' and np.random.random() < 0.3:
                augmented = self.time_shift(augmented)
            elif aug_type == 'pitch_shift' and np.random.random() < 0.3:
                augmented = self.pitch_shift(augmented)
            elif aug_type == 'time_stretch' and np.random.random() < 0.3:
                augmented = self.time_stretch(augmented)
            elif aug_type == 'volume_scale' and np.random.random() < 0.3:
                augmented = self.volume_scale(augmented)
        
        return augmented


def create_audio_processor(sample_rate: int = 22050, **kwargs) -> AudioProcessor:
    """Factory function to create AudioProcessor"""
    return AudioProcessor(sample_rate=sample_rate, **kwargs)


if __name__ == "__main__":
    # Example usage
    processor = create_audio_processor()
    
    # Test with a sample audio file
    # result = processor.preprocess_audio("sample_audio.wav")
    # print("Preprocessing result keys:", result.keys())
    
    # Test augmentation
    augmenter = AudioAugmentation()
    # sample_audio = np.random.randn(22050)  # 1 second of random audio
    # augmented = augmenter.apply_augmentation(sample_audio)
    # print("Augmentation completed")
