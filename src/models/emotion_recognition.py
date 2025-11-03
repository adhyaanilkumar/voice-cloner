"""
Emotion Recognition Model
CNN/RNN-based emotion classifier for happy, sad, and neutral tones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import librosa


class EmotionCNN(nn.Module):
    """CNN-based emotion recognition model for mel-spectrograms"""
    
    def __init__(self, n_mels: int = 80, num_classes: int = 3, dropout: float = 0.5):
        super(EmotionCNN, self).__init__()
        
        self.n_mels = n_mels
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 2))
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # x shape: (batch_size, 1, time_steps, n_mels)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class EmotionRNN(nn.Module):
    """RNN-based emotion recognition model for mel-spectrograms"""
    
    def __init__(self, n_mels: int = 80, hidden_size: int = 128, num_layers: int = 2, 
                 num_classes: int = 3, dropout: float = 0.5, bidirectional: bool = True):
        super(EmotionRNN, self).__init__()
        
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # x shape: (batch_size, time_steps, n_mels)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM
        # lstm_out shape: (batch_size, time_steps, hidden_size * 2)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class EmotionRecognizer:
    """Emotion recognition wrapper class"""
    
    def __init__(self, model_path: str = None, model_type: str = "cnn", device: str = "cpu"):
        self.device = torch.device(device)
        self.model_type = model_type
        # Emotion labels matching training: {'neutral': 0, 'happy': 1, 'sad': 2}
        self.emotion_labels = {0: "neutral", 1: "happy", 2: "sad"}
        
        # Initialize model
        if model_type == "cnn":
            self.model = EmotionCNN()
        elif model_type == "rnn":
            self.model = EmotionRNN()
        else:
            raise ValueError("model_type must be 'cnn' or 'rnn'")
        
        self.model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded emotion recognition model from {model_path}")
    
    def preprocess_audio(self, audio_path: str, sr: int = 22050, n_mels: int = 80, 
                        n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
        """Extract mel-spectrogram from audio file"""
        try:
            # Try librosa first
            try:
                y, sr = librosa.load(audio_path, sr=sr)
            except Exception as load_error:
                print(f"Librosa failed to load {audio_path}: {load_error}")
                # Try soundfile as fallback
                try:
                    import soundfile as sf
                    y, sr = sf.read(audio_path)
                    # Resample if needed
                    if sr != 22050:
                        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                        sr = 22050
                except Exception as sf_error:
                    print(f"Soundfile failed: {sf_error}")
                    # Try torchaudio as last resort
                    try:
                        import torchaudio
                        y, sr = torchaudio.load(audio_path)
                        y = y.numpy()
                        if len(y.shape) > 1:
                            y = y[0]  # Take first channel if stereo
                        # Resample if needed
                        if sr != 22050:
                            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                            sr = 22050
                    except Exception as ta_error:
                        print(f"Torchaudio failed: {ta_error}")
                        return None
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return log_mel_spec.T  # Transpose to (time_steps, n_mels)
            
        except Exception as e:
            print(f"Error preprocessing audio {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_emotion(self, audio_path: str) -> Dict[str, Any]:
        """Predict emotion from audio file"""
        # Preprocess audio
        mel_spec = self.preprocess_audio(audio_path)
        if mel_spec is None:
            return {"error": "Failed to preprocess audio"}
        
        # Convert to tensor
        if self.model_type == "cnn":
            # Add batch and channel dimensions: (1, 1, time_steps, n_mels)
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        else:  # RNN
            # Add batch dimension: (1, time_steps, n_mels)
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        mel_tensor = mel_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(mel_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "emotion": self.emotion_labels[predicted_class],
            "confidence": confidence,
            "probabilities": {
                self.emotion_labels[i]: probabilities[0][i].item() 
                for i in range(len(self.emotion_labels))
            }
        }
    
    def predict_emotion_from_array(self, mel_spec: np.ndarray) -> Dict[str, Any]:
        """Predict emotion from mel-spectrogram array"""
        # Convert to tensor
        if self.model_type == "cnn":
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
        else:  # RNN
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        mel_tensor = mel_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(mel_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "emotion": self.emotion_labels[predicted_class],
            "confidence": confidence,
            "probabilities": {
                self.emotion_labels[i]: probabilities[0][i].item() 
                for i in range(len(self.emotion_labels))
            }
        }


def create_emotion_model(model_type: str = "cnn", num_classes: int = 3) -> nn.Module:
    """Factory function to create emotion recognition models"""
    if model_type == "cnn":
        return EmotionCNN(num_classes=num_classes)
    elif model_type == "rnn":
        return EmotionRNN(num_classes=num_classes)
    else:
        raise ValueError("model_type must be 'cnn' or 'rnn'")


if __name__ == "__main__":
    # Example usage
    recognizer = EmotionRecognizer(model_type="cnn", device="cpu")
    
    # Test with a sample audio file (if available)
    # result = recognizer.predict_emotion("sample_audio.wav")
    # print(result)
