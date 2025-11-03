"""
Configuration Management for Voice Cloner
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    fmin: int = 0
    fmax: Optional[int] = None
    max_duration: float = 10.0  # seconds
    max_file_size: int = 10 * 1024 * 1024  # 10MB (ElevenLabs file size limit)

@dataclass
class ModelConfig:
    """Model configuration"""
    device: str = "auto"
    models_dir: str = "models"
    emotion_model_type: str = "cnn"
    vocab_size: int = 99  # Actual vocab size: 4 special tokens + 95 characters
    max_text_length: int = 200
    max_mel_length: int = 1000
    num_emotions: int = 3
    emotion_labels: list = None
    
    def __post_init__(self):
        if self.emotion_labels is None:
            self.emotion_labels = ["neutral", "happy", "sad"]

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./voice_cloner.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

@dataclass
class WebConfig:
    """Web dashboard configuration"""
    static_dir: str = "web_dashboard/static"
    template_dir: str = "web_dashboard"
    max_upload_size: int = 10 * 1024 * 1024  # 10MB (ElevenLabs limit)
    allowed_extensions: list = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = [".wav", ".mp3", ".flac", ".m4a"]

@dataclass
class ElevenLabsConfig:
    """ElevenLabs API configuration"""
    api_key: Optional[str] = None
    model_id: str = "eleven_v3"  # Use Eleven v3 (Alpha) model
    stability: str = "natural"  # Stability preset: "creative", "natural", or "robust" (v3 only)
    use_enhancement: bool = True  # Use v3 enhancement (v3 only)
    use_instant_cloning: bool = True  # Use instant voice cloning for better results
    reuse_voice: bool = True  # Reuse existing voices instead of creating new ones (recommended for starter plan)

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config_path = Path(self.config_file)
        
        # Initialize default configurations
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.web = WebConfig()
        self.elevenlabs = ElevenLabsConfig()
        
        # Load configuration from file if it exists
        self.load_config()
        
        # Override with environment variables
        self.load_from_env()
    
    def load_config(self):
        """Load configuration from JSON file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations
                if 'audio' in config_data:
                    for key, value in config_data['audio'].items():
                        if hasattr(self.audio, key):
                            setattr(self.audio, key, value)
                
                if 'model' in config_data:
                    for key, value in config_data['model'].items():
                        if hasattr(self.model, key):
                            setattr(self.model, key, value)
                
                if 'database' in config_data:
                    for key, value in config_data['database'].items():
                        if hasattr(self.database, key):
                            setattr(self.database, key, value)
                
                if 'api' in config_data:
                    for key, value in config_data['api'].items():
                        if hasattr(self.api, key):
                            setattr(self.api, key, value)
                
                if 'web' in config_data:
                    for key, value in config_data['web'].items():
                        if hasattr(self.web, key):
                            setattr(self.web, key, value)
                
                if 'elevenlabs' in config_data:
                    for key, value in config_data['elevenlabs'].items():
                        if hasattr(self.elevenlabs, key):
                            setattr(self.elevenlabs, key, value)
                
                logger.info(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.config_path}: {e}")
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Audio configuration
        self.audio.sample_rate = int(os.getenv('AUDIO_SAMPLE_RATE', self.audio.sample_rate))
        self.audio.n_fft = int(os.getenv('AUDIO_N_FFT', self.audio.n_fft))
        self.audio.hop_length = int(os.getenv('AUDIO_HOP_LENGTH', self.audio.hop_length))
        self.audio.n_mels = int(os.getenv('AUDIO_N_MELS', self.audio.n_mels))
        self.audio.max_duration = float(os.getenv('AUDIO_MAX_DURATION', self.audio.max_duration))
        self.audio.max_file_size = int(os.getenv('AUDIO_MAX_FILE_SIZE', self.audio.max_file_size))
        
        # Model configuration
        self.model.device = os.getenv('MODEL_DEVICE', self.model.device)
        self.model.models_dir = os.getenv('MODEL_DIR', self.model.models_dir)
        self.model.emotion_model_type = os.getenv('EMOTION_MODEL_TYPE', self.model.emotion_model_type)
        self.model.vocab_size = int(os.getenv('VOCAB_SIZE', self.model.vocab_size))
        self.model.max_text_length = int(os.getenv('MAX_TEXT_LENGTH', self.model.max_text_length))
        self.model.max_mel_length = int(os.getenv('MAX_MEL_LENGTH', self.model.max_mel_length))
        
        # Database configuration
        self.database.url = os.getenv('DATABASE_URL', self.database.url)
        self.database.echo = os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        
        # API configuration
        self.api.host = os.getenv('API_HOST', self.api.host)
        self.api.port = int(os.getenv('API_PORT', self.api.port))
        self.api.workers = int(os.getenv('API_WORKERS', self.api.workers))
        self.api.reload = os.getenv('API_RELOAD', 'false').lower() == 'true'
        self.api.log_level = os.getenv('API_LOG_LEVEL', self.api.log_level)
        
        # CORS origins from environment
        cors_origins_env = os.getenv('CORS_ORIGINS')
        if cors_origins_env:
            self.api.cors_origins = [origin.strip() for origin in cors_origins_env.split(',')]
        
        # ElevenLabs configuration
        self.elevenlabs.api_key = os.getenv('ELEVENLABS_API_KEY', self.elevenlabs.api_key)
        self.elevenlabs.model_id = os.getenv('ELEVENLABS_MODEL_ID', self.elevenlabs.model_id)
        # Stability must be "creative", "natural", or "robust" (v3 presets only)
        stability_env = os.getenv('ELEVENLABS_STABILITY', str(self.elevenlabs.stability))
        if stability_env.lower() in ["creative", "natural", "robust"]:
            self.elevenlabs.stability = stability_env.lower()
        else:
            self.elevenlabs.stability = "natural"  # Default to natural for v3
            logger.warning(f"Invalid stability preset '{stability_env}', defaulting to 'natural'")
        self.elevenlabs.use_enhancement = os.getenv('ELEVENLABS_USE_ENHANCEMENT', 'true').lower() == 'true'
        self.elevenlabs.use_instant_cloning = os.getenv('ELEVENLABS_USE_INSTANT_CLONING', 'true').lower() == 'true'
        self.elevenlabs.reuse_voice = os.getenv('ELEVENLABS_REUSE_VOICE', 'true').lower() == 'true'
    
    def save_config(self):
        """Save current configuration to JSON file"""
        config_data = {
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'n_fft': self.audio.n_fft,
                'hop_length': self.audio.hop_length,
                'n_mels': self.audio.n_mels,
                'fmin': self.audio.fmin,
                'fmax': self.audio.fmax,
                'max_duration': self.audio.max_duration,
                'max_file_size': self.audio.max_file_size
            },
            'model': {
                'device': self.model.device,
                'models_dir': self.model.models_dir,
                'emotion_model_type': self.model.emotion_model_type,
                'vocab_size': self.model.vocab_size,
                'max_text_length': self.model.max_text_length,
                'max_mel_length': self.model.max_mel_length,
                'num_emotions': self.model.num_emotions,
                'emotion_labels': self.model.emotion_labels
            },
            'database': {
                'url': self.database.url,
                'echo': self.database.echo,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'workers': self.api.workers,
                'reload': self.api.reload,
                'log_level': self.api.log_level,
                'cors_origins': self.api.cors_origins
            },
            'web': {
                'static_dir': self.web.static_dir,
                'template_dir': self.web.template_dir,
                'max_upload_size': self.web.max_upload_size,
                'allowed_extensions': self.web.allowed_extensions
            },
            'elevenlabs': {
                'api_key': None,  # Never save API key to config file
                'model_id': self.elevenlabs.model_id,
                'stability': self.elevenlabs.stability,
                'use_enhancement': self.elevenlabs.use_enhancement,
                'use_instant_cloning': self.elevenlabs.use_instant_cloning,
                'reuse_voice': self.elevenlabs.reuse_voice
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {self.config_path}: {e}")
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model file paths"""
        models_dir = Path(self.model.models_dir)
        return {
            'emotion_cnn': str(models_dir / "emotion_cnn_best.pth"),
            'emotion_rnn': str(models_dir / "emotion_rnn_best.pth"),
            'tacotron2': str(models_dir / "tacotron2_best.pth"),
            'waveglow': str(models_dir / "waveglow_pretrained.pth")  # Optional - will use Griffin-Lim if not found
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate audio configuration
        if self.audio.sample_rate <= 0:
            errors.append("Audio sample rate must be positive")
        
        if self.audio.n_fft <= 0:
            errors.append("Audio n_fft must be positive")
        
        if self.audio.hop_length <= 0:
            errors.append("Audio hop_length must be positive")
        
        if self.audio.n_mels <= 0:
            errors.append("Audio n_mels must be positive")
        
        # Validate model configuration
        if self.model.device not in ['auto', 'cpu', 'cuda']:
            errors.append("Model device must be 'auto', 'cpu', or 'cuda'")
        
        if self.model.emotion_model_type not in ['cnn', 'rnn']:
            errors.append("Emotion model type must be 'cnn' or 'rnn'")
        
        if self.model.vocab_size <= 0:
            errors.append("Vocabulary size must be positive")
        
        if self.model.num_emotions <= 0:
            errors.append("Number of emotions must be positive")
        
        # Validate API configuration
        if self.api.port <= 0 or self.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        if self.api.workers <= 0:
            errors.append("API workers must be positive")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        elevenlabs_dict = self.elevenlabs.__dict__.copy()
        # Don't expose API key in dict
        elevenlabs_dict['api_key'] = '***' if elevenlabs_dict.get('api_key') else None
        
        return {
            'audio': self.audio.__dict__,
            'model': self.model.__dict__,
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'web': self.web.__dict__,
            'elevenlabs': elevenlabs_dict
        }


# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration from file and environment"""
    global config
    config = Config(config.config_file)

if __name__ == "__main__":
    # Example usage
    config = get_config()
    print("Current configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Validate configuration
    if config.validate():
        print("Configuration is valid")
    else:
        print("Configuration has errors")

