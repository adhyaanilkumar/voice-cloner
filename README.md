# 🎙️ Voice Cloner with ElevenLabs API

A comprehensive **voice cloning system** powered by **ElevenLabs AI** that enables you to clone voices and generate speech with high-quality results. The system provides a user-friendly web interface and REST API for voice cloning operations.

## 🚀 Features

- **Voice Cloning**: Clone voices from audio files or URLs using ElevenLabs API
- **Voice Management**: Create, list, and manage your cloned voices
- **Text-to-Speech**: Generate speech using your cloned voices with customizable parameters
- **Web Dashboard**: User-friendly interface for all operations
- **REST API**: Complete backend API for integration with other applications
- **Optional Emotion Recognition**: CNN/RNN-based emotion detection (optional feature)

## 🏗️ Architecture

```
User Input → ElevenLabs API → Cloned Voice → Text-to-Speech → Generated Audio
     ↓              ↓                    ↓              ↓              ↓
Audio File    Voice Creation      Voice ID      Speech Gen    MP3 Output
```

### Core Components

1. **ElevenLabs Service**: Integration with ElevenLabs Voice Cloning API (v3 Alpha)
2. **FastAPI Backend**: RESTful API server with web dashboard
3. **Web Dashboard**: HTML/CSS/JavaScript interface for voice operations
4. **Database**: SQLite database for tracking voice samples and cloned voices
5. **Optional Emotion Recognition**: CNN/RNN-based emotion classifier (if model available)

## 📂 Project Structure

```
voice-cloner/
├── src/
│   ├── api/                    # FastAPI backend
│   │   └── main.py             # Main API server
│   ├── services/               # External service integrations
│   │   └── elevenlabs_service.py
│   ├── models/                 # ML models (optional)
│   │   └── emotion_recognition.py
│   ├── preprocessing/          # Audio processing
│   │   └── audio_processor.py
│   ├── database/               # Database models
│   │   ├── models.py
│   │   └── database.py
│   └── config.py               # Configuration management
├── web_dashboard/              # Web interface
│   ├── index.html
│   ├── create-voice.html
│   ├── manage-voices.html
│   ├── clone-voice.html
│   └── static/
│       ├── style.css
│       └── *.js
├── generated_audio/             # Generated audio files (created at runtime)
├── setup.py
├── setup_api_key.bat           # Windows batch script for API key setup
├── setup_api_key.ps1           # PowerShell script for API key setup
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **API Integration**: ElevenLabs Voice Cloning API
- **Database**: SQLAlchemy, SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Audio Processing**: Librosa, SoundFile (for optional features)
- **Optional ML**: PyTorch (for optional emotion recognition)

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- ElevenLabs API key ([Get one here](https://elevenlabs.io/))

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/adhyaanilkumar/voice-cloner.git
cd voice-cloner

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

**Windows (PowerShell):**
```powershell
.\setup_api_key.ps1
```

**Windows (Batch):**
```cmd
setup_api_key.bat
```

**Manual Setup:**
```bash
# Windows PowerShell
$env:ELEVENLABS_API_KEY="your-api-key-here"

# Windows CMD
set ELEVENLABS_API_KEY=your-api-key-here

# Linux/Mac
export ELEVENLABS_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:
```
ELEVENLABS_API_KEY=your-api-key-here
```

### 4. Start the Server

```bash
# Start FastAPI backend
python -m src.api.main

# Server will be available at http://localhost:8000
```

### 5. Access the Web Dashboard

Open your browser and navigate to:
- **Web Dashboard**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 API Usage

### Create a Voice

```python
import requests

# Create voice from audio file
with open('voice_sample.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/create-voice',
        files={'audio_file': f},
        data={'voice_name': 'My Voice', 'voice_language': 'en'}
    )
    
result = response.json()
print(f"Voice ID: {result['voice_id']}")
print(f"Voice Name: {result['voice_name']}")
```

### List Voices

```python
# List all voices
response = requests.get('http://localhost:8000/elevenlabs/voices')
voices = response.json()
print(f"Available voices: {len(voices['voices'])}")
```

### Clone Voice (Text-to-Speech)

```python
# Generate speech using a voice
response = requests.post(
    'http://localhost:8000/clone-voice',
    data={
        'text': 'Hello, this is a test of voice cloning.',
        'voice_id': 'your-voice-id-here',
        'stability': 'natural',  # Options: "creative", "natural", "robust"
        'use_enhancement': 'true'
    }
)

# Save the audio file
with open('output.mp3', 'wb') as f:
    f.write(response.content)
```

### Delete a Voice

```python
# Delete a voice
response = requests.delete(
    f'http://localhost:8000/elevenlabs/voices/{voice_id}'
)
result = response.json()
print(result['message'])
```

## 🔧 Configuration

The application can be configured via environment variables or a `config.json` file.

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# ElevenLabs Configuration
ELEVENLABS_API_KEY=your-api-key
ELEVENLABS_MODEL_ID=eleven_v3
ELEVENLABS_STABILITY=natural  # Options: creative, natural, robust
ELEVENLABS_USE_ENHANCEMENT=true

# Audio Configuration
AUDIO_SAMPLE_RATE=22050
AUDIO_MAX_FILE_SIZE=10485760  # 10MB (ElevenLabs limit)
```

### Configuration File

Create a `config.json` file in the project root:

```json
{
  "elevenlabs": {
    "model_id": "eleven_v3",
    "stability": "natural",
    "use_enhancement": true
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "audio": {
    "max_file_size": 10485760
  }
}
```

## 🌐 Web Dashboard Features

The web dashboard provides a user-friendly interface for:

1. **Create Voice**: Upload audio files or provide URLs to create new voices
2. **Manage Voices**: View and delete your cloned voices
3. **Clone Voice**: Generate speech using your voices with customizable text and parameters

## 🔒 Security & Privacy

- **API Key Security**: API keys are stored in environment variables, never in code
- **Data Handling**: Uploaded audio files are processed through ElevenLabs API
- **Temporary Files**: Local temporary files are cleaned up after processing
- **Consent Required**: Users must have permission to clone voices

## 🚧 Limitations

- **File Size**: Maximum 10MB per audio file (ElevenLabs limit)
- **API Quota**: Subject to your ElevenLabs subscription plan limits
- **Language Support**: Depends on ElevenLabs model capabilities
- **Processing Time**: Depends on ElevenLabs API response time

## 📝 API Endpoints

### Web Pages
- `GET /` - Home page
- `GET /create-voice` - Create voice page
- `GET /manage-voices` - Manage voices page
- `GET /clone-voice` - Clone voice page

### API Endpoints
- `GET /health` - Health check
- `GET /api-info` - API information
- `GET /elevenlabs/voices` - List all voices
- `POST /create-voice` - Create a new voice
- `POST /clone-voice` - Generate speech from text
- `DELETE /elevenlabs/voices/{voice_id}` - Delete a voice

### Debug Endpoints
- `GET /debug/logs` - View recent logs

## 🔮 Future Enhancements

- [ ] Support for more ElevenLabs models
- [ ] Voice preview functionality
- [ ] Batch voice creation
- [ ] Voice cloning from multiple samples
- [ ] Enhanced emotion control
- [ ] Mobile app development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ElevenLabs**: For providing the Voice Cloning API
- **FastAPI**: For the excellent web framework
- **All Contributors**: Who help improve this project

## 📞 Support

- **Repository**: https://github.com/adhyaanilkumar/voice-cloner
- **Issues**: [GitHub Issues](https://github.com/adhyaanilkumar/voice-cloner/issues)

---

✨ *A modern voice cloning system powered by ElevenLabs AI*
