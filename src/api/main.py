"""
FastAPI Backend for Voice Cloning System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import uuid
import logging
from pathlib import Path
import torch  # Only needed for optional emotion recognition
import shutil

# Set up logging - DEBUG level for troubleshooting
# In serverless environments (like Vercel), file logging is not available
is_serverless = os.getenv('VERCEL') or os.getenv('LAMBDA_TASK_ROOT') or not os.path.exists('/tmp')
if is_serverless:
    # Serverless environment - only use StreamHandler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
else:
    # Local environment - use both StreamHandler and FileHandler
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('voice_cloner.log', encoding='utf-8')
        ]
    )
logger = logging.getLogger(__name__)

# Import our models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Note: Local models removed - using ElevenLabs API exclusively
from src.models.emotion_recognition import EmotionRecognizer
from src.preprocessing.audio_processor import AudioProcessor
from src.database.database import get_db, init_database
from src.database.models import VoiceSample, ClonedVoice, EmotionClassification
from src.config import get_config
from src.services.elevenlabs_service import ElevenLabsService, create_elevenlabs_service

app = FastAPI(title="Voice Cloner API", version="1.0.0")

# Mount static files
# Get project root directory (parent of src directory)
project_root = Path(__file__).parent.parent.parent
static_dir = project_root / "web_dashboard" / "static"
try:
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Static files mounted from: {static_dir}")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
except RuntimeError:
    pass  # Already mounted

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (only keep what's needed for ElevenLabs + optional emotion recognition)
emotion_recognizer = None  # Optional: can be removed if not needed
audio_processor = None  # Optional: used for audio duration calculation only
elevenlabs_service = None
use_elevenlabs = False

# Request/Response models
class VoiceCloneRequest(BaseModel):
    text: str
    emotion: Optional[str] = "neutral"
    speaker_id: Optional[str] = None

class EmotionRecognitionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]

class VoiceCloneResponse(BaseModel):
    audio_file_path: str
    emotion_detected: str
    processing_time: float

# Initialize models
@app.on_event("startup")
async def startup_event():
    global emotion_recognizer, audio_processor, elevenlabs_service, use_elevenlabs
    
    try:
        # Load configuration
        config = get_config()
        if not config.validate():
            raise Exception("Invalid configuration")
        
        # Initialize ElevenLabs service - REQUIRED
        if not config.elevenlabs.api_key:
            raise Exception("ElevenLabs API key is required. Please set ELEVENLABS_API_KEY environment variable.")
        
        try:
            elevenlabs_service = create_elevenlabs_service(api_key=config.elevenlabs.api_key)
            use_elevenlabs = True
            logger.info("ElevenLabs API service initialized - using v3 (Alpha) model")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs service: {e}")
            raise
        
        # Initialize database (skip in serverless if SQLite is not available)
        try:
            init_database()
            logger.info("Database initialized successfully")
        except Exception as db_error:
            if is_serverless:
                logger.warning(f"Database initialization skipped in serverless environment: {db_error}")
            else:
                logger.error(f"Database initialization failed: {db_error}")
                raise
        
        # Optional: Initialize audio processor for duration calculation only
        try:
            audio_processor = AudioProcessor(
                sample_rate=config.audio.sample_rate,
                n_fft=config.audio.n_fft,
                hop_length=config.audio.hop_length,
                n_mels=config.audio.n_mels,
                fmin=config.audio.fmin,
                fmax=config.audio.fmax
            )
        except Exception as e:
            logger.warning(f"Failed to initialize audio processor (optional): {e}")
            audio_processor = None
        
        # Optional: Initialize emotion recognition if model exists
        try:
            emotion_model_path = config.get_model_paths()['emotion_cnn']
            if Path(emotion_model_path).exists():
                emotion_recognizer = EmotionRecognizer(
                    model_path=emotion_model_path,
                    model_type=config.model.emotion_model_type,
                    device=str(config.model.device) if config.model.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("Emotion recognition model loaded (optional)")
            else:
                logger.info("Emotion recognition model not found - skipping (optional)")
                emotion_recognizer = None
        except Exception as e:
            logger.warning(f"Failed to initialize emotion recognition (optional): {e}")
            emotion_recognizer = None
        
        logger.info("Startup completed - ElevenLabs API ready")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()
        raise

def serve_page(page_name: str):
    """Helper function to serve HTML pages"""
    # Get project root directory (parent of src directory)
    project_root = Path(__file__).parent.parent.parent
    page_path = project_root / "web_dashboard" / f"{page_name}.html"
    
    logger.debug(f"Serving page: {page_name}, path: {page_path}, exists: {page_path.exists()}")
    
    if page_path.exists():
        with open(page_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return HTMLResponse(content=content, media_type="text/html")
    logger.error(f"Page not found: {page_path}")
    return HTMLResponse(content=f"<h1>Page {page_name} not found</h1><p>Path: {page_path}</p>", media_type="text/html")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the home page"""
    return serve_page("index")

@app.get("/create-voice", response_class=HTMLResponse)
async def create_voice_page():
    """Serve the create voice page"""
    return serve_page("create-voice")

@app.get("/manage-voices", response_class=HTMLResponse)
async def manage_voices_page():
    """Serve the manage voices page"""
    return serve_page("manage-voices")

@app.get("/clone-voice", response_class=HTMLResponse)
async def clone_voice_page():
    """Serve the clone voice page"""
    return serve_page("clone-voice")


@app.get("/api-info")
async def api_info():
    """API information endpoint"""
    return {"message": "Voice Cloner API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "use_elevenlabs": use_elevenlabs,
        "elevenlabs_available": elevenlabs_service is not None,
        "models_loaded": all([
            emotion_recognizer is not None
        ])
    }

@app.get("/debug/logs")
async def get_debug_logs(lines: int = 100):
    """Get recent log entries for debugging"""
    try:
        log_file = Path("voice_cloner.log")
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                return {"logs": recent_lines, "total_lines": len(all_lines)}
        else:
            return {"logs": [], "message": "Log file not found"}
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
        return {"error": str(e)}

@app.get("/elevenlabs/voices")
async def list_elevenlabs_voices():
    """List all ElevenLabs voices (including cloned voices)"""
    if not use_elevenlabs or elevenlabs_service is None:
        raise HTTPException(status_code=503, detail="ElevenLabs service not available")
    
    try:
        voices_response = elevenlabs_service.get_voices()
        # ElevenLabs API returns {"voices": [...]}
        voices_list = voices_response.get("voices", []) if isinstance(voices_response, dict) else voices_response
        cached_voices = elevenlabs_service.list_cached_voices()
        return {
            "voices": voices_list,
            "cached_voices": cached_voices
        }
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@app.delete("/elevenlabs/voices/{voice_id}")
async def delete_elevenlabs_voice(voice_id: str):
    """Delete an ElevenLabs voice"""
    if not use_elevenlabs or elevenlabs_service is None:
        raise HTTPException(status_code=503, detail="ElevenLabs service not available")
    
    try:
        success = elevenlabs_service.delete_voice(voice_id)
        if success:
            return {"success": True, "message": f"Voice {voice_id} deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete voice")
    except Exception as e:
        logger.error(f"Failed to delete voice: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")

class CreateVoiceResponse(BaseModel):
    """Response for voice creation"""
    voice_id: str
    voice_name: str
    success: bool

@app.post("/create-voice", response_model=CreateVoiceResponse)
async def create_voice(
    audio_file: Optional[UploadFile] = File(None),
    audio_url: Optional[str] = Form(None),
    voice_name: str = Form(...),
    voice_language: str = Form("en")  # Default to English
):
    """Create a voice in ElevenLabs from uploaded audio file or URL"""
    if not use_elevenlabs or elevenlabs_service is None:
        raise HTTPException(
            status_code=503, 
            detail="ElevenLabs API is required. Please set your ELEVENLABS_API_KEY environment variable."
        )
    
    # Validate that either audio_file or audio_url is provided
    if not audio_file and not audio_url:
        raise HTTPException(status_code=400, detail="Either audio_file or audio_url must be provided")
    
    if audio_file and audio_url:
        raise HTTPException(status_code=400, detail="Please provide either audio_file OR audio_url, not both")
    
    tmp_file_path = None
    db = next(get_db())
    try:
        # Handle URL-based creation
        if audio_url:
            logger.info(f"Creating voice from URL: {audio_url}, name={voice_name}")
            try:
                voice_id = elevenlabs_service.clone_voice_from_url(
                    audio_url=audio_url,
                    voice_name=voice_name,
                    voice_language=voice_language
                )
                logger.info(f"Voice created successfully from URL: voice_id={voice_id}, name={voice_name}")
                
                return CreateVoiceResponse(
                    voice_id=voice_id,
                    voice_name=voice_name,
                    success=True
                )
            except Exception as e:
                error_message = str(e)
                logger.error(f"Failed to create voice from URL: {error_message}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Provide more helpful error messages
                if "unauthorized" in error_message.lower() or "401" in error_message or "api key" in error_message.lower():
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid or missing ElevenLabs API key. Please check your ELEVENLABS_API_KEY environment variable."
                    )
                elif "quota" in error_message.lower() or "limit" in error_message.lower() or "402" in error_message:
                    raise HTTPException(
                        status_code=402,
                        detail="ElevenLabs API quota exceeded or plan limit reached. Please check your ElevenLabs subscription."
                    )
                elif "too large" in error_message.lower() or "10mb" in error_message.lower():
                    raise HTTPException(
                        status_code=400,
                        detail="File size exceeds 10MB limit. Please use a shorter audio file or compress it."
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to create voice from URL: {error_message}"
                    )
        
        # Handle file-based creation
        # Validate file type
        if audio_file.content_type and not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Validate file size (ElevenLabs has 10MB limit)
        content = await audio_file.read()
        file_size_mb = len(content) / (1024 * 1024)
        original_filename = audio_file.filename or "audio"
        logger.info(f"Uploaded file: {original_filename}, size: {len(content)} bytes ({file_size_mb:.2f} MB), content_type: {audio_file.content_type}")
        
        config = get_config()
        max_size = 10 * 1024 * 1024  # 10MB limit for ElevenLabs
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail=f"File size must be less than 10MB. Your file is {file_size_mb:.2f}MB")
        
        # Check if file is empty
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty. Please upload a valid audio file.")
        
        # Validate file extension
        file_extension = Path(original_filename).suffix or ".wav"
        allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        if file_extension.lower() not in allowed_extensions:
            logger.warning(f"File extension '{file_extension}' not in allowed list, but attempting upload anyway")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            logger.debug(f"Saved temporary file: {tmp_file_path} ({len(content)} bytes)")
        
        logger.info(f"Creating voice in ElevenLabs: name={voice_name}, file={original_filename}, size={len(content)} bytes")
        
        # Create voice in ElevenLabs
        try:
            logger.info(f"Attempting to create voice: name={voice_name}, file={original_filename}, size={len(content)} bytes ({len(content) / (1024*1024):.2f}MB)")
            
            voice_id = elevenlabs_service.clone_voice_from_file(
                audio_file_path=tmp_file_path,
                voice_name=voice_name,
                voice_language=voice_language
            )
            
            logger.info(f"Voice created successfully: voice_id={voice_id}, name={voice_name}")
            
            return CreateVoiceResponse(
                voice_id=voice_id,
                voice_name=voice_name,
                success=True
            )
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to create voice in ElevenLabs: {error_message}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Provide more helpful error messages
            if "too large" in error_message.lower() or "10mb" in error_message.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds 10MB limit. Your file is {len(content) / (1024*1024):.2f}MB. Please use a shorter audio file or compress it."
                )
            elif "unauthorized" in error_message.lower() or "401" in error_message or "api key" in error_message.lower():
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing ElevenLabs API key. Please check your ELEVENLABS_API_KEY environment variable."
                )
            elif "quota" in error_message.lower() or "limit" in error_message.lower() or "402" in error_message:
                raise HTTPException(
                    status_code=402,
                    detail="ElevenLabs API quota exceeded or plan limit reached. Please check your ElevenLabs subscription."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create voice in ElevenLabs: {error_message}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating voice: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_file_path}: {e}")
        db.close()

async def clone_voice_with_voice_id(
    text: str,
    voice_id: str,
    target_emotion: Optional[str] = "neutral",
    stability: Optional[str] = None,  # v3 presets: "creative", "natural", "robust"
    use_enhancement: Optional[str] = None
) -> VoiceCloneResponse:
    """Clone voice using ElevenLabs API with existing voice_id"""
    import time
    
    start_time = time.time()
    db = next(get_db())
    
    try:
        config = get_config()
        
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not voice_id.strip():
            raise HTTPException(status_code=400, detail="Voice ID cannot be empty")
        
        logger.info(f"Cloning voice with voice_id: {voice_id}")
        
        # Use parameters from request or fallback to config defaults (v3 only supports stability and enhancement)
        
        # Handle stability - v3 supports presets ("creative", "natural", "robust")
        if stability is not None:
            if isinstance(stability, str) and stability.lower() in ["creative", "natural", "robust"]:
                stability_value = stability.lower()
            else:
                logger.warning(f"Invalid stability preset '{stability}', using default 'natural'")
                stability_value = "natural"
        else:
            stability_value = config.elevenlabs.stability if isinstance(config.elevenlabs.stability, str) else "natural"
        
        # Handle boolean conversion for use_enhancement
        if use_enhancement is not None:
            if isinstance(use_enhancement, str):
                use_enhancement_value = use_enhancement.lower() == 'true'
            else:
                use_enhancement_value = bool(use_enhancement)
        else:
            use_enhancement_value = config.elevenlabs.use_enhancement
        
        # Use ElevenLabs v3 (Alpha) to generate speech with the voice_id
        logger.info(f"Using ElevenLabs v3 (Alpha) API to generate speech")
        logger.info(f"Voice ID: {voice_id}")
        logger.info(f"Model: eleven_v3 (Alpha) with enhancement={use_enhancement_value}")
        logger.info(f"v3 Parameters: stability={stability_value}, enhancement={use_enhancement_value}")
        logger.info(f"THIS REQUEST SHOULD CONSUME CREDITS")
        
        # Generate speech using the voice_id directly
        audio_data = elevenlabs_service.text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id="eleven_v3",
            stability=stability_value,
            use_enhancement=use_enhancement_value
        )
        
        # Save audio to file (use /tmp in serverless environments)
        if is_serverless:
            output_dir = Path("/tmp/generated_audio")
        else:
            output_dir = Path("generated_audio")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_filename = f"cloned_voice_{voice_id[:8]}_{int(time.time())}.mp3"
        output_path = output_dir / output_filename
        
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Speech generated successfully: {output_path} ({len(audio_data)} bytes)")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"THIS REQUEST SHOULD HAVE CONSUMED CREDITS")
        
        # Log to database (optional)
        detected_emotion = target_emotion or "neutral"
        try:
            # Create a dummy voice sample if we don't have one (for database consistency)
            voice_sample = VoiceSample(
                filename=f"voice_{voice_id[:8]}",
                file_path="",  # No local file, using ElevenLabs voice_id
                duration=0.0,
                sample_rate=22050,
                emotion_label=detected_emotion,
                confidence=1.0
            )
            db.add(voice_sample)
            db.commit()
            db.refresh(voice_sample)
            
            cloned_voice = ClonedVoice(
                source_sample_id=voice_sample.id,
                text_input=text,
                emotion_detected=detected_emotion,
                emotion_target=target_emotion,
                audio_file_path=str(output_path),
                processing_time=processing_time
            )
            db.add(cloned_voice)
            db.commit()
        except Exception as db_error:
            logger.warning(f"Failed to log to database: {db_error}")
        
        return VoiceCloneResponse(
            audio_file_path=str(output_path).replace('\\', '/'),
            emotion_detected=detected_emotion,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ElevenLabs voice cloning: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        db.close()

@app.post("/clone-voice")
async def clone_voice(
    text: str = Form(...),
    voice_id: str = Form(...),  # Use voice_id instead of audio_file
    target_emotion: Optional[str] = Form("neutral"),
    stability: Optional[str] = Form(None),  # v3 presets: "creative", "natural", "robust"
    use_enhancement: Optional[str] = Form(None)
):
    """Clone voice using ElevenLabs v3 (Alpha) API with existing voice_id
    Returns audio file directly as MP3"""
    logger.info(f"Clone voice request received - text: {text[:50]}, voice_id: {voice_id}")
    
    # Require ElevenLabs API
    if not use_elevenlabs or elevenlabs_service is None:
        raise HTTPException(
            status_code=503, 
            detail="ElevenLabs API is required for voice cloning. Please set your ELEVENLABS_API_KEY environment variable."
        )
    
    try:
        # Parse stability and use_enhancement
        stability_value = stability if stability else "natural"
        use_enhancement_value = use_enhancement and use_enhancement.lower() in ["true", "1", "yes"]
        
        logger.info(f"Generating speech with voice_id={voice_id}, stability={stability_value}, enhancement={use_enhancement_value}")
        
        # Generate speech using the voice_id directly
        audio_data = elevenlabs_service.text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id="eleven_v3",
            stability=stability_value,
            use_enhancement=use_enhancement_value
        )
        
        # Log audio data size for debugging
        logger.info(f"Audio data received: {len(audio_data)} bytes")
        if len(audio_data) == 0:
            logger.error("ERROR: Audio data is empty! This should not happen.")
            raise HTTPException(status_code=500, detail="Received empty audio data from ElevenLabs API")
        
        # Return audio directly as StreamingResponse
        audio_stream = BytesIO(audio_data)
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="cloned_voice_{voice_id[:8]}.mp3"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning voice: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    # Use /tmp in serverless environments
    if is_serverless:
        file_path = Path("/tmp/generated_audio") / filename
    else:
        file_path = Path("generated_audio") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/mpeg"
    )

@app.post("/batch-process")
async def batch_process_audio(
    audio_files: list[UploadFile] = File(...),
    text: str = Form(...)
):
    """Process multiple audio files for emotion recognition"""
    if emotion_recognizer is None:
        raise HTTPException(status_code=500, detail="Emotion recognition model not loaded")
    
    results = []
    
    for audio_file in audio_files:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content = await audio_file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Recognize emotion
            result = emotion_recognizer.predict_emotion(tmp_file_path)
            result["filename"] = audio_file.filename
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": audio_file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
