"""
ElevenLabs Voice Cloning API Service
Replaces Tacotron2/WaveGlow with direct API calls
"""

import os
import requests
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, BinaryIO
import logging
import time
import hashlib

logger = logging.getLogger(__name__)


class ElevenLabsService:
    """Service for interacting with ElevenLabs Voice Cloning API"""
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs service
        
        Args:
            api_key: ElevenLabs API key. If None, reads from ELEVENLABS_API_KEY env var
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not provided. "
                "Set ELEVENLABS_API_KEY environment variable or pass api_key parameter."
            )
        
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Cache for voice IDs (maps voice_name -> voice_id)
        self._voice_cache = {}
        
        # Cache for reference audio hashes (maps hash -> voice_id)
        # This allows us to reuse voices for the same reference audio
        self._reference_cache = {}
        
        logger.info("ElevenLabs service initialized")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make API request with error handling"""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            error_msg = f"ElevenLabs API error: {e}"
            if response.status_code == 401:
                error_msg = "Invalid API key. Please check your ELEVENLABS_API_KEY."
            elif response.status_code == 429:
                error_msg = "Rate limit exceeded. Please try again later."
            elif response.status_code == 402:
                error_msg = "Insufficient credits. Please upgrade your plan."
            
            try:
                error_detail = response.json()
                if "detail" in error_detail:
                    error_msg = f"{error_msg} Details: {error_detail['detail']}"
            except:
                pass
            
            logger.error(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_voices(self) -> Dict[str, Any]:
        """Get list of available voices"""
        response = self._make_request("GET", "/voices")
        return response.json()
    
    def _get_reference_hash(self, audio_path: str) -> str:
        """Generate hash for reference audio file to identify duplicates"""
        try:
            with open(audio_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.warning(f"Failed to hash reference audio: {e}")
            return str(time.time())  # Fallback to timestamp
    
    def _find_existing_voice_by_hash(self, audio_hash: str) -> Optional[str]:
        """Check if we already created a voice for this reference audio"""
        if audio_hash in self._reference_cache:
            voice_id = self._reference_cache[audio_hash]
            # Verify the voice still exists
            try:
                self.get_voice_info(voice_id)
                return voice_id
            except Exception:
                # Voice was deleted, remove from cache
                del self._reference_cache[audio_hash]
        return None
    
    def _find_voice_by_name(self, name: str) -> Optional[str]:
        """Find voice ID by name from existing voices"""
        try:
            voices_response = self.get_voices()
            voices = voices_response.get("voices", [])
            for voice in voices:
                if voice.get("name") == name:
                    voice_id = voice.get("voice_id")
                    if voice_id:
                        return voice_id
        except Exception as e:
            logger.warning(f"Failed to search for existing voice: {e}")
        return None
    
    def create_voice(self, 
                    name: str, 
                    audio_files: list[BinaryIO],
                    description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new voice clone from audio files
        
        Args:
            name: Name for the voice clone
            audio_files: List of file-like objects containing audio data
            description: Optional description
            
        Returns:
            Voice creation response with voice_id
        """
        files = []
        for i, audio_file in enumerate(audio_files):
            files.append(("files", (f"audio_{i}.mp3", audio_file, "audio/mpeg")))
        
        data = {"name": name}
        if description:
            data["description"] = description
        
        # Remove Content-Type from headers for multipart form data
        headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{self.BASE_URL}/voices/add",
            headers=headers,
            data=data,
            files=files
        )
        
        response.raise_for_status()
        result = response.json()
        
        voice_id = result.get("voice_id")
        if voice_id:
            self._voice_cache[name] = voice_id
        
        logger.info(f"Created voice '{name}' with ID: {voice_id}")
        return result
    
    def clone_voice_from_file(self,
                              audio_file_path: str,
                              voice_name: Optional[str] = None,
                              voice_language: str = "en",
                              auto_delete: bool = False,
                              reuse_existing: bool = True) -> str:
        """
        Create a voice clone from an audio file
        
        Args:
            audio_file_path: Path to audio file
            voice_name: Name for the voice (auto-generated if None)
            voice_language: Language code for the voice (e.g., "en", "es", "fr") - REQUIRED by ElevenLabs
            auto_delete: If True, delete the voice after cloning (for temporary voices) - NOT RECOMMENDED for starter plan
            reuse_existing: If True, check for existing voice with same name/content before creating new one
            
        Returns:
            Voice ID
        """
        if voice_name is None:
            voice_name = f"voice_{int(time.time())}"
        
        # Check cache first
        if voice_name in self._voice_cache:
            voice_id = self._voice_cache[voice_name]
            # Verify it still exists
            try:
                self.get_voice_info(voice_id)
                logger.info(f"Reusing cached voice: {voice_name} ({voice_id})")
                return voice_id
            except Exception:
                # Voice was deleted, remove from cache
                del self._voice_cache[voice_name]
        
        # Check for existing voice by name (reuse_existing mode)
        if reuse_existing:
            existing_voice_id = self._find_voice_by_name(voice_name)
            if existing_voice_id:
                logger.info(f"Found existing voice with name '{voice_name}': {existing_voice_id}")
                self._voice_cache[voice_name] = existing_voice_id
                return existing_voice_id
            
            # Check by reference audio hash
            audio_hash = self._get_reference_hash(audio_file_path)
            existing_voice_id = self._find_existing_voice_by_hash(audio_hash)
            if existing_voice_id:
                logger.info(f"Found existing voice for same reference audio: {existing_voice_id}")
                self._reference_cache[audio_hash] = existing_voice_id
                return existing_voice_id
        
        # Read audio file and prepare for upload
        with open(audio_file_path, "rb") as audio_file:
            audio_content = audio_file.read()
        
        # Validate file size
        audio_size_mb = len(audio_content) / (1024 * 1024)
        if audio_size_mb > 10:
            raise Exception(f"Audio file is too large ({audio_size_mb:.2f}MB). ElevenLabs requires files to be under 10MB.")
        
        # Create BytesIO object and ensure pointer is at beginning
        from io import BytesIO
        audio_bytes = BytesIO(audio_content)
        audio_bytes.seek(0)  # Reset to beginning
        
        # Determine file extension and proper MIME type
        file_ext = Path(audio_file_path).suffix.lower()
        filename = Path(audio_file_path).name or f"audio{file_ext}"
        
        # Proper MIME type mapping
        mime_type_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg"
        }
        mime_type = mime_type_map.get(file_ext, "audio/mpeg")
        
        logger.info(f"Uploading voice: name={voice_name}, file={filename}, size={audio_size_mb:.2f}MB, type={mime_type}")
        
        # Prepare multipart form data - ElevenLabs expects files as tuple
        # Format: files = [("files", (filename, file_obj, content_type))]
        # The "files" key is important - ElevenLabs API expects this specific key
        files = [
            ("files", (filename, audio_bytes, mime_type))
        ]
        
        # ElevenLabs requires name and language parameters
        data = {
            "name": voice_name,
            "language": voice_language
        }
        
        # Remove Content-Type from headers - requests will set it automatically for multipart
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        
        # Ensure audio_bytes is at the beginning
        audio_bytes.seek(0)
        
        try:
            logger.info("=" * 80)
            logger.info(f"POST {self.BASE_URL}/voices/add")
            logger.info(f"Voice name: {voice_name}")
            logger.info(f"Voice language: {voice_language}")
            logger.info(f"File: {filename}")
            logger.info(f"MIME type: {mime_type}")
            logger.info(f"File size: {len(audio_content)} bytes ({audio_size_mb:.2f} MB)")
            logger.info(f"API Key present: {bool(self.api_key)}")
            logger.info(f"API Key prefix: {self.api_key[:10] if self.api_key else 'N/A'}...")
            logger.debug(f"Headers (without Content-Type): {headers}")
            logger.debug(f"Data: {data}")
            logger.info("=" * 80)
            
            response = requests.post(
                f"{self.BASE_URL}/voices/add",
                headers=headers,
                data=data,
                files=files,
                timeout=120  # Longer timeout for large files
            )
            
            logger.info("=" * 80)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if not response.ok:
                error_detail = ""
                error_json = None
                try:
                    error_json = response.json()
                    error_detail = str(error_json)
                    logger.error(f"ERROR JSON Response: {error_json}")
                    if isinstance(error_json, dict):
                        if 'detail' in error_json:
                            logger.error(f"Error detail field: {error_json['detail']}")
                        if 'message' in error_json:
                            logger.error(f"Error message field: {error_json['message']}")
                        if 'error' in error_json:
                            logger.error(f"Error field: {error_json['error']}")
                except Exception as json_error:
                    error_text = ""
                    try:
                        error_text = response.text[:2000] if hasattr(response, 'text') else str(response.content[:2000])
                        logger.error(f"ERROR TEXT Response (first 2000 chars): {error_text}")
                        error_detail = error_text
                    except Exception as text_error:
                        logger.error(f"Could not read error text: {text_error}")
                        error_detail = f"Status {response.status_code}: {response.reason}"
                
                logger.error(f"FAILED to create voice - HTTP {response.status_code}")
                logger.error(f"Full error response: {error_detail}")
                logger.info("=" * 80)
                
                # Create detailed error message and raise
                if error_json and isinstance(error_json, dict):
                    if 'detail' in error_json:
                        raise Exception(f"ElevenLabs API error: {error_json['detail']}")
                    elif 'message' in error_json:
                        raise Exception(f"ElevenLabs API error: {error_json['message']}")
                    else:
                        raise Exception(f"ElevenLabs API error (HTTP {response.status_code}): {error_json}")
                else:
                    raise Exception(f"ElevenLabs API error (HTTP {response.status_code}): {error_detail}")
            
            result = response.json()
            logger.info(f"Success! Response JSON: {result}")
            logger.info("=" * 80)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to upload voice to ElevenLabs: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    if isinstance(error_detail, dict) and 'detail' in error_detail:
                        error_msg = f"{error_msg} Details: {error_detail['detail']}"
                    elif isinstance(error_detail, dict):
                        error_msg = f"{error_msg} Response: {error_detail}"
                    else:
                        error_msg = f"{error_msg} Response: {e.response.text[:500]}"
                except:
                    error_msg = f"{error_msg} Response: {e.response.text[:500] if hasattr(e.response, 'text') else str(e.response.content[:500])}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        
        voice_id = result.get("voice_id")
        if not voice_id:
            raise Exception("Failed to create voice: voice_id not in response")
        
        # Cache the voice
        self._voice_cache[voice_name] = voice_id
        
        # Cache by reference audio hash
        audio_hash = self._get_reference_hash(audio_file_path)
        self._reference_cache[audio_hash] = voice_id
        
        logger.info(f"Created new voice: {voice_name} ({voice_id})")
        
        return voice_id
    
    def clone_voice_from_url(self,
                             audio_url: str,
                             voice_name: Optional[str] = None,
                             voice_language: str = "en",
                             auto_delete: bool = False,
                             reuse_existing: bool = True) -> str:
        """
        Create a voice clone from an external URL
        
        Args:
            audio_url: URL to audio file (must be publicly accessible)
            voice_name: Name for the voice (auto-generated if None)
            voice_language: Language code for the voice (e.g., "en", "es", "fr") - REQUIRED by ElevenLabs
            auto_delete: If True, delete the voice after cloning (for temporary voices) - NOT RECOMMENDED for starter plan
            reuse_existing: If True, check for existing voice with same name/content before creating new one
            
        Returns:
            Voice ID
        """
        import urllib.request
        import urllib.parse
        
        if voice_name is None:
            voice_name = f"voice_{int(time.time())}"
        
        # Download the file temporarily
        tmp_file_path = None
        try:
            logger.info(f"Downloading audio from URL: {audio_url}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file_path = tmp_file.name
            
            # Download with timeout
            urllib.request.urlretrieve(audio_url, tmp_file_path)
            
            # Validate file size
            file_size = os.path.getsize(tmp_file_path)
            file_size_mb = file_size / (1024 * 1024)
            if file_size_mb > 10:
                raise Exception(f"Downloaded file is too large ({file_size_mb:.2f}MB). ElevenLabs requires files to be under 10MB.")
            
            if file_size == 0:
                raise Exception("Downloaded file is empty")
            
            logger.info(f"Downloaded {file_size} bytes ({file_size_mb:.2f}MB) from {audio_url}")
            
            # Use existing file-based method
            voice_id = self.clone_voice_from_file(
                audio_file_path=tmp_file_path,
                voice_name=voice_name,
                voice_language=voice_language,
                auto_delete=auto_delete,
                reuse_existing=reuse_existing
            )
            
            return voice_id
            
        except urllib.error.URLError as e:
            error_msg = f"Failed to download audio from URL {audio_url}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Error processing audio from URL {audio_url}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        finally:
            # Clean up temp file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logger.debug(f"Cleaned up temporary file: {tmp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {tmp_file_path}: {cleanup_error}")
    
    def text_to_speech(self,
                      text: str,
                      voice_id: str,
                      model_id: str = "eleven_v3",
                      stability: str = "natural",  # v3 presets: "creative", "natural", "robust"
                      use_enhancement: bool = True) -> bytes:
        """
        Generate speech from text using Eleven v3 (Alpha) model
        v3 only supports: stability (preset) and enhancement
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            model_id: Model to use (default: eleven_v3)
            stability: Stability preset - "creative", "natural", or "robust" (default: "natural")
            use_enhancement: Use v3 enhancement (default: True)
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        # Use endpoint path, not full URL (to avoid double BASE_URL)
        endpoint = f"text-to-speech/{voice_id}"
        
        # Convert stability preset string to numeric value (0.0 to 1.0)
        # ElevenLabs API requires exact values: 0.0 (Creative), 0.5 (Natural), or 1.0 (Robust)
        stability_preset = stability.lower() if isinstance(stability, str) else str(stability).lower()
        stability_map = {
            "creative": 0.0,  # Creative - lowest stability
            "natural": 0.5,   # Natural - balanced stability
            "robust": 1.0     # Robust - highest stability
        }
        
        # Convert preset to numeric value, or use as-is if already numeric
        if stability_preset in stability_map:
            stability_value = stability_map[stability_preset]
        else:
            # Try to parse as float if it's a string number, otherwise default to natural
            try:
                stability_value = float(stability)
                if stability_value < 0.0 or stability_value > 1.0:
                    logger.warning(f"Stability value {stability_value} out of range [0.0, 1.0], defaulting to 0.5")
                    stability_value = 0.5
            except (ValueError, TypeError):
                logger.warning(f"Invalid stability value '{stability}', defaulting to 0.5 (natural)")
                stability_value = 0.5
        
        # Build voice settings for v3 (stability as float, enhancement as boolean)
        voice_settings = {
            "stability": stability_value
        }
        
        # Add enhancement for v3 model
        if use_enhancement:
            voice_settings["enhancement"] = True
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        logger.info(f"TTS request (v3): voice_id={voice_id}, model={model_id}, stability={stability_value}, enhancement={use_enhancement}")
        logger.debug(f"Payload: {payload}")
        
        response = self._make_request("POST", endpoint, json=payload)
        
        # Debug: Log response details
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        logger.info(f"Response content type: {response.headers.get('content-type', 'unknown')}")
        logger.info(f"Response content length: {len(response.content)} bytes")
        
        if len(response.content) == 0:
            logger.error("ERROR: Response content is empty!")
            logger.error(f"Response text (first 500 chars): {response.text[:500] if hasattr(response, 'text') else 'N/A'}")
            raise Exception("Received empty audio data from ElevenLabs API")
        
        return response.content
    
    def text_to_speech_with_reference(self,
                                     text: str,
                                     reference_audio_path: str,
                                     model_id: str = "eleven_v3",
                                     stability: str = "natural",  # v3 presets: "creative", "natural", "robust"
                                     use_enhancement: bool = True,
                                     reuse_voice: bool = True,
                                     voice_name: Optional[str] = None,
                                     use_instant_cloning: bool = True) -> bytes:
        """
        True Instant Voice Cloning: Generate speech from text using reference audio directly
        Supports Eleven v3 (Alpha) with enhancement
        v3 only supports: stability (preset) and enhancement
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio file for voice cloning
            model_id: Model to use (default: eleven_v3)
            stability: Stability preset - "creative", "natural", or "robust" (default: "natural")
            use_enhancement: Use v3 enhancement (default: True)
            reuse_voice: If True, reuse existing voice instead of creating new one each time
            voice_name: Optional name for the voice (for reuse)
            use_instant_cloning: If True, use instant voice cloning (creates voice temporarily if needed)
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        # Try to reuse existing voice if enabled
        if reuse_voice:
            audio_hash = self._get_reference_hash(reference_audio_path)
            existing_voice_id = self._find_existing_voice_by_hash(audio_hash)
            
            if existing_voice_id:
                logger.info(f"Reusing existing voice for reference audio: {existing_voice_id}")
                logger.info(f"Using v3 settings: stability={stability}, enhancement={use_enhancement}")
                # Use existing voice with v3 settings
                return self.text_to_speech(
                    text=text,
                    voice_id=existing_voice_id,
                    model_id=model_id,
                    stability=stability,
                    use_enhancement=use_enhancement
                )
            
            # Check by name if provided
            if voice_name:
                existing_voice_id = self._find_voice_by_name(voice_name)
                if existing_voice_id:
                    logger.info(f"Reusing existing voice with name '{voice_name}': {existing_voice_id}")
                    logger.info(f"Using v3 settings: stability={stability}, enhancement={use_enhancement}")
                    self._reference_cache[audio_hash] = existing_voice_id
                    return self.text_to_speech(
                        text=text,
                        voice_id=existing_voice_id,
                        model_id=model_id,
                        stability=stability,
                        use_enhancement=use_enhancement
                    )
        
        # Use True Instant Voice Cloning - Create voice then use it for TTS
        # This approach creates a voice first, then uses it for text-to-speech
        from io import BytesIO
        
        # Read reference audio
        with open(reference_audio_path, "rb") as audio_file:
            audio_content = audio_file.read()
        
        # Validate audio file size (ElevenLabs has 10MB limit)
        audio_size_mb = len(audio_content) / (1024 * 1024)
        if audio_size_mb > 10:
            raise Exception(f"Audio file is too large ({audio_size_mb:.2f}MB). ElevenLabs requires files to be under 10MB.")
        if audio_size_mb < 0.1:
            logger.warning(f"Audio file is very small ({audio_size_mb:.2f}MB). For better voice cloning results, use longer audio samples.")
        
        # Determine MIME type
        file_ext = Path(reference_audio_path).suffix.lower()
        mime_type_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/m4a",
            ".ogg": "audio/ogg"
        }
        mime_type = mime_type_map.get(file_ext, "audio/mpeg")
        
        # Generate voice name if not provided
        if voice_name is None:
            if reuse_voice:
                # Use hash-based name for better reuse
                audio_hash = self._get_reference_hash(reference_audio_path)
                voice_name = f"voice_{audio_hash[:8]}"
            else:
                voice_name = f"instant_clone_{int(time.time())}"
        
        logger.info("=" * 80)
        logger.info("INSTANT VOICE CLONING - STEP 1: Creating voice from reference audio")
        logger.info(f"Reference audio: {reference_audio_path}")
        logger.info(f"File size: {len(audio_content)} bytes ({audio_size_mb:.2f}MB)")
        logger.info(f"Voice name: {voice_name}")
        logger.info("=" * 80)
        
        # Reset BytesIO to beginning for file upload
        audio_bytes = BytesIO(audio_content)
        audio_bytes.seek(0)
        
        # Prepare multipart form data for voice creation
        files = [("files", (f"reference{file_ext}", audio_bytes, mime_type))]
        data = {
            "name": voice_name,
        }
        
        # Remove Content-Type from headers for multipart form data
        headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
        
        try:
            # STEP 1: Create instant voice clone using /voices/add
            logger.info(f"POST {self.BASE_URL}/voices/add")
            logger.info(f"Request: name={voice_name}, file_size={len(audio_content)} bytes")
            
            response = requests.post(
                f"{self.BASE_URL}/voices/add",
                headers=headers,
                data=data,
                files=files,
                timeout=60
            )
            
            logger.info(f"Voice creation response status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text[:500] if hasattr(response, 'text') else str(response.content[:500])
                logger.error(f"Voice creation failed with status {response.status_code}")
                logger.error(f"Error response: {error_text}")
                response.raise_for_status()
            
            voice_data = response.json()
            logger.info(f"Voice creation response: {voice_data}")
            
            voice_id = voice_data.get("voice_id")
            if not voice_id:
                logger.error(f"No voice_id in response! Response: {voice_data}")
                raise Exception(f"Failed to create instant voice clone: voice_id not in response. Response: {voice_data}")
            
            logger.info("=" * 80)
            logger.info(f"✓ Voice created successfully! Voice ID: {voice_id}")
            logger.info("=" * 80)
            
            # Cache the voice for reuse
            if reuse_voice:
                audio_hash = self._get_reference_hash(reference_audio_path)
                self._reference_cache[audio_hash] = voice_id
                self._voice_cache[voice_name] = voice_id
                logger.info(f"Cached voice {voice_id} for reuse with hash {audio_hash[:8]}")
            
            # STEP 2: Use voice for text-to-speech (THIS SHOULD CONSUME CREDITS)
            logger.info("=" * 80)
            logger.info("INSTANT VOICE CLONING - STEP 2: Generating speech with cloned voice")
            logger.info(f"Voice ID: {voice_id}")
            logger.info(f"Model: {model_id} (v3 Alpha)")
            logger.info(f"Text: {text[:100]}...")
            logger.info(f"Settings: stability={stability}, enhancement={use_enhancement}")
            logger.info("=" * 80)
            
            audio_data = self.text_to_speech(
                text=text,
                voice_id=voice_id,  # Use the cloned voice ID - THIS CONSUMES CREDITS
                model_id=model_id,
                stability=stability,
                use_enhancement=use_enhancement
            )
            
            logger.info("=" * 80)
            logger.info(f"✓ Speech generated successfully!")
            logger.info(f"Audio data size: {len(audio_data)} bytes")
            logger.info(f"THIS REQUEST SHOULD HAVE CONSUMED CREDITS")
            logger.info("=" * 80)
            
            return audio_data
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"Instant voice cloning failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    if "detail" in error_detail:
                        error_msg = f"{error_msg} Details: {error_detail['detail']}"
                    # Check for starter plan limitations
                    if e.response.status_code == 402:
                        error_msg = f"{error_msg} This might be a plan limitation. Check your ElevenLabs subscription."
                    elif e.response.status_code == 400:
                        error_msg = f"{error_msg} Check that your reference audio is under 10MB and high quality."
                except:
                    pass
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def clone_and_synthesize(self,
                             text: str,
                             reference_audio_path: str,
                             voice_name: Optional[str] = None,
                             output_path: Optional[str] = None,
                             stability: str = "natural",  # v3 presets: "creative", "natural", "robust"
                             use_enhancement: bool = True,
                             use_instant_cloning: bool = True,
                             reuse_voice: bool = True) -> Dict[str, Any]:
        """
        Complete voice cloning pipeline using Eleven v3 (Alpha) model with enhancement
        v3 only supports: stability (preset) and enhancement
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio for voice cloning
            voice_name: Name for the cloned voice (auto-generated if None)
            output_path: Path to save output audio (creates temp file if None)
            stability: Stability preset - "creative", "natural", or "robust" (default: "natural")
            use_enhancement: Use v3 enhancement (default: True)
            use_instant_cloning: Use instant voice cloning (recommended) vs creating voice first
            reuse_voice: If True, reuse existing voice instead of creating new one (recommended for starter plan)
            
        Returns:
            Dictionary with output_path, voice_id, and processing_time
        """
        start_time = time.time()
        
        try:
            # Use true instant voice cloning by default (better quality)
            if use_instant_cloning:
                logger.info(f"Using true instant voice cloning with Eleven v3 (Alpha) from: {reference_audio_path}")
                logger.info(f"v3 Parameters: stability={stability}, enhancement={use_enhancement}")
                if reuse_voice:
                    logger.info("Voice reuse enabled - will check for existing voices before creating new one")
                
                # Step 1: Generate speech with reference audio using true instant voice cloning
                audio_data = self.text_to_speech_with_reference(
                    text=text,
                    reference_audio_path=reference_audio_path,
                    model_id="eleven_v3",  # Use v3 Alpha model
                    stability=stability,
                    use_enhancement=use_enhancement,
                    reuse_voice=reuse_voice,
                    voice_name=voice_name,
                    use_instant_cloning=True
                )
                
                # Get actual voice_id if we reused or created one
                audio_hash = self._get_reference_hash(reference_audio_path)
                voice_id = self._reference_cache.get(audio_hash, "instant_clone")
            else:
                # Fallback: Create voice first, then use it (with reuse enabled)
                logger.info(f"Creating voice clone from: {reference_audio_path}")
                voice_id = self.clone_voice_from_file(
                    reference_audio_path, 
                    voice_name, 
                    auto_delete=False,  # Don't auto-delete for starter plan
                    reuse_existing=reuse_voice
                )
                logger.info(f"Voice cloned/reused with ID: {voice_id}")
                
                # Generate speech
                logger.info(f"Generating speech for text: {text[:50]}...")
                audio_data = self.text_to_speech(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_v3",  # Use v3 Alpha model
                    stability=stability,
                    use_enhancement=use_enhancement
                )
            
            # Step 2: Save audio
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp3", prefix="elevenlabs_clone_")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Voice cloning completed in {processing_time:.2f}s")
            logger.info(f"Output saved to: {output_path}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "voice_id": voice_id,
                "processing_time": processing_time,
                "audio_format": "mp3",
                "text_length": len(text)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Voice cloning failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def delete_voice(self, voice_id: str, update_cache: bool = True) -> bool:
        """Delete a voice clone"""
        try:
            self._make_request("DELETE", f"/voices/{voice_id}")
            logger.info(f"Deleted voice: {voice_id}")
            
            # Remove from caches
            if update_cache:
                # Remove from voice cache
                self._voice_cache = {k: v for k, v in self._voice_cache.items() if v != voice_id}
                # Remove from reference cache
                self._reference_cache = {k: v for k, v in self._reference_cache.items() if v != voice_id}
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {e}")
            return False
    
    def list_cached_voices(self) -> Dict[str, str]:
        """Get list of cached voice IDs (name -> voice_id mapping)"""
        return self._voice_cache.copy()
    
    def clear_voice_cache(self):
        """Clear internal voice cache (does not delete voices from ElevenLabs)"""
        self._voice_cache.clear()
        self._reference_cache.clear()
        logger.info("Voice cache cleared")
    
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get information about a voice"""
        response = self._make_request("GET", f"/voices/{voice_id}")
        return response.json()


def create_elevenlabs_service(api_key: Optional[str] = None) -> ElevenLabsService:
    """Factory function to create ElevenLabs service"""
    return ElevenLabsService(api_key=api_key)

