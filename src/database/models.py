"""
Database Models for Voice Cloning System
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class VoiceSample(Base):
    """Voice sample storage"""
    __tablename__ = "voice_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    speaker_id = Column(String(100), nullable=True)
    duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    emotion_label = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cloned_voices = relationship("ClonedVoice", back_populates="source_sample")

class ClonedVoice(Base):
    """Generated cloned voice storage"""
    __tablename__ = "cloned_voices"
    
    id = Column(Integer, primary_key=True, index=True)
    source_sample_id = Column(Integer, ForeignKey("voice_samples.id"), nullable=False)
    text_input = Column(Text, nullable=False)
    emotion_detected = Column(String(50), nullable=True)
    emotion_target = Column(String(50), nullable=True)
    audio_file_path = Column(String(500), nullable=False)
    processing_time = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source_sample = relationship("VoiceSample", back_populates="cloned_voices")

class EmotionClassification(Base):
    """Emotion classification results"""
    __tablename__ = "emotion_classifications"
    
    id = Column(Integer, primary_key=True, index=True)
    voice_sample_id = Column(Integer, ForeignKey("voice_samples.id"), nullable=False)
    emotion_predicted = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    neutral_prob = Column(Float, nullable=True)
    happy_prob = Column(Float, nullable=True)
    sad_prob = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    voice_sample = relationship("VoiceSample")

class ModelMetrics(Base):
    """Model performance metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    metric_type = Column(String(50), nullable=False)  # accuracy, loss, etc.
    metric_value = Column(Float, nullable=False)
    dataset_split = Column(String(20), nullable=True)  # train, val, test
    epoch = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, nullable=False)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    api_requests = relationship("APIRequest", back_populates="session")

class APIRequest(Base):
    """API request logging"""
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id"), nullable=False)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Float, nullable=True)
    request_size = Column(Integer, nullable=True)
    response_size = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("UserSession", back_populates="api_requests")
