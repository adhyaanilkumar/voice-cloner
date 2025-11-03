"""
Database connection and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base
import os
from typing import Generator

# Database configuration
# In serverless environments, use /tmp for SQLite or a serverless database
is_serverless = os.getenv('VERCEL') or os.getenv('LAMBDA_TASK_ROOT')
if is_serverless:
    # Use /tmp for SQLite in serverless (note: SQLite is not ideal for serverless)
    # Consider using a proper serverless database like PostgreSQL
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/voice_cloner.db")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./voice_cloner.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    poolclass=StaticPool if "sqlite" in DATABASE_URL else None,
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with tables"""
    create_tables()
    print("Database initialized successfully")
