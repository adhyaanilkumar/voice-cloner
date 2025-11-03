"""
Vercel Serverless Handler for FastAPI Application
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the FastAPI app
from src.api.main import app

# Vercel's Python runtime automatically detects ASGI apps
# Export the app variable for Vercel to use

