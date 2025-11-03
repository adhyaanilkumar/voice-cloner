"""
Setup script for Voice Cloner with Emotion Recognition
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "models",
        "data/raw",
        "data/processed", 
        "generated_audio",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def test_installation():
    """Test if the installation was successful"""
    print("🧪 Testing installation...")
    
    # Try importing main modules to verify installation
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.models.emotion_recognition import create_emotion_model
        from src.models.tacotron2 import create_tacotron2_model
        from src.api.main import app
        print("✅ Core modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def setup_database():
    """Initialize database"""
    print("🗄️ Setting up database...")
    
    try:
        # Import and initialize database
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.database.database import init_database
        init_database()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎙️ Voice Cloner Setup")
    print("=" * 50)
    
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up database", setup_database),
        ("Testing installation", test_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_func():
            print(f"❌ Setup failed at step: {step_name}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    print("\n📖 Next steps:")
    print("1. Download datasets (optional for training):")
    print("   - RAVDESS: https://zenodo.org/record/1188976")
    print("   - LJSpeech: https://keithito.com/LJ-Speech-Dataset/")
    print("\n2. Train models (optional):")
    print("   python scripts/train_all_models.py --data_dir data")
    print("\n3. Start the API server:")
    print("   python -m src.api.main")
    print("\n4. Open the web dashboard:")
    print("   open web_dashboard/index.html")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
