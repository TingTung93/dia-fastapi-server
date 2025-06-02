#!/usr/bin/env python3
"""
Fixed Whisper Installation Script
Uses the working method for Python 3.13 compatibility
"""

import subprocess
import sys
import os

def install_whisper():
    """Install Whisper using the working method"""
    print("🔧 Installing OpenAI Whisper (Python 3.13 compatible)")
    print("📦 Method: Direct from GitHub repository")
    print("⏱️  This may take a few minutes...\n")
    
    try:
        # Use the working installation method
        cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"]
        
        print("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Whisper installed successfully!")
        
        # Test the installation
        print("\n🧪 Testing installation...")
        test_cmd = [sys.executable, "-c", "import whisper; print('✅ Import successful')"]
        subprocess.run(test_cmd, check=True)
        
        print("\n🎉 Installation complete and verified!")
        print("\n📋 Available models:")
        print("  • tiny    (39MB)  - Fastest, good for testing")
        print("  • base    (74MB)  - Recommended default")
        print("  • small   (244MB) - Better accuracy")
        print("  • medium  (769MB) - High accuracy")
        print("  • large   (1550MB)- Best accuracy")
        
        print("\n🚀 Quick start:")
        print("  python whisper_simple_setup.py check")
        print("  python whisper_simple_setup.py batch")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed!")
        print(f"Error: {e}")
        print(f"Output: {e.output if hasattr(e, 'output') else 'No output'}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_requirements():
    """Check if required dependencies are available"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"  Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check pip
    try:
        import pip
        print("  ✅ pip available")
    except ImportError:
        print("  ❌ pip not available")
        return False
    
    # Check PyTorch
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  🎯 CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  ⚠️  PyTorch not found (will be installed with Whisper)")
    
    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✅ FFmpeg available")
        else:
            print("  ❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ❌ FFmpeg not found")
        print("     Install: https://ffmpeg.org/download.html")
        return False
    
    print("✅ All requirements satisfied")
    return True

def main():
    print("🎙️ Whisper Installation Tool (Python 3.13 Compatible)")
    print("=" * 60)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing dependencies.")
        return False
    
    print("\n" + "=" * 60)
    
    # Check if already installed
    try:
        import whisper
        print("ℹ️  Whisper appears to already be installed.")
        choice = input("Reinstall anyway? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("Installation cancelled.")
            return True
    except ImportError:
        pass
    
    success = install_whisper()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Whisper is ready to use!")
        print("\nNext steps:")
        print("1. Test: python whisper_simple_setup.py check")
        print("2. Transcribe: python whisper_simple_setup.py batch")
        print("3. Read: WHISPER_SOLUTION_GUIDE.md")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 