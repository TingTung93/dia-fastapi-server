#!/usr/bin/env python3
"""
Simple setup script for Dia FastAPI TTS Server
Installs all dependencies with CUDA support
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    """Main setup function"""
    print("🚀 Dia TTS Server Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    # Paths
    script_dir = Path(__file__).parent.absolute()
    venv_dir = script_dir / 'venv'
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        venv_python = venv_dir / 'Scripts' / 'python.exe'
        pip_executable = venv_dir / 'Scripts' / 'pip.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'
        pip_executable = venv_dir / 'bin' / 'pip'
    
    # Create venv if needed
    if not venv_python.exists():
        print("📦 Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', str(venv_dir)])
        print("✅ Virtual environment created")
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    subprocess.check_call([str(pip_executable), 'install', 'pip'])

    # Step 1: Install Dia model
    print("\n🎯 Installing Dia TTS model...")
    subprocess.check_call([
        str(pip_executable), 'install',
        'git+https://github.com/nari-labs/dia.git'
    ])
    print("✅ Dia model installed")
    
    # Step 2: Install PyTorch with CUDA 12.1
    print("\n🔥 Installing PyTorch with CUDA 12.8...")
    subprocess.check_call([
        str(pip_executable), 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu128'
    ])
    print("✅ PyTorch with CUDA installed")
    
    # Step 3: Install NumPy < 2.0 to avoid compatibility issues
    print("\n📦 Installing NumPy (compatible version)...")
    subprocess.check_call([
        str(pip_executable), 'install',
        'numpy<2.0'
    ])
    print("✅ NumPy installed")
    
    # Step 4: Install core dependencies
    print("\n📦 Installing core dependencies...")
    core_deps = [
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'python-multipart>=0.0.6',
        'pydantic>=2.0.0',
        'aiofiles>=23.1.0',
        'rich>=13.0.0',
        'click>=8.1.0'
    ]
    subprocess.check_call([str(pip_executable), 'install'] + core_deps)
    print("✅ Core dependencies installed")
    
    # Step 5: Install audio dependencies
    print("\n🎵 Installing audio dependencies...")
    audio_deps = [
        'soundfile>=0.13.1',
        'librosa>=0.10.0',
        'scipy>=1.10.0'
    ]
    subprocess.check_call([str(pip_executable), 'install'] + audio_deps)
    print("✅ Audio dependencies installed")
    
    # Step 6: Install ML dependencies
    print("\n🤖 Installing ML dependencies...")
    ml_deps = [
        'transformers>=4.35.0',
        'accelerate>=0.21.0',
        'datasets>=2.14.0',
        'huggingface_hub>=0.19.0',
        'safetensors>=0.4.0'
    ]
    subprocess.check_call([str(pip_executable), 'install'] + ml_deps)
    print("✅ ML dependencies installed")
    
    # Step 7: Quick verification
    print("\n🔍 Verifying installation...")
    verify_cmd = [
        str(venv_python), '-c',
        'import torch; print(f"PyTorch: {torch.__version__}"); print(f"CUDA: {torch.cuda.is_available()}"); print(f"GPUs: {torch.cuda.device_count()}")'
    ]
    subprocess.run(verify_cmd)
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    if is_windows:
        print("1. Activate venv: .\\venv\\Scripts\\activate")
    else:
        print("1. Activate venv: source venv/bin/activate")
    print("2. Start server: python start_server.py")
    print("3. Development mode: python start_server.py --dev")

if __name__ == "__main__":
    main()