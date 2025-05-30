#!/usr/bin/env python3
"""
Simple startup script for Dia FastAPI TTS Server
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
import traceback
from pathlib import Path

def ensure_venv_and_requirements():
    """Ensure script runs inside venv and all requirements are installed"""
    script_dir = Path(__file__).parent.absolute()
    venv_dir = script_dir / 'venv'
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        venv_python = venv_dir / 'Scripts' / 'python.exe'
        pip_executable = venv_dir / 'Scripts' / 'pip.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'
        pip_executable = venv_dir / 'bin' / 'pip'
    
    req_file = script_dir / 'requirements.txt'

    # 1. Create venv if missing
    if not venv_python.exists():
        print(f"🔧 Creating virtual environment at {venv_dir} ...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', str(venv_dir)])
            print("✅ Virtual environment created.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Failed to create virtual environment!")
            print("\nPossible solutions:")
            print("1. On Debian/Ubuntu: sudo apt install python3-venv")
            print("2. On Fedora: sudo dnf install python3-devel")
            print("3. On macOS: Install Python from python.org (not system Python)")
            print("4. On Windows: Ensure Python was installed with 'Add to PATH' option")
            sys.exit(1)

    # 2. Check if we're in the venv, if not restart in venv
    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    
    if current_python != target_python:
        print(f"🔄 Switching to virtual environment...")
        try:
            # Use subprocess instead of os.execv to avoid the loop issues
            cmd = [str(venv_python)] + sys.argv
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ Failed to switch to venv Python: {e}")
            sys.exit(1)

    # 3. Check requirements by trying imports
    required_packages = {
        "fastapi": "fastapi>=0.104.0",
        "uvicorn": "uvicorn[standard]>=0.24.0", 
        "torch": "torch>=2.0.0",
        "transformers": "transformers>=4.35.0",
        "librosa": "librosa>=0.10.0",
        "soundfile": "soundfile>=0.13.1"
    }
    
    missing_packages = []
    for package_name, requirement in required_packages.items():
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(requirement)
    
    # Check for dia separately since it might be installed but not importable
    try:
        from dia import Dia
    except ImportError:
        if "git+https://github.com/nari-labs/dia.git" not in missing_packages:
            missing_packages.append("git+https://github.com/nari-labs/dia.git")

    if missing_packages:
        print(f"📦 Missing packages detected")
        print("📦 Installing dependencies with correct CUDA support...")
        print("   This may take several minutes on first run")
        
        try:
            # Step 1: Always install/reinstall PyTorch with CUDA 12.8 first
            print("\n🔥 Installing PyTorch 2.6+ with CUDA 12.8 support...")
            print("   This ensures GPU acceleration for both Dia TTS and Whisper")
            
            # Remove any existing PyTorch to avoid conflicts
            subprocess.run([
                str(pip_executable), 'uninstall', '-y',
                'torch', 'torchaudio', 'torchvision'
            ], capture_output=True)
            
            # Install PyTorch with CUDA 12.8
            subprocess.check_call([
                str(pip_executable), 'install',
                '--index-url', 'https://download.pytorch.org/whl/cu128',
                'torch>=2.6.0', 'torchaudio>=2.6.0', 'torchvision>=0.21.0'
            ])
            print("✅ PyTorch with CUDA 12.8 installed successfully")
            
            # Verify CUDA installation immediately
            try:
                result = subprocess.run([
                    str(venv_python), '-c', 
                    'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"GPU count: {torch.cuda.device_count()}")'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("🔍 CUDA verification:")
                    print(f"   {result.stdout.strip()}")
                else:
                    print("⚠️  CUDA verification failed - continuing with installation")
            except:
                print("⚠️  Could not verify CUDA - continuing with installation")
            
            # Step 2: Install core dependencies
            print("\n📦 Installing core FastAPI dependencies...")
            core_packages = [
                'fastapi>=0.104.0',
                'uvicorn[standard]>=0.24.0',
                'python-multipart>=0.0.6',
                'pydantic>=2.0.0',
                'aiofiles>=23.1.0',
                'rich>=13.0.0',
                'click>=8.1.0'
            ]
            subprocess.check_call([str(pip_executable), 'install'] + core_packages)
            print("✅ Core dependencies installed")
            
            # Step 3: Install audio processing dependencies
            print("\n🎵 Installing audio processing dependencies...")
            audio_packages = [
                'soundfile>=0.13.1',
                'librosa>=0.10.0',
                'numpy>=1.24.0,<2.3.0',
                'scipy>=1.10.0'
            ]
            subprocess.check_call([str(pip_executable), 'install'] + audio_packages)
            print("✅ Audio processing dependencies installed")
            
            # Step 4: Install Transformers and ML dependencies
            print("\n🤖 Installing Transformers and ML dependencies...")
            ml_packages = [
                'transformers>=4.35.0',
                'accelerate>=0.21.0',
                'datasets>=2.14.0',
                'huggingface_hub>=0.19.0',
                'safetensors>=0.4.0'
            ]
            subprocess.check_call([str(pip_executable), 'install'] + ml_packages)
            print("✅ ML dependencies installed")
            
            # Step 5: Install development dependencies
            print("\n🛠️  Installing development dependencies...")
            dev_packages = [
                'pytest>=7.4.0',
                'pytest-asyncio>=0.21.0',
                'httpx>=0.24.0'
            ]
            subprocess.check_call([str(pip_executable), 'install'] + dev_packages)
            print("✅ Development dependencies installed")
            
            # Step 6: Install Dia model
            print("\n🎯 Installing Dia TTS model...")
            subprocess.check_call([
                str(pip_executable), 'install', 
                'git+https://github.com/nari-labs/dia.git'
            ])
            print("✅ Dia TTS model installed")
            
            print("\n✅ All dependencies installed successfully!")
            print("🔄 Restarting to load new packages...")
            
            # Restart to ensure imports work
            cmd = [str(venv_python)] + sys.argv
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Installation failed: {e}")
            print("\n🔧 Manual installation steps:")
            print("1. Install PyTorch with CUDA:")
            print(f"   {pip_executable} install --index-url https://download.pytorch.org/whl/cu128 torch>=2.6.0 torchaudio>=2.6.0 torchvision>=0.21.0")
            print("2. Install other dependencies:")
            print(f"   {pip_executable} install fastapi uvicorn[standard] transformers librosa soundfile rich")
            print("3. Install Dia model:")
            print(f"   {pip_executable} install git+https://github.com/nari-labs/dia.git")
            print("4. Check NVIDIA drivers: nvidia-smi")
            print("5. Check CUDA toolkit version matches PyTorch")
            sys.exit(1)
    else:
        print("✅ All requirements satisfied.")

def check_environment():
    """Check if the environment is properly set up"""
    issues = []

    # Check if required packages are available
    try:
        import fastapi
        import uvicorn
        import torch
        from dia import Dia
        print("✅ Core packages available")
    except ImportError as e:
        issues.append(f"❌ Missing required package: {e}")
        return issues

    # Check Python version compatibility
    python_version = sys.version_info
    if python_version >= (3, 13):
        print("⚠️  Warning: Python 3.13+ may have compatibility issues with some packages")
        print("   Consider using Python 3.11 or 3.12 for best compatibility")
    elif python_version < (3, 9):
        issues.append("❌ Python 3.9+ required")
        return issues
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} compatibility good")

    # Check CUDA availability
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            print(f"✅ CUDA {cuda_version} available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f}GB)")
        else:
            print("❌ CUDA not available - will use CPU (much slower)")
            print("   Possible issues:")
            print("   - PyTorch CPU version installed instead of CUDA version")
            print("   - NVIDIA drivers not installed or outdated")
            print("   - CUDA toolkit version mismatch")
            print("   💡 Fix by reinstalling PyTorch with CUDA:")
            print("      pip uninstall torch torchaudio torchvision -y")
            print("      pip install --index-url https://download.pytorch.org/whl/cu128 torch>=2.6.0 torchaudio>=2.6.0 torchvision>=0.21.0")
    except Exception as e:
        print(f"⚠️  Could not check CUDA status: {e}")
    
    # Check if model cache exists
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_patterns = ["models--nari-labs--Dia-1.6B", "*dia*"]
        found_model = False
        
        for pattern in model_patterns:
            if list(cache_dir.glob(pattern)):
                found_model = True
                break
        
        if found_model:
            print("✅ Dia model found in cache")
        else:
            print("ℹ️  Dia model not cached, will download on first run (~3.2GB)")
    except Exception:
        print("ℹ️  Could not check model cache")

    return issues

def main():
    """Main entry point"""
    try:
        # Ensure we're in the right environment first
        ensure_venv_and_requirements()
        
        # Add project root to PYTHONPATH to fix module imports
        script_dir = Path(__file__).parent.absolute()
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Check environment and CUDA status
        check_environment()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Start Dia TTS Server')
        parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
        parser.add_argument('--port', type=int, default=7860, help='Port to bind to')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        parser.add_argument('--save-outputs', action='store_true', help='Save audio outputs')
        parser.add_argument('--show-prompts', action='store_true', help='Show prompts in logs')
        parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
        parser.add_argument('--retention', type=int, default=24, help='Output retention in hours')
        parser.add_argument('--gpu', choices=['auto', 'force', 'disable'], default='auto',
                          help='GPU mode: auto, force, or disable')
        args = parser.parse_args()

        print("\n🚀 Dia TTS Server Startup")
        print("=" * 50)
        print("📋 Configuration:")
        print(f"   Host: {args.host}:{args.port}")
        print(f"   Debug: {'Yes' if args.debug else 'No'}")
        print(f"   Save outputs: {'Yes' if args.save_outputs else 'No'}")
        print(f"   Show prompts: {'Yes' if args.show_prompts else 'No'}")
        print(f"   Auto-reload: {'Yes' if args.reload else 'No'}")
        print(f"   Retention: {args.retention} hours")
        print(f"   GPU mode: {args.gpu}")
        print()

        # Set environment variables
        env = os.environ.copy()
        env['DIA_DEBUG'] = str(args.debug).lower()
        env['DIA_SAVE_OUTPUTS'] = str(args.save_outputs).lower()
        env['DIA_SHOW_PROMPTS'] = str(args.show_prompts).lower()
        env['DIA_OUTPUT_RETENTION'] = str(args.retention)
        env['DIA_GPU_MODE'] = args.gpu
        
        # Build the uvicorn command
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'src.server:app',
            '--host', args.host,
            '--port', str(args.port)
        ]
        
        if args.reload:
            cmd.append('--reload')
            
        if args.debug:
            cmd.append('--log-level=debug')
        else:
            cmd.append('--log-level=info')

        # Print working directory for debugging
        if args.debug:
            script_dir = Path(__file__).parent.absolute()
            print(f"🔧 Working directory: {script_dir}")
            print()

        # Run the server
        result = subprocess.run(cmd, env=env, cwd=script_dir, check=False)
        
        if result.returncode == 0:
            print("\n✅ Server exited cleanly")
        elif result.returncode in (1, 3221225786):  # Common Ctrl+C codes
            print("\n👋 Server stopped by user")
        else:
            print(f"\n❌ Server exited with code {result.returncode}")
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        print("\n👋 Startup interrupted")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()