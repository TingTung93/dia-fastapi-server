#!/usr/bin/env python3
"""
Enhanced GPU-aware startup script for Dia FastAPI TTS Server
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
import traceback
import json

# Clear problematic Python environment variables
for var in ['PYTHONHOME', 'PYTHONEXECUTABLE']:
    if var in os.environ:
        print(f"ℹ️  Clearing {var} from environment")
        del os.environ[var]

def detect_cuda():
    """Detect CUDA availability and version"""
    cuda_info = {
        "available": False,
        "version": None,
        "nvcc_path": None,
        "gpu_count": 0,
        "gpu_names": []
    }
    
    # Check for nvcc
    nvcc_paths = [
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin\\nvcc.exe"
    ]
    
    for nvcc in nvcc_paths:
        if nvcc and os.path.exists(nvcc):
            try:
                result = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    cuda_info["nvcc_path"] = nvcc
                    # Extract version
                    for line in result.stdout.split('\n'):
                        if "release" in line:
                            parts = line.split("release")[-1].strip().split(",")[0]
                            cuda_info["version"] = parts
                            break
                    break
            except:
                pass
    
    # Check nvidia-smi for GPU info
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        name, memory = line.split(', ')
                        cuda_info["gpu_names"].append(f"{name} ({memory})")
                cuda_info["gpu_count"] = len(cuda_info["gpu_names"])
                cuda_info["available"] = True
        except:
            pass
    
    return cuda_info

def get_torch_cuda_version():
    """Determine appropriate PyTorch CUDA version"""
    cuda_info = detect_cuda()
    
    if not cuda_info["available"]:
        return None
    
    cuda_ver = cuda_info["version"]
    if cuda_ver:
        # Map CUDA versions to PyTorch indexes
        if "12" in cuda_ver:
            return "cu121"  # CUDA 12.1
        elif "11.8" in cuda_ver:
            return "cu118"
        elif "11.7" in cuda_ver:
            return "cu117"
        else:
            return "cu118"  # Default to 11.8
    
    return "cu118"  # Default

def ensure_venv_and_requirements():
    """Ensure script runs inside venv and all requirements are installed"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')
    is_windows = platform.system() == 'Windows'
    venv_python = os.path.join(venv_dir, 'Scripts' if is_windows else 'bin', 'python.exe' if is_windows else 'python')
    pip_executable = os.path.join(venv_dir, 'Scripts' if is_windows else 'bin', 'pip.exe' if is_windows else 'pip')
    
    # Requirements files
    req_file = os.path.join(script_dir, 'requirements.txt')
    req_cuda_file = os.path.join(script_dir, 'requirements-cuda.txt')

    # 1. Create venv if missing
    if not os.path.exists(venv_python):
        print(f"🔧 Creating virtual environment at {venv_dir} ...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
            print("✅ Virtual environment created.")
            print(f"🔄 Re-launching with venv Python...")
            os.execv(venv_python, [venv_python] + sys.argv)
        except subprocess.CalledProcessError as e:
            print("\n❌ Failed to create virtual environment!")
            print("\nPossible solutions:")
            print("1. On Debian/Ubuntu: sudo apt install python3-venv")
            print("2. On Fedora: sudo dnf install python3-devel")
            print("3. On macOS: Install Python from python.org")
            sys.exit(1)

    # 2. Relaunch if not in venv
    if os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        print(f"🔄 Switching to venv Python...")
        clean_env = os.environ.copy()
        for var in ['PYTHONHOME', 'PYTHONEXECUTABLE']:
            clean_env.pop(var, None)
        
        try:
            os.execve(venv_python, [venv_python] + sys.argv, clean_env)
        except OSError as e:
            print(f"❌ Failed to switch to venv: {e}")
            sys.exit(1)

    # 3. Detect CUDA and determine requirements
    print("\n🔍 Detecting GPU/CUDA setup...")
    cuda_info = detect_cuda()
    
    if cuda_info["available"]:
        print(f"✅ CUDA detected: {cuda_info['version']}")
        print(f"✅ Found {cuda_info['gpu_count']} GPU(s):")
        for gpu in cuda_info["gpu_names"]:
            print(f"   - {gpu}")
        use_cuda = True
        torch_cuda = get_torch_cuda_version()
        print(f"✅ Will install PyTorch with {torch_cuda} support")
    else:
        print("ℹ️  No CUDA detected, will use CPU (slower)")
        use_cuda = False
        torch_cuda = None

    # 4. Check if we need to install/update packages
    force_reinstall = False
    try:
        import torch
        # Check if PyTorch has correct CUDA support
        if use_cuda and not torch.cuda.is_available():
            print("⚠️  PyTorch installed but CUDA not available - will reinstall")
            force_reinstall = True
        elif use_cuda:
            print(f"✅ PyTorch with CUDA support already installed")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("📦 PyTorch not installed")
        force_reinstall = True

    # 5. Install requirements
    if force_reinstall or not all(can_import(m) for m in ["fastapi", "uvicorn", "torch", "soundfile"]):
        print("\n📦 Installing requirements...")
        
        if use_cuda and os.path.exists(req_cuda_file):
            print(f"📦 Using CUDA requirements from {req_cuda_file}")
            # Install CUDA requirements
            try:
                subprocess.check_call([pip_executable, 'install', '-r', req_cuda_file])
            except:
                print("⚠️  CUDA requirements failed, falling back to CPU")
                subprocess.check_call([pip_executable, 'install', '-r', req_file])
        else:
            # For CUDA but no cuda requirements file, install PyTorch with CUDA manually
            if use_cuda and torch_cuda:
                print(f"📦 Installing PyTorch with {torch_cuda} support...")
                torch_packages = [
                    f"torch==2.5.1+{torch_cuda}",
                    f"torchaudio==2.5.1+{torch_cuda}"
                ]
                index_url = f"https://download.pytorch.org/whl/{torch_cuda}"
                
                try:
                    subprocess.check_call([
                        pip_executable, 'install'] + torch_packages + 
                        ['--index-url', index_url]
                    )
                    print("✅ PyTorch with CUDA installed")
                except:
                    print("⚠️  Failed to install CUDA PyTorch, trying CPU version")
                    use_cuda = False
            
            # Install other requirements
            subprocess.check_call([pip_executable, 'install', '-r', req_file])
        
        print("✅ All requirements installed")
        
        # Relaunch to ensure clean imports
        print("🔄 Relaunching with updated packages...")
        os.execv(venv_python, [venv_python] + sys.argv)

    # 6. Install Dia if missing
    try:
        import dia
        print("✅ Dia package installed")
    except ImportError:
        print("📦 Installing Dia TTS model...")
        subprocess.check_call([pip_executable, 'install', 'dia-tts'])
        print("✅ Dia installed")
        os.execv(venv_python, [venv_python] + sys.argv)

def can_import(module):
    """Check if a module can be imported"""
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def check_gpu_environment():
    """Final GPU environment check"""
    print("\n🎮 GPU Environment Check:")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: Available")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   Memory: {memory:.1f} GB")
            
            # Test GPU
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                print("✅ GPU test: Success")
            except Exception as e:
                print(f"❌ GPU test failed: {e}")
        else:
            print("ℹ️  PyTorch CUDA: Not available (CPU mode)")
            
    except ImportError:
        print("❌ PyTorch not available")
    
    # Check environment variables
    print("\n📋 GPU Environment Variables:")
    for var in ['CUDA_VISIBLE_DEVICES', 'DIA_GPU_MODE', 'DIA_GPU_IDS']:
        val = os.environ.get(var)
        if val:
            print(f"   {var}={val}")

def main():
    ensure_venv_and_requirements()
    
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI Server (GPU Enhanced)")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--check-only", action="store_true", help="Only check environment")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-outputs", action="store_true", help="Save generated audio")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts in console")
    
    # GPU options
    parser.add_argument("--gpu-mode", choices=["single", "multi", "auto"], default="auto",
                       help="GPU mode: single, multi, or auto")
    parser.add_argument("--gpus", type=str, help="GPU IDs to use (e.g., '0,1')")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile")
    
    # Performance
    parser.add_argument("--workers", type=int, help="Number of workers")
    
    # Presets
    parser.add_argument("--dev", action="store_true", help="Development mode")
    parser.add_argument("--production", action="store_true", help="Production mode")
    
    args = parser.parse_args()
    
    # Handle presets
    if args.dev:
        args.debug = True
        args.save_outputs = True
        args.show_prompts = True
        args.reload = True
    
    if args.production:
        args.debug = False
        args.save_outputs = False
        args.show_prompts = False
        args.reload = False
    
    print("\n🚀 Dia TTS Server Startup (GPU Enhanced)")
    print("=" * 50)
    
    # GPU environment check
    check_gpu_environment()
    
    if args.check_only:
        print("\n✅ Environment check complete!")
        return
    
    # Prepare environment variables
    env = os.environ.copy()
    
    # GPU configuration
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        print("\n⚠️  Forcing CPU mode (no GPU)")
    elif args.gpus:
        env["DIA_GPU_IDS"] = args.gpus
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"\n✅ Using GPUs: {args.gpus}")
    
    if args.gpu_mode:
        env["DIA_GPU_MODE"] = args.gpu_mode
    
    if args.workers:
        env["DIA_MAX_WORKERS"] = str(args.workers)
    
    if args.no_torch_compile:
        env["DIA_DISABLE_TORCH_COMPILE"] = "1"
    
    # Server info
    print(f"\n🌐 Starting server on {args.host}:{args.port}")
    print("\n📋 Configuration:")
    print(f"   GPU Mode: {args.gpu_mode}")
    print(f"   Debug: {'✅' if args.debug else '❌'}")
    print(f"   Save outputs: {'✅' if args.save_outputs else '❌'}")
    print(f"   Auto-reload: {'✅' if args.reload else '❌'}")
    
    print("\n🔗 Endpoints:")
    print(f"   Health: http://{args.host}:{args.port}/health")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   GPU Status: http://{args.host}:{args.port}/gpu/status")
    
    print("\nPress Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    # Build command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')
    is_windows = platform.system() == 'Windows'
    venv_python = os.path.join(venv_dir, 'Scripts' if is_windows else 'bin', 'python.exe' if is_windows else 'python')
    
    cmd = [
        venv_python,
        "-m", "uvicorn",
        "src.server:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    # Server args via environment
    if args.debug:
        env["DIA_DEBUG"] = "1"
    if args.save_outputs:
        env["DIA_SAVE_OUTPUTS"] = "1"
    if args.show_prompts:
        env["DIA_SHOW_PROMPTS"] = "1"
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()