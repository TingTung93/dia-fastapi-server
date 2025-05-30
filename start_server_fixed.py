#!/usr/bin/env python3
"""
Fixed startup script for Dia FastAPI TTS Server - No loops!
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
import json
import time

# Clear problematic environment variables
for var in ['PYTHONHOME', 'PYTHONEXECUTABLE']:
    if var in os.environ:
        print(f"ℹ️  Clearing {var} from environment")
        del os.environ[var]

def run_command(cmd, check=True, capture_output=False):
    """Run command with error handling"""
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, check=check)
            return result.returncode == 0, "", ""
    except subprocess.CalledProcessError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", str(e)

def detect_cuda():
    """Detect CUDA availability"""
    # Check nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, None, []
    
    success, stdout, _ = run_command([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                                   capture_output=True)
    if not success:
        return False, None, []
    
    gpus = []
    for line in stdout.strip().split('\n'):
        if line:
            gpus.append(line)
    
    # Check CUDA version
    cuda_version = None
    for nvcc_path in [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]:
        if nvcc_path and os.path.exists(nvcc_path):
            success, stdout, _ = run_command([nvcc_path, "--version"], capture_output=True)
            if success and "release" in stdout:
                for line in stdout.split('\n'):
                    if "release" in line:
                        cuda_version = line.split("release")[-1].split(",")[0].strip()
                        break
                break
    
    return len(gpus) > 0, cuda_version, gpus

def setup_venv():
    """Setup virtual environment without loops"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')
    is_windows = platform.system() == 'Windows'
    
    # Define paths
    if is_windows:
        venv_python = os.path.join(venv_dir, 'Scripts', 'python.exe')
        venv_pip = os.path.join(venv_dir, 'Scripts', 'pip.exe')
        venv_activate = os.path.join(venv_dir, 'Scripts', 'activate.bat')
    else:
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        venv_pip = os.path.join(venv_dir, 'bin', 'pip')
        venv_activate = os.path.join(venv_dir, 'bin', 'activate')
    
    # Check if we're already in the venv
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    # If already in venv and it's the right one, we're good
    if in_venv and os.path.abspath(sys.prefix) == os.path.abspath(venv_dir):
        print("✅ Already in virtual environment")
        return True, venv_python, venv_pip
    
    # Create venv if it doesn't exist
    if not os.path.exists(venv_python):
        print(f"🔧 Creating virtual environment...")
        success, _, err = run_command([sys.executable, '-m', 'venv', venv_dir])
        if not success:
            print(f"❌ Failed to create venv: {err}")
            print("\nTry creating manually:")
            print(f"  {sys.executable} -m venv {venv_dir}")
            if is_windows:
                print(f"  {venv_activate}")
            else:
                print(f"  source {venv_activate}")
            return False, None, None
        print("✅ Virtual environment created")
    
    # If not in venv, instruct user to activate
    if not in_venv:
        print("\n⚠️  Please activate the virtual environment:")
        if is_windows:
            print(f"   {venv_activate}")
        else:
            print(f"   source {venv_activate}")
        print("\nThen run this script again.")
        return False, None, None
    
    return True, venv_python, venv_pip

def install_requirements(pip_path, use_cuda=False):
    """Install requirements - no loops!"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(script_dir, 'requirements.txt')
    req_cuda_file = os.path.join(script_dir, 'requirements-cuda.txt')
    
    print("\n📦 Checking installed packages...")
    
    # Check what's already installed
    success, stdout, _ = run_command([pip_path, 'list', '--format=json'], capture_output=True)
    installed_packages = {}
    if success:
        try:
            packages = json.loads(stdout)
            installed_packages = {p['name'].lower(): p['version'] for p in packages}
        except:
            pass
    
    # Check key packages
    required = ['fastapi', 'uvicorn', 'torch', 'numpy', 'soundfile']
    missing = [p for p in required if p not in installed_packages]
    
    if not missing:
        # Check if torch has CUDA
        if use_cuda:
            print("🔍 Checking PyTorch CUDA support...")
            check_script = "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"
            success, stdout, _ = run_command([sys.executable, '-c', check_script], capture_output=True)
            
            if success and 'CPU' in stdout and use_cuda:
                print("⚠️  PyTorch installed but no CUDA support")
                print("📦 Reinstalling PyTorch with CUDA...")
                
                # Uninstall CPU version
                run_command([pip_path, 'uninstall', 'torch', 'torchaudio', '-y'], check=False)
                
                # Install CUDA version
                if os.path.exists(req_cuda_file):
                    print(f"📦 Installing from {req_cuda_file}")
                    success, _, err = run_command([pip_path, 'install', '-r', req_cuda_file])
                else:
                    print("📦 Installing PyTorch with CUDA 11.8")
                    success, _, err = run_command([
                        pip_path, 'install', 
                        'torch==2.5.1+cu118', 
                        'torchaudio==2.5.1+cu118',
                        '--index-url', 'https://download.pytorch.org/whl/cu118'
                    ])
                
                if not success:
                    print(f"⚠️  Failed to install CUDA PyTorch: {err}")
                    print("   Continuing with CPU version...")
            else:
                print("✅ PyTorch with CUDA support detected" if 'CUDA' in stdout else "ℹ️  Using CPU PyTorch")
        
        print("✅ All required packages installed")
        return True
    
    # Install missing packages
    print(f"📦 Installing missing packages: {missing}")
    
    if use_cuda and os.path.exists(req_cuda_file):
        print(f"📦 Installing from {req_cuda_file}")
        success, _, err = run_command([pip_path, 'install', '-r', req_cuda_file])
    else:
        print(f"📦 Installing from {req_file}")
        success, _, err = run_command([pip_path, 'install', '-r', req_file])
    
    if not success:
        print(f"❌ Failed to install requirements: {err}")
        return False
    
    # Install dia-tts if missing
    if 'dia-tts' not in installed_packages:
        print("📦 Installing Dia TTS...")
        success, _, err = run_command([pip_path, 'install', 'dia-tts'])
        if not success:
            print(f"⚠️  Failed to install dia-tts: {err}")
    
    print("✅ Requirements installed")
    return True

def start_server(args, venv_python):
    """Start the server"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build command
    cmd = [
        venv_python,
        "-m", "uvicorn",
        "src.server:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    # Setup environment
    env = os.environ.copy()
    
    # GPU settings
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        print("⚠️  Forcing CPU mode")
    elif args.gpus:
        env["DIA_GPU_IDS"] = args.gpus
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"✅ Using GPUs: {args.gpus}")
    
    if args.gpu_mode:
        env["DIA_GPU_MODE"] = args.gpu_mode
    
    if args.workers:
        env["DIA_MAX_WORKERS"] = str(args.workers)
    
    if args.no_torch_compile:
        env["DIA_DISABLE_TORCH_COMPILE"] = "1"
    
    # Debug settings
    if args.debug:
        env["DIA_DEBUG"] = "1"
    if args.save_outputs:
        env["DIA_SAVE_OUTPUTS"] = "1"
    if args.show_prompts:
        env["DIA_SHOW_PROMPTS"] = "1"
    
    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Start Dia TTS Server (Fixed)")
    
    # Server options
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Auto-reload for development")
    
    # GPU options
    parser.add_argument("--gpu-mode", choices=["single", "multi", "auto"], default="auto")
    parser.add_argument("--gpus", help="GPU IDs to use (e.g., '0,1')")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-outputs", action="store_true", help="Save generated audio")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts")
    
    # Other options
    parser.add_argument("--workers", type=int, help="Number of workers")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    parser.add_argument("--check-only", action="store_true", help="Check environment only")
    
    args = parser.parse_args()
    
    # Dev mode
    if args.dev:
        args.debug = True
        args.save_outputs = True
        args.show_prompts = True
        args.reload = True
    
    print("🚀 Dia TTS Server Startup\n")
    
    # Step 1: Setup venv
    success, venv_python, venv_pip = setup_venv()
    if not success:
        return 1
    
    # Step 2: Detect CUDA
    print("\n🔍 Checking GPU/CUDA...")
    has_cuda, cuda_version, gpus = detect_cuda()
    
    if has_cuda:
        print(f"✅ CUDA {cuda_version or 'detected'}")
        print(f"✅ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"   - {gpu}")
        use_cuda = not args.force_cpu
    else:
        print("ℹ️  No CUDA detected, will use CPU")
        use_cuda = False
    
    # Step 3: Install requirements
    if not install_requirements(venv_pip, use_cuda):
        print("\n❌ Failed to install requirements")
        print("\nTry manually:")
        print(f"  {venv_pip} install -r requirements.txt")
        return 1
    
    # Step 4: Final check
    print("\n🔍 Final check...")
    check_script = """
import sys
try:
    import torch
    import fastapi
    import dia
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)
"""
    
    success, stdout, _ = run_command([venv_python, '-c', check_script], capture_output=True)
    if success:
        print(stdout)
    else:
        print("❌ Environment check failed")
        return 1
    
    if args.check_only:
        print("\n✅ Environment ready!")
        return 0
    
    # Step 5: Start server
    if not start_server(args, venv_python):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())