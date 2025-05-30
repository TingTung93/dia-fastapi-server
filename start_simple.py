#!/usr/bin/env python3
"""
Simple, no-loop startup script for Dia TTS Server
"""

import os
import sys
import subprocess
import platform

def main():
    """Main startup function"""
    print("🚀 Dia TTS Server - Simple Startup\n")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("❌ Not in virtual environment!")
        print("\nPlease activate your virtual environment first:")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_dir = os.path.join(script_dir, 'venv')
        is_windows = platform.system() == 'Windows'
        
        if is_windows:
            activate_cmd = os.path.join(venv_dir, 'Scripts', 'activate.bat')
            print(f"   {activate_cmd}")
        else:
            activate_cmd = os.path.join(venv_dir, 'bin', 'activate')
            print(f"   source {activate_cmd}")
        
        print("\nOr create one if it doesn't exist:")
        print(f"   python -m venv {venv_dir}")
        return 1
    
    print("✅ Virtual environment active")
    
    # Check if required packages are installed
    missing_packages = []
    for package in ['fastapi', 'uvicorn', 'torch', 'soundfile', 'rich']:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("\nInstall with:")
        print("   pip install -r requirements.txt")
        print("   pip install dia-tts")
        
        # For CUDA users
        print("\nFor GPU support, install PyTorch with CUDA:")
        print("   pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
        return 1
    
    print("✅ Required packages installed")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  No GPU detected, using CPU")
    except:
        print("⚠️  Could not check GPU status")
    
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", help="GPU IDs (e.g., '0,1')")
    parser.add_argument("--workers", type=int)
    
    args = parser.parse_args()
    
    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m", "uvicorn", 
        "src.server:app",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    # Set environment variables
    env = os.environ.copy()
    
    if args.gpus:
        env["DIA_GPU_IDS"] = args.gpus
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"✅ Using GPUs: {args.gpus}")
    
    if args.workers:
        env["DIA_MAX_WORKERS"] = str(args.workers)
        print(f"✅ Workers: {args.workers}")
    
    if args.debug:
        env["DIA_DEBUG"] = "1"
        print("✅ Debug mode enabled")
    
    # Show startup info
    print(f"\n🌐 Starting server on http://{args.host}:{args.port}")
    print("📋 Key endpoints:")
    print(f"   Health: http://{args.host}:{args.port}/health")
    print(f"   GPU Status: http://{args.host}:{args.port}/gpu/status")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print("\nPress Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    # Start server
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())