# """
# Simple startup script for Dia FastAPI TTS Server
# """

import argparse
import os
import sys
import subprocess
import platform
import shutil
import traceback

# Attempt to mitigate issues from potentially problematic Python environment variables [REH]
# These can sometimes interfere with how Python finds its home or executable,
# especially in virtual environments or when system Python is unusually configured.
# It's best to ensure these are not set globally if they cause issues,
# but this script will try to clear them for its own execution context.
if 'PYTHONHOME' in os.environ:
    print(f"ℹ️  Detected PYTHONHOME: {os.environ['PYTHONHOME']}. Clearing for this script's execution context.")
    del os.environ['PYTHONHOME']
if 'PYTHONEXECUTABLE' in os.environ:
    print(f"ℹ️  Detected PYTHONEXECUTABLE: {os.environ['PYTHONEXECUTABLE']}. Clearing for this script's execution context.")
    del os.environ['PYTHONEXECUTABLE']

def ensure_venv_and_requirements():
    """Ensure script runs inside venv and all requirements are installed"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')
    is_windows = platform.system() == 'Windows'
    venv_python = os.path.join(venv_dir, 'Scripts' if is_windows else 'bin', 'python.exe' if is_windows else 'python')
    pip_executable = os.path.join(venv_dir, 'Scripts' if is_windows else 'bin', 'pip.exe' if is_windows else 'pip')
    req_file = os.path.join(script_dir, 'requirements.txt')

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
            print("3. On macOS: Install Python from python.org (not system Python)")
            print("\nAlternatively, create venv manually:")
            print(f"  {sys.executable} -m venv {venv_dir}")
            sys.exit(1)

    # 2. Relaunch if not in venv
    if os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        print(f"🔄 Switching to venv Python (using execve with cleaned env)...")
        # Ensure a clean environment is passed to the new process
        clean_env = os.environ.copy()
        # These should have been cleared by the top-level code, but ensure they are not re-introduced.
        if 'PYTHONHOME' in clean_env:
            print(f"ℹ️  PYTHONHOME found in env before execve: {clean_env['PYTHONHOME']}. Removing.")
            del clean_env['PYTHONHOME']
        if 'PYTHONEXECUTABLE' in clean_env:
            print(f"ℹ️  PYTHONEXECUTABLE found in env before execve: {clean_env['PYTHONEXECUTABLE']}. Removing.")
            del clean_env['PYTHONEXECUTABLE']
        
        print(f"🔧 Attempting to execute: {venv_python}")
        try:
            os.execve(venv_python, [venv_python] + sys.argv, clean_env)
        except OSError as e:
            print(f"❌ Failed to execve to venv Python: {e}")
            print(f"   Executable path was: {venv_python}")
            print(f"   Ensure this path is correct and the Python interpreter in the venv is not corrupted.")
            sys.exit(1)

    # 3. Ensure setuptools (for pkg_resources) is installed
    try:
        import pkg_resources
    except ImportError:
        print("📦 Installing setuptools...")
        subprocess.check_call([pip_executable, 'install', 'setuptools'])
        print("✅ Setuptools installed.")
        os.execv(venv_python, [venv_python] + sys.argv)

    # 4. Check requirements by import
    required_imports = ["fastapi", "uvicorn", "torch", "dia"]
    missing_imports = []
    for mod in required_imports:
        try:
            __import__(mod)
        except ImportError:
            missing_imports.append(mod)
    if missing_imports:
        print(f"📦 Missing required packages: {missing_imports}")
        print("📦 Installing requirements (this may take a few minutes on first run)...")
        subprocess.check_call([pip_executable, 'install', '-r', req_file])
        print("✅ All requirements installed.")
        os.execv(venv_python, [venv_python] + sys.argv)
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
        import dia
        print("✅ Required packages are available")
    except ImportError as e:
        issues.append(f"❌ Missing required package: {e}")
        issues.append("   Install with: pip install -r requirements.txt")

    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
        else:
            print("ℹ️  CUDA not available, will use CPU (slower)")
    except:
        pass
    
    # Check if model will be downloaded
    try:
        import huggingface_hub
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = "models--nari-labs--Dia-1.6B"
        if not (cache_dir / model_id).exists():
            print("ℹ️  Dia model not found in cache, will download on first run (~3.2GB)")
        else:
            print("✅ Dia model found in cache")
    except:
        pass

    return issues

def main():
    ensure_venv_and_requirements()
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI Server")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--check-only", action="store_true", help="Only check environment, don't start server")

    # Debug and logging options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--save-outputs", action="store_true", help="Save all generated audio files")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts and processing details in console")
    parser.add_argument("--retention-hours", type=int, default=24, help="File retention period in hours (default: 24)")

    # Performance options
    parser.add_argument("--workers", type=int, help="Number of worker threads (default: auto-detect)")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile optimization")

    # GPU options
    parser.add_argument("--gpu-mode", choices=["single", "multi", "auto"], default="auto",
                       help="GPU mode: single (use one GPU), multi (one model per GPU), auto (detect)")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')")

    # Quick preset options
    parser.add_argument("--dev", action="store_true", help="Development mode (debug + save outputs + show prompts + reload)")
    parser.add_argument("--production", action="store_true", help="Production mode (optimized settings)")

    args = parser.parse_args()

    # Handle preset modes
    if args.dev:
        args.debug = True
        args.save_outputs = True
        args.show_prompts = True
        args.reload = True
        print("🔧 Development mode enabled")

    if args.production:
        args.debug = False
        args.save_outputs = False
        args.show_prompts = False
        args.reload = False
        print("🏭 Production mode enabled")

    print("🚀 Dia TTS Server Startup")
    print("=" * 40)

    # Show configuration
    print("📋 Server Configuration:")
    print(f"   Debug mode: {'✅' if args.debug else '❌'}")
    print(f"   Save outputs: {'✅' if args.save_outputs else '❌'}")
    print(f"   Show prompts: {'✅' if args.show_prompts else '❌'}")
    print(f"   Auto-reload: {'✅' if args.reload else '❌'}")
    print(f"   Retention: {args.retention_hours} hours")
    if args.workers:
        print(f"   Workers: {args.workers}")
    print(f"   GPU mode: {args.gpu_mode}")
    if args.gpus:
        print(f"   GPU IDs: {args.gpus}")
    print()

    # Check environment
    issues = check_environment()

    if issues:
        print("\n⚠️  Environment Issues:")
        for issue in issues:
            print(issue)

        if args.check_only:
            sys.exit(1)
        else:
            print("\n⚠️  Please fix the above issues before continuing.")
            sys.exit(1)

    if args.check_only:
        print("\n✅ Environment check passed!")
        return

    print(f"\n🌐 Starting server on {args.host}:{args.port}")
    print("📋 SillyTavern Configuration:")
    print("   TTS Provider: OpenAI Compatible")
    print("   Model: dia")
    print("   API Key: sk-anything")
    print(f"   Endpoint URL: http://{args.host}:{args.port}/v1/audio/speech")
    print()
    print("🔗 Server endpoints:")
    print(f"   Health Check: http://{args.host}:{args.port}/health")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print(f"   Voice List: http://{args.host}:{args.port}/v1/voices")
    print(f"   Queue Stats: http://{args.host}:{args.port}/v1/queue/stats")
    print(f"   GPU Status: http://{args.host}:{args.port}/gpu/status")
    if args.debug:
        print(f"   Config API: http://{args.host}:{args.port}/v1/config")
        print(f"   Generation Logs: http://{args.host}:{args.port}/v1/logs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)

    # Get script directory and venv python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')
    is_windows = platform.system() == 'Windows'
    venv_python = os.path.join(venv_dir, 'Scripts' if is_windows else 'bin', 'python.exe' if is_windows else 'python')

    # Build command to start the server
    cmd = [
        venv_python,  # Use the virtual environment's Python executable
        "-m", "uvicorn", # Run uvicorn as a module
        "src.server:app", # Specify the application instance
        "--host", args.host,
        "--port", str(args.port)
    ]

    # Add flags
    if args.reload:
        cmd.append("--reload")
    if args.debug:
        cmd.append("--debug")
    if args.save_outputs:
        cmd.append("--save-outputs")
    if args.show_prompts:
        cmd.append("--show-prompts")
    if args.retention_hours != 24:
        cmd.extend(["--retention-hours", str(args.retention_hours)])

    # Environment variables for advanced options
    env = os.environ.copy()
    if args.workers:
        env["DIA_MAX_WORKERS"] = str(args.workers)
    if args.no_torch_compile:
        env["DIA_DISABLE_TORCH_COMPILE"] = "1"
    if args.gpu_mode:
        env["DIA_GPU_MODE"] = args.gpu_mode
    if args.gpus:
        env["DIA_GPU_IDS"] = args.gpus

    # Start the server
    try:
        if args.debug:
            print(f"🔧 Command: {' '.join(cmd)}")
            if args.workers:
                print(f"🔧 Workers: {args.workers}")
            if args.no_torch_compile:
                print("🔧 Torch compile: disabled")
            print()

        # Start the server process with proper environment variables
        server_process = subprocess.run(cmd, env=env, check=False)
        
        # On Windows, Ctrl+C often results in specific exit codes like 3221225786 (0xC000013A STATUS_CONTROL_C_EXIT)
        # or sometimes 1 if the Python interpreter handles SIGINT and exits with 1.
        # A clean exit is usually 0.
        if server_process.returncode != 0 and server_process.returncode != 3221225786 and server_process.returncode != 1:
            print(f"\n❌ Server process failed with unexpected exit code {server_process.returncode}")
            sys.exit(server_process.returncode)
        elif server_process.returncode == 0:
            print("\n✅ Server exited cleanly.")
        else:
            # Handles Ctrl+C cases gracefully
            print("\n👋 Server stopped by user (Ctrl+C detected).")

    except KeyboardInterrupt:
        # This outer KeyboardInterrupt might catch a Ctrl+C if it happens before or during subprocess.run() itself,
        # though typically the subprocess handles it first.
        print("\n👋 Server startup interrupted or main script stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()