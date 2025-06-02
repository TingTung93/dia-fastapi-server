#!/usr/bin/env python3
"""
Simple server startup script for Dia FastAPI TTS Server
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the Dia TTS server"""
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI Server")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Debug and logging options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--save-outputs", action="store_true", help="Save all generated audio files")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts and processing details")
    parser.add_argument("--retention-hours", type=int, default=24, help="File retention period in hours")

    # Performance options
    parser.add_argument("--workers", type=int, help="Number of worker threads (default: auto-detect)")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile optimization")

    # GPU options
    parser.add_argument("--gpu-mode", choices=["single", "multi", "auto"], default="auto",
                       help="GPU mode: single, multi, or auto-detect")
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU IDs (e.g., '0,1,2')")

    # Quick preset options
    parser.add_argument("--dev", action="store_true", help="Development mode (debug + save + show + reload)")
    parser.add_argument("--production", action="store_true", help="Production mode (optimized)")

    args = parser.parse_args()

    # Handle preset modes
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

    print("🚀 Starting Dia TTS Server")
    print("=" * 50)

    # Show configuration
    print("📋 Configuration:")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Debug: {'Yes' if args.debug else 'No'}")
    print(f"   Save outputs: {'Yes' if args.save_outputs else 'No'}")
    print(f"   Show prompts: {'Yes' if args.show_prompts else 'No'}")
    print(f"   Auto-reload: {'Yes' if args.reload else 'No'}")
    print(f"   Retention: {args.retention_hours} hours")
    print(f"   GPU mode: {args.gpu_mode}")
    if args.workers:
        print(f"   Workers: {args.workers}")
    if args.gpus:
        print(f"   GPU IDs: {args.gpus}")
    print()

    # Quick environment check
    try:
        import torch
        import fastapi
        from dia import Dia
        print("✅ Core packages available")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
        else:
            print("⚠️  CUDA not available - using CPU")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run setup.py first to install dependencies")
        sys.exit(1)

    print()
    print("🌐 Server Endpoints:")
    print(f"   Health: http://{args.host}:{args.port}/health")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   SillyTavern: http://{args.host}:{args.port}/v1/audio/speech")
    print(f"   Voices: http://{args.host}:{args.port}/voices")
    print(f"   Audio Prompts: http://{args.host}:{args.port}/audio_prompts")
    print(f"   Whisper Status: http://{args.host}:{args.port}/whisper/status")
    if args.debug:
        print(f"   Config: http://{args.host}:{args.port}/config")
        print(f"   Logs: http://{args.host}:{args.port}/logs")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 50)

    # Set up environment variables
    env = os.environ.copy()
    if args.workers:
        env["DIA_MAX_WORKERS"] = str(args.workers)
    if args.no_torch_compile:
        env["DIA_DISABLE_TORCH_COMPILE"] = "1"
    if args.gpu_mode:
        env["DIA_GPU_MODE"] = args.gpu_mode
    if args.gpus:
        env["DIA_GPU_IDS"] = args.gpus

    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    server_script = script_dir / "src" / "server.py"
    
    # Build server command
    cmd = [sys.executable, str(server_script)]
    
    # Add server arguments
    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])
    
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

    # Start the server
    try:
        if args.debug:
            print(f"🔧 Command: {' '.join(cmd)}")
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
        print("\n👋 Server interrupted")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()