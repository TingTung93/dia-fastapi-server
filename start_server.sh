#!/bin/bash

echo "🚀 Dia TTS Server - Linux/Mac Startup"
echo

# Check if venv exists
if [ ! -f "venv/bin/python" ]; then
    echo "❌ Virtual environment not found!"
    echo
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        echo "Try: sudo apt install python3-venv (Ubuntu/Debian)"
        exit 1
    fi
    echo "✅ Virtual environment created"
    echo
fi

# Activate venv
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if packages are installed
echo "📦 Checking packages..."
python -c "import fastapi, uvicorn, torch, soundfile" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    pip install dia-tts
    
    # Try to install CUDA PyTorch if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        echo "Installing PyTorch with CUDA support..."
        pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    fi
fi

# Check GPU
echo "🎮 Checking GPU..."
python -c "import torch; print('GPU available' if torch.cuda.is_available() else 'CPU only'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else '')"

# Set environment variables
export DIA_DEBUG=1
export DIA_GPU_MODE=auto

# Parse command line arguments
GPU_IDS=""
WORKERS=""
DEBUG_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Set GPU environment if specified
if [ ! -z "$GPU_IDS" ]; then
    export DIA_GPU_IDS="$GPU_IDS"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "✅ Using GPUs: $GPU_IDS"
fi

if [ ! -z "$WORKERS" ]; then
    export DIA_MAX_WORKERS="$WORKERS"
    echo "✅ Workers: $WORKERS"
fi

if [ ! -z "$DEBUG_MODE" ]; then
    export DIA_DEBUG="1"
    echo "✅ Debug mode enabled"
fi

echo
echo "🌐 Starting server on http://localhost:7860"
echo "📋 Key endpoints:"
echo "   Health: http://localhost:7860/health"
echo "   GPU Status: http://localhost:7860/gpu/status"
echo "   API Docs: http://localhost:7860/docs"
echo
echo "Press Ctrl+C to stop"
echo

# Start server
python -m uvicorn src.server:app --host 0.0.0.0 --port 7860 --reload