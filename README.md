# Dia FastAPI TTS Server

A high-performance FastAPI server for the Dia text-to-speech model with multi-GPU support, real-time progress tracking, and SillyTavern compatibility.

## Features

- 🚀 **Multi-GPU Support** - Automatically distributes load across multiple GPUs with CUDA optimization
- 📊 **Real-time Progress** - Beautiful console output with progress bars and performance metrics
- 🔄 **Async/Sync Modes** - Support for both immediate and job-based processing
- 🎙️ **Voice Cloning** - Upload custom audio samples for voice cloning
- 🎯 **Whisper Integration** - Automatic transcription of audio prompts for enhanced voice cloning
- 🎛️ **Flexible Configuration** - Full control over model parameters and generation settings
- 📝 **Comprehensive Logging** - Debug mode with detailed performance tracking
- 🐳 **Docker Ready** - Easy deployment with Docker support
- 🎮 **SillyTavern Compatible** - Compatible TTS API for voice generation

## Quick Start

### Prerequisites

- **Windows 10/11** or **Linux/macOS**
- **Python 3.11+** (can be installed automatically)
- **~5GB disk space** for dependencies and model
- **CUDA-capable GPU** (recommended) or CPU
- **Internet connection** for downloads

### Installation

#### Windows: All-In-One Setup (Recommended)

**Option 1: Batch Script (Easiest)**
```cmd
# Download project and run:
setup_windows_aio.bat
```

**Option 2: PowerShell (Advanced)**
```powershell
# Run as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows_aio.ps1
```

The Windows AIO scripts automatically handle:
- ✅ Python installation (if needed)
- ✅ Virtual environment setup
- ✅ All dependencies (PyTorch, FastAPI, Whisper, etc.)
- ✅ GPU/CUDA configuration
- ✅ Model downloads
- ✅ Server startup

📖 **Detailed Windows instructions**: [`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md)

#### Linux/macOS: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dia-fastapi-server.git
cd dia-fastapi-server
```

2. Run the server (dependencies will be installed automatically):
```bash
python start_server.py
```

The startup script will:
- Create a virtual environment if needed
- Install all required dependencies
- Download the Dia model on first run (~3.2GB)
- Start the server

#### Method 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dia-fastapi-server.git
cd dia-fastapi-server
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python src/server.py
```

#### Method 3: Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dia-fastapi-server.git
cd dia-fastapi-server
```

2. Build and run with Docker Compose:
```bash
docker-compose up -d
```

### Running the Server

#### Basic Usage
```bash
python start_server.py
```

#### Development Mode (with debug output)
```bash
python start_server.py --dev
```

#### Production Mode
```bash
python start_server.py --production
```

#### Multi-GPU Configuration
```bash
# Auto-detect and use all GPUs
python start_server.py --gpu-mode auto

# Use specific GPUs only
python start_server.py --gpu-mode multi --gpus "0,2,3"

# Force single GPU mode
python start_server.py --gpu-mode single
```

## API Endpoints

### Text-to-Speech Generation

#### Synchronous Mode (immediate response)
```bash
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "aria"}' \
  --output speech.wav
```

#### Asynchronous Mode (job-based)
```bash
# Submit job
curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "aria"}'

# Check job status
curl "http://localhost:7860/jobs/{job_id}"

# Download result
curl "http://localhost:7860/jobs/{job_id}/result" -o result.wav
```

### Voice Management

#### List Available Voices
```bash
curl "http://localhost:7860/voices"
```

#### Upload Custom Voice
```bash
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@voice_sample.wav"
```

#### Create Voice Mapping
```bash
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "custom_voice",
    "style": "friendly",
    "primary_speaker": "S1",
    "audio_prompt": "my_voice"
  }'
```

### System Status

#### GPU Status
```bash
curl "http://localhost:7860/gpu/status"
```

#### Queue Statistics
```bash
curl "http://localhost:7860/queue/stats"
```

#### Generation Logs (debug mode)
```bash
curl "http://localhost:7860/logs"
```

## Configuration Options

### Server Options
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 7860)
- `--reload`: Enable auto-reload for development
- `--debug`: Enable debug mode with verbose logging
- `--save-outputs`: Save all generated audio files
- `--show-prompts`: Show prompts and processing details
- `--retention-hours`: File retention period (default: 24)

### Performance Options
- `--workers`: Number of worker threads (default: auto-detect)
- `--no-torch-compile`: Disable torch.compile optimization
- `--gpu-mode`: GPU mode (single/multi/auto)
- `--gpus`: Comma-separated list of GPU IDs to use

### Generation Parameters
- `temperature` (0.1-2.0): Controls randomness and creativity
- `cfg_scale` (1.0-10.0): Classifier-free guidance strength
- `top_p` (0.0-1.0): Nucleus sampling threshold
- `max_tokens` (100-10000): Maximum generation length

## SillyTavern Integration

1. Start the server:
```bash
python start_server.py
```

2. In SillyTavern:
   - Navigate to Settings → Text-to-Speech
   - Set TTS Provider: **Custom API**
   - Model: **dia**
   - API Key: **dia-server** (any value works)
   - Endpoint URL: **http://localhost:7860/v1/audio/speech**

3. Select a voice:
   - Choose from: aria, atlas, luna, kai, zara, nova
   - Or use your custom uploaded voices

## Performance Metrics

The server displays real-time performance metrics including:
- Token generation speed (tokens/sec)
- Realtime factor (e.g., 3.5x realtime)
- GPU memory usage
- Queue statistics
- Worker utilization

Example output:
```
✓ Generation Complete
Time: 3.45s
Audio Duration: 5.12s
Estimated Tokens: 440
Speed: 127.5 tokens/sec
Realtime Factor: 1.48x
```

## Docker Deployment

### Build the Docker image:
```bash
docker build -t dia-fastapi-server .
```

### Run with Docker:
```bash
docker run -d \
  --gpus all \
  -p 7860:7860 \
  -e HF_TOKEN="your_token" \
  -v $(pwd)/audio_outputs:/app/audio_outputs \
  dia-fastapi-server
```

## Environment Variables

- `HF_TOKEN`: Hugging Face token for model download (required)
- `DIA_MAX_WORKERS`: Override default worker count
- `DIA_DISABLE_TORCH_COMPILE`: Set to "1" to disable torch.compile
- `DIA_GPU_MODE`: Set GPU mode (single/multi/auto)
- `DIA_GPU_IDS`: Comma-separated list of GPU IDs

## Testing & Validation

The server includes comprehensive test suites to validate functionality:

### CUDA Optimization Tests
```bash
python test_cuda_optimizations.py
```
Tests multi-GPU functionality, memory management, and performance optimizations.

### Whisper Integration Tests
```bash
python test_whisper_integration.py
```
Validates automatic transcription, audio prompt processing, and voice cloning with transcripts.

### Basic Functionality Tests
```bash
python test_gpu.py
```
Tests basic GPU detection and model loading.

### API Test Suite
```bash
cd tests/
pytest -v
```
Comprehensive API endpoint testing with the pytest framework.

## Troubleshooting

### Installation Issues

#### Python Version Error
**Problem**: `Python 3.11+ is required`
**Solution**: 
- Check your Python version: `python --version`
- Install Python 3.11 or higher from [python.org](https://python.org)
- On Ubuntu/Debian: `sudo apt update && sudo apt install python3.11`
- On macOS with Homebrew: `brew install python@3.11`

#### Virtual Environment Creation Fails
**Problem**: `Error creating virtual environment`
**Solution**:
- Ensure python-venv is installed: `sudo apt install python3.11-venv` (Ubuntu/Debian)
- Try creating manually: `python -m venv venv`
- Use absolute paths if needed

#### Dependency Installation Errors
**Problem**: `pip install fails with error`
**Solution**:
- Update pip: `pip install --upgrade pip`
- Try installing with legacy resolver: `pip install -r requirements.txt --use-deprecated=legacy-resolver`
- For PyTorch issues, install manually: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### GPU/CUDA Issues

#### CUDA Not Available
**Problem**: `CUDA is not available. GPU mode disabled.`
**Solution**:
- Check NVIDIA driver: `nvidia-smi`
- Install CUDA toolkit: Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA support

#### CUDA Out of Memory
**Problem**: `RuntimeError: CUDA out of memory`
**Solution**:
- Reduce worker count: `--workers 1`
- Use single GPU mode: `--gpu-mode single`
- Clear GPU cache: Add `torch.cuda.empty_cache()` in code
- Monitor GPU memory: `nvidia-smi -l 1`
- Use CPU mode as fallback: `python start_server.py --gpu-mode cpu`

#### Multiple GPU Detection Issues
**Problem**: `Only detecting one GPU when multiple are available`
**Solution**:
- Check all GPUs are visible: `nvidia-smi`
- Set CUDA_VISIBLE_DEVICES: `export CUDA_VISIBLE_DEVICES=0,1,2`
- Force multi-GPU mode: `--gpu-mode multi --gpus "0,1,2"`
- Check PyTorch GPU count: `python -c "import torch; print(torch.cuda.device_count())"`

### Model Loading Issues

#### Model Download Fails
**Problem**: `Failed to download model`
**Solution**:
- Check internet connection
- Set Hugging Face token: `export HF_TOKEN=your_token_here`
- Try manual download: `huggingface-cli download nari-labs/dia`
- Use offline mode if model exists: Set environment variable
- Check disk space (needs ~3.2GB)

#### Model Loading Timeout
**Problem**: `Model loading takes forever`
**Solution**:
- Be patient on first run (model download takes time)
- Check download progress in terminal
- Use faster internet connection
- Pre-download model separately

### Performance Issues

#### Slow Generation
**Problem**: `Generation is slower than expected`
**Solution**:
- Enable torch.compile (default): Remove `--no-torch-compile`
- Use GPU instead of CPU: Check GPU is being used
- Check GPU utilization: `nvidia-smi` or use `--debug`
- Reduce batch size if processing multiple requests
- Update NVIDIA drivers to latest version

#### High Memory Usage
**Problem**: `Server using too much RAM/VRAM`
**Solution**:
- Reduce retention period: `--retention-hours 1`
- Lower worker count: `--workers 2`
- Use garbage collection: Enabled by default
- Monitor memory: `htop` or `nvidia-smi`

### API/Network Issues

#### Connection Refused
**Problem**: `curl: (7) Failed to connect to localhost port 7860`
**Solution**:
- Check server is running: `ps aux | grep server.py`
- Check correct port: Default is 7860
- Try different host: `--host 127.0.0.1`
- Check firewall settings
- Verify no other service on port: `lsof -i :7860`

#### CORS Errors
**Problem**: `Cross-Origin Request Blocked`
**Solution**:
- CORS is enabled by default for all origins
- Check browser console for specific errors
- Try using proxy for development
- Verify API endpoint URL is correct

### Docker Issues

#### Docker Build Fails
**Problem**: `docker build fails with error`
**Solution**:
- Update Docker: `docker --version` (need 20.10+)
- Enable BuildKit: `export DOCKER_BUILDKIT=1`
- Check disk space for Docker
- Use CPU Dockerfile if no GPU: `docker build -f Dockerfile.cpu`

#### Container Won't Start
**Problem**: `Container exits immediately`
**Solution**:
- Check logs: `docker logs container_name`
- Verify GPU support: `docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi`
- Check environment variables in docker-compose.yml
- Ensure proper volume mounts

### Audio Issues

#### Generated Audio is Silent
**Problem**: `WAV file created but no sound`
**Solution**:
- Check input text is not empty
- Verify audio player supports 44.1kHz WAV
- Test with simple text: "Hello world"
- Check volume settings

#### Audio Quality Issues
**Problem**: `Audio sounds distorted or robotic`
**Solution**:
- Adjust temperature: Lower for more stable output
- Try different voice_id
- Check audio_prompt quality (if using custom voice)
- Verify sample rate is correct (44.1kHz)

### Common Error Messages

#### "No module named 'dia'"
**Solution**: The model will be downloaded automatically on first run. Ensure internet connection.

#### "RuntimeError: Expected all tensors to be on the same device"
**Solution**: This is a multi-GPU synchronization issue. Use `--gpu-mode single` or update PyTorch.

#### "AssertionError: Torch not compiled with CUDA enabled"
**Solution**: Install PyTorch with CUDA support. See installation section.

## Documentation

📚 **Complete documentation available in [`docs/`](docs/) folder:**

- 🖥️ **[Windows Setup](docs/WINDOWS_SETUP.md)** - Complete Windows installation guide
- 🚀 **[GPU Startup Guide](docs/GPU_STARTUP_GUIDE.md)** - CUDA optimization and multi-GPU setup
- 🎤 **[Audio Prompts Guide](docs/AUDIO_PROMPT_TRANSCRIPT_GUIDE.md)** - Voice cloning and audio prompts
- 🎯 **[Whisper Setup](docs/WHISPER_SETUP_GUIDE.md)** - Automatic transcription setup
- 🎮 **[Speaker Tags](docs/SPEAKER_TAG_GUIDE.md)** - SillyTavern integration
- 🛠️ **[Development Guide](docs/CLAUDE.md)** - Architecture and development info
- 🔧 **[Troubleshooting](docs/STARTUP_TROUBLESHOOTING.md)** - Common issues and solutions

### Getting Help

If you encounter issues not covered in the documentation:

1. **Check documentation first**: Browse the [`docs/`](docs/) folder for your specific issue
2. **Enable debug mode**: `python start_server.py --debug`
3. **Run tests**: `python test_gpu.py` or `python test_whisper_integration.py`
4. **Search existing GitHub issues**
5. **Create a new issue** with:
   - System info (OS, Python version, GPU)
   - Full error message and debug logs
   - Steps to reproduce
   - What documentation you've already checked

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

Built on top of the [Dia TTS model](https://github.com/nari-labs/dia) by Nari Labs.