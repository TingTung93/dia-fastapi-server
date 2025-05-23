# Dia FastAPI TTS Server

A high-performance FastAPI server for the Dia text-to-speech model with multi-GPU support, real-time progress tracking, and SillyTavern compatibility.

## Features

- 🚀 **Multi-GPU Support** - Automatically distributes load across multiple GPUs
- 📊 **Real-time Progress** - Beautiful console output with progress bars and performance metrics
- 🔄 **Async/Sync Modes** - Support for both immediate and job-based processing
- 🎙️ **Voice Cloning** - Upload custom audio samples for voice cloning
- 🎛️ **Flexible Configuration** - Full control over model parameters
- 📝 **Comprehensive Logging** - Debug mode with detailed performance tracking
- 🐳 **Docker Ready** - Easy deployment with Docker support
- 🎮 **SillyTavern Compatible** - Drop-in replacement for OpenAI TTS API

## Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended) or CPU
- Hugging Face token for model download

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dia-fastapi-server.git
cd dia-fastapi-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Hugging Face token:
```bash
export HF_TOKEN="your_huggingface_token"
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
  -d '{"text": "Hello, world!", "voice_id": "alloy"}' \
  --output speech.wav
```

#### Asynchronous Mode (job-based)
```bash
# Submit job
curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "alloy"}'

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
   - Set TTS Provider: **OpenAI Compatible**
   - Model: **dia**
   - API Key: **sk-anything** (any value works)
   - Endpoint URL: **http://localhost:7860/v1/audio/speech**

3. Select a voice:
   - Choose from: alloy, echo, fable, nova, onyx, shimmer
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

## Troubleshooting

### CUDA Out of Memory
- Reduce worker count: `--workers 1`
- Use single GPU mode: `--gpu-mode single`
- Use float16 precision (automatic on most GPUs)

### Slow Generation
- Enable torch.compile (default)
- Use GPU instead of CPU
- Check GPU utilization with `--debug`

### Model Loading Issues
- Ensure HF_TOKEN is set correctly
- Check internet connection for model download
- Verify CUDA installation for GPU support

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

Built on top of the [Dia TTS model](https://github.com/nari-labs/dia) by Nari Labs.