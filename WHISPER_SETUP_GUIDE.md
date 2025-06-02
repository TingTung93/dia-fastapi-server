# Whisper Integration Setup Guide

## Overview

The Dia TTS server now includes full OpenAI Whisper integration for automatic audio transcription. This enables automatic generation of transcripts for uploaded audio prompts, improving voice cloning quality and user experience.

## Installation

### 1. Install OpenAI Whisper

```bash
# Install OpenAI Whisper
pip install openai-whisper

# Or install with the server requirements
pip install -r requirements.txt
```

### 2. System Requirements

- **Python**: 3.8+
- **FFmpeg**: Required for audio processing
- **GPU**: CUDA-compatible GPU recommended for faster transcription
- **Memory**: 1-8GB depending on model size

#### Install FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html or use chocolatey:
```bash
choco install ffmpeg
```

### 3. GPU Support (Optional but Recommended)

For faster transcription with CUDA:
```bash
# Ensure PyTorch with CUDA is installed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

### Server Configuration

The server automatically detects Whisper availability and configures transcription settings:

```python
# Default configuration in server
SERVER_CONFIG = ServerConfig(
    auto_transcribe=True,           # Enable automatic transcription
    whisper_model_size="base",      # Model size: tiny, base, small, medium, large
    auto_discover_prompts=True      # Auto-discover and transcribe audio prompts
)
```

### Model Sizes

Choose the appropriate Whisper model size based on your needs:

| Model | Size | Speed | Accuracy | VRAM | Use Case |
|-------|------|-------|----------|------|----------|
| `tiny` | 39 MB | Fastest | Good | ~1GB | Testing/Development |
| `base` | 74 MB | Fast | Better | ~1GB | Default recommendation |
| `small` | 244 MB | Medium | Good | ~2GB | Balanced performance |
| `medium` | 769 MB | Slow | Very Good | ~5GB | High accuracy needed |
| `large` | 1550 MB | Slowest | Best | ~10GB | Maximum accuracy |

### Configure Model Size

Update server configuration:
```bash
# Via environment variable
export WHISPER_MODEL_SIZE="base"

# Or via API
curl -X PUT "http://localhost:7860/config" \
  -H "Content-Type: application/json" \
  -d '{"whisper_model_size": "base", "auto_transcribe": true}'
```

## Usage

### 1. Automatic Transcription

When you upload audio prompts, transcription happens automatically:

```bash
# Upload audio prompt - transcription happens automatically
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@voice_sample.wav"
```

### 2. Manual Transcription

Force transcription of existing audio prompts:

```bash
# Transcribe specific audio prompt
curl -X POST "http://localhost:7860/audio_prompts/my_voice/transcribe"

# Re-discover all prompts with forced transcription
curl -X POST "http://localhost:7860/audio_prompts/discover" \
  -H "Content-Type: application/json" \
  -d '{"force_retranscribe": true}'
```

### 3. Check Whisper Status

Monitor Whisper integration:

```bash
# Check Whisper availability and status
curl "http://localhost:7860/whisper/status"

# Load Whisper model manually
curl -X POST "http://localhost:7860/whisper/load"
```

### 4. Audio Prompt Discovery

Auto-discover and transcribe audio files:

```bash
# Discover all audio prompts in audio_prompts/ directory
curl -X POST "http://localhost:7860/audio_prompts/discover"
```

## File Structure

### Audio Prompt Directory

```
audio_prompts/
├── my_voice.wav              # Audio prompt file
├── my_voice.txt              # Auto-generated transcript
├── my_voice.reference.txt    # Manual reference transcript (highest priority)
├── another_voice.wav
├── another_voice.txt
└── custom_voice.wav
```

### Transcript Priority

The system uses transcripts in this priority order:

1. **`.reference.txt`** - Manual reference transcript (highest priority)
2. **`.txt`** - Generated or manual transcript
3. **Existing metadata** - Previously generated transcript
4. **Whisper transcription** - Automatic generation

## API Endpoints

### Whisper Management

```bash
# Get Whisper status
GET /whisper/status

# Load Whisper model
POST /whisper/load

# Get server configuration
GET /config

# Update server configuration  
PUT /config
```

### Audio Prompt Management

```bash
# Upload audio prompt (auto-transcribes)
POST /audio_prompts/upload

# List audio prompts with metadata
GET /audio_prompts/metadata

# Transcribe specific prompt
POST /audio_prompts/{prompt_id}/transcribe

# Update transcript manually
PUT /audio_prompts/{prompt_id}/transcript

# Discover and transcribe all prompts
POST /audio_prompts/discover
```

## Testing

### Run Whisper Integration Tests

```bash
# Test Whisper functionality
python test_whisper_integration.py
```

### Manual Testing

```bash
# 1. Start server with debug mode
python start_server.py --debug --show-prompts

# 2. Upload test audio file
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=test_voice" \
  -F "audio_file=@test_audio.wav"

# 3. Check transcription
curl "http://localhost:7860/audio_prompts/metadata/test_voice"

# 4. Test voice generation
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "test_voice"}' \
  --output test_generation.wav
```

## Troubleshooting

### Common Issues

#### 1. Whisper Not Available
```
⚠️  OpenAI Whisper not available. Install with: pip install openai-whisper
```
**Solution**: Install Whisper and restart server
```bash
pip install openai-whisper
python start_server.py
```

#### 2. FFmpeg Not Found
```
Error: ffmpeg not found
```
**Solution**: Install FFmpeg system-wide
```bash
# Linux
sudo apt install ffmpeg

# macOS  
brew install ffmpeg

# Windows
choco install ffmpeg
```

#### 3. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use smaller model: `"whisper_model_size": "tiny"`
- Force CPU mode: `device="cpu"` in model loading
- Free GPU memory: restart server

#### 4. Slow Transcription
**Solutions**:
- Use GPU if available
- Use smaller model (`tiny` or `base`)
- Ensure audio files are < 25MB
- Check CPU/GPU utilization

#### 5. Poor Transcription Quality
**Solutions**:
- Use larger model (`medium` or `large`)
- Ensure audio quality is good
- Use shorter audio clips (2-30 seconds)
- Manually create `.reference.txt` files

### Debug Mode

Enable detailed Whisper logging:

```bash
# Start server with debug output
python start_server.py --debug --show-prompts

# Check server logs for Whisper operations
curl "http://localhost:7860/logs" | grep -i whisper
```

### Performance Optimization

#### GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Monitor GPU usage during transcription
nvidia-smi -l 1
```

#### Model Optimization
```bash
# Use appropriate model size for your hardware
# tiny: Testing/Development
# base: Production default  
# small: Better accuracy
# medium/large: Maximum accuracy
```

## Integration with Voice Cloning

### Automatic Voice Enhancement

When transcripts are available, the server automatically:

1. **Prepends transcript** to generation text for better voice matching
2. **Associates transcript** with voice mappings
3. **Improves voice cloning** quality through context

### Example Voice Cloning Workflow

```bash
# 1. Upload audio sample
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=custom_speaker" \
  -F "audio_file=@speaker_sample.wav"

# 2. Create voice mapping (transcript auto-associated)
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "custom_voice",
    "style": "conversational",
    "primary_speaker": "S1", 
    "audio_prompt": "custom_speaker"
  }'

# 3. Generate with enhanced voice cloning
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this should sound like the custom speaker!",
    "voice_id": "custom_voice"
  }' \
  --output enhanced_voice.wav
```

## Best Practices

### Audio Quality
- **Duration**: 3-30 seconds optimal
- **Format**: WAV preferred, MP3 acceptable
- **Quality**: 44.1kHz, 16-bit minimum
- **Content**: Clear speech, minimal background noise

### Transcript Quality
- **Manual Review**: Check auto-generated transcripts
- **Reference Files**: Use `.reference.txt` for important voices
- **Consistency**: Keep transcript style consistent

### Performance
- **Model Selection**: Balance speed vs accuracy
- **GPU Usage**: Enable CUDA for production
- **File Management**: Keep audio files organized
- **Monitoring**: Use debug mode for troubleshooting

## Advanced Configuration

### Environment Variables

```bash
# Whisper model configuration
export WHISPER_MODEL_SIZE="base"
export WHISPER_DEVICE="cuda"  # or "cpu"

# Server configuration
export AUTO_TRANSCRIBE="true"
export AUTO_DISCOVER_PROMPTS="true"
```

### Custom Configuration

```python
# Custom server configuration
SERVER_CONFIG = ServerConfig(
    debug_mode=True,
    auto_transcribe=True,
    whisper_model_size="base",
    auto_discover_prompts=True
)
```

The Whisper integration is now ready for production use with automatic transcription, enhanced voice cloning, and comprehensive error handling.