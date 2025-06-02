# Dia TTS Server API Documentation

## Overview

The Dia TTS Server provides a FastAPI-based REST API for text-to-speech generation using the Dia 1.6B model. It supports both synchronous and asynchronous generation, voice cloning with audio prompts, and reproducible generation with seed parameters.

**Base URL**: `http://localhost:7860`  
**API Version**: 1.0.0  
**Compatible with**: SillyTavern, OpenAI Audio API

---

## Authentication

The server accepts any bearer token for authentication (optional):

```bash
curl -H "Authorization: Bearer your-token-here" ...
```

---

## Core Generation Endpoints

### 1. Main Generation Endpoint

**`POST /generate`**

Primary TTS generation endpoint supporting both sync and async modes.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | ✅ | - | Text to convert to speech (max 4096 chars) |
| `voice_id` | string | ✅ | - | Voice identifier (no default voices) |
| `response_format` | string | ❌ | `"wav"` | Audio format (`wav`, `mp3`) |
| `speed` | float | ❌ | `1.0` | Speech speed (0.25-4.0) |
| `role` | string | ❌ | `null` | Speaker role (`user`, `assistant`, `system`) |
| `temperature` | float | ❌ | `null` | Sampling temperature (0.1-2.0) |
| `cfg_scale` | float | ❌ | `null` | Classifier-free guidance scale (1.0-10.0) |
| `top_p` | float | ❌ | `null` | Top-p sampling (0.0-1.0) |
| `max_tokens` | integer | ❌ | `null` | Maximum tokens to generate (100-10000) |
| `use_torch_compile` | boolean | ❌ | `null` | Enable torch.compile optimization |
| **`seed`** | integer | ❌ | `null` | **Random seed for reproducible generation** |

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_mode` | boolean | `false` | Return job ID for async processing |

#### Request Examples

**Synchronous with Seed (Recommended):**
```json
{
  "text": "[S1] Hello world! [S2] How are you today?",
  "voice_id": "seraphina_voice",
  "seed": 42,
  "temperature": 1.2,
  "cfg_scale": 3.0,
  "top_p": 0.95,
  "speed": 1.0
}
```

**Asynchronous Generation:**
```bash
curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[S1] Long text for background processing",
    "voice_id": "seraphina_voice",
    "seed": 1234
  }'
```

#### Response (Sync Mode)
- **Content-Type**: `audio/wav`
- **Headers**: 
  - `Content-Disposition: attachment; filename=speech.wav`
  - `X-Generation-ID: <log_id>` (if debug mode enabled)

#### Response (Async Mode)
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Job queued for processing"
}
```

---

### 2. OpenAI Compatible Endpoint

**`POST /v1/audio/speech`**

SillyTavern and OpenAI-compatible TTS endpoint.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | ✅ | - | Text to convert to speech |
| `voice` | string | ✅ | - | Voice identifier |
| `model` | string | ❌ | `"dia"` | Model name |
| `response_format` | string | ❌ | `"wav"` | Audio format |
| `speed` | float | ❌ | `1.0` | Speech speed |
| **`seed`** | integer | ❌ | `null` | **Random seed for reproducible generation** |

#### Request Example

```json
{
  "input": "Hello, this is a test message",
  "voice": "seraphina_voice",
  "model": "dia",
  "speed": 1.0,
  "seed": 42
}
```

#### cURL Example

```bash
curl -X POST "http://localhost:7860/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "voice": "seraphina_voice",
    "seed": 2024
  }'
```

---

## Voice Management

### 3. List Voices

**`GET /voices`**

Get all available voices and their configurations.

#### Response Example

```json
{
  "voices": [
    {
      "id": "seraphina_voice",
      "name": "seraphina_voice",
      "style": "conversational",
      "primary_speaker": "S1",
      "has_audio_prompt": true,
      "preview_url": "/preview/seraphina_voice"
    }
  ]
}
```

### 4. Voice Mappings

**`GET /voice_mappings`** - Get voice mappings  
**`POST /voice_mappings`** - Create voice mapping  
**`PUT /voice_mappings/{voice_id}`** - Update voice mapping  
**`DELETE /voice_mappings/{voice_id}`** - Delete voice mapping  

#### Create Voice Mapping Example

```json
{
  "voice_id": "my_custom_voice",
  "style": "neutral",
  "primary_speaker": "S1",
  "audio_prompt": "my_audio_prompt_id",
  "audio_prompt_transcript": "Transcript of the audio prompt"
}
```

### 5. Voice Preview

**`GET /preview/{voice_id}`**

Generate a preview sample for testing voices.

```bash
curl "http://localhost:7860/preview/seraphina_voice" > preview.wav
```

---

## Audio Prompt Management

### 6. Upload Audio Prompt

**`POST /audio_prompts/upload`**

Upload audio files for voice cloning.

#### Form Data Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt_id` | string | ✅ | Unique identifier for the audio prompt |
| `audio_file` | file | ✅ | Audio file (WAV, MP3, FLAC, etc.) |

#### cURL Example

```bash
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@/path/to/voice_sample.wav"
```

#### Response

```json
{
  "message": "Audio prompt 'my_voice' uploaded successfully",
  "duration": 8.55,
  "sample_rate": 44100,
  "original_sample_rate": 22050,
  "channels": "mono"
}
```

### 7. Audio Prompt Discovery

**`POST /audio_prompts/discover`**

Automatically discover audio prompts and transcribe them.

#### Request

```json
{
  "force_retranscribe": false
}
```

### 8. Transcription Management

**`POST /audio_prompts/{prompt_id}/transcribe`** - Transcribe with Whisper  
**`PUT /audio_prompts/{prompt_id}/transcript`** - Update transcript manually  

#### Manual Transcript Update

```json
{
  "transcript": "This is the spoken content of the audio prompt",
  "prompt_id": "my_voice"
}
```

---

## Job Management (Async Mode)

### 9. Job Status

**`GET /jobs/{job_id}`** - Get job status  
**`GET /jobs/{job_id}/result`** - Download completed result  
**`DELETE /jobs/{job_id}`** - Cancel job  

#### Job Status Response

```json
{
  "id": "job-uuid",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:05Z",
  "text": "Generated text",
  "voice_id": "seraphina_voice",
  "seed": 42,
  "generation_time": 4.2,
  "worker_id": "worker_1"
}
```

### 10. Queue Management

**`GET /jobs`** - List jobs with filtering  
**`GET /queue/stats`** - Get queue statistics  
**`DELETE /jobs`** - Clear completed jobs  

---

## System Information

### 11. Health Check

**`GET /health`**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1642234567.89
}
```

### 12. GPU Status

**`GET /gpu/status`**

```json
{
  "gpu_mode": "multi",
  "gpu_count": 2,
  "allowed_gpus": [0, 1],
  "use_multi_gpu": true,
  "models_loaded": {
    "gpu_0": true,
    "gpu_1": true
  },
  "gpu_memory": {
    "gpu_0": {
      "allocated_gb": 3.2,
      "reserved_gb": 4.1,
      "total_gb": 24.0,
      "free_gb": 19.9
    }
  }
}
```

### 13. Configuration

**`GET /config`** - Get server configuration  
**`PUT /config`** - Update server configuration  

#### Configuration Example

```json
{
  "debug_mode": false,
  "save_outputs": false,
  "show_prompts": false,
  "output_retention_hours": 24,
  "auto_discover_prompts": true,
  "auto_transcribe": true,
  "whisper_model_size": "base"
}
```

---

## Reproducible Generation with Seeds

### Key Features

1. **Deterministic Output**: Same seed + same parameters = consistent results
2. **Cross-Platform**: Works on single/multi-GPU setups
3. **Thread-Safe**: Isolated per generation request
4. **Debug Support**: Seed values shown in debug output

### Best Practices

```json
{
  "text": "[S1] Your text here",
  "voice_id": "voice_with_audio_prompt",
  "seed": 42,
  "temperature": 1.2,
  "cfg_scale": 3.0,
  "top_p": 0.95
}
```

### Consistency Levels

| Configuration | Consistency | Use Case |
|---------------|-------------|----------|
| Seed + Audio Prompt + Fixed Params | **Highest** | Production, A/B testing |
| Seed + Fixed Params | **Medium** | General reproducibility |
| Seed Only | **Basic** | Quick testing |
| No Seed | **Random** | Creative generation |

---

## Error Handling

### Common HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid parameters, empty text |
| `404` | Not Found | Voice ID not found, job not found |
| `425` | Too Early | Job not completed yet |
| `500` | Server Error | Model error, generation failure |
| `503` | Service Unavailable | Worker pool not available |

### Error Response Format

```json
{
  "detail": "Voice 'invalid_voice' not found. Available voices: ['seraphina_voice']"
}
```

---

## Rate Limiting & Performance

### Recommendations

- **Concurrent Requests**: Up to number of GPU workers
- **Request Timeout**: 60 seconds for generation
- **Text Length**: Keep under 4096 characters
- **Audio Prompts**: 3-30 seconds, under 25MB

### Performance Optimization

```json
{
  "text": "[S1] Optimized request",
  "voice_id": "cached_voice",
  "seed": 42,
  "temperature": 1.2,
  "use_torch_compile": true
}
```

---

## SDK Examples

### Python

```python
import requests

# Reproducible generation
response = requests.post("http://localhost:7860/generate", json={
    "text": "[S1] Hello world",
    "voice_id": "seraphina_voice",
    "seed": 42,
    "temperature": 1.2
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### JavaScript

```javascript
const response = await fetch('http://localhost:7860/v1/audio/speech', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: "Hello world",
    voice: "seraphina_voice",
    seed: 42
  })
});

const audioBlob = await response.blob();
```

### cURL

```bash
# Generate with seed
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text":"[S1] Test", "voice_id":"seraphina_voice", "seed":42}' \
  --output result.wav

# Check generation status
curl "http://localhost:7860/health"
```

---

## Migration Guide

### From v0.x to v1.0

1. **Voice Changes**: No default voices - create explicit voice mappings
2. **Seed Support**: Add `seed` parameter for reproducible generation
3. **Error Handling**: Updated error messages include available voices
4. **Audio Prompts**: Enhanced automatic discovery and transcription

### Updating Existing Code

```python
# Before (v0.x)
response = requests.post("/generate", json={
    "text": "Hello",
    "voice_id": "alloy"  # Default voice
})

# After (v1.0)
response = requests.post("/generate", json={
    "text": "Hello", 
    "voice_id": "seraphina_voice",  # Explicit voice mapping
    "seed": 42  # For reproducibility
})
```

---

## Support & Troubleshooting

### Common Issues

1. **"Voice not found"**: Create voice mapping first
2. **"Model not loaded"**: Check `/health` endpoint
3. **"Empty text"**: Ensure text parameter is not empty
4. **Inconsistent results**: Use seeds with audio prompts

### Debug Mode

Enable debug mode for detailed logging:

```json
{
  "debug_mode": true,
  "show_prompts": true
}
```

### Test Script

Use the provided test script to verify functionality:

```bash
python test_seed_functionality.py --url http://localhost:7860
```

---

## Changelog

### v1.0.0 (Current)
- ✅ Added seed parameter for reproducible generation
- ✅ Removed default voices (zero built-in voices)
- ✅ Enhanced audio prompt discovery with Whisper
- ✅ Improved error handling and validation
- ✅ Multi-GPU support with load balancing
- ✅ Comprehensive API documentation

---

*For more information about specific features, see the detailed guides:*
- [Seed Parameter Guide](SEED_PARAMETER_GUIDE.md)
- [Zero Voices Configuration](ZERO_VOICES_CHANGES.md) 