# Dia FastAPI TTS Server - API Reference

Complete API documentation for the Dia text-to-speech server.

## Base URL

```
http://localhost:7860
```

## Interactive Documentation

- **Swagger UI**: http://localhost:7860/docs
- **ReDoc**: http://localhost:7860/redoc

## Authentication

No authentication required for local usage. For production deployments, consider implementing authentication layers.

---

## Core TTS Endpoints

### Generate Speech

#### `POST /generate`

Generate text-to-speech audio.

**Request Body:**
```json
{
  "text": "Hello, world!",
  "voice_id": "aria",
  "response_format": "wav",
  "speed": 1.0,
  "role": "assistant",
  "temperature": 0.7,
  "cfg_scale": 3.0,
  "top_p": 0.9,
  "max_tokens": 2048,
  "use_torch_compile": true
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to convert to speech (max 4096 chars) |
| `voice_id` | string | "aria" | Voice identifier |
| `response_format` | string | "wav" | Audio format (wav, mp3) |
| `speed` | float | 1.0 | Speech speed (0.25 - 4.0) |
| `role` | string | null | Speaker role (user, assistant, system) |
| `temperature` | float | null | Sampling temperature (0.1 - 2.0) |
| `cfg_scale` | float | null | Classifier-free guidance scale (1.0 - 10.0) |
| `top_p` | float | null | Top-p sampling (0.0 - 1.0) |
| `max_tokens` | int | null | Maximum tokens to generate (100 - 10000) |
| `use_torch_compile` | bool | null | Enable torch.compile optimization |

**Response:**
- **Content-Type**: `audio/wav` or `audio/mpeg`
- **Body**: Binary audio data

**Example:**
```bash
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "aria"}' \
  --output speech.wav
```

#### `POST /generate?async_mode=true`

Generate speech asynchronously (returns job ID).

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `async_mode` | bool | Enable asynchronous processing |

**Response:**
```json
{
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "status": "pending",
  "message": "Job queued for processing"
}
```

**Example:**
```bash
curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, async world!", "voice_id": "aria"}'
```

---

## OpenAI-Compatible Endpoint

### `POST /v1/audio/speech`

OpenAI-compatible speech generation endpoint for SillyTavern integration.

**Request Body:**
```json
{
  "model": "dia",
  "input": "Hello, world!",
  "voice": "aria",
  "response_format": "wav",
  "speed": 1.0
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | "dia" | Model name (always "dia") |
| `input` | string | required | Text to convert to speech |
| `voice` | string | "aria" | Voice identifier |
| `response_format` | string | "wav" | Audio format (wav, mp3) |
| `speed` | float | 1.0 | Speech speed (0.25 - 4.0) |

**Response:**
- **Content-Type**: `audio/wav` or `audio/mpeg`  
- **Body**: Binary audio data

**Example:**
```bash
curl -X POST "http://localhost:7860/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model": "dia", "input": "Hello from SillyTavern!", "voice": "luna"}' \
  --output openai_speech.wav
```

---

## Voice Management

### List Voices

#### `GET /voices`

Get list of available voices.

**Response:**
```json
{
  "voices": [
    {
      "name": "Aria",
      "voice_id": "aria",
      "preview_url": null
    },
    {
      "name": "Atlas", 
      "voice_id": "atlas",
      "preview_url": null
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:7860/voices"
```

### Voice Mappings

#### `GET /voice_mappings`

Get current voice mappings configuration.

**Response:**
```json
{
  "aria": {
    "style": "neutral",
    "primary_speaker": "S1",
    "audio_prompt": null,
    "audio_prompt_transcript": null
  },
  "custom_voice": {
    "style": "conversational",
    "primary_speaker": "S1", 
    "audio_prompt": "my_audio_prompt",
    "audio_prompt_transcript": "Hello, this is my custom voice sample."
  }
}
```

#### `POST /voice_mappings`

Create or update voice mapping.

**Request Body:**
```json
{
  "voice_id": "custom_voice",
  "style": "conversational",
  "primary_speaker": "S1",
  "audio_prompt": "my_audio_prompt",
  "audio_prompt_transcript": "Custom transcript text"
}
```

#### `DELETE /voice_mappings/{voice_id}`

Delete voice mapping.

**Example:**
```bash
curl -X DELETE "http://localhost:7860/voice_mappings/custom_voice"
```

---

## Audio Prompt Management

### Upload Audio Prompt

#### `POST /audio_prompts/upload`

Upload audio file for voice cloning.

**Request (multipart/form-data):**
```bash
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@voice_sample.wav"
```

**Form Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `prompt_id` | string | Unique identifier for the audio prompt |
| `audio_file` | file | Audio file (WAV, MP3, etc.) |

**Response:**
```json
{
  "message": "Audio prompt uploaded successfully",
  "prompt_id": "my_voice",
  "file_path": "audio_prompts/my_voice.wav",
  "duration": 3.45,
  "sample_rate": 44100,
  "transcript": "Auto-generated transcript if Whisper is enabled"
}
```

### List Audio Prompts

#### `GET /audio_prompts`

Get list of uploaded audio prompts.

**Response:**
```json
{
  "audio_prompts": [
    {
      "prompt_id": "my_voice",
      "file_path": "audio_prompts/my_voice.wav", 
      "duration": 3.45,
      "sample_rate": 44100
    }
  ]
}
```

#### `GET /audio_prompts/metadata`

Get detailed metadata for all audio prompts.

**Response:**
```json
{
  "audio_prompts": [
    {
      "prompt_id": "my_voice",
      "file_path": "audio_prompts/my_voice.wav",
      "duration": 3.45,
      "sample_rate": 44100,
      "transcript": "Hello, this is a voice sample for cloning.",
      "transcript_source": "whisper_auto",
      "file_size": 152064,
      "created_at": "2025-06-02T10:30:00Z"
    }
  ]
}
```

#### `GET /audio_prompts/metadata/{prompt_id}`

Get metadata for specific audio prompt.

#### `DELETE /audio_prompts/{prompt_id}`

Delete audio prompt.

**Example:**
```bash
curl -X DELETE "http://localhost:7860/audio_prompts/my_voice"
```

### Audio Prompt Discovery

#### `POST /audio_prompts/discover`

Scan audio_prompts/ directory for new files.

**Request Body:**
```json
{
  "force_retranscribe": false
}
```

**Response:**
```json
{
  "message": "Audio prompt discovery completed",
  "total_prompts": 5,
  "discovered": [
    {
      "prompt_id": "new_voice",
      "file_path": "audio_prompts/new_voice.wav",
      "transcript": "Automatically discovered and transcribed"
    }
  ]
}
```

---

## Whisper Integration

### Transcribe Audio Prompt

#### `POST /audio_prompts/{prompt_id}/transcribe`

Generate transcript for specific audio prompt.

**Response:**
```json
{
  "message": "Transcription completed",
  "prompt_id": "my_voice", 
  "transcript": "Hello, this is my voice sample for cloning.",
  "saved_to": "audio_prompts/my_voice.txt"
}
```

#### `PUT /audio_prompts/{prompt_id}/transcript`

Update transcript manually.

**Request Body:**
```json
{
  "transcript": "Manually corrected transcript text"
}
```

### Whisper Status

#### `GET /whisper/status`

Check Whisper availability and configuration.

**Response:**
```json
{
  "available": true,
  "model_loaded": true,
  "model_size": "base",
  "auto_transcribe": true,
  "device": "cuda"
}
```

#### `POST /whisper/load`

Load or reload Whisper model.

**Response:**
```json
{
  "message": "Whisper model loaded successfully",
  "model_size": "base",
  "device": "cuda"
}
```

---

## Job Management (Async Mode)

### Check Job Status

#### `GET /jobs/{job_id}`

Get status of asynchronous job.

**Response:**
```json
{
  "job_id": "12345678-1234-1234-1234-123456789abc",
  "status": "completed",
  "created_at": "2025-06-02T10:30:00Z",
  "completed_at": "2025-06-02T10:30:05Z",
  "duration": 5.2,
  "result_url": "/jobs/12345678-1234-1234-1234-123456789abc/result"
}
```

**Status Values:**
- `pending` - Job queued
- `processing` - Currently generating
- `completed` - Finished successfully  
- `failed` - Error occurred

#### `GET /jobs/{job_id}/result`

Download result of completed job.

**Response:**
- **Content-Type**: `audio/wav` or `audio/mpeg`
- **Body**: Binary audio data

#### `DELETE /jobs/{job_id}`

Cancel job or delete result.

### List Jobs

#### `GET /jobs`

Get list of recent jobs.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (pending, processing, completed, failed) |
| `limit` | int | Maximum number of jobs to return (default: 50) |

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "12345678-1234-1234-1234-123456789abc",
      "status": "completed", 
      "created_at": "2025-06-02T10:30:00Z",
      "text_preview": "Hello, world!"
    }
  ]
}
```

---

## Server Management

### Health Check

#### `GET /health`

Check server health and status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_count": 2,
  "worker_count": 4,
  "active_jobs": 1,
  "uptime": "2h 45m 12s"
}
```

### Server Configuration

#### `GET /config`

Get current server configuration.

**Response:**
```json
{
  "debug_mode": false,
  "save_outputs": true,
  "show_prompts": false,
  "output_retention_hours": 24,
  "auto_discover_prompts": true,
  "auto_transcribe": true,
  "whisper_model_size": "base",
  "gpu_mode": "auto",
  "worker_count": 4,
  "torch_compile_enabled": true
}
```

#### `PUT /config`

Update server configuration.

**Request Body:**
```json
{
  "debug_mode": true,
  "save_outputs": true,
  "auto_transcribe": true,
  "whisper_model_size": "small"
}
```

### Server Stats

#### `GET /stats`

Get detailed server statistics.

**Response:**
```json
{
  "requests": {
    "total": 1234,
    "successful": 1200,
    "failed": 34,
    "success_rate": 97.2
  },
  "performance": {
    "avg_generation_time": 2.1,
    "avg_tokens_per_second": 450,
    "total_audio_generated": "45m 23s"
  },
  "resources": {
    "cpu_usage": 15.3,
    "memory_usage": 68.7,
    "gpu_usage": [85.2, 78.9],
    "gpu_memory": [4.2, 3.8]
  }
}
```

---

## WebSocket API (Real-time)

### Real-time Generation

#### `WS /ws/generate`

WebSocket endpoint for real-time TTS generation with progress updates.

**Message Format:**
```json
{
  "type": "generate",
  "data": {
    "text": "Hello, world!",
    "voice_id": "aria"
  }
}
```

**Response Messages:**
```json
// Progress update
{
  "type": "progress",
  "job_id": "12345",
  "progress": 45,
  "stage": "generating_tokens"
}

// Completion
{
  "type": "complete",
  "job_id": "12345", 
  "audio_url": "/jobs/12345/result"
}

// Error
{
  "type": "error",
  "message": "Generation failed"
}
```

---

## Error Responses

### Standard Error Format

```json
{
  "detail": "Error message description",
  "error_code": "INVALID_VOICE_ID",
  "timestamp": "2025-06-02T10:30:00Z"
}
```

### Common HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Invalid input format |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_VOICE_ID` | Voice ID not found |
| `TEXT_TOO_LONG` | Text exceeds maximum length |
| `MODEL_NOT_LOADED` | Dia model not available |
| `WHISPER_NOT_AVAILABLE` | Whisper not installed |
| `GPU_OUT_OF_MEMORY` | Insufficient GPU memory |
| `AUDIO_PROCESSING_ERROR` | Audio file processing failed |

---

## Rate Limiting

### Default Limits

- **Text Generation**: 10 requests/minute per IP
- **File Upload**: 5 uploads/minute per IP  
- **Job Queries**: 100 requests/minute per IP

### Headers

Rate limit information included in response headers:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1638360000
```

---

## SDKs and Examples

### Python SDK Example

```python
import requests
import json

class DiaTTSClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
    
    def generate_speech(self, text, voice_id="aria", **kwargs):
        """Generate speech from text"""
        data = {"text": text, "voice_id": voice_id, **kwargs}
        response = requests.post(f"{self.base_url}/generate", json=data)
        return response.content
    
    def upload_audio_prompt(self, prompt_id, audio_file_path):
        """Upload audio prompt for voice cloning"""
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'prompt_id': prompt_id}
            response = requests.post(
                f"{self.base_url}/audio_prompts/upload",
                files=files, data=data
            )
        return response.json()
    
    def list_voices(self):
        """Get available voices"""
        response = requests.get(f"{self.base_url}/voices")
        return response.json()

# Usage
client = DiaTTSClient()
audio = client.generate_speech("Hello, world!", voice_id="aria")
with open("output.wav", "wb") as f:
    f.write(audio)
```

### JavaScript/Node.js Example

```javascript
class DiaTTSClient {
    constructor(baseUrl = 'http://localhost:7860') {
        this.baseUrl = baseUrl;
    }
    
    async generateSpeech(text, voiceId = 'aria', options = {}) {
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice_id: voiceId, ...options })
        });
        return response.arrayBuffer();
    }
    
    async listVoices() {
        const response = await fetch(`${this.baseUrl}/voices`);
        return response.json();
    }
}

// Usage
const client = new DiaTTSClient();
const audio = await client.generateSpeech("Hello, world!", "aria");
```

### cURL Examples

```bash
# Basic generation
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "aria"}' \
  --output speech.wav

# Async generation
curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, async!", "voice_id": "luna"}'

# Upload audio prompt
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@sample.wav"

# Check job status
curl "http://localhost:7860/jobs/12345678-1234-1234-1234-123456789abc"
```

---

This API reference covers all endpoints and functionality of the Dia FastAPI TTS Server. For interactive testing, visit http://localhost:7860/docs when the server is running.