# Dia TTS API Quick Reference

## 🎯 Most Common Endpoints

### Generate Speech (Sync)
```bash
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[S1] Your text here",
    "voice_id": "seraphina_voice",
    "seed": 42
  }' --output speech.wav
```

### OpenAI Compatible
```bash
curl -X POST "http://localhost:7860/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text here",
    "voice": "seraphina_voice",
    "seed": 42
  }' --output speech.wav
```

### Check Available Voices
```bash
curl "http://localhost:7860/voices"
```

### Create Voice Mapping
```bash
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "my_voice",
    "style": "conversational",
    "primary_speaker": "S1",
    "audio_prompt": "my_audio_id"
  }'
```

### Upload Audio Prompt
```bash
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_audio_id" \
  -F "audio_file=@voice_sample.wav"
```

---

## 🔄 Reproducible Generation

### With Fixed Seed (Recommended)
```json
{
  "text": "[S1] Consistent output every time",
  "voice_id": "seraphina_voice",
  "seed": 42,
  "temperature": 1.2,
  "cfg_scale": 3.0,
  "top_p": 0.95
}
```

---

## 📊 System Status

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Server health check |
| `GET /voices` | List available voices |
| `GET /gpu/status` | GPU memory and status |
| `GET /queue/stats` | Job queue statistics |

---

## ⚡ Quick Setup

1. **Start Server**: `python start_server.py`
2. **Upload Voice**: Use `/audio_prompts/upload`
3. **Create Mapping**: Use `/voice_mappings`
4. **Generate**: Use `/generate` with seed

---

## 🎯 Parameters at a Glance

| Parameter | Type | Required | Range | Example |
|-----------|------|----------|-------|---------|
| `text` | string | ✅ | 1-4096 chars | `"[S1] Hello"` |
| `voice_id` | string | ✅ | - | `"seraphina_voice"` |
| `seed` | integer | ❌ | any | `42` |
| `temperature` | float | ❌ | 0.1-2.0 | `1.2` |
| `cfg_scale` | float | ❌ | 1.0-10.0 | `3.0` |
| `top_p` | float | ❌ | 0.0-1.0 | `0.95` |
| `speed` | float | ❌ | 0.25-4.0 | `1.0` |

---

## 🚨 Common Errors

| Error | Solution |
|-------|----------|
| `Voice 'X' not found` | Check `/voices` or create mapping |
| `Text cannot be empty` | Provide non-empty text |
| `Model not loaded` | Check `/health` endpoint |
| `Job not completed` | Wait or check `/jobs/{id}` |

---

## 🧪 Testing

```bash
# Test seed functionality
python test_seed_functionality.py

# Health check
curl "http://localhost:7860/health"

# Get voice list
curl "http://localhost:7860/voices" | python -m json.tool
``` 