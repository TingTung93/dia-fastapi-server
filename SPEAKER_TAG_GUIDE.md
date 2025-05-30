# Speaker Tag Guide for Dia Model

## The Problem: Wrong Gender Voice

If your female audio prompt is producing a masculine voice, it's likely because of the **speaker tag** setting.

## Quick Fix

Change your voice's `primary_speaker` from `S1` to `S2`:

```bash
curl -X PUT "http://localhost:7860/voice_mappings/seraphina" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_id": "seraphina",
       "primary_speaker": "S2"
     }'
```

## Understanding Speaker Tags

The Dia model has two primary speaker tags that significantly affect voice characteristics:

| Tag | Typical Characteristics | Best For |
|-----|------------------------|----------|
| [S1] | Masculine, deeper, neutral | Male voices, neutral narration |
| [S2] | Feminine, higher, expressive | Female voices, varied expression |

## How It Works

When you generate speech, the text is wrapped with speaker tags:
- With S1: `[S1] Your text here [S1]`
- With S2: `[S2] Your text here [S2]`

These tags influence the voice characteristics **even when using an audio prompt**.

## Complete Setup for Female Voice

```bash
# 1. Upload female audio prompt
curl -X POST "http://localhost:7860/audio_prompts/upload" \
     -F "prompt_id=seraphina_voice" \
     -F "audio_file=@seraphina_voice.wav"

# 2. Create voice with S2 speaker tag
curl -X POST "http://localhost:7860/voice_mappings" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_id": "seraphina",
       "style": "elegant",
       "primary_speaker": "S2",  # ← Critical for female voice!
       "audio_prompt": "seraphina_voice",
       "audio_prompt_transcript": "Hello, I am Seraphina. I have a warm, feminine voice."
     }'
```

## Testing Different Configurations

### Test with S1 (masculine):
```bash
curl -X POST "http://localhost:7860/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "[S1] This will sound more masculine [S1]",
       "voice_id": "seraphina"
     }' --output test_s1.wav
```

### Test with S2 (feminine):
```bash
curl -X POST "http://localhost:7860/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "[S2] This will sound more feminine [S2]",
       "voice_id": "seraphina"
     }' --output test_s2.wav
```

## Advanced Tips

### 1. Mixed Speakers
You can use both tags in one generation for dialogue:
```
"[S2] Hello, said the woman. [S1] Hello, replied the man. [S2] How are you today?"
```

### 2. Role-Based Override
The `role` parameter can override speaker selection:
```json
{
  "text": "Hello there",
  "voice_id": "seraphina",
  "role": "assistant"  // Forces S2
}
```

### 3. Fine-Tuning Gender Perception
Adjust these parameters for better gender matching:
- **temperature**: Lower (0.8) for consistency
- **cfg_scale**: Higher (3.5-4.0) for stronger audio prompt influence
- **top_p**: Lower (0.9) for more predictable output

### 4. Default Voice Mappings
Built-in voices and their default speakers:
- alloy, echo, nova, shimmer → S1 (masculine/neutral)
- fable, onyx → S2 (feminine/varied)

## Troubleshooting

### Still sounds masculine with S2?
1. Check audio prompt quality - ensure it's clearly feminine
2. Verify transcript matches exactly
3. Try higher cfg_scale (4.0-5.0)
4. Use longer audio prompt (15-20 seconds)

### Voice is inconsistent?
1. Lower temperature to 0.8-1.0
2. Ensure consistent speaker tags throughout
3. Keep audio prompt transcript accurate

### Want more control?
1. Use explicit tags in every request: `"[S2] text [S2]"`
2. Experiment with base voices (fable vs custom)
3. Try different audio prompt recordings

## Quick Diagnostic

Run this to check and fix your setup:
```bash
python3 fix_voice_gender.py
```

This tool will:
1. Show current speaker tag setting
2. Let you switch between S1/S2
3. Generate test files for comparison
4. Provide specific recommendations