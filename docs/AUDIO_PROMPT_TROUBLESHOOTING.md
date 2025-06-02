# Audio Prompt Troubleshooting Guide

## Problem: Empty Audio When Using Audio Prompts

If you're getting empty audio clips when using an audio prompt like `seraphina_voice.wav`, here's how to diagnose and fix the issue.

## Step 1: Verify Audio Prompt Upload

First, check if your audio prompt is properly uploaded:

```bash
# Check uploaded audio prompts
curl http://localhost:7860/audio_prompts
```

You should see something like:
```json
{
  "seraphina_voice": {
    "file_path": "audio_prompts/seraphina_voice.wav",
    "duration": 5.2,
    "sample_rate": 44100,
    "exists": true
  }
}
```

If not listed, upload it:
```bash
curl -X POST "http://localhost:7860/audio_prompts/upload" \
     -F "prompt_id=seraphina_voice" \
     -F "audio_file=@seraphina_voice.wav"
```

## Step 2: Create or Update Voice Mapping

The audio prompt needs to be associated with a voice. Check current voice mappings:

```bash
curl http://localhost:7860/voice_mappings
```

### Create a new voice with the audio prompt:
```bash
curl -X POST "http://localhost:7860/voice_mappings" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_id": "seraphina",
       "style": "elegant",
       "primary_speaker": "S1",
       "audio_prompt": "seraphina_voice",
       "audio_prompt_transcript": "Hello, this is Seraphina speaking with a clear and elegant voice."
     }'
```

### Or update an existing voice:
```bash
curl -X PUT "http://localhost:7860/voice_mappings/seraphina" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_id": "seraphina",
       "audio_prompt": "seraphina_voice",
       "audio_prompt_transcript": "Hello, this is Seraphina speaking with a clear and elegant voice."
     }'
```

## Step 3: Enable Debug Mode

Enable debug mode to see what's happening:

```bash
curl -X PUT "http://localhost:7860/config" \
     -H "Content-Type: application/json" \
     -d '{"debug_mode": true, "show_prompts": true}'
```

## Step 4: Test Generation

Test with your configured voice:

```bash
curl -X POST "http://localhost:7860/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, this is a test of the Seraphina voice. I hope it sounds elegant and clear.",
       "voice_id": "seraphina",
       "speed": 1.0
     }' \
     --output test_seraphina.wav
```

## Step 5: Check Server Logs

Look for these messages in the server output:
- `Audio Prompt: Yes` - Confirms audio prompt is being used
- `Warning: Audio prompt file not found` - Indicates file path issue
- Generation time and token speed information

## Common Issues and Solutions

### 1. Empty Audio Output
**Cause**: Audio prompt file not found or not properly loaded.
**Solution**: 
- Verify the file exists in `audio_prompts/` directory
- Check file permissions
- Ensure the prompt_id matches exactly (without .wav extension)

### 2. Audio Prompt Not Applied
**Cause**: Voice mapping not configured correctly.
**Solution**:
- Make sure `audio_prompt` field in voice mapping matches the prompt_id
- Include `audio_prompt_transcript` - it significantly helps voice cloning

### 3. Poor Voice Cloning
**Cause**: Audio prompt quality or transcript mismatch.
**Solution**:
- Use 3-30 seconds of clear speech
- Ensure transcript accurately represents the audio
- Audio should be 44.1kHz mono WAV
- Avoid background noise or music

### 4. Model Ignoring Audio Prompt
**Cause**: Text preprocessing removing speaker tags.
**Solution**:
- The processed text should include speaker tags: `[S1] text [S1]`
- Check debug output to see the processed text

## Python Debugging Script

Run the debugging script to check your setup:

```bash
python3 debug_audio_prompt.py
```

Or check manually:

```bash
python3 check_seraphina.py
```

## Advanced Troubleshooting

### Check if audio prompt is being passed to model:

In `server.py`, the audio prompt is passed as:
```python
audio_output = model_instance.generate(
    processed_text,
    audio_prompt=audio_prompt,  # This should be the file path
    **generation_params
)
```

### Verify the audio prompt path:

Add logging to see the exact path being used:
```python
if audio_prompt:
    print(f"DEBUG: Using audio prompt: {audio_prompt}")
    print(f"DEBUG: File exists: {os.path.exists(audio_prompt)}")
```

### Test with different parameters:

Sometimes adjusting generation parameters helps:
```json
{
  "text": "Test text",
  "voice_id": "seraphina",
  "temperature": 1.0,
  "cfg_scale": 2.5,
  "top_p": 0.9
}
```

## Expected Behavior

When working correctly, you should see:
1. Server logs showing "Audio Prompt: Yes"
2. Generation takes slightly longer (loading audio prompt)
3. Output audio matches the voice characteristics of the prompt
4. File size is normal (not just WAV headers)

## Need More Help?

1. Check the Dia model documentation for audio prompt requirements
2. Ensure your conda environment has all dependencies
3. Try with a known-good audio prompt file first
4. Check GPU memory usage during generation