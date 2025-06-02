# Audio Prompt Transcript Guide

## Why You're Getting Gibberish Output

When you use an audio prompt without a transcript, the Dia model tries to clone the voice but doesn't know what words are being spoken in the reference audio. This results in:
- Gibberish that sounds like the target voice
- Garbled speech with the right tone but wrong words
- Sometimes repeating sounds or phonemes

## The Solution: Audio Prompt Transcript

The `audio_prompt_transcript` field tells the model exactly what is being said in your audio prompt file. This is **critical** for proper voice cloning.

## How to Set the Transcript

### Method 1: When Creating a Voice

```bash
curl -X POST "http://localhost:7860/voice_mappings" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_id": "seraphina",
       "style": "elegant",
       "primary_speaker": "S1",
       "audio_prompt": "seraphina_voice",
       "audio_prompt_transcript": "Hello, my name is Seraphina. I speak with a warm, elegant voice that is clear and articulate."
     }'
```

### Method 2: Updating an Existing Voice

```bash
curl -X PUT "http://localhost:7860/voice_mappings/seraphina" \
     -H "Content-Type: application/json" \
     -d '{
       "voice_id": "seraphina",
       "audio_prompt_transcript": "Hello, my name is Seraphina. I speak with a warm, elegant voice that is clear and articulate."
     }'
```

### Method 3: Using the Python Script

```bash
python3 fix_audio_prompt_transcript.py
```

## How It Works

When you generate audio with a voice that has both `audio_prompt` and `audio_prompt_transcript`:

1. The model loads your audio prompt file (e.g., `seraphina_voice.wav`)
2. It prepends the transcript to your generation text
3. This helps the model understand the voice characteristics and speaking style
4. The output matches the voice from the audio prompt

### Example Flow:

Your request:
```json
{
  "text": "Welcome to our presentation.",
  "voice_id": "seraphina"
}
```

What the model actually processes:
```
"Hello, my name is Seraphina. I speak with a warm, elegant voice that is clear and articulate. Welcome to our presentation."
```

The model uses the transcript portion to understand the voice, then generates the new text in that voice.

## Best Practices for Transcripts

### 1. Be Exact
The transcript should match your audio prompt word-for-word:
- ✅ Include all words spoken
- ✅ Include fillers like "um" or "uh" if present
- ✅ Match the punctuation to the speech rhythm

### 2. Length Matters
- Ideal: 1-3 sentences (5-15 seconds of audio)
- Too short: Not enough voice information
- Too long: May affect generation quality

### 3. Content Tips
Good transcript examples:
- "Hello, my name is Seraphina. I have a warm and sophisticated voice."
- "This is David speaking. My voice is deep and authoritative, perfect for narration."
- "Hi there! I'm Luna, and I speak with energy and enthusiasm!"

### 4. What to Include
- The speaker's name (helps with identity)
- Voice characteristics description
- Natural speech patterns
- Any unique pronunciations

## Creating Your Audio Prompt

### Recording Tips:
1. **Clear audio**: No background noise or music
2. **Natural speech**: Speak normally, not too fast or slow
3. **Good microphone**: Use decent recording equipment
4. **Consistent volume**: Avoid volume spikes
5. **WAV format**: 44.1kHz, 16-bit, mono

### Sample Script Templates:

**Professional Voice:**
```
"Good morning. My name is [Name], and I speak with a clear, professional tone 
that's perfect for business presentations and educational content."
```

**Friendly Voice:**
```
"Hey there! I'm [Name], and I love speaking with warmth and enthusiasm. 
My voice is friendly and approachable, great for casual conversations!"
```

**Narrator Voice:**
```
"Welcome. I am [Name], your narrator. I speak with measured pace and clear 
articulation, ideal for storytelling and documentary narration."
```

## Troubleshooting

### Still Getting Gibberish?
1. Verify transcript is set: `curl http://localhost:7860/voice_mappings`
2. Check transcript accuracy - re-listen to your audio
3. Ensure audio quality is good
4. Try adjusting generation parameters

### Voice Doesn't Match?
1. Audio prompt too short (aim for 5-15 seconds)
2. Background noise in audio prompt
3. Transcript doesn't match audio exactly
4. Try different cfg_scale values (2.5-4.0)

### Generation Fails?
1. Check audio file format (must be readable by soundfile)
2. Verify file permissions
3. Enable debug mode to see errors

## Quick Test

After setting the transcript:

```bash
# Test with simple text
curl -X POST "http://localhost:7860/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Testing one, two, three. This should sound clear.",
       "voice_id": "seraphina"
     }' \
     --output test_clear.wav

# Compare with/without transcript
# The difference should be dramatic!
```

## Remember

The transcript is not just metadata - it's actively used by the model to understand your audio prompt. Without it, the model is essentially trying to clone a voice while "deaf" to what's being said, resulting in gibberish output that only captures the acoustic properties of the voice.