# Zero Built-in Voices Configuration

## Changes Made ✅

The server has been successfully configured to start with **zero built-in voices**. All voices must now be explicitly configured.

## What Was Changed

### 1. Voice Mapping Dictionary (`src/server.py`)

**Before:**
```python
VOICE_MAPPING: Dict[str, Dict[str, Any]] = {
    "aria": {"style": "neutral", "primary_speaker": "S1", ...},
    "atlas": {"style": "calm", "primary_speaker": "S1", ...}, 
    "luna": {"style": "expressive", "primary_speaker": "S2", ...},
    "kai": {"style": "friendly", "primary_speaker": "S1", ...},
    "zara": {"style": "deep", "primary_speaker": "S2", ...},
    "nova": {"style": "bright", "primary_speaker": "S1", ...},
}
```

**After:**
```python
# Server starts with zero built-in voices - all voices must be explicitly configured
VOICE_MAPPING: Dict[str, Dict[str, Any]] = {}
```

### 2. API Model Validation

**Updated models to require voice_id:**
- `TTSRequest.voice_id`: Now required field (no default)
- `TTSGenerateRequest.voice_id`: Now required field
- OpenAI endpoint validation: Rejects empty voice strings

### 3. Function Signatures

**Updated `generate_audio_from_text()`:**
```python
# Before: voice_id: str = "alloy"
# After:  voice_id: str  (required parameter)
```

### 4. Delete Endpoint

**Removed protection for "default" voices:**
- Can now delete any voice (no restricted voice names)
- Simplified deletion logic

### 5. Test Configuration (`tests/conftest.py`)

**Added test voice fixtures:**
```python
@pytest.fixture
def test_voice_mappings():
    return {
        "test_voice": {"style": "neutral", "primary_speaker": "S1", ...},
        "alloy": {"style": "neutral", "primary_speaker": "S1", ...}  # For backward compatibility
    }
```

## Current Behavior

### ✅ Server Startup
- Starts with empty voice mapping: `{}`
- No voices available until explicitly created
- Audio prompt discovery still works (creates audio prompts without voice mappings)

### ✅ API Validation
- All endpoints require valid `voice_id`
- Helpful error messages when voice_id is missing or empty
- OpenAI compatibility endpoint validates required fields

### ✅ Voice Management
- Must use `POST /voice_mappings` to create voices
- Can delete any voice (no restrictions)
- Supports audio prompt association

## Usage Examples

### Create Your First Voice

```bash
# 1. Upload audio prompt (optional)
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice_sample" \
  -F "audio_file=@voice.wav"

# 2. Create voice mapping
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "my_voice",
    "style": "friendly",
    "primary_speaker": "S1",
    "audio_prompt": "my_voice_sample"
  }'

# 3. Generate speech
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world!",
    "voice_id": "my_voice"
  }' \
  --output speech.wav
```

### Check Available Voices

```bash
# List all voices (will be empty initially)
curl "http://localhost:7860/voices"

# Get voice mappings
curl "http://localhost:7860/voice_mappings"
```

## Error Handling

### Missing Voice ID
```bash
# This will fail:
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}'

# Error: {"detail": "Field required"}
```

### Empty Voice ID (OpenAI endpoint)
```bash
# This will fail:
curl -X POST "http://localhost:7860/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello", "voice": ""}'

# Error: {"detail": "Voice identifier is required (no default voices available)"}
```

### Non-existent Voice
```bash
# This will fail:
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "voice_id": "nonexistent"}'

# Error: {"detail": "Voice 'nonexistent' not found"}
```

## Migration Notes

### For Existing Users
- **Action Required**: Create voice mappings for any previously used voices
- Old voice names (`aria`, `atlas`, `luna`, `kai`, `zara`, `nova`) must be explicitly recreated
- API calls with default voice_id will now fail

### For New Users
- Clean slate - configure only the voices you need
- More explicit and predictable behavior
- Better control over available voices

## Benefits

1. **🎯 Explicit Configuration**: No hidden/surprise voices
2. **🧹 Clean Slate**: Server starts minimal and clean
3. **🔒 Controlled Environment**: Only configured voices are available
4. **📝 Clear API Contract**: All endpoints require explicit voice specification
5. **🧪 Better Testing**: Tests use known voice configurations

## Rollback (if needed)

To restore the old behavior, simply revert the `VOICE_MAPPING` initialization:

```python
VOICE_MAPPING: Dict[str, Dict[str, Any]] = {
    "aria": {"style": "neutral", "primary_speaker": "S1", "audio_prompt": None, "audio_prompt_transcript": None},
    # ... add other voices ...
}
```

## Verification

```bash
# Verify zero voices on startup
curl "http://localhost:7860/voices" | jq '.voices | length'
# Expected: 0

# Verify empty voice mapping
curl "http://localhost:7860/voice_mappings" | jq 'keys | length'  
# Expected: 0
```

**Result: ✅ Server now starts with zero built-in voices as requested.** 