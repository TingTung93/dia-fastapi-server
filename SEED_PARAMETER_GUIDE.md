# Seed Parameter Guide for Reproducible Generation

## Overview ✅

The Dia TTS server now supports **seed parameters** for reproducible audio generation. This feature addresses the community-requested ability to generate consistent outputs by controlling the randomness in the model.

## What is a Seed?

A **seed** is an integer value that controls the random number generators used during audio generation. When you use the same seed:
- ✅ The model will produce more consistent results
- ✅ You can reproduce similar voice characteristics 
- ✅ Multiple generations with the same seed and text should be similar

**⚠️ Important Notes:**
- Seeds help with consistency but don't guarantee identical outputs due to model architecture
- For best voice consistency, combine seeds with **voice mappings** (audio prompts)
- The Dia model has some inherent non-deterministic behavior that seeds may not fully control

## How to Use Seeds

### 1. API Endpoints

#### Main Generation Endpoint (`/generate`)
```json
{
  "text": "[S1] Hello world [S2] How are you today?",
  "voice_id": "seraphina_voice",
  "seed": 42
}
```

#### OpenAI Compatible Endpoint (`/v1/audio/speech`)
```json
{
  "input": "Hello, this is a test",
  "voice": "seraphina_voice", 
  "seed": 1234
}
```

### 2. Seed Values

- **Integer (e.g., 42, 1234, 9999)**: Fixed seed for reproducible generation
- **null or omitted**: Random generation (default behavior)

### 3. Best Practices for Consistency

**Most Consistent (Recommended):**
```json
{
  "text": "Your text here",
  "voice_id": "voice_with_audio_prompt",  // Use a voice with audio prompt
  "seed": 42,                            // Fixed seed
  "temperature": 1.0,                    // Fixed parameters
  "cfg_scale": 3.0,
  "top_p": 0.95
}
```

**Less Consistent:**
```json
{
  "text": "Your text here", 
  "voice_id": "unmapped_voice",  // No audio prompt
  "seed": 42                     // Seed alone may not be enough
}
```

## Technical Implementation

### Seed Application
When a seed is provided, the system:

1. **Sets all random generators**:
   - Python `random.seed(seed)`
   - NumPy `np.random.seed(seed)`
   - PyTorch `torch.manual_seed(seed)`
   - CUDA `torch.cuda.manual_seed_all(seed)`

2. **Enables deterministic mode**:
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`

3. **Applies before generation**: Seed is set at the start of each generation call

### Debug Output
When debug mode is enabled (`debug_mode: true`), you'll see:
```
🌱 Generation seed set to: 42
```

## Example Usage

### Testing Voice Consistency
```bash
# First generation
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[S1] This is a test with seed",
    "voice_id": "seraphina_voice",
    "seed": 12345
  }'

# Second generation (should be similar)
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[S1] This is a test with seed", 
    "voice_id": "seraphina_voice",
    "seed": 12345
  }'
```

### Random vs Fixed Comparison
```python
import requests

# Random generation (different each time)
response1 = requests.post("http://localhost:7860/generate", json={
    "text": "Hello world",
    "voice_id": "seraphina_voice"
    # No seed = random
})

# Fixed generation (reproducible)
response2 = requests.post("http://localhost:7860/generate", json={
    "text": "Hello world", 
    "voice_id": "seraphina_voice",
    "seed": 777
})
```

## Limitations and Known Issues

### Current Limitations
1. **Partial Determinism**: Dia model architecture has some non-deterministic elements that seeds cannot fully control
2. **Voice Selection**: Seeds don't control the fundamental voice characteristics - use audio prompts for voice consistency
3. **Hardware Differences**: Results may vary slightly between different GPU models/drivers

### Community Feedback
Based on [GitHub Issue #109](https://github.com/nari-labs/dia/issues/109), users report:
- Seeds help with generation consistency
- Voice characteristics still vary between runs without audio prompts
- Best results when combining seeds with voice cloning/audio prompts

## Troubleshooting

### Seed Not Working?
1. **Check voice mapping**: Ensure your `voice_id` has an audio prompt configured
2. **Verify parameters**: Use the same temperature, cfg_scale, etc. for consistency
3. **Debug mode**: Enable debug to see seed confirmation message
4. **Hardware**: Different GPUs may produce slight variations

### Still Getting Different Voices?
```json
{
  "text": "Your text",
  "voice_id": "voice_with_audio_prompt",  // ← Most important!
  "seed": 42,                            // ← Helps with consistency
  "temperature": 1.2,                    // ← Keep parameters fixed
  "cfg_scale": 3.0
}
```

## Examples in the Wild

Check out the test script to see seed usage:
```bash
python -c "
import requests
response = requests.post('http://localhost:7860/generate', json={
    'text': '[S1] Testing seed consistency',
    'voice_id': 'seraphina_voice', 
    'seed': 2024
})
print('Generated with seed 2024')
"
```

---

## Contributing

Found issues with seed consistency? Please report them with:
- Seed value used
- Voice configuration
- Generation parameters
- Hardware details (GPU model)

This helps improve the reproducibility features for everyone! [RAT: AC, SD, SF] 