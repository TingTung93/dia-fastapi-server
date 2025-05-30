#!/usr/bin/env python3
"""Fix Seraphina audio prompt issues"""

import requests
import json
import os

BASE_URL = "http://localhost:7860"

def diagnose_issue():
    print("🔍 Diagnosing Seraphina Audio Prompt Issue\n")
    
    print("Your test results show:")
    print("✅ Female voice: low_temp (0.8) and no_prompt")
    print("❌ Masculine voice: with audio prompt")
    print("\nThis suggests the audio prompt is working but having the wrong effect.\n")
    
    # Get current configuration
    try:
        response = requests.get(f"{BASE_URL}/voice_mappings")
        if response.status_code == 200:
            voices = response.json()
            if 'seraphina' in voices:
                config = voices['seraphina']
                print("Current Seraphina configuration:")
                print(f"  Primary speaker: {config.get('primary_speaker')}")
                print(f"  Audio prompt: {config.get('audio_prompt')}")
                print(f"  Transcript: {config.get('audio_prompt_transcript', 'Not set')}")
    except:
        pass

def fix_approaches():
    print("\n🔧 Fix Approaches:\n")
    
    print("1. **Check Audio Prompt Quality**")
    print("   Your seraphina_voice.wav might:")
    print("   - Be too short/long (ideal: 5-15 seconds)")
    print("   - Have background noise")
    print("   - Be ambiguous in gender")
    print("   - Have processing/effects that confuse the model")
    
    print("\n2. **Try Inverse CFG Scale**")
    print("   Since audio prompt makes it masculine, try negative guidance:")
    
    inverse_config = {
        "text": "This is a test of the Seraphina voice.",
        "voice_id": "seraphina", 
        "cfg_scale": 1.0,  # Lower CFG = less audio prompt influence
        "temperature": 0.8  # Lower temp worked better
    }
    
    print(f"""
curl -X POST "{BASE_URL}/generate" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(inverse_config, indent=2)}' \\
     --output test_low_cfg.wav
""")
    
    print("\n3. **Create Alternative Voice Without Audio Prompt**")
    print("   Since low_temp without prompt worked well:")
    
    alt_voice = {
        "voice_id": "seraphina_base",
        "style": "elegant",
        "primary_speaker": "S2",
        "audio_prompt": None,
        "audio_prompt_transcript": None
    }
    
    print(f"""
# Create base voice without audio prompt
curl -X POST "{BASE_URL}/voice_mappings" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(alt_voice, indent=2)}'

# Generate with low temperature
curl -X POST "{BASE_URL}/generate" \\
     -H "Content-Type: application/json" \\
     -d '{{"text": "Test text", "voice_id": "seraphina_base", "temperature": 0.8}}' \\
     --output test_base.wav
""")
    
    print("\n4. **Experiment with Transcript**")
    print("   The transcript might be affecting generation:")
    print("   - Try without transcript")
    print("   - Try with explicit gender markers")
    print("   - Try with different transcript content")
    
    print("\n5. **Test Different Parameters**")
    params_to_test = [
        {"name": "minimal_cfg", "cfg_scale": 0.5, "temperature": 0.8},
        {"name": "no_cfg", "cfg_scale": 0.0, "temperature": 0.8},
        {"name": "high_temp_low_cfg", "cfg_scale": 1.5, "temperature": 1.5},
        {"name": "explicit_s2", "text": "[S2] This is Seraphina speaking [S2]", "cfg_scale": 2.0}
    ]
    
    print("\n   Test configurations:")
    for params in params_to_test:
        name = params.pop("name")
        print(f"\n   # Test: {name}")
        cmd_params = {"voice_id": "seraphina", **params}
        print(f'   curl -X POST "{BASE_URL}/generate" \\')
        print(f'        -H "Content-Type: application/json" \\')
        print(f"        -d '{json.dumps(cmd_params)}' \\")
        print(f'        --output test_{name}.wav')

def audio_prompt_analysis():
    print("\n\n📊 Audio Prompt Analysis Needed:\n")
    
    print("1. **Check the WAV file properties:**")
    print("""
import soundfile as sf
import numpy as np

data, sr = sf.read('audio_prompts/seraphina_voice.wav')
print(f"Duration: {len(data)/sr:.1f}s")
print(f"Sample rate: {sr}Hz")
print(f"RMS level: {np.sqrt(np.mean(data**2)):.3f}")
print(f"Peak level: {np.max(np.abs(data)):.3f}")

# Check frequency content (pitch)
from scipy import signal
f, Pxx = signal.periodogram(data, sr)
peak_freq = f[np.argmax(Pxx)]
print(f"Dominant frequency: {peak_freq:.1f}Hz")
""")
    
    print("\n2. **Common Issues:**")
    print("   - Male voice in 'female' recording")
    print("   - Pitch-shifted or processed audio")
    print("   - Multiple speakers in one file")
    print("   - Very short clip (under 3 seconds)")
    print("   - Background music/effects")
    
    print("\n3. **Quick Fix - Use Working Config:**")
    print("   Since 'no prompt + low temp' worked, just use that!")
    print("""
# Best working configuration based on your tests:
curl -X POST "{BASE_URL}/generate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "Your text here",
       "voice_id": "seraphina",
       "temperature": 0.8,
       "cfg_scale": 0.0
     }' --output output.wav
""")

def main():
    diagnose_issue()
    fix_approaches()
    audio_prompt_analysis()
    
    print("\n\n🎯 Recommended Action:\n")
    print("1. First, try cfg_scale=0.0 or 0.5 to reduce audio prompt influence")
    print("2. Create a base voice without audio prompt for comparison")
    print("3. Check your audio file - it might not be what the model expects")
    print("4. Use the working config (no prompt, temp=0.8) for now")
    
    print("\n💡 The fact that the audio prompt makes it MORE masculine suggests:")
    print("   - The audio file might contain male characteristics")
    print("   - Or the model is interpreting it incorrectly")
    print("   - Or there's a mismatch between transcript and audio")

if __name__ == "__main__":
    main()