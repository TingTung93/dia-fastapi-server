#!/usr/bin/env python3
"""Quick check for Seraphina voice setup"""

import os
import json

def check_setup():
    print("🔍 Checking Seraphina Voice Setup\n")
    
    # Check if audio file exists
    audio_prompt_dir = "audio_prompts"
    seraphina_file = os.path.join(audio_prompt_dir, "seraphina_voice.wav")
    
    print("1. Audio Prompt File:")
    if os.path.exists(seraphina_file):
        file_size = os.path.getsize(seraphina_file)
        print(f"   ✅ Found: {seraphina_file}")
        print(f"   📊 Size: {file_size:,} bytes")
        
        # Check if file is valid
        try:
            import soundfile as sf
            data, samplerate = sf.read(seraphina_file)
            duration = len(data) / samplerate
            print(f"   🎵 Duration: {duration:.2f} seconds")
            print(f"   🎵 Sample rate: {samplerate} Hz")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    else:
        print(f"   ❌ Not found: {seraphina_file}")
        print(f"   💡 Make sure the file is uploaded to the audio_prompts/ directory")
    
    print("\n2. Voice Configuration:")
    print("   To use this audio prompt, you need to:")
    print("   a) Upload the audio prompt (if not already done)")
    print("   b) Create a voice mapping that references it")
    
    print("\n3. Example Setup Commands:")
    print("\n   # Upload audio prompt (if needed):")
    print('   curl -X POST "http://localhost:7860/audio_prompts/upload" \\')
    print('        -F "prompt_id=seraphina_voice" \\')
    print('        -F "audio_file=@seraphina_voice.wav"')
    
    print("\n   # Create voice with audio prompt:")
    print('   curl -X POST "http://localhost:7860/voice_mappings" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"voice_id": "seraphina", "style": "elegant", "primary_speaker": "S1", "audio_prompt": "seraphina_voice", "audio_prompt_transcript": "Hello, this is Seraphina speaking."}\'')
    
    print("\n   # Or update existing voice:")
    print('   curl -X PUT "http://localhost:7860/voice_mappings/seraphina" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"voice_id": "seraphina", "audio_prompt": "seraphina_voice", "audio_prompt_transcript": "Hello, this is Seraphina speaking."}\'')
    
    print("\n   # Test generation:")
    print('   curl -X POST "http://localhost:7860/generate" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"text": "Hello, this is a test.", "voice_id": "seraphina"}\' \\')
    print('        --output test_seraphina.wav')
    
    print("\n4. Common Issues:")
    print("   - Empty audio: Usually means the audio prompt isn't being found or loaded")
    print("   - Check server logs for 'Warning: Audio prompt file not found' messages")
    print("   - Make sure the audio_prompt_transcript is set (it helps the model)")
    print("   - The audio prompt should be 3-30 seconds of clear speech")

if __name__ == "__main__":
    check_setup()