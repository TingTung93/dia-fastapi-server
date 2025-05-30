#!/usr/bin/env python3
"""Check voice and audio prompt setup"""

import requests
import json

BASE_URL = "http://localhost:7860"

def check_setup():
    print("🔍 Checking Voice and Audio Prompt Setup\n")
    
    # 1. Check audio prompts
    print("1. Audio Prompts:")
    try:
        response = requests.get(f"{BASE_URL}/audio_prompts")
        if response.status_code == 200:
            prompts = response.json()
            if prompts:
                print("   ✅ Audio prompts found:")
                for prompt_id, info in prompts.items():
                    print(f"      - {prompt_id}: {info.get('file_path', 'Unknown path')}")
                    if 'duration' in info:
                        print(f"        Duration: {info['duration']:.1f}s")
            else:
                print("   ❌ No audio prompts uploaded")
        else:
            print(f"   ❌ Failed to get audio prompts: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Check voice mappings
    print("\n2. Voice Mappings:")
    try:
        response = requests.get(f"{BASE_URL}/voice_mappings")
        if response.status_code == 200:
            mappings = response.json()
            
            # Look for voices using audio prompts
            voices_with_prompts = []
            for voice_id, config in mappings.items():
                if config.get('audio_prompt'):
                    voices_with_prompts.append((voice_id, config))
            
            if voices_with_prompts:
                print("   ✅ Voices with audio prompts:")
                for voice_id, config in voices_with_prompts:
                    print(f"      - Voice ID: '{voice_id}'")
                    print(f"        Audio prompt: {config.get('audio_prompt')}")
                    print(f"        Primary speaker: {config.get('primary_speaker')}")
                    if config.get('audio_prompt_transcript'):
                        print(f"        Transcript: \"{config['audio_prompt_transcript'][:50]}...\"")
                    else:
                        print(f"        ⚠️  No transcript set!")
            else:
                print("   ❌ No voices configured with audio prompts")
                print("\n   Available voice IDs:")
                for voice_id in mappings.keys():
                    print(f"      - {voice_id}")
        else:
            print(f"   ❌ Failed to get voice mappings: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("\n💡 Understanding the Setup:")
    print("\n1. Audio Prompt = The WAV file (e.g., 'seraphina_voice')")
    print("2. Voice ID = The voice name you use in generation (e.g., 'seraphina')")
    print("3. You need to create a voice that uses the audio prompt")
    
    print("\n🔧 Quick Setup Commands:")
    print("\nIf you have 'seraphina_voice.wav' uploaded but no voice created:")
    print("""
# Create a voice called 'seraphina' that uses the audio prompt:
curl -X POST "http://localhost:7860/voice_mappings" \\
     -H "Content-Type: application/json" \\
     -d '{
       "voice_id": "seraphina",
       "style": "elegant",
       "primary_speaker": "S2",
       "audio_prompt": "seraphina_voice",
       "audio_prompt_transcript": "Hello, I am Seraphina. I speak with a warm, elegant voice."
     }'
""")
    
    print("\nThen generate with:")
    print("""
curl -X POST "http://localhost:7860/generate" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "This is a test of the voice cloning.",
       "voice_id": "seraphina"
     }' --output test.wav
""")
    
    print("\n📋 Common Mistakes:")
    print("   ❌ Using 'seraphina_voice' as voice_id (that's the audio file)")
    print("   ✅ Using 'seraphina' as voice_id (that's the voice name)")
    print("   ❌ Not setting the transcript")
    print("   ✅ Setting transcript to match audio content")

if __name__ == "__main__":
    check_setup()