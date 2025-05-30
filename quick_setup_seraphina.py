#!/usr/bin/env python3
"""Quick setup for Seraphina voice"""

import requests
import json

BASE_URL = "http://localhost:7860"

def main():
    print("🎭 Quick Seraphina Voice Setup\n")
    
    # 1. Check what's currently available
    print("1. Checking current setup...")
    
    # Check audio prompts
    try:
        response = requests.get(f"{BASE_URL}/audio_prompts")
        if response.status_code == 200:
            prompts = response.json()
            print("\n📁 Audio Prompts:")
            if prompts:
                for name, info in prompts.items():
                    print(f"   ✅ {name}")
                    if isinstance(info, dict) and 'duration' in info:
                        print(f"      Duration: {info['duration']:.1f}s")
            else:
                print("   ❌ No audio prompts found")
                print("   Please upload seraphina_voice.wav first!")
                return
    except Exception as e:
        print(f"❌ Error checking audio prompts: {e}")
        return
    
    # Check voices
    try:
        response = requests.get(f"{BASE_URL}/voice_mappings")
        if response.status_code == 200:
            voices = response.json()
            print("\n🎤 Current Voices:")
            for voice_id in voices.keys():
                audio_prompt = voices[voice_id].get('audio_prompt')
                if audio_prompt:
                    print(f"   ✅ {voice_id} (uses audio prompt: {audio_prompt})")
                else:
                    print(f"   - {voice_id} (no audio prompt)")
    except Exception as e:
        print(f"❌ Error checking voices: {e}")
        return
    
    # 2. Check if seraphina_voice exists
    if 'seraphina_voice' not in prompts:
        print("\n❌ Audio prompt 'seraphina_voice' not found!")
        print("\n📝 To upload it:")
        print('curl -X POST "http://localhost:7860/audio_prompts/upload" \\')
        print('     -F "prompt_id=seraphina_voice" \\')
        print('     -F "audio_file=@seraphina_voice.wav"')
        return
    
    # 3. Create seraphina voice
    print("\n2. Creating 'seraphina' voice...")
    
    print("\n📝 Please provide the transcript")
    print("   (The exact words spoken in seraphina_voice.wav)")
    print("   Example: 'Hello, I am Seraphina. I have a warm, elegant voice.'")
    
    transcript = input("\nTranscript: ").strip()
    if not transcript:
        # Use a default if empty
        transcript = "This is Seraphina speaking with my natural voice."
        print(f"Using default: {transcript}")
    
    # Create the voice
    voice_config = {
        "voice_id": "seraphina",
        "style": "elegant",
        "primary_speaker": "S2",  # S2 for feminine
        "audio_prompt": "seraphina_voice",
        "audio_prompt_transcript": transcript
    }
    
    print("\n3. Creating voice configuration...")
    response = requests.post(
        f"{BASE_URL}/voice_mappings",
        json=voice_config
    )
    
    if response.status_code == 200:
        print("✅ Voice 'seraphina' created successfully!")
        
        print("\n📋 Configuration:")
        print(f"   Voice ID: seraphina")
        print(f"   Audio Prompt: seraphina_voice")
        print(f"   Speaker: S2 (feminine)")
        print(f"   Transcript: '{transcript}'")
        
        print("\n4. Testing voice...")
        print("Generating test audio...")
        
        test_response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": "This is a test of the Seraphina voice. It should sound like the audio prompt.",
                "voice_id": "seraphina",
                "speed": 1.0
            }
        )
        
        if test_response.status_code == 200:
            with open("seraphina_test.wav", "wb") as f:
                f.write(test_response.content)
            print("✅ Test audio saved: seraphina_test.wav")
        else:
            print(f"❌ Test generation failed: {test_response.status_code}")
        
        print("\n✅ Setup complete!")
        print("\nNow you can:")
        print("1. Run debug: python debug_audio_prompt_deep.py")
        print("   Enter voice ID: seraphina")
        print("\n2. Or generate directly:")
        print('   curl -X POST "http://localhost:7860/generate" \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"text": "Your text", "voice_id": "seraphina"}\' \\')
        print('        --output output.wav')
        
    else:
        print(f"❌ Failed to create voice: {response.status_code}")
        if response.status_code == 400:
            print("   Voice might already exist. Try:")
            print(f'   curl -X PUT "http://localhost:7860/voice_mappings/seraphina" \\')
            print(f'        -H "Content-Type: application/json" \\')
            print(f'        -d \'{json.dumps(voice_config)}\'')

if __name__ == "__main__":
    try:
        # Check server
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not responding")
            exit(1)
    except:
        print("❌ Cannot connect to server at", BASE_URL)
        print("   Make sure server is running")
        exit(1)
    
    main()