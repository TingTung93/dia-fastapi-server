#!/usr/bin/env python3
"""Create Seraphina voice with audio prompt"""

import requests
import json
import sys

BASE_URL = "http://localhost:7860"

def create_seraphina_voice():
    print("🎭 Creating Seraphina Voice\n")
    
    # 1. Check if audio prompt exists
    print("1. Checking audio prompts...")
    response = requests.get(f"{BASE_URL}/audio_prompts")
    if response.status_code != 200:
        print("❌ Failed to get audio prompts")
        return False
    
    prompts = response.json()
    if "seraphina_voice" not in prompts:
        print("❌ Audio prompt 'seraphina_voice' not found!")
        print("   Please upload seraphina_voice.wav first")
        return False
    
    print("✅ Found audio prompt: seraphina_voice")
    if 'duration' in prompts['seraphina_voice']:
        print(f"   Duration: {prompts['seraphina_voice']['duration']:.1f}s")
    
    # 2. Check if voice already exists
    print("\n2. Checking existing voices...")
    response = requests.get(f"{BASE_URL}/voice_mappings")
    if response.status_code == 200:
        mappings = response.json()
        if "seraphina" in mappings:
            print("⚠️  Voice 'seraphina' already exists")
            current = mappings['seraphina']
            print(f"   Current audio prompt: {current.get('audio_prompt')}")
            print(f"   Current speaker: {current.get('primary_speaker')}")
            
            update = input("\nUpdate existing voice? (y/n): ")
            if update.lower() != 'y':
                return False
    
    # 3. Get transcript
    print("\n3. Setting up transcript...")
    print("\n📝 The transcript should be the EXACT words spoken in seraphina_voice.wav")
    print("   This is crucial for proper voice cloning!")
    print("\n   Example transcripts:")
    print('   - "Hello, I am Seraphina. I have a warm and elegant voice."')
    print('   - "This is Seraphina speaking. My voice is clear and refined."')
    
    transcript = input("\nEnter the transcript: ").strip()
    if not transcript:
        print("❌ Transcript is required!")
        return False
    
    # 4. Create or update voice
    print("\n4. Creating voice configuration...")
    
    voice_data = {
        "voice_id": "seraphina",
        "style": "elegant",
        "primary_speaker": "S2",  # S2 for feminine voice
        "audio_prompt": "seraphina_voice",
        "audio_prompt_transcript": transcript
    }
    
    # Try update first, then create if needed
    response = requests.put(
        f"{BASE_URL}/voice_mappings/seraphina",
        json=voice_data
    )
    
    if response.status_code == 404:
        # Voice doesn't exist, create it
        response = requests.post(
            f"{BASE_URL}/voice_mappings",
            json=voice_data
        )
    
    if response.status_code == 200:
        print("✅ Voice 'seraphina' configured successfully!")
        
        # Show configuration
        print("\n📋 Voice Configuration:")
        print(f"   Voice ID: seraphina")
        print(f"   Audio Prompt: seraphina_voice")
        print(f"   Speaker Tag: S2 (feminine)")
        print(f"   Transcript: \"{transcript}\"")
        
        # 5. Test generation
        print("\n5. Testing voice...")
        test = input("\nGenerate test audio? (y/n): ")
        if test.lower() == 'y':
            test_text = "This is a test of the Seraphina voice. The voice cloning should now work properly with the audio prompt and transcript."
            
            print(f"\nGenerating test audio...")
            response = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": test_text,
                    "voice_id": "seraphina",
                    "speed": 1.0
                }
            )
            
            if response.status_code == 200:
                with open("test_seraphina.wav", "wb") as f:
                    f.write(response.content)
                print("✅ Test audio saved to: test_seraphina.wav")
                print("   Listen to verify it matches the voice from your audio prompt")
            else:
                print(f"❌ Generation failed: {response.status_code}")
        
        print("\n✅ Setup complete!")
        print("\nYou can now use voice_id='seraphina' in your generations:")
        print('   {"text": "Your text", "voice_id": "seraphina"}')
        
        return True
    else:
        print(f"❌ Failed to create voice: {response.status_code}")
        if response.text:
            print(f"   Error: {response.text}")
        return False

if __name__ == "__main__":
    # Check server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not responding at", BASE_URL)
            sys.exit(1)
    except:
        print("❌ Cannot connect to server at", BASE_URL)
        print("   Make sure the server is running")
        sys.exit(1)
    
    create_seraphina_voice()