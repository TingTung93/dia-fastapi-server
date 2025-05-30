#!/usr/bin/env python3
"""Debug script for audio prompt issues"""

import requests
import json
import os
import sys

BASE_URL = "http://localhost:7860"

def check_audio_prompts():
    """Check current audio prompts"""
    print("📋 Checking Audio Prompts...")
    try:
        response = requests.get(f"{BASE_URL}/audio_prompts")
        if response.status_code == 200:
            prompts = response.json()
            print(f"\nFound {len(prompts)} audio prompts:")
            for prompt_id, info in prompts.items():
                print(f"  - {prompt_id}: {info}")
        else:
            print(f"❌ Failed to get audio prompts: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_voice_mappings():
    """Check current voice mappings"""
    print("\n📋 Checking Voice Mappings...")
    try:
        response = requests.get(f"{BASE_URL}/voice_mappings")
        if response.status_code == 200:
            mappings = response.json()
            print(f"\nFound {len(mappings)} voice mappings:")
            for voice_id, config in mappings.items():
                if config.get("audio_prompt"):
                    print(f"  ✅ {voice_id}: uses audio prompt '{config['audio_prompt']}'")
                else:
                    print(f"  - {voice_id}: no audio prompt")
        else:
            print(f"❌ Failed to get voice mappings: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_voice_with_prompt(voice_id, audio_prompt_id):
    """Create or update a voice to use an audio prompt"""
    print(f"\n🔧 Creating/updating voice '{voice_id}' with audio prompt '{audio_prompt_id}'...")
    
    # Check if voice exists
    response = requests.get(f"{BASE_URL}/voice_mappings")
    mappings = response.json()
    
    if voice_id in mappings:
        # Update existing voice
        print(f"  Updating existing voice '{voice_id}'...")
        response = requests.put(
            f"{BASE_URL}/voice_mappings/{voice_id}",
            json={
                "voice_id": voice_id,
                "audio_prompt": audio_prompt_id,
                "audio_prompt_transcript": "This is Seraphina speaking."  # You can customize this
            }
        )
    else:
        # Create new voice
        print(f"  Creating new voice '{voice_id}'...")
        response = requests.post(
            f"{BASE_URL}/voice_mappings",
            json={
                "voice_id": voice_id,
                "style": "custom",
                "primary_speaker": "S1",
                "audio_prompt": audio_prompt_id,
                "audio_prompt_transcript": "This is Seraphina speaking."  # You can customize this
            }
        )
    
    if response.status_code == 200:
        print(f"  ✅ Voice '{voice_id}' configured successfully!")
        return True
    else:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return False

def test_generation(voice_id, text):
    """Test audio generation with the voice"""
    print(f"\n🎤 Testing generation with voice '{voice_id}'...")
    print(f"  Text: '{text}'")
    
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": text,
            "voice_id": voice_id,
            "speed": 1.0
        }
    )
    
    if response.status_code == 200:
        audio_data = response.content
        print(f"  ✅ Generated audio: {len(audio_data)} bytes")
        
        # Save to file for inspection
        output_file = f"test_output_{voice_id}.wav"
        with open(output_file, "wb") as f:
            f.write(audio_data)
        print(f"  💾 Saved to: {output_file}")
        
        # Check if audio is empty (just headers)
        if len(audio_data) < 1000:  # WAV header is usually around 44 bytes
            print(f"  ⚠️  WARNING: Audio file seems very small ({len(audio_data)} bytes)")
        
        return True
    else:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return False

def main():
    print("🔍 Audio Prompt Debugging Tool\n")
    
    # Check server health
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not responding properly!")
            sys.exit(1)
        print("✅ Server is running\n")
    except:
        print("❌ Cannot connect to server at", BASE_URL)
        print("   Please make sure the server is running.")
        sys.exit(1)
    
    # Step 1: Check audio prompts
    check_audio_prompts()
    
    # Step 2: Check voice mappings
    check_voice_mappings()
    
    # Step 3: Setup voice with audio prompt
    print("\n" + "="*50)
    print("\n📝 To use 'seraphina_voice.wav' as an audio prompt:")
    print("\n1. First, make sure it's uploaded:")
    print("   - The file should be in the audio_prompts/ directory")
    print("   - Or upload it via the API")
    
    print("\n2. Create a voice that uses this audio prompt:")
    user_input = input("\nDo you want to create/update a voice 'seraphina' with this audio prompt? (y/n): ")
    
    if user_input.lower() == 'y':
        # Assuming the audio prompt ID is 'seraphina_voice' (without .wav)
        if create_voice_with_prompt("seraphina", "seraphina_voice"):
            # Test the voice
            test_text = input("\nEnter text to test (or press Enter for default): ").strip()
            if not test_text:
                test_text = "Hello, this is a test of the Seraphina voice. How does it sound?"
            
            test_generation("seraphina", test_text)
    
    print("\n" + "="*50)
    print("\n💡 Troubleshooting Tips:")
    print("1. Enable debug mode to see what's happening:")
    print("   curl -X PUT http://localhost:7860/config -H 'Content-Type: application/json' -d '{\"debug_mode\": true}'")
    print("\n2. Check server logs for warnings about audio prompt files")
    print("\n3. Make sure the audio prompt transcript is set (it helps with voice cloning)")
    print("\n4. Try different generation parameters (temperature, cfg_scale)")

if __name__ == "__main__":
    main()