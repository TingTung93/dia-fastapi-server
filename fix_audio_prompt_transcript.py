#!/usr/bin/env python3
"""Fix audio prompt transcript for better voice cloning"""

import requests
import json
import sys

BASE_URL = "http://localhost:7860"

def show_current_setup(voice_id):
    """Show current voice configuration"""
    print(f"📋 Current setup for voice '{voice_id}':\n")
    
    response = requests.get(f"{BASE_URL}/voice_mappings")
    if response.status_code == 200:
        mappings = response.json()
        if voice_id in mappings:
            config = mappings[voice_id]
            print(f"  Voice ID: {voice_id}")
            print(f"  Style: {config.get('style', 'N/A')}")
            print(f"  Primary Speaker: {config.get('primary_speaker', 'N/A')}")
            print(f"  Audio Prompt: {config.get('audio_prompt', 'None')}")
            print(f"  Audio Prompt Transcript: {config.get('audio_prompt_transcript', 'None')}")
            
            if config.get('audio_prompt') and not config.get('audio_prompt_transcript'):
                print("\n  ⚠️  WARNING: Audio prompt is set but transcript is missing!")
                print("     This is likely causing the gibberish output.")
            return config
        else:
            print(f"  ❌ Voice '{voice_id}' not found")
            return None
    else:
        print(f"  ❌ Failed to get voice mappings")
        return None

def update_transcript(voice_id, transcript):
    """Update the audio prompt transcript for a voice"""
    print(f"\n🔧 Updating transcript for voice '{voice_id}'...")
    
    response = requests.put(
        f"{BASE_URL}/voice_mappings/{voice_id}",
        json={
            "voice_id": voice_id,
            "audio_prompt_transcript": transcript
        }
    )
    
    if response.status_code == 200:
        print(f"  ✅ Transcript updated successfully!")
        return True
    else:
        print(f"  ❌ Failed: {response.status_code} - {response.text}")
        return False

def test_with_transcript(voice_id, test_text):
    """Test generation with the updated transcript"""
    print(f"\n🎤 Testing generation...")
    
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": test_text,
            "voice_id": voice_id,
            "speed": 1.0
        }
    )
    
    if response.status_code == 200:
        audio_data = response.content
        output_file = f"test_{voice_id}_fixed.wav"
        with open(output_file, "wb") as f:
            f.write(audio_data)
        print(f"  ✅ Generated audio saved to: {output_file}")
        print(f"  📊 Size: {len(audio_data):,} bytes")
        return True
    else:
        print(f"  ❌ Failed: {response.status_code}")
        return False

def main():
    print("🔍 Audio Prompt Transcript Fixer\n")
    print("This tool helps fix gibberish output from audio prompts by setting the transcript.\n")
    
    # Get voice ID
    voice_id = input("Enter the voice ID (e.g., 'seraphina'): ").strip()
    if not voice_id:
        print("❌ Voice ID is required")
        sys.exit(1)
    
    # Show current setup
    config = show_current_setup(voice_id)
    if not config:
        sys.exit(1)
    
    if not config.get('audio_prompt'):
        print("\n❌ This voice doesn't have an audio prompt set.")
        print("   First upload an audio prompt and configure the voice to use it.")
        sys.exit(1)
    
    # Get transcript
    print("\n" + "="*60)
    print("\n📝 IMPORTANT: Audio Prompt Transcript")
    print("\nThe transcript should be the EXACT words spoken in your audio prompt file.")
    print("This tells the model what's being said in the reference audio.\n")
    print("For example, if your audio prompt says:")
    print('  "Hello, my name is Seraphina. I have a warm and elegant voice."')
    print("\nThen that exact text should be your transcript.\n")
    
    transcript = input("Enter the transcript of your audio prompt: ").strip()
    if not transcript:
        print("❌ Transcript is required")
        sys.exit(1)
    
    # Update transcript
    if update_transcript(voice_id, transcript):
        # Test generation
        print("\n" + "="*60)
        test_text = input("\nEnter text to test (or press Enter for default): ").strip()
        if not test_text:
            test_text = "This is a test of the voice cloning. It should now sound much clearer and more natural."
        
        test_with_transcript(voice_id, test_text)
        
        print("\n" + "="*60)
        print("\n✅ Transcript has been set!")
        print("\n🎯 How it works:")
        print("1. The model prepends your transcript to the generation text")
        print("2. This helps it understand the voice characteristics")
        print("3. The output should now match the voice in your audio prompt")
        
        print("\n💡 Tips for best results:")
        print("- Make sure the transcript is 100% accurate")
        print("- Use punctuation to match the speech rhythm")
        print("- Include any specific pronunciations or emphasis")
        print("- The audio prompt should be clear speech without background noise")

if __name__ == "__main__":
    main()