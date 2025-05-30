#!/usr/bin/env python3
"""Fix voice gender by adjusting speaker tags"""

import requests
import json
import sys

BASE_URL = "http://localhost:7860"

def check_voice_config(voice_id):
    """Check current voice configuration"""
    print(f"📋 Current configuration for '{voice_id}':\n")
    
    response = requests.get(f"{BASE_URL}/voice_mappings")
    if response.status_code == 200:
        mappings = response.json()
        if voice_id in mappings:
            config = mappings[voice_id]
            print(f"  Voice ID: {voice_id}")
            print(f"  Primary Speaker: {config.get('primary_speaker', 'N/A')} {'← This affects gender!' if config.get('primary_speaker') else ''}")
            print(f"  Audio Prompt: {config.get('audio_prompt', 'None')}")
            print(f"  Transcript: {config.get('audio_prompt_transcript', 'None')[:50]}..." if config.get('audio_prompt_transcript') else "  Transcript: None")
            
            speaker = config.get('primary_speaker', 'S1')
            print(f"\n  Current Gender Tendency:")
            if speaker == 'S1':
                print("  🔷 S1 = More masculine/neutral voice")
                print("  💡 Consider switching to S2 for feminine voice")
            else:
                print("  🔷 S2 = More feminine voice")
            
            return config
        else:
            print(f"❌ Voice '{voice_id}' not found")
            return None
    return None

def update_speaker_tag(voice_id, speaker_tag):
    """Update the primary speaker tag"""
    print(f"\n🔧 Updating speaker tag to {speaker_tag}...")
    
    response = requests.put(
        f"{BASE_URL}/voice_mappings/{voice_id}",
        json={
            "voice_id": voice_id,
            "primary_speaker": speaker_tag
        }
    )
    
    if response.status_code == 200:
        print(f"  ✅ Speaker tag updated successfully!")
        return True
    else:
        print(f"  ❌ Failed: {response.status_code}")
        return False

def test_generation(voice_id, test_text, filename):
    """Test generation with current settings"""
    print(f"\n🎤 Generating test audio...")
    
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": test_text,
            "voice_id": voice_id,
            "speed": 1.0
        }
    )
    
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"  ✅ Saved to: {filename}")
        return True
    return False

def main():
    print("🔍 Voice Gender Fix Tool\n")
    print("This tool helps fix masculine/feminine voice issues with audio prompts.\n")
    
    # Get voice ID
    voice_id = input("Enter voice ID (e.g., 'seraphina'): ").strip()
    if not voice_id:
        voice_id = "seraphina"
    
    # Check current config
    config = check_voice_config(voice_id)
    if not config:
        print("\n❌ Voice not found. Create it first.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("\n🎯 Understanding Speaker Tags in Dia Model:")
    print("\nThe Dia model uses two speaker tags that affect voice characteristics:")
    print("  • [S1] = Tends toward masculine/neutral voices")
    print("  • [S2] = Tends toward feminine voices")
    print("\nEven with a female audio prompt, using S1 can make it sound masculine!")
    
    current_speaker = config.get('primary_speaker', 'S1')
    
    if current_speaker == 'S1':
        print(f"\n⚠️  Your voice is currently using {current_speaker} (masculine tendency)")
        print("   This is likely why it sounds masculine despite the female audio prompt.")
        
        change = input("\nSwitch to S2 for feminine voice? (y/n): ").lower()
        if change == 'y':
            if update_speaker_tag(voice_id, 'S2'):
                print("\n✅ Updated to S2 (feminine voice)")
                current_speaker = 'S2'
    
    # Test both options
    print("\n" + "="*60)
    print("\n🧪 Testing Voice Generation:")
    
    test_text = "Hello, this is a test of the voice cloning system. I hope my voice sounds natural and matches the audio prompt."
    
    print("\n1. Generating with current settings...")
    test_generation(voice_id, test_text, f"test_{voice_id}_current.wav")
    
    # Offer to test the opposite
    opposite = 'S1' if current_speaker == 'S2' else 'S2'
    test_opposite = input(f"\nGenerate comparison with {opposite}? (y/n): ").lower()
    
    if test_opposite == 'y':
        print(f"\n2. Temporarily testing with {opposite}...")
        # Save current setting
        original = current_speaker
        
        # Update to opposite
        if update_speaker_tag(voice_id, opposite):
            test_generation(voice_id, test_text, f"test_{voice_id}_{opposite}.wav")
            
            # Restore original
            print(f"\n🔄 Restoring original setting ({original})...")
            update_speaker_tag(voice_id, original)
    
    print("\n" + "="*60)
    print("\n💡 Additional Tips:")
    print("\n1. Speaker Tag Override:")
    print("   You can override per request by including tags in your text:")
    print('   "text": "[S2] Your text here [S2]"')
    
    print("\n2. Fine-tuning:")
    print("   Try adjusting these parameters for better results:")
    print("   • temperature: 0.8-1.2 (lower = more consistent)")
    print("   • cfg_scale: 2.0-4.0 (higher = stronger prompt adherence)")
    
    print("\n3. Audio Prompt Quality:")
    print("   Ensure your audio prompt is:")
    print("   • Clear female voice without ambiguity")
    print("   • No background music or effects")
    print("   • Natural speech, not whispered or altered")
    
    print("\n4. Alternative Approach:")
    print("   If S2 alone doesn't work, try:")
    print("   • Different base voice (fable, onyx use S2 by default)")
    print("   • Adjust audio prompt transcript to emphasize feminine qualities")
    print("   • Use longer audio prompt (10-20 seconds)")

if __name__ == "__main__":
    main()