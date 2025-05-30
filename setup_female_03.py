#!/usr/bin/env python3
"""Setup female_03 voice with transcript from reference file"""

import requests
import json
import os

BASE_URL = "http://localhost:7860"

def setup_female_03():
    print("🎭 Setting up female_03 voice\n")
    
    # 1. Check if reference file exists and read transcript
    reference_file = "female_03.reference.txt"
    transcript = None
    
    if os.path.exists(reference_file):
        print(f"📄 Reading transcript from {reference_file}")
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            print(f"✅ Transcript loaded: \"{transcript[:100]}...\"" if len(transcript) > 100 else f"✅ Transcript loaded: \"{transcript}\"")
        except Exception as e:
            print(f"❌ Error reading transcript: {e}")
    else:
        print(f"⚠️  Reference file not found: {reference_file}")
        print("   Looking in audio_prompts directory...")
        
        alt_path = os.path.join("audio_prompts", reference_file)
        if os.path.exists(alt_path):
            try:
                with open(alt_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                print(f"✅ Transcript loaded from {alt_path}")
            except Exception as e:
                print(f"❌ Error reading transcript: {e}")
    
    if not transcript:
        print("\n📝 No transcript file found. Please enter manually:")
        transcript = input("Transcript: ").strip()
        if not transcript:
            print("❌ Transcript is required!")
            return False
    
    # 2. Upload audio prompt
    print("\n📤 Uploading female_03.wav...")
    
    audio_file_path = "female_03.wav"
    if not os.path.exists(audio_file_path):
        audio_file_path = os.path.join("audio_prompts", "female_03.wav")
        if not os.path.exists(audio_file_path):
            print(f"❌ Audio file not found: female_03.wav")
            return False
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': ('female_03.wav', f, 'audio/wav')}
            data = {'prompt_id': 'female_03'}
            
            response = requests.post(
                f"{BASE_URL}/audio_prompts/upload",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Audio prompt uploaded successfully")
                print(f"   Duration: {result.get('duration', 'Unknown')}s")
                print(f"   Sample rate: {result.get('sample_rate', 'Unknown')}Hz")
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"   {response.text}")
                # Check if already exists
                if response.status_code == 400 and "already exists" in response.text:
                    print("   Audio prompt already uploaded, continuing...")
                else:
                    return False
    except Exception as e:
        print(f"❌ Error uploading audio: {e}")
        return False
    
    # 3. Create voice configuration
    print("\n🔧 Creating voice configuration...")
    
    voice_configs = [
        {
            "name": "female_03",
            "config": {
                "voice_id": "female_03",
                "style": "natural",
                "primary_speaker": "S2",  # S2 for feminine
                "audio_prompt": "female_03",
                "audio_prompt_transcript": transcript
            }
        },
        # Also create a variant with different parameters
        {
            "name": "female_03_low",
            "config": {
                "voice_id": "female_03_low",
                "style": "natural",
                "primary_speaker": "S2",
                "audio_prompt": "female_03",
                "audio_prompt_transcript": transcript
            }
        }
    ]
    
    for voice_info in voice_configs:
        name = voice_info["name"]
        config = voice_info["config"]
        
        print(f"\n📝 Creating voice: {name}")
        
        # Try to create or update
        response = requests.post(
            f"{BASE_URL}/voice_mappings",
            json=config
        )
        
        if response.status_code == 400:
            # Already exists, update it
            response = requests.put(
                f"{BASE_URL}/voice_mappings/{name}",
                json=config
            )
        
        if response.status_code == 200:
            print(f"✅ Voice '{name}' configured successfully")
        else:
            print(f"❌ Failed to configure voice '{name}': {response.status_code}")
    
    # 4. Test generation with different parameters
    print("\n🧪 Testing voice generation...")
    
    test_configs = [
        {"name": "default", "voice_id": "female_03", "params": {}},
        {"name": "low_temp", "voice_id": "female_03", "params": {"temperature": 0.8}},
        {"name": "low_cfg", "voice_id": "female_03", "params": {"cfg_scale": 1.5}},
        {"name": "both_low", "voice_id": "female_03_low", "params": {"temperature": 0.8, "cfg_scale": 1.5}},
    ]
    
    test_text = "Hello, this is a test of the female voice. It should sound natural and clear."
    
    for test in test_configs:
        print(f"\n  Testing {test['name']}...")
        
        request_data = {
            "text": test_text,
            "voice_id": test["voice_id"],
            "speed": 1.0,
            **test["params"]
        }
        
        response = requests.post(
            f"{BASE_URL}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            filename = f"test_female_03_{test['name']}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"  ✅ Saved: {filename}")
        else:
            print(f"  ❌ Generation failed: {response.status_code}")
    
    print("\n✅ Setup complete!")
    print("\n📊 Generated test files:")
    print("  - test_female_03_default.wav (standard settings)")
    print("  - test_female_03_low_temp.wav (temperature 0.8)")
    print("  - test_female_03_low_cfg.wav (cfg_scale 1.5)")
    print("  - test_female_03_both_low.wav (both lowered)")
    
    print("\n💡 Usage:")
    print('  curl -X POST "http://localhost:7860/generate" \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"text": "Your text", "voice_id": "female_03"}\' \\')
    print('       --output output.wav')
    
    print("\n🎯 If the voice still sounds masculine, try:")
    print("  - voice_id: 'female_03' with temperature: 0.8")
    print("  - voice_id: 'female_03' with cfg_scale: 1.0")
    print("  - Check the audio file quality (clear female voice, no effects)")

if __name__ == "__main__":
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not responding")
            exit(1)
    except:
        print("❌ Cannot connect to server at", BASE_URL)
        print("   Make sure server is running")
        exit(1)
    
    setup_female_03()