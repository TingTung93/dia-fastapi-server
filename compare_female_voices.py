#!/usr/bin/env python3
"""Compare female voices with different settings"""

import requests
import json
import os

BASE_URL = "http://localhost:7860"

def compare_voices():
    print("🎭 Female Voice Comparison Test\n")
    
    # Test text
    test_text = "Hello, this is a test of the text to speech system. I should sound like a natural female voice with clear pronunciation."
    
    # Voice configurations to test
    test_configs = [
        # Seraphina tests
        {"name": "seraphina_no_prompt", "voice": "seraphina", "audio_prompt": None, "temp": 0.8, "cfg": 3.0},
        {"name": "seraphina_with_prompt", "voice": "seraphina", "audio_prompt": "seraphina_voice", "temp": 0.8, "cfg": 3.0},
        {"name": "seraphina_low_cfg", "voice": "seraphina", "audio_prompt": "seraphina_voice", "temp": 0.8, "cfg": 1.0},
        
        # Female_03 tests
        {"name": "female03_default", "voice": "female_03", "audio_prompt": "female_03", "temp": 1.0, "cfg": 3.0},
        {"name": "female03_low_temp", "voice": "female_03", "audio_prompt": "female_03", "temp": 0.8, "cfg": 3.0},
        {"name": "female03_low_cfg", "voice": "female_03", "audio_prompt": "female_03", "temp": 1.0, "cfg": 1.5},
        {"name": "female03_both_low", "voice": "female_03", "audio_prompt": "female_03", "temp": 0.8, "cfg": 1.5},
        
        # Baseline S2 without any audio prompt
        {"name": "baseline_s2", "voice": "fable", "audio_prompt": None, "temp": 0.8, "cfg": 3.0},
    ]
    
    print(f"📝 Test text: \"{test_text}\"\n")
    print("🧪 Generating voice samples...\n")
    
    results = []
    
    for config in test_configs:
        print(f"Generating: {config['name']}...")
        
        # First, update voice if needed (for testing without audio prompt)
        if config['voice'] == 'seraphina' and config['audio_prompt'] is None:
            # Temporarily remove audio prompt
            requests.put(
                f"{BASE_URL}/voice_mappings/seraphina",
                json={
                    "voice_id": "seraphina",
                    "primary_speaker": "S2",
                    "audio_prompt": None
                }
            )
        
        # Generate audio
        request_data = {
            "text": test_text,
            "voice_id": config['voice'],
            "speed": 1.0,
            "temperature": config['temp'],
            "cfg_scale": config['cfg']
        }
        
        response = requests.post(
            f"{BASE_URL}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            filename = f"compare_{config['name']}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"  ✅ Saved: {filename}")
            results.append({
                "name": config['name'],
                "file": filename,
                "voice": config['voice'],
                "prompt": config['audio_prompt'],
                "temp": config['temp'],
                "cfg": config['cfg']
            })
        else:
            print(f"  ❌ Failed: {response.status_code}")
    
    # Restore seraphina audio prompt if it was removed
    requests.put(
        f"{BASE_URL}/voice_mappings/seraphina",
        json={
            "voice_id": "seraphina",
            "primary_speaker": "S2", 
            "audio_prompt": "seraphina_voice"
        }
    )
    
    # Print comparison table
    print("\n📊 Comparison Results:")
    print("\n| File | Voice | Audio Prompt | Temp | CFG |")
    print("|------|-------|--------------|------|-----|")
    for r in results:
        prompt = r['prompt'] or "None"
        print(f"| {r['file']} | {r['voice']} | {prompt} | {r['temp']} | {r['cfg']} |")
    
    print("\n🎧 Listen to the files and compare:")
    print("  1. Which sounds most feminine?")
    print("  2. Which matches the audio prompt best?")
    print("  3. Which has the best quality?")
    
    print("\n💡 Key comparisons:")
    print("  - compare_seraphina_no_prompt.wav vs compare_seraphina_with_prompt.wav")
    print("    (Shows effect of seraphina audio prompt)")
    print("  - compare_female03_default.wav vs compare_female03_both_low.wav")
    print("    (Shows effect of parameter tuning)")
    print("  - compare_baseline_s2.wav")
    print("    (Pure S2 voice without any audio prompt)")
    
    print("\n🎯 Based on your previous results:")
    print("  - If no_prompt sounds more feminine, the audio prompt has issues")
    print("  - If low_temp helps, use temperature 0.8")
    print("  - If low_cfg helps, the audio prompt needs less influence")

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
    
    compare_voices()