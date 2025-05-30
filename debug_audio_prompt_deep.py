#!/usr/bin/env python3
"""Deep debugging for audio prompt issues"""

import requests
import json
import os
import sys
import numpy as np
import soundfile as sf

BASE_URL = "http://localhost:7860"

def enable_debug_mode():
    """Enable debug mode on server"""
    print("🔧 Enabling debug mode...")
    response = requests.put(
        f"{BASE_URL}/config",
        json={"debug_mode": True, "show_prompts": True}
    )
    if response.status_code == 200:
        print("✅ Debug mode enabled\n")

def check_audio_file(filepath):
    """Analyze the audio prompt file"""
    print(f"🎵 Analyzing audio file: {filepath}\n")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    try:
        data, samplerate = sf.read(filepath)
        duration = len(data) / samplerate
        
        print(f"  ✅ File exists")
        print(f"  📊 Duration: {duration:.2f} seconds")
        print(f"  🎵 Sample rate: {samplerate} Hz")
        print(f"  📐 Shape: {data.shape}")
        print(f"  📏 Samples: {len(data):,}")
        
        # Check if mono
        if len(data.shape) > 1:
            print(f"  ⚠️  Stereo audio detected ({data.shape[1]} channels)")
            print(f"     Converting to mono might help")
        else:
            print(f"  ✅ Mono audio")
        
        # Check duration
        if duration < 3:
            print(f"  ⚠️  Very short audio ({duration:.1f}s) - aim for 5-15 seconds")
        elif duration > 30:
            print(f"  ⚠️  Very long audio ({duration:.1f}s) - might affect quality")
        else:
            print(f"  ✅ Good duration for voice cloning")
            
        # Check for silence
        rms = np.sqrt(np.mean(data**2))
        print(f"  🔊 RMS level: {rms:.4f}")
        if rms < 0.01:
            print(f"  ⚠️  Very quiet audio - might be mostly silence")
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")
        return False

def test_without_audio_prompt(voice_id, text):
    """Test generation without audio prompt for comparison"""
    print("\n🧪 Test 1: WITHOUT audio prompt (baseline)")
    
    # Create temporary voice without audio prompt
    temp_voice = f"{voice_id}_no_prompt"
    
    # Get current config
    response = requests.get(f"{BASE_URL}/voice_mappings")
    if response.status_code == 200:
        mappings = response.json()
        if voice_id in mappings:
            config = mappings[voice_id].copy()
            config['voice_id'] = temp_voice
            config['audio_prompt'] = None
            config['audio_prompt_transcript'] = None
            
            # Create temp voice
            requests.post(f"{BASE_URL}/voice_mappings", json=config)
    
    # Generate
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"text": text, "voice_id": temp_voice, "speed": 1.0}
    )
    
    if response.status_code == 200:
        with open(f"test_no_prompt.wav", "wb") as f:
            f.write(response.content)
        print("  ✅ Saved: test_no_prompt.wav")
    
    # Clean up temp voice
    requests.delete(f"{BASE_URL}/voice_mappings/{temp_voice}")

def test_with_explicit_tags(voice_id, text):
    """Test with explicit speaker tags"""
    print("\n🧪 Test 2: With explicit [S2] tags")
    
    tagged_text = f"[S2] {text} [S2]"
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"text": tagged_text, "voice_id": voice_id, "speed": 1.0}
    )
    
    if response.status_code == 200:
        with open(f"test_explicit_s2.wav", "wb") as f:
            f.write(response.content)
        print("  ✅ Saved: test_explicit_s2.wav")

def test_different_params(voice_id, text):
    """Test with different generation parameters"""
    print("\n🧪 Test 3: Different generation parameters")
    
    params_sets = [
        {"name": "low_temp", "temperature": 0.8, "cfg_scale": 2.0},
        {"name": "high_cfg", "temperature": 1.0, "cfg_scale": 5.0},
        {"name": "focused", "temperature": 0.9, "cfg_scale": 3.5, "top_p": 0.85}
    ]
    
    for params in params_sets:
        name = params.pop("name")
        print(f"\n  Testing {name}: {params}")
        
        request_data = {
            "text": text,
            "voice_id": voice_id,
            "speed": 1.0,
            **params
        }
        
        response = requests.post(f"{BASE_URL}/generate", json=request_data)
        
        if response.status_code == 200:
            filename = f"test_{name}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"  ✅ Saved: {filename}")

def check_model_loading():
    """Check if model is receiving audio prompt"""
    print("\n🔍 Checking model configuration...")
    
    # Check GPU status
    response = requests.get(f"{BASE_URL}/gpu/status")
    if response.status_code == 200:
        gpu_info = response.json()
        print(f"  GPU Mode: {gpu_info.get('gpu_mode')}")
        print(f"  Models loaded: {gpu_info.get('models_loaded')}")

def test_direct_api():
    """Test if we can generate with explicit paths"""
    print("\n🧪 Test 4: Direct generation test")
    
    # This tests if the issue is in voice mapping or generation
    test_data = {
        "text": "[S2] Testing direct generation with female voice [S2]",
        "voice_id": "fable",  # Uses S2 by default
        "speed": 1.0,
        "temperature": 1.0,
        "cfg_scale": 3.0
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=test_data)
    if response.status_code == 200:
        with open("test_fable_s2.wav", "wb") as f:
            f.write(response.content)
        print("  ✅ Saved: test_fable_s2.wav (baseline female voice)")

def main():
    print("🔍 Deep Audio Prompt Debugging\n")
    
    # Enable debug mode
    enable_debug_mode()
    
    # Get voice info
    voice_id = input("Enter voice ID (default: seraphina): ").strip() or "seraphina"
    
    # Check voice config
    print(f"\n📋 Checking voice configuration for '{voice_id}'...")
    response = requests.get(f"{BASE_URL}/voice_mappings")
    if response.status_code != 200:
        print("❌ Failed to get voice mappings")
        return
    
    mappings = response.json()
    if voice_id not in mappings:
        print(f"❌ Voice '{voice_id}' not found")
        return
    
    config = mappings[voice_id]
    print(f"\n  Current Configuration:")
    print(f"  • Primary Speaker: {config.get('primary_speaker')}")
    print(f"  • Audio Prompt: {config.get('audio_prompt')}")
    print(f"  • Transcript: {config.get('audio_prompt_transcript', 'None')[:50]}...")
    
    # Check audio prompt file
    if config.get('audio_prompt'):
        audio_prompt_path = f"audio_prompts/{config['audio_prompt']}.wav"
        check_audio_file(audio_prompt_path)
    else:
        print("\n❌ No audio prompt configured")
        return
    
    # Test text
    test_text = "Hello, this is a test of the voice cloning system. I should sound like the audio prompt."
    
    print("\n" + "="*60)
    print("\n🧪 Running comparison tests...")
    print(f"Test text: '{test_text}'")
    
    # Run tests
    test_without_audio_prompt(voice_id, test_text)
    test_with_explicit_tags(voice_id, test_text)
    test_different_params(voice_id, test_text)
    test_direct_api()
    
    # Check server logs
    print("\n" + "="*60)
    print("\n📋 IMPORTANT: Check server console output!")
    print("Look for these messages:")
    print("  • 'Audio Prompt: Yes' or 'Audio Prompt: No'")
    print("  • 'Using audio prompt: /path/to/file'")
    print("  • Any warning messages about audio prompts")
    
    print("\n💡 Troubleshooting Steps:")
    print("\n1. Compare the generated files:")
    print("   - test_no_prompt.wav (baseline without audio prompt)")
    print("   - test_explicit_s2.wav (with forced S2 tags)")
    print("   - test_fable_s2.wav (default female voice)")
    print("   - test_high_cfg.wav (stronger prompt adherence)")
    
    print("\n2. If all sound similar (ignoring audio prompt):")
    print("   • The audio prompt might not be loading properly")
    print("   • Check file permissions on audio_prompts directory")
    print("   • Try re-uploading the audio prompt")
    print("   • Check if the Dia model version supports audio prompts")
    
    print("\n3. If S2 still sounds masculine:")
    print("   • The model's S2 voice might not match expectations")
    print("   • Try the 'fable' voice as baseline (uses S2)")
    print("   • Your audio prompt might be ambiguous")
    
    print("\n4. Alternative approach:")
    print("   • Try without audio prompt first to establish baseline")
    print("   • Use a different female voice recording")
    print("   • Ensure recording is very clearly feminine")
    
    print("\n5. Server-side debugging:")
    print("   • Add logging to server.py in generate_audio_from_text()")
    print("   • Log the actual audio_prompt path being passed")
    print("   • Check if model.generate() is receiving the audio_prompt")

if __name__ == "__main__":
    main()