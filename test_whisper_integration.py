#!/usr/bin/env python3
"""
Test script to validate Whisper integration and audio transcription
"""

import os
import sys
import time
import requests
import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio():
    """Create a test audio file for transcription testing"""
    print("🎵 Creating test audio file...")
    
    # Create a simple test audio (2 seconds, 44.1kHz)
    sample_rate = 44100
    duration = 2.0
    
    # Generate a sine wave with speech-like characteristics
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix multiple frequencies to simulate speech
    frequency1 = 440  # A4
    frequency2 = 880  # A5
    frequency3 = 220  # A3
    
    audio_data = (
        0.3 * np.sin(2 * np.pi * frequency1 * t) +
        0.2 * np.sin(2 * np.pi * frequency2 * t) +
        0.1 * np.sin(2 * np.pi * frequency3 * t)
    )
    
    # Add some amplitude variation to make it more speech-like
    envelope = np.exp(-t * 0.5) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    audio_data = audio_data * envelope
    
    # Save test audio
    test_audio_path = Path("test_audio_for_whisper.wav")
    sf.write(test_audio_path, audio_data, sample_rate)
    
    print(f"✅ Created test audio: {test_audio_path}")
    return test_audio_path

def test_whisper_availability():
    """Test if Whisper is available"""
    print("\n🔍 Testing Whisper Availability...")
    
    try:
        import whisper
        print("✅ OpenAI Whisper imported successfully")
        
        # Test model loading
        try:
            model = whisper.load_model("tiny")
            print("✅ Whisper 'tiny' model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
            
    except ImportError:
        print("❌ OpenAI Whisper not available")
        print("   Install with: pip install openai-whisper")
        return False

def test_server_whisper_status():
    """Test server's Whisper status endpoint"""
    print("\n📊 Testing Server Whisper Status...")
    
    try:
        response = requests.get('http://localhost:7860/whisper/status', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Whisper status endpoint working")
            print(f"   Available: {data.get('available')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Auto-transcribe: {data.get('auto_transcribe')}")
            print(f"   Model size: {data.get('model_size')}")
            return data.get('available', False)
        else:
            print(f"❌ Whisper status failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Whisper status error: {e}")
        return False

def test_whisper_loading():
    """Test Whisper model loading via API"""
    print("\n🔄 Testing Whisper Model Loading...")
    
    try:
        response = requests.post('http://localhost:7860/whisper/load', timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Whisper model loading successful")
            print(f"   Message: {data.get('message')}")
            return True
        elif response.status_code == 503:
            print("⚠️  Whisper not installed on server")
            return False
        else:
            print(f"❌ Whisper loading failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail')}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ Whisper loading error: {e}")
        return False

def test_audio_prompt_upload_and_transcription():
    """Test audio prompt upload with transcription"""
    print("\n🎤 Testing Audio Prompt Upload and Transcription...")
    
    # Create test audio file
    test_audio_path = create_test_audio()
    
    try:
        # Upload audio prompt
        with open(test_audio_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'prompt_id': 'whisper_test_voice'}
            response = requests.post(
                'http://localhost:7860/audio_prompts/upload',
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            print("✅ Audio prompt upload successful")
            upload_data = response.json()
            print(f"   Duration: {upload_data.get('duration'):.2f}s")
            
            # Test transcription
            return test_prompt_transcription('whisper_test_voice')
        else:
            print(f"❌ Audio prompt upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Audio prompt upload error: {e}")
        return False
    finally:
        # Cleanup test file
        if test_audio_path.exists():
            test_audio_path.unlink()

def test_prompt_transcription(prompt_id):
    """Test transcription of a specific audio prompt"""
    print(f"\n🗣️ Testing Transcription for '{prompt_id}'...")
    
    try:
        response = requests.post(f'http://localhost:7860/audio_prompts/{prompt_id}/transcribe', timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Transcription successful")
            print(f"   Transcript: {data.get('transcript')}")
            print(f"   Saved to: {data.get('saved_to')}")
            return True
        elif response.status_code == 503:
            print("⚠️  Whisper not available for transcription")
            return False
        else:
            print(f"❌ Transcription failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail')}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return False

def test_audio_prompt_discovery():
    """Test audio prompt discovery with transcription"""
    print("\n🔍 Testing Audio Prompt Discovery...")
    
    try:
        response = requests.post(
            'http://localhost:7860/audio_prompts/discover',
            json={"force_retranscribe": False},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Audio prompt discovery successful")
            print(f"   Message: {data.get('message')}")
            print(f"   Total prompts: {data.get('total_prompts')}")
            
            discovered = data.get('discovered', [])
            if discovered:
                print("   Discovered prompts:")
                for prompt in discovered:
                    transcript = prompt.get('transcript', 'No transcript')
                    transcript_preview = transcript[:50] + "..." if len(transcript) > 50 else transcript
                    print(f"     • {prompt['prompt_id']}: {transcript_preview}")
            
            return True
        else:
            print(f"❌ Discovery failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Discovery error: {e}")
        return False

def test_generation_with_transcribed_prompt():
    """Test TTS generation using a voice with transcribed prompt"""
    print("\n🎵 Testing Generation with Transcribed Prompt...")
    
    try:
        # Try to generate with the test voice
        response = requests.post(
            'http://localhost:7860/generate',
            json={
                "text": "This is a test of voice cloning using the transcribed audio prompt.",
                "voice_id": "whisper_test_voice",
                "speed": 1.0
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Generation with transcribed prompt successful")
            
            # Save test output
            with open("test_whisper_voice_generation.wav", "wb") as f:
                f.write(response.content)
            print("   Audio saved to test_whisper_voice_generation.wav")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def cleanup_test_data():
    """Clean up test audio prompts and files"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # Delete test voice mapping
        response = requests.delete('http://localhost:7860/voice_mappings/whisper_test_voice')
        if response.status_code == 200:
            print("✅ Cleaned up test voice mapping")
        
        # Delete test audio prompt
        response = requests.delete('http://localhost:7860/audio_prompts/whisper_test_voice')
        if response.status_code == 200:
            print("✅ Cleaned up test audio prompt")
        
        # Remove local test files
        test_files = [
            "test_audio_for_whisper.wav",
            "test_whisper_voice_generation.wav"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"✅ Removed {file_path}")
        
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

def main():
    """Run all Whisper integration tests"""
    print("🎙️ Testing Whisper Integration for Dia TTS Server")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:7860/health', timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the server first.")
            sys.exit(1)
    except:
        print("❌ Cannot connect to server. Please start the server first.")
        sys.exit(1)
    
    tests = [
        ("Whisper Library", test_whisper_availability),
        ("Server Whisper Status", test_server_whisper_status),
        ("Whisper Model Loading", test_whisper_loading),
        ("Audio Upload & Transcription", test_audio_prompt_upload_and_transcription),
        ("Audio Prompt Discovery", test_audio_prompt_discovery),
        ("Generation with Transcript", test_generation_with_transcribed_prompt),
    ]
    
    passed = 0
    total = len(tests)
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Cleanup
    cleanup_test_data()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Whisper Integration Test Results: {passed}/{total} passed")
    
    print("\n📋 Test Summary:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 All Whisper integration tests passed!")
        print("   Whisper is properly configured for automatic transcription.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        if not results.get("Whisper Library", False):
            print("   💡 Install Whisper: pip install openai-whisper")
        if not results.get("Server Whisper Status", False):
            print("   💡 Check server configuration and restart if needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())