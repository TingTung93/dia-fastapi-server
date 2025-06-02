#!/usr/bin/env python3
"""
Test script to validate CUDA optimizations and audio prompting functionality
"""

import os
import sys
import time
import requests
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cuda_availability():
    """Test CUDA availability and configuration"""
    print("🔍 Testing CUDA Configuration...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device=f"cuda:{i}")
            del test_tensor
            torch.cuda.empty_cache()
            print(f"   ✅ GPU {i} memory test passed")
        except Exception as e:
            print(f"   ❌ GPU {i} memory test failed: {e}")
    
    return True

def test_audio_prompt_upload():
    """Test audio prompt upload functionality"""
    print("\n🎤 Testing Audio Prompt Upload...")
    
    # Create a simple test audio file
    sample_rate = 44100
    duration = 2.0  # 2 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save test audio
    test_audio_path = "test_audio_prompt.wav"
    sf.write(test_audio_path, audio_data, sample_rate)
    
    try:
        # Upload audio prompt
        with open(test_audio_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'prompt_id': 'test_voice'}
            response = requests.post(
                'http://localhost:7860/audio_prompts/upload',
                files=files,
                data=data,
                timeout=10
            )
        
        if response.status_code == 200:
            print("✅ Audio prompt upload successful")
            return True
        else:
            print(f"❌ Audio prompt upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Audio prompt upload error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)

def test_voice_mapping():
    """Test voice mapping with audio prompt"""
    print("\n🗣️ Testing Voice Mapping...")
    
    try:
        # Create voice mapping
        voice_data = {
            "voice_id": "test_voice_mapped",
            "style": "test",
            "primary_speaker": "S1",
            "audio_prompt": "test_voice"
        }
        
        response = requests.post(
            'http://localhost:7860/voice_mappings',
            json=voice_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Voice mapping creation successful")
            return True
        else:
            print(f"❌ Voice mapping failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Voice mapping error: {e}")
        return False

def test_generation_sync():
    """Test synchronous TTS generation"""
    print("\n🎵 Testing Synchronous Generation...")
    
    try:
        tts_data = {
            "text": "Hello, this is a test of the optimized CUDA TTS system.",
            "voice_id": "aria",
            "speed": 1.0
        }
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:7860/generate',
            json=tts_data,
            timeout=30
        )
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Sync generation successful ({generation_time:.2f}s)")
            
            # Save test output
            with open("test_output_sync.wav", "wb") as f:
                f.write(response.content)
            print("   Audio saved to test_output_sync.wav")
            return True
        else:
            print(f"❌ Sync generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Sync generation error: {e}")
        return False

def test_generation_async():
    """Test asynchronous TTS generation"""
    print("\n⚡ Testing Asynchronous Generation...")
    
    try:
        tts_data = {
            "text": "This is an asynchronous test of the CUDA optimized server.",
            "voice_id": "atlas",
            "speed": 1.0
        }
        
        # Submit job
        response = requests.post(
            'http://localhost:7860/generate?async_mode=true',
            json=tts_data,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Async job submission failed: {response.status_code}")
            return False
            
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ Job submitted: {job_id}")
        
        # Poll for completion
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f'http://localhost:7860/jobs/{job_id}')
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data['status'] == 'completed':
                    print("✅ Async job completed")
                    
                    # Download result
                    result_response = requests.get(f'http://localhost:7860/jobs/{job_id}/result')
                    if result_response.status_code == 200:
                        with open("test_output_async.wav", "wb") as f:
                            f.write(result_response.content)
                        print("   Audio saved to test_output_async.wav")
                        return True
                elif status_data['status'] == 'failed':
                    print(f"❌ Async job failed: {status_data.get('error_message', 'Unknown error')}")
                    return False
            
            time.sleep(1)
        
        print("❌ Async job timeout")
        return False
        
    except Exception as e:
        print(f"❌ Async generation error: {e}")
        return False

def test_gpu_status():
    """Test GPU status endpoint"""
    print("\n📊 Testing GPU Status...")
    
    try:
        response = requests.get('http://localhost:7860/gpu/status', timeout=10)
        
        if response.status_code == 200:
            gpu_data = response.json()
            print("✅ GPU status endpoint working")
            print(f"   GPU Mode: {gpu_data.get('gpu_mode')}")
            print(f"   GPU Count: {gpu_data.get('gpu_count')}")
            print(f"   Multi-GPU: {gpu_data.get('use_multi_gpu')}")
            
            if 'gpu_memory' in gpu_data:
                for gpu_name, memory_info in gpu_data['gpu_memory'].items():
                    if 'total_gb' in memory_info:
                        print(f"   {gpu_name}: {memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB")
            
            return True
        else:
            print(f"❌ GPU status failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ GPU status error: {e}")
        return False

def test_queue_stats():
    """Test queue statistics with memory pressure"""
    print("\n📈 Testing Queue Stats...")
    
    try:
        response = requests.get('http://localhost:7860/queue/stats', timeout=10)
        
        if response.status_code == 200:
            stats_data = response.json()
            print("✅ Queue stats endpoint working")
            print(f"   Active Workers: {stats_data.get('active_workers')}")
            print(f"   Total Workers: {stats_data.get('total_workers')}")
            
            if 'memory_pressure' in stats_data:
                memory_pressure = stats_data['memory_pressure']
                for gpu_name, pressure_info in memory_pressure.items():
                    if 'pressure' in pressure_info:
                        pressure_pct = pressure_info['pressure'] * 100
                        status = pressure_info['status']
                        print(f"   {gpu_name}: {pressure_pct:.1f}% ({status})")
            
            return True
        else:
            print(f"❌ Queue stats failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Queue stats error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing CUDA Optimized TTS Server (No OpenAI Dependencies)")
    print("=" * 50)
    
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
        test_cuda_availability,
        test_gpu_status,
        test_queue_stats,
        test_generation_sync,
        test_generation_async,
        test_audio_prompt_upload,
        test_voice_mapping,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! CUDA optimizations are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())