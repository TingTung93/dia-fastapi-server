# """
# Simple test script for Dia FastAPI TTS Server
# """

import requests
import time
import sys
import os
from pathlib import Path

# Server configuration
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("\nTesting models endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Found {len(models['models'])} models")
            for model in models['models']:
                print(f"   - {model['id']}: {model['name']}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False

def test_voices():
    """Test voices endpoint"""
    print("\nTesting voices endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/voices")
        if response.status_code == 200:
            voices = response.json()
            print(f"✅ Found {len(voices['voices'])} voices")
            for voice in voices['voices'][:5]:  # Show first 5
                print(f"   - {voice['id']} ({voice['style']})")
            return True
        else:
            print(f"❌ Voices endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Voices endpoint failed: {e}")
        return False

def test_sync_generation():
    """Test synchronous TTS generation"""
    print("\nTesting synchronous TTS generation...")

    test_text = "[S1] Hello! This is a test of the Dia text-to-speech system."

    try:
        start_time = time.time()
        response = requests.post(
            f"{SERVER_URL}/generate",
            json={
                "text": test_text,
                "voice_id": "alloy"
            },
            stream=True
        )

        if response.status_code == 200:
            # Save the audio file
            output_file = "test_output_sync.wav"
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            generation_time = time.time() - start_time
            file_size = os.path.getsize(output_file) / 1024  # KB

            print(f"✅ Sync generation successful")
            print(f"   - Time: {generation_time:.2f}s")
            print(f"   - File size: {file_size:.1f} KB")
            print(f"   - Saved to: {output_file}")
            return True
        else:
            print(f"❌ Sync generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Sync generation failed: {e}")
        return False

def test_async_generation():
    """Test asynchronous TTS generation"""
    print("\nTesting asynchronous TTS generation...")

    test_text = "[S1] Testing async mode. [S2] This should return a job ID."

    try:
        # Submit job
        response = requests.post(
            f"{SERVER_URL}/generate?async_mode=true",
            json={
                "text": test_text,
                "voice_id": "echo",
                "temperature": 1.0
            }
        )

        if response.status_code != 200:
            print(f"❌ Async job submission failed: {response.status_code}")
            return False

        job_data = response.json()
        job_id = job_data.get("job_id")
        print(f"✅ Job submitted: {job_id}")

        # Poll for completion
        max_wait = 30  # seconds
        poll_interval = 0.5
        elapsed = 0

        while elapsed < max_wait:
            status_response = requests.get(f"{SERVER_URL}/jobs/{job_id}")
            if status_response.status_code == 200:
                job_status = status_response.json()

                if job_status["status"] == "completed":
                    print(f"✅ Job completed in {elapsed:.1f}s")

                    # Download result
                    result_response = requests.get(f"{SERVER_URL}/jobs/{job_id}/result")
                    if result_response.status_code == 200:
                        output_file = "test_output_async.wav"
                        with open(output_file, "wb") as f:
                            f.write(result_response.content)

                        file_size = os.path.getsize(output_file) / 1024  # KB
                        print(f"✅ Async result downloaded")
                        print(f"   - File size: {file_size:.1f} KB")
                        print(f"   - Saved to: {output_file}")
                        return True
                    else:
                        print(f"❌ Failed to get async result: {result_response.status_code}")
                        return False

            time.sleep(poll_interval)
            elapsed += poll_interval

        print("❌ Job timed out before completing")
        return False

    except Exception as e:
        print(f"❌ Async generation failed: {e}")
        return False

def test_gpu_status():
    """Test GPU status endpoint"""
    print("\nTesting GPU status...")
    try:
        response = requests.get(f"{SERVER_URL}/gpu/status")
        if response.status_code == 200:
            gpu_info = response.json()
            print("✅ GPU status:")
            for gpu in gpu_info["gpus"]:
                print(f"   - {gpu['name']}: {gpu['memory_total']} MB total, {gpu['memory_free']} MB free")
            return True
        else:
            print(f"❌ GPU status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ GPU status failed: {e}")
        return False

def test_queue_stats():
    """Test queue statistics endpoint"""
    print("\nTesting queue stats...")
    try:
        response = requests.get(f"{SERVER_URL}/queue/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Queue stats:")
            print(f"   - Pending: {stats['pending']}")
            print(f"   - Processing: {stats['processing']}")
            print(f"   - Completed: {stats['completed']}")
            return True
        else:
            print(f"❌ Queue stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Queue stats failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Dia TTS Server Test Suite")
    print("=" * 40)

    success = True

    # Run health check first
    if not test_health():
        success = False

    # Only run other tests if health check passes
    else:
        if not test_models():
            success = False
        if not test_voices():
            success = False
        if not test_sync_generation():
            success = False
        if not test_async_generation():
            success = False
        if not test_gpu_status():
            success = False
        if not test_queue_stats():
            success = False

    print("\n🔍 Test Summary")
    print("=" * 40)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()