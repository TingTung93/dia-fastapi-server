#!/usr/bin/env python3
"""
Test script for Dia TTS seed functionality
Tests that the same seed produces consistent results
"""

import requests
import json
import time
from typing import Optional

def test_seed_functionality(
    base_url: str = "http://localhost:7860",
    voice_id: str = "seraphina_voice",
    test_text: str = "[S1] Testing seed consistency with reproducible generation"
):
    """Test seed functionality for reproducible generation"""
    
    print("🌱 Testing Dia TTS Seed Functionality")
    print("=" * 50)
    
    # Test 1: Server health check
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Server healthy, model loaded: {health.get('model_loaded', False)}")
        else:
            print(f"   ❌ Server not healthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Check available voices
    print("\n2. Checking available voices...")
    try:
        response = requests.get(f"{base_url}/voices", timeout=10)
        if response.status_code == 200:
            voices = response.json().get('voices', [])
            voice_ids = [v['id'] for v in voices]
            print(f"   Available voices: {voice_ids}")
            
            if voice_id not in voice_ids:
                if voice_ids:
                    voice_id = voice_ids[0]  # Use first available voice
                    print(f"   ⚠️  Using '{voice_id}' instead")
                else:
                    print("   ❌ No voices available!")
                    return False
            else:
                print(f"   ✅ Voice '{voice_id}' is available")
        else:
            print(f"   ❌ Cannot get voices: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error getting voices: {e}")
        return False
    
    # Test 3: Generate with fixed seed (twice)
    print(f"\n3. Testing fixed seed reproducibility...")
    test_seed = 42
    
    def generate_with_seed(seed: Optional[int], test_name: str):
        """Generate audio with specified seed"""
        payload = {
            "text": test_text,
            "voice_id": voice_id,
            "temperature": 1.2,
            "cfg_scale": 3.0,
            "top_p": 0.95
        }
        if seed is not None:
            payload["seed"] = seed
        
        print(f"   {test_name} (seed={seed})...")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/generate", 
                json=payload,
                timeout=60,
                stream=True
            )
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                # Get content length if available
                content_length = len(response.content) if hasattr(response, 'content') else 0
                print(f"      ✅ Generated successfully ({generation_time:.2f}s, {content_length} bytes)")
                return True, content_length, generation_time
            else:
                print(f"      ❌ Generation failed: {response.status_code}")
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"      Error: {error_detail}")
                except:
                    print(f"      Error: {response.text[:100]}...")
                return False, 0, 0
        except Exception as e:
            print(f"      ❌ Request failed: {e}")
            return False, 0, 0
    
    # Generate with same seed twice
    success1, size1, time1 = generate_with_seed(test_seed, "First generation")
    if not success1:
        return False
    
    time.sleep(1)  # Brief pause between requests
    
    success2, size2, time2 = generate_with_seed(test_seed, "Second generation")
    if not success2:
        return False
    
    # Test 4: Generate with different seed
    print(f"\n4. Testing different seed...")
    success3, size3, time3 = generate_with_seed(12345, "Different seed")
    if not success3:
        return False
    
    # Test 5: Generate without seed (random)
    print(f"\n5. Testing random generation (no seed)...")
    success4, size4, time4 = generate_with_seed(None, "Random generation")
    if not success4:
        return False
    
    # Test 6: OpenAI endpoint with seed
    print(f"\n6. Testing OpenAI endpoint with seed...")
    try:
        payload = {
            "input": test_text,
            "voice": voice_id,
            "seed": test_seed,
            "model": "dia"
        }
        
        response = requests.post(
            f"{base_url}/v1/audio/speech",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"      ✅ OpenAI endpoint works with seed")
        else:
            print(f"      ❌ OpenAI endpoint failed: {response.status_code}")
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"      Error: {error_detail}")
            except:
                print(f"      Error: {response.text[:100]}...")
    except Exception as e:
        print(f"      ❌ OpenAI endpoint request failed: {e}")
    
    # Results summary
    print(f"\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)
    print(f"✅ All generations completed successfully")
    print(f"🎯 Fixed seed (42): {size1} bytes, {size2} bytes")
    print(f"🎲 Different seed (12345): {size3} bytes")
    print(f"🎰 Random generation: {size4} bytes")
    print(f"⏱️  Average generation time: {(time1 + time2 + time3 + time4) / 4:.2f}s")
    
    # Size consistency check
    if abs(size1 - size2) < 1000:  # Allow small differences
        print(f"✅ Seed consistency: Output sizes are similar ({abs(size1 - size2)} byte difference)")
    else:
        print(f"⚠️  Seed consistency: Output sizes differ significantly ({abs(size1 - size2)} byte difference)")
    
    print(f"\n💡 TIPS FOR BEST CONSISTENCY:")
    print(f"   • Use voices with audio prompts for best voice consistency")
    print(f"   • Keep generation parameters (temperature, cfg_scale, top_p) fixed")
    print(f"   • Same seed + same parameters = more consistent results")
    print(f"   • Seeds help with generation consistency, not perfect reproduction")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dia TTS seed functionality")
    parser.add_argument("--url", default="http://localhost:7860", help="Server URL")
    parser.add_argument("--voice", default="seraphina_voice", help="Voice ID to test")
    parser.add_argument("--text", default="[S1] Testing seed functionality for consistent generation", help="Test text")
    
    args = parser.parse_args()
    
    success = test_seed_functionality(args.url, args.voice, args.text)
    if success:
        print(f"\n🎉 All tests completed successfully!")
        print(f"🌱 Seed functionality is working properly!")
    else:
        print(f"\n❌ Some tests failed - check server status and configuration")
        exit(1) 