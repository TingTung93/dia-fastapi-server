#!/usr/bin/env python3
"""Advanced debugging for audio prompt issues with current server features"""

import requests
import json
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
import time

BASE_URL = "http://localhost:7860"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 60)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def check_server_status():
    """Check if server is running and get basic status"""
    print_section("Server Status Check")
    
    try:
        # Health check
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print_success(f"Server is running (status: {health_data.get('status')})")
            print_info(f"Model loaded: {health_data.get('model_loaded')}")
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
            
        # GPU status
        response = requests.get(f"{BASE_URL}/gpu/status")
        if response.status_code == 200:
            gpu_data = response.json()
            print_info(f"GPU Mode: {gpu_data.get('gpu_mode')}")
            print_info(f"GPU Count: {gpu_data.get('gpu_count')}")
            print_info(f"Multi-GPU: {gpu_data.get('use_multi_gpu')}")
            
            if gpu_data.get('gpu_memory'):
                for gpu_id, memory in gpu_data['gpu_memory'].items():
                    if 'error' not in memory:
                        allocated = memory.get('allocated_gb', 0)
                        total = memory.get('total_gb', 0)
                        print_info(f"{gpu_id}: {allocated:.1f}GB / {total:.1f}GB used")
        
        # Whisper status
        response = requests.get(f"{BASE_URL}/whisper/status")
        if response.status_code == 200:
            whisper_data = response.json()
            print_info(f"Whisper available: {whisper_data.get('available')}")
            print_info(f"Whisper loaded: {whisper_data.get('model_loaded')}")
            if whisper_data.get('model_size'):
                print_info(f"Whisper model: {whisper_data.get('model_size')}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print_error(f"Cannot connect to server: {e}")
        print_info(f"Make sure server is running on {BASE_URL}")
        return False

def enable_debug_mode():
    """Enable comprehensive debug mode"""
    print_section("Enabling Debug Mode")
    
    try:
        response = requests.put(
            f"{BASE_URL}/config",
            json={
                "debug_mode": True, 
                "show_prompts": True,
                "save_outputs": True,
                "auto_discover_prompts": True,
                "auto_transcribe": True
            }
        )
        if response.status_code == 200:
            print_success("Enhanced debug mode enabled")
            return True
        else:
            print_error(f"Failed to enable debug mode: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error enabling debug mode: {e}")
        return False

def discover_audio_prompts():
    """Trigger audio prompt discovery"""
    print_section("Audio Prompt Discovery")
    
    try:
        # Trigger discovery with forced retranscription
        response = requests.post(
            f"{BASE_URL}/audio_prompts/discover",
            json={"force_retranscribe": False}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Discovery complete: {data['message']}")
            print_info(f"Total prompts: {data['total_prompts']}")
            
            if data.get('discovered'):
                for prompt in data['discovered']:
                    print_info(f"  • {prompt['prompt_id']}: {prompt['duration']:.1f}s")
                    if prompt.get('transcript'):
                        preview = prompt['transcript'][:50] + "..." if len(prompt['transcript']) > 50 else prompt['transcript']
                        print_info(f"    Transcript: {preview}")
            return True
        else:
            print_error(f"Discovery failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Discovery error: {e}")
        return False

def get_audio_prompt_metadata():
    """Get detailed metadata for all audio prompts"""
    print_section("Audio Prompt Metadata")
    
    try:
        response = requests.get(f"{BASE_URL}/audio_prompts/metadata")
        if response.status_code == 200:
            metadata = response.json()
            
            if not metadata:
                print_warning("No audio prompts found")
                return {}
            
            for prompt_id, info in metadata.items():
                print_info(f"Prompt: {prompt_id}")
                print(f"    Duration: {info['duration']}s")
                print(f"    Sample Rate: {info['sample_rate']}Hz")
                print(f"    File: {Path(info['file_path']).name}")
                
                if info.get('transcript'):
                    preview = info['transcript'][:80] + "..." if len(info['transcript']) > 80 else info['transcript']
                    print(f"    Transcript ({info.get('transcript_source', 'unknown')}): {preview}")
                else:
                    print_warning("    No transcript available")
                
                if info.get('discovered_at'):
                    print(f"    Discovered: {info['discovered_at']}")
                    
            return metadata
        else:
            print_error(f"Failed to get metadata: {response.status_code}")
            return {}
            
    except Exception as e:
        print_error(f"Metadata error: {e}")
        return {}

def check_voice_mappings():
    """Check current voice mappings and audio prompt assignments"""
    print_section("Voice Mappings Analysis")
    
    try:
        response = requests.get(f"{BASE_URL}/voice_mappings")
        if response.status_code == 200:
            mappings = response.json()
            
            for voice_id, config in mappings.items():
                print_info(f"Voice: {voice_id}")
                print(f"    Style: {config.get('style')}")
                print(f"    Primary Speaker: {config.get('primary_speaker')}")
                
                if config.get('audio_prompt'):
                    print_success(f"    Audio Prompt: {config['audio_prompt']}")
                    if config.get('audio_prompt_transcript'):
                        transcript_preview = config['audio_prompt_transcript'][:60] + "..." if len(config['audio_prompt_transcript']) > 60 else config['audio_prompt_transcript']
                        print(f"    Transcript: {transcript_preview}")
                    else:
                        print_warning("    No transcript configured")
                else:
                    print_warning("    No audio prompt configured")
                    
            return mappings
        else:
            print_error(f"Failed to get voice mappings: {response.status_code}")
            return {}
            
    except Exception as e:
        print_error(f"Voice mapping error: {e}")
        return {}

def test_whisper_transcription(prompt_id):
    """Test Whisper transcription for a specific prompt"""
    print_section(f"Testing Whisper Transcription for '{prompt_id}'")
    
    try:
        response = requests.post(f"{BASE_URL}/audio_prompts/{prompt_id}/transcribe")
        if response.status_code == 200:
            data = response.json()
            print_success("Transcription successful")
            print_info(f"Transcript: {data['transcript']}")
            print_info(f"Saved to: {data['saved_to']}")
            return data['transcript']
        elif response.status_code == 503:
            print_warning("Whisper not available")
            return None
        else:
            print_error(f"Transcription failed: {response.status_code}")
            try:
                error_data = response.json()
                print_error(f"Error: {error_data.get('detail', 'Unknown error')}")
            except:
                pass
            return None
            
    except Exception as e:
        print_error(f"Transcription error: {e}")
        return None

def test_generation_sync(voice_id, text, test_name="sync"):
    """Test synchronous generation"""
    print_section(f"Testing Sync Generation ({test_name})")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": text,
                "voice_id": voice_id,
                "speed": 1.0,
                "temperature": 1.0,
                "cfg_scale": 3.0
            }
        )
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            filename = f"test_{test_name}_{voice_id}.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print_success(f"Generation successful in {generation_time:.2f}s")
            print_info(f"Saved: {filename}")
            print_info(f"File size: {len(response.content):,} bytes")
            
            # Check for debug headers
            if 'X-Generation-ID' in response.headers:
                print_info(f"Generation ID: {response.headers['X-Generation-ID']}")
                
            return True
        else:
            print_error(f"Generation failed: {response.status_code}")
            try:
                error_data = response.json()
                print_error(f"Error: {error_data.get('detail', 'Unknown error')}")
            except:
                pass
            return False
            
    except Exception as e:
        print_error(f"Generation error: {e}")
        return False

def test_generation_async(voice_id, text, test_name="async"):
    """Test asynchronous generation"""
    print_section(f"Testing Async Generation ({test_name})")
    
    try:
        # Start async job
        response = requests.post(
            f"{BASE_URL}/generate?async_mode=true",
            json={
                "text": text,
                "voice_id": voice_id,
                "speed": 1.0,
                "temperature": 1.0,
                "cfg_scale": 3.0
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print_success(f"Job queued: {job_id}")
            
            # Poll for completion
            max_wait = 30  # 30 seconds max
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                job_response = requests.get(f"{BASE_URL}/jobs/{job_id}")
                if job_response.status_code == 200:
                    job_data = job_response.json()
                    status = job_data['status']
                    
                    if status == 'completed':
                        print_success(f"Job completed in {job_data.get('generation_time', 0):.2f}s")
                        
                        # Download result
                        result_response = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                        if result_response.status_code == 200:
                            filename = f"test_{test_name}_{voice_id}_async.wav"
                            with open(filename, "wb") as f:
                                f.write(result_response.content)
                            print_info(f"Saved: {filename}")
                            return True
                        else:
                            print_error("Failed to download result")
                            return False
                            
                    elif status == 'failed':
                        print_error(f"Job failed: {job_data.get('error_message')}")
                        return False
                        
                    elif status in ['pending', 'processing']:
                        print_info(f"Job status: {status}")
                        time.sleep(1)
                    else:
                        print_warning(f"Unknown job status: {status}")
                        
                else:
                    print_error(f"Failed to check job status: {job_response.status_code}")
                    return False
                    
            print_error("Job timed out")
            return False
            
        else:
            print_error(f"Failed to queue job: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Async generation error: {e}")
        return False

def compare_generations():
    """Compare different generation approaches"""
    print_section("Generation Comparison")
    
    test_text = "Hello, this is a comprehensive test of the voice cloning system. The audio prompt should significantly influence the voice characteristics."
    
    # Test parameters for comparison
    test_configs = [
        {"name": "baseline", "temperature": 1.0, "cfg_scale": 3.0},
        {"name": "high_cfg", "temperature": 1.0, "cfg_scale": 5.0},
        {"name": "low_temp", "temperature": 0.8, "cfg_scale": 3.0},
        {"name": "high_temp", "temperature": 1.5, "cfg_scale": 3.0},
    ]
    
    for config in test_configs:
        try:
            response = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": test_text,
                    "voice_id": "seraphina",  # Assuming this exists
                    "speed": 1.0,
                    "temperature": config["temperature"],
                    "cfg_scale": config["cfg_scale"]
                }
            )
            
            if response.status_code == 200:
                filename = f"comparison_{config['name']}.wav"
                with open(filename, "wb") as f:
                    f.write(response.content)
                print_success(f"{config['name']}: Saved {filename}")
            else:
                print_error(f"{config['name']}: Generation failed ({response.status_code})")
                
        except Exception as e:
            print_error(f"{config['name']}: Error - {e}")

def check_generation_logs():
    """Check recent generation logs"""
    print_section("Recent Generation Logs")
    
    try:
        response = requests.get(f"{BASE_URL}/logs?limit=10")
        if response.status_code == 200:
            data = response.json()
            logs = data.get('logs', [])
            
            if not logs:
                print_warning("No generation logs found")
                return
                
            print_info(f"Found {len(logs)} recent generations")
            
            for log in logs[:5]:  # Show last 5
                timestamp = log['timestamp']
                text_preview = log['text'][:40] + "..." if len(log['text']) > 40 else log['text']
                print_info(f"[{timestamp}] {log['voice']} - \"{text_preview}\"")
                print(f"    Generation time: {log['generation_time']:.2f}s")
                print(f"    Audio prompt used: {log['audio_prompt_used']}")
                if log.get('file_path'):
                    print(f"    File: {Path(log['file_path']).name}")
                    
        else:
            print_error(f"Failed to get logs: {response.status_code}")
            
    except Exception as e:
        print_error(f"Log check error: {e}")

def run_comprehensive_tests():
    """Run comprehensive audio prompt tests"""
    print_header("COMPREHENSIVE AUDIO PROMPT TESTING")
    
    # Get available voice with audio prompt
    voice_mappings = check_voice_mappings()
    test_voice = None
    
    for voice_id, config in voice_mappings.items():
        if config.get('audio_prompt'):
            test_voice = voice_id
            break
    
    if not test_voice:
        print_warning("No voices with audio prompts found. Creating test setup...")
        # Check if we have any audio prompts available
        metadata = get_audio_prompt_metadata()
        if metadata:
            # Use first available prompt
            first_prompt = list(metadata.keys())[0]
            test_voice = "test_voice"
            
            # Create a test voice mapping
            try:
                response = requests.post(
                    f"{BASE_URL}/voice_mappings",
                    json={
                        "voice_id": test_voice,
                        "style": "custom",
                        "primary_speaker": "S2",
                        "audio_prompt": first_prompt,
                        "audio_prompt_transcript": metadata[first_prompt].get('transcript')
                    }
                )
                if response.status_code == 200:
                    print_success(f"Created test voice '{test_voice}' with prompt '{first_prompt}'")
                else:
                    print_error("Failed to create test voice")
                    return
            except Exception as e:
                print_error(f"Error creating test voice: {e}")
                return
        else:
            print_error("No audio prompts available for testing")
            print_info("Please upload an audio prompt file to the audio_prompts/ directory")
            return
    
    print_info(f"Using voice '{test_voice}' for testing")
    
    test_text = "This is a detailed test of the audio prompt system. The voice should closely match the characteristics of the uploaded audio sample."
    
    # Test 1: Sync generation
    test_generation_sync(test_voice, test_text, "with_prompt")
    
    # Test 2: Async generation
    test_generation_async(test_voice, test_text, "with_prompt")
    
    # Test 3: Compare with default voice
    test_generation_sync("aria", test_text, "baseline_aria")
    
    # Test 4: Parameter comparison
    compare_generations()
    
    # Test 5: Check logs
    check_generation_logs()

def main():
    """Main debug function"""
    print_header("DIA TTS SERVER - COMPREHENSIVE DEBUG TOOL")
    print_info("This tool will thoroughly test your audio prompt setup")
    
    # Step 1: Check server status
    if not check_server_status():
        return
    
    # Step 2: Enable debug mode
    if not enable_debug_mode():
        return
    
    # Step 3: Discover audio prompts
    if not discover_audio_prompts():
        print_warning("Audio prompt discovery failed, continuing with existing prompts...")
    
    # Step 4: Get metadata
    metadata = get_audio_prompt_metadata()
    
    # Step 5: Check voice mappings
    voice_mappings = check_voice_mappings()
    
    # Step 6: Test Whisper if available
    if metadata:
        first_prompt = list(metadata.keys())[0]
        test_whisper_transcription(first_prompt)
    
    # Step 7: Run comprehensive tests
    run_comprehensive_tests()
    
    # Final recommendations
    print_header("TROUBLESHOOTING RECOMMENDATIONS")
    
    print_section("Generated Test Files")
    print_info("Compare the following files to diagnose issues:")
    print("  • test_with_prompt_*.wav - Voice with audio prompt")
    print("  • test_baseline_aria.wav - Default voice for comparison")
    print("  • comparison_*.wav - Different parameter settings")
    
    print_section("Common Issues & Solutions")
    print("1. Audio prompt not loading:")
    print("   → Check file permissions in audio_prompts/ directory")
    print("   → Verify audio file format (WAV recommended)")
    print("   → Check server logs for file access errors")
    
    print("2. Voice sounds similar to default:")
    print("   → Audio prompt may be too short (<3s) or too long (>30s)")
    print("   → Try higher cfg_scale values (4.0-6.0)")
    print("   → Ensure audio prompt has clear, distinct characteristics")
    
    print("3. Transcription issues:")
    print("   → Install Whisper: pip install openai-whisper")
    print("   → Manually create .txt files with transcripts")
    print("   → Use .reference.txt for highest priority transcripts")
    
    print("4. GPU/Performance issues:")
    print("   → Check GPU memory usage in server status")
    print("   → Reduce batch size or use CPU mode if needed")
    print("   → Monitor server console for CUDA errors")
    
    print_section("Next Steps")
    print("1. Listen to generated test files")
    print("2. Check server console output for detailed logs")
    print("3. Review generation logs via /logs endpoint")
    print("4. Adjust audio prompt or parameters based on results")
    print("5. Use async mode for production to avoid timeouts")
    
    print_success("Debug session complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDebug session interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()