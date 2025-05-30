#!/usr/bin/env python3
"""
Audio prompt management tool
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

BASE_URL = "http://localhost:7860"

def check_audio_prompts_folder():
    """Check and organize audio prompts folder"""
    audio_dir = Path("audio_prompts")
    audio_dir.mkdir(exist_ok=True)
    
    print("📁 Checking audio_prompts folder...\n")
    
    audio_files = []
    transcript_files = []
    
    for file in audio_dir.iterdir():
        if file.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            audio_files.append(file)
        elif file.suffix.lower() in ['.txt']:
            transcript_files.append(file)
    
    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f"  🎵 {f.name}")
        
        # Check for transcripts
        ref_file = f.with_suffix('.reference.txt')
        txt_file = f.with_suffix('.txt')
        
        if ref_file.exists():
            print(f"     ✅ Has reference transcript")
        elif txt_file.exists():
            print(f"     ✅ Has transcript")
        else:
            print(f"     ⚠️  No transcript found")
    
    return audio_files

def transcribe_with_whisper(audio_file: Path) -> Optional[str]:
    """Transcribe audio file with Whisper"""
    if not WHISPER_AVAILABLE:
        print("❌ Whisper not installed. Install with: pip install openai-whisper")
        return None
    
    try:
        print(f"🎤 Transcribing {audio_file.name}...")
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_file))
        transcript = result["text"].strip()
        print(f"✅ Transcribed: \"{transcript}\"")
        return transcript
    except Exception as e:
        print(f"❌ Failed to transcribe: {e}")
        return None

def create_voice_from_prompt(prompt_id: str, voice_id: Optional[str] = None):
    """Create a voice configuration from an audio prompt"""
    if not voice_id:
        voice_id = prompt_id
    
    # Check if audio prompt exists on server
    response = requests.get(f"{BASE_URL}/audio_prompts")
    if response.status_code != 200:
        print("❌ Could not get audio prompts from server")
        return
    
    prompts = response.json()
    if prompt_id not in prompts:
        print(f"❌ Audio prompt '{prompt_id}' not found on server")
        return
    
    # Get transcript
    audio_file = Path(f"audio_prompts/{prompt_id}.wav")
    transcript = None
    
    # Check for existing transcript
    ref_file = audio_file.with_suffix('.reference.txt')
    txt_file = audio_file.with_suffix('.txt')
    
    if ref_file.exists():
        transcript = ref_file.read_text(encoding='utf-8').strip()
        print(f"✅ Using reference transcript")
    elif txt_file.exists():
        transcript = txt_file.read_text(encoding='utf-8').strip()
        print(f"✅ Using existing transcript")
    else:
        print("📝 No transcript found")
        use_whisper = input("Transcribe with Whisper? (y/n): ")
        if use_whisper.lower() == 'y':
            transcript = transcribe_with_whisper(audio_file)
            if transcript:
                # Save transcript
                txt_file.write_text(transcript, encoding='utf-8')
        else:
            transcript = input("Enter transcript manually: ").strip()
            if transcript:
                txt_file.write_text(transcript, encoding='utf-8')
    
    if not transcript:
        print("⚠️  No transcript provided, voice quality may be affected")
        transcript = ""
    
    # Create voice
    voice_config = {
        "voice_id": voice_id,
        "style": "natural",
        "primary_speaker": "S2",  # Default to feminine
        "audio_prompt": prompt_id,
        "audio_prompt_transcript": transcript
    }
    
    print(f"\n🔧 Creating voice '{voice_id}'...")
    response = requests.post(
        f"{BASE_URL}/voice_mappings",
        json=voice_config
    )
    
    if response.status_code == 200:
        print(f"✅ Voice '{voice_id}' created successfully")
        return True
    else:
        print(f"❌ Failed to create voice: {response.status_code}")
        return False

def test_voice_parameters(voice_id: str):
    """Test voice with different parameters"""
    test_text = "This is a test of the text to speech voice. It should sound natural and clear."
    
    # Parameter sets to test
    param_sets = [
        {"name": "default", "params": {}},
        {"name": "low_cfg", "params": {"cfg_scale": 1.5}},
        {"name": "very_low_cfg", "params": {"cfg_scale": 1.0}},
        {"name": "low_temp", "params": {"temperature": 0.8}},
        {"name": "both_low", "params": {"temperature": 0.8, "cfg_scale": 1.5}},
    ]
    
    print(f"\n🧪 Testing voice '{voice_id}' with different parameters...")
    
    for test in param_sets:
        print(f"\n  Testing {test['name']}...")
        
        request_data = {
            "text": test_text,
            "voice_id": voice_id,
            "speed": 1.0,
            **test['params']
        }
        
        response = requests.post(
            f"{BASE_URL}/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            filename = f"test_{voice_id}_{test['name']}.wav"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Check file size (empty audio issue)
            file_size = os.path.getsize(filename)
            if file_size < 10000:  # Less than 10KB is suspiciously small
                print(f"  ⚠️  WARNING: Output is very small ({file_size} bytes) - might be empty")
            else:
                print(f"  ✅ Saved: {filename} ({file_size:,} bytes)")
        else:
            print(f"  ❌ Failed: {response.status_code}")

def fix_empty_audio_issue():
    """Diagnose and fix empty audio generation"""
    print("\n🔧 Diagnosing empty audio issue...\n")
    
    print("Common causes of empty/20-second silent audio:")
    print("1. Temperature too low (< 0.7) can cause generation to fail")
    print("2. Conflicting parameters (low temp + high cfg_scale)")
    print("3. Audio prompt issues (corrupted/incompatible file)")
    print("4. Model timeout or memory issues")
    
    print("\n💡 Recommended fixes:")
    print("1. Use temperature >= 0.9 (default 1.0)")
    print("2. Use cfg_scale 1.5-2.0 with audio prompts")
    print("3. Don't combine very low temperature with audio prompts")
    print("4. Check server logs for generation errors")
    
    print("\n🧪 Safe parameter combinations:")
    safe_configs = [
        {"cfg_scale": 1.5, "temperature": 1.0},
        {"cfg_scale": 2.0, "temperature": 0.9},
        {"cfg_scale": 1.0, "temperature": 1.0},
    ]
    
    for i, config in enumerate(safe_configs):
        print(f"\nOption {i+1}:")
        print(f"  cfg_scale: {config['cfg_scale']}")
        print(f"  temperature: {config['temperature']}")

def main():
    print("🎙️ Audio Prompt Manager\n")
    
    while True:
        print("\nOptions:")
        print("1. Check audio prompts folder")
        print("2. Create voice from audio prompt")
        print("3. Test voice with different parameters")
        print("4. Fix empty audio issues")
        print("5. Install Whisper for transcription")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            check_audio_prompts_folder()
        
        elif choice == "2":
            files = check_audio_prompts_folder()
            if files:
                prompt_id = input("\nEnter prompt ID (filename without extension): ").strip()
                voice_id = input("Enter voice ID (or press Enter to use prompt ID): ").strip()
                create_voice_from_prompt(prompt_id, voice_id or None)
        
        elif choice == "3":
            voice_id = input("\nEnter voice ID to test: ").strip()
            test_voice_parameters(voice_id)
        
        elif choice == "4":
            fix_empty_audio_issue()
        
        elif choice == "5":
            print("\nTo install Whisper for automatic transcription:")
            print("pip install openai-whisper")
            print("\nNote: Whisper will download models on first use (~1.5GB)")
        
        elif choice == "0":
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    # Check server connection
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not responding")
            sys.exit(1)
    except:
        print("❌ Cannot connect to server at", BASE_URL)
        print("   Make sure server is running")
        sys.exit(1)
    
    main()