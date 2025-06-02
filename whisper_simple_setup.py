#!/usr/bin/env python3
"""
Simple Whisper setup and transcription tool
This provides a working alternative while the server integration is stabilized
"""

import os
import sys
import argparse
from pathlib import Path

def check_whisper():
    """Check if Whisper is properly installed"""
    try:
        import whisper
        print("✅ Whisper is installed")
        return True
    except ImportError:
        print("❌ Whisper not installed")
        print("   Run: pip install git+https://github.com/openai/whisper.git")
        return False

def transcribe_audio_file(audio_path, model_size="base", output_file=None):
    """Transcribe a single audio file"""
    try:
        import whisper
        
        print(f"🔄 Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        
        print(f"🎤 Transcribing {audio_path}...")
        result = model.transcribe(str(audio_path))
        transcript = result["text"].strip()
        
        print(f"✅ Transcription complete!")
        print(f"📝 Transcript: {transcript}")
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(transcript, encoding='utf-8')
            print(f"💾 Saved to: {output_path}")
        
        return transcript
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def batch_transcribe_directory(directory="audio_prompts", model_size="base"):
    """Transcribe all audio files in a directory"""
    audio_dir = Path(directory)
    if not audio_dir.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = [f for f in audio_dir.iterdir() 
                   if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print(f"❌ No audio files found in {directory}")
        return
    
    print(f"🔍 Found {len(audio_files)} audio files")
    
    try:
        import whisper
        print(f"🔄 Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        
        for audio_file in audio_files:
            print(f"\n🎤 Processing: {audio_file.name}")
            
            # Check if transcript already exists
            transcript_file = audio_file.with_suffix('.txt')
            reference_file = audio_file.with_suffix('.reference.txt')
            
            if reference_file.exists():
                print(f"   ⏭️  Skipping (has .reference.txt)")
                continue
            elif transcript_file.exists():
                print(f"   ⏭️  Skipping (has .txt)")
                continue
            
            try:
                result = model.transcribe(str(audio_file))
                transcript = result["text"].strip()
                
                # Save transcript
                transcript_file.write_text(transcript, encoding='utf-8')
                
                print(f"   ✅ Transcribed: {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
                print(f"   💾 Saved: {transcript_file.name}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print(f"\n🎉 Batch transcription complete!")
        
    except Exception as e:
        print(f"❌ Batch transcription failed: {e}")

def update_requirements():
    """Update requirements.txt to use the working Whisper installation"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return
    
    # Read current requirements
    content = requirements_file.read_text()
    
    # Replace the problematic openai-whisper line
    if "openai-whisper>=20231117" in content:
        new_content = content.replace(
            "openai-whisper>=20231117",
            "# Install Whisper from GitHub for Python 3.13 compatibility\n# Run: pip install git+https://github.com/openai/whisper.git"
        )
        
        requirements_file.write_text(new_content)
        print("✅ Updated requirements.txt with working Whisper installation note")
    else:
        print("ℹ️  requirements.txt doesn't need updating")

def main():
    parser = argparse.ArgumentParser(description="Simple Whisper transcription tool")
    parser.add_argument("command", choices=["check", "transcribe", "batch", "update-requirements"], 
                       help="Command to run")
    parser.add_argument("--file", help="Audio file to transcribe")
    parser.add_argument("--output", help="Output transcript file")
    parser.add_argument("--directory", default="audio_prompts", help="Directory for batch processing")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    
    args = parser.parse_args()
    
    if args.command == "check":
        check_whisper()
        
    elif args.command == "transcribe":
        if not args.file:
            print("❌ --file required for transcribe command")
            return
        if not check_whisper():
            return
        transcribe_audio_file(args.file, args.model, args.output)
        
    elif args.command == "batch":
        if not check_whisper():
            return
        batch_transcribe_directory(args.directory, args.model)
        
    elif args.command == "update-requirements":
        update_requirements()

if __name__ == "__main__":
    # If run without arguments, show help
    if len(sys.argv) == 1:
        print("🎙️ Simple Whisper Setup & Transcription Tool")
        print("\nUsage examples:")
        print("  python whisper_simple_setup.py check")
        print("  python whisper_simple_setup.py transcribe --file audio.wav")
        print("  python whisper_simple_setup.py batch --directory audio_prompts")
        print("  python whisper_simple_setup.py update-requirements")
        print("\nModel sizes: tiny, base (default), small, medium, large")
        print("\n🔧 Installation:")
        print("  pip install git+https://github.com/openai/whisper.git")
        sys.exit(0)
    
    main() 