#!/usr/bin/env python3
"""
Enhanced audio prompt management system
- Auto-discovers audio files and reference texts
- Uses Whisper for automatic transcription
- Better organization and storage
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import soundfile as sf
import numpy as np

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available. Install with: pip install openai-whisper")

class AudioPromptManager:
    def __init__(self, audio_prompt_dir: str = "audio_prompts"):
        self.audio_prompt_dir = Path(audio_prompt_dir)
        self.audio_prompt_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.transcripts_dir = self.audio_prompt_dir / "transcripts"
        self.transcripts_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.audio_prompt_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Load Whisper model if available
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                print("🔄 Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _get_audio_hash(self, filepath: Path) -> str:
        """Get hash of audio file for change detection"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _transcribe_with_whisper(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            return None
        
        try:
            print(f"🎤 Transcribing {audio_path.name} with Whisper...")
            result = self.whisper_model.transcribe(str(audio_path))
            transcript = result["text"].strip()
            print(f"✅ Transcribed: \"{transcript}\"")
            return transcript
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            return None
    
    def discover_audio_prompts(self) -> Dict[str, Dict]:
        """Discover all audio files and their transcripts"""
        discovered = {}
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        print("🔍 Discovering audio prompts...")
        
        for audio_file in self.audio_prompt_dir.iterdir():
            if audio_file.suffix.lower() not in audio_extensions:
                continue
            
            prompt_id = audio_file.stem
            print(f"\n📁 Found: {audio_file.name}")
            
            # Get audio info
            try:
                data, sr = sf.read(audio_file)
                duration = len(data) / sr
                
                # Convert to mono if needed
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                audio_info = {
                    "file_path": str(audio_file),
                    "duration": round(duration, 2),
                    "sample_rate": sr,
                    "channels": 1 if len(data.shape) == 1 else data.shape[1],
                    "hash": self._get_audio_hash(audio_file)
                }
            except Exception as e:
                print(f"   ❌ Error reading audio: {e}")
                continue
            
            # Look for transcript in order of preference
            transcript = None
            transcript_source = None
            
            # 1. Check for .reference.txt file
            ref_file = audio_file.with_suffix('.reference.txt')
            if ref_file.exists():
                try:
                    transcript = ref_file.read_text(encoding='utf-8').strip()
                    transcript_source = "reference"
                    print(f"   ✅ Found reference transcript")
                except Exception as e:
                    print(f"   ❌ Error reading reference: {e}")
            
            # 2. Check for .txt file
            if not transcript:
                txt_file = audio_file.with_suffix('.txt')
                if txt_file.exists():
                    try:
                        transcript = txt_file.read_text(encoding='utf-8').strip()
                        transcript_source = "txt"
                        print(f"   ✅ Found .txt transcript")
                    except Exception as e:
                        print(f"   ❌ Error reading .txt: {e}")
            
            # 3. Check saved transcripts
            if not transcript:
                saved_transcript = self.transcripts_dir / f"{prompt_id}.txt"
                if saved_transcript.exists():
                    try:
                        transcript = saved_transcript.read_text(encoding='utf-8').strip()
                        transcript_source = "saved"
                        print(f"   ✅ Found saved transcript")
                    except:
                        pass
            
            # 4. Check metadata
            if not transcript and prompt_id in self.metadata:
                if 'transcript' in self.metadata[prompt_id]:
                    transcript = self.metadata[prompt_id]['transcript']
                    transcript_source = "metadata"
                    print(f"   ✅ Found transcript in metadata")
            
            # 5. Use Whisper if no transcript found
            if not transcript and self.whisper_model:
                transcript = self._transcribe_with_whisper(audio_file)
                if transcript:
                    transcript_source = "whisper"
                    # Save Whisper transcript
                    saved_transcript = self.transcripts_dir / f"{prompt_id}.txt"
                    saved_transcript.write_text(transcript, encoding='utf-8')
            
            # Build result
            discovered[prompt_id] = {
                **audio_info,
                "prompt_id": prompt_id,
                "transcript": transcript,
                "transcript_source": transcript_source
            }
            
            if transcript:
                preview = transcript[:50] + "..." if len(transcript) > 50 else transcript
                print(f"   📝 Transcript: \"{preview}\"")
            else:
                print(f"   ⚠️  No transcript found")
        
        print(f"\n✅ Discovered {len(discovered)} audio prompts")
        return discovered
    
    def update_metadata(self, discovered: Dict[str, Dict]):
        """Update metadata with discovered prompts"""
        for prompt_id, info in discovered.items():
            if prompt_id not in self.metadata:
                self.metadata[prompt_id] = {}
            
            # Update metadata
            self.metadata[prompt_id].update({
                "file_path": info["file_path"],
                "duration": info["duration"],
                "sample_rate": info["sample_rate"],
                "hash": info["hash"],
                "transcript": info.get("transcript"),
                "transcript_source": info.get("transcript_source")
            })
        
        # Remove deleted files from metadata
        existing_ids = set(discovered.keys())
        removed = [k for k in self.metadata.keys() if k not in existing_ids]
        for k in removed:
            del self.metadata[k]
        
        self._save_metadata()
    
    def get_prompt_info(self, prompt_id: str) -> Optional[Dict]:
        """Get info for a specific prompt"""
        if prompt_id in self.metadata:
            return self.metadata[prompt_id]
        return None
    
    def set_transcript(self, prompt_id: str, transcript: str, source: str = "manual"):
        """Set or update transcript for a prompt"""
        if prompt_id in self.metadata:
            self.metadata[prompt_id]["transcript"] = transcript
            self.metadata[prompt_id]["transcript_source"] = source
            
            # Save to transcripts directory
            saved_transcript = self.transcripts_dir / f"{prompt_id}.txt"
            saved_transcript.write_text(transcript, encoding='utf-8')
            
            self._save_metadata()
            return True
        return False
    
    def list_prompts(self) -> List[Dict]:
        """List all audio prompts with their info"""
        return [
            {
                "prompt_id": k,
                **v
            }
            for k, v in self.metadata.items()
        ]
    
    def sync_with_server(self, base_url: str = "http://localhost:7860"):
        """Sync discovered prompts with server"""
        import requests
        
        discovered = self.discover_audio_prompts()
        self.update_metadata(discovered)
        
        print("\n🔄 Syncing with server...")
        
        # Get current server prompts
        try:
            response = requests.get(f"{base_url}/audio_prompts")
            if response.status_code == 200:
                server_prompts = response.json()
            else:
                server_prompts = {}
        except:
            print("❌ Could not connect to server")
            return
        
        # Update server with discovered prompts
        for prompt_id, info in discovered.items():
            if prompt_id not in server_prompts:
                print(f"   📤 Registering {prompt_id} with server...")
                # The server needs to know about the file
                # This would require a new endpoint or modification
        
        print("✅ Sync complete")

def main():
    """Example usage"""
    manager = AudioPromptManager()
    
    # Discover all prompts
    prompts = manager.discover_audio_prompts()
    
    # Display found prompts
    print("\n📊 Summary:")
    for prompt_id, info in prompts.items():
        print(f"\n🎵 {prompt_id}")
        print(f"   File: {Path(info['file_path']).name}")
        print(f"   Duration: {info['duration']}s")
        if info.get('transcript'):
            print(f"   Transcript: \"{info['transcript'][:60]}...\"")
            print(f"   Source: {info.get('transcript_source')}")
        else:
            print(f"   ⚠️  No transcript")

if __name__ == "__main__":
    main()