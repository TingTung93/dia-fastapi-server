#!/usr/bin/env python3
"""
Apply audio prompt discovery integration to server.py
"""

import os
import re
import shutil
from datetime import datetime

def create_backup():
    """Create backup of server.py"""
    src = "src/server.py"
    if os.path.exists(src):
        backup = f"src/server.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(src, backup)
        print(f"✅ Backup created: {backup}")
        return backup
    return None

def apply_integration():
    """Apply the integration patches to server.py"""
    
    server_file = "src/server.py"
    if not os.path.exists(server_file):
        print(f"❌ {server_file} not found!")
        return False
    
    # Read current server.py
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already integrated
    if "whisper" in content and "discover_audio_prompts" in content:
        print("⚠️  Server appears to already have integration!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create backup
    backup = create_backup()
    if not backup:
        print("❌ Could not create backup!")
        return False
    
    print("\n🔧 Applying integration patches...\n")
    
    # 1. Add imports after existing imports
    import_marker = "from dia import Dia"
    import_addition = '''

# Additional imports for audio prompt discovery and Whisper
from pathlib import Path
import hashlib
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
    console.print("[yellow]⚠️  Whisper not available. Install with: pip install openai-whisper[/yellow]")
'''
    
    if import_marker in content and "import whisper" not in content:
        content = content.replace(import_marker, import_marker + import_addition)
        print("✅ Added Whisper imports")
    
    # 2. Add extended models after ServerConfig
    model_marker = "class ServerConfig(BaseModel):"
    model_addition = '''

# Extended models for audio prompt discovery
class AudioPromptInfo(BaseModel):
    prompt_id: str
    file_path: str
    duration: float
    sample_rate: int
    transcript: Optional[str] = None
    transcript_source: Optional[str] = None
    hash: Optional[str] = None
    discovered_at: Optional[datetime] = None

class AudioPromptDiscoveryRequest(BaseModel):
    force_retranscribe: bool = Field(default=False, description="Force re-transcription with Whisper")
    
class TranscriptUpdateRequest(BaseModel):
    transcript: str = Field(..., description="The transcript text")
    prompt_id: str = Field(..., description="The audio prompt ID")
'''
    
    # Find end of ServerConfig class
    if model_marker in content and "AudioPromptInfo" not in content:
        # Find the end of ServerConfig
        match = re.search(r'class ServerConfig\(BaseModel\):.*?\n\n', content, re.DOTALL)
        if match:
            end_pos = match.end()
            content = content[:end_pos] + model_addition + "\n" + content[end_pos:]
            print("✅ Added extended models")
    
    # 3. Update ServerConfig fields
    serverconfig_pattern = r'(class ServerConfig\(BaseModel\):.*?output_retention_hours: int = 24)'
    serverconfig_addition = '''
    auto_discover_prompts: bool = True
    auto_transcribe: bool = True
    whisper_model_size: str = "base"  # tiny, base, small, medium, large'''
    
    if "auto_discover_prompts" not in content:
        content = re.sub(
            serverconfig_pattern,
            r'\1' + serverconfig_addition,
            content,
            flags=re.DOTALL
        )
        print("✅ Extended ServerConfig")
    
    # 4. Add global variables after AUDIO_PROMPT_DIR
    globals_marker = 'AUDIO_PROMPT_DIR = "audio_prompts"'
    globals_addition = '''

# Whisper model instance
WHISPER_MODEL: Optional[Any] = None
WHISPER_LOADING = False
WHISPER_LOCK = threading.Lock()

# Audio prompt metadata
AUDIO_PROMPT_METADATA: Dict[str, AudioPromptInfo] = {}
'''
    
    if globals_marker in content and "WHISPER_MODEL" not in content:
        content = content.replace(globals_marker, globals_marker + globals_addition)
        print("✅ Added global variables")
    
    # 5. Add discovery functions before load_model
    functions_marker = "def load_model():"
    discovery_functions = '''
def get_audio_file_hash(filepath: Path) -> str:
    """Get hash of audio file for change detection"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_whisper_model():
    """Load Whisper model if not already loaded"""
    global WHISPER_MODEL, WHISPER_LOADING
    
    if not WHISPER_AVAILABLE or not SERVER_CONFIG.auto_transcribe:
        return None
    
    with WHISPER_LOCK:
        if WHISPER_MODEL is not None or WHISPER_LOADING:
            return WHISPER_MODEL
        
        WHISPER_LOADING = True
        try:
            console.print(f"[yellow]🔄 Loading Whisper model ({SERVER_CONFIG.whisper_model_size})...[/yellow]")
            WHISPER_MODEL = whisper.load_model(SERVER_CONFIG.whisper_model_size)
            console.print("[green]✅ Whisper model loaded successfully[/green]")
            return WHISPER_MODEL
        except Exception as e:
            console.print(f"[red]❌ Failed to load Whisper model: {e}[/red]")
            return None
        finally:
            WHISPER_LOADING = False

def transcribe_with_whisper(audio_path: Path) -> Optional[str]:
    """Transcribe audio using Whisper"""
    model = load_whisper_model()
    if not model:
        return None
    
    try:
        console.print(f"[yellow]🎤 Transcribing {audio_path.name}...[/yellow]")
        result = model.transcribe(str(audio_path), language="en")
        transcript = result["text"].strip()
        console.print(f"[green]✅ Transcribed: \\"{transcript[:50]}...\\\"[/green]" if len(transcript) > 50 else f"[green]✅ Transcribed: \\"{transcript}\\\"[/green]")
        return transcript
    except Exception as e:
        console.print(f"[red]❌ Transcription failed: {e}[/red]")
        return None

def discover_audio_prompts(force_retranscribe: bool = False) -> Dict[str, AudioPromptInfo]:
    """Automatically discover audio prompts and their transcripts"""
    global AUDIO_PROMPTS, AUDIO_PROMPT_METADATA
    
    audio_prompt_dir = Path(AUDIO_PROMPT_DIR)
    if not audio_prompt_dir.exists():
        return {}
    
    discovered = {}
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    console.print("[cyan]🔍 Discovering audio prompts...[/cyan]")
    
    for audio_file in audio_prompt_dir.iterdir():
        if audio_file.suffix.lower() not in audio_extensions:
            continue
        
        prompt_id = audio_file.stem
        file_path = str(audio_file.absolute())
        
        # Get file info
        try:
            audio_data, sr = sf.read(file_path)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            duration = len(audio_data) / sr
            file_hash = get_audio_file_hash(audio_file)
        except Exception as e:
            console.print(f"[red]❌ Error reading {audio_file.name}: {e}[/red]")
            continue
        
        # Check if file has changed
        existing_metadata = AUDIO_PROMPT_METADATA.get(prompt_id)
        file_changed = not existing_metadata or existing_metadata.hash != file_hash
        
        # Look for transcript
        transcript = None
        transcript_source = None
        
        # 1. Check for .reference.txt file (highest priority)
        ref_file = audio_file.with_suffix('.reference.txt')
        if ref_file.exists() and not force_retranscribe:
            try:
                transcript = ref_file.read_text(encoding='utf-8').strip()
                transcript_source = "reference"
            except Exception as e:
                console.print(f"[yellow]⚠️  Error reading {ref_file.name}: {e}[/yellow]")
        
        # 2. Check for .txt file
        if not transcript and not force_retranscribe:
            txt_file = audio_file.with_suffix('.txt')
            if txt_file.exists():
                try:
                    transcript = txt_file.read_text(encoding='utf-8').strip()
                    transcript_source = "txt"
                except:
                    pass
        
        # 3. Check existing metadata (if file hasn't changed)
        if not transcript and not file_changed and existing_metadata and existing_metadata.transcript:
            transcript = existing_metadata.transcript
            transcript_source = existing_metadata.transcript_source
        
        # 4. Use Whisper if no transcript found or forced
        if (not transcript or force_retranscribe) and SERVER_CONFIG.auto_transcribe:
            whisper_transcript = transcribe_with_whisper(audio_file)
            if whisper_transcript:
                transcript = whisper_transcript
                transcript_source = "whisper"
                
                # Save Whisper transcript
                transcript_file = audio_file.with_suffix('.txt')
                try:
                    transcript_file.write_text(transcript, encoding='utf-8')
                    console.print(f"[green]💾 Saved transcript to {transcript_file.name}[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not save transcript: {e}[/yellow]")
        
        # Create metadata
        prompt_info = AudioPromptInfo(
            prompt_id=prompt_id,
            file_path=file_path,
            duration=round(duration, 2),
            sample_rate=sr,
            transcript=transcript,
            transcript_source=transcript_source,
            hash=file_hash,
            discovered_at=datetime.now()
        )
        
        # Register audio prompt
        AUDIO_PROMPTS[prompt_id] = file_path
        AUDIO_PROMPT_METADATA[prompt_id] = prompt_info
        discovered[prompt_id] = prompt_info
        
        # Log discovery
        console.print(f"[green]✅ {prompt_id}[/green]")
        if transcript:
            preview = transcript[:60] + "..." if len(transcript) > 60 else transcript
            console.print(f"   [dim]Transcript ({transcript_source}): {preview}[/dim]")
        else:
            console.print(f"   [yellow]⚠️  No transcript[/yellow]")
        
        # Update voice mappings with discovered transcript
        for voice_id, voice_config in VOICE_MAPPING.items():
            if voice_config.get("audio_prompt") == prompt_id and transcript:
                if not voice_config.get("audio_prompt_transcript") or file_changed:
                    voice_config["audio_prompt_transcript"] = transcript
                    console.print(f"   [dim]Updated voice '{voice_id}' transcript[/dim]")
    
    console.print(f"[bold green]✅ Discovered {len(discovered)} audio prompts[/bold green]")
    return discovered

def sync_audio_prompts_with_voices():
    """Sync discovered audio prompts with voice mappings"""
    updated_count = 0
    
    for voice_id, voice_config in VOICE_MAPPING.items():
        audio_prompt_id = voice_config.get("audio_prompt")
        if audio_prompt_id and audio_prompt_id in AUDIO_PROMPT_METADATA:
            metadata = AUDIO_PROMPT_METADATA[audio_prompt_id]
            if metadata.transcript and not voice_config.get("audio_prompt_transcript"):
                voice_config["audio_prompt_transcript"] = metadata.transcript
                updated_count += 1
                console.print(f"[dim]Synced transcript for voice '{voice_id}'[/dim]")
    
    if updated_count > 0:
        console.print(f"[green]✅ Updated {updated_count} voice transcripts[/green]")


'''
    
    if functions_marker in content and "discover_audio_prompts" not in content:
        content = content.replace(functions_marker, discovery_functions + functions_marker)
        print("✅ Added discovery functions")
    
    # 6. Update startup event
    startup_marker = "ensure_audio_prompt_dir()"
    startup_addition = '''
    
    # Discover audio prompts on startup
    if SERVER_CONFIG.auto_discover_prompts:
        discovered = discover_audio_prompts()
        sync_audio_prompts_with_voices()
        
        # Load Whisper model in background if needed
        if SERVER_CONFIG.auto_transcribe and WHISPER_AVAILABLE and not WHISPER_MODEL:
            threading.Thread(target=load_whisper_model, daemon=True).start()
'''
    
    if startup_marker in content and "discover_audio_prompts()" not in content:
        content = content.replace(startup_marker, startup_marker + startup_addition)
        print("✅ Updated startup event")
    
    # 7. Add new endpoints before the last endpoint (if __name__)
    endpoints_marker = 'if __name__ == "__main__":'
    new_endpoints = '''

# Audio Prompt Discovery Endpoints

@app.post("/audio_prompts/discover")
async def discover_prompts(request: AudioPromptDiscoveryRequest = AudioPromptDiscoveryRequest()):
    """Manually trigger audio prompt discovery"""
    discovered = discover_audio_prompts(force_retranscribe=request.force_retranscribe)
    sync_audio_prompts_with_voices()
    
    return {
        "message": f"Discovered {len(discovered)} audio prompts",
        "total_prompts": len(AUDIO_PROMPTS),
        "discovered": [info.dict() for info in discovered.values()]
    }

@app.get("/audio_prompts/metadata")
async def get_audio_prompt_metadata():
    """Get detailed metadata for all audio prompts"""
    return {
        prompt_id: info.dict() 
        for prompt_id, info in AUDIO_PROMPT_METADATA.items()
    }

@app.get("/audio_prompts/metadata/{prompt_id}")
async def get_prompt_metadata(prompt_id: str):
    """Get metadata for a specific audio prompt"""
    if prompt_id not in AUDIO_PROMPT_METADATA:
        raise HTTPException(status_code=404, detail=f"Audio prompt '{prompt_id}' not found")
    
    return AUDIO_PROMPT_METADATA[prompt_id]

@app.put("/audio_prompts/{prompt_id}/transcript")
async def update_prompt_transcript(prompt_id: str, request: TranscriptUpdateRequest):
    """Update transcript for an audio prompt"""
    if prompt_id not in AUDIO_PROMPTS:
        raise HTTPException(status_code=404, detail=f"Audio prompt '{prompt_id}' not found")
    
    # Update metadata
    if prompt_id in AUDIO_PROMPT_METADATA:
        AUDIO_PROMPT_METADATA[prompt_id].transcript = request.transcript
        AUDIO_PROMPT_METADATA[prompt_id].transcript_source = "manual"
    
    # Save to file
    audio_file = Path(AUDIO_PROMPTS[prompt_id])
    transcript_file = audio_file.with_suffix('.txt')
    try:
        transcript_file.write_text(request.transcript, encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save transcript: {e}")
    
    # Update voice mappings
    updated_voices = []
    for voice_id, voice_config in VOICE_MAPPING.items():
        if voice_config.get("audio_prompt") == prompt_id:
            voice_config["audio_prompt_transcript"] = request.transcript
            updated_voices.append(voice_id)
    
    return {
        "message": "Transcript updated successfully",
        "prompt_id": prompt_id,
        "updated_voices": updated_voices
    }

@app.post("/audio_prompts/{prompt_id}/transcribe")
async def transcribe_audio_prompt(prompt_id: str):
    """Transcribe an audio prompt using Whisper"""
    if prompt_id not in AUDIO_PROMPTS:
        raise HTTPException(status_code=404, detail=f"Audio prompt '{prompt_id}' not found")
    
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Whisper is not installed")
    
    audio_path = Path(AUDIO_PROMPTS[prompt_id])
    transcript = transcribe_with_whisper(audio_path)
    
    if not transcript:
        raise HTTPException(status_code=500, detail="Transcription failed")
    
    # Update metadata
    if prompt_id in AUDIO_PROMPT_METADATA:
        AUDIO_PROMPT_METADATA[prompt_id].transcript = transcript
        AUDIO_PROMPT_METADATA[prompt_id].transcript_source = "whisper"
    
    # Save transcript
    transcript_file = audio_path.with_suffix('.txt')
    transcript_file.write_text(transcript, encoding='utf-8')
    
    return {
        "prompt_id": prompt_id,
        "transcript": transcript,
        "saved_to": str(transcript_file)
    }

@app.get("/whisper/status")
async def get_whisper_status():
    """Get Whisper model status"""
    return {
        "available": WHISPER_AVAILABLE,
        "model_loaded": WHISPER_MODEL is not None,
        "model_size": SERVER_CONFIG.whisper_model_size if WHISPER_AVAILABLE else None,
        "auto_transcribe": SERVER_CONFIG.auto_transcribe
    }

@app.post("/whisper/load")
async def load_whisper():
    """Manually load Whisper model"""
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Whisper is not installed")
    
    model = load_whisper_model()
    if model:
        return {"message": "Whisper model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load Whisper model")


'''
    
    if endpoints_marker in content and "/audio_prompts/discover" not in content:
        content = content.replace(endpoints_marker, new_endpoints + endpoints_marker)
        print("✅ Added new API endpoints")
    
    # Write updated content
    try:
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("\n✅ Successfully integrated audio prompt discovery!")
        print(f"   Backup saved as: {backup}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to write changes: {e}")
        print(f"   Restoring from backup: {backup}")
        shutil.copy2(backup, server_file)
        return False

def main():
    print("🔧 Audio Prompt Discovery Integration Tool\n")
    
    # Check server file exists
    if not os.path.exists("src/server.py"):
        print("❌ src/server.py not found!")
        print("   Make sure you're in the project root directory")
        return
    
    print("This will integrate the following features:")
    print("✅ Auto-discovery of audio prompts")
    print("✅ Support for .reference.txt and .txt transcripts")
    print("✅ Local Whisper transcription")
    print("✅ New API endpoints for management")
    print("✅ Automatic sync with voice mappings")
    
    response = input("\nProceed with integration? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Apply integration
    if apply_integration():
        print("\n🎉 Integration complete!")
        print("\n📝 Next steps:")
        print("1. Install Whisper: pip install openai-whisper")
        print("2. Restart the server")
        print("3. Place audio files in audio_prompts/")
        print("4. Server will auto-discover on startup")
        print("\n📋 New API endpoints:")
        print("   POST /audio_prompts/discover - Force discovery")
        print("   GET  /audio_prompts/metadata - Get all metadata")
        print("   POST /audio_prompts/{id}/transcribe - Transcribe with Whisper")
        print("   PUT  /audio_prompts/{id}/transcript - Update transcript")
        print("   GET  /whisper/status - Check Whisper status")
    else:
        print("\n❌ Integration failed!")

if __name__ == "__main__":
    main()