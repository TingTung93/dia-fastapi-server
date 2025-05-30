#!/usr/bin/env python3
"""
Complete integration patch for audio prompt discovery and Whisper transcription
"""

import os
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

def generate_integrated_code():
    """Generate the complete integrated server code with audio prompt discovery"""
    
    # Part 1: Additional imports
    additional_imports = '''
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

    # Part 2: Extended models
    extended_models = '''
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

    # Part 3: Extended ServerConfig
    serverconfig_extension = '''
# Add these fields to ServerConfig class:
    auto_discover_prompts: bool = True
    auto_transcribe: bool = True
    whisper_model_size: str = "base"  # tiny, base, small, medium, large
'''

    # Part 4: Global variables
    new_globals = '''
# Whisper model instance
WHISPER_MODEL: Optional[Any] = None
WHISPER_LOADING = False
WHISPER_LOCK = threading.Lock()

# Audio prompt metadata
AUDIO_PROMPT_METADATA: Dict[str, AudioPromptInfo] = {}
'''

    # Part 5: Discovery functions
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

    # Part 6: New API endpoints
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

    # Part 7: Modified startup
    modified_startup = '''
# Add this to the startup_event function after ensure_audio_prompt_dir():

    # Discover audio prompts on startup
    if SERVER_CONFIG.auto_discover_prompts:
        discovered = discover_audio_prompts()
        sync_audio_prompts_with_voices()
        
        # Load Whisper model in background if needed
        if SERVER_CONFIG.auto_transcribe and WHISPER_AVAILABLE and not WHISPER_MODEL:
            threading.Thread(target=load_whisper_model, daemon=True).start()
'''

    # Part 8: Modified list_audio_prompts endpoint
    modified_list_prompts = '''
# Replace the existing list_audio_prompts endpoint with:

@app.get("/audio_prompts")
async def list_audio_prompts():
    """List available audio prompts with metadata"""
    prompts = {}
    
    for prompt_id, file_path in AUDIO_PROMPTS.items():
        # Get metadata if available
        metadata = AUDIO_PROMPT_METADATA.get(prompt_id)
        
        if metadata:
            prompts[prompt_id] = {
                "file_path": file_path,
                "duration": metadata.duration,
                "sample_rate": metadata.sample_rate,
                "transcript": metadata.transcript,
                "transcript_source": metadata.transcript_source,
                "exists": os.path.exists(file_path)
            }
        else:
            # Fallback to basic info
            if os.path.exists(file_path):
                try:
                    audio_data, sr = sf.read(file_path)
                    prompts[prompt_id] = {
                        "file_path": file_path,
                        "duration": len(audio_data) / sr,
                        "sample_rate": sr,
                        "exists": True
                    }
                except:
                    prompts[prompt_id] = {
                        "file_path": file_path,
                        "exists": True,
                        "error": "Could not read audio file"
                    }
            else:
                prompts[prompt_id] = {
                    "file_path": file_path,
                    "exists": False
                }
    
    return prompts
'''

    return {
        "imports": additional_imports,
        "models": extended_models,
        "serverconfig": serverconfig_extension,
        "globals": new_globals,
        "functions": discovery_functions,
        "endpoints": new_endpoints,
        "startup": modified_startup,
        "modified_endpoints": modified_list_prompts
    }

def create_installation_script():
    """Create a script to install Whisper"""
    content = '''#!/usr/bin/env python3
"""Install Whisper for local transcription"""

import subprocess
import sys

def install_whisper():
    print("🔧 Installing OpenAI Whisper...")
    print("This will download and install Whisper and its dependencies.")
    print("Note: The first model download will be ~1-2GB\\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        print("\\n✅ Whisper installed successfully!")
        
        print("\\n📋 Available model sizes:")
        print("  - tiny    (39M parameters, ~1GB)    - fastest, lowest quality")
        print("  - base    (74M parameters, ~1.5GB)  - good balance")
        print("  - small   (244M parameters, ~2GB)   - better quality")
        print("  - medium  (769M parameters, ~5GB)   - high quality")
        print("  - large   (1550M parameters, ~10GB) - best quality")
        
        print("\\nThe 'base' model is recommended for most use cases.")
        print("Models will be downloaded on first use.")
        
    except subprocess.CalledProcessError:
        print("\\n❌ Failed to install Whisper")
        print("Try manually: pip install openai-whisper")
        return False
    
    return True

if __name__ == "__main__":
    install_whisper()
'''
    
    with open("install_whisper.py", "w") as f:
        f.write(content)
    print("✅ Created install_whisper.py")

def main():
    print("🔧 FastAPI Server Integration Tool\n")
    
    # Create backup
    backup = create_backup()
    
    # Generate integration code
    patches = generate_integrated_code()
    
    print("\n📋 Integration Instructions:\n")
    print("1. Install Whisper (for local transcription):")
    print("   python install_whisper.py\n")
    
    print("2. Add the following code sections to src/server.py:\n")
    
    for section, code in patches.items():
        print(f"\n{'='*60}")
        print(f"SECTION: {section.upper()}")
        print(f"{'='*60}")
        print(code)
    
    print("\n3. After adding all sections, restart the server\n")
    
    print("📝 Summary of new features:")
    print("✅ Auto-discovery of audio prompts on startup")
    print("✅ Support for .reference.txt and .txt transcripts")
    print("✅ Local Whisper transcription")
    print("✅ Metadata tracking with file hashing")
    print("✅ New API endpoints for discovery and transcription")
    print("✅ Automatic sync with voice mappings")
    
    # Create installation script
    create_installation_script()
    
    print("\n💡 Quick test after integration:")
    print("1. Place audio files in audio_prompts/")
    print("2. Start server")
    print("3. Check discovery: GET /audio_prompts/metadata")
    print("4. Force discovery: POST /audio_prompts/discover")
    print("5. Transcribe: POST /audio_prompts/{prompt_id}/transcribe")

if __name__ == "__main__":
    main()
'''