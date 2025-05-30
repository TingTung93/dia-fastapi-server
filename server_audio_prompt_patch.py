#!/usr/bin/env python3
"""
Patch for server.py to add automatic audio prompt discovery
"""

print("""
🔧 Server Audio Prompt Discovery Patch

Add this code to server.py to enable automatic discovery of audio prompts:

==== 1. Add imports (after existing imports) ====

from pathlib import Path
try:
    import whisper
    WHISPER_MODEL = None
except ImportError:
    WHISPER_MODEL = None
    whisper = None

==== 2. Add discovery function (after model loading functions) ====

def discover_audio_prompts():
    \"\"\"Automatically discover audio prompts and their transcripts\"\"\"
    global AUDIO_PROMPTS, WHISPER_MODEL
    
    audio_prompt_dir = Path(AUDIO_PROMPT_DIR)
    if not audio_prompt_dir.exists():
        return
    
    console.print("[cyan]🔍 Discovering audio prompts...[/cyan]")
    
    # Load Whisper model if available and not loaded
    if whisper and WHISPER_MODEL is None and SERVER_CONFIG.auto_transcribe:
        try:
            console.print("[yellow]Loading Whisper model for transcription...[/yellow]")
            WHISPER_MODEL = whisper.load_model("base")
            console.print("[green]✅ Whisper model loaded[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load Whisper: {e}[/red]")
    
    discovered_count = 0
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    for audio_file in audio_prompt_dir.iterdir():
        if audio_file.suffix.lower() not in audio_extensions:
            continue
        
        prompt_id = audio_file.stem
        file_path = str(audio_file.absolute())
        
        # Skip if already loaded
        if prompt_id in AUDIO_PROMPTS and AUDIO_PROMPTS[prompt_id] == file_path:
            continue
        
        # Look for transcript
        transcript = None
        transcript_source = None
        
        # 1. Check for .reference.txt
        ref_file = audio_file.with_suffix('.reference.txt')
        if ref_file.exists():
            try:
                transcript = ref_file.read_text(encoding='utf-8').strip()
                transcript_source = "reference"
            except:
                pass
        
        # 2. Check for .txt
        if not transcript:
            txt_file = audio_file.with_suffix('.txt')
            if txt_file.exists():
                try:
                    transcript = txt_file.read_text(encoding='utf-8').strip()
                    transcript_source = "txt"
                except:
                    pass
        
        # 3. Use Whisper if enabled
        if not transcript and WHISPER_MODEL and SERVER_CONFIG.auto_transcribe:
            try:
                console.print(f"[yellow]Transcribing {audio_file.name}...[/yellow]")
                result = WHISPER_MODEL.transcribe(str(audio_file))
                transcript = result["text"].strip()
                transcript_source = "whisper"
                
                # Save transcript
                transcript_file = audio_file.with_suffix('.txt')
                transcript_file.write_text(transcript, encoding='utf-8')
                console.print(f"[green]✅ Transcribed and saved[/green]")
            except Exception as e:
                console.print(f"[red]❌ Transcription failed: {e}[/red]")
        
        # Register audio prompt
        AUDIO_PROMPTS[prompt_id] = file_path
        discovered_count += 1
        
        # Log discovery
        console.print(f"[green]✅ Discovered: {prompt_id}[/green]")
        if transcript:
            preview = transcript[:50] + "..." if len(transcript) > 50 else transcript
            console.print(f"   [dim]Transcript ({transcript_source}): {preview}[/dim]")
        
        # Update voice mappings if they use this prompt
        for voice_id, voice_config in VOICE_MAPPING.items():
            if voice_config.get("audio_prompt") == prompt_id and transcript:
                if not voice_config.get("audio_prompt_transcript"):
                    voice_config["audio_prompt_transcript"] = transcript
                    console.print(f"   [dim]Updated voice '{voice_id}' with transcript[/dim]")
    
    if discovered_count > 0:
        console.print(f"[bold green]✅ Discovered {discovered_count} new audio prompts[/bold green]")
    
    return discovered_count

==== 3. Update ServerConfig class (add to existing class) ====

class ServerConfig(BaseModel):
    debug_mode: bool = False
    save_outputs: bool = False
    show_prompts: bool = False
    output_retention_hours: int = 24
    auto_transcribe: bool = True  # Add this line

==== 4. Add discovery endpoint (add with other endpoints) ====

@app.post("/audio_prompts/discover")
async def discover_prompts():
    \"\"\"Manually trigger audio prompt discovery\"\"\"
    count = discover_audio_prompts()
    return {
        "message": f"Discovered {count} new audio prompts",
        "total_prompts": len(AUDIO_PROMPTS),
        "prompts": list(AUDIO_PROMPTS.keys())
    }

@app.get("/audio_prompts/details")
async def get_audio_prompt_details():
    \"\"\"Get detailed info about all audio prompts\"\"\"
    details = {}
    
    for prompt_id, file_path in AUDIO_PROMPTS.items():
        if os.path.exists(file_path):
            try:
                # Get audio info
                audio_data, sr = sf.read(file_path)
                duration = len(audio_data) / sr
                
                # Look for transcript
                audio_file = Path(file_path)
                transcript = None
                transcript_source = None
                
                # Check for transcript files
                for suffix, source in [('.reference.txt', 'reference'), ('.txt', 'txt')]:
                    transcript_file = audio_file.with_suffix(suffix)
                    if transcript_file.exists():
                        try:
                            transcript = transcript_file.read_text(encoding='utf-8').strip()
                            transcript_source = source
                            break
                        except:
                            pass
                
                # Check voice mappings for transcript
                if not transcript:
                    for voice_config in VOICE_MAPPING.values():
                        if voice_config.get("audio_prompt") == prompt_id:
                            transcript = voice_config.get("audio_prompt_transcript")
                            if transcript:
                                transcript_source = "voice_mapping"
                                break
                
                details[prompt_id] = {
                    "file_path": file_path,
                    "duration": round(duration, 2),
                    "sample_rate": sr,
                    "transcript": transcript,
                    "transcript_source": transcript_source,
                    "exists": True
                }
            except Exception as e:
                details[prompt_id] = {
                    "file_path": file_path,
                    "exists": True,
                    "error": str(e)
                }
        else:
            details[prompt_id] = {
                "file_path": file_path,
                "exists": False
            }
    
    return details

==== 5. Update startup event (modify existing startup_event) ====

@app.on_event("startup")
async def startup_event():
    \"\"\"Load model on startup\"\"\"
    load_model()
    initialize_worker_pool()
    
    # Ensure audio prompt directory exists
    ensure_audio_prompt_dir()
    
    # Discover audio prompts
    discover_audio_prompts()
    
    # Continue with existing startup code...

==== 6. Add install command for Whisper (optional) ====

To enable automatic transcription, install Whisper:
pip install openai-whisper

""")

print("""
📝 To apply this patch:

1. Open src/server.py
2. Add the code sections shown above in the appropriate places
3. Restart the server

Features added:
✅ Automatic discovery of audio files in audio_prompts/
✅ Support for .reference.txt and .txt transcript files
✅ Automatic transcription with Whisper (if installed)
✅ New endpoints for discovery and details
✅ Updates voice mappings with discovered transcripts

Usage:
- Place audio files in audio_prompts/
- Add transcripts as filename.reference.txt or filename.txt
- Server will auto-discover on startup
- Or trigger manually: POST /audio_prompts/discover
""")

if __name__ == "__main__":
    print("\n💡 This is a guide for manual patching.")
    print("   Copy the code sections above into server.py")
    print("   Or use the new audio_prompt_manager.py module")