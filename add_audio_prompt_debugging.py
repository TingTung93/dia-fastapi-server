#!/usr/bin/env python3
"""Add debugging to server.py for audio prompt issues"""

import os
import shutil
from datetime import datetime

def add_debugging():
    """Add debug logging to server.py"""
    
    # Backup original
    server_file = "src/server.py"
    backup_file = f"src/server.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not os.path.exists(server_file):
        print(f"❌ Server file not found: {server_file}")
        return
    
    print(f"📋 Creating backup: {backup_file}")
    shutil.copy2(server_file, backup_file)
    
    print("\n🔧 Adding debug code to server.py...")
    print("\nAdd this code in the generate_audio_from_text function:")
    print("\n" + "="*60)
    
    debug_code = '''
    # DEBUG: Audio prompt investigation
    if voice_config.get("audio_prompt"):
        console.print(f"[yellow]DEBUG: Audio prompt configured: {voice_config['audio_prompt']}[/yellow]")
        console.print(f"[yellow]DEBUG: Looking in AUDIO_PROMPTS dict...[/yellow]")
        console.print(f"[yellow]DEBUG: AUDIO_PROMPTS keys: {list(AUDIO_PROMPTS.keys())}[/yellow]")
        
        audio_prompt_path = AUDIO_PROMPTS.get(voice_config["audio_prompt"])
        console.print(f"[yellow]DEBUG: Retrieved path: {audio_prompt_path}[/yellow]")
        
        if audio_prompt_path:
            console.print(f"[yellow]DEBUG: Checking if path exists: {os.path.exists(audio_prompt_path)}[/yellow]")
            console.print(f"[yellow]DEBUG: Absolute path: {os.path.abspath(audio_prompt_path) if audio_prompt_path else 'None'}[/yellow]")
            
            if os.path.exists(audio_prompt_path):
                file_size = os.path.getsize(audio_prompt_path)
                console.print(f"[yellow]DEBUG: File size: {file_size} bytes[/yellow]")
                
                # Try to read the file to ensure it's valid
                try:
                    import soundfile as sf
                    data, sr = sf.read(audio_prompt_path)
                    console.print(f"[yellow]DEBUG: Audio file is valid: {len(data)} samples @ {sr}Hz[/yellow]")
                except Exception as e:
                    console.print(f"[red]DEBUG: Error reading audio file: {e}[/red]")
    
    # Right before model.generate() call:
    console.print(f"[yellow]DEBUG: Calling model.generate with:[/yellow]")
    console.print(f"[yellow]  - processed_text: {processed_text[:100]}...[/yellow]")
    console.print(f"[yellow]  - audio_prompt: {audio_prompt}[/yellow]")
    console.print(f"[yellow]  - audio_prompt type: {type(audio_prompt)}[/yellow]")
    console.print(f"[yellow]  - generation_params: {generation_params}[/yellow]")
'''
    
    print(debug_code)
    print("="*60)
    
    print("\n📍 Where to add this code:")
    print("1. First block: After line ~630 (where audio_prompt is set)")
    print("2. Second block: Before line ~720 (before model_instance.generate call)")
    
    print("\n💡 This will help identify:")
    print("• If the audio prompt path is being found correctly")
    print("• If the file exists and is readable")
    print("• What's being passed to the model")
    print("• Whether the model is receiving the audio prompt parameter")
    
    print("\n🔧 To apply debugging:")
    print("1. Edit src/server.py and add the debug code")
    print("2. Restart the server")
    print("3. Enable debug mode: curl -X PUT http://localhost:7860/config -H 'Content-Type: application/json' -d '{\"debug_mode\": true}'")
    print("4. Generate audio and watch the server console output")
    
    print(f"\n💾 Backup saved to: {backup_file}")
    print("   To restore: cp", backup_file, server_file)

if __name__ == "__main__":
    add_debugging()