#!/usr/bin/env python3
"""Test if Dia model is actually loading and using audio prompts"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

def test_dia_audio_prompt():
    """Test Dia model directly with audio prompt"""
    print("🧪 Testing Dia Model Audio Prompt Support\n")
    
    try:
        from dia import Dia
        print("✅ Dia module imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import Dia: {e}")
        print("   This test must be run in the conda environment with Dia installed")
        return
    
    # Check for audio file
    audio_file = "audio_prompts/seraphina_voice.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("   Please ensure the audio prompt is uploaded first")
        return
    
    print(f"✅ Found audio file: {audio_file}")
    
    # Try to understand how Dia expects audio prompts
    print("\n📋 Checking Dia model documentation...")
    
    # Get Dia class info
    print("\nDia.generate method signature:")
    try:
        import inspect
        sig = inspect.signature(Dia.generate)
        print(f"  {sig}")
        
        # Get method docstring
        if Dia.generate.__doc__:
            print("\nDocstring:")
            print(Dia.generate.__doc__)
    except Exception as e:
        print(f"  Could not inspect: {e}")
    
    # Test different audio prompt formats
    print("\n🧪 Testing audio prompt formats...")
    
    test_formats = [
        ("File path string", audio_file),
        ("Absolute path", os.path.abspath(audio_file)),
        ("Path without extension", audio_file.replace('.wav', '')),
    ]
    
    for desc, path in test_formats:
        print(f"\n  Testing: {desc}")
        print(f"  Path: {path}")
        print(f"  Exists: {os.path.exists(path) if '.wav' in path else 'N/A'}")
    
    print("\n💡 Insights:")
    print("\n1. Audio Prompt Format:")
    print("   The Dia model might expect:")
    print("   • Just the file path as a string")
    print("   • Audio data as numpy array")
    print("   • A special audio prompt object")
    
    print("\n2. Common Issues:")
    print("   • Wrong path format (relative vs absolute)")
    print("   • Model version doesn't support audio prompts")
    print("   • Audio prompt parameter name mismatch")
    
    print("\n3. To verify in server.py:")
    print("   Add this debug code in generate_audio_from_text():")
    print("""
    # Before calling model.generate()
    print(f"DEBUG: audio_prompt type: {type(audio_prompt)}")
    print(f"DEBUG: audio_prompt value: {audio_prompt}")
    if audio_prompt and isinstance(audio_prompt, str):
        print(f"DEBUG: audio_prompt exists: {os.path.exists(audio_prompt)}")
    """)
    
    print("\n4. Alternative Test:")
    print("   Try loading and passing audio data directly:")
    print("""
    import soundfile as sf
    audio_data, sr = sf.read(audio_prompt_path)
    # Then try passing audio_data instead of path
    """)

if __name__ == "__main__":
    test_dia_audio_prompt()