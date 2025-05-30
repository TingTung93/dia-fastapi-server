#!/usr/bin/env python3
"""Direct GPU test of Dia model with audio prompts"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import time

# Add src to path
sys.path.insert(0, 'src')

def test_dia_gpu():
    """Test Dia model directly on GPU with audio prompts"""
    print("🚀 GPU-Accelerated Dia Model Test\n")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPU.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        from dia import Dia
        print("✅ Dia module imported")
    except ImportError as e:
        print(f"❌ Cannot import Dia: {e}")
        print("   Run this in the conda environment: conda activate dia")
        return
    
    # Load model on GPU
    print("\n🔄 Loading Dia model on GPU...")
    device = torch.device("cuda:0")
    
    # Determine compute dtype
    if torch.cuda.is_bf16_supported():
        compute_dtype = "bfloat16"
        print("✅ Using BFloat16 precision (optimal for modern GPUs)")
    else:
        compute_dtype = "float16"
        print("✅ Using Float16 precision")
    
    start_time = time.time()
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B",
        compute_dtype=compute_dtype,
        device=device
    )
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Check model location
    print(f"\n📍 Model device: {next(model.parameters()).device}")
    
    # Test configurations
    test_configs = [
        {
            "name": "baseline_s1",
            "text": "[S1] Hello, this is a test of the voice synthesis system. [S1]",
            "audio_prompt": None
        },
        {
            "name": "baseline_s2", 
            "text": "[S2] Hello, this is a test of the voice synthesis system. [S2]",
            "audio_prompt": None
        }
    ]
    
    # Add audio prompt tests if file exists
    audio_file = "audio_prompts/seraphina_voice.wav"
    if os.path.exists(audio_file):
        print(f"\n✅ Found audio prompt: {audio_file}")
        
        # Analyze audio file
        data, sr = sf.read(audio_file)
        print(f"   Duration: {len(data)/sr:.2f}s, Sample rate: {sr}Hz")
        
        test_configs.extend([
            {
                "name": "with_prompt_s1",
                "text": "[S1] Hello, this is a test with audio prompt. [S1]",
                "audio_prompt": audio_file
            },
            {
                "name": "with_prompt_s2",
                "text": "[S2] Hello, this is a test with audio prompt. [S2]",
                "audio_prompt": audio_file
            },
            {
                "name": "with_prompt_and_transcript",
                "text": "This is Seraphina speaking. [S2] Hello, this is a test with audio prompt. [S2]",
                "audio_prompt": audio_file
            }
        ])
    else:
        print(f"\n⚠️  Audio prompt not found: {audio_file}")
    
    # Generation parameters
    gen_params = {
        "temperature": 1.0,
        "cfg_scale": 3.0,
        "top_p": 0.95,
        "use_torch_compile": torch.cuda.get_device_capability()[0] >= 8,  # Ampere+
        "verbose": True
    }
    
    print(f"\n🔧 Generation parameters: {gen_params}")
    print(f"   Torch compile: {'Enabled' if gen_params['use_torch_compile'] else 'Disabled'}")
    
    # Run tests
    print("\n" + "="*60)
    print("🧪 Running generation tests...\n")
    
    for config in test_configs:
        print(f"\n📋 Test: {config['name']}")
        print(f"   Text: {config['text'][:50]}...")
        print(f"   Audio prompt: {config['audio_prompt'] or 'None'}")
        
        try:
            # Time the generation
            torch.cuda.synchronize()  # Ensure GPU is ready
            start_time = time.time()
            
            # Generate audio
            if config['audio_prompt']:
                audio = model.generate(
                    config['text'],
                    audio_prompt=config['audio_prompt'],
                    **gen_params
                )
            else:
                audio = model.generate(
                    config['text'],
                    **gen_params
                )
            
            torch.cuda.synchronize()  # Wait for GPU to finish
            gen_time = time.time() - start_time
            
            if audio is not None and len(audio) > 0:
                # Save output
                output_file = f"gpu_test_{config['name']}.wav"
                sf.write(output_file, audio, 44100, format='WAV', subtype='PCM_16')
                
                duration = len(audio) / 44100
                print(f"   ✅ Generated {duration:.2f}s of audio in {gen_time:.2f}s")
                print(f"   ✅ Saved to: {output_file}")
                print(f"   📊 Realtime factor: {duration/gen_time:.2f}x")
                
                # GPU memory usage
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"   🎮 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            else:
                print(f"   ❌ No audio generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("\n📊 Results Summary:")
    print("\n1. Compare the generated files:")
    print("   - gpu_test_baseline_s1.wav (masculine tendency)")
    print("   - gpu_test_baseline_s2.wav (feminine tendency)")
    print("   - gpu_test_with_prompt_s*.wav (with audio prompt)")
    
    print("\n2. If audio prompt has no effect:")
    print("   • Check if model.generate() accepts audio_prompt parameter")
    print("   • Try passing audio data instead of file path")
    print("   • Verify Dia model version supports voice cloning")
    
    print("\n3. GPU Performance:")
    print(f"   • Model loaded on: {device}")
    print(f"   • Precision: {compute_dtype}")
    print(f"   • Torch compile: {'Available' if torch.cuda.get_device_capability()[0] >= 8 else 'Not available'}")

if __name__ == "__main__":
    test_dia_gpu()