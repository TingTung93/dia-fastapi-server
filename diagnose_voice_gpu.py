#!/usr/bin/env python3
"""GPU-accelerated voice diagnostics"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import json
import time

def diagnose_voice_issue():
    """Diagnose voice cloning issues on GPU"""
    print("🔍 GPU-Accelerated Voice Diagnostics\n")
    
    # Quick GPU check
    if not torch.cuda.is_available():
        print("❌ No GPU available. Server may be running on CPU.")
        print("   This significantly affects performance and quality.")
        return
    
    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "compute_capability": torch.cuda.get_device_capability(),
        "bf16_supported": torch.cuda.is_bf16_supported()
    }
    
    print(f"✅ GPU: {gpu_info['name']}")
    print(f"   Memory: {gpu_info['memory_gb']:.1f}GB")
    print(f"   BF16: {'Yes' if gpu_info['bf16_supported'] else 'No'}")
    
    # Check server
    print("\n📡 Checking server...")
    try:
        import requests
        
        # Check GPU status on server
        response = requests.get("http://localhost:7860/gpu/status")
        if response.status_code == 200:
            server_gpu = response.json()
            print(f"✅ Server GPU mode: {server_gpu.get('gpu_mode')}")
            print(f"   GPUs in use: {server_gpu.get('allowed_gpus')}")
            print(f"   Multi-GPU: {server_gpu.get('use_multi_gpu')}")
        
    except Exception as e:
        print(f"⚠️  Cannot connect to server: {e}")
    
    print("\n" + "="*60)
    print("\n🎯 Quick Diagnosis Steps:\n")
    
    print("1. **Run GPU benchmark** (in conda environment):")
    print("   ```bash")
    print("   conda activate dia")
    print("   python benchmark_audio_prompt_gpu.py")
    print("   ```")
    print("   This will generate test files showing GPU performance")
    
    print("\n2. **Direct GPU test** (bypass server):")
    print("   ```bash")
    print("   conda activate dia") 
    print("   python test_dia_gpu_direct.py")
    print("   ```")
    print("   This tests the model directly on GPU")
    
    print("\n3. **Server-side debugging**:")
    print("   Add this to server.py to verify GPU usage:")
    print("""
    # In generate_audio_from_text():
    if model_instance is not None:
        device = next(model_instance.parameters()).device
        console.print(f"[cyan]Model on device: {device}[/cyan]")
        if device.type == 'cuda':
            console.print(f"[green]✓ Using GPU {device.index}[/green]")
        else:
            console.print(f"[yellow]⚠ Using CPU (slow!)[/yellow]")
    """)
    
    print("\n4. **Common GPU issues**:")
    print("   • Model on CPU: Check CUDA availability during load")
    print("   • Wrong GPU: Check ALLOWED_GPUS environment variable")
    print("   • Out of memory: Monitor with nvidia-smi")
    print("   • Slow generation: Enable torch.compile on Ampere+ GPUs")
    
    print("\n5. **Audio prompt GPU tips**:")
    print("   • Larger cfg_scale (4-5) needs more GPU memory")
    print("   • BF16 is faster than FP16 on modern GPUs")
    print("   • Multi-GPU doesn't help single generation")
    print("   • Audio prompts add ~10-20% generation time")
    
    print("\n💡 **For your specific issue** (masculine voice):")
    print("\n   Since S1/S2 changes didn't help, likely causes:")
    print("   a) Audio prompt not loading (check with benchmark)")
    print("   b) Model not supporting voice cloning properly")
    print("   c) Audio prompt file issues (quality/format)")
    print("   d) Need different generation parameters")
    
    print("\n🔧 **Recommended test sequence**:")
    print("   1. Run benchmark to verify GPU is being used")
    print("   2. Compare files with/without audio prompt")
    print("   3. If no difference, audio prompt isn't working")
    print("   4. If difference but wrong gender, try:")
    print("      - Different cfg_scale values (2.0 vs 5.0)")
    print("      - Longer audio prompt (15-20 seconds)")
    print("      - More explicit transcript")

if __name__ == "__main__":
    diagnose_voice_issue()