#!/usr/bin/env python3
"""Quick GPU test"""

print("🔍 Quick GPU Test\n")

# Test 1: Check if nvidia-smi works
print("1. NVIDIA Driver:")
import subprocess
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        gpus = result.stdout.strip().split('\n')
        print(f"   ✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu}")
    else:
        print("   ❌ nvidia-smi failed")
except Exception as e:
    print(f"   ❌ nvidia-smi error: {e}")

# Test 2: PyTorch CUDA
print("\n2. PyTorch:")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA compiled: {torch.version.cuda}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {name} ({mem:.1f} GB)")
        
        # Test GPU tensor
        try:
            x = torch.tensor([1.0]).cuda()
            print("   ✅ GPU tensor test: Success")
        except Exception as e:
            print(f"   ❌ GPU tensor test failed: {e}")
    else:
        print("   ❌ No CUDA devices detected by PyTorch")
        
        # Check if it's a CPU-only installation
        if "+cpu" in torch.__version__ or "cpu" in torch.__version__:
            print("   ⚠️  This appears to be a CPU-only PyTorch installation")
            print("   💡 Need to install CUDA version:")
            print("      pip uninstall torch torchaudio")
            print("      pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
        
except ImportError:
    print("   ❌ PyTorch not installed")

# Test 3: Environment
print("\n3. Environment:")
import os
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible is not None:
    if cuda_visible == "":
        print("   ⚠️  CUDA_VISIBLE_DEVICES is empty (GPU disabled)")
    else:
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
else:
    print("   CUDA_VISIBLE_DEVICES: not set (all GPUs available)")

# Test 4: CUDA installation
print("\n4. CUDA Installation:")
cuda_paths = [
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    "/usr/local/cuda"
]

for path in cuda_paths:
    if os.path.exists(path):
        print(f"   ✅ CUDA found: {path}")
        break
else:
    print("   ⚠️  CUDA toolkit not found in standard locations")

print("\n" + "="*50)

# Give recommendations
try:
    import torch
    if not torch.cuda.is_available():
        print("\n🔧 RECOMMENDED FIXES:")
        print("\n1. Install PyTorch with CUDA:")
        print("   pip uninstall torch torchaudio")
        print("   pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n2. Or try different CUDA version:")
        print("   pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n3. After reinstalling, restart terminal and test again")
    else:
        print("\n✅ GPU setup looks good!")
        print("\nIf server still doesn't detect GPU, check:")
        print("- Restart terminal after PyTorch installation")
        print("- Make sure no CUDA_VISIBLE_DEVICES=\"\" is set")
        print("- Try: python start_simple.py --debug")
        
except ImportError:
    print("\n🔧 Install PyTorch first:")
    print("   pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118")