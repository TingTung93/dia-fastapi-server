#!/usr/bin/env python3
"""Fix GPU detection issues"""

import os
import sys
import subprocess

def check_gpu_setup():
    """Comprehensive GPU check"""
    print("🔍 GPU Detection Diagnostics\n")
    
    # 1. Check NVIDIA driver
    print("1. NVIDIA Driver Check:")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "Driver Version" in line:
                    print(f"   ✅ {line.strip()}")
                    break
            # Count GPUs
            gpu_count = result.stdout.count("GeForce") + result.stdout.count("RTX") + result.stdout.count("GTX")
            print(f"   ✅ Found {gpu_count} GPU(s)")
        else:
            print("   ❌ nvidia-smi failed - driver issue?")
            return False
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found - NVIDIA drivers not installed?")
        return False
    
    # 2. Check CUDA installation
    print("\n2. CUDA Installation:")
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "/usr/local/cuda",
        "/opt/cuda"
    ]
    
    cuda_found = False
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"   ✅ CUDA found at: {path}")
            cuda_found = True
            break
    
    if not cuda_found:
        print("   ⚠️  CUDA toolkit not found in standard locations")
    
    # 3. Check PyTorch CUDA
    print("\n3. PyTorch CUDA Support:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   CUDA version: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"           Memory: {mem:.1f} GB")
            return True
        else:
            print("   ❌ PyTorch can't see CUDA")
            return False
            
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error checking PyTorch: {e}")
        return False

def fix_pytorch_cuda():
    """Fix PyTorch CUDA installation"""
    print("\n🔧 Fixing PyTorch CUDA Installation\n")
    
    # Check current PyTorch
    try:
        import torch
        print(f"Current PyTorch: {torch.__version__}")
        if "cu" in torch.__version__:
            print("   PyTorch appears to have CUDA support")
        else:
            print("   PyTorch is CPU-only version")
    except ImportError:
        print("PyTorch not installed")
    
    print("\nReinstalling PyTorch with CUDA 11.8 support...")
    
    commands = [
        # Uninstall current version
        [sys.executable, "-m", "pip", "uninstall", "torch", "torchaudio", "-y"],
        
        # Install CUDA version
        [sys.executable, "-m", "pip", "install", 
         "torch==2.5.1+cu118", "torchaudio==2.5.1+cu118",
         "--index-url", "https://download.pytorch.org/whl/cu118"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Success")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            return False
    
    # Test new installation
    print("\n🧪 Testing new installation...")
    try:
        # Import in a fresh process to avoid cached imports
        test_script = """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
"""
        
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        return False

def check_environment_variables():
    """Check GPU-related environment variables"""
    print("\n4. Environment Variables:")
    
    gpu_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER", 
        "DIA_GPU_MODE",
        "DIA_GPU_IDS"
    ]
    
    for var in gpu_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}={value}")
            if var == "CUDA_VISIBLE_DEVICES" and value == "":
                print("   ⚠️  CUDA_VISIBLE_DEVICES is empty - this disables GPU!")
        else:
            print(f"   {var}=<not set>")

def main():
    print("🚀 GPU Detection Fix Tool\n")
    
    # Step 1: Comprehensive check
    if check_gpu_setup():
        print("\n✅ GPU setup looks good!")
        
        # Check environment
        check_environment_variables()
        
        print("\n💡 If server still doesn't detect GPU:")
        print("1. Restart your terminal/command prompt")
        print("2. Make sure CUDA_VISIBLE_DEVICES is not set to empty")
        print("3. Try: set DIA_GPU_MODE=single")
        print("4. Check Windows NVIDIA Control Panel > System Information")
        
        return
    
    # Step 2: Try to fix PyTorch
    print("\n🔧 Attempting to fix PyTorch CUDA...")
    
    user_input = input("\nTry to reinstall PyTorch with CUDA? (y/n): ")
    if user_input.lower() == 'y':
        if fix_pytorch_cuda():
            print("\n✅ PyTorch CUDA fixed!")
            print("\nNow restart your terminal and try:")
            print("   python start_simple.py")
        else:
            print("\n❌ Failed to fix PyTorch CUDA")
            print("\nTry manual installation:")
            print("   pip uninstall torch torchaudio")
            print("   pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n📋 Additional troubleshooting:")
    print("1. Restart your computer (sometimes needed after CUDA install)")
    print("2. Check Windows Device Manager for GPU")
    print("3. Update NVIDIA drivers from nvidia.com")
    print("4. Try different CUDA versions (cu121, cu118)")

if __name__ == "__main__":
    main()