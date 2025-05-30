# Startup Troubleshooting Guide

## The Loop Problem

The original `start_server.py` script gets into loops because it uses `os.execv()` to restart itself after installing packages. If installation fails or there are import issues, it keeps restarting indefinitely.

## Fixed Startup Options

### Option 1: Simple Python Script (Recommended)
```bash
python start_simple.py
python start_simple.py --debug --gpus "0,1"
```

### Option 2: No-Loop Script
```bash
python start_server_fixed.py
python start_server_fixed.py --check-only  # Just check environment
```

### Option 3: Platform-Specific Scripts

**Windows:**
```cmd
start_server.bat
```

**Linux/Mac:**
```bash
./start_server.sh
./start_server.sh --gpus "0" --workers 4
```

## Manual Setup (Most Reliable)

If all scripts fail, do it manually:

### 1. Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Requirements
```bash
# Basic requirements
pip install -r requirements.txt

# For GPU support
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# TTS model
pip install dia-tts
```

### 3. Start Server
```bash
# Basic start
python -m uvicorn src.server:app --host 0.0.0.0 --port 7860

# With GPU configuration
DIA_GPU_MODE=single CUDA_VISIBLE_DEVICES=0 python -m uvicorn src.server:app --host 0.0.0.0 --port 7860 --reload
```

## Common Issues & Solutions

### 1. Loop on Package Installation
**Problem:** Script keeps restarting after trying to install packages

**Solution:** 
- Use `start_simple.py` instead
- Or activate venv manually first

### 2. CUDA Installation Fails
**Problem:** PyTorch CUDA installation fails repeatedly

**Solution:**
```bash
# Force CPU version first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then manually install CUDA version if needed
pip uninstall torch torchaudio
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Virtual Environment Issues
**Problem:** Can't create or activate venv

**Solutions:**
- **Ubuntu/Debian:** `sudo apt install python3-venv`
- **Windows:** Use `py -m venv venv` instead of `python -m venv venv`
- **Mac:** Install Python from python.org, not system Python

### 4. Import Errors After Installation
**Problem:** Packages install but can't be imported

**Solution:**
```bash
# Clear Python cache
python -c "import sys; print(sys.path)"
rm -rf __pycache__ src/__pycache__

# Reinstall in clean environment
pip uninstall -y fastapi uvicorn torch dia-tts
pip install --no-cache-dir -r requirements.txt
```

### 5. GPU Not Detected
**Problem:** Server starts but doesn't use GPU

**Check:**
```bash
# Verify CUDA
nvidia-smi
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check server GPU status
curl http://localhost:7860/gpu/status
```

## Environment Variables for Troubleshooting

```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Use specific GPUs
export CUDA_VISIBLE_DEVICES="0,1"
export DIA_GPU_IDS="0,1"

# Debug mode
export DIA_DEBUG=1
export DIA_SHOW_PROMPTS=1

# Disable optimizations
export DIA_DISABLE_TORCH_COMPILE=1
```

## Verification Steps

1. **Check venv is active:**
   ```bash
   which python  # Should point to venv/bin/python
   ```

2. **Check packages:**
   ```bash
   pip list | grep -E "(torch|fastapi|dia)"
   ```

3. **Test imports:**
   ```bash
   python -c "import torch, fastapi, dia; print('All imports OK')"
   ```

4. **Test GPU:**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
   ```

5. **Start server:**
   ```bash
   python -m uvicorn src.server:app --host 0.0.0.0 --port 7860
   ```

## Quick Start Commands

**Just want to test?**
```bash
# Activate existing venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Start with minimal config
python -m uvicorn src.server:app --port 7860
```

**For development:**
```bash
source venv/bin/activate
export DIA_DEBUG=1
python -m uvicorn src.server:app --host 0.0.0.0 --port 7860 --reload
```

**For production:**
```bash
source venv/bin/activate
export DIA_GPU_MODE=single
export CUDA_VISIBLE_DEVICES=0
python -m uvicorn src.server:app --host 0.0.0.0 --port 7860 --workers 1
```