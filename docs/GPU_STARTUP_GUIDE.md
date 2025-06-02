# GPU-Enabled Server Startup Guide

## Quick Start with GPU

### 1. Use the Enhanced GPU Startup Script

```bash
# Basic GPU startup
python start_server_gpu.py

# Development mode with GPU
python start_server_gpu.py --dev

# Specific GPUs
python start_server_gpu.py --gpus "0,1"

# Check GPU setup only
python start_server_gpu.py --check-only
```

### 2. Manual Setup (if script fails)

```bash
# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install CUDA requirements
pip install -r requirements-cuda.txt

# Or install PyTorch with specific CUDA version
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Start server with GPU
python -m uvicorn src.server:app --host 0.0.0.0 --port 7860
```

## Environment Variables

Set these before starting the server:

```bash
# GPU Configuration
export DIA_GPU_MODE=single      # or "multi" or "auto"
export DIA_GPU_IDS=0,1          # Specific GPUs to use
export CUDA_VISIBLE_DEVICES=0,1 # Limit visible GPUs

# Performance
export DIA_MAX_WORKERS=4        # Worker threads
export DIA_DISABLE_TORCH_COMPILE=0  # Enable torch.compile

# Debug/Logging
export DIA_DEBUG=1              # Enable debug mode
export DIA_SAVE_OUTPUTS=1       # Save generated audio
export DIA_SHOW_PROMPTS=1       # Show processing details
```

## Verify GPU is Being Used

### 1. Check Server Startup Messages

Look for:
```
✅ CUDA available with 2 GPU(s)
✅ Using GPU 0 for single model mode
[green]Using BFloat16 precision for better performance[/green]
```

### 2. Check GPU Status Endpoint

```bash
curl http://localhost:7860/gpu/status
```

Should show:
```json
{
  "gpu_mode": "single",
  "gpu_count": 2,
  "allowed_gpus": [0, 1],
  "models_loaded": {"single_model": true},
  "gpu_memory": {
    "gpu_0": {
      "allocated_gb": 3.2,
      "total_gb": 24.0
    }
  }
}
```

### 3. Monitor GPU Usage

```bash
# Watch GPU usage
nvidia-smi -l 1

# Or use nvtop
nvtop
```

## Troubleshooting

### Server Not Using GPU

1. **Check CUDA Installation**
   ```bash
   nvidia-smi  # Should show GPUs
   nvcc --version  # Should show CUDA version
   ```

2. **Verify PyTorch CUDA**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.device_count())   # Should show GPU count
   ```

3. **Force Reinstall PyTorch with CUDA**
   ```bash
   pip uninstall torch torchaudio -y
   pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

### Common Issues

1. **"CUDA out of memory"**
   - Reduce workers: `--workers 2`
   - Use single GPU mode: `--gpu-mode single`
   - Lower batch processing

2. **"No CUDA GPUs are available"**
   - Check CUDA_VISIBLE_DEVICES isn't set to empty
   - Verify driver: `nvidia-smi`
   - Reinstall CUDA toolkit if needed

3. **Slow Generation Despite GPU**
   - Enable torch.compile (Ampere+ GPUs)
   - Check if model is on GPU in logs
   - Verify BFloat16 is being used

## Optimal GPU Settings

### For RTX 3090/4090 (24GB)
```bash
python start_server_gpu.py \
  --gpu-mode single \
  --workers 4 \
  --gpus 0
```

### For Multiple GPUs
```bash
python start_server_gpu.py \
  --gpu-mode multi \
  --gpus "0,1,2,3" \
  --workers 4  # One per GPU
```

### For Limited VRAM (<8GB)
```bash
python start_server_gpu.py \
  --gpu-mode single \
  --workers 1 \
  --no-torch-compile
```

## Performance Expectations

With proper GPU setup:
- Model loading: 5-10 seconds
- Generation speed: 10-30x realtime
- Memory usage: 3-5GB VRAM
- Tokens/sec: 200-500+

## Final Checklist

- [ ] CUDA installed and working (`nvidia-smi` works)
- [ ] PyTorch with CUDA support installed
- [ ] Server shows "CUDA available" on startup
- [ ] GPU memory allocated when generating
- [ ] Generation speed is >5x realtime
- [ ] No CPU fallback warnings in logs