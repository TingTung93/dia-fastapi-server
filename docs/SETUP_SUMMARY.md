# Setup Options Summary

Quick reference for all available setup methods for Dia FastAPI TTS Server.

## Windows Setup Options

### 🚀 Option 1: All-In-One Batch Script (Easiest)
**Best for: Complete beginners, one-click setup**

```cmd
# Simply double-click or run:
setup_windows_aio.bat
```

**What it does:**
- ✅ Installs Python automatically if needed
- ✅ Creates virtual environment
- ✅ Installs all dependencies (PyTorch, FastAPI, Whisper, etc.)
- ✅ Configures GPU/CUDA support automatically
- ✅ Downloads Dia model
- ✅ Starts the server
- ✅ No manual configuration needed

**Requirements:**
- Windows 10/11
- Internet connection
- ~5GB free space
- Administrator privileges (recommended)

---

### 🔧 Option 2: PowerShell Script (Advanced)
**Best for: Advanced users, custom configuration**

```powershell
# Run as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows_aio.ps1
```

**Advanced Options:**
```powershell
# Skip Python installation (if already installed)
.\setup_windows_aio.ps1 -SkipPython

# Force CPU-only mode (no GPU)
.\setup_windows_aio.ps1 -CpuOnly

# Quiet installation (minimal output)
.\setup_windows_aio.ps1 -Quiet

# Combined options
.\setup_windows_aio.ps1 -SkipPython -CpuOnly
```

**Same features as batch script plus:**
- ✅ Advanced command-line options
- ✅ Better error handling
- ✅ Colored output and progress indicators
- ✅ More detailed logging

---

### 📖 Option 3: Manual Installation
**Best for: Developers, custom setups, troubleshooting**

Follow the detailed guide: [`docs/WINDOWS_SETUP.md`](WINDOWS_SETUP.md)

**When to use:**
- Custom Python version needed
- Specific dependency versions required
- Virtual environment already exists
- Troubleshooting installation issues
- Learning the setup process

---

## Linux/macOS Setup

### Manual Installation Only
**Currently available method:**

```bash
# Clone repository
git clone https://github.com/yourusername/dia-fastapi-server.git
cd dia-fastapi-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
python start_server.py
```

**Future:** Linux/macOS AIO scripts planned for future releases.

---

## Quick Start After Setup

### Starting the Server

**Option 1: Simple Launcher (Windows)**
```cmd
start_server_simple.bat
```

**Option 2: Direct Python**
```cmd
# Activate environment first
venv\Scripts\activate.bat    # Windows
source venv/bin/activate     # Linux/macOS

# Start server
python start_server.py
```

**Option 3: With Options**
```cmd
# Development mode
python start_server.py --dev

# Specific GPU setup
python start_server.py --gpu-mode multi --gpus "0,1"

# CPU-only mode
python start_server.py --gpu-mode cpu
```

### Accessing the Server

Once running, the server is available at:
- **Main endpoint**: http://localhost:7860
- **API documentation**: http://localhost:7860/docs
- **Health check**: http://localhost:7860/health

### First Steps

1. **Test basic generation:**
   ```bash
   curl -X POST "http://localhost:7860/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world!", "voice_id": "aria"}' \
     --output test.wav
   ```

2. **Upload custom voices** to `audio_prompts/` folder

3. **Configure SillyTavern** (see [`SPEAKER_TAG_GUIDE.md`](SPEAKER_TAG_GUIDE.md))

---

## Comparison Table

| Feature | AIO Batch | AIO PowerShell | Manual |
|---------|-----------|----------------|--------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Customization** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Error Handling** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Learning Value** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Troubleshooting** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting Quick Links

**Installation Issues:**
- [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md) - Complete Windows guide
- [`STARTUP_TROUBLESHOOTING.md`](STARTUP_TROUBLESHOOTING.md) - Common startup issues

**Performance Issues:**
- [`GPU_STARTUP_GUIDE.md`](GPU_STARTUP_GUIDE.md) - GPU optimization
- [`CUDA_OPTIMIZATION_SUMMARY.md`](CUDA_OPTIMIZATION_SUMMARY.md) - CUDA tuning

**Feature Setup:**
- [`WHISPER_SETUP_GUIDE.md`](WHISPER_SETUP_GUIDE.md) - Transcription setup
- [`AUDIO_PROMPT_TRANSCRIPT_GUIDE.md`](AUDIO_PROMPT_TRANSCRIPT_GUIDE.md) - Voice cloning

**Integration:**
- [`SPEAKER_TAG_GUIDE.md`](SPEAKER_TAG_GUIDE.md) - SillyTavern setup

---

## Support

If you encounter issues:

1. **Check logs**: Run with `--debug` flag
2. **Test components**: Use `test_gpu.py` or `test_whisper_integration.py`
3. **Review documentation**: See appropriate guide above
4. **Report issues**: Create GitHub issue with debug logs

Choose the setup method that best fits your experience level and requirements!