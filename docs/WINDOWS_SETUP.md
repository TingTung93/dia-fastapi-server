# Windows Setup Guide

Complete installation guide for running Dia FastAPI TTS Server on Windows 10/11.

## Quick Start (Recommended)

### Method 1: All-In-One Batch Script

1. **Download the repository**:
   - Download as ZIP from GitHub and extract
   - Or clone: `git clone https://github.com/yourusername/dia-fastapi-server.git`

2. **Run the setup script**:
   ```cmd
   setup_windows_aio.bat
   ```

3. **Follow the prompts** - the script will automatically:
   - Install Python if needed
   - Create virtual environment
   - Install all dependencies
   - Configure GPU support
   - Start the server

### Method 2: PowerShell Script (Advanced)

1. **Open PowerShell as Administrator**

2. **Allow script execution**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Run the setup**:
   ```powershell
   .\setup_windows_aio.ps1
   ```

   **Options available**:
   ```powershell
   # Skip Python installation (if already installed)
   .\setup_windows_aio.ps1 -SkipPython
   
   # Force CPU-only mode
   .\setup_windows_aio.ps1 -CpuOnly
   
   # Quiet installation
   .\setup_windows_aio.ps1 -Quiet
   ```

## Manual Installation

### Prerequisites

- **Windows 10/11** (64-bit)
- **~5GB free disk space**
- **Internet connection**
- **Administrator privileges** (for optimal setup)

#### Optional but Recommended:
- **NVIDIA GPU** with CUDA support
- **Git** for cloning repository

### Step 1: Install Python

#### Option A: Automatic (via setup script)
The AIO scripts will install Python automatically.

#### Option B: Manual Installation
1. Download Python 3.11+ from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```cmd
   python --version
   ```

#### Option C: Via Package Manager
```powershell
# Install Chocolatey first
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python
choco install python311 -y
```

### Step 2: Download the Project

#### Option A: Download ZIP
1. Go to the GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract to desired location

#### Option B: Git Clone
```cmd
git clone https://github.com/yourusername/dia-fastapi-server.git
cd dia-fastapi-server
```

### Step 3: Setup Environment

1. **Open Command Prompt or PowerShell** in the project directory

2. **Create virtual environment**:
   ```cmd
   python -m venv venv
   ```

3. **Activate virtual environment**:
   ```cmd
   # Command Prompt
   venv\Scripts\activate.bat
   
   # PowerShell
   venv\Scripts\Activate.ps1
   ```

4. **Upgrade pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

### Step 4: Install Dependencies

#### For GPU Systems (NVIDIA with CUDA):
```cmd
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

#### For CPU-only Systems:
```cmd
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install other dependencies  
pip install -r requirements.txt
```

#### Install Whisper for Transcription:
```cmd
# Install from GitHub (Python 3.13 compatible)
pip install git+https://github.com/openai/whisper.git
```

### Step 5: Configuration

1. **Create environment file** (optional):
   ```cmd
   copy .env.example .env
   ```

2. **Edit .env file** with your preferences:
   ```env
   HF_TOKEN=your_huggingface_token_here
   DIA_DEBUG=0
   DIA_SAVE_OUTPUTS=1
   DIA_GPU_MODE=auto
   ```

3. **Create audio directories**:
   ```cmd
   mkdir audio_outputs
   mkdir audio_prompts
   ```

### Step 6: Start the Server

#### Basic Start:
```cmd
python start_server.py
```

#### With Options:
```cmd
# Development mode with debug output
python start_server.py --dev

# Specific GPU configuration
python start_server.py --gpu-mode multi --gpus "0,1"

# CPU-only mode
python start_server.py --gpu-mode cpu

# Production mode
python start_server.py --production
```

## Troubleshooting

### Common Issues

#### Python Not Found
**Error**: `'python' is not recognized as an internal or external command`

**Solutions**:
1. Reinstall Python with "Add to PATH" checked
2. Add Python to PATH manually:
   - Open System Properties → Environment Variables
   - Add Python installation directory to PATH
   - Restart Command Prompt

#### Virtual Environment Issues
**Error**: `cannot be loaded because running scripts is disabled`

**Solution** (PowerShell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### GPU Not Detected
**Error**: Server starts but shows CPU mode only

**Solutions**:
1. Install NVIDIA drivers: [nvidia.com](https://www.nvidia.com/drivers/)
2. Install CUDA Toolkit: [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads)
3. Verify with: `nvidia-smi`
4. Reinstall PyTorch with CUDA support

#### Torch Compilation Errors
**Error**: `RuntimeError: Torch not compiled with CUDA enabled`

**Solution**:
```cmd
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Whisper Installation Issues
**Error**: `KeyError: '__version__'` during Whisper install

**Solution**:
```cmd
# Use GitHub version instead of PyPI
pip install git+https://github.com/openai/whisper.git
```

#### Dependency Conflicts
**Error**: Various package conflicts

**Solution**:
```cmd
# Create fresh environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### Firewall/Antivirus Issues
**Error**: Server blocked or files deleted

**Solutions**:
1. Add Python and project folder to antivirus exceptions
2. Allow Python through Windows Firewall
3. Temporarily disable real-time protection during setup

### Performance Issues

#### Slow Generation Speed
**Symptoms**: Generation takes >10 seconds

**Solutions**:
1. Ensure GPU is being used (check startup logs)
2. Close other GPU-intensive applications
3. Try lower torch compile optimization
4. Check GPU memory usage with `nvidia-smi`

#### High Memory Usage
**Symptoms**: System becomes slow/unresponsive

**Solutions**:
1. Use single GPU mode: `--gpu-mode single`
2. Reduce worker count: `--workers 2`
3. Disable torch compile: `--no-torch-compile`
4. Close unnecessary applications

### Installation Verification

#### Test Installation:
```cmd
# Activate environment
venv\Scripts\activate.bat

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import fastapi; print('FastAPI: OK')"

# Test server (quick check)
python start_server.py --check-only
```

#### Test API:
```cmd
# Basic generation test
curl -X POST "http://localhost:7860/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Hello world\", \"voice_id\": \"aria\"}" ^
  --output test.wav

# Check API documentation
# Open: http://localhost:7860/docs
```

## Advanced Configuration

### Environment Variables

Set in `.env` file or system environment:

```env
# Core settings
HF_TOKEN=your_huggingface_token_here
DIA_DEBUG=1                    # Enable debug logging
DIA_SAVE_OUTPUTS=1            # Save generated audio
DIA_SHOW_PROMPTS=1            # Show audio prompts in logs

# GPU settings
DIA_GPU_MODE=auto             # auto, single, multi, cpu
DIA_GPU_IDS=0,1,2             # Specific GPU IDs
DIA_MAX_WORKERS=4             # Override worker count
DIA_DISABLE_TORCH_COMPILE=0   # Disable optimization

# Server settings
DIA_RETENTION_HOURS=24        # Audio file retention
DIA_HOST=0.0.0.0             # Server host
DIA_PORT=7860                # Server port
```

### Startup Scripts

#### Create Desktop Shortcut:
1. Create `Start Dia Server.bat`:
   ```batch
   @echo off
   cd /d "C:\path\to\dia-fastapi-server"
   call venv\Scripts\activate.bat
   python start_server.py --dev
   pause
   ```

2. Create shortcut to this batch file on desktop

#### Windows Service (Advanced):
Use NSSM or similar to run as Windows service for automatic startup.

### Updating

#### Update Server Code:
```cmd
git pull origin main
```

#### Update Dependencies:
```cmd
venv\Scripts\activate.bat
pip install --upgrade -r requirements.txt
```

#### Update Models:
Models are cached automatically. To force re-download:
```cmd
# Clear HuggingFace cache
rmdir /s %USERPROFILE%\.cache\huggingface
```

## Security Considerations

### Windows Defender
- Add project folder to exclusions for better performance
- Allow Python through firewall when prompted

### Network Access
- Server binds to localhost:7860 by default
- To allow network access: set `DIA_HOST=0.0.0.0`
- Consider firewall rules for network deployment

### API Keys
- Store HuggingFace tokens securely in `.env` file
- Don't commit `.env` to version control
- Use environment variables for production deployment

## Getting Help

### Log Files
- Server logs: Console output or redirect to file
- Error logs: Check startup messages
- Debug mode: Use `--dev` flag for detailed output

### Common Commands
```cmd
# Check system info
python start_server.py --check-only

# Test specific features
python test_gpu.py
python test_whisper_integration.py

# View help
python start_server.py --help
```

### Support Resources
1. **Documentation**: See `docs/` folder
2. **GitHub Issues**: Report bugs and feature requests
3. **Community**: Discussions and Q&A
4. **Logs**: Always include debug logs when reporting issues

## Next Steps

After successful installation:

1. **Configure Voices**: Upload audio prompts to `audio_prompts/` folder
2. **Test API**: Visit http://localhost:7860/docs for interactive testing
3. **SillyTavern Integration**: See `docs/SPEAKER_TAG_GUIDE.md`
4. **Performance Tuning**: See `docs/GPU_STARTUP_GUIDE.md`
5. **Voice Cloning**: See `docs/AUDIO_PROMPT_TRANSCRIPT_GUIDE.md`

The server is now ready for production use!