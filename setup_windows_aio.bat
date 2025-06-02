@echo off
setlocal enabledelayedexpansion

:: Dia FastAPI TTS Server - Windows All-In-One Setup
:: This script automatically installs everything needed to run the server

title Dia TTS Server - Windows Setup
color 0A

echo.
echo ████████████████████████████████████████████████████████████████
echo   Dia FastAPI TTS Server - Windows All-In-One Setup
echo ████████████████████████████████████████████████████████████████
echo.
echo This script will automatically:
echo   [1] Check system requirements
echo   [2] Install Python if needed  
echo   [3] Create virtual environment
echo   [4] Install all dependencies
echo   [5] Download Dia model
echo   [6] Configure GPU support
echo   [7] Start the server
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo [WARNING] This script requires administrator privileges for some operations.
    echo Please run as administrator for best results.
    echo.
    echo Continuing in 5 seconds...
    timeout /t 5 > nul
)

echo.
echo [1/7] Checking system requirements...
echo ════════════════════════════════════════

:: Check Windows version
ver | findstr /i "10\|11" > nul
if %errorLevel% neq 0 (
    echo [ERROR] Windows 10 or 11 required
    goto :error_exit
)
echo ✓ Windows version supported

:: Check if Python is installed
python --version > nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo [2/7] Installing Python...
    echo ════════════════════════════════════════
    echo Python not found. Installing Python 3.11...
    
    :: Download Python installer
    echo Downloading Python 3.11.9...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python_installer.exe'}"
    
    if not exist "python_installer.exe" (
        echo [ERROR] Failed to download Python installer
        goto :error_exit
    )
    
    :: Install Python silently
    echo Installing Python (this may take a few minutes)...
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    :: Wait for installation to complete
    timeout /t 30 > nul
    
    :: Clean up installer
    del python_installer.exe
    
    :: Refresh PATH
    call refreshenv > nul 2>&1
    
    :: Check if Python is now available
    python --version > nul 2>&1
    if %errorLevel% neq 0 (
        echo [ERROR] Python installation failed. Please install Python manually.
        goto :error_exit
    )
    echo ✓ Python installed successfully
) else (
    echo ✓ Python already installed
    python --version
)

echo.
echo [3/7] Setting up virtual environment...
echo ════════════════════════════════════════

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorLevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        goto :error_exit
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip > nul 2>&1

echo.
echo [4/7] Installing dependencies...
echo ════════════════════════════════════════

:: Check if CUDA is available
nvidia-smi > nul 2>&1
if %errorLevel% equ 0 (
    echo ✓ NVIDIA GPU detected - installing CUDA dependencies
    set CUDA_AVAILABLE=1
    
    :: Install PyTorch with CUDA
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %errorLevel% neq 0 (
        echo [WARNING] CUDA PyTorch installation failed, trying CPU version...
        pip install torch torchvision torchaudio
    )
) else (
    echo [INFO] No NVIDIA GPU detected - installing CPU version
    set CUDA_AVAILABLE=0
    pip install torch torchvision torchaudio
)

:: Install other requirements
echo Installing FastAPI dependencies...
pip install fastapi uvicorn[standard] python-multipart pydantic aiofiles

echo Installing audio processing libraries...
pip install soundfile librosa numpy scipy

echo Installing Whisper for transcription...
pip install git+https://github.com/openai/whisper.git

echo Installing additional dependencies...
pip install huggingface_hub safetensors rich click

echo Installing Dia model...
pip install git+https://github.com/nari-labs/dia.git

echo Installing testing dependencies...
pip install pytest pytest-asyncio httpx

echo ✓ All dependencies installed

echo.
echo [5/7] Configuring environment...
echo ════════════════════════════════════════

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating environment configuration...
    (
        echo # Dia TTS Server Configuration
        echo HF_TOKEN=your_huggingface_token_here
        echo DIA_DEBUG=0
        echo DIA_SAVE_OUTPUTS=1
        echo DIA_SHOW_PROMPTS=0
        echo DIA_RETENTION_HOURS=24
        if !CUDA_AVAILABLE! equ 1 (
            echo DIA_GPU_MODE=auto
        ) else (
            echo DIA_GPU_MODE=cpu
        )
    ) > .env
    echo ✓ Environment file created
) else (
    echo ✓ Environment file already exists
)

:: Create audio directories
if not exist "audio_outputs" mkdir audio_outputs
if not exist "audio_prompts" mkdir audio_prompts
echo ✓ Audio directories ready

echo.
echo [6/7] Testing installation...
echo ════════════════════════════════════════

:: Test Python imports
echo Testing core imports...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
if %errorLevel% neq 0 (
    echo [ERROR] PyTorch import failed
    goto :error_exit
)

python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
if %errorLevel% neq 0 (
    echo [ERROR] FastAPI import failed
    goto :error_exit
)

:: Test GPU if available
if !CUDA_AVAILABLE! equ 1 (
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2> nul
)

echo ✓ Installation test passed

echo.
echo [7/7] Starting server...
echo ════════════════════════════════════════

echo Server will start shortly...
echo You can access the server at: http://localhost:7860
echo.
echo Available endpoints:
echo   - Generate TTS: POST /generate
echo   - Voice list: GET /voices
echo   - Health check: GET /health
echo   - API docs: GET /docs
echo.

if !CUDA_AVAILABLE! equ 1 (
    echo Starting with GPU acceleration...
    python start_server.py --gpu-mode auto
) else (
    echo Starting with CPU mode...
    python start_server.py --gpu-mode cpu
)

goto :end

:error_exit
echo.
echo ████████████████████████████████████████████████████████████████
echo   Setup failed. Please check the error messages above.
echo ████████████████████████████████████████████████████████████████
echo.
echo Common solutions:
echo   1. Run as administrator
echo   2. Check internet connection
echo   3. Ensure Windows 10/11
echo   4. Free up disk space (need ~5GB)
echo.
echo For help, see: docs/WINDOWS_SETUP.md or create an issue on GitHub
echo.
pause
exit /b 1

:end
echo.
echo ████████████████████████████████████████████████████████████████
echo   Setup completed successfully!
echo ████████████████████████████████████████████████████████████████
echo.
echo Next steps:
echo   1. Set your Hugging Face token in .env file
echo   2. Upload audio prompts to audio_prompts/ folder
echo   3. Test the API at http://localhost:7860/docs
echo.
echo To start the server later, run: start_server.bat
echo.
pause