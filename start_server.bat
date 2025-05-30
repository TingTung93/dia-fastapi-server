@echo off
echo 🚀 Dia TTS Server - Windows Startup
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found!
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
    echo.
)

REM Activate venv
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if packages are installed
echo 📦 Checking packages...
python -c "import fastapi, uvicorn, torch, soundfile" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
    pip install dia-tts
    
    REM Try to install CUDA PyTorch
    echo Installing PyTorch with CUDA support...
    pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
)

REM Check GPU
echo 🎮 Checking GPU...
python -c "import torch; print('GPU available:' if torch.cuda.is_available() else 'CPU only'); print(f'Device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else '')"

REM Set environment variables
set DIA_DEBUG=1
set DIA_GPU_MODE=auto

REM Start server
echo.
echo 🌐 Starting server on http://localhost:7860
echo Press Ctrl+C to stop
echo.

python -m uvicorn src.server:app --host 0.0.0.0 --port 7860 --reload

pause