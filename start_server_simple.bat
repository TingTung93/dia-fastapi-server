@echo off
title Dia TTS Server
color 0A

echo.
echo ████████████████████████████████████████████████████████████████
echo   Dia FastAPI TTS Server - Quick Start
echo ████████████████████████████████████████████████████████████████
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup first:
    echo   setup_windows_aio.bat
    echo.
    echo Or see: docs/WINDOWS_SETUP.md
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if server file exists
if not exist "start_server.py" (
    echo [ERROR] Server files not found!
    echo Please ensure you're in the correct directory.
    pause
    exit /b 1
)

:: Start server
echo.
echo Starting Dia TTS Server...
echo Server will be available at: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

python start_server.py

echo.
echo Server stopped.
pause