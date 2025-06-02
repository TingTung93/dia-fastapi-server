@echo off
echo 🔧 Installing PyTorch with GPU Support
echo.

echo Your hardware:
echo   GPU 0: RTX 3090 (24GB)
echo   GPU 1: RTX 3080 (10GB)
echo   CUDA: Installed
echo.

echo Current PyTorch: CPU-only version
echo Installing: GPU version with CUDA 12.1
echo.

echo Step 1: Uninstalling CPU version...
pip uninstall torch torchaudio torchvision -y

echo.
echo Step 2: Installing GPU version...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo.
echo Step 3: Testing installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPUs detected')"

echo.
echo ✅ Done! Now restart your terminal and run:
echo    python start_server.py
echo.
pause