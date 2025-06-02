# Dia FastAPI TTS Server - Windows PowerShell All-In-One Setup
# Run with: PowerShell -ExecutionPolicy Bypass -File setup_windows_aio.ps1

param(
    [switch]$SkipPython,
    [switch]$CpuOnly,
    [switch]$Quiet
)

# Set console properties
$Host.UI.RawUI.WindowTitle = "Dia TTS Server - PowerShell Setup"
if (!$Quiet) {
    Clear-Host
    Write-Host ""
    Write-Host "████████████████████████████████████████████████████████████████" -ForegroundColor Green
    Write-Host "  Dia FastAPI TTS Server - Windows PowerShell Setup" -ForegroundColor Green  
    Write-Host "████████████████████████████████████████████████████████████████" -ForegroundColor Green
    Write-Host ""
    Write-Host "This script will automatically:" -ForegroundColor Cyan
    Write-Host "  [1] Check system requirements" -ForegroundColor Yellow
    Write-Host "  [2] Install Python if needed" -ForegroundColor Yellow
    Write-Host "  [3] Create virtual environment" -ForegroundColor Yellow
    Write-Host "  [4] Install all dependencies" -ForegroundColor Yellow
    Write-Host "  [5] Download Dia model" -ForegroundColor Yellow
    Write-Host "  [6] Configure GPU support" -ForegroundColor Yellow
    Write-Host "  [7] Start the server" -ForegroundColor Yellow
    Write-Host ""
    
    if (!$SkipPython -and !$CpuOnly) {
        Write-Host "Press any key to continue or Ctrl+C to cancel..." -ForegroundColor White
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
}

# Function to write colored output
function Write-Status {
    param($Message, $Type = "Info")
    $color = switch ($Type) {
        "Success" { "Green" }
        "Warning" { "Yellow" }
        "Error" { "Red" }
        "Info" { "Cyan" }
        default { "White" }
    }
    Write-Host $Message -ForegroundColor $color
}

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to install Chocolatey
function Install-Chocolatey {
    Write-Status "Installing Chocolatey package manager..." "Info"
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    try {
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        refreshenv
        Write-Status "✓ Chocolatey installed successfully" "Success"
        return $true
    } catch {
        Write-Status "[ERROR] Failed to install Chocolatey: $($_.Exception.Message)" "Error"
        return $false
    }
}

# Function to install Python via Chocolatey
function Install-Python {
    Write-Status "Installing Python 3.11..." "Info"
    try {
        choco install python311 -y --force
        refreshenv
        # Update PATH for current session
        $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
        Write-Status "✓ Python installed successfully" "Success"
        return $true
    } catch {
        Write-Status "[ERROR] Failed to install Python: $($_.Exception.Message)" "Error"
        return $false
    }
}

# Function to test Python installation
function Test-Python {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ Python found: $pythonVersion" "Success"
            return $true
        }
    } catch {
        # Python not found
    }
    return $false
}

# Function to test GPU availability
function Test-GPU {
    try {
        $result = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ NVIDIA GPU detected" "Success"
            return $true
        }
    } catch {
        # No GPU found
    }
    Write-Status "[INFO] No NVIDIA GPU detected - using CPU mode" "Warning"
    return $false
}

# Function to create virtual environment
function New-VirtualEnvironment {
    Write-Status "Creating virtual environment..." "Info"
    try {
        if (Test-Path "venv") {
            Write-Status "✓ Virtual environment already exists" "Success"
        } else {
            python -m venv venv
            if ($LASTEXITCODE -eq 0) {
                Write-Status "✓ Virtual environment created" "Success"
            } else {
                throw "Failed to create virtual environment"
            }
        }
        
        # Activate virtual environment
        Write-Status "Activating virtual environment..." "Info"
        & .\venv\Scripts\Activate.ps1
        
        # Upgrade pip
        python -m pip install --upgrade pip --quiet
        Write-Status "✓ Virtual environment ready" "Success"
        return $true
    } catch {
        Write-Status "[ERROR] Virtual environment setup failed: $($_.Exception.Message)" "Error"
        return $false
    }
}

# Function to install Python packages
function Install-Dependencies {
    param([bool]$HasGPU)
    
    Write-Status "Installing dependencies..." "Info"
    
    try {
        # Install PyTorch
        if ($HasGPU -and !$CpuOnly) {
            Write-Status "Installing PyTorch with CUDA support..." "Info"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
            if ($LASTEXITCODE -ne 0) {
                Write-Status "[WARNING] CUDA PyTorch failed, installing CPU version..." "Warning"
                pip install torch torchvision torchaudio --quiet
            }
        } else {
            Write-Status "Installing PyTorch (CPU version)..." "Info"
            pip install torch torchvision torchaudio --quiet
        }
        
        # Install FastAPI dependencies
        Write-Status "Installing FastAPI dependencies..." "Info"
        pip install fastapi uvicorn[standard] python-multipart pydantic aiofiles --quiet
        
        # Install audio processing
        Write-Status "Installing audio processing libraries..." "Info"
        pip install soundfile librosa numpy scipy --quiet
        
        # Install Whisper
        Write-Status "Installing Whisper for transcription..." "Info"
        pip install git+https://github.com/openai/whisper.git --quiet
        
        # Install additional dependencies  
        Write-Status "Installing additional dependencies..." "Info"
        pip install huggingface_hub safetensors rich click --quiet
        
        # Install Dia model
        Write-Status "Installing Dia model..." "Info"
        pip install git+https://github.com/nari-labs/dia.git --quiet
        
        # Install testing dependencies
        pip install pytest pytest-asyncio httpx --quiet
        
        Write-Status "✓ All dependencies installed successfully" "Success"
        return $true
    } catch {
        Write-Status "[ERROR] Dependency installation failed: $($_.Exception.Message)" "Error"
        return $false
    }
}

# Function to create configuration files
function New-Configuration {
    param([bool]$HasGPU)
    
    Write-Status "Creating configuration files..." "Info"
    
    # Create .env file
    if (!(Test-Path ".env")) {
        $envContent = @"
# Dia TTS Server Configuration
HF_TOKEN=your_huggingface_token_here
DIA_DEBUG=0
DIA_SAVE_OUTPUTS=1
DIA_SHOW_PROMPTS=0
DIA_RETENTION_HOURS=24
DIA_GPU_MODE=$(if ($HasGPU -and !$CpuOnly) { "auto" } else { "cpu" })
"@
        $envContent | Out-File -FilePath ".env" -Encoding UTF8
        Write-Status "✓ Environment file created" "Success"
    } else {
        Write-Status "✓ Environment file already exists" "Success"
    }
    
    # Create directories
    @("audio_outputs", "audio_prompts") | ForEach-Object {
        if (!(Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ | Out-Null
        }
    }
    Write-Status "✓ Audio directories ready" "Success"
}

# Function to test installation
function Test-Installation {
    Write-Status "Testing installation..." "Info"
    
    try {
        # Test Python imports
        $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ PyTorch: $torchVersion" "Success"
        } else {
            throw "PyTorch import failed"
        }
        
        $fastapiVersion = python -c "import fastapi; print(fastapi.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ FastAPI: $fastapiVersion" "Success"
        } else {
            throw "FastAPI import failed"
        }
        
        # Test GPU if available
        $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ CUDA available: $cudaAvailable" "Success"
            if ($cudaAvailable -eq "True") {
                $gpuCount = python -c "import torch; print(torch.cuda.device_count())" 2>&1
                Write-Status "✓ GPU count: $gpuCount" "Success"
            }
        }
        
        Write-Status "✓ Installation test passed" "Success"
        return $true
    } catch {
        Write-Status "[ERROR] Installation test failed: $($_.Exception.Message)" "Error"
        return $false
    }
}

# Main execution
try {
    # Check system requirements
    Write-Status ""
    Write-Status "[1/7] Checking system requirements..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    if (!(Test-Administrator)) {
        Write-Status "[WARNING] Running without administrator privileges" "Warning"
        Write-Status "Some operations may fail. Consider running as administrator." "Warning"
    }
    
    $osVersion = [Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        throw "Windows 10 or later required"
    }
    Write-Status "✓ Windows version supported" "Success"
    
    # Check/Install Python
    Write-Status ""
    Write-Status "[2/7] Checking Python installation..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    if (!(Test-Python) -and !$SkipPython) {
        # Try to install Chocolatey first
        $chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
        if (!$chocoInstalled) {
            if (!(Install-Chocolatey)) {
                throw "Failed to install Chocolatey"
            }
        }
        
        if (!(Install-Python)) {
            throw "Failed to install Python"
        }
        
        if (!(Test-Python)) {
            throw "Python installation verification failed"
        }
    } elseif ($SkipPython) {
        Write-Status "✓ Skipping Python installation (--SkipPython)" "Success"
    }
    
    # Create virtual environment
    Write-Status ""
    Write-Status "[3/7] Setting up virtual environment..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    if (!(New-VirtualEnvironment)) {
        throw "Virtual environment setup failed"
    }
    
    # Install dependencies
    Write-Status ""
    Write-Status "[4/7] Installing dependencies..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    $hasGPU = Test-GPU
    if (!(Install-Dependencies -HasGPU $hasGPU)) {
        throw "Dependency installation failed"
    }
    
    # Configure environment
    Write-Status ""
    Write-Status "[5/7] Configuring environment..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    New-Configuration -HasGPU $hasGPU
    
    # Test installation
    Write-Status ""
    Write-Status "[6/7] Testing installation..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    if (!(Test-Installation)) {
        throw "Installation test failed"
    }
    
    # Start server
    Write-Status ""
    Write-Status "[7/7] Starting server..." "Info"
    Write-Status "════════════════════════════════════════" "Info"
    
    Write-Status "Server will start shortly..." "Info"
    Write-Status "Access URL: http://localhost:7860" "Success"
    Write-Status ""
    Write-Status "Available endpoints:" "Info"
    Write-Status "  - Generate TTS: POST /generate" "Info"
    Write-Status "  - Voice list: GET /voices" "Info"
    Write-Status "  - Health check: GET /health" "Info"
    Write-Status "  - API docs: GET /docs" "Info"
    Write-Status ""
    
    if ($hasGPU -and !$CpuOnly) {
        Write-Status "Starting with GPU acceleration..." "Success"
        python start_server.py --gpu-mode auto
    } else {
        Write-Status "Starting with CPU mode..." "Success"
        python start_server.py --gpu-mode cpu
    }
    
} catch {
    Write-Status ""
    Write-Status "████████████████████████████████████████████████████████████████" "Error"
    Write-Status "  Setup failed: $($_.Exception.Message)" "Error"
    Write-Status "████████████████████████████████████████████████████████████████" "Error"
    Write-Status ""
    Write-Status "Common solutions:" "Warning"
    Write-Status "  1. Run as administrator: Right-click → 'Run as administrator'" "Warning"
    Write-Status "  2. Check internet connection" "Warning"
    Write-Status "  3. Ensure Windows 10/11" "Warning"
    Write-Status "  4. Free up disk space (need ~5GB)" "Warning"
    Write-Status "  5. Try with -CpuOnly if GPU issues" "Warning"
    Write-Status ""
    Write-Status "For help, see: docs/WINDOWS_SETUP.md" "Info"
    
    if (!$Quiet) {
        Write-Host "Press any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
    exit 1
}

Write-Status ""
Write-Status "████████████████████████████████████████████████████████████████" "Success"
Write-Status "  Setup completed successfully!" "Success"
Write-Status "████████████████████████████████████████████████████████████████" "Success"
Write-Status ""
Write-Status "Next steps:" "Info"
Write-Status "  1. Set your Hugging Face token in .env file" "Yellow"
Write-Status "  2. Upload audio prompts to audio_prompts/ folder" "Yellow"
Write-Status "  3. Test the API at http://localhost:7860/docs" "Yellow"
Write-Status ""
Write-Status "To start the server later, run: start_server.bat" "Info"

if (!$Quiet) {
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}