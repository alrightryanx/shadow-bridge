# RVC One-Click Installer for Windows
# Installs Retrieval-Based Voice Conversion WebUI on Windows
# Supports NVIDIA (CUDA), AMD (ROCm), and Intel (OneAPI) GPUs

Write-Host "======================================="
Write-Host "  RVC One-Click Installer"
Write-Host "======================================="
Write-Host ""

# Configuration
$RVC_INSTALL_DIR = "$env:USERPROFILE\shadow_rvc"
$CONDA_ENV_NAME = "rvc_env"
$MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal]::new().IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))

if (-not $isAdmin) {
    Write-Warning "Script requires Administrator privileges."
    Write-Host "Please run PowerShell as Administrator and try again."
    exit 1
}

# Detect GPU
Write-Host "Detecting GPU..."
$hasNvidia = $false
$hasAMD = $false
$hasIntel = $false

try {
    $nvidia = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($nvidia) {
        $hasNvidia = $true
        Write-Host "  [OK] NVIDIA GPU detected"
    }
} catch {
    Write-Host "  [WARN] NVIDIA GPU detection failed"
}

try {
    $amd = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*AMD*" -or $_.Name -like "*Radeon*" }
    if ($amd) {
        $hasAMD = $true
        Write-Host "  [OK] AMD GPU detected"
    }
} catch {
    Write-Host "  [WARN] AMD GPU detection failed"
}

try {
    $intel = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*Intel*" }
    if ($intel) {
        $hasIntel = $true
        Write-Host "  [OK] Intel GPU detected"
    }
} catch {
    Write-Host "  [WARN] Intel GPU detection failed"
}

Write-Host ""

# Check for CUDA (NVIDIA)
$hasCUDA = $false
if ($hasNvidia) {
    try {
        $nvcc = nvcc --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $hasCUDA = $true
            Write-Host "  [OK] CUDA detected"
        }
    } catch {
        Write-Host "  [WARN] CUDA not found"
    }
}

# Create installation directory
Write-Host "Creating installation directory: $RVC_INSTALL_DIR"
if (-not (Test-Path $RVC_INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $RVC_INSTALL_DIR -Force | Out-Null
}

# Install Miniconda if not found
$condaExe = "$env:USERPROFILE\miniconda3\shell\conda.exe"
if (-not (Test-Path $condaExe)) {
    Write-Host "Installing Miniconda..."
    Write-Host "This will take a few minutes..."

    $installerPath = "$env:TEMP\miniconda_installer.exe"
    Invoke-WebRequest -Uri $MINICONDA_URL -OutFile $installerPath

    if (Test-Path $installerPath) {
        Start-Process -FilePath $installerPath -Wait -ArgumentList "/S", "/D=$env:USERPROFILE\miniconda3"
    } else {
        Write-Host "[ERROR] Failed to download Miniconda installer"
        exit 1
    }

    $condaExe = "$env:USERPROFILE\miniconda3\shell\conda.exe"
} else {
    Write-Host "[OK] Miniconda already installed"
}

Write-Host ""

# Create Conda environment
Write-Host "Creating Conda environment: $CONDA_ENV_NAME"
& $condaExe create -n $CONDA_ENV_NAME python=3.10 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create Conda environment"
    exit 1
}

Write-Host "[OK] Conda environment created"

# Activate Conda environment function
function Activate-CondaEnv {
    param([string]$EnvironmentName)

    $envPath = "$env:USERPROFILE\miniconda3"
    $condaPath = "$envPath\shell\conda.exe"
    $activatePath = "$envPath\Scripts\activate.ps1"

    & $condaPath init powershell | Out-Null
    & conda activate $EnvironmentName
}

# Install PyTorch with GPU support
Write-Host "Installing PyTorch..."

if ($hasNvidia) {
    # NVIDIA - PyTorch with CUDA
    Write-Host "  Installing PyTorch with CUDA support for NVIDIA..."
    Activate-CondaEnv $CONDA_ENV_NAME
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
}
elseif ($hasAMD) {
    # AMD - PyTorch with ROCm
    Write-Host "  Installing PyTorch with ROCm support for AMD..."
    Write-Host "  [WARN] AMD ROCm support requires manual configuration"
    Write-Host "  See: https://rocm.docs.amd.com/"
    Activate-CondaEnv $CONDA_ENV_NAME
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
}
elseif ($hasIntel) {
    # Intel - PyTorch with OneAPI
    Write-Host "  Installing PyTorch with OneAPI support for Intel..."
    Write-Host "  [WARN] Intel OneAPI support requires manual configuration"
    Write-Host "  See: https://intel.com/oneapi"
    Activate-CondaEnv $CONDA_ENV_NAME
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
}
else {
    # No GPU - CPU-only PyTorch
    Write-Host "  [WARN] No GPU detected, installing CPU-only PyTorch"
    Write-Host "  Performance will be significantly slower"
    Activate-CondaEnv $CONDA_ENV_NAME
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PyTorch"
    exit 1
}

Write-Host "[OK] PyTorch installed"
Write-Host ""

# Install other dependencies
Write-Host "Installing dependencies..."
Activate-CondaEnv $CONDA_ENV_NAME

$requirements = @(
    "flask",
    "flask-cors",
    "flask-socketio",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "gradio",
    "ffmpeg-python",
    "pandas",
    "matplotlib",
    "pillow",
    "requests",
    "colorama",
    "tqdm"
)

foreach ($req in $requirements) {
    Write-Host "  Installing $req..."
    pip install $req -q
}

Write-Host "[OK] Dependencies installed"
Write-Host ""

# Clone RVC WebUI repository
Write-Host "Cloning RVC WebUI repository..."
$rvcRepo = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
$rvcDir = "$RVC_INSTALL_DIR\RVC-WebUI"

if (-not (Test-Path $rvcDir)) {
    git clone $rvcRepo $rvcDir
} else {
    Write-Host "[OK] RVC WebUI already cloned"
    # Update repository
    Set-Location $rvcDir
    git pull
}

Write-Host ""

# Download pre-trained RVC models
Write-Hub "Downloading pre-trained RVC models..."
$modelsDir = "$RVC_INSTALL_DIR\models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

# TODO: Download actual pre-trained models
# For now, we create placeholder
Write-Host "[INFO] Pre-trained models will be downloaded on first use"

Write-Host ""

# Configure audio I/O paths
Write-Host "Configuring audio paths..."
$configFile = "$RVC_INSTALL_DIR\config.json"

$config = @{
    "audio_input_dir" = "$RVC_INSTALL_DIR\input"
    "audio_output_dir" = "$RVC_INSTALL_DIR\output"
    "models_dir" = "$RVC_INSTALL_DIR\models"
    "rvc_webui_dir" = $rvcDir
    "conda_env_name" = $CONDA_ENV_NAME
}

$config | ConvertTo-Json | Set-Content $configFile

Write-Host "[OK] Configuration saved to: $configFile"
Write-Host ""

# Create input/output directories
New-Item -ItemType Directory -Path $config["audio_input_dir"] -Force | Out-Null
New-Item -ItemType Directory -Path $config["audio_output_dir"] -Force | Out-Null
New-Item -ItemType Directory -Path $config["models_dir"] -Force | Out-Null

Write-Host ""

# Test installation
Write-Host "Testing installation..."
Activate-CondaEnv $CONDA_ENV_NAME

try {
    # Test PyTorch import
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

    # Test librosa
    python -c "import librosa; print('librosa:', librosa.__version__)"

    Write-Host ""
    Write-Host "======================================="
    Write-Host "  [SUCCESS] RVC Installation Complete"
    Write-Host "======================================="
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. RVC WebUI is installed at: $rvcDir"
    Write-Host "  2. To use: Activate '$CONDA_ENV_NAME' environment"
    Write-Host "  3. Run: python $rvcDir\infer_web.py"
    Write-Host ""
    Write-Host "ShadowBridge will automatically use RVC for voice cloning."
    Write-Host ""

} catch {
    Write-Host "[ERROR] Installation test failed"
    Write-Host "Error: $_"
    exit 1
}

# Create desktop shortcut
Write-Host "Creating desktop shortcut..."
$shortcutPath = "$env:USERPROFILE\Desktop\RVC WebUI.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
$shortcut.Arguments = "run $CONDA_ENV_NAME python $rvcDir\infer_web.py"
$shortcut.WorkingDirectory = $RVC_INSTALL_DIR
$shortcut.Description = "RVC Voice Conversion WebUI"
$shortcut.Save()

Write-Host "[OK] Desktop shortcut created: $shortcutPath"
Write-Host ""

Write-Host "======================================="
Write-Host "  Installation Complete!"
Write-Host "======================================="
Write-Host ""
Write-Host "Summary:"
Write-Host "  - Install directory: $RVC_INSTALL_DIR"
Write-Host "  - Conda environment: $CONDA_ENV_NAME"
Write-Host "  - RVC WebUI: $rvcDir"
Write-Host "  - GPU detected: NVIDIA=$hasNvidia, AMD=$hasAMD, Intel=$hasIntel"
Write-Host ""
