$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Stop-ShadowBridge {
    Write-Host "Checking for running ShadowBridge instances..."
    $processes = Get-Process -Name "ShadowBridge" -ErrorAction SilentlyContinue
    if ($processes) {
        Write-Host "Stopping $($processes.Count) instance(s) of ShadowBridge..."
        Stop-Process -Name "ShadowBridge" -Force -ErrorAction SilentlyContinue
        # Wait for processes to actually exit
        $timeout = 10 # seconds
        $elapsed = 0
        while ((Get-Process -Name "ShadowBridge" -ErrorAction SilentlyContinue) -and ($elapsed -lt $timeout)) {
            Start-Sleep -Seconds 1
            $elapsed++
        }
        if (Get-Process -Name "ShadowBridge" -ErrorAction SilentlyContinue) {
            Write-Warning "ShadowBridge failed to stop within $timeout seconds."
        } else {
            Write-Host "ShadowBridge stopped successfully."
        }
    }
}

# 0. Ensure singleton execution
Stop-ShadowBridge

# 0. Auto-Increment Version
Write-Host "Auto-incrementing ShadowBridge version..."
try {
    py update_version.py
} catch {
    Write-Warning "Version auto-increment failed: $_"
}

# 1. Cleanup
Stop-ShadowBridge

Write-Host "Cleaning up previous builds and logs..."
if (Test-Path "build") {
    try {
        Remove-Item -Path "build" -Recurse -Force -ErrorAction Stop
    } catch {
        Write-Warning "Could not remove build dir (file lock). Trying robocopy wipe..."
        $emptyDir = "$env:TEMP\shadow_empty_$$"
        New-Item -ItemType Directory -Path $emptyDir -Force | Out-Null
        robocopy $emptyDir "build\exe.win-amd64-3.13" /MIR /NFL /NDL /NJH /NJS 2>$null
        Remove-Item $emptyDir -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path "build\exe.win-amd64-3.13") {
            Write-Warning "Build dir still locked. Renaming and continuing..."
            $ts = Get-Date -Format "yyyyMMddHHmmss"
            Rename-Item "build" "build_old_$ts" -ErrorAction SilentlyContinue
        }
    }
}
# NOTE: dist directory is NOT cleared to preserve previous MSI builds
$logFile = "$env:USERPROFILE\.shadowai\shadowbridge.log"
try {
    if (Test-Path $logFile) { Remove-Item -Path $logFile -Force -ErrorAction SilentlyContinue }
} catch {
    Write-Warning "Could not delete log file. Proceeding anyway..."
}

# 1. Build MSI
Write-Host "Building MSI..."
py setup.py bdist_msi

# 2. Find the MSI
$msiFile = Get-ChildItem dist\*.msi | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $msiFile) {
    Write-Error "MSI not found in dist/"
}
Write-Host "Found MSI: $($msiFile.FullName)"

# 3. Kill running instances again before install
Stop-ShadowBridge
Start-Sleep -Seconds 2

# 4. Uninstall old version (Optional, but clean)
# We skip explicit uninstall as msiexec /i usually handles upgrade if versions differ

# 5. Install MSI
Write-Host "Installing MSI... (User may need to approve UAC)"
# /passive shows progress bar but is automated. /qn is silent.
# Using Start-Process -Wait to block until done.
$installProcess = Start-Process "msiexec.exe" -ArgumentList "/i `"$($msiFile.FullName)`" /passive" -PassThru -Wait

if ($installProcess.ExitCode -ne 0) {
    Write-Warning "Install failed or was cancelled. Exit code: $($installProcess.ExitCode)"
    # Attempting to run from build directory as fallback
    $buildExe = "build\exe.win-amd64-3.13\ShadowBridge.exe"
    if (Test-Path $buildExe) {
        Write-Host "Falling back to running from build directory: $buildExe"
        Stop-ShadowBridge # One last check
        Start-Process $buildExe
        Start-Sleep -Seconds 10
    }
} else {
    Write-Host "Installation complete."
    
    # 6. Run the Installed App
    $installedPath = "${env:ProgramFiles}\ShadowBridge\ShadowBridge.exe"
    if (-not (Test-Path $installedPath)) {
        # Try x86 path just in case
        $installedPath = "${env:ProgramFiles(x86)}\ShadowBridge\ShadowBridge.exe"
    }
    if (-not (Test-Path $installedPath)) {
        # Try LocalAppData (per-user install default)
        $installedPath = "$env:LOCALAPPDATA\Programs\ShadowBridge\ShadowBridge.exe"
    }

    if (Test-Path $installedPath) {
        Write-Host "Launching installed app..."
        Stop-ShadowBridge # Ensure no zombie process
        Start-Process $installedPath -ArgumentList "--aidev"
        
        # 7. Wait and Collect Logs
        Write-Host "Waiting 10 seconds for app to initialize..."
        Start-Sleep -Seconds 10
    } else {
        Write-Error "Installed executable not found at $installedPath"
    }
}

# 8. Show Logs
$logFile = "$env:USERPROFILE\.shadowai\shadowbridge.log"
if (Test-Path $logFile) {
    Write-Host "`n--- Recent Logs ---"
    Get-Content $logFile -Tail 20
} else {
    Write-Warning "Log file not found at $logFile"
}
