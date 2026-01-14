$ErrorActionPreference = "Stop"

# 0. Cleanup
Write-Host "Killing any running ShadowBridge instances..."
try { Stop-Process -Name "ShadowBridge" -Force -ErrorAction SilentlyContinue } catch {}

Write-Host "Cleaning up previous builds and logs..."
if (Test-Path "build") { Remove-Item -Path "build" -Recurse -Force }
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

# 3. Kill running instances
Write-Host "Stopping ShadowBridge..."
Stop-Process -Name "ShadowBridge" -ErrorAction SilentlyContinue
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
        Start-Process $buildExe
        Start-Sleep -Seconds 10
        Stop-Process -Name "ShadowBridge" -ErrorAction SilentlyContinue
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
        Start-Process $installedPath
        
        # 7. Wait and Collect Logs
        Write-Host "Waiting 10 seconds for app to initialize..."
        Start-Sleep -Seconds 10
        
        Write-Host "Stopping app..."
        Stop-Process -Name "ShadowBridge" -ErrorAction SilentlyContinue
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
