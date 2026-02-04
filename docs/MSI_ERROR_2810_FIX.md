# MSI Error 2810 Fix - ShadowBridge 1.44

## Problem
MSI installer was failing at the end with error 2810, which is a Windows Installer dialog control flow validation error.

## Root Cause
The `setup.py` file had malformed MSI dialog definitions:

1. **AIInstallCheckbox Control** - Had 12 parameters instead of the required 11
2. **InstallAIDeps CustomAction** - Tried to execute with command-line arguments, which isn't supported in MSI custom actions
3. **ControlEvent** - Referenced the problematic InstallAIDeps action

## Changes Made

### setup.py
- **Removed** malformed AIInstallCheckbox control
- **Removed** InstallAIDeps custom action
- **Simplified** ControlEvent to only launch ShadowBridge on finish
- **Removed** unused AIDependencies feature
- **Removed** unused INSTALL_AI_DEPS property
- **Fixed** LaunchApp custom action type (210 → 226 for proper async executable launch)

### shadow_bridge_gui.py
- **Incremented** APP_VERSION from "1.043" to "1.044"

## Result
New MSI file: `C:\shadow\shadow-bridge\dist\ShadowBridge-1.44-win64.msi`

The MSI will now:
1. Install ShadowBridge to the target directory
2. Create desktop and start menu shortcuts
3. Launch ShadowBridge after installation completes
4. **No longer** show AI dependencies checkbox (this was causing the error)

## Testing
To test the fix:
1. Uninstall any existing ShadowBridge version
2. Run the new MSI: `dist\ShadowBridge-1.44-win64.msi`
3. Complete the installation wizard
4. Verify no error 2810 appears at the end
5. Verify ShadowBridge launches successfully

## Python Traceback Errors
If you're still seeing Python traceback errors after installation, please provide:
- The full error message
- When the error occurs (during install, at launch, or during specific operations)
- Any log files from `%TEMP%` or the ShadowBridge install directory

The lines mentioned (27, 137, 8376, 8330) in `shadow_bridge_gui.py` appear correct in the source, so the traceback might be from:
- A different file
- Runtime errors after launch
- MSI installation process errors

## Future AI Dependencies
If AI dependencies (PyTorch, AudioCraft) installation is needed in the future, consider:
1. Post-install download script triggered from within ShadowBridge GUI
2. Separate optional installer package
3. Requirements.txt-based pip install with progress indicator
