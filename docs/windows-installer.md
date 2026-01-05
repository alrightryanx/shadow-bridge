# Windows Installer & Audio Dependencies

ShadowBridge includes a compact GUI plus a Flask-powered web dashboard. To keep the download size small we now provide a proper MSI installer instead of a monolithic portable EXE. The installer ships the GUI, tray helpers, and dashboard assets while leaving large AI dependencies (PyTorch + AudioCraft) optional so they can be downloaded separately if needed.

## Building the MSI installer

1. Install Python 3.8+ on a Windows machine and create a virtual environment.
2. From the root of the repository run:
   ```powershell
   python -m pip install -r requirements.txt "cx-Freeze>=6.12"
   ```
3. Build the MSI with:
   ```powershell
   python setup.py bdist_msi
   ```
   The script bundles `shadow_bridge_gui.py`, the `web/` package, icons, and static assets while explicitly excluding optional PyTorch / AudioCraft dependencies.
4. The resulting MSI lives under `dist/ShadowBridge-<version>.msi` and can be distributed like any other Windows installer. The `bdist_msi` target also drops a desktop shortcut for convenience.
5. For faster iterations you can rebuild just the executable with `python setup.py build_exe` before running `bdist_msi`.

## Audio dependencies

### Manual download

Audio generation relies on PyTorch and AudioCraft, which weigh ~2.5 GB combined. These packages are **not** bundled into the MSI; instead, users should install them explicitly after ShadowBridge is set up:

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install audiocraft
```

If CUDA is unavailable you can omit the `--index-url` flag to pull CPU-only builds, or substitute the appropriate PyTorch wheel for your GPU. Once the packages are installed, restart ShadowBridge and check `/audio` in the web dashboard. The UI now surfaces the manual commands and a “Manual install guide” link so users can follow these steps even when running from a frozen bundle.

## Installation notes

- The MSI registers ShadowBridge in `Program Files` and adds a shortcut to the desktop.
- During installer creation the `bdist_msi` options use the upgrade code `{B87662D2-8B6B-4B0B-A69C-2F7AC7E2F3A6}` so new builds will replace older ones.
- Keep the optional AI dependencies out of the MSI to avoid multi-gigabyte downloads and let users control when/where GPU drivers are installed.

## Uninstall & Cleanup

Using the Windows "Uninstall a program" feature removes the application files and shortcuts. 

**Persistence Notes:**
- The `%USERPROFILE%\.shadowai` directory (containing logs, notes, and project data) is **preserved** to prevent data loss.
- Registered SSH keys and the `%APPDATA%\ShadowBridge` folder are also preserved.
- **Full Cleanup**: To completely remove all trace of ShadowBridge, manually delete the folders mentioned above after uninstalling.
