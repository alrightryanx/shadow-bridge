# -*- mode: python ; coding: utf-8 -*-
import sv_ttk
import os

# Get sv_ttk theme files path
sv_ttk_path = os.path.dirname(sv_ttk.__file__)

a = Analysis(
    ['shadow_bridge_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('web', 'web'),  # Bundle web templates, static files, and routes
        (sv_ttk_path, 'sv_ttk'),  # Bundle sv_ttk theme files for Windows 11 look
    ],
    hiddenimports=['sv_ttk'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',  # PyTorch - auto-installs on demand
        'diffusers', 'transformers', 'accelerate',  # SD models - auto-install
        'rembg',  # Background removal - auto-install
        'onnxruntime', 'onnx',  # ONNX runtime
        'tensorflow', 'keras',  # Not used
        'scipy', 'sklearn', 'scikit-learn',  # Not needed
        'matplotlib', 'pandas', 'numpy.distutils',  # Not needed for core
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ShadowBridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
