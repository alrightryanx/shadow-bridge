from __future__ import annotations

import re
from pathlib import Path
import sys

from cx_Freeze import Executable, setup

BASE_DIR = Path(__file__).resolve().parent


def _read_version() -> str:
    pattern = re.compile(r'APP_VERSION\s*=\s*"([^"]+)"')
    with open(BASE_DIR / "shadow_bridge_gui.py", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return match.group(1)
    return "0.0.1"


include_files = [
    (str(BASE_DIR / "web" / "templates"), "web/templates"),
    (str(BASE_DIR / "web" / "static"), "web/static"),
    (str(BASE_DIR / "images"), "images"),
    (str(BASE_DIR / "icon.ico"), "icon.ico"),
]

data_dir = BASE_DIR / "data"
if data_dir.exists():
    include_files.append((str(data_dir), "data"))

build_exe_options = {
    "include_files": include_files,
    "include_msvcr": True,
    "packages": [
        "flask",
        "flask_socketio",
        "pystray",
        "web",
        "web.routes",
        "web.services",
        "cryptography",
        "qrcode",
        "psutil",
        "requests",
        "engineio",
        "bidict",
    ],
    "includes": ["sv_ttk"],
    "excludes": [
        "torch",
        "torchvision",
        "torchaudio",
        "diffusers",
        "transformers",
        "accelerate",
        "rembg",
        "onnxruntime",
        "onnx",
        "tensorflow",
        "keras",
        "scipy",
        "sklearn",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "numpy.distutils",
    ],
}

bdist_msi_options = {
    "upgrade_code": "{B87662D2-8B6B-4B0B-A69C-2F7AC7E2F3A6}",
    "initial_target_dir": r"[ProgramFilesFolder]\ShadowBridge",
    "add_to_path": False,
    "data": {
        "Shortcut": [
            (
                "DesktopShortcut",
                "DesktopFolder",
                "ShadowBridge",
                "TARGETDIR",
                "[TARGETDIR]ShadowBridge.exe",
                None,
                "ShadowBridge",
                None,
                str(BASE_DIR / "icon.ico"),
                None,
                "TARGETDIR",
                None,
            )
        ]
    },
}

base = "gui" if sys.platform == "win32" else None

executables = [
    Executable(
        script=str(BASE_DIR / "shadow_bridge_gui.py"),
        base=base,
        target_name="ShadowBridge.exe",
        icon=str(BASE_DIR / "icon.ico"),
    )
]

setup(
    name="ShadowBridge",
    version=_read_version(),
    description="Secure bridge between ShadowAI on Android and your PC.",
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=executables,
)
