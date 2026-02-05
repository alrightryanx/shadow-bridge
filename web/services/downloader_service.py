"""
Model Weight Downloader Service
Handles automatic downloading of large model weights from Hugging Face.
"""

import os
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

# Base models directory
MODELS_DIR = os.path.join(os.path.expanduser("~"), ".shadowai", "video_models")

# Model Weight Repositories
WEIGHT_REPOS = {
    "hunyuan-15": {
        "repo_id": "Tencent-Hunyuan/HunyuanVideo",
        "local_dir": os.path.join(MODELS_DIR, "hunyuan", "weights"),
        "essential_files": ["hunyuan_video_720_cfg_v1.safetensors"]
    },
    "wan-21": {
        "repo_id": "Wan-Video/Wan2.1-T2V-1.3B",
        "local_dir": os.path.join(MODELS_DIR, "wan", "weights"),
        "essential_files": ["diffusion_pytorch_model.safetensors"]
    },
    "ltx-video": {
        "repo_id": "Lightricks/LTX-Video",
        "local_dir": os.path.join(MODELS_DIR, "ltx", "weights"),
        "essential_files": ["ltx-video-2b-v0.9.safetensors"]
    }
}

class DownloaderService:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_downloads = {} # model_id -> progress %

    def is_weight_present(self, model_id: str) -> bool:
        """Check if essential weights are present for a model."""
        if model_id not in WEIGHT_REPOS:
            return False
        
        config = WEIGHT_REPOS[model_id]
        local_dir = config["local_dir"]
        
        if not os.path.exists(local_dir):
            return False
            
        for f in config["essential_files"]:
            if not os.path.exists(os.path.join(local_dir, f)):
                # Check for subdirectories if it's a full repo download
                found = False
                for root, dirs, files in os.walk(local_dir):
                    if f in files:
                        found = True
                        break
                if not found:
                    return False
        
        return True

    def download_model_weights(self, model_id: str, progress_callback: Optional[Callable] = None):
        """Download model weights with progress updates."""
        if model_id not in WEIGHT_REPOS:
            raise ValueError(f"Unknown model ID: {model_id}")

        config = WEIGHT_REPOS[model_id]
        repo_id = config["repo_id"]
        local_dir = config["local_dir"]

        os.makedirs(local_dir, exist_ok=True)

        logger.info(f"Starting weight download for {model_id} from {repo_id}")
        
        if progress_callback:
            progress_callback({
                "status": "Downloading",
                "message": f"Downloading {model_id} weights (~10-20GB)...",
                "progress": 0
            })

        try:
            # Use snapshot_download for full model repos
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False, # Better for portability
                resume_download=True
            )
            
            if progress_callback:
                progress_callback({
                    "status": "Ready",
                    "message": f"Weights for {model_id} downloaded successfully",
                    "progress": 100
                })
            
            return True
        except Exception as e:
            logger.error(f"Failed to download weights for {model_id}: {e}")
            if progress_callback:
                progress_callback({
                    "status": "Error",
                    "message": f"Download failed: {str(e)}",
                    "progress": 0
                })
            raise

# Global instance
_downloader_service = DownloaderService()

def get_downloader_service():
    return _downloader_service
