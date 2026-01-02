"""
Image Generation Service for ShadowBridge

Provides:
- Text-to-image generation using Stable Diffusion (via diffusers)
- Background removal using rembg
- Image inpainting

GPU Requirements:
- NVIDIA GPU with CUDA support recommended
- At least 8GB VRAM for SD-XL, 4GB for SD 1.5
- Falls back to CPU if GPU unavailable (slow)
"""

import base64
import io
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional, Dict, Any, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-load heavy dependencies
_diffusers_available = False
_rembg_available = False
_torch_available = False
_pipeline = None
_inpaint_pipeline = None
_setup_thread = None
_setup_lock = threading.Lock()
_setup_status = {
    "state": "idle",  # idle, running, ready, error
    "stage": None,    # checking, installing, downloading, ready, error
    "progress": 0,
    "message": "Idle",
    "error": None,
    "updated_at": int(time.time() * 1000)
}

def _check_dependencies():
    """Check which optional dependencies are available."""
    global _diffusers_available, _rembg_available, _torch_available

    try:
        import torch
        _torch_available = True
        logger.info(f"PyTorch available: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    except ImportError:
        logger.warning("PyTorch not installed. Image generation will not be available.")

    try:
        import diffusers
        _diffusers_available = True
        logger.info(f"Diffusers available: {diffusers.__version__}")
    except ImportError:
        logger.warning("Diffusers not installed. Image generation will not be available.")

    try:
        import rembg
        _rembg_available = True
        logger.info("rembg available for background removal")
    except ImportError:
        logger.warning("rembg not installed. Background removal will not be available.")

# Check on module load
_check_dependencies()

def _set_setup_status(state: str, stage: str, progress: int, message: str, error: Optional[str] = None):
    with _setup_lock:
        _setup_status.update({
            "state": state,
            "stage": stage,
            "progress": progress,
            "message": message,
            "error": error,
            "updated_at": int(time.time() * 1000)
        })

def _install_python_packages(packages):
    if not packages:
        return

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    logger.info(f"Installing image dependencies: {' '.join(packages)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("pip not available. Install Python and run: pip install torch diffusers transformers accelerate") from e

    if result.returncode != 0:
        output = (result.stderr or result.stdout or "pip install failed").strip()
        raise RuntimeError(output)

def _run_image_setup(model_id: Optional[str] = None):
    try:
        _set_setup_status("running", "checking", 5, "Checking image dependencies")
        _check_dependencies()

        missing = []
        if not _torch_available:
            missing.append("torch")
        if not _diffusers_available:
            missing.extend(["diffusers", "transformers", "accelerate"])

        if missing:
            _set_setup_status("running", "installing", 15, "Installing image dependencies")
            _install_python_packages(missing)
            _set_setup_status("running", "installing", 35, "Verifying installation")
            _check_dependencies()

        if not _torch_available or not _diffusers_available:
            raise RuntimeError("Image dependencies not available after install.")

        _set_setup_status("running", "downloading", 60, "Downloading Stable Diffusion model")
        service = get_image_generation_service()
        service.warmup_model(model_id)

        _set_setup_status("ready", "ready", 100, "Image generation ready")
    except Exception as e:
        logger.error(f"Image setup failed: {e}")
        _set_setup_status("error", "error", 100, "Image setup failed", error=str(e))

def start_image_setup(model_id: Optional[str] = None) -> Dict[str, Any]:
    global _setup_thread
    with _setup_lock:
        if _setup_thread is not None and _setup_thread.is_alive():
            return dict(_setup_status)

        _setup_thread = threading.Thread(target=_run_image_setup, args=(model_id,), daemon=True)
        _setup_thread.start()
        return dict(_setup_status)

def get_image_setup_status() -> Dict[str, Any]:
    with _setup_lock:
        return dict(_setup_status)


class ImageGenerationService:
    """Service for generating and manipulating images."""

    # Supported models
    MODELS = {
        "sd-1.5": "runwayml/stable-diffusion-v1-5",
        "sd-xl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd-xl-turbo": "stabilityai/sdxl-turbo",
    }

    DEFAULT_MODEL = "sd-xl-turbo"  # Fast and good quality

    def __init__(self):
        self.current_model = None
        self.device = None
        self._initialized = False

    def _initialize(self, model_id: str = None):
        """Lazy initialization of the pipeline."""
        global _pipeline

        if not _diffusers_available or not _torch_available:
            raise RuntimeError("Image generation requires PyTorch and diffusers. "
                             "Install with: pip install torch diffusers transformers accelerate")

        import torch
        from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting

        model_id = model_id or self.DEFAULT_MODEL
        model_name = self.MODELS.get(model_id, model_id)

        # Skip if already loaded
        if self._initialized and self.current_model == model_id:
            return

        logger.info(f"Loading image generation model: {model_name}")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            torch_dtype = torch.float32
            logger.warning("GPU not available, using CPU (will be slow)")

        # Load pipeline
        try:
            _pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                variant="fp16" if torch_dtype == torch.float16 else None,
                use_safetensors=True
            )
            _pipeline.to(self.device)

            # Enable memory optimizations
            if self.device == "cuda":
                _pipeline.enable_attention_slicing()
                try:
                    _pipeline.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass  # xformers not available

            self.current_model = model_id
            self._initialized = True
            logger.info(f"Model loaded successfully: {model_id}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def warmup_model(self, model_id: str = None) -> str:
        """Ensure the requested model is downloaded and ready."""
        self._initialize(model_id)
        return self.current_model

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.

        Returns:
            Dict with 'success', 'image' (base64), 'seed', 'generation_time_ms'
        """
        start_time = time.time()

        try:
            self._initialize(model)

            import torch

            # Set seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                seed = torch.randint(0, 2**32, (1,)).item()
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # Generate image
            result = _pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, bad quality, distorted",
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

            image = result.images[0]

            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            generation_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "image": image_base64,
                "seed": seed,
                "generation_time_ms": generation_time,
                "width": width,
                "height": height,
                "model": self.current_model
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time_ms": int((time.time() - start_time) * 1000)
            }

    def inpaint_image(
        self,
        image_base64: str,
        mask_base64: str,
        prompt: str,
        negative_prompt: str = None,
        steps: int = 20,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """
        Inpaint an image using a mask.

        Args:
            image_base64: Original image as base64
            mask_base64: Mask image as base64 (white = area to regenerate)
            prompt: Text prompt for the regenerated area

        Returns:
            Dict with 'success', 'image' (base64)
        """
        global _inpaint_pipeline

        start_time = time.time()

        try:
            if not _diffusers_available or not _torch_available:
                raise RuntimeError("Inpainting requires PyTorch and diffusers")

            import torch
            from diffusers import AutoPipelineForInpainting

            # Load inpainting pipeline if not loaded
            if _inpaint_pipeline is None:
                logger.info("Loading inpainting pipeline...")

                if torch.cuda.is_available():
                    device = "cuda"
                    torch_dtype = torch.float16
                else:
                    device = "cpu"
                    torch_dtype = torch.float32

                _inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch_dtype,
                    variant="fp16" if torch_dtype == torch.float16 else None,
                )
                _inpaint_pipeline.to(device)

                if device == "cuda":
                    _inpaint_pipeline.enable_attention_slicing()

            # Decode images
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
            mask = Image.open(io.BytesIO(base64.b64decode(mask_base64))).convert("L")

            # Ensure same size
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)

            # Generate
            result = _inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "blurry, bad quality",
                image=image,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )

            output_image = result.images[0]

            # Convert to base64
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            output_base64 = base64.b64encode(buffer.getvalue()).decode()

            generation_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "image": output_base64,
                "generation_time_ms": generation_time
            }

        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time_ms": int((time.time() - start_time) * 1000)
            }


class BackgroundRemovalService:
    """Service for removing backgrounds from images using rembg."""

    def __init__(self):
        self._session = None

    def _get_session(self):
        """Lazy load rembg session."""
        if self._session is None:
            if not _rembg_available:
                raise RuntimeError("Background removal requires rembg. Install with: pip install rembg")

            from rembg import new_session
            logger.info("Initializing rembg session...")
            self._session = new_session("u2net")  # Good balance of speed/quality
            logger.info("rembg session ready")

        return self._session

    def remove_background(self, image_base64: str) -> Dict[str, Any]:
        """
        Remove background from an image.

        Args:
            image_base64: Image as base64 string

        Returns:
            Dict with 'success', 'image' (base64 PNG with transparency)
        """
        start_time = time.time()

        try:
            from rembg import remove

            # Decode input image
            input_bytes = base64.b64decode(image_base64)
            input_image = Image.open(io.BytesIO(input_bytes))

            # Remove background
            session = self._get_session()
            output_image = remove(input_image, session=session)

            # Ensure RGBA mode for transparency
            if output_image.mode != "RGBA":
                output_image = output_image.convert("RGBA")

            # Convert to base64 PNG
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            output_base64 = base64.b64encode(buffer.getvalue()).decode()

            processing_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "image": output_base64,
                "processing_time_ms": processing_time
            }

        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }


# Global service instances
_image_gen_service = None
_bg_removal_service = None


def get_image_generation_service() -> ImageGenerationService:
    """Get or create the image generation service singleton."""
    global _image_gen_service
    if _image_gen_service is None:
        _image_gen_service = ImageGenerationService()
    return _image_gen_service


def get_bg_removal_service() -> BackgroundRemovalService:
    """Get or create the background removal service singleton."""
    global _bg_removal_service
    if _bg_removal_service is None:
        _bg_removal_service = BackgroundRemovalService()
    return _bg_removal_service


def get_image_service_status() -> Dict[str, Any]:
    """Get status of image services."""
    status = {
        "diffusers_available": _diffusers_available,
        "rembg_available": _rembg_available,
        "torch_available": _torch_available,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None
    }

    if _torch_available:
        try:
            import torch
            status["gpu_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        except Exception:
            pass

    return status
