"""
AI-based image upscaling using Real-ESRGAN and Real-CUGAN with ncnn-Vulkan backend.
Supports AMD, NVIDIA, and Intel GPUs via Vulkan.

Real-CUGAN: Sharp edges, preserves detail (recommended for thermal)
Real-ESRGAN: Smooth, removes noise (general photos)
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional
from enum import Enum
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """Suppress stdout to hide progress spam from ncnn."""
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# Try to import upscaling libraries
REALESRGAN_AVAILABLE = False
REALCUGAN_AVAILABLE = False

try:
    from realesrgan_ncnn_py import Realesrgan
    REALESRGAN_AVAILABLE = True
except ImportError:
    Realesrgan = None

try:
    from realcugan_ncnn_py import Realcugan
    REALCUGAN_AVAILABLE = True
except ImportError:
    Realcugan = None


class AIUpscaleModel(Enum):
    """Available AI upscaling models."""
    OFF = "off"
    # Real-CUGAN models (sharp, fast - recommended for thermal)
    CUGAN_SE = "cugan-se"           # Standard, balanced
    CUGAN_PRO = "cugan-pro"         # Professional, best quality
    CUGAN_NOSE = "cugan-nose"       # No denoise, sharpest
    # Real-ESRGAN models (smooth)
    ESRGAN_ANIME = "esrgan-anime"   # Fast video model
    ESRGAN_X4PLUS = "esrgan-x4plus" # Best quality (slow)


MODEL_INFO = {
    AIUpscaleModel.OFF: {
        "display_name": "Off",
        "backend": None,
        "scale": 1,
    },
    # models-se supports 2x, 3x, 4x
    AIUpscaleModel.CUGAN_SE: {
        "display_name": "CUGAN 4x Sharp (Fast)",
        "backend": "cugan",
        "model": "models-se",
        "noise": -1,  # No denoise
        "scale": 4,
    },
    # models-pro only supports 2x, 3x (NO 4x!)
    AIUpscaleModel.CUGAN_PRO: {
        "display_name": "CUGAN 3x Pro",
        "backend": "cugan",
        "model": "models-pro",
        "noise": -1,
        "scale": 3,
    },
    # models-nose only supports 2x with no-denoise (noise=0, not -1!)
    AIUpscaleModel.CUGAN_NOSE: {
        "display_name": "CUGAN 2x NoSE",
        "backend": "cugan",
        "model": "models-nose",
        "noise": 0,  # Must be 0 - NoSE only has no-denoise variant
        "scale": 2,
    },
    AIUpscaleModel.ESRGAN_ANIME: {
        "display_name": "ESRGAN 4x Video (Smooth)",
        "backend": "esrgan",
        "model_id": 2,  # animevideov3-x4
        "scale": 4,
    },
    AIUpscaleModel.ESRGAN_X4PLUS: {
        "display_name": "ESRGAN 4x Plus (Slow)",
        "backend": "esrgan",
        "model_id": 4,  # x4plus
        "scale": 4,
    },
}


def apply_sharpening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Apply aggressive sharpening to an image using unsharp mask.

    Args:
        image: BGR uint8 image
        strength: Sharpening strength (0.0 - 1.0)

    Returns:
        Sharpened image
    """
    if strength <= 0:
        return image

    # More aggressive sharpening with larger kernel and stronger effect
    # strength 0.0-1.0 maps to effect multiplier 0.0-3.0
    effect = strength * 3.0

    # Larger sigma for more visible sharpening on upscaled images
    blurred = cv2.GaussianBlur(image, (0, 0), 5)

    # Unsharp mask: original + effect * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + effect, blurred, -effect, 0)
    return sharpened


class AIUpscaler:
    """
    AI upscaler supporting Real-CUGAN and Real-ESRGAN via ncnn-Vulkan.
    """

    def __init__(self):
        self._upscaler = None
        self._current_model: Optional[AIUpscaleModel] = None
        self._gpu_id = 0
        self._scale = 4
        self._sharpen_strength = 0.3  # Default light sharpening

        # Check availability
        self._available = REALESRGAN_AVAILABLE or REALCUGAN_AVAILABLE
        self._gpu_available = False
        self._gpu_name = "CPU"

        if self._available:
            self._detect_gpu()

    def _detect_gpu(self):
        """Detect GPU availability."""
        # Try Real-CUGAN first (it's typically what we want for thermal)
        for gpu_id in [0, -1]:  # 0=GPU, -1=CPU
            try:
                if REALCUGAN_AVAILABLE:
                    with suppress_stdout():
                        test = Realcugan(gpuid=gpu_id, scale=2)
                    del test
                elif REALESRGAN_AVAILABLE:
                    with suppress_stdout():
                        test = Realesrgan(gpuid=gpu_id, model=2)
                    del test

                self._gpu_available = (gpu_id >= 0)
                self._gpu_name = "Vulkan GPU" if gpu_id >= 0 else "CPU"
                self._gpu_id = gpu_id
                print(f"AI Upscale: {self._gpu_name} mode available")
                break
            except Exception as e:
                continue
        else:
            print("AI Upscale: No working backend found")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    @property
    def gpu_name(self) -> str:
        return self._gpu_name

    @property
    def current_model(self) -> Optional[AIUpscaleModel]:
        return self._current_model

    @property
    def sharpen_strength(self) -> float:
        return self._sharpen_strength

    @sharpen_strength.setter
    def sharpen_strength(self, value: float):
        self._sharpen_strength = max(0.0, min(1.0, value))

    def load_model(self, model: AIUpscaleModel) -> bool:
        """Load a model for upscaling."""
        if model == AIUpscaleModel.OFF:
            self._upscaler = None
            self._current_model = None
            return True

        if not self._available:
            return False

        info = MODEL_INFO[model]
        backend = info.get("backend")

        try:
            with suppress_stdout():
                if backend == "cugan" and REALCUGAN_AVAILABLE:
                    self._upscaler = Realcugan(
                        gpuid=self._gpu_id,
                        scale=info["scale"],
                        noise=info.get("noise", -1),
                        model=info.get("model", "models-se"),
                    )
                elif backend == "esrgan" and REALESRGAN_AVAILABLE:
                    self._upscaler = Realesrgan(
                        gpuid=self._gpu_id,
                        model=info["model_id"],
                        tta_mode=False,
                        tilesize=0,
                    )
                else:
                    print(f"Backend {backend} not available")
                    return False

            self._current_model = model
            self._scale = info["scale"]
            print(f"Loaded: {info['display_name']}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self._upscaler = None
            self._current_model = None
            return False

    def upscale(self, image: np.ndarray, apply_sharpen: bool = True) -> np.ndarray:
        """
        Upscale an image.

        Args:
            image: Input BGR image (uint8)
            apply_sharpen: Whether to apply post-sharpening

        Returns:
            Upscaled BGR image (uint8)
        """
        if self._upscaler is None or self._current_model is None:
            return image

        try:
            with suppress_stdout():
                result = self._upscaler.process_cv2(image)

            # Apply sharpening if enabled
            if apply_sharpen and self._sharpen_strength > 0:
                result = apply_sharpening(result, self._sharpen_strength)

            return result

        except Exception as e:
            print(f"Upscale error: {e}")
            return image


# Global instance
_upscaler: Optional[AIUpscaler] = None


def get_ai_upscaler() -> AIUpscaler:
    """Get the global AI upscaler instance."""
    global _upscaler
    if _upscaler is None:
        _upscaler = AIUpscaler()
    return _upscaler


def get_model_names() -> list[str]:
    """Get list of available model display names for UI."""
    names = ["Off"]

    # Real-CUGAN models (preferred for thermal - sharp)
    if REALCUGAN_AVAILABLE:
        names.extend([
            MODEL_INFO[AIUpscaleModel.CUGAN_SE]["display_name"],
            MODEL_INFO[AIUpscaleModel.CUGAN_PRO]["display_name"],
            MODEL_INFO[AIUpscaleModel.CUGAN_NOSE]["display_name"],
        ])

    # Real-ESRGAN models (smooth)
    if REALESRGAN_AVAILABLE:
        names.extend([
            MODEL_INFO[AIUpscaleModel.ESRGAN_ANIME]["display_name"],
            MODEL_INFO[AIUpscaleModel.ESRGAN_X4PLUS]["display_name"],
        ])

    return names


def get_model_by_name(name: str) -> AIUpscaleModel:
    """Get model enum from display name."""
    if name == "Off":
        return AIUpscaleModel.OFF
    for model, info in MODEL_INFO.items():
        if info.get("display_name") == name:
            return model
    return AIUpscaleModel.OFF
