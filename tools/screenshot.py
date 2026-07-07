"""
Screenshot capture functionality.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class ScreenshotCapture:
    """
    Handles screenshot capture and saving.
    """

    def __init__(self, save_directory: str = None):
        self._save_dir = save_directory or str(Path.home() / "Pictures" / "ThermalCaptures")
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure save directory exists."""
        os.makedirs(self._save_dir, exist_ok=True)

    @property
    def save_directory(self) -> str:
        return self._save_dir

    @save_directory.setter
    def save_directory(self, path: str):
        self._save_dir = path
        self._ensure_directory()

    def capture(
        self,
        image: np.ndarray,
        temperature_data: np.ndarray = None,
        filename: str = None,
        format: str = "png",
        include_temp_data: bool = True
    ) -> str:
        """
        Save a screenshot.

        Args:
            image: BGR image to save
            temperature_data: Optional temperature array for metadata
            filename: Optional filename (auto-generated if None)
            format: Image format ('png' or 'jpg')
            include_temp_data: If True, save temperature data as sidecar file

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thermal_{timestamp}"

        # Ensure correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        filepath = os.path.join(self._save_dir, filename)

        # Convert BGR to RGB for saving
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save image
        pil_image = Image.fromarray(rgb_image)

        if format == "png":
            pil_image.save(filepath, "PNG", compress_level=6)
        else:
            pil_image.save(filepath, "JPEG", quality=95)

        # Save temperature data as sidecar if requested
        if include_temp_data and temperature_data is not None:
            temp_filepath = filepath.rsplit('.', 1)[0] + "_temps.npy"
            np.save(temp_filepath, temperature_data)

        return filepath

    def capture_with_overlays(
        self,
        thermal_image: np.ndarray,
        display_image: np.ndarray,
        temperature_data: np.ndarray = None,
        include_overlays: bool = True
    ) -> tuple[str, str]:
        """
        Capture both raw thermal and display (with overlays) images.

        Args:
            thermal_image: Raw colorized thermal image
            display_image: Image with overlays (crosshair, ROIs, etc.)
            temperature_data: Temperature array
            include_overlays: If True, save overlay version

        Returns:
            Tuple of (raw_filepath, overlay_filepath or None)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw thermal
        raw_path = self.capture(
            thermal_image,
            temperature_data,
            f"thermal_{timestamp}_raw",
            "png",
            include_temp_data=True
        )

        overlay_path = None
        if include_overlays:
            overlay_path = self.capture(
                display_image,
                None,
                f"thermal_{timestamp}_overlay",
                "png",
                include_temp_data=False
            )

        return raw_path, overlay_path


# Global instance
_screenshot: Optional[ScreenshotCapture] = None


def get_screenshot_capture() -> ScreenshotCapture:
    """Get the global screenshot capture instance."""
    global _screenshot
    if _screenshot is None:
        _screenshot = ScreenshotCapture()
    return _screenshot


def take_screenshot(image: np.ndarray, temperature_data: np.ndarray = None) -> str:
    """Convenience function to take a screenshot."""
    return get_screenshot_capture().capture(image, temperature_data)
