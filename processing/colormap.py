"""
Colormap/palette system for thermal image visualization.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum, auto


class Palette(Enum):
    """Available thermal color palettes."""
    # OpenCV built-in colormaps
    JET = auto()
    HOT = auto()
    INFERNO = auto()
    PLASMA = auto()
    MAGMA = auto()
    VIRIDIS = auto()
    BONE = auto()
    OCEAN = auto()
    RAINBOW = auto()
    TURBO = auto()

    # Custom palettes
    WHITE_HOT = auto()
    BLACK_HOT = auto()
    IRON = auto()
    ARCTIC = auto()
    MEDICAL = auto()


# Mapping from Palette enum to OpenCV colormap constants
_OPENCV_COLORMAPS: Dict[Palette, int] = {
    Palette.JET: cv2.COLORMAP_JET,
    Palette.HOT: cv2.COLORMAP_HOT,
    Palette.INFERNO: cv2.COLORMAP_INFERNO,
    Palette.PLASMA: cv2.COLORMAP_PLASMA,
    Palette.MAGMA: cv2.COLORMAP_MAGMA,
    Palette.VIRIDIS: cv2.COLORMAP_VIRIDIS,
    Palette.BONE: cv2.COLORMAP_BONE,
    Palette.OCEAN: cv2.COLORMAP_OCEAN,
    Palette.RAINBOW: cv2.COLORMAP_RAINBOW,
    Palette.TURBO: cv2.COLORMAP_TURBO,
}


def _create_custom_lut(colors: list[Tuple[int, int, int]]) -> np.ndarray:
    """
    Create a 256-entry lookup table from a list of color stops.

    Args:
        colors: List of (B, G, R) color tuples

    Returns:
        256x1x3 uint8 array for cv2.LUT
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    n_colors = len(colors)

    for i in range(256):
        # Linear interpolation between color stops
        pos = i / 255.0 * (n_colors - 1)
        idx = int(pos)
        frac = pos - idx

        if idx >= n_colors - 1:
            lut[i, 0] = colors[-1]
        else:
            c1 = np.array(colors[idx])
            c2 = np.array(colors[idx + 1])
            lut[i, 0] = (c1 * (1 - frac) + c2 * frac).astype(np.uint8)

    return lut


# Custom palette LUTs
_CUSTOM_LUTS: Dict[Palette, np.ndarray] = {}


def _get_white_hot_lut() -> np.ndarray:
    """White-hot: black (cold) to white (hot)."""
    colors = [
        (0, 0, 0),       # Black
        (255, 255, 255)  # White
    ]
    return _create_custom_lut(colors)


def _get_black_hot_lut() -> np.ndarray:
    """Black-hot: white (cold) to black (hot)."""
    colors = [
        (255, 255, 255),  # White
        (0, 0, 0)         # Black
    ]
    return _create_custom_lut(colors)


def _get_iron_lut() -> np.ndarray:
    """Iron/Ironbow palette."""
    colors = [
        (0, 0, 0),        # Black
        (0, 0, 128),      # Dark blue
        (128, 0, 128),    # Purple
        (0, 0, 255),      # Red (BGR)
        (0, 128, 255),    # Orange
        (0, 255, 255),    # Yellow
        (255, 255, 255)   # White
    ]
    return _create_custom_lut(colors)


def _get_arctic_lut() -> np.ndarray:
    """Arctic palette: cool colors."""
    colors = [
        (128, 64, 0),      # Dark teal
        (128, 128, 0),     # Teal
        (255, 128, 0),     # Cyan
        (255, 255, 128),   # Light cyan
        (255, 255, 255)    # White
    ]
    return _create_custom_lut(colors)


def _get_medical_lut() -> np.ndarray:
    """Medical imaging palette."""
    colors = [
        (128, 0, 0),       # Dark blue
        (255, 0, 0),       # Blue
        (255, 128, 0),     # Cyan
        (0, 255, 0),       # Green
        (0, 255, 255),     # Yellow
        (0, 128, 255),     # Orange
        (0, 0, 255)        # Red
    ]
    return _create_custom_lut(colors)


def _initialize_custom_luts():
    """Initialize custom palette LUTs."""
    global _CUSTOM_LUTS
    _CUSTOM_LUTS = {
        Palette.WHITE_HOT: _get_white_hot_lut(),
        Palette.BLACK_HOT: _get_black_hot_lut(),
        Palette.IRON: _get_iron_lut(),
        Palette.ARCTIC: _get_arctic_lut(),
        Palette.MEDICAL: _get_medical_lut(),
    }


# Initialize on module load
_initialize_custom_luts()


def apply_colormap(
    grayscale: np.ndarray,
    palette: Palette,
    invert: bool = False
) -> np.ndarray:
    """
    Apply a color palette to a grayscale thermal image.

    Args:
        grayscale: uint8 grayscale image (0-255)
        palette: Color palette to apply
        invert: If True, invert the temperature mapping

    Returns:
        BGR color image
    """
    # Invert if requested
    if invert:
        grayscale = 255 - grayscale

    # Apply colormap
    if palette in _OPENCV_COLORMAPS:
        return cv2.applyColorMap(grayscale, _OPENCV_COLORMAPS[palette])
    elif palette in _CUSTOM_LUTS:
        # For custom LUTs, use cv2.LUT
        gray_3ch = cv2.merge([grayscale, grayscale, grayscale])
        return cv2.LUT(gray_3ch, _CUSTOM_LUTS[palette])
    else:
        # Fallback to JET
        return cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)


def get_palette_preview(palette: Palette, width: int = 256, height: int = 20) -> np.ndarray:
    """
    Generate a preview gradient for a palette.

    Args:
        palette: Palette to preview
        width: Preview width
        height: Preview height

    Returns:
        BGR image showing the palette gradient
    """
    # Create horizontal gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(gradient, (height, 1))

    return apply_colormap(gradient, palette)


def get_palette_names() -> list[str]:
    """Get list of available palette names."""
    return [p.name.replace('_', ' ').title() for p in Palette]


def get_palette_by_name(name: str) -> Palette:
    """Get palette enum from display name."""
    name_normalized = name.upper().replace(' ', '_')
    try:
        return Palette[name_normalized]
    except KeyError:
        return Palette.JET


def get_contrast_color(bgr_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Get a contrasting color for overlay visibility.

    Args:
        bgr_color: Background color (B, G, R)

    Returns:
        Contrasting color (B, G, R)
    """
    b, g, r = bgr_color

    # Calculate luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Return white or black based on luminance
    if luminance > 128:
        return (0, 0, 0)  # Black for light backgrounds
    else:
        return (255, 255, 255)  # White for dark backgrounds


def get_dynamic_contour_color(
    thermal_normalized: np.ndarray,
    palette: Palette,
    x: int,
    y: int
) -> Tuple[int, int, int]:
    """
    Get a contrasting contour color based on the thermal value at a point.

    Args:
        thermal_normalized: Normalized thermal image (0-255)
        palette: Current color palette
        x, y: Pixel coordinates

    Returns:
        BGR color that contrasts with the background
    """
    # Get the thermal value at this point
    value = thermal_normalized[y, x]

    # Create a single-pixel colormap lookup
    single_pixel = np.array([[value]], dtype=np.uint8)
    colored = apply_colormap(single_pixel, palette)
    bg_color = tuple(int(c) for c in colored[0, 0])

    return get_contrast_color(bg_color)


class ColormapManager:
    """Manages colormap application with caching and temperature range control."""

    def __init__(self):
        self._current_palette = Palette.JET
        self._invert = False
        self._cached_lut: Optional[np.ndarray] = None

        # Temperature range settings
        self._auto_range = True  # Auto-scale to frame min/max
        self._manual_min = 15.0  # Manual minimum temperature (°C)
        self._manual_max = 40.0  # Manual maximum temperature (°C)

        # Track actual range for display
        self._last_min = 0.0
        self._last_max = 100.0

    @property
    def palette(self) -> Palette:
        return self._current_palette

    @palette.setter
    def palette(self, value: Palette):
        if value != self._current_palette:
            self._current_palette = value
            self._cached_lut = None

    @property
    def invert(self) -> bool:
        return self._invert

    @invert.setter
    def invert(self, value: bool):
        if value != self._invert:
            self._invert = value
            self._cached_lut = None

    @property
    def auto_range(self) -> bool:
        return self._auto_range

    @auto_range.setter
    def auto_range(self, value: bool):
        self._auto_range = value

    @property
    def manual_min(self) -> float:
        return self._manual_min

    @manual_min.setter
    def manual_min(self, value: float):
        self._manual_min = value
        # Ensure min < max
        if self._manual_min >= self._manual_max:
            self._manual_max = self._manual_min + 1.0

    @property
    def manual_max(self) -> float:
        return self._manual_max

    @manual_max.setter
    def manual_max(self, value: float):
        self._manual_max = value
        # Ensure max > min
        if self._manual_max <= self._manual_min:
            self._manual_min = self._manual_max - 1.0

    @property
    def effective_min(self) -> float:
        """Get the effective minimum temperature being used."""
        return self._last_min

    @property
    def effective_max(self) -> float:
        """Get the effective maximum temperature being used."""
        return self._last_max

    def normalize_temperature(self, temp_data: np.ndarray) -> np.ndarray:
        """
        Normalize temperature data to 0-255 for colormap application.

        Uses either auto-range (per-frame min/max) or manual range.

        Args:
            temp_data: Temperature array in Celsius

        Returns:
            Normalized uint8 array (0-255)
        """
        if self._auto_range:
            min_temp = float(np.min(temp_data))
            max_temp = float(np.max(temp_data))
        else:
            min_temp = self._manual_min
            max_temp = self._manual_max

        # Store for display
        self._last_min = min_temp
        self._last_max = max_temp

        # Normalize
        temp_range = max_temp - min_temp
        if temp_range < 0.01:
            temp_range = 0.01

        normalized = (temp_data - min_temp) / temp_range
        normalized = np.clip(normalized, 0, 1)

        return (normalized * 255).astype(np.uint8)

    def apply(self, grayscale: np.ndarray) -> np.ndarray:
        """Apply the current colormap to a grayscale image."""
        return apply_colormap(grayscale, self._current_palette, self._invert)

    def apply_to_temperature(self, temp_data: np.ndarray) -> np.ndarray:
        """
        Normalize temperature data and apply colormap in one step.

        Args:
            temp_data: Temperature array in Celsius

        Returns:
            BGR color image
        """
        normalized = self.normalize_temperature(temp_data)
        return self.apply(normalized)

    def get_preview(self, width: int = 256, height: int = 20) -> np.ndarray:
        """Get a preview of the current palette."""
        return get_palette_preview(self._current_palette, width, height)
