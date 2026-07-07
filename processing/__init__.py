"""Processing modules for thermal image data."""

from .temperature import (
    raw_to_celsius,
    celsius_to_raw,
    normalize_temperature,
    get_temperature_stats,
    get_roi_stats,
    TemperatureStats,
    TemperatureHistory,
)
from .colormap import (
    Palette,
    apply_colormap,
    get_palette_names,
    get_palette_by_name,
    ColormapManager,
)
from .contour import ContourSettings, ContourExtractor, ContourOverlay
from .gpu_scale import GPUScaler, ScaleAlgorithm, get_scaler, scale_image

__all__ = [
    # Temperature
    'raw_to_celsius',
    'celsius_to_raw',
    'normalize_temperature',
    'get_temperature_stats',
    'get_roi_stats',
    'TemperatureStats',
    'TemperatureHistory',
    # Colormap
    'Palette',
    'apply_colormap',
    'get_palette_names',
    'get_palette_by_name',
    'ColormapManager',
    # Contour
    'ContourSettings',
    'ContourExtractor',
    'ContourOverlay',
    # GPU
    'GPUScaler',
    'ScaleAlgorithm',
    'get_scaler',
    'scale_image',
]
