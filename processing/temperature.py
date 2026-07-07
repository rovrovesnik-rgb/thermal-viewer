"""
Temperature processing utilities for P2Pro thermal camera.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TemperatureStats:
    """Temperature statistics for a region."""
    min_temp: float
    max_temp: float
    avg_temp: float
    min_pos: Tuple[int, int]  # (x, y) of minimum
    max_pos: Tuple[int, int]  # (x, y) of maximum


def raw_to_celsius(raw_data: np.ndarray) -> np.ndarray:
    """
    Convert raw 16-bit temperature data to Celsius.

    The P2Pro stores temperature as: raw = (kelvin * 64)
    So: celsius = (raw / 64) - 273.15

    Args:
        raw_data: Raw 16-bit temperature values

    Returns:
        Temperature array in Celsius (float32)
    """
    return (raw_data.astype(np.float32) / 64.0) - 273.15


def celsius_to_raw(celsius: float) -> int:
    """
    Convert Celsius temperature to raw camera value.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Raw 16-bit value
    """
    kelvin = celsius + 273.15
    return int(kelvin * 64)


def celsius_to_fahrenheit(celsius: np.ndarray) -> np.ndarray:
    """Convert Celsius to Fahrenheit."""
    return celsius * 1.8 + 32.0


def fahrenheit_to_celsius(fahrenheit: np.ndarray) -> np.ndarray:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) / 1.8


def get_temperature_stats(temp_data: np.ndarray, mask: Optional[np.ndarray] = None) -> TemperatureStats:
    """
    Calculate temperature statistics for an array or masked region.

    Args:
        temp_data: Temperature array in Celsius
        mask: Optional boolean mask (True = include)

    Returns:
        TemperatureStats with min, max, avg and positions
    """
    if mask is not None:
        data = temp_data[mask]
        # Find positions in original array
        indices = np.where(mask)
        flat_idx = np.argmin(data)
        min_pos = (int(indices[1][flat_idx]), int(indices[0][flat_idx]))
        flat_idx = np.argmax(data)
        max_pos = (int(indices[1][flat_idx]), int(indices[0][flat_idx]))
    else:
        data = temp_data
        min_idx = np.unravel_index(np.argmin(data), data.shape)
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        min_pos = (int(min_idx[1]), int(min_idx[0]))
        max_pos = (int(max_idx[1]), int(max_idx[0]))

    return TemperatureStats(
        min_temp=float(np.min(data)),
        max_temp=float(np.max(data)),
        avg_temp=float(np.mean(data)),
        min_pos=min_pos,
        max_pos=max_pos
    )


def get_roi_stats(temp_data: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> TemperatureStats:
    """
    Get temperature statistics for a rectangular region of interest.

    Args:
        temp_data: Temperature array in Celsius
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner

    Returns:
        TemperatureStats for the ROI
    """
    h, w = temp_data.shape

    # Clamp and sort coordinates
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    roi = temp_data[y1:y2+1, x1:x2+1]

    min_idx = np.unravel_index(np.argmin(roi), roi.shape)
    max_idx = np.unravel_index(np.argmax(roi), roi.shape)

    return TemperatureStats(
        min_temp=float(np.min(roi)),
        max_temp=float(np.max(roi)),
        avg_temp=float(np.mean(roi)),
        min_pos=(x1 + int(min_idx[1]), y1 + int(min_idx[0])),
        max_pos=(x1 + int(max_idx[1]), y1 + int(max_idx[0]))
    )


def normalize_temperature(
    temp_data: np.ndarray,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None
) -> np.ndarray:
    """
    Normalize temperature data to 0-255 range for display.

    Args:
        temp_data: Temperature array in Celsius
        min_temp: Minimum temperature for scaling (auto if None)
        max_temp: Maximum temperature for scaling (auto if None)

    Returns:
        Normalized uint8 array (0-255)
    """
    if min_temp is None:
        min_temp = np.min(temp_data)
    if max_temp is None:
        max_temp = np.max(temp_data)

    # Avoid division by zero
    temp_range = max_temp - min_temp
    if temp_range < 0.01:
        temp_range = 0.01

    normalized = (temp_data - min_temp) / temp_range
    normalized = np.clip(normalized, 0, 1)

    return (normalized * 255).astype(np.uint8)


def apply_temperature_correction(
    temp_celsius: float,
    emissivity: float = 1.0,
    distance: float = 1.0,
    reflection_temp: float = 25.0,
    atmospheric_temp: float = 25.0,
    transmittance: float = 1.0
) -> float:
    """
    Apply environmental correction to temperature reading.

    This is a simplified correction model. For accurate results,
    use the camera's built-in correction with proper calibration data.

    Args:
        temp_celsius: Raw temperature reading in Celsius
        emissivity: Surface emissivity (0-1)
        distance: Distance to target in meters
        reflection_temp: Background/reflection temperature in Celsius
        atmospheric_temp: Ambient air temperature in Celsius
        transmittance: Atmospheric transmittance (0-1)

    Returns:
        Corrected temperature in Celsius
    """
    # Simplified Stefan-Boltzmann correction
    # This is an approximation - real correction needs calibration data

    # Convert to Kelvin
    t_obj = temp_celsius + 273.15
    t_refl = reflection_temp + 273.15
    t_atm = atmospheric_temp + 273.15

    # Apply corrections
    tau = transmittance

    # Radiance contributions
    w_obj = emissivity * tau  # Object contribution weight
    w_refl = (1 - emissivity) * tau  # Reflected contribution weight
    w_atm = 1 - tau  # Atmospheric contribution weight

    # Total radiance (proportional to T^4)
    t_total_4 = t_obj**4  # Simplified - actual would need sensor response curve

    # Corrected temperature
    if w_obj > 0:
        t_corrected_4 = (t_total_4 - w_refl * t_refl**4 - w_atm * t_atm**4) / w_obj
        if t_corrected_4 > 0:
            t_corrected = t_corrected_4 ** 0.25
        else:
            t_corrected = t_obj
    else:
        t_corrected = t_obj

    return t_corrected - 273.15


class TemperatureHistory:
    """Circular buffer for temperature history (for histogram)."""

    def __init__(self, max_samples: int = 1500):  # 60s at 25fps
        self.max_samples = max_samples
        self.min_temps: list[float] = []
        self.max_temps: list[float] = []
        self.avg_temps: list[float] = []
        self.timestamps: list[float] = []

    def add(self, stats: TemperatureStats, timestamp: float):
        """Add a temperature reading to the history."""
        self.min_temps.append(stats.min_temp)
        self.max_temps.append(stats.max_temp)
        self.avg_temps.append(stats.avg_temp)
        self.timestamps.append(timestamp)

        # Trim to max size
        if len(self.timestamps) > self.max_samples:
            self.min_temps = self.min_temps[-self.max_samples:]
            self.max_temps = self.max_temps[-self.max_samples:]
            self.avg_temps = self.avg_temps[-self.max_samples:]
            self.timestamps = self.timestamps[-self.max_samples:]

    def clear(self):
        """Clear all history."""
        self.min_temps.clear()
        self.max_temps.clear()
        self.avg_temps.clear()
        self.timestamps.clear()

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get history as numpy arrays (timestamps, min, max, avg)."""
        return (
            np.array(self.timestamps),
            np.array(self.min_temps),
            np.array(self.max_temps),
            np.array(self.avg_temps)
        )

    def __len__(self) -> int:
        return len(self.timestamps)
