"""Capture modules for thermal and visible cameras."""

from .device_detect import DeviceDetector, VideoDevice, detect_p2pro_cameras
from .thermal_capture import ThermalCapture, ThermalFrame
from .visible_capture import VisibleCapture, VisibleFrame
from .camera_control import P2ProControl, GainMode, ThermalParams

__all__ = [
    'DeviceDetector',
    'VideoDevice',
    'detect_p2pro_cameras',
    'ThermalCapture',
    'ThermalFrame',
    'VisibleCapture',
    'VisibleFrame',
    'P2ProControl',
    'GainMode',
    'ThermalParams',
]
