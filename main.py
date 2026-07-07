#!/usr/bin/env python3
"""
P2Pro Thermal Camera Viewer

A GUI application for Infiray P2Pro thermal cameras with:
- Real-time thermal imaging with multiple color palettes
- Visible light camera contour overlay
- Temperature measurement tools (crosshair, ROI rectangles)
- Temperature histogram tracking
- GPU-accelerated scaling
- Screenshot capture

Usage:
    python main.py

Requirements:
    - PyQt6
    - OpenCV
    - NumPy
    - PyQtGraph
    - PyUSB (for camera control)
    - PyOpenCL (optional, for GPU acceleration)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import run_application


def main():
    """Main entry point."""
    # Check for required modules
    missing = []

    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import pyqtgraph
    except ImportError:
        missing.append("pyqtgraph")

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    # Optional modules
    try:
        import pyopencl
        print("OpenCL support: Available")
    except ImportError:
        print("OpenCL support: Not available (GPU scaling disabled)")

    try:
        import usb.core
        print("USB control: Available")
    except ImportError:
        print("USB control: Not available (camera settings disabled)")

    # Run application
    sys.exit(run_application())


if __name__ == "__main__":
    main()
