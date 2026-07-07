"""
Visible light camera capture module for P2Pro.
Captures from the supplementary visible camera for contour overlay.
"""

import cv2
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from threading import Thread, Event
import time


@dataclass
class VisibleFrame:
    """Contains visible camera frame data."""
    # BGR image
    image: np.ndarray
    # Grayscale version
    gray: np.ndarray
    # Frame timestamp
    timestamp: float

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


class VisibleCapture:
    """
    Captures frames from P2Pro visible light camera.
    Used to extract edge contours for overlay on thermal image.
    """

    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    def __init__(self, device_path: str = "/dev/video0"):
        self.device_path = device_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._frame_callback: Optional[Callable[[VisibleFrame], None]] = None
        self._last_frame: Optional[VisibleFrame] = None
        self._width = self.DEFAULT_WIDTH
        self._height = self.DEFAULT_HEIGHT

    def open(self, width: int = 640, height: int = 480, use_mjpeg: bool = False) -> bool:
        """Open the visible camera device."""
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            return False

        self._width = width
        self._height = height

        # Set format
        if use_mjpeg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        else:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 25)

        return True

    def close(self):
        """Close the camera device."""
        self.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> Optional[VisibleFrame]:
        """Read and process a single frame."""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        timestamp = time.time()

        # Convert to grayscale
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return VisibleFrame(
            image=frame,
            gray=gray,
            timestamp=timestamp
        )

    def start(self, callback: Optional[Callable[[VisibleFrame], None]] = None):
        """Start continuous capture in background thread."""
        if self.running:
            return

        if not self.cap or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError(f"Failed to open camera: {self.device_path}")

        self._frame_callback = callback
        self._stop_event.clear()
        self.running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop continuous capture."""
        if not self.running:
            return

        self._stop_event.set()
        self.running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Clear last frame so stale data isn't used
        self._last_frame = None

    def _capture_loop(self):
        """Background capture loop."""
        while not self._stop_event.is_set():
            frame = self.read_frame()
            if frame:
                self._last_frame = frame

                if self._frame_callback:
                    self._frame_callback(frame)

    @property
    def last_frame(self) -> Optional[VisibleFrame]:
        """Most recent captured frame."""
        return self._last_frame

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


if __name__ == "__main__":
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"

    capture = VisibleCapture(device)
    if not capture.open():
        print(f"Failed to open {device}")
        sys.exit(1)

    print(f"Opened {device}")
    print("Reading frames... (Ctrl+C to stop)")

    try:
        while True:
            frame = capture.read_frame()
            if frame:
                print(f"Frame: {frame.width}x{frame.height}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    capture.close()
    print("Done")
