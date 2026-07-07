"""
Thermal camera capture module for P2Pro.
Handles frame capture and extraction of thermal + temperature data.
"""

import cv2
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from threading import Thread, Event
import time

# Import logger (lazy to avoid circular imports)
_logger = None
def _get_logger():
    global _logger
    if _logger is None:
        try:
            from ui.console_log import get_logger
            _logger = get_logger()
        except ImportError:
            class DummyLogger:
                def camera(self, msg): print(f"[CAM] {msg}")
                def warn(self, msg): print(f"[WARN] {msg}")
                def debug(self, msg): pass
            _logger = DummyLogger()
    return _logger


@dataclass
class ThermalFrame:
    """Contains processed thermal frame data."""
    # Visual thermal image (256x192, uint8 grayscale or colorized)
    thermal_image: np.ndarray
    # Raw temperature data (256x192, float32 in Celsius)
    temperature_data: np.ndarray
    # Frame timestamp
    timestamp: float
    # Frame statistics
    min_temp: float
    max_temp: float
    avg_temp: float
    # Center point temperature
    center_temp: float

    @property
    def width(self) -> int:
        return self.thermal_image.shape[1]

    @property
    def height(self) -> int:
        return self.thermal_image.shape[0]


class ThermalCapture:
    """
    Captures and processes frames from P2Pro thermal camera.

    The P2Pro outputs 256x384 frames where:
    - Top half (256x192): Pre-processed thermal image
    - Bottom half (256x192): Raw 16-bit temperature data

    Temperature formula: temp_celsius = (raw_value / 64) - 273.15
    """

    THERMAL_WIDTH = 256
    THERMAL_HEIGHT = 192
    FRAME_HEIGHT = 384  # Double height (image + temp data)
    WARMUP_FRAMES = 100  # Camera needs ~3-4 seconds to start

    def __init__(self, device_path: str = "/dev/video2"):
        self.device_path = device_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._frame_callback: Optional[Callable[[ThermalFrame], None]] = None
        self._last_frame: Optional[ThermalFrame] = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()
        self._warmup_complete = False
        self._warmup_count = 0
        self._read_failures = 0
        self._consecutive_failures = 0
        self._reopen_attempts = 0
        self._is_recalibrating = False
        self._recalibration_start_time = 0.0  # For display persistence
        self._recalibration_end_time = 0.0  # Track when recal ended for display
        self._recalibration_display_duration = 2.0  # Show status for 2 seconds after

    def open(self) -> bool:
        """Open the thermal camera device."""
        log = _get_logger()
        log.camera(f"Opening {self.device_path} with V4L2 backend...")

        # Try V4L2 backend explicitly
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            log.error(f"Failed to open {self.device_path}")
            return False

        # Configure for raw YUYV capture at 256x384
        r1 = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        r2 = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.THERMAL_WIDTH)
        r3 = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        r4 = self.cap.set(cv2.CAP_PROP_FPS, 25)

        # CRITICAL: Disable automatic RGB conversion to get raw data
        r5 = self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        # CRITICAL: Set buffer size to 1 to avoid V4L2 timeout issues
        r6 = self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Verify settings
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_rgb = self.cap.get(cv2.CAP_PROP_CONVERT_RGB)
        actual_buf = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)

        log.camera(f"Set results: FOURCC={r1}, W={r2}, H={r3}, FPS={r4}, RGB={r5}, BUF={r6}")
        log.camera(f"Actual: {actual_w}x{actual_h}, CONVERT_RGB={actual_rgb}, BUFFERSIZE={actual_buf}")

        # Validate this is actually a thermal camera (256x384)
        if actual_w != self.THERMAL_WIDTH or actual_h != self.FRAME_HEIGHT:
            log.warn(f"Device {self.device_path} is not a thermal camera!")
            log.warn(f"Expected {self.THERMAL_WIDTH}x{self.FRAME_HEIGHT}, got {actual_w}x{actual_h}")
            log.warn("Please select the correct thermal camera device (should support 256x384)")
            self.cap.release()
            self.cap = None
            return False

        # Reset warmup state
        self._warmup_complete = False
        self._warmup_count = 0
        self._consecutive_failures = 0
        self._read_failures = 0

        return True

    def close(self):
        """Close the camera device."""
        self.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> Optional[ThermalFrame]:
        """Read and process a single frame."""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, raw_frame = self.cap.read()
        if not ret or raw_frame is None:
            # Track read failures
            self._read_failures += 1
            self._consecutive_failures += 1

            # Count failures toward warmup so progress advances
            if not self._warmup_complete:
                self._warmup_count += 1

            if self._consecutive_failures <= 3:
                log = _get_logger()
                log.warn(f"cap.read() failed #{self._read_failures}: ret={ret}")
            return None

        # Reset failure counter on success
        self._consecutive_failures = 0
        return self._process_frame(raw_frame)

    def _process_frame(self, raw_frame: np.ndarray) -> Optional[ThermalFrame]:
        """Process raw YUYV frame into thermal data."""
        timestamp = time.time()

        # Expected shape from OpenCV with CONVERT_RGB=0: (384, 256, 2)
        # Channel 0 = Y (luminance), Channel 1 = UV (chrominance)
        expected_bytes = self.FRAME_HEIGHT * self.THERMAL_WIDTH * 2  # 196608 for 384x256 YUYV

        # OpenCV 4.11+ may return raw bytes as (1, N) - need to reshape
        if raw_frame.ndim == 2 and raw_frame.shape[0] == 1 and raw_frame.size == expected_bytes:
            raw_frame = raw_frame.reshape((self.FRAME_HEIGHT, self.THERMAL_WIDTH, 2))

        if raw_frame.ndim != 3 or raw_frame.shape != (self.FRAME_HEIGHT, self.THERMAL_WIDTH, 2):
            # Try to diagnose the issue
            log = _get_logger()
            total_bytes = raw_frame.size * raw_frame.itemsize if hasattr(raw_frame, 'itemsize') else raw_frame.size
            log.warn(f"Unexpected frame: shape={raw_frame.shape}, dtype={raw_frame.dtype}, bytes={total_bytes}")

            # Check if this looks like 1080p visible camera (1920x1080 YUYV = 4147200 bytes)
            if total_bytes == 4147200:
                log.warn("Frame size matches 1080p - wrong camera device? Check /dev/video* selection")
            # Check if CONVERT_RGB might be enabled (would give 3 channels)
            elif raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
                log.warn("Got 3-channel frame - CONVERT_RGB may not be disabled properly")

            return None

        # Split into top and bottom halves
        top_half = raw_frame[:self.THERMAL_HEIGHT]      # Thermal image
        bottom_half = raw_frame[self.THERMAL_HEIGHT:]   # Temperature data

        # Extract thermal image from Y channel
        thermal_image = top_half[:, :, 0].copy()

        # Extract 16-bit temperature data (little-endian: lo + hi*256)
        # Y channel = low byte, UV channel = high byte
        temp_low = bottom_half[:, :, 0].astype(np.uint16)
        temp_high = bottom_half[:, :, 1].astype(np.uint16)
        raw_temp = temp_low + (temp_high << 8)

        # Convert to Celsius: (raw / 64) - 273.15
        temperature_data = (raw_temp.astype(np.float32) / 64.0) - 273.15

        # Check if data is valid
        # During warmup/NUC: raw=0x8000 (32768) -> 238.85°C, Y channel all 0
        # Valid data: temperature varies, Y channel has real values
        max_temp = float(np.max(temperature_data))
        min_temp = float(np.min(temperature_data))
        temp_variance = max_temp - min_temp
        y_max = int(np.max(thermal_image))

        # Detect warmup or recalibration (NUC) - data is invalid when:
        # 1. Temperature variance < 1°C (all pixels same value)
        # 2. Y channel is all zeros (no image data)
        # 3. Temperature is in unrealistic range
        is_calibrating = (temp_variance < 1.0) or (y_max < 5) or (max_temp > 200)
        log = _get_logger()

        if is_calibrating:
            self._warmup_count += 1
            if not self._warmup_complete:
                # Initial warmup
                if self._warmup_count % 25 == 0:  # Log every second
                    log.camera(f"Warming up... frame {self._warmup_count} "
                              f"(max_temp={max_temp:.1f}°C)")
                return None
            else:
                # Recalibration (NUC) event during operation
                if not self._is_recalibrating:
                    log.camera("Recalibration (NUC) started - shutter closed")
                    self._recalibration_start_time = timestamp
                self._is_recalibrating = True
                return None
        else:
            if not self._warmup_complete:
                self._warmup_complete = True
                log.camera(f"Warmup complete after {self._warmup_count} frames "
                          f"(temp range: {min_temp:.1f} - {max_temp:.1f}°C)")
            elif self._is_recalibrating:
                self._is_recalibrating = False
                self._recalibration_end_time = timestamp
                duration = timestamp - self._recalibration_start_time
                log.camera(f"Recalibration (NUC) complete - shutter open (took {duration:.2f}s)")

        return self._create_thermal_frame(thermal_image, temperature_data, timestamp)

    def _create_thermal_frame(
        self,
        thermal_image: np.ndarray,
        temperature_data: np.ndarray,
        timestamp: float
    ) -> ThermalFrame:
        """Create ThermalFrame with statistics."""
        min_temp = float(np.min(temperature_data))
        max_temp = float(np.max(temperature_data))
        avg_temp = float(np.mean(temperature_data))

        # Center point temperature
        cy, cx = temperature_data.shape[0] // 2, temperature_data.shape[1] // 2
        center_temp = float(temperature_data[cy, cx])

        return ThermalFrame(
            thermal_image=thermal_image,
            temperature_data=temperature_data,
            timestamp=timestamp,
            min_temp=min_temp,
            max_temp=max_temp,
            avg_temp=avg_temp,
            center_temp=center_temp
        )

    def start(self, callback: Optional[Callable[[ThermalFrame], None]] = None):
        """Start continuous capture in background thread."""
        if self.running:
            return

        # Close any existing capture - we'll reopen in the thread
        if self.cap:
            self.cap.release()
            self.cap = None

        self._frame_callback = callback
        self._stop_event.clear()
        self.running = True
        self._open_in_thread = True  # Flag to open in thread
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

    def _capture_loop(self):
        """Background capture loop."""
        log = _get_logger()
        log.camera("Capture loop started")

        # Open camera in this thread to avoid cross-thread issues
        if getattr(self, '_open_in_thread', False):
            log.camera("Opening camera in capture thread...")
            max_open_attempts = 3
            for attempt in range(max_open_attempts):
                if self.open():
                    break
                log.warn(f"Open attempt {attempt + 1}/{max_open_attempts} failed, retrying...")
                time.sleep(0.5)
            else:
                log.error("Failed to open camera after retries!")
                self.running = False
                return
            self._open_in_thread = False

        loop_count = 0
        last_log = 0

        while not self._stop_event.is_set():
            loop_count += 1

            # Check for too many consecutive failures - try to reopen camera
            if self._consecutive_failures >= 50 and self._reopen_attempts < 3:
                log.warn(f"Too many read failures ({self._consecutive_failures}), reopening camera...")
                self._reopen_attempts += 1
                self._consecutive_failures = 0

                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(0.3)

                if not self.open():
                    log.error("Failed to reopen camera!")
                    continue
                log.camera("Camera reopened successfully")

            frame = self.read_frame()

            # Log progress every 50 frames or when state changes
            if loop_count - last_log >= 50 or (frame and last_log == 0):
                if frame:
                    log.camera(f"Loop {loop_count}: Got frame, temp={frame.center_temp:.1f}°C")
                else:
                    log.camera(f"Loop {loop_count}: No frame (warmup={self._warmup_count}, "
                              f"complete={self._warmup_complete}, failures={self._consecutive_failures})")
                last_log = loop_count

            if frame:
                self._last_frame = frame
                self._reopen_attempts = 0  # Reset on successful frame
                self._update_fps()

                if self._frame_callback:
                    self._frame_callback(frame)

    def _update_fps(self):
        """Update FPS calculation."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

    @property
    def fps(self) -> float:
        """Current frames per second."""
        return self._fps

    @property
    def last_frame(self) -> Optional[ThermalFrame]:
        """Most recent captured frame."""
        return self._last_frame

    @property
    def is_warming_up(self) -> bool:
        """True if camera is still in warmup phase."""
        return not self._warmup_complete

    @property
    def is_recalibrating(self) -> bool:
        """True if camera is currently recalibrating (NUC) or recently finished."""
        if self._is_recalibrating:
            return True
        # Keep showing recalibration status for a bit after it completes
        if self._recalibration_end_time > 0:
            elapsed = time.time() - self._recalibration_end_time
            if elapsed < self._recalibration_display_duration:
                return True
        return False

    @property
    def recalibration_active(self) -> bool:
        """True only while recalibration is actively in progress."""
        return self._is_recalibrating

    @property
    def warmup_progress(self) -> float:
        """Warmup progress as fraction 0.0 - 1.0."""
        if self._warmup_complete:
            return 1.0
        return min(1.0, self._warmup_count / self.WARMUP_FRAMES)

    def get_temperature_at(self, x: int, y: int) -> Optional[float]:
        """Get temperature at specific pixel coordinate."""
        if self._last_frame is None:
            return None

        if 0 <= x < self.THERMAL_WIDTH and 0 <= y < self.THERMAL_HEIGHT:
            return float(self._last_frame.temperature_data[y, x])
        return None

    def get_roi_stats(
        self,
        x1: int, y1: int,
        x2: int, y2: int
    ) -> Optional[tuple[float, float, float]]:
        """
        Get temperature statistics for a region of interest.
        Returns (min, max, avg) temperatures.
        """
        if self._last_frame is None:
            return None

        # Clamp coordinates
        x1 = max(0, min(x1, self.THERMAL_WIDTH - 1))
        x2 = max(0, min(x2, self.THERMAL_WIDTH - 1))
        y1 = max(0, min(y1, self.THERMAL_HEIGHT - 1))
        y2 = max(0, min(y2, self.THERMAL_HEIGHT - 1))

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        roi = self._last_frame.temperature_data[y1:y2+1, x1:x2+1]
        return float(np.min(roi)), float(np.max(roi)), float(np.mean(roi))


if __name__ == "__main__":
    # Test capture
    import sys

    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video2"

    capture = ThermalCapture(device)
    if not capture.open():
        print(f"Failed to open {device}")
        sys.exit(1)

    print(f"Opened {device}")
    print("Reading frames... (Ctrl+C to stop)")

    try:
        while True:
            frame = capture.read_frame()
            if frame:
                print(f"Frame: {frame.width}x{frame.height}, "
                      f"Temp: {frame.min_temp:.1f}-{frame.max_temp:.1f}°C, "
                      f"Center: {frame.center_temp:.1f}°C")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    capture.close()
    print("Done")
