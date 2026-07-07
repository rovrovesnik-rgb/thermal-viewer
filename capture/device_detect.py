"""
Device detection module for P2Pro thermal camera.
Identifies thermal and visible cameras by USB VID/PID and video capabilities.
Uses parallel scanning for fast device discovery.
"""

import subprocess
import re
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


@dataclass
class VideoDevice:
    """Represents a video capture device."""
    path: str
    name: str
    usb_path: str
    vid: Optional[str] = None
    pid: Optional[str] = None
    formats: list = field(default_factory=list)
    scan_complete: bool = False
    scan_failed: bool = False

    @property
    def max_resolution(self) -> tuple[int, int]:
        """Get the maximum resolution supported by this device."""
        max_w, max_h = 0, 0
        for fmt in self.formats:
            match = re.search(r'(\d+)x(\d+)', fmt)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                if w * h > max_w * max_h:
                    max_w, max_h = w, h
        return max_w, max_h

    @property
    def is_thermal(self) -> bool:
        """Check if this is likely the P2Pro thermal camera."""
        # Primary check: 256x384 resolution is unique to thermal cameras
        has_thermal_res = any("256x384" in fmt for fmt in self.formats)
        if has_thermal_res:
            return True

        # Secondary check: VID/PID match (if USB detection worked)
        if self.vid == "0bda" and self.pid == "5830":
            return True

        return False

    @property
    def is_visible(self) -> bool:
        """Check if this is likely the P2Pro visible light camera."""
        # P2Pro visible: VID 31da, PID 5846 (ZXHY)
        if self.vid == "31da" and self.pid == "5846":
            return True

        # P2Pro visible camera is LOW resolution only (640x480 max)
        # Exclude HD cameras (GoPro, webcams, capture cards)
        has_vga = any("640x480" in fmt for fmt in self.formats)
        has_thermal_res = any("256x384" in fmt or "256x192" in fmt for fmt in self.formats)

        # Check max resolution - P2Pro visible should NOT have HD resolutions
        max_w, max_h = self.max_resolution
        is_hd_or_higher = max_w >= 1280 or max_h >= 720

        # Must have VGA, no thermal res, and NOT be HD capable
        return has_vga and not has_thermal_res and not is_hd_or_higher

    @property
    def thermal_resolution(self) -> Optional[str]:
        """Get the thermal resolution if available."""
        for fmt in self.formats:
            if "256x384" in fmt:
                return "256x384"
            if "256x192" in fmt:
                return "256x192"
        return None

    @property
    def best_visible_format(self) -> Optional[tuple]:
        """Get the best visible format (prefer YUYV 640x480@25fps)."""
        for fmt in self.formats:
            if "YUYV" in fmt and "640x480" in fmt:
                return ("YUYV", 640, 480)
        for fmt in self.formats:
            if "MJPG" in fmt and "640x480" in fmt:
                return ("MJPG", 640, 480)
        return None


class DeviceDetector:
    """Detects and identifies P2Pro camera devices with parallel scanning."""

    # Known USB IDs
    P2PRO_THERMAL_VID = "0bda"
    P2PRO_THERMAL_PID = "5830"
    P2PRO_VISIBLE_VID = "31da"
    P2PRO_VISIBLE_PID = "5846"

    # Timeout for device queries (seconds)
    QUERY_TIMEOUT = 2

    def __init__(self):
        self.devices: list[VideoDevice] = []
        self._lock = threading.Lock()

    def scan(self) -> list[VideoDevice]:
        """Scan for all video devices and identify them (blocking)."""
        self.devices = []
        video_devices = self._get_video_devices()

        # Scan all devices in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._scan_single_device, path, name, usb): path
                for path, name, usb in video_devices
            }

            for future in as_completed(futures):
                try:
                    device = future.result()
                    if device:
                        self.devices.append(device)
                except Exception as e:
                    path = futures[future]
                    print(f"Error scanning {path}: {e}")

        return self.devices

    def scan_async(
        self,
        on_device: Callable[[VideoDevice], None],
        on_complete: Callable[[], None] = None
    ):
        """
        Scan devices in background, calling on_device as each is discovered.
        Devices are emitted immediately when scanned, allowing fast cameras
        to appear while slow ones are still being probed.
        """
        def worker():
            self.devices = []
            video_devices = self._get_video_devices()

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(self._scan_single_device, path, name, usb): path
                    for path, name, usb in video_devices
                }

                for future in as_completed(futures):
                    try:
                        device = future.result()
                        if device:
                            with self._lock:
                                self.devices.append(device)
                            on_device(device)
                    except Exception as e:
                        path = futures[future]
                        print(f"Error scanning {path}: {e}")

            if on_complete:
                on_complete()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def _scan_single_device(
        self, dev_path: str, dev_name: str, usb_path: str
    ) -> Optional[VideoDevice]:
        """Scan a single device (called in parallel)."""
        device = VideoDevice(
            path=dev_path,
            name=dev_name,
            usb_path=usb_path
        )

        # Get supported formats (this is what can be slow)
        device.formats = self._get_formats(dev_path)

        # If no formats, device might be unresponsive
        if not device.formats:
            device.scan_failed = True
            return device

        # Get USB VID/PID (only for devices with thermal/visible resolution)
        if device.is_thermal or "640x480" in str(device.formats):
            device.vid, device.pid = self._get_usb_ids_for_device(dev_path)

        device.scan_complete = True
        return device

    def find_thermal_camera(self) -> Optional[VideoDevice]:
        """Find the P2Pro thermal camera device."""
        for device in self.devices:
            if device.is_thermal:
                return device
        return None

    def find_visible_camera(self) -> Optional[VideoDevice]:
        """Find the P2Pro visible light camera device."""
        for device in self.devices:
            if device.is_visible:
                return device
        return None

    def _get_video_devices(self) -> list[tuple[str, str, str]]:
        """Get list of video devices with their names and USB paths (fast)."""
        devices = []

        try:
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True,
                text=True,
                timeout=self.QUERY_TIMEOUT
            )

            if result.returncode != 0:
                return self._fallback_device_list()

            lines = result.stdout.strip().split("\n")
            current_name = ""
            current_usb = ""

            for line in lines:
                if not line.startswith("\t") and line.strip():
                    current_name = line.strip().rstrip(":")
                    usb_match = re.search(r'\(usb-([^)]+)\)', current_name)
                    current_usb = usb_match.group(1) if usb_match else ""
                elif line.strip().startswith("/dev/video"):
                    dev_path = line.strip()
                    devices.append((dev_path, current_name, current_usb))

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._fallback_device_list()

        return devices

    def _fallback_device_list(self) -> list[tuple[str, str, str]]:
        """Fallback: list /dev/video* directly."""
        devices = []
        for i in range(10):
            dev_path = f"/dev/video{i}"
            if Path(dev_path).exists():
                devices.append((dev_path, f"Video Device {i}", ""))
        return devices

    def _get_usb_ids_for_device(self, dev_path: str) -> tuple[Optional[str], Optional[str]]:
        """Get USB VID/PID by reading sysfs for the specific device."""
        try:
            # Get the real path of the video device
            video_num = dev_path.split("/dev/video")[1]
            sysfs_path = Path(f"/sys/class/video4linux/video{video_num}/device")

            if not sysfs_path.exists():
                return None, None

            # Walk up to find USB device with idVendor/idProduct
            real_path = sysfs_path.resolve()
            current = real_path

            for _ in range(10):  # Limit traversal depth
                vid_path = current / "idVendor"
                pid_path = current / "idProduct"

                if vid_path.exists() and pid_path.exists():
                    vid = vid_path.read_text().strip().lower()
                    pid = pid_path.read_text().strip().lower()
                    return vid, pid

                parent = current.parent
                if parent == current:
                    break
                current = parent

        except Exception:
            pass

        return None, None

    def _get_formats(self, dev_path: str) -> list[str]:
        """Get supported video formats for a device."""
        formats = []

        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", dev_path, "--list-formats-ext"],
                capture_output=True,
                text=True,
                timeout=self.QUERY_TIMEOUT
            )

            if result.returncode == 0:
                current_format = ""
                for line in result.stdout.split("\n"):
                    fmt_match = re.search(r"'(\w+)'", line)
                    if fmt_match:
                        current_format = fmt_match.group(1)

                    size_match = re.search(r'Size: Discrete (\d+x\d+)', line)
                    if size_match and current_format:
                        formats.append(f"{current_format} {size_match.group(1)}")

        except subprocess.TimeoutExpired:
            # Device is unresponsive - mark as failed
            pass
        except FileNotFoundError:
            pass

        return formats


def detect_p2pro_cameras() -> tuple[Optional[VideoDevice], Optional[VideoDevice]]:
    """
    Convenience function to detect P2Pro cameras.
    Returns (thermal_device, visible_device) tuple.
    """
    detector = DeviceDetector()
    detector.scan()
    return detector.find_thermal_camera(), detector.find_visible_camera()


if __name__ == "__main__":
    # Test detection
    detector = DeviceDetector()
    devices = detector.scan()

    print("Detected video devices:")
    for dev in devices:
        print(f"  {dev.path}: {dev.name}")
        print(f"    USB: VID={dev.vid} PID={dev.pid}")
        print(f"    Formats: {dev.formats}")
        print(f"    Max res: {dev.max_resolution}")
        print(f"    Is Thermal: {dev.is_thermal}")
        print(f"    Is Visible: {dev.is_visible}")
        print()

    thermal, visible = detect_p2pro_cameras()
    print(f"Thermal camera: {thermal.path if thermal else 'Not found'}")
    print(f"Visible camera: {visible.path if visible else 'Not found'}")
