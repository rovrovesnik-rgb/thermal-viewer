"""
USB camera control module for P2Pro.
Sends vendor commands to control camera settings like emissivity, gain, and shutter.
"""

import usb.core
import usb.util
from typing import Optional
from dataclasses import dataclass
from enum import IntEnum
import struct

# Import logger (lazy to avoid circular imports)
_logger = None
def _get_logger():
    global _logger
    if _logger is None:
        try:
            from ui.console_log import get_logger
            _logger = get_logger()
        except ImportError:
            # Fallback if console_log not available
            class DummyLogger:
                def usb(self, msg): print(f"[USB] {msg}")
                def warn(self, msg): print(f"[WARN] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
            _logger = DummyLogger()
    return _logger


class GainMode(IntEnum):
    """Gain mode selection."""
    LOW = 0   # Wide temperature range, lower sensitivity
    HIGH = 1  # Narrow range, higher sensitivity
    # Note: Some firmware versions return other values (e.g., 17)
    # These are handled gracefully as "unknown"


@dataclass
class CameraInfo:
    """Camera device information."""
    serial_number: str
    sensor_temp: float  # Current sensor temperature
    shutter_temp: float  # Shutter temperature at last NUC


@dataclass
class ThermalParams:
    """Thermal measurement parameters."""
    emissivity: float      # 0.01 - 1.0
    distance: float        # meters (0 - 200)
    reflection_temp: float # Celsius
    atmospheric_temp: float # Celsius
    transmittance: float   # 0.01 - 1.0
    gain_mode: GainMode


class P2ProControl:
    """
    USB control interface for P2Pro thermal camera.

    Uses USB vendor commands to control camera settings.
    Based on reverse-engineered protocol from community projects.
    """

    # P2Pro USB IDs
    VID = 0x0bda
    PID = 0x5830

    # USB control transfer parameters
    CTRL_OUT = 0x41  # Vendor request, device-to-host
    CTRL_IN = 0xC1   # Vendor request, host-to-device
    REQ_WRITE = 0x45
    REQ_READ = 0x44

    # Command codes (from SDK reverse engineering)
    CMD_GET_DEVICE_INFO = 0x8405
    CMD_PSEUDO_COLOR = 0x8409
    CMD_TPD_PARAMS = 0x8514
    CMD_CURRENT_VOLTAGE = 0x8b0d
    CMD_PREVIEW_START = 0xc10f
    CMD_PREVIEW_STOP = 0x020f

    # Direction flags
    DIR_GET = 0x0000
    DIR_SET = 0x4000

    # TPD parameter indices (from SDK CommonParams.PropTPDParams)
    TPD_DISTANCE = 0
    TPD_TU = 1        # Reflection temperature
    TPD_TA = 2        # Atmospheric temperature
    TPD_EMS = 3       # Emissivity
    TPD_TAU = 4       # Transmittance
    TPD_GAIN_SEL = 5  # Gain selection

    # Shutter/NUC commands (from SDK IRCMD)
    # updateOOCOrB types: 0=OOC_UPDATE, 1=B_UPDATE, 2=OOC_B_UPDATE
    CMD_UPDATE_OOC_B = 0x8502  # updateOOCOrB command
    CMD_SHUTTER_MANUAL = 0x8501  # setShutterManualSwitch
    CMD_SHUTTER_STATUS = 0x8503  # setShutterStatus

    # Scaling factors from SDK (setEnvCorrectParams)
    # emissivity * 16384, transmittance * 16384, temp_K * 16
    SCALE_EMS = 16384
    SCALE_TAU = 16384
    SCALE_TEMP = 16

    def __init__(self):
        self.device: Optional[usb.core.Device] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to P2Pro camera via USB."""
        log = _get_logger()
        try:
            log.usb(f"Searching for P2Pro (VID={self.VID:04x}, PID={self.PID:04x})")
            self.device = usb.core.find(idVendor=self.VID, idProduct=self.PID)

            if self.device is None:
                log.warn("P2Pro USB device not found")
                return False

            # Try to get device info (may fail without permissions)
            try:
                name = f"{self.device.manufacturer} {self.device.product}"
            except (usb.core.USBError, ValueError):
                name = f"VID={self.VID:04x} PID={self.PID:04x}"
            log.usb(f"Found P2Pro: {name}")

            # Try to set configuration
            try:
                self.device.set_configuration()
                log.usb("USB configuration set")
            except usb.core.USBError as e:
                # May already be configured or permission denied
                if "permission" in str(e).lower() or "langid" in str(e).lower():
                    log.error("USB permission denied. Add udev rule or run: sudo chmod 666 /dev/bus/usb/...")
                    return False
                log.usb(f"Configuration note: {e}")

            self._connected = True
            log.usb("Connected to P2Pro USB control")
            return True

        except usb.core.USBError as e:
            if "permission" in str(e).lower() or "langid" in str(e).lower():
                log.error("USB permission denied. Run: sudo udevadm control --reload-rules && sudo udevadm trigger")
            else:
                log.error(f"USB error: {e}")
            return False
        except ValueError as e:
            # langid error from string descriptors
            log.error("USB permission issue - replug camera after setting udev rules")
            return False

    def disconnect(self):
        """Disconnect from camera."""
        if self.device:
            usb.util.dispose_resources(self.device)
            self.device = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.device is not None

    def _send_command(self, cmd: int, data: bytes = b'', direction: int = DIR_SET) -> Optional[bytes]:
        """Send a vendor command to the camera."""
        if not self.is_connected:
            return None

        try:
            wValue = cmd | direction
            wIndex = 0x200

            if direction == self.DIR_SET:
                # Write command
                self.device.ctrl_transfer(
                    self.CTRL_OUT, self.REQ_WRITE,
                    wValue, wIndex, data
                )
                return b''
            else:
                # Read command
                result = self.device.ctrl_transfer(
                    self.CTRL_IN, self.REQ_READ,
                    wValue, wIndex, len(data) if data else 64
                )
                return bytes(result)

        except usb.core.USBError as e:
            print(f"USB error: {e}")
            return None

    def trigger_nuc(self) -> bool:
        """
        Trigger Non-Uniformity Correction (shutter/FFC).
        This performs a flat-field calibration.

        Based on SDK: updateOOCOrB(B_UPDATE) for P2Pro.
        """
        log = _get_logger()
        if not self.is_connected:
            log.warn("Cannot trigger NUC: not connected")
            return False

        log.usb("Triggering NUC (shutter calibration)...")

        # SDK command: updateOOCOrB(B_UPDATE)
        # B_UPDATE = 1 in SDK enum UpdateOOCOrBType
        B_UPDATE = 1

        # Try the SDK-documented commands in order of likelihood
        commands_to_try = [
            # Primary: updateOOCOrB (SDK IRCMD method)
            # The command sends the update type as the data
            (self.CTRL_OUT, self.REQ_WRITE,
             self.CMD_UPDATE_OOC_B | self.DIR_SET, 0x200,
             struct.pack('<H', B_UPDATE), "updateOOCOrB(B_UPDATE)"),

            # Alternative: Direct shutter manual switch (close then open triggers NUC)
            (self.CTRL_OUT, self.REQ_WRITE,
             self.CMD_SHUTTER_MANUAL | self.DIR_SET, 0x200,
             struct.pack('<H', 0), "setShutterManual(CLOSE)"),

            # TPD param 6 (gain switch) sometimes triggers recal
            (self.CTRL_OUT, self.REQ_WRITE,
             self.CMD_TPD_PARAMS | self.DIR_SET, 0x200,
             struct.pack('<HH', 6, 1), "TPD GAIN trigger"),
        ]

        success = False
        for bmRequestType, bRequest, wValue, wIndex, data, desc in commands_to_try:
            try:
                log.usb(f"  Trying {desc}: wValue=0x{wValue:04x}, data={data.hex()}")
                self.device.ctrl_transfer(
                    bmRequestType, bRequest,
                    wValue, wIndex, data
                )
                log.usb(f"  {desc}: OK")
                success = True
                break  # Stop at first success - SDK only uses one command
            except usb.core.USBError as e:
                log.usb(f"  {desc}: failed ({e})")
                continue

        if success:
            log.usb("NUC triggered successfully")
        else:
            log.error("NUC trigger failed - all commands rejected")
        return success

    def set_emissivity(self, value: float) -> bool:
        """
        Set surface emissivity for temperature calculation.

        Based on SDK: setEnvCorrectParams uses emissivity * 16384

        Args:
            value: Emissivity value (0.01 - 1.0)

        Returns:
            True if successful
        """
        if not 0.01 <= value <= 1.0:
            raise ValueError("Emissivity must be between 0.01 and 1.0")

        # SDK format: emissivity * 16384 (so 1.0 = 16384, 0.95 = 15564)
        raw_value = int(value * self.SCALE_EMS)
        raw_value = max(164, min(16384, raw_value))  # 0.01 to 1.0

        return self._set_tpd_param(self.TPD_EMS, raw_value)

    def get_emissivity(self) -> Optional[float]:
        """Get current emissivity setting."""
        raw = self._get_tpd_param(self.TPD_EMS)
        if raw is not None:
            # SDK format: emissivity * 16384
            return min(1.0, raw / self.SCALE_EMS)
        return None

    def set_distance(self, meters: float) -> bool:
        """
        Set target distance for temperature calculation.

        Args:
            meters: Distance in meters (0 - 200)

        Returns:
            True if successful
        """
        if not 0 <= meters <= 200:
            raise ValueError("Distance must be between 0 and 200 meters")

        # Convert to camera format: 128 counts = 1 meter
        raw_value = int(meters * 128)
        raw_value = max(0, min(25600, raw_value))

        return self._set_tpd_param(self.TPD_DISTANCE, raw_value)

    def get_distance(self) -> Optional[float]:
        """Get current distance setting in meters."""
        raw = self._get_tpd_param(self.TPD_DISTANCE)
        if raw is not None:
            # Value at offset 14 appears to be in cm
            return raw / 100.0
        return None

    def set_gain_mode(self, mode: GainMode) -> bool:
        """
        Set gain mode (high/low sensitivity).

        High gain: More sensitive, narrower temperature range
        Low gain: Less sensitive, wider temperature range

        Args:
            mode: GainMode.HIGH or GainMode.LOW

        Returns:
            True if successful
        """
        return self._set_tpd_param(self.TPD_GAIN_SEL, int(mode))

    def get_gain_mode(self) -> Optional[GainMode]:
        """Get current gain mode."""
        raw = self._get_tpd_param(self.TPD_GAIN_SEL)
        if raw is not None:
            try:
                return GainMode(raw)
            except ValueError:
                # Unknown gain mode value - some firmware returns different values
                print(f"Unknown gain mode value: {raw}, treating as HIGH")
                return GainMode.HIGH
        return None

    def set_reflection_temp(self, celsius: float) -> bool:
        """
        Set reflection/background temperature.

        Based on SDK: setEnvCorrectParams uses temp_K * 16

        Args:
            celsius: Temperature in Celsius

        Returns:
            True if successful
        """
        # Convert to Kelvin * 16 (SDK format)
        kelvin = celsius + 273.15
        raw_value = int(kelvin * self.SCALE_TEMP)
        raw_value = max(3680, min(14400, raw_value))  # ~-43°C to ~627°C

        return self._set_tpd_param(self.TPD_TU, raw_value)

    def get_reflection_temp(self) -> Optional[float]:
        """Get reflection temperature in Celsius."""
        raw = self._get_tpd_param(self.TPD_TU)
        if raw is not None:
            # SDK format: temp_K * 16
            kelvin = raw / self.SCALE_TEMP
            return kelvin - 273.15
        return None

    def set_atmospheric_temp(self, celsius: float) -> bool:
        """
        Set atmospheric/ambient temperature.

        Based on SDK: setEnvCorrectParams uses temp_K * 16

        Args:
            celsius: Temperature in Celsius

        Returns:
            True if successful
        """
        # Convert to Kelvin * 16 (SDK format)
        kelvin = celsius + 273.15
        raw_value = int(kelvin * self.SCALE_TEMP)
        raw_value = max(3680, min(14400, raw_value))  # ~-43°C to ~627°C

        return self._set_tpd_param(self.TPD_TA, raw_value)

    def get_atmospheric_temp(self) -> Optional[float]:
        """Get atmospheric temperature in Celsius."""
        raw = self._get_tpd_param(self.TPD_TA)
        if raw is not None:
            # SDK format: temp_K * 16
            kelvin = raw / self.SCALE_TEMP
            return kelvin - 273.15
        return None

    def set_transmittance(self, value: float) -> bool:
        """
        Set atmospheric transmittance.

        Based on SDK: setEnvCorrectParams uses transmittance * 16384

        Args:
            value: Transmittance (0.01 - 1.0)

        Returns:
            True if successful
        """
        if not 0.01 <= value <= 1.0:
            raise ValueError("Transmittance must be between 0.01 and 1.0")

        # SDK format: transmittance * 16384 (same as emissivity)
        raw_value = int(value * self.SCALE_TAU)
        raw_value = max(164, min(16384, raw_value))

        return self._set_tpd_param(self.TPD_TAU, raw_value)

    def get_transmittance(self) -> Optional[float]:
        """Get atmospheric transmittance."""
        raw = self._get_tpd_param(self.TPD_TAU)
        if raw is not None:
            # SDK format: transmittance * 16384
            return min(1.0, raw / self.SCALE_TAU)
        return None

    def _set_tpd_param(self, param_index: int, value: int) -> bool:
        """Set a TPD (thermal parameter data) value."""
        log = _get_logger()
        param_names = {0: "distance", 1: "reflect_temp", 2: "ambient_temp",
                       3: "emissivity", 4: "transmittance", 5: "gain_mode"}
        param_name = param_names.get(param_index, f"param_{param_index}")

        if not self.is_connected:
            log.warn(f"Cannot set {param_name}: not connected")
            return False

        try:
            # Pack parameter: index + value
            data = struct.pack('<HH', param_index, value)

            log.usb(f"SET {param_name} = {value} (0x{value:04x})")

            self.device.ctrl_transfer(
                self.CTRL_OUT, self.REQ_WRITE,
                self.CMD_TPD_PARAMS | self.DIR_SET,
                0x200, data
            )

            log.usb(f"SET {param_name}: OK")
            return True

        except usb.core.USBError as e:
            log.error(f"SET {param_name} failed: {e}")
            return False

    def _get_tpd_param(self, param_index: int) -> Optional[int]:
        """Get a TPD parameter value."""
        log = _get_logger()
        param_names = {0: "distance", 1: "reflect_temp", 2: "ambient_temp",
                       3: "emissivity", 4: "transmittance", 5: "gain_mode"}
        param_name = param_names.get(param_index, f"param_{param_index}")

        if not self.is_connected:
            log.warn(f"Cannot get {param_name}: not connected")
            return None

        try:
            # Request 64-byte TPD structure
            result = self.device.ctrl_transfer(
                self.CTRL_IN, self.REQ_READ,
                self.CMD_TPD_PARAMS | self.DIR_GET,
                0x200, 64
            )

            if len(result) >= 16:
                # P2Pro returns a 64-byte structure
                # Known offsets (tentative, based on reverse engineering):
                # Offset 12-13: emissivity (scaled)
                # Offset 14-15: distance (in cm?)
                # This needs more research - for now, return default values
                raw_data = bytes(result)
                log.debug(f"TPD raw: {raw_data[:20].hex()}...")

                # Parse based on offset mapping (verified via raw data analysis)
                # Raw data structure (little-endian uint16 values):
                # Offset 4-5: emissivity * 16384
                # Offset 14-15: distance in cm
                # Offset 16-17: reflect_temp (K * 16)
                # Offset 18-19: ambient_temp (K * 16)
                # Offset 20-21: transmittance * 16384
                # Offset 6-7: gain_mode
                offset_map = {
                    3: 4,   # emissivity (TPD_EMS)
                    0: 14,  # distance (TPD_DISTANCE)
                    1: 16,  # reflect_temp (TPD_TU)
                    2: 18,  # ambient_temp (TPD_TA)
                    4: 20,  # transmittance (TPD_TAU)
                    5: 6,   # gain_mode (TPD_GAIN_SEL)
                }

                offset = offset_map.get(param_index, 0)
                if offset + 2 <= len(result):
                    value = struct.unpack('<H', raw_data[offset:offset+2])[0]
                    log.usb(f"GET {param_name} @ offset {offset} = {value} (0x{value:04x})")
                    return value
            else:
                log.warn(f"GET {param_name}: short response ({len(result)} bytes)")

        except usb.core.USBError as e:
            log.error(f"GET {param_name} failed: {e}")

        return None

    def get_raw_tpd_data(self) -> Optional[bytes]:
        """Get raw TPD data structure for debugging."""
        if not self.is_connected:
            return None
        try:
            result = self.device.ctrl_transfer(
                self.CTRL_IN, self.REQ_READ,
                self.CMD_TPD_PARAMS | self.DIR_GET,
                0x200, 64
            )
            return bytes(result)
        except usb.core.USBError:
            return None

    def get_params(self) -> Optional[ThermalParams]:
        """Get all thermal parameters."""
        try:
            emissivity = self.get_emissivity()
            distance = self.get_distance()
            reflection = self.get_reflection_temp()
            atmospheric = self.get_atmospheric_temp()
            transmittance = self.get_transmittance()
            gain = self.get_gain_mode()

            if all(v is not None for v in [emissivity, distance, reflection, atmospheric, transmittance, gain]):
                return ThermalParams(
                    emissivity=emissivity,
                    distance=distance,
                    reflection_temp=reflection,
                    atmospheric_temp=atmospheric,
                    transmittance=transmittance,
                    gain_mode=gain
                )
        except Exception as e:
            print(f"Error getting params: {e}")

        return None


# Singleton instance for easy access
_control_instance: Optional[P2ProControl] = None


def get_camera_control() -> P2ProControl:
    """Get the camera control singleton."""
    global _control_instance
    if _control_instance is None:
        _control_instance = P2ProControl()
    return _control_instance


if __name__ == "__main__":
    # Test camera control
    control = P2ProControl()

    if not control.connect():
        print("Failed to connect to P2Pro camera")
        print("Make sure the camera is connected and you have USB permissions")
        exit(1)

    print("Connected to P2Pro")

    # Try to read current settings
    params = control.get_params()
    if params:
        print(f"Current settings:")
        print(f"  Emissivity: {params.emissivity:.2f}")
        print(f"  Distance: {params.distance:.1f}m")
        print(f"  Reflection temp: {params.reflection_temp:.1f}°C")
        print(f"  Atmospheric temp: {params.atmospheric_temp:.1f}°C")
        print(f"  Transmittance: {params.transmittance:.2f}")
        print(f"  Gain mode: {'High' if params.gain_mode == GainMode.HIGH else 'Low'}")
    else:
        print("Could not read camera parameters")

    control.disconnect()
    print("Disconnected")
