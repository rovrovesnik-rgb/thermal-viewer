"""
Camera selection dialog for choosing thermal and visible cameras.
Uses progressive device discovery - devices appear as soon as they're scanned.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject

from capture.device_detect import DeviceDetector, VideoDevice


class DeviceSignals(QObject):
    """Signals for cross-thread device updates."""
    device_found = pyqtSignal(object)  # VideoDevice
    scan_complete = pyqtSignal()


class CameraSelectorDialog(QDialog):
    """
    Dialog for selecting thermal and visible camera devices.
    Auto-detects devices progressively - fast cameras appear immediately.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Camera Selection")
        self.setMinimumWidth(450)

        self._detector = DeviceDetector()
        self._devices: list[VideoDevice] = []
        self._signals = DeviceSignals()
        self._scanning = False

        self._thermal_device: VideoDevice = None
        self._visible_device: VideoDevice = None

        # Track auto-selection
        self._auto_thermal_selected = False
        self._auto_visible_selected = False

        self._setup_ui()

        # Connect signals for thread-safe UI updates
        self._signals.device_found.connect(self._on_device_found)
        self._signals.scan_complete.connect(self._on_scan_complete)

        # Start scanning
        self._scan_devices_async()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Thermal camera group
        thermal_group = QGroupBox("Thermal Camera")
        thermal_layout = QFormLayout(thermal_group)

        self._thermal_combo = QComboBox()
        self._thermal_combo.currentIndexChanged.connect(self._on_thermal_changed)
        thermal_layout.addRow("Device:", self._thermal_combo)

        self._thermal_info = QLabel("Not selected")
        self._thermal_info.setStyleSheet("color: gray;")
        thermal_layout.addRow("Info:", self._thermal_info)

        layout.addWidget(thermal_group)

        # Visible camera group
        visible_group = QGroupBox("Visible Light Camera")
        visible_layout = QFormLayout(visible_group)

        self._visible_combo = QComboBox()
        self._visible_combo.currentIndexChanged.connect(self._on_visible_changed)
        visible_layout.addRow("Device:", self._visible_combo)

        self._visible_info = QLabel("Not selected")
        self._visible_info.setStyleSheet("color: gray;")
        visible_layout.addRow("Info:", self._visible_info)

        layout.addWidget(visible_group)

        # Status/scan button row
        status_row = QHBoxLayout()
        self._scan_status = QLabel("")
        self._scan_status.setStyleSheet("color: #888; font-size: 11px;")
        status_row.addWidget(self._scan_status)
        status_row.addStretch()

        scan_btn = QPushButton("Rescan")
        scan_btn.clicked.connect(self._scan_devices_async)
        status_row.addWidget(scan_btn)

        layout.addLayout(status_row)

        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        # Initialize combos with None option
        self._thermal_combo.addItem("(None)", None)
        self._visible_combo.addItem("(None)", None)

    def _scan_devices_async(self):
        """Start scanning for devices in background."""
        if self._scanning:
            return

        self._scanning = True
        self._devices = []
        self._auto_thermal_selected = False
        self._auto_visible_selected = False

        # Clear combos but keep None option
        self._thermal_combo.clear()
        self._visible_combo.clear()
        self._thermal_combo.addItem("(None)", None)
        self._visible_combo.addItem("(None)", None)

        self._scan_status.setText("Scanning devices...")
        self._scan_status.setStyleSheet("color: #888; font-size: 11px;")

        # Start async scan - devices will be emitted as discovered
        self._detector.scan_async(
            on_device=lambda dev: self._signals.device_found.emit(dev),
            on_complete=lambda: self._signals.scan_complete.emit()
        )

    def _on_device_found(self, device: VideoDevice):
        """Handle device discovery (called on main thread via signal)."""
        self._devices.append(device)

        # Add to combos
        label = f"{device.path} - {device.name}"
        if device.scan_failed:
            label += " (timeout)"

        self._thermal_combo.addItem(label, device)
        self._visible_combo.addItem(label, device)

        # Auto-select if this is a P2Pro camera and not already selected
        if device.is_thermal and not self._auto_thermal_selected:
            for i in range(self._thermal_combo.count()):
                if self._thermal_combo.itemData(i) == device:
                    self._thermal_combo.setCurrentIndex(i)
                    self._auto_thermal_selected = True
                    break

        if device.is_visible and not self._auto_visible_selected:
            for i in range(self._visible_combo.count()):
                if self._visible_combo.itemData(i) == device:
                    self._visible_combo.setCurrentIndex(i)
                    self._auto_visible_selected = True
                    break

        # Update status
        self._scan_status.setText(f"Found {len(self._devices)} device(s)...")

    def _on_scan_complete(self):
        """Handle scan completion."""
        self._scanning = False

        count = len(self._devices)
        failed = sum(1 for d in self._devices if d.scan_failed)

        if failed > 0:
            self._scan_status.setText(f"Found {count} device(s), {failed} unresponsive")
            self._scan_status.setStyleSheet("color: #a80; font-size: 11px;")
        else:
            self._scan_status.setText(f"Found {count} device(s)")
            self._scan_status.setStyleSheet("color: #080; font-size: 11px;")

        # If no thermal found, show warning
        if not self._auto_thermal_selected:
            self._thermal_info.setText("No P2Pro thermal camera detected")
            self._thermal_info.setStyleSheet("color: orange;")

    def _on_thermal_changed(self, index: int):
        """Handle thermal camera selection change."""
        device = self._thermal_combo.itemData(index)
        self._thermal_device = device

        if device:
            info_parts = []
            if device.vid and device.pid:
                info_parts.append(f"USB: {device.vid}:{device.pid}")
            if device.formats:
                # Show first 3 formats
                info_parts.append(f"Formats: {', '.join(device.formats[:3])}")
            if device.is_thermal:
                info_parts.append("P2Pro thermal detected")

            self._thermal_info.setText("\n".join(info_parts) if info_parts else "No info")
            self._thermal_info.setStyleSheet("color: green;" if device.is_thermal else "color: orange;")
        else:
            self._thermal_info.setText("Not selected")
            self._thermal_info.setStyleSheet("color: gray;")

    def _on_visible_changed(self, index: int):
        """Handle visible camera selection change."""
        device = self._visible_combo.itemData(index)
        self._visible_device = device

        if device:
            info_parts = []
            if device.vid and device.pid:
                info_parts.append(f"USB: {device.vid}:{device.pid}")
            if device.formats:
                info_parts.append(f"Max: {device.max_resolution[0]}x{device.max_resolution[1]}")
            if device.is_visible:
                info_parts.append("P2Pro visible detected")
            elif device.max_resolution[0] >= 1280:
                info_parts.append("HD camera (not P2Pro)")

            self._visible_info.setText("\n".join(info_parts) if info_parts else "No info")
            if device.is_visible:
                self._visible_info.setStyleSheet("color: green;")
            elif device.max_resolution[0] >= 1280:
                self._visible_info.setStyleSheet("color: #a80;")  # Orange for HD
            else:
                self._visible_info.setStyleSheet("color: orange;")
        else:
            self._visible_info.setText("Not selected (optional)")
            self._visible_info.setStyleSheet("color: gray;")

    def _on_accept(self):
        """Validate and accept selection."""
        if not self._thermal_device:
            QMessageBox.warning(
                self,
                "No Thermal Camera",
                "Please select a thermal camera device."
            )
            return

        self.accept()

    @property
    def thermal_device(self) -> VideoDevice:
        """Get selected thermal device."""
        return self._thermal_device

    @property
    def visible_device(self) -> VideoDevice:
        """Get selected visible device (may be None)."""
        return self._visible_device


def select_cameras(parent=None) -> tuple[VideoDevice, VideoDevice]:
    """
    Show camera selection dialog and return selected devices.

    Returns:
        (thermal_device, visible_device) tuple
        Returns (None, None) if cancelled
    """
    dialog = CameraSelectorDialog(parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.thermal_device, dialog.visible_device
    return None, None
