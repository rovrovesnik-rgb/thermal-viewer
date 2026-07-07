"""
Main application window for P2Pro Thermal Camera Viewer.
"""

import sys
import time
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QStatusBar, QMessageBox, QSplitter,
    QApplication, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QAction, QKeySequence

from ui.thermal_view import ThermalView
from ui.control_panel import ControlPanel
from ui.camera_selector import select_cameras
from ui.roi_modal import ROIManager
from ui.console_log import ConsoleLogWindow, get_logger

from capture.thermal_capture import ThermalCapture, ThermalFrame
from capture.visible_capture import VisibleCapture
from capture.camera_control import P2ProControl, GainMode
from capture.device_detect import VideoDevice

from processing.colormap import ColormapManager, Palette, get_palette_by_name
from processing.contour import ContourOverlay, FusionMode, EdgeMethod, PipPosition
from processing.temperature import get_roi_stats, normalize_temperature
from processing.ai_upscale import get_ai_upscaler, get_model_by_name, AIUpscaleModel, MODEL_INFO

from tools.screenshot import get_screenshot_capture


class MainWindow(QMainWindow):
    """
    Main application window.

    Integrates:
    - Thermal and visible camera capture
    - Colormap application
    - Contour overlay
    - GPU scaling
    - ROI management
    - Camera control
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("P2Pro Thermal Camera Viewer")
        self.setMinimumSize(800, 600)

        # Components
        self._thermal_capture: ThermalCapture = None
        self._visible_capture: VisibleCapture = None
        self._camera_control = P2ProControl()

        self._colormap = ColormapManager()
        self._contour = ContourOverlay()
        self._ai_upscaler = get_ai_upscaler()
        self._roi_manager = ROIManager(self)
        self._screenshot = get_screenshot_capture()
        self._console_log = ConsoleLogWindow(self)
        self._logger = get_logger()

        # State
        self._thermal_device: VideoDevice = None
        self._visible_device: VideoDevice = None
        self._last_frame: ThermalFrame = None
        self._last_display_image: np.ndarray = None
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0
        self._thermal_rotation = 0  # 0, 90, 180, 270 degrees
        self._ai_upscale_enabled = False  # AI upscaling active

        # Settings
        self._settings = QSettings("P2ProViewer", "ThermalViewer")

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._connect_signals()

        # Load settings
        self._load_settings()

        # Update timer (30 FPS display refresh)
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(33)

        # Status update timer
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

    def _setup_ui(self):
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Thermal view (main area)
        self._thermal_view = ThermalView()
        splitter.addWidget(self._thermal_view)

        # Control panel (sidebar)
        self._control_panel = ControlPanel()
        splitter.addWidget(self._control_panel)

        # Set initial sizes (70% view, 30% controls)
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("No camera connected")
        self._status_bar.addWidget(self._status_label)

        self._camera_state_label = QLabel("")
        self._camera_state_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
        self._status_bar.addWidget(self._camera_state_label)

        self._fps_label = QLabel("0 FPS")
        self._status_bar.addPermanentWidget(self._fps_label)

        self._temp_label = QLabel("--.-°C")
        self._status_bar.addPermanentWidget(self._temp_label)

        # Update AI upscaler status
        self._control_panel.set_ai_upscale_available(
            self._ai_upscaler.available,
            self._ai_upscaler.gpu_name if self._ai_upscaler.gpu_available else ""
        )

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        select_cam_action = QAction("&Select Cameras...", self)
        select_cam_action.triggered.connect(self._select_cameras)
        file_menu.addAction(select_cam_action)

        file_menu.addSeparator()

        screenshot_action = QAction("&Screenshot", self)
        screenshot_action.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_action.triggered.connect(self._take_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        fullscreen_action = QAction("&Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        view_menu.addSeparator()

        crosshair_action = QAction("Show &Crosshair", self)
        crosshair_action.setCheckable(True)
        crosshair_action.setChecked(True)
        crosshair_action.triggered.connect(
            lambda checked: setattr(self._thermal_view, 'show_crosshair', checked)
        )
        view_menu.addAction(crosshair_action)

        minmax_action = QAction("Show &Min/Max", self)
        minmax_action.setCheckable(True)
        minmax_action.setChecked(True)
        minmax_action.triggered.connect(
            lambda checked: setattr(self._thermal_view, 'show_min_max', checked)
        )
        view_menu.addAction(minmax_action)

        view_menu.addSeparator()

        console_action = QAction("&Console Log...", self)
        console_action.setShortcut(QKeySequence("Ctrl+L"))
        console_action.triggered.connect(self._show_console_log)
        view_menu.addAction(console_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        pass  # Already handled in menu actions

    def _connect_signals(self):
        """Connect control panel signals to handlers."""
        cp = self._control_panel

        # Palette
        cp.palette_changed.connect(self._on_palette_changed)
        cp.palette_inverted.connect(self._on_palette_inverted)

        # Temperature range
        cp.temp_auto_range.connect(lambda v: setattr(self._colormap, 'auto_range', v))
        cp.temp_min.connect(lambda v: setattr(self._colormap, 'manual_min', v))
        cp.temp_max.connect(lambda v: setattr(self._colormap, 'manual_max', v))

        # Fusion (MSX-style overlay)
        cp.fusion_enabled.connect(lambda v: setattr(self._contour.settings, 'enabled', v))
        cp.fusion_mode.connect(self._on_fusion_mode_changed)
        cp.fusion_edge_method.connect(self._on_edge_method_changed)
        cp.fusion_edge_strength.connect(lambda v: setattr(self._contour.settings, 'edge_strength', v))
        cp.fusion_edge_detail.connect(lambda v: setattr(self._contour.settings, 'edge_detail', v))
        cp.fusion_edge_threshold.connect(lambda v: setattr(self._contour.settings, 'edge_threshold', v))
        cp.fusion_alpha.connect(lambda v: setattr(self._contour.settings, 'fusion_alpha', v))
        cp.fusion_zoom.connect(lambda v: setattr(self._contour.settings, 'zoom', v))
        cp.fusion_offset_x.connect(lambda v: setattr(self._contour.settings, 'offset_x', v))
        cp.fusion_offset_y.connect(lambda v: setattr(self._contour.settings, 'offset_y', v))
        cp.fusion_rotation.connect(lambda v: setattr(self._contour.settings, 'rotation', v))
        cp.fusion_denoise.connect(lambda v: setattr(self._contour.settings, 'denoise_strength', v))
        cp.fusion_temporal.connect(self._on_fusion_temporal_changed)
        cp.fusion_pip_position.connect(self._on_pip_position_changed)
        cp.fusion_pip_size.connect(lambda v: setattr(self._contour.settings, 'pip_size', v))

        # Legacy contour signals (for compatibility)
        cp.contour_enabled.connect(lambda v: setattr(self._contour.settings, 'enabled', v))

        # Camera
        cp.emissivity_changed.connect(self._on_emissivity_changed)
        cp.distance_changed.connect(self._on_distance_changed)
        cp.gain_mode_changed.connect(self._on_gain_changed)
        cp.reflection_temp_changed.connect(self._on_reflection_temp_changed)
        cp.atmospheric_temp_changed.connect(self._on_atmospheric_temp_changed)
        cp.trigger_nuc.connect(self._on_trigger_nuc)

        # Display
        cp.scale_factor_changed.connect(self._on_scale_changed)
        cp.thermal_rotation_changed.connect(self._on_thermal_rotation_changed)
        cp.ai_upscale_changed.connect(self._on_ai_upscale_changed)
        cp.ai_sharpen_changed.connect(lambda v: setattr(self._ai_upscaler, 'sharpen_strength', v))

        # Actions
        cp.screenshot_requested.connect(self._take_screenshot)

        # Thermal view
        self._thermal_view.roi_created.connect(self._on_roi_created)
        self._thermal_view.roi_selected.connect(self._on_roi_selected)

    def _load_settings(self):
        """Load settings from persistent storage."""
        # Window geometry
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Load contour settings
        self._load_contour_settings()

        # Load temperature range settings
        self._load_temp_range_settings()

        # Load display settings (rotation, AI upscale)
        self._load_display_settings()

        # Last used devices
        thermal_path = self._settings.value("thermal_device")
        visible_path = self._settings.value("visible_device")

        # Try to auto-connect if devices were saved
        if thermal_path:
            self._auto_connect(thermal_path, visible_path)

    def _load_fusion_settings(self):
        """Load fusion overlay settings from storage."""
        # Load all fusion settings
        enabled = self._settings.value("fusion/enabled", True, type=bool)
        mode = self._settings.value("fusion/mode", 2, type=int)  # Edge White default
        edge_strength = self._settings.value("fusion/edge_strength", 1.5, type=float)
        edge_detail = self._settings.value("fusion/edge_detail", 3, type=int)
        edge_threshold = self._settings.value("fusion/edge_threshold", 10, type=int)
        fusion_alpha = self._settings.value("fusion/alpha", 0.3, type=float)
        zoom = self._settings.value("fusion/zoom", 1.0, type=float)
        offset_x = self._settings.value("fusion/offset_x", 0, type=int)
        offset_y = self._settings.value("fusion/offset_y", 0, type=int)
        rotation = self._settings.value("fusion/rotation", 0.0, type=float)
        denoise = self._settings.value("fusion/denoise", 9, type=int)
        temporal = self._settings.value("fusion/temporal", 1, type=int)
        pip_position = self._settings.value("fusion/pip_position", 3, type=int)
        pip_size = self._settings.value("fusion/pip_size", 0.25, type=float)

        # Apply to processor
        self._contour.settings.enabled = enabled
        self._contour.settings.mode = FusionMode(mode)
        self._contour.settings.edge_strength = edge_strength
        self._contour.settings.edge_detail = edge_detail
        self._contour.settings.edge_threshold = edge_threshold
        self._contour.settings.fusion_alpha = fusion_alpha
        self._contour.settings.zoom = zoom
        self._contour.settings.offset_x = offset_x
        self._contour.settings.offset_y = offset_y
        self._contour.settings.rotation = rotation
        self._contour.settings.denoise_strength = denoise
        self._contour.settings.temporal_frames = temporal
        self._contour.settings.pip_position = PipPosition(pip_position)
        self._contour.settings.pip_size = pip_size

        # Update control panel
        self._control_panel.set_fusion_settings(
            enabled=enabled,
            mode=mode,
            edge_strength=edge_strength,
            edge_detail=edge_detail,
            edge_threshold=edge_threshold,
            fusion_alpha=fusion_alpha,
            zoom=zoom,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation=rotation,
            denoise=denoise,
            temporal=temporal,
            pip_position=pip_position,
            pip_size=pip_size
        )

    # Keep old method name for compatibility
    def _load_contour_settings(self):
        """Load contour overlay settings (legacy, redirects to fusion)."""
        self._load_fusion_settings()

    def _load_temp_range_settings(self):
        """Load temperature range settings from storage."""
        auto_range = self._settings.value("temp_range/auto", True, type=bool)
        min_temp = self._settings.value("temp_range/min", 15.0, type=float)
        max_temp = self._settings.value("temp_range/max", 40.0, type=float)

        # Apply to colormap manager
        self._colormap.auto_range = auto_range
        self._colormap.manual_min = min_temp
        self._colormap.manual_max = max_temp

        # Update control panel
        self._control_panel.set_temp_range_settings(
            auto_range=auto_range,
            min_temp=min_temp,
            max_temp=max_temp
        )

    def _load_display_settings(self):
        """Load display settings (rotation, AI upscale) from storage."""
        rotation = self._settings.value("display/rotation", 0, type=int)
        self._thermal_rotation = rotation
        self._control_panel.set_thermal_rotation(rotation)

        # Load AI upscale model
        ai_model_name = self._settings.value("display/ai_upscale", "Off", type=str)
        if ai_model_name != "Off" and self._ai_upscaler.available:
            model = get_model_by_name(ai_model_name)
            if self._ai_upscaler.load_model(model):
                self._ai_upscale_enabled = (model != AIUpscaleModel.OFF)
                self._control_panel.set_ai_upscale_model(ai_model_name)
                if self._ai_upscale_enabled:
                    info = MODEL_INFO.get(model, {})
                    self._control_panel.set_ai_upscale_status(f"Model: {info.get('display_name', ai_model_name)}")

    def _save_settings(self):
        """Save settings to persistent storage."""
        self._settings.setValue("geometry", self.saveGeometry())

        if self._thermal_device:
            self._settings.setValue("thermal_device", self._thermal_device.path)
        if self._visible_device:
            self._settings.setValue("visible_device", self._visible_device.path)

        # Save fusion settings
        self._settings.setValue("fusion/enabled", self._contour.settings.enabled)
        self._settings.setValue("fusion/mode", int(self._contour.settings.mode))
        self._settings.setValue("fusion/edge_strength", self._contour.settings.edge_strength)
        self._settings.setValue("fusion/edge_detail", self._contour.settings.edge_detail)
        self._settings.setValue("fusion/edge_threshold", self._contour.settings.edge_threshold)
        self._settings.setValue("fusion/alpha", self._contour.settings.fusion_alpha)
        self._settings.setValue("fusion/zoom", self._contour.settings.zoom)
        self._settings.setValue("fusion/offset_x", self._contour.settings.offset_x)
        self._settings.setValue("fusion/offset_y", self._contour.settings.offset_y)
        self._settings.setValue("fusion/rotation", self._contour.settings.rotation)
        self._settings.setValue("fusion/denoise", self._contour.settings.denoise_strength)
        self._settings.setValue("fusion/temporal", self._contour.settings.temporal_frames)
        self._settings.setValue("fusion/pip_position", int(self._contour.settings.pip_position))
        self._settings.setValue("fusion/pip_size", self._contour.settings.pip_size)

        # Save temperature range settings
        self._settings.setValue("temp_range/auto", self._colormap.auto_range)
        self._settings.setValue("temp_range/min", self._colormap.manual_min)
        self._settings.setValue("temp_range/max", self._colormap.manual_max)

        # Save display settings
        self._settings.setValue("display/rotation", self._thermal_rotation)

        # Save AI upscale model
        if self._ai_upscaler.current_model:
            info = MODEL_INFO.get(self._ai_upscaler.current_model, {})
            self._settings.setValue("display/ai_upscale", info.get("display_name", "Off"))
        else:
            self._settings.setValue("display/ai_upscale", "Off")

    def _auto_connect(self, thermal_path: str, visible_path: str = None):
        """Try to auto-connect to saved devices."""
        try:
            self._start_capture(thermal_path, visible_path)
        except Exception as e:
            print(f"Auto-connect failed: {e}")

    def _select_cameras(self):
        """Show camera selection dialog."""
        thermal, visible = select_cameras(self)

        if thermal:
            self._stop_capture()

            self._thermal_device = thermal
            self._visible_device = visible

            try:
                self._start_capture(
                    thermal.path,
                    visible.path if visible else None
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Camera Error",
                    f"Failed to start capture: {e}"
                )

    def _start_capture(self, thermal_path: str, visible_path: str = None):
        """Start camera capture."""
        self._logger.camera(f"Starting capture from {thermal_path}")

        # Thermal camera - start() will open it in its own thread
        self._thermal_capture = ThermalCapture(thermal_path)
        self._thermal_capture.start()
        self._logger.camera("Thermal capture started")

        # Visible camera (optional)
        if visible_path:
            self._logger.camera(f"Opening visible camera: {visible_path}")
            self._visible_capture = VisibleCapture(visible_path)
            if self._visible_capture.open():
                self._visible_capture.start()
                self._logger.camera("Visible capture started")
            else:
                self._logger.warn(f"Failed to open visible camera: {visible_path}")
                self._visible_capture = None

        # Try to connect USB control
        if self._camera_control.connect():
            self._load_camera_settings()
            self._logger.info("USB control connected, loaded camera settings")
            self._control_panel.set_usb_connected(True)
        else:
            self._logger.warn("USB control not available")
            self._control_panel.set_usb_connected(False)

        # Update visible camera availability status in control panel
        visible_available = self._visible_capture is not None
        self._control_panel.set_visible_camera_available(visible_available)
        if not visible_available:
            self._logger.info("Running in thermal-only mode (no visible camera)")

        self._status_label.setText(f"Connected: {thermal_path}")

    def _stop_capture(self):
        """Stop camera capture."""
        if self._thermal_capture:
            self._thermal_capture.stop()
            self._thermal_capture.close()
            self._thermal_capture = None

        if self._visible_capture:
            self._visible_capture.stop()
            self._visible_capture.close()
            self._visible_capture = None

        self._camera_control.disconnect()

        # Update UI to reflect no visible camera available
        self._control_panel.set_visible_camera_available(False)
        self._control_panel.set_usb_connected(False)

    def _handle_visible_camera_lost(self):
        """Handle visible camera disconnection during operation."""
        if self._visible_capture:
            self._logger.warn("Visible camera disconnected or stopped responding")
            self._visible_capture.stop()
            self._visible_capture.close()
            self._visible_capture = None

        # Update UI
        self._control_panel.set_visible_camera_available(False)
        self._logger.info("Switched to thermal-only mode")

    def _load_camera_settings(self):
        """Load current settings from camera."""
        self._logger.info("Loading camera settings...")
        params = self._camera_control.get_params()
        if params:
            self._logger.info(f"Camera params: emissivity={params.emissivity:.2f}, "
                             f"distance={params.distance:.1f}m, "
                             f"gain={'HIGH' if params.gain_mode == GainMode.HIGH else 'LOW'}")
            self._control_panel.set_emissivity(params.emissivity)
            self._control_panel.set_distance(params.distance)
            self._control_panel.set_gain_mode(params.gain_mode == GainMode.HIGH)
        else:
            self._logger.warn("Failed to load camera settings")

    def _update_display(self):
        """Update the thermal display."""
        if not self._thermal_capture or not self._thermal_capture.running:
            return

        frame = self._thermal_capture.last_frame
        if frame is None or frame is self._last_frame:
            return

        self._last_frame = frame
        self._frame_count += 1

        # Get temperature data (may be rotated)
        temp_data = frame.temperature_data
        if self._thermal_rotation != 0:
            temp_data = self._rotate_image(temp_data)

        # Normalize temperature data and apply colormap
        # This uses the colormap manager's range settings (auto or manual)
        colored = self._colormap.apply_to_temperature(temp_data)

        # Update the control panel with effective range (for display)
        self._control_panel.update_effective_range(
            self._colormap.effective_min,
            self._colormap.effective_max
        )

        # Apply contour overlay if enabled and visible camera available
        if self._contour.settings.enabled and self._visible_capture:
            visible_frame = self._visible_capture.last_frame
            # Check if visible frame is fresh (within 0.5 seconds)
            if visible_frame and (frame.timestamp - visible_frame.timestamp) < 0.5:
                # Rotate visible frame to match thermal rotation
                visible_gray = visible_frame.gray
                if self._thermal_rotation != 0:
                    visible_gray = self._rotate_image(visible_gray)
                colored = self._contour.process(visible_gray, colored)
            elif visible_frame:
                # Frame is stale - camera may have disconnected
                # Disable overlay and update UI
                self._handle_visible_camera_lost()

        # Scale for display (AI or Lanczos fallback)
        scale = self._thermal_view.scale_factor
        if scale > 1:
            if self._ai_upscale_enabled and self._ai_upscaler.current_model:
                # Use AI upscaling (always 4x, then resize if needed)
                colored = self._ai_upscaler.upscale(colored)
                # If scale != 4, resize to target
                if scale != 4:
                    target_h = int(temp_data.shape[0] * scale)
                    target_w = int(temp_data.shape[1] * scale)
                    colored = cv2.resize(colored, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                # Use Lanczos scaling (high quality fallback)
                target_h = int(temp_data.shape[0] * scale)
                target_w = int(temp_data.shape[1] * scale)
                colored = cv2.resize(colored, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        self._last_display_image = colored

        # Find min/max positions (in rotated coordinates)
        min_idx = np.unravel_index(
            np.argmin(temp_data),
            temp_data.shape
        )
        max_idx = np.unravel_index(
            np.argmax(temp_data),
            temp_data.shape
        )

        # Update thermal view (use rotated temp_data for hover readings)
        self._thermal_view.set_frame(
            colored,
            temp_data,
            float(np.min(temp_data)),
            float(np.max(temp_data)),
            frame.center_temp,
            (int(min_idx[1]), int(min_idx[0])),
            (int(max_idx[1]), int(max_idx[0]))
        )

        # Update ROI modals (pass rotated temp_data)
        self._update_roi_stats_with_data(temp_data, frame.timestamp)

    def _update_roi_stats(self, frame: ThermalFrame):
        """Update statistics for all ROIs (legacy, uses frame data)."""
        self._update_roi_stats_with_data(frame.temperature_data, frame.timestamp)

    def _update_roi_stats_with_data(self, temp_data: np.ndarray, timestamp: float):
        """Update statistics for all ROIs with given temperature data."""
        for i, roi in enumerate(self._thermal_view.get_rois()):
            stats = get_roi_stats(
                temp_data,
                roi.left(), roi.top(),
                roi.right(), roi.bottom()
            )
            self._roi_manager.update_roi_stats(i, stats, timestamp)

    def _update_status(self):
        """Update status bar."""
        # Calculate FPS
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = now

        self._fps_label.setText(f"{self._fps:.1f} FPS")

        # Update camera state indicator
        if self._thermal_capture:
            if self._thermal_capture.is_warming_up:
                progress = int(self._thermal_capture.warmup_progress * 100)
                self._camera_state_label.setText(f"WARMING UP ({progress}%)")
                self._camera_state_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
            elif self._thermal_capture.is_recalibrating:
                # Check if actively recalibrating or just showing post-recal status
                if self._thermal_capture.recalibration_active:
                    self._camera_state_label.setText("NUC IN PROGRESS...")
                    self._camera_state_label.setStyleSheet("color: #ff6600; font-weight: bold;")
                else:
                    self._camera_state_label.setText("NUC COMPLETE")
                    self._camera_state_label.setStyleSheet("color: #00ff00; font-weight: bold;")
            else:
                self._camera_state_label.setText("")
        else:
            self._camera_state_label.setText("")

        # Update temperature display
        if self._last_frame:
            self._temp_label.setText(f"Center: {self._last_frame.center_temp:.1f}°C")

    # Signal handlers
    def _on_palette_changed(self, name: str):
        self._colormap.palette = get_palette_by_name(name)

    def _on_palette_inverted(self, inverted: bool):
        self._colormap.invert = inverted

    def _on_fusion_mode_changed(self, mode: int):
        """Handle fusion mode change."""
        self._contour.settings.mode = FusionMode(mode)

    def _on_edge_method_changed(self, method: int):
        """Handle edge method change."""
        self._contour.settings.edge_method = EdgeMethod(method)

    def _on_fusion_temporal_changed(self, frames: int):
        """Handle temporal smoothing change."""
        self._contour.settings.temporal_frames = frames
        self._contour._processor.clear_history()

    def _on_pip_position_changed(self, position: int):
        """Handle PiP position change."""
        self._contour.settings.pip_position = PipPosition(position)

    # Legacy handler
    def _on_contour_temporal_changed(self, frames: int):
        self._on_fusion_temporal_changed(frames)

    def _on_emissivity_changed(self, value: float):
        if self._camera_control.is_connected:
            if self._camera_control.set_emissivity(value):
                self._logger.info(f"Emissivity set to {value:.2f}")
            else:
                self._logger.error(f"Failed to set emissivity to {value:.2f}")
        else:
            self._logger.warn(f"Cannot set emissivity: USB not connected")

    def _on_distance_changed(self, value: float):
        if self._camera_control.is_connected:
            if self._camera_control.set_distance(value):
                self._logger.info(f"Distance set to {value:.1f}m")
            else:
                self._logger.error(f"Failed to set distance to {value:.1f}m")
        else:
            self._logger.warn(f"Cannot set distance: USB not connected")

    def _on_gain_changed(self, high_gain: int):
        if self._camera_control.is_connected:
            mode = GainMode.HIGH if high_gain else GainMode.LOW
            if self._camera_control.set_gain_mode(mode):
                self._logger.info(f"Gain mode set to {'HIGH' if high_gain else 'LOW'}")
            else:
                self._logger.error(f"Failed to set gain mode")
        else:
            self._logger.warn(f"Cannot set gain: USB not connected")

    def _on_reflection_temp_changed(self, value: float):
        if self._camera_control.is_connected:
            if self._camera_control.set_reflection_temp(value):
                self._logger.info(f"Reflection temp set to {value:.1f}C")
            else:
                self._logger.error(f"Failed to set reflection temp")
        else:
            self._logger.warn(f"Cannot set reflection temp: USB not connected")

    def _on_atmospheric_temp_changed(self, value: float):
        if self._camera_control.is_connected:
            if self._camera_control.set_atmospheric_temp(value):
                self._logger.info(f"Atmospheric temp set to {value:.1f}C")
            else:
                self._logger.error(f"Failed to set atmospheric temp")
        else:
            self._logger.warn(f"Cannot set atmospheric temp: USB not connected")

    def _on_trigger_nuc(self):
        if self._camera_control.is_connected:
            if self._camera_control.trigger_nuc():
                self._status_bar.showMessage("NUC triggered", 2000)
            else:
                self._status_bar.showMessage("NUC trigger failed", 2000)

    def _on_scale_changed(self, factor: float):
        self._thermal_view.scale_factor = factor

    def _on_thermal_rotation_changed(self, degrees: int):
        """Handle thermal rotation change."""
        self._thermal_rotation = degrees

    def _on_ai_upscale_changed(self, model_name: str):
        """Handle AI upscale model change."""
        if not self._ai_upscaler.available:
            self._control_panel.set_ai_upscale_status("ncnn not available")
            return

        model = get_model_by_name(model_name)

        if model == AIUpscaleModel.OFF:
            self._ai_upscale_enabled = False
            self._ai_upscaler.load_model(AIUpscaleModel.OFF)
            self._control_panel.set_ai_upscale_status("Off")
            self._logger.info("AI upscaling disabled")
            return

        # Try to load the model
        self._control_panel.set_ai_upscale_status("Downloading/Loading...")

        if self._ai_upscaler.load_model(model):
            self._ai_upscale_enabled = True
            info = MODEL_INFO.get(model, {})
            self._control_panel.set_ai_upscale_status(f"Active: {info.get('display_name', model_name)}")
            self._logger.info(f"AI upscaling enabled: {model_name}")
        else:
            self._ai_upscale_enabled = False
            self._control_panel.set_ai_upscale_status("Failed to load model")
            self._logger.warn(f"Failed to load AI model: {model_name}")

    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate image by current thermal rotation setting."""
        if self._thermal_rotation == 0:
            return image
        elif self._thermal_rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self._thermal_rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self._thermal_rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _on_roi_created(self, rect):
        """Handle new ROI created."""
        pass  # ROI is automatically tracked by thermal view

    def _on_roi_selected(self, index: int):
        """Handle ROI selection (double-click to open modal)."""
        rois = self._thermal_view.get_rois()
        if 0 <= index < len(rois):
            roi = rois[index]
            self._roi_manager.open_modal(
                index,
                roi.left(), roi.top(),
                roi.right(), roi.bottom()
            )

    def _take_screenshot(self):
        """Take a screenshot."""
        if self._last_display_image is None:
            return

        temp_data = self._last_frame.temperature_data if self._last_frame else None

        filepath = self._screenshot.capture(
            self._last_display_image,
            temp_data
        )

        self._status_bar.showMessage(f"Screenshot saved: {filepath}", 3000)

    def _toggle_fullscreen(self, fullscreen: bool):
        if fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

    def _show_console_log(self):
        """Show the console log window."""
        self._console_log.show()
        self._console_log.raise_()
        self._console_log.activateWindow()

    def _show_about(self):
        QMessageBox.about(
            self,
            "About P2Pro Thermal Viewer",
            "P2Pro Thermal Camera Viewer\n\n"
            "A GUI application for Infiray P2Pro thermal cameras.\n\n"
            "Features:\n"
            "- Real-time thermal imaging\n"
            "- Visible light contour overlay\n"
            "- Multiple color palettes\n"
            "- Temperature measurement tools\n"
            "- GPU-accelerated scaling\n\n"
            "License: MIT"
        )

    def closeEvent(self, event):
        """Handle window close."""
        self._stop_capture()
        self._roi_manager.close_all()
        self._save_settings()
        event.accept()


def run_application():
    """Run the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 3px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #3c3c3c;
            border: 1px solid #555;
            padding: 3px;
        }
        QPushButton {
            background: #404040;
            border: 1px solid #555;
            padding: 5px 15px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background: #505050;
        }
        QPushButton:pressed {
            background: #606060;
        }
        QSlider::groove:horizontal {
            background: #404040;
            height: 6px;
        }
        QSlider::handle:horizontal {
            background: #888;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QScrollArea {
            border: none;
        }
        QStatusBar {
            background: #1e1e1e;
        }
    """)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_application())
