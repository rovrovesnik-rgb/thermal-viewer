"""
Collapsible control panel sidebar with all camera and display settings.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSlider, QComboBox, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from processing.colormap import Palette, get_palette_names
from processing.contour import FusionMode, EdgeMethod, PipPosition
from processing.ai_upscale import get_model_names as get_ai_model_names


class CollapsibleSection(QWidget):
    """A collapsible section widget."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        self._is_collapsed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header button
        self._header = QPushButton(f"▼ {title}")
        self._header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                background: #404040;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #505050;
            }
        """)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        # Content widget
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(5, 5, 5, 5)
        self._content_layout.setSpacing(5)
        layout.addWidget(self._content)

        self._title = title

    def add_widget(self, widget: QWidget):
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area."""
        self._content_layout.addLayout(layout)

    def _toggle(self):
        """Toggle collapsed state."""
        self._is_collapsed = not self._is_collapsed
        self._content.setVisible(not self._is_collapsed)
        arrow = "▶" if self._is_collapsed else "▼"
        self._header.setText(f"{arrow} {self._title}")


class ControlPanel(QWidget):
    """
    Control panel sidebar with all settings.

    Sections:
    - Palette selection
    - Fusion/overlay settings (MSX-style)
    - Camera settings (emissivity, gain, etc.)
    - Display settings (scale, GPU)
    """

    # Signals for settings changes
    palette_changed = pyqtSignal(str)
    palette_inverted = pyqtSignal(bool)

    # Temperature range signals
    temp_auto_range = pyqtSignal(bool)
    temp_min = pyqtSignal(float)
    temp_max = pyqtSignal(float)

    # Fusion signals (new MSX-style)
    fusion_enabled = pyqtSignal(bool)
    fusion_mode = pyqtSignal(int)  # FusionMode value
    fusion_edge_method = pyqtSignal(int)  # EdgeMethod value
    fusion_edge_strength = pyqtSignal(float)
    fusion_edge_detail = pyqtSignal(int)   # Kernel size: 3, 5, 7
    fusion_edge_threshold = pyqtSignal(int)  # Threshold: 0-50
    fusion_alpha = pyqtSignal(float)
    fusion_zoom = pyqtSignal(float)
    fusion_offset_x = pyqtSignal(int)
    fusion_offset_y = pyqtSignal(int)
    fusion_rotation = pyqtSignal(float)
    fusion_denoise = pyqtSignal(int)
    fusion_temporal = pyqtSignal(int)
    fusion_pip_position = pyqtSignal(int)
    fusion_pip_size = pyqtSignal(float)

    # Legacy signals for compatibility
    contour_enabled = pyqtSignal(bool)
    contour_zoom = pyqtSignal(float)
    contour_offset_x = pyqtSignal(int)
    contour_offset_y = pyqtSignal(int)
    contour_rotation = pyqtSignal(float)
    contour_threshold_low = pyqtSignal(int)
    contour_threshold_high = pyqtSignal(int)
    contour_opacity = pyqtSignal(float)
    contour_blur = pyqtSignal(int)
    contour_temporal_frames = pyqtSignal(int)
    contour_update_rate = pyqtSignal(int)
    contour_color_mode = pyqtSignal(str)
    contour_custom_color = pyqtSignal(tuple)
    contour_preset_changed = pyqtSignal(str)

    emissivity_changed = pyqtSignal(float)
    distance_changed = pyqtSignal(float)
    gain_mode_changed = pyqtSignal(int)
    reflection_temp_changed = pyqtSignal(float)
    atmospheric_temp_changed = pyqtSignal(float)
    trigger_nuc = pyqtSignal()

    scale_factor_changed = pyqtSignal(float)
    thermal_rotation_changed = pyqtSignal(int)  # 0, 90, 180, 270 degrees
    ai_upscale_changed = pyqtSignal(str)  # Model name
    ai_sharpen_changed = pyqtSignal(float)  # Sharpening strength 0-1

    screenshot_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumWidth(250)
        self.setMaximumWidth(350)

        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)

        # Add sections
        self._create_palette_section(layout)
        self._create_fusion_section(layout)
        self._create_camera_section(layout)
        self._create_display_section(layout)
        self._create_actions_section(layout)

        # Spacer at bottom
        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _create_palette_section(self, parent_layout):
        """Create palette selection section."""
        section = CollapsibleSection("Palette")

        # Palette dropdown
        row = QHBoxLayout()
        row.addWidget(QLabel("Color Map:"))
        self._palette_combo = QComboBox()
        self._palette_combo.addItems(get_palette_names())
        self._palette_combo.currentTextChanged.connect(self.palette_changed.emit)
        row.addWidget(self._palette_combo)
        section.add_layout(row)

        # Invert checkbox
        self._invert_check = QCheckBox("Invert Palette")
        self._invert_check.toggled.connect(self.palette_inverted.emit)
        section.add_widget(self._invert_check)

        # Temperature range controls
        section.add_widget(QLabel(""))  # Spacer

        # Auto-range checkbox
        self._auto_range_check = QCheckBox("Auto Range")
        self._auto_range_check.setChecked(True)
        self._auto_range_check.setToolTip(
            "When enabled, color scale adjusts to each frame's min/max.\n"
            "Disable for manual control of temperature thresholds."
        )
        self._auto_range_check.toggled.connect(self._on_auto_range_changed)
        section.add_widget(self._auto_range_check)

        # Min temperature slider
        row = QHBoxLayout()
        row.addWidget(QLabel("Min Temp:"))
        self._temp_min_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_min_slider.setRange(-20, 100)  # -20°C to 100°C
        self._temp_min_slider.setValue(15)
        self._temp_min_slider.setEnabled(False)  # Disabled when auto-range
        self._temp_min_slider.setToolTip(
            "Minimum temperature (cold end of color scale).\n"
            "Anything below this shows as coldest color."
        )
        self._temp_min_slider.valueChanged.connect(self._on_temp_min_changed)
        row.addWidget(self._temp_min_slider)
        self._temp_min_label = QLabel("15°C")
        self._temp_min_label.setMinimumWidth(45)
        row.addWidget(self._temp_min_label)
        section.add_layout(row)

        # Max temperature slider
        row = QHBoxLayout()
        row.addWidget(QLabel("Max Temp:"))
        self._temp_max_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_max_slider.setRange(-20, 150)  # -20°C to 150°C
        self._temp_max_slider.setValue(40)
        self._temp_max_slider.setEnabled(False)  # Disabled when auto-range
        self._temp_max_slider.setToolTip(
            "Maximum temperature (hot end of color scale).\n"
            "Anything above this shows as hottest color."
        )
        self._temp_max_slider.valueChanged.connect(self._on_temp_max_changed)
        row.addWidget(self._temp_max_slider)
        self._temp_max_label = QLabel("40°C")
        self._temp_max_label.setMinimumWidth(45)
        row.addWidget(self._temp_max_label)
        section.add_layout(row)

        parent_layout.addWidget(section)

    def _create_fusion_section(self, parent_layout):
        """Create MSX-style fusion settings section."""
        section = CollapsibleSection("Fusion Overlay (MSX)")

        # Enable checkbox
        self._fusion_enable = QCheckBox("Enable Fusion")
        self._fusion_enable.setChecked(True)
        self._fusion_enable.toggled.connect(self._on_fusion_enabled)
        section.add_widget(self._fusion_enable)

        # Fusion mode dropdown
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self._fusion_mode_combo = QComboBox()
        self._fusion_mode_combo.addItems([
            "Thermal Only",       # 0
            "Visible Only",       # 1
            "Edge White (MSX)",   # 2
            "Fusion Blend",       # 3
            "Picture-in-Picture", # 4
            "Edge Black",         # 5
        ])
        self._fusion_mode_combo.setCurrentIndex(2)  # Default to Edge White
        self._fusion_mode_combo.setToolTip(
            "Fusion mode (like Android app):\n"
            "• Thermal Only: No overlay\n"
            "• Visible Only: Show visible camera\n"
            "• Edge White: White edges on thermal (MSX default)\n"
            "• Fusion Blend: Alpha blend thermal + visible\n"
            "• Picture-in-Picture: Visible thumbnail\n"
            "• Edge Black: Black edges on thermal"
        )
        self._fusion_mode_combo.currentIndexChanged.connect(self._on_fusion_mode_changed)
        row.addWidget(self._fusion_mode_combo)
        section.add_layout(row)

        # Edge method dropdown
        row = QHBoxLayout()
        row.addWidget(QLabel("Edge Method:"))
        self._edge_method_combo = QComboBox()
        self._edge_method_combo.addItems([
            "Laplacian",    # 0 - Recommended
            "Sobel",        # 1
            "High-Pass",    # 2
        ])
        self._edge_method_combo.setCurrentIndex(0)
        self._edge_method_combo.setToolTip(
            "Edge detection algorithm:\n"
            "• Laplacian: Best for MSX-style (recommended)\n"
            "• Sobel: Directional gradient, sharper\n"
            "• High-Pass: Simplest, subtle edges"
        )
        self._edge_method_combo.currentIndexChanged.connect(
            lambda i: self.fusion_edge_method.emit(i)
        )
        row.addWidget(self._edge_method_combo)
        section.add_layout(row)

        # Edge strength slider
        row = QHBoxLayout()
        row.addWidget(QLabel("Edge Strength:"))
        self._edge_strength = QSlider(Qt.Orientation.Horizontal)
        self._edge_strength.setRange(5, 30)  # 0.5 - 3.0
        self._edge_strength.setValue(15)     # 1.5 default
        self._edge_strength.setToolTip(
            "Edge visibility strength (0.5 - 3.0)\n"
            "Higher = more visible edges"
        )
        self._edge_strength.valueChanged.connect(
            lambda v: self.fusion_edge_strength.emit(v / 10.0)
        )
        row.addWidget(self._edge_strength)
        self._edge_strength_label = QLabel("1.5")
        self._edge_strength.valueChanged.connect(
            lambda v: self._edge_strength_label.setText(f"{v/10:.1f}")
        )
        row.addWidget(self._edge_strength_label)
        section.add_layout(row)

        # Edge detail (kernel size) dropdown
        row = QHBoxLayout()
        row.addWidget(QLabel("Edge Detail:"))
        self._edge_detail_combo = QComboBox()
        self._edge_detail_combo.addItems([
            "Fine (1)",      # ksize=1 - sharpest
            "Medium (3)",    # ksize=3
            "Coarse (5)",    # ksize=5 - thicker lines
        ])
        self._edge_detail_combo.setCurrentIndex(0)  # Default to fine
        self._edge_detail_combo.setToolTip(
            "Edge detection kernel size:\n"
            "• Fine (1): Sharpest, finest details\n"
            "• Medium (3): Balanced line weight\n"
            "• Coarse (5): Thicker lines, bolder edges"
        )
        self._edge_detail_combo.currentIndexChanged.connect(
            lambda i: self.fusion_edge_detail.emit([1, 3, 5][i])
        )
        row.addWidget(self._edge_detail_combo)
        section.add_layout(row)

        # Edge threshold slider
        row = QHBoxLayout()
        row.addWidget(QLabel("Edge Threshold:"))
        self._edge_threshold = QSlider(Qt.Orientation.Horizontal)
        self._edge_threshold.setRange(0, 50)
        self._edge_threshold.setValue(10)  # Default
        self._edge_threshold.setToolTip(
            "Minimum edge intensity (0-50)\n"
            "Higher = cleaner edges, removes noise\n"
            "Lower = more detail, may be noisier"
        )
        self._edge_threshold.valueChanged.connect(self.fusion_edge_threshold.emit)
        row.addWidget(self._edge_threshold)
        self._edge_threshold_label = QLabel("10")
        self._edge_threshold.valueChanged.connect(
            lambda v: self._edge_threshold_label.setText(str(v))
        )
        row.addWidget(self._edge_threshold_label)
        section.add_layout(row)

        # Fusion alpha slider (for FUSION_BLEND mode)
        row = QHBoxLayout()
        row.addWidget(QLabel("Blend Alpha:"))
        self._fusion_alpha = QSlider(Qt.Orientation.Horizontal)
        self._fusion_alpha.setRange(0, 100)
        self._fusion_alpha.setValue(30)  # 0.3 default
        self._fusion_alpha.setToolTip(
            "Blend amount for Fusion Blend mode (0-100%)\n"
            "0% = thermal only, 100% = visible only"
        )
        self._fusion_alpha.valueChanged.connect(
            lambda v: self.fusion_alpha.emit(v / 100.0)
        )
        row.addWidget(self._fusion_alpha)
        self._fusion_alpha_label = QLabel("30%")
        self._fusion_alpha.valueChanged.connect(
            lambda v: self._fusion_alpha_label.setText(f"{v}%")
        )
        row.addWidget(self._fusion_alpha_label)
        section.add_layout(row)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        section.add_widget(sep)

        # Alignment section label
        align_label = QLabel("Alignment:")
        align_label.setStyleSheet("font-weight: bold;")
        section.add_widget(align_label)

        # Zoom
        row = QHBoxLayout()
        row.addWidget(QLabel("Zoom:"))
        self._fusion_zoom = QDoubleSpinBox()
        self._fusion_zoom.setRange(0.5, 3.0)
        self._fusion_zoom.setSingleStep(0.05)
        self._fusion_zoom.setValue(1.0)
        self._fusion_zoom.setToolTip("Zoom factor for visible camera alignment")
        self._fusion_zoom.valueChanged.connect(self.fusion_zoom.emit)
        row.addWidget(self._fusion_zoom)
        section.add_layout(row)

        # Offset X
        row = QHBoxLayout()
        row.addWidget(QLabel("Offset X:"))
        self._fusion_offset_x = QSpinBox()
        self._fusion_offset_x.setRange(-500, 500)
        self._fusion_offset_x.setValue(0)
        self._fusion_offset_x.setToolTip("Horizontal offset (negative=left, positive=right)")
        self._fusion_offset_x.valueChanged.connect(self.fusion_offset_x.emit)
        row.addWidget(self._fusion_offset_x)
        section.add_layout(row)

        # Offset Y
        row = QHBoxLayout()
        row.addWidget(QLabel("Offset Y:"))
        self._fusion_offset_y = QSpinBox()
        self._fusion_offset_y.setRange(-500, 500)
        self._fusion_offset_y.setValue(0)
        self._fusion_offset_y.setToolTip("Vertical offset (negative=up, positive=down)")
        self._fusion_offset_y.valueChanged.connect(self.fusion_offset_y.emit)
        row.addWidget(self._fusion_offset_y)
        section.add_layout(row)

        # Rotation
        row = QHBoxLayout()
        row.addWidget(QLabel("Rotation:"))
        self._fusion_rotation = QDoubleSpinBox()
        self._fusion_rotation.setRange(-15.0, 15.0)
        self._fusion_rotation.setSingleStep(0.5)
        self._fusion_rotation.setValue(0.0)
        self._fusion_rotation.setToolTip("Rotation in degrees")
        self._fusion_rotation.valueChanged.connect(self.fusion_rotation.emit)
        row.addWidget(self._fusion_rotation)
        section.add_layout(row)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        section.add_widget(sep2)

        # Advanced section label
        adv_label = QLabel("Advanced:")
        adv_label.setStyleSheet("font-weight: bold;")
        section.add_widget(adv_label)

        # Denoise strength
        row = QHBoxLayout()
        row.addWidget(QLabel("Denoise:"))
        self._fusion_denoise = QSpinBox()
        self._fusion_denoise.setRange(5, 15)
        self._fusion_denoise.setValue(9)
        self._fusion_denoise.setToolTip(
            "Bilateral filter strength (5-15)\n"
            "Higher = smoother edges, less noise"
        )
        self._fusion_denoise.valueChanged.connect(self.fusion_denoise.emit)
        row.addWidget(self._fusion_denoise)
        section.add_layout(row)

        # Temporal smoothing
        row = QHBoxLayout()
        row.addWidget(QLabel("Smoothing:"))
        self._fusion_temporal = QSpinBox()
        self._fusion_temporal.setRange(1, 5)
        self._fusion_temporal.setValue(1)
        self._fusion_temporal.setToolTip(
            "Temporal frame averaging (1=off)\n"
            "Higher = smoother but more lag"
        )
        self._fusion_temporal.valueChanged.connect(self.fusion_temporal.emit)
        row.addWidget(self._fusion_temporal)
        section.add_layout(row)

        # PiP position (for PiP mode)
        row = QHBoxLayout()
        row.addWidget(QLabel("PiP Position:"))
        self._pip_position_combo = QComboBox()
        self._pip_position_combo.addItems([
            "Top Left",
            "Top Right",
            "Bottom Left",
            "Bottom Right",
        ])
        self._pip_position_combo.setCurrentIndex(3)  # Bottom right
        self._pip_position_combo.setToolTip("Picture-in-Picture corner position")
        self._pip_position_combo.currentIndexChanged.connect(
            lambda i: self.fusion_pip_position.emit(i)
        )
        row.addWidget(self._pip_position_combo)
        section.add_layout(row)

        # PiP size
        row = QHBoxLayout()
        row.addWidget(QLabel("PiP Size:"))
        self._pip_size = QSlider(Qt.Orientation.Horizontal)
        self._pip_size.setRange(15, 50)  # 15% - 50%
        self._pip_size.setValue(25)
        self._pip_size.setToolTip("Picture-in-Picture size (15-50% of image)")
        self._pip_size.valueChanged.connect(
            lambda v: self.fusion_pip_size.emit(v / 100.0)
        )
        row.addWidget(self._pip_size)
        self._pip_size_label = QLabel("25%")
        self._pip_size.valueChanged.connect(
            lambda v: self._pip_size_label.setText(f"{v}%")
        )
        row.addWidget(self._pip_size_label)
        section.add_layout(row)

        parent_layout.addWidget(section)

    def _on_fusion_enabled(self, enabled: bool):
        """Handle fusion enable/disable."""
        self.fusion_enabled.emit(enabled)
        self.contour_enabled.emit(enabled)  # Legacy

    def _on_auto_range_changed(self, checked: bool):
        """Handle auto-range checkbox change."""
        self._temp_min_slider.setEnabled(not checked)
        self._temp_max_slider.setEnabled(not checked)
        self.temp_auto_range.emit(checked)

    def _on_temp_min_changed(self, value: int):
        """Handle minimum temperature slider change."""
        # Ensure min < max
        if value >= self._temp_max_slider.value():
            self._temp_max_slider.setValue(value + 1)
        self._temp_min_label.setText(f"{value}°C")
        self.temp_min.emit(float(value))

    def _on_temp_max_changed(self, value: int):
        """Handle maximum temperature slider change."""
        # Ensure max > min
        if value <= self._temp_min_slider.value():
            self._temp_min_slider.setValue(value - 1)
        self._temp_max_label.setText(f"{value}°C")
        self.temp_max.emit(float(value))

    def _on_fusion_mode_changed(self, index: int):
        """Handle fusion mode change."""
        self.fusion_mode.emit(index)

        # Show/hide relevant controls based on mode
        is_edge_mode = index in [2, 5]  # Edge White, Edge Black
        is_blend_mode = index == 3       # Fusion Blend
        is_pip_mode = index == 4         # Picture-in-Picture

        self._edge_method_combo.setEnabled(is_edge_mode)
        self._edge_strength.setEnabled(is_edge_mode)
        self._edge_detail_combo.setEnabled(is_edge_mode)
        self._edge_threshold.setEnabled(is_edge_mode)
        self._fusion_alpha.setEnabled(is_blend_mode)
        self._pip_position_combo.setEnabled(is_pip_mode)
        self._pip_size.setEnabled(is_pip_mode)

    def _create_camera_section(self, parent_layout):
        """Create camera settings section."""
        section = CollapsibleSection("Camera Settings")

        # USB status indicator
        self._usb_status_label = QLabel("USB: Not connected")
        self._usb_status_label.setStyleSheet("color: #a80; font-size: 10px; padding: 2px;")
        section.add_widget(self._usb_status_label)

        # Emissivity
        row = QHBoxLayout()
        row.addWidget(QLabel("Emissivity:"))
        self._emissivity = QDoubleSpinBox()
        self._emissivity.setRange(0.01, 1.0)
        self._emissivity.setSingleStep(0.01)
        self._emissivity.setValue(0.95)
        self._emissivity.setToolTip(
            "Surface emissivity (0.01-1.0)\n"
            "• Skin/organic: 0.95-0.98\n"
            "• Painted metal: 0.90-0.95\n"
            "• Bare metal: 0.05-0.20\n"
            "• Plastic/rubber: 0.90-0.95\n"
            "Higher values = higher temp readings"
        )
        self._emissivity.valueChanged.connect(self.emissivity_changed.emit)
        row.addWidget(self._emissivity)
        section.add_layout(row)

        # Distance
        row = QHBoxLayout()
        row.addWidget(QLabel("Distance (m):"))
        self._distance = QDoubleSpinBox()
        self._distance.setRange(0.0, 200.0)
        self._distance.setSingleStep(0.1)
        self._distance.setValue(0.5)
        self._distance.setToolTip(
            "Distance to target in meters\n"
            "Affects atmospheric absorption compensation.\n"
            "For close-up work (<2m), effect is minimal."
        )
        self._distance.valueChanged.connect(self.distance_changed.emit)
        row.addWidget(self._distance)
        section.add_layout(row)

        # Gain mode
        row = QHBoxLayout()
        row.addWidget(QLabel("Gain:"))
        self._gain_combo = QComboBox()
        self._gain_combo.addItems(["High (Sensitive)", "Low (Wide Range)"])
        self._gain_combo.setToolTip(
            "Sensor gain mode:\n"
            "• High: Better sensitivity, narrow temp range\n"
            "• Low: Wide temp range, less detail\n"
            "Use High for room-temp objects, Low for hot objects"
        )
        self._gain_combo.currentIndexChanged.connect(lambda i: self.gain_mode_changed.emit(1 - i))
        row.addWidget(self._gain_combo)
        section.add_layout(row)

        # Reflection temp
        row = QHBoxLayout()
        row.addWidget(QLabel("Reflect T (°C):"))
        self._reflect_temp = QDoubleSpinBox()
        self._reflect_temp.setRange(-40.0, 500.0)
        self._reflect_temp.setSingleStep(1.0)
        self._reflect_temp.setValue(25.0)
        self._reflect_temp.setToolTip(
            "Reflected temperature (°C)\n"
            "Temperature of surroundings reflected by target.\n"
            "Usually room temperature (~20-25°C).\n"
            "Important for shiny/reflective surfaces."
        )
        self._reflect_temp.valueChanged.connect(self.reflection_temp_changed.emit)
        row.addWidget(self._reflect_temp)
        section.add_layout(row)

        # Atmospheric temp
        row = QHBoxLayout()
        row.addWidget(QLabel("Ambient T (°C):"))
        self._ambient_temp = QDoubleSpinBox()
        self._ambient_temp.setRange(-40.0, 500.0)
        self._ambient_temp.setSingleStep(1.0)
        self._ambient_temp.setValue(25.0)
        self._ambient_temp.setToolTip(
            "Ambient/atmospheric temperature (°C)\n"
            "Temperature of air between camera and target.\n"
            "Affects long-distance measurements."
        )
        self._ambient_temp.valueChanged.connect(self.atmospheric_temp_changed.emit)
        row.addWidget(self._ambient_temp)
        section.add_layout(row)

        # NUC trigger button
        self._nuc_button = QPushButton("Trigger NUC (Shutter)")
        self._nuc_button.setToolTip(
            "Non-Uniformity Correction (NUC)\n"
            "Triggers internal shutter calibration.\n"
            "Camera does this automatically every few minutes,\n"
            "but you can force it for better accuracy."
        )
        self._nuc_button.clicked.connect(self.trigger_nuc.emit)
        section.add_widget(self._nuc_button)

        parent_layout.addWidget(section)

    def _create_display_section(self, parent_layout):
        """Create display settings section."""
        section = CollapsibleSection("Display")

        # Thermal rotation
        row = QHBoxLayout()
        row.addWidget(QLabel("Rotation:"))
        self._thermal_rotation_combo = QComboBox()
        self._thermal_rotation_combo.addItems(["0°", "90°", "180°", "270°"])
        self._thermal_rotation_combo.setCurrentIndex(0)
        self._thermal_rotation_combo.setToolTip(
            "Rotate thermal camera view.\n"
            "Use if camera is mounted at an angle."
        )
        self._thermal_rotation_combo.currentIndexChanged.connect(
            lambda i: self.thermal_rotation_changed.emit(i * 90)
        )
        row.addWidget(self._thermal_rotation_combo)
        section.add_layout(row)

        # Scale factor
        row = QHBoxLayout()
        row.addWidget(QLabel("Scale:"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["1x", "2x", "3x", "4x"])
        self._scale_combo.setCurrentIndex(1)  # Default 2x
        self._scale_combo.currentIndexChanged.connect(
            lambda i: self.scale_factor_changed.emit(float(i + 1))
        )
        row.addWidget(self._scale_combo)
        section.add_layout(row)

        # AI Upscaling
        row = QHBoxLayout()
        row.addWidget(QLabel("AI Upscale:"))
        self._ai_upscale_combo = QComboBox()
        self._ai_upscale_combo.addItems(get_ai_model_names())
        self._ai_upscale_combo.setCurrentIndex(0)  # Off by default
        self._ai_upscale_combo.setToolTip(
            "AI-based upscaling using Real-ESRGAN:\n"
            "• Off: Use traditional scaling (fast)\n"
            "• AnimVideo V3: Fast, good for real-time\n"
            "• x4plus: Best quality (slower)\n"
            "• x4plus Anime: Sharp edges (slower)\n\n"
            "Models auto-download on first use (~6MB each)"
        )
        self._ai_upscale_combo.currentTextChanged.connect(self._on_ai_upscale_changed)
        row.addWidget(self._ai_upscale_combo)
        section.add_layout(row)

        # AI Upscale status label
        self._ai_status_label = QLabel("")
        self._ai_status_label.setStyleSheet("color: #888; font-size: 10px;")
        section.add_widget(self._ai_status_label)

        # Sharpening slider (for AI upscale post-processing)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sharpen:"))
        self._sharpen_slider = QSlider(Qt.Orientation.Horizontal)
        self._sharpen_slider.setRange(0, 100)
        self._sharpen_slider.setSingleStep(10)  # Scroll/arrow in 10% increments
        self._sharpen_slider.setPageStep(10)    # Click track in 10% increments
        self._sharpen_slider.setValue(30)  # 0.3 default
        self._sharpen_slider.setToolTip(
            "Post-sharpening after AI upscale (0-100%)\n"
            "Enhances edges and detail. More aggressive at higher values.\n"
            "0 = Off, 50 = Moderate, 100 = Maximum"
        )
        self._sharpen_slider.valueChanged.connect(
            lambda v: self.ai_sharpen_changed.emit(v / 100.0)
        )
        row.addWidget(self._sharpen_slider)
        self._sharpen_label = QLabel("30%")
        self._sharpen_label.setMinimumWidth(35)
        self._sharpen_slider.valueChanged.connect(
            lambda v: self._sharpen_label.setText(f"{v}%")
        )
        row.addWidget(self._sharpen_label)
        section.add_layout(row)

        parent_layout.addWidget(section)

    def _create_actions_section(self, parent_layout):
        """Create actions section."""
        section = CollapsibleSection("Actions")

        # Screenshot button
        self._screenshot_button = QPushButton("Take Screenshot")
        self._screenshot_button.clicked.connect(self.screenshot_requested.emit)
        section.add_widget(self._screenshot_button)

        parent_layout.addWidget(section)

    # Methods to update UI from external sources
    def set_emissivity(self, value: float):
        self._emissivity.blockSignals(True)
        self._emissivity.setValue(value)
        self._emissivity.blockSignals(False)

    def set_distance(self, value: float):
        self._distance.blockSignals(True)
        self._distance.setValue(value)
        self._distance.blockSignals(False)

    def set_gain_mode(self, high_gain: bool):
        self._gain_combo.blockSignals(True)
        self._gain_combo.setCurrentIndex(0 if high_gain else 1)
        self._gain_combo.blockSignals(False)

    def set_usb_connected(self, connected: bool):
        """Update USB connection status indicator."""
        if connected:
            self._usb_status_label.setText("USB: Connected")
            self._usb_status_label.setStyleSheet("color: #080; font-size: 10px; padding: 2px;")
        else:
            self._usb_status_label.setText("USB: Not connected (settings won't apply)")
            self._usb_status_label.setStyleSheet("color: #a80; font-size: 10px; padding: 2px;")

    def set_fusion_settings(
        self,
        enabled: bool = True,
        mode: int = 2,
        edge_strength: float = 1.5,
        edge_detail: int = 3,
        edge_threshold: int = 10,
        fusion_alpha: float = 0.3,
        zoom: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
        rotation: float = 0.0,
        denoise: int = 9,
        temporal: int = 1,
        pip_position: int = 3,
        pip_size: float = 0.25
    ):
        """Set all fusion settings from external source (e.g., loaded settings)."""
        self._fusion_enable.blockSignals(True)
        self._fusion_enable.setChecked(enabled)
        self._fusion_enable.blockSignals(False)

        self._fusion_mode_combo.blockSignals(True)
        self._fusion_mode_combo.setCurrentIndex(mode)
        self._fusion_mode_combo.blockSignals(False)

        self._edge_strength.blockSignals(True)
        self._edge_strength.setValue(int(edge_strength * 10))
        self._edge_strength_label.setText(f"{edge_strength:.1f}")
        self._edge_strength.blockSignals(False)

        # Set edge detail (kernel size) - map value to combo index
        self._edge_detail_combo.blockSignals(True)
        detail_index = {1: 0, 3: 1, 5: 2}.get(edge_detail, 0)
        self._edge_detail_combo.setCurrentIndex(detail_index)
        self._edge_detail_combo.blockSignals(False)

        # Set edge threshold
        self._edge_threshold.blockSignals(True)
        self._edge_threshold.setValue(edge_threshold)
        self._edge_threshold_label.setText(str(edge_threshold))
        self._edge_threshold.blockSignals(False)

        self._fusion_alpha.blockSignals(True)
        self._fusion_alpha.setValue(int(fusion_alpha * 100))
        self._fusion_alpha_label.setText(f"{int(fusion_alpha * 100)}%")
        self._fusion_alpha.blockSignals(False)

        self._fusion_zoom.blockSignals(True)
        self._fusion_zoom.setValue(zoom)
        self._fusion_zoom.blockSignals(False)

        self._fusion_offset_x.blockSignals(True)
        self._fusion_offset_x.setValue(offset_x)
        self._fusion_offset_x.blockSignals(False)

        self._fusion_offset_y.blockSignals(True)
        self._fusion_offset_y.setValue(offset_y)
        self._fusion_offset_y.blockSignals(False)

        self._fusion_rotation.blockSignals(True)
        self._fusion_rotation.setValue(rotation)
        self._fusion_rotation.blockSignals(False)

        self._fusion_denoise.blockSignals(True)
        self._fusion_denoise.setValue(denoise)
        self._fusion_denoise.blockSignals(False)

        self._fusion_temporal.blockSignals(True)
        self._fusion_temporal.setValue(temporal)
        self._fusion_temporal.blockSignals(False)

        self._pip_position_combo.blockSignals(True)
        self._pip_position_combo.setCurrentIndex(pip_position)
        self._pip_position_combo.blockSignals(False)

        self._pip_size.blockSignals(True)
        self._pip_size.setValue(int(pip_size * 100))
        self._pip_size_label.setText(f"{int(pip_size * 100)}%")
        self._pip_size.blockSignals(False)

        # Update control states
        self._on_fusion_mode_changed(mode)

    def set_temp_range_settings(
        self,
        auto_range: bool = True,
        min_temp: float = 15.0,
        max_temp: float = 40.0
    ):
        """Set temperature range settings from external source."""
        self._auto_range_check.blockSignals(True)
        self._auto_range_check.setChecked(auto_range)
        self._auto_range_check.blockSignals(False)

        self._temp_min_slider.blockSignals(True)
        self._temp_min_slider.setValue(int(min_temp))
        self._temp_min_label.setText(f"{int(min_temp)}°C")
        self._temp_min_slider.setEnabled(not auto_range)
        self._temp_min_slider.blockSignals(False)

        self._temp_max_slider.blockSignals(True)
        self._temp_max_slider.setValue(int(max_temp))
        self._temp_max_label.setText(f"{int(max_temp)}°C")
        self._temp_max_slider.setEnabled(not auto_range)
        self._temp_max_slider.blockSignals(False)

    def update_effective_range(self, min_temp: float, max_temp: float):
        """Update the displayed temperature range (when using auto-range)."""
        if self._auto_range_check.isChecked():
            self._temp_min_label.setText(f"{min_temp:.0f}°C")
            self._temp_max_label.setText(f"{max_temp:.0f}°C")

    def set_thermal_rotation(self, degrees: int):
        """Set thermal rotation from external source (e.g., loaded settings)."""
        index = {0: 0, 90: 1, 180: 2, 270: 3}.get(degrees, 0)
        self._thermal_rotation_combo.blockSignals(True)
        self._thermal_rotation_combo.setCurrentIndex(index)
        self._thermal_rotation_combo.blockSignals(False)

    def set_visible_camera_available(self, available: bool):
        """Enable/disable fusion controls based on visible camera availability."""
        # Store the widgets that should be disabled when no visible camera
        fusion_widgets = [
            self._fusion_enable,
            self._fusion_mode_combo,
            self._edge_method_combo,
            self._edge_strength,
            self._edge_detail_combo,
            self._edge_threshold,
            self._fusion_alpha,
            self._fusion_zoom,
            self._fusion_offset_x,
            self._fusion_offset_y,
            self._fusion_rotation,
            self._fusion_denoise,
            self._fusion_temporal,
            self._pip_position_combo,
            self._pip_size,
        ]

        for widget in fusion_widgets:
            widget.setEnabled(available)

        if not available:
            # Set tooltip explaining why controls are disabled
            tooltip = "Visible camera not available.\nConnect a visible camera to use fusion features."
            for widget in fusion_widgets:
                widget.setToolTip(tooltip)
            # Uncheck fusion enable when no camera and emit signal to update state
            if self._fusion_enable.isChecked():
                self._fusion_enable.setChecked(False)  # This emits toggled signal
            else:
                # Already unchecked, but ensure signal is emitted for state sync
                self.fusion_enabled.emit(False)
        else:
            # Restore original tooltips - let mode change handler update states
            self._on_fusion_mode_changed(self._fusion_mode_combo.currentIndex())

    def _on_ai_upscale_changed(self, name: str):
        """Handle AI upscale model change."""
        self._ai_status_label.setText("Loading model...")
        self.ai_upscale_changed.emit(name)

    def set_ai_upscale_status(self, status: str):
        """Update AI upscale status label."""
        self._ai_status_label.setText(status)

    def set_ai_upscale_available(self, available: bool, gpu_name: str = ""):
        """Update AI upscale availability status."""
        if available:
            self._ai_upscale_combo.setEnabled(True)
            if gpu_name:
                self._ai_status_label.setText(f"GPU: {gpu_name}")
            else:
                self._ai_status_label.setText("Ready (CPU mode)")
        else:
            self._ai_upscale_combo.setEnabled(False)
            self._ai_upscale_combo.setCurrentIndex(0)
            self._ai_status_label.setText("ncnn not installed")
            self._ai_upscale_combo.setToolTip(
                "AI upscaling not available.\n"
                "Install ncnn: pip install ncnn"
            )

    def set_ai_upscale_model(self, model_name: str):
        """Set AI upscale model from external source (e.g., loaded settings)."""
        index = self._ai_upscale_combo.findText(model_name)
        if index >= 0:
            self._ai_upscale_combo.blockSignals(True)
            self._ai_upscale_combo.setCurrentIndex(index)
            self._ai_upscale_combo.blockSignals(False)
