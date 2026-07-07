"""
Thermal image display widget with crosshair and ROI overlay support.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent


class ThermalView(QWidget):
    """
    Widget for displaying thermal imagery with overlays.

    Features:
    - Thermal image display with colormap
    - Center crosshair with temperature readout
    - ROI rectangle drawing and editing
    - Mouse position temperature tracking
    """

    # Signals
    roi_created = pyqtSignal(QRect)  # Emitted when a new ROI is drawn
    roi_selected = pyqtSignal(int)   # Emitted when an ROI is clicked (index)
    mouse_position = pyqtSignal(int, int, float)  # x, y, temperature

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image: QImage = None
        self._pixmap: QPixmap = None
        self._temperature_data: np.ndarray = None

        # Display settings
        self._scale_factor = 2.0
        self._show_crosshair = True
        self._show_min_max = True
        self._crosshair_temp = 0.0
        self._min_temp = 0.0
        self._max_temp = 0.0
        self._min_pos = (0, 0)
        self._max_pos = (0, 0)

        # ROI drawing state
        self._drawing_roi = False
        self._roi_start = QPoint()
        self._roi_current = QPoint()
        self._rois: list[QRect] = []
        self._selected_roi = -1

        # Mouse tracking
        self._mouse_temp = 0.0
        self._mouse_pos = QPoint()
        self.setMouseTracking(True)

        # Actual scale (computed from image vs temperature data)
        self._actual_scale = 1.0

        # Setup UI
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(256, 192)

    def set_frame(
        self,
        image: np.ndarray,
        temperature_data: np.ndarray,
        min_temp: float,
        max_temp: float,
        center_temp: float,
        min_pos: tuple = None,
        max_pos: tuple = None
    ):
        """
        Update the displayed frame.

        Args:
            image: BGR image to display (already scaled by GPU scaler)
            temperature_data: Temperature array in Celsius (always 256x192)
            min_temp: Minimum temperature
            max_temp: Maximum temperature
            center_temp: Center point temperature
            min_pos: (x, y) of minimum temperature (in original coords)
            max_pos: (x, y) of maximum temperature (in original coords)
        """
        self._temperature_data = temperature_data
        self._crosshair_temp = center_temp
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._min_pos = min_pos or (0, 0)
        self._max_pos = max_pos or (0, 0)

        # Convert numpy array to QImage
        if image.ndim == 2:
            # Grayscale
            h, w = image.shape
            bytes_per_line = w
            self._image = QImage(
                image.data, w, h, bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
        else:
            # BGR to RGB
            h, w, c = image.shape
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            bytes_per_line = w * 3
            self._image = QImage(
                rgb.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )

        # Image is already scaled by GPU scaler - just convert to pixmap
        # Calculate actual scale from image size vs temperature data size
        self._actual_scale = w / temperature_data.shape[1]
        self._pixmap = QPixmap.fromImage(self._image)

        self.update()

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float):
        self._scale_factor = max(1.0, min(8.0, value))
        # Note: Actual scaling is done by GPU scaler in main_window
        # This just stores the requested scale factor
        self.update()

    @property
    def show_crosshair(self) -> bool:
        return self._show_crosshair

    @show_crosshair.setter
    def show_crosshair(self, value: bool):
        self._show_crosshair = value
        self.update()

    @property
    def show_min_max(self) -> bool:
        return self._show_min_max

    @show_min_max.setter
    def show_min_max(self, value: bool):
        self._show_min_max = value
        self.update()

    def add_roi(self, rect: QRect):
        """Add an ROI rectangle."""
        self._rois.append(rect)
        self.update()

    def remove_roi(self, index: int):
        """Remove an ROI by index."""
        if 0 <= index < len(self._rois):
            del self._rois[index]
            if self._selected_roi >= len(self._rois):
                self._selected_roi = -1
            self.update()

    def clear_rois(self):
        """Remove all ROIs."""
        self._rois.clear()
        self._selected_roi = -1
        self.update()

    def get_rois(self) -> list[QRect]:
        """Get all ROI rectangles."""
        return self._rois.copy()

    def _image_to_widget_coords(self, x: int, y: int) -> QPoint:
        """Convert original image coordinates (256x192) to widget coordinates."""
        if not self._pixmap:
            return QPoint(x, y)

        # Calculate offset for centering
        offset_x = (self.width() - self._pixmap.width()) // 2
        offset_y = (self.height() - self._pixmap.height()) // 2

        # Use actual scale from the displayed image
        return QPoint(
            int(x * self._actual_scale) + offset_x,
            int(y * self._actual_scale) + offset_y
        )

    def _widget_to_image_coords(self, pos: QPoint) -> tuple[int, int]:
        """Convert widget coordinates to original image coordinates (256x192)."""
        if not self._pixmap or self._temperature_data is None:
            return (0, 0)

        offset_x = (self.width() - self._pixmap.width()) // 2
        offset_y = (self.height() - self._pixmap.height()) // 2

        # Use actual scale to convert back to original temperature_data coordinates
        x = int((pos.x() - offset_x) / self._actual_scale)
        y = int((pos.y() - offset_y) / self._actual_scale)

        # Clamp to temperature data bounds (always 256x192)
        h, w = self._temperature_data.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        return (x, y)

    def _get_temperature_at(self, x: int, y: int) -> float:
        """Get temperature at image coordinates."""
        if self._temperature_data is None:
            return 0.0

        h, w = self._temperature_data.shape
        if 0 <= x < w and 0 <= y < h:
            return float(self._temperature_data[y, x])
        return 0.0

    def paintEvent(self, event):
        """Paint the thermal image and overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if not self._pixmap:
            # No image yet
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No thermal image")
            return

        # Draw thermal image centered
        x = (self.width() - self._pixmap.width()) // 2
        y = (self.height() - self._pixmap.height()) // 2
        painter.drawPixmap(x, y, self._pixmap)

        # Draw crosshair
        if self._show_crosshair:
            self._draw_crosshair(painter)

        # Draw min/max markers
        if self._show_min_max:
            self._draw_min_max_markers(painter)

        # Draw ROIs
        self._draw_rois(painter)

        # Draw current ROI being drawn
        if self._drawing_roi:
            self._draw_current_roi(painter)

        # Draw mouse temperature
        self._draw_mouse_temp(painter)

    def _draw_crosshair(self, painter: QPainter):
        """Draw center crosshair with temperature."""
        if not self._pixmap:
            return

        cx = self.width() // 2
        cy = self.height() // 2
        size = 20

        # Crosshair lines
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawLine(cx - size, cy, cx - 5, cy)
        painter.drawLine(cx + 5, cy, cx + size, cy)
        painter.drawLine(cx, cy - size, cx, cy - 5)
        painter.drawLine(cx, cy + 5, cx, cy + size)

        # Temperature text
        font = QFont("Monospace", 12, QFont.Weight.Bold)
        painter.setFont(font)

        text = f"{self._crosshair_temp:.1f}°C"

        # Draw with shadow
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(cx + size + 6, cy + 6, text)
        painter.setPen(QColor(255, 255, 0))
        painter.drawText(cx + size + 5, cy + 5, text)

    def _draw_min_max_markers(self, painter: QPainter):
        """Draw min/max temperature markers."""
        if not self._pixmap or not self._image:
            return

        # Min marker (blue)
        min_widget = self._image_to_widget_coords(*self._min_pos)
        pen = QPen(QColor(0, 100, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(min_widget.x() - 8, min_widget.y(), min_widget.x() + 8, min_widget.y())
        painter.drawLine(min_widget.x(), min_widget.y() - 8, min_widget.x(), min_widget.y() + 8)

        font = QFont("Monospace", 9)
        painter.setFont(font)
        painter.drawText(min_widget.x() + 10, min_widget.y() - 5, f"{self._min_temp:.1f}°C")

        # Max marker (red)
        max_widget = self._image_to_widget_coords(*self._max_pos)
        pen = QPen(QColor(255, 50, 50))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(max_widget.x() - 8, max_widget.y(), max_widget.x() + 8, max_widget.y())
        painter.drawLine(max_widget.x(), max_widget.y() - 8, max_widget.x(), max_widget.y() + 8)

        painter.drawText(max_widget.x() + 10, max_widget.y() - 5, f"{self._max_temp:.1f}°C")

    def _draw_rois(self, painter: QPainter):
        """Draw ROI rectangles."""
        for i, roi in enumerate(self._rois):
            # Convert to widget coordinates
            tl = self._image_to_widget_coords(roi.left(), roi.top())
            br = self._image_to_widget_coords(roi.right(), roi.bottom())
            widget_rect = QRect(tl, br)

            # Selected ROI has different color
            if i == self._selected_roi:
                pen = QPen(QColor(0, 255, 0))
                pen.setWidth(2)
            else:
                pen = QPen(QColor(255, 255, 0))
                pen.setWidth(1)

            painter.setPen(pen)
            painter.drawRect(widget_rect)

            # Draw ROI temperature stats
            if self._temperature_data is not None:
                roi_data = self._temperature_data[roi.top():roi.bottom()+1, roi.left():roi.right()+1]
                if roi_data.size > 0:
                    roi_min = float(np.min(roi_data))
                    roi_max = float(np.max(roi_data))
                    roi_avg = float(np.mean(roi_data))

                    font = QFont("Monospace", 8)
                    painter.setFont(font)
                    text = f"{roi_max:.1f}°C"
                    painter.drawText(widget_rect.left() + 2, widget_rect.top() - 3, text)

    def _draw_current_roi(self, painter: QPainter):
        """Draw the ROI currently being drawn."""
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)

        rect = QRect(self._roi_start, self._roi_current).normalized()
        painter.drawRect(rect)

    def _draw_mouse_temp(self, painter: QPainter):
        """Draw temperature at mouse position."""
        if not self._mouse_pos or self._temperature_data is None:
            return

        # Draw small indicator at mouse position
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)

        x, y = self._mouse_pos.x(), self._mouse_pos.y()

        # Small crosshair
        painter.drawLine(x - 4, y, x + 4, y)
        painter.drawLine(x, y - 4, x, y + 4)

        # Temperature text
        font = QFont("Monospace", 9)
        painter.setFont(font)
        text = f"{self._mouse_temp:.1f}°C"
        painter.drawText(x + 8, y - 3, text)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for ROI drawing."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on existing ROI
            img_x, img_y = self._widget_to_image_coords(event.pos())

            for i, roi in enumerate(self._rois):
                if roi.contains(img_x, img_y):
                    self._selected_roi = i
                    self.roi_selected.emit(i)
                    self.update()
                    return

            # Start drawing new ROI
            self._drawing_roi = True
            self._roi_start = event.pos()
            self._roi_current = event.pos()
            self._selected_roi = -1

        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click: deselect or delete ROI
            img_x, img_y = self._widget_to_image_coords(event.pos())

            for i, roi in enumerate(self._rois):
                if roi.contains(img_x, img_y):
                    self.remove_roi(i)
                    return

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for ROI drawing and temperature tracking."""
        self._mouse_pos = event.pos()

        # Update mouse temperature
        img_x, img_y = self._widget_to_image_coords(event.pos())
        self._mouse_temp = self._get_temperature_at(img_x, img_y)

        # Emit signal
        self.mouse_position.emit(img_x, img_y, self._mouse_temp)

        # Update ROI drawing
        if self._drawing_roi:
            self._roi_current = event.pos()

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to finish ROI drawing."""
        if event.button() == Qt.MouseButton.LeftButton and self._drawing_roi:
            self._drawing_roi = False

            # Convert to image coordinates
            start_img = self._widget_to_image_coords(self._roi_start)
            end_img = self._widget_to_image_coords(event.pos())

            # Create normalized rect (minimum size 5x5)
            rect = QRect(
                QPoint(start_img[0], start_img[1]),
                QPoint(end_img[0], end_img[1])
            ).normalized()

            if rect.width() >= 5 and rect.height() >= 5:
                self._rois.append(rect)
                self.roi_created.emit(rect)

            self.update()

    def sizeHint(self):
        """Return preferred size."""
        if self._pixmap:
            return self._pixmap.size()
        return QSize(512, 384)
