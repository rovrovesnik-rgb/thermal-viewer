"""
ROI detail modal with temperature histogram.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from processing.temperature import TemperatureHistory, TemperatureStats


class ROIModal(QDialog):
    """
    Modal dialog showing detailed ROI information with temperature histogram.
    """

    def __init__(self, roi_index: int, roi_name: str = None, parent=None):
        super().__init__(parent)

        self.roi_index = roi_index
        self.roi_name = roi_name or f"ROI {roi_index + 1}"

        self.setWindowTitle(f"ROI: {self.roi_name}")
        self.setMinimumSize(500, 400)

        # Temperature history (60 seconds at 25fps = 1500 samples)
        self._history = TemperatureHistory(max_samples=1500)
        self._start_time = 0

        self._setup_ui()

        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_plot)
        self._timer.start(100)  # Update plot at 10Hz

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Current stats group
        stats_group = QGroupBox("Current Temperature")
        stats_layout = QFormLayout(stats_group)

        self._min_label = QLabel("--.-°C")
        self._min_label.setStyleSheet("font-size: 14px; color: #6699ff;")
        stats_layout.addRow("Minimum:", self._min_label)

        self._max_label = QLabel("--.-°C")
        self._max_label.setStyleSheet("font-size: 14px; color: #ff6666;")
        stats_layout.addRow("Maximum:", self._max_label)

        self._avg_label = QLabel("--.-°C")
        self._avg_label.setStyleSheet("font-size: 14px; color: #66ff66;")
        stats_layout.addRow("Average:", self._avg_label)

        layout.addWidget(stats_group)

        # Histogram plot
        plot_group = QGroupBox("Temperature History (60s)")
        plot_layout = QVBoxLayout(plot_group)

        # PyQtGraph plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground('k')
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel('left', 'Temperature', units='°C')
        self._plot_widget.setLabel('bottom', 'Time', units='s')

        # Create plot lines
        self._min_curve = self._plot_widget.plot(pen=pg.mkPen('#6699ff', width=1), name='Min')
        self._max_curve = self._plot_widget.plot(pen=pg.mkPen('#ff6666', width=1), name='Max')
        self._avg_curve = self._plot_widget.plot(pen=pg.mkPen('#66ff66', width=2), name='Avg')

        # Legend
        self._plot_widget.addLegend()

        plot_layout.addWidget(self._plot_widget)
        layout.addWidget(plot_group)

        # Location info
        self._location_label = QLabel("Location: (0,0) to (0,0)")
        self._location_label.setStyleSheet("color: gray;")
        layout.addWidget(self._location_label)

        # Buttons
        button_layout = QHBoxLayout()

        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        button_layout.addWidget(export_btn)

        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        button_layout.addWidget(clear_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def update_stats(self, stats: TemperatureStats, timestamp: float):
        """
        Update ROI statistics and add to history.

        Args:
            stats: Current temperature statistics
            timestamp: Frame timestamp
        """
        # Update labels
        self._min_label.setText(f"{stats.min_temp:.1f}°C")
        self._max_label.setText(f"{stats.max_temp:.1f}°C")
        self._avg_label.setText(f"{stats.avg_temp:.1f}°C")

        # Add to history
        if self._start_time == 0:
            self._start_time = timestamp

        self._history.add(stats, timestamp - self._start_time)

    def set_location(self, x1: int, y1: int, x2: int, y2: int):
        """Set the ROI location display."""
        self._location_label.setText(f"Location: ({x1},{y1}) to ({x2},{y2})")

    def _update_plot(self):
        """Update the histogram plot."""
        if len(self._history) < 2:
            return

        timestamps, mins, maxs, avgs = self._history.get_arrays()

        self._min_curve.setData(timestamps, mins)
        self._max_curve.setData(timestamps, maxs)
        self._avg_curve.setData(timestamps, avgs)

    def _clear_history(self):
        """Clear temperature history."""
        self._history.clear()
        self._start_time = 0
        self._min_curve.setData([], [])
        self._max_curve.setData([], [])
        self._avg_curve.setData([], [])

    def _export_csv(self):
        """Export temperature history to CSV."""
        if len(self._history) == 0:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Temperature Data",
            f"roi_{self.roi_index}_temps.csv",
            "CSV Files (*.csv)"
        )

        if not filename:
            return

        timestamps, mins, maxs, avgs = self._history.get_arrays()

        with open(filename, 'w') as f:
            f.write("Time (s),Min (°C),Max (°C),Avg (°C)\n")
            for t, mn, mx, av in zip(timestamps, mins, maxs, avgs):
                f.write(f"{t:.3f},{mn:.2f},{mx:.2f},{av:.2f}\n")

    def closeEvent(self, event):
        """Handle dialog close."""
        self._timer.stop()
        super().closeEvent(event)


class ROIManager:
    """
    Manages multiple ROIs and their modals.
    """

    def __init__(self, parent=None):
        self._parent = parent
        self._modals: dict[int, ROIModal] = {}
        self._roi_names: dict[int, str] = {}

    def open_modal(self, roi_index: int, x1: int, y1: int, x2: int, y2: int):
        """
        Open or focus the modal for an ROI.

        Args:
            roi_index: ROI index
            x1, y1, x2, y2: ROI coordinates
        """
        if roi_index in self._modals:
            # Focus existing modal
            self._modals[roi_index].raise_()
            self._modals[roi_index].activateWindow()
        else:
            # Create new modal
            name = self._roi_names.get(roi_index, f"ROI {roi_index + 1}")
            modal = ROIModal(roi_index, name, self._parent)
            modal.set_location(x1, y1, x2, y2)
            modal.finished.connect(lambda: self._on_modal_closed(roi_index))
            modal.show()
            self._modals[roi_index] = modal

    def close_modal(self, roi_index: int):
        """Close modal for an ROI."""
        if roi_index in self._modals:
            self._modals[roi_index].close()
            del self._modals[roi_index]

    def close_all(self):
        """Close all open modals."""
        for modal in list(self._modals.values()):
            modal.close()
        self._modals.clear()

    def _on_modal_closed(self, roi_index: int):
        """Handle modal closed."""
        if roi_index in self._modals:
            del self._modals[roi_index]

    def update_roi_stats(self, roi_index: int, stats: TemperatureStats, timestamp: float):
        """
        Update stats for an ROI (if its modal is open).

        Args:
            roi_index: ROI index
            stats: Temperature statistics
            timestamp: Frame timestamp
        """
        if roi_index in self._modals:
            self._modals[roi_index].update_stats(stats, timestamp)

    def set_roi_name(self, roi_index: int, name: str):
        """Set custom name for an ROI."""
        self._roi_names[roi_index] = name
        if roi_index in self._modals:
            self._modals[roi_index].roi_name = name
            self._modals[roi_index].setWindowTitle(f"ROI: {name}")

    def remove_roi(self, roi_index: int):
        """Remove an ROI and close its modal."""
        self.close_modal(roi_index)
        if roi_index in self._roi_names:
            del self._roi_names[roi_index]
