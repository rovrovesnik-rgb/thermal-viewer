"""
Console log window for debugging camera communication.
"""

import sys
from datetime import datetime
from typing import Optional
from collections import deque

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton,
    QHBoxLayout, QCheckBox, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QTextCursor


class LogCapture(QObject):
    """Captures log messages and emits them as signals."""

    log_message = pyqtSignal(str, str)  # (level, message)

    _instance: Optional['LogCapture'] = None

    def __init__(self):
        super().__init__()
        self._messages = deque(maxlen=1000)  # Keep last 1000 messages

    @classmethod
    def instance(cls) -> 'LogCapture':
        if cls._instance is None:
            cls._instance = LogCapture()
        return cls._instance

    def log(self, level: str, message: str):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        full_msg = f"[{timestamp}] [{level}] {message}"
        self._messages.append(full_msg)
        self.log_message.emit(level, full_msg)

    def info(self, message: str):
        self.log("INFO", message)

    def warn(self, message: str):
        self.log("WARN", message)

    def error(self, message: str):
        self.log("ERROR", message)

    def debug(self, message: str):
        self.log("DEBUG", message)

    def usb(self, message: str):
        """Log USB communication."""
        self.log("USB", message)

    def camera(self, message: str):
        """Log camera events."""
        self.log("CAM", message)

    def get_history(self) -> list[str]:
        """Get all logged messages."""
        return list(self._messages)


# Global logger instance
def get_logger() -> LogCapture:
    return LogCapture.instance()


class ConsoleLogWindow(QDialog):
    """Console log window showing camera communication."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Console Log")
        self.setMinimumSize(600, 400)
        self.resize(800, 500)

        # Don't block main window
        self.setModal(False)

        layout = QVBoxLayout(self)

        # Filter checkboxes
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))

        self._filters = {}
        for level in ["INFO", "WARN", "ERROR", "DEBUG", "USB", "CAM"]:
            cb = QCheckBox(level)
            cb.setChecked(True)
            cb.toggled.connect(self._refresh_display)
            self._filters[level] = cb
            filter_layout.addWidget(cb)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Log text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Monospace", 9))
        self._text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #404040;
            }
        """)
        layout.addWidget(self._text)

        # Buttons
        btn_layout = QHBoxLayout()

        self._auto_scroll = QCheckBox("Auto-scroll")
        self._auto_scroll.setChecked(True)
        btn_layout.addWidget(self._auto_scroll)

        btn_layout.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        btn_layout.addWidget(clear_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        # Connect to logger
        self._logger = get_logger()
        self._logger.log_message.connect(self._on_log_message)

        # Load history
        for msg in self._logger.get_history():
            self._append_message(msg)

    def _on_log_message(self, level: str, message: str):
        """Handle new log message."""
        if self._filters.get(level, self._filters.get("INFO")).isChecked():
            self._append_message(message)

    def _append_message(self, message: str):
        """Append message to text area with coloring."""
        # Color based on level
        color = "#d4d4d4"  # default
        if "[ERROR]" in message:
            color = "#f14c4c"
        elif "[WARN]" in message:
            color = "#cca700"
        elif "[USB]" in message:
            color = "#3dc9b0"
        elif "[CAM]" in message:
            color = "#569cd6"
        elif "[DEBUG]" in message:
            color = "#808080"

        self._text.append(f'<span style="color: {color}">{message}</span>')

        if self._auto_scroll.isChecked():
            self._text.moveCursor(QTextCursor.MoveOperation.End)

    def _refresh_display(self):
        """Refresh display with current filters."""
        self._text.clear()
        for msg in self._logger.get_history():
            # Check if message matches any enabled filter
            for level, cb in self._filters.items():
                if f"[{level}]" in msg and cb.isChecked():
                    self._append_message(msg)
                    break

    def _clear(self):
        """Clear the log display."""
        self._text.clear()
