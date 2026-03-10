# Made by Claude (claude-sonnet-4-6, Anthropic AI)
# NOTE: This GUI requires a successful installation of FreeTrace to function.
"""
FreeTrace GUI — run localization and tracking by clicking.
Launch with:  python gui.py
"""
import math
import os
import sys
import traceback

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QTextEdit, QSplitter,
    QTabWidget, QScrollArea, QProgressBar, QMessageBox, QSizePolicy,
)

# Base window size — font sizes are defined relative to this
_BASE_W, _BASE_H = 1050, 720


# ---------------------------------------------------------------------------
# Worker thread — runs FreeTrace without blocking the UI
# ---------------------------------------------------------------------------
class FreeTraceWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)   # percent, stage label
    finished = pyqtSignal(bool, str)  # success, output_dir

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from FreeTrace import Localization, Tracking

            p = self.params
            os.makedirs(p["output_dir"], exist_ok=True)

            self.log.emit("Starting localization…")
            self.progress.emit(5, "Localization")

            loc = Localization.run_process(
                input_video_path=p["video_path"],
                output_path=p["output_dir"],
                window_size=p["window_size"],
                threshold=p["threshold"],
                gpu_on=p["gpu_localization"],
                save_video=p["save_loc_video"],
                realtime_visualization=False,
                verbose=1,
                batch=False,
            )

            self.progress.emit(50, "Localization done")
            self.log.emit("Localization complete.")

            if not loc:
                self.finished.emit(False, "Localization returned no results.")
                return

            self.log.emit("Starting tracking…")
            self.progress.emit(55, "Tracking")

            Tracking.run_process(
                input_video_path=p["video_path"],
                output_path=p["output_dir"],
                graph_depth=p["graph_depth"],
                cutoff=p["cutoff"],
                jump_threshold=p["jump_threshold"] if p["jump_threshold"] > 0 else None,
                gpu_on=p["fbm_mode"],
                save_video=p["save_track_video"],
                realtime_visualization=False,
                verbose=1,
                batch=False,
            )

            self.progress.emit(100, "Done")
            self.log.emit("Tracking complete.")
            self.finished.emit(True, p["output_dir"])

        except Exception:
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, "An error occurred — see log.")


# ---------------------------------------------------------------------------
# Collapsible section widget
# ---------------------------------------------------------------------------
class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title_text = title
        self._toggle = QPushButton(f"▼  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.toggled.connect(self._on_toggle)
        self._apply_toggle_style(14)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

    def _apply_toggle_style(self, font_px: int):
        self._toggle.setStyleSheet(
            f"QPushButton {{ text-align:left; font-weight:bold; font-size:{font_px}px;"
            "border:none; background:#2d2d2d; color:#ccc; padding:6px 8px; border-radius:4px; }"
            "QPushButton:checked { background:#3a3a3a; }"
        )

    def set_font_size(self, font_px: int):
        self._apply_toggle_style(font_px)

    def _on_toggle(self, checked):
        self._body.setVisible(checked)
        self._toggle.setText(
            f"{'▼' if checked else '▶'}  {self._toggle.text()[3:]}"
        )

    def add_widget(self, widget):
        self._body_layout.addWidget(widget)

    def add_layout(self, layout):
        self._body_layout.addLayout(layout)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class FreeTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeTrace")
        self.setMinimumSize(_BASE_W, _BASE_H)
        self._worker = None
        self._output_dir = None
        # Debounce timer — fires 80 ms after the last resize event
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._apply_fonts)
        self._setup_ui()
        self._apply_fonts()

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------
    def _scale(self) -> float:
        """Return a scale factor relative to the base window size."""
        s = math.sqrt(self.width() * self.height()) / math.sqrt(_BASE_W * _BASE_H)
        return max(0.8, min(2.5, s))

    def _f(self, base_px: int) -> int:
        """Scale a base pixel font size and clamp to at least 8 px."""
        return max(8, round(base_px * self._scale()))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 670])

    # ---- left panel (controls) ----------------------------------------
    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # Title
        self._title_label = QLabel("FreeTrace")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color:#7ec8e3; margin:6px 0;")
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("Single-molecule tracking · fBm inference")
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._subtitle_label)

        self._install_notice = QLabel("Valid after successful installation of FreeTrace")
        self._install_notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._install_notice)

        # File I/O
        self._io_sec = CollapsibleSection("Input / Output")
        io_grid = QGridLayout()
        io_grid.setColumnStretch(1, 1)

        io_grid.addWidget(QLabel("Input video (.tiff):"), 0, 0)
        self._video_path = QLineEdit("inputs/sample0.tiff")
        io_grid.addWidget(self._video_path, 0, 1)
        btn_vid = QPushButton("Browse")
        btn_vid.clicked.connect(self._browse_video)
        io_grid.addWidget(btn_vid, 0, 2)

        io_grid.addWidget(QLabel("Output folder:"), 1, 0)
        self._output_path = QLineEdit("outputs")
        io_grid.addWidget(self._output_path, 1, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        io_grid.addWidget(btn_out, 1, 2)

        self._io_sec.add_layout(io_grid)
        layout.addWidget(self._io_sec)

        # Basic parameters
        self._basic_sec = CollapsibleSection("Basic Parameters")
        basic_grid = QGridLayout()
        basic_grid.setColumnStretch(1, 1)

        basic_grid.addWidget(QLabel("Window size:"), 0, 0)
        self._window_size = QSpinBox()
        self._window_size.setRange(3, 21)
        self._window_size.setSingleStep(2)
        self._window_size.setValue(7)
        self._window_size.setToolTip("Sliding window size for particle localisation (odd number).")
        basic_grid.addWidget(self._window_size, 0, 1)

        basic_grid.addWidget(QLabel("Detection threshold:"), 1, 0)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.1, 10.0)
        self._threshold.setSingleStep(0.1)
        self._threshold.setValue(1.0)
        self._threshold.setToolTip("Signal-to-noise threshold for particle detection.")
        basic_grid.addWidget(self._threshold, 1, 1)

        basic_grid.addWidget(QLabel("Min trajectory length:"), 2, 0)
        self._cutoff = QSpinBox()
        self._cutoff.setRange(1, 50)
        self._cutoff.setValue(3)
        self._cutoff.setToolTip("Minimum number of frames a trajectory must span to be kept.")
        basic_grid.addWidget(self._cutoff, 2, 1)

        self._basic_sec.add_layout(basic_grid)
        layout.addWidget(self._basic_sec)

        # Advanced parameters
        self._adv_sec = CollapsibleSection("Advanced Parameters")
        adv_grid = QGridLayout()
        adv_grid.setColumnStretch(1, 1)

        adv_grid.addWidget(QLabel("Graph depth (Δt):"), 0, 0)
        self._graph_depth = QSpinBox()
        self._graph_depth.setRange(1, 10)
        self._graph_depth.setValue(3)
        self._graph_depth.setToolTip("Number of future frames considered for trajectory reconnection.")
        adv_grid.addWidget(self._graph_depth, 0, 1)

        adv_grid.addWidget(QLabel("Jump threshold (0=auto):"), 1, 0)
        self._jump_threshold = QDoubleSpinBox()
        self._jump_threshold.setRange(0, 500)
        self._jump_threshold.setSingleStep(1.0)
        self._jump_threshold.setValue(0)
        self._jump_threshold.setToolTip("Max jump distance in pixels. 0 = inferred automatically.")
        adv_grid.addWidget(self._jump_threshold, 1, 1)

        self._fbm_mode = QCheckBox("FBM mode (fBm inference, slower)")
        self._fbm_mode.setChecked(True)
        self._fbm_mode.setToolTip("Use fractional Brownian motion model for tracking (requires GPU for speed).")
        adv_grid.addWidget(self._fbm_mode, 2, 0, 1, 2)

        self._gpu_loc = QCheckBox("GPU for localisation (CUDA)")
        self._gpu_loc.setChecked(True)
        adv_grid.addWidget(self._gpu_loc, 3, 0, 1, 2)

        self._save_loc_video = QCheckBox("Save localisation video")
        self._save_loc_video.setChecked(False)
        adv_grid.addWidget(self._save_loc_video, 4, 0, 1, 2)

        self._save_track_video = QCheckBox("Save tracking video")
        self._save_track_video.setChecked(False)
        adv_grid.addWidget(self._save_track_video, 5, 0, 1, 2)

        self._adv_sec.add_layout(adv_grid)
        layout.addWidget(self._adv_sec)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%  %v")
        layout.addWidget(self._progress_bar)

        self._stage_label = QLabel("")
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stage_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run FreeTrace")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#43a047; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background:#c62828; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#e53935; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        return panel

    # ---- right panel (log + output images) ----------------------------
    def _build_right_panel(self):
        tabs = QTabWidget()

        # Log tab
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background:#1a1a1a; color:#ccc; border:none;")
        tabs.addTab(self._log, "Log")

        # Results tab
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_widget = QWidget()
        self._results_layout = QVBoxLayout(results_widget)
        self._results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._no_results_label = QLabel("Results will appear here after running FreeTrace.")
        self._no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results_layout.addWidget(self._no_results_label)

        results_scroll.setWidget(results_widget)
        tabs.addTab(results_scroll, "Results")

        self._tabs = tabs
        return tabs

    # ------------------------------------------------------------------
    # Dynamic font scaling
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Debounce: wait 80 ms after the last resize before updating fonts
        self._resize_timer.start(80)

    def _apply_fonts(self):
        """Recompute all font sizes based on current window dimensions."""
        f = self._f

        # Stylesheet — all font-size values are scaled
        self._apply_dark_theme(f)

        # QFont-based widgets
        self._title_label.setFont(QFont("Arial", f(18), QFont.Weight.Bold))
        self._run_btn.setFont(QFont("Arial", f(12), QFont.Weight.Bold))
        self._stop_btn.setFont(QFont("Arial", f(12), QFont.Weight.Bold))
        self._log.setFont(QFont("Courier New", f(13)))

        # Inline-styled labels
        self._subtitle_label.setStyleSheet(
            f"color:#888; font-size:{f(14)}px; margin-bottom:4px;"
        )
        self._install_notice.setStyleSheet(
            f"color:#f0a500; font-size:{f(12)}px; font-style:italic; margin-bottom:8px;"
        )
        self._stage_label.setStyleSheet(f"color:#888; font-size:{f(13)}px;")
        self._no_results_label.setStyleSheet(
            f"color:#666; font-size:{f(15)}px; margin:40px;"
        )

        # CollapsibleSection toggles
        for sec in (self._io_sec, self._basic_sec, self._adv_sec):
            sec.set_font_size(f(14))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input video", "", "TIFF files (*.tiff *.tif);;All files (*)"
        )
        if path:
            self._video_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_path.setText(path)

    def _on_run(self):
        video = self._video_path.text().strip()
        if not os.path.exists(video):
            QMessageBox.warning(self, "File not found", f"Cannot find:\n{video}")
            return

        params = {
            "video_path": video,
            "output_dir": self._output_path.text().strip() or "outputs",
            "window_size": self._window_size.value(),
            "threshold": self._threshold.value(),
            "cutoff": self._cutoff.value(),
            "graph_depth": self._graph_depth.value(),
            "jump_threshold": self._jump_threshold.value(),
            "fbm_mode": self._fbm_mode.isChecked(),
            "gpu_localization": self._gpu_loc.isChecked(),
            "save_loc_video": self._save_loc_video.isChecked(),
            "save_track_video": self._save_track_video.isChecked(),
        }

        self._log.clear()
        self._log.append(f"<b>Input:</b> {params['video_path']}")
        self._log.append(f"<b>Output:</b> {params['output_dir']}")
        self._log.append(f"<b>FBM mode:</b> {params['fbm_mode']}")
        self._log.append("-" * 60)
        self._progress_bar.setValue(0)
        self._stage_label.setText("")
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._tabs.setCurrentIndex(0)

        self._worker = FreeTraceWorker(params)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
            self._append_log("⚠ Stopped by user.")
            self._reset_buttons()

    def _append_log(self, text: str):
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _update_progress(self, value: int, label: str):
        self._progress_bar.setValue(value)
        self._stage_label.setText(label)

    def _on_finished(self, success: bool, message: str):
        self._reset_buttons()
        if success:
            self._output_dir = message
            self._append_log(f"✓ Done. Results saved to: {message}")
            self._load_results(message)
            self._tabs.setCurrentIndex(1)
        else:
            self._append_log(f"✗ {message}")
            QMessageBox.critical(self, "FreeTrace error", message)

    def _reset_buttons(self):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _load_results(self, output_dir: str):
        # Clear previous results
        for i in reversed(range(self._results_layout.count())):
            w = self._results_layout.itemAt(i).widget()
            if w:
                w.deleteLater()

        image_files = {
            "Trajectory Map": "_traces.png",
            "Localisation Density": "_loc_2d_density.png",
        }

        found = False
        for title, suffix in image_files.items():
            matches = [
                f for f in os.listdir(output_dir)
                if f.endswith(suffix)
            ]
            for fname in matches:
                fpath = os.path.join(output_dir, fname)
                if not os.path.exists(fpath):
                    continue
                found = True

                header = QLabel(f"<b>{title}</b> — {fname}")
                header.setStyleSheet(
                    f"color:#aaa; font-size:{self._f(15)}px; margin-top:12px;"
                )
                self._results_layout.addWidget(header)

                img_label = QLabel()
                pixmap = QPixmap(fpath)
                if not pixmap.isNull():
                    pixmap = pixmap.scaledToWidth(
                        600, Qt.TransformationMode.SmoothTransformation
                    )
                    img_label.setPixmap(pixmap)
                else:
                    img_label.setText(f"(could not load {fname})")
                    img_label.setStyleSheet("color:#888;")
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._results_layout.addWidget(img_label)

        if not found:
            lbl = QLabel("No output images found in the output folder.")
            lbl.setStyleSheet(f"color:#666; font-size:{self._f(15)}px; margin:20px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._results_layout.addWidget(lbl)

    # ------------------------------------------------------------------
    # Dark theme — font sizes passed in as a callable
    # ------------------------------------------------------------------
    def _apply_dark_theme(self, f):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: #1e1e1e;
                color: #ddd;
                font-size: {f(14)}px;
            }}
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 6px;
                font-weight: bold;
                font-size: {f(14)}px;
                color: #bbb;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {f(14)}px;
                color: #eee;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid #7ec8e3;
            }}
            QPushButton {{
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: {f(14)}px;
                color: #ddd;
            }}
            QPushButton:hover {{ background: #4a4a4a; }}
            QCheckBox {{ color: #ccc; spacing: 8px; font-size: {f(14)}px; }}
            QCheckBox::indicator {{
                width: {f(17)}px; height: {f(17)}px;
                border: 1px solid #666;
                border-radius: 3px;
                background: #2a2a2a;
            }}
            QCheckBox::indicator:checked {{
                background: #7ec8e3;
                border-color: #7ec8e3;
            }}
            QTabWidget::pane {{ border: 1px solid #444; background: #1e1e1e; }}
            QTabBar::tab {{
                background: #2a2a2a; color: #aaa;
                padding: 8px 18px; font-size: {f(14)}px;
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{ background: #1e1e1e; color: #fff; }}
            QProgressBar {{
                border: 1px solid #555; border-radius: 4px;
                background: #2a2a2a; color: #eee; text-align: center;
                font-size: {f(13)}px; height: {f(22)}px;
            }}
            QProgressBar::chunk {{ background: #2e7d32; border-radius: 3px; }}
            QScrollArea {{ border: none; }}
            QSplitter::handle {{ background: #333; }}
            QLabel {{ color: #ccc; font-size: {f(14)}px; }}
        """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setApplicationName("FreeTrace")
    win = FreeTraceGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
