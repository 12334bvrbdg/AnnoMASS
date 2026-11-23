#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import random
import re
import shutil
import signal
import struct
import string
import subprocess
import termios
import threading
import time
import fcntl
import json
import pty
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import chain


def _prepare_qt_input_method_environment_for_direct_run():
    """Make `python annomass_app.py` use the same fcitx setup as run_gui.sh.

    This must run before importing PyQt5. launch_gui.py/run_gui.sh already set
    this environment, so only do the heavier profile/startup fallback when this
    file is executed directly as the GUI entry point.
    """
    if __name__ != "__main__" or not sys.platform.startswith("linux"):
        return
    os.environ.setdefault("GTK_IM_MODULE", "fcitx")
    os.environ.setdefault("QT_IM_MODULE", "fcitx")
    os.environ.setdefault("XMODIFIERS", "@im=fcitx")
    os.environ.setdefault("SDL_IM_MODULE", "fcitx")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    system_qt_plugins = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
    plugin_path = os.environ.get("QT_PLUGIN_PATH", "")
    if os.path.isdir(system_qt_plugins) and system_qt_plugins not in plugin_path.split(":"):
        os.environ["QT_PLUGIN_PATH"] = (
            system_qt_plugins + (":" + plugin_path if plugin_path else "")
        )

    try:
        profile_dir = Path.home() / ".config" / "fcitx5"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile").write_text(
            """[Groups/0]
# Group Name
Name=Default
# Layout
Default Layout=us
# Default Input Method
DefaultIM=pinyin

[Groups/0/Items/0]
# Name
Name=keyboard-us
# Layout
Layout=

[Groups/0/Items/1]
# Name
Name=pinyin
# Layout
Layout=

[GroupOrder]
0=Default
""",
            encoding="utf-8",
        )
    except Exception:
        pass

    try:
        if shutil.which("fcitx5"):
            import subprocess
            running = subprocess.run(
                ["pgrep", "-x", "fcitx5"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            cmd = ["fcitx5", "-r", "-d", "--disable", "wayland"] if running else [
                "fcitx5", "-d", "--disable", "wayland"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        pass


_prepare_qt_input_method_environment_for_direct_run()

import numpy as np
import pandas as pd
import hnswlib
from hnswlib import Index

import matchms.filtering as msfilters
from gensim.models import Word2Vec
from molmass import Formula

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, QVariant, QProcess
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS, AllChem

from core.importing.load_from_files import load_from_files
from core.main import identify_unknown, prebuild_database_cache
from core.Spectrum import Spectrum
from core.flavclass_filter import classify as flavonoid_classify
from core.icefrag_adapter import (predict_and_score as mf_predict,
                                  is_available as mf_available,
                                  predict_batch as mf_predict_batch,
                                  render_fragment_svg as mf_render_fragment_svg)
import threading as _threading

from core.staged_levels import select_topk_per_spectrum as _select_topk_per_spectrum
from core import annomass_algorithm

# One-time ~/.flavomass / ~/.deepmass2 -> ~/.annomass config-directory
# migration (API key, agent memory, chat sessions). Runs at import time so every entry point
# (launch_gui.py, run_pipeline_headless.py, public_pip_ablation.py, the
# webapi server) does it exactly once, before any code reads/writes config.
from core.agent_config import migrate_legacy_config_dir as _migrate_legacy_config_dir
_migrate_legacy_config_dir()

_LOCAL_DREAMS_REF_DIR = Path(__file__).resolve().parent / "data" / "reference_libraries"
# Kept in sync with core.dreams_adapter.DEFAULT_REF_HDF5 (see that module for
# the flavonoid-only-switch rationale, 2026-09-16); this module-level default
# is only the pre-_ensure_dreams_adapter() fallback value.
DEFAULT_REF_HDF5 = _LOCAL_DREAMS_REF_DIR / "MassSpecGym_DreaMS_flavonoid_only.hdf5"
_DREAMS_AVAILABLE = None
_DREAMS_IMPORT_ERROR = None
dreams_spectra_to_mgf = None
run_dreams_search = None
list_reference_libraries = None


def _ensure_dreams_adapter():
    global _DREAMS_AVAILABLE, _DREAMS_IMPORT_ERROR
    global dreams_spectra_to_mgf, run_dreams_search, list_reference_libraries, DEFAULT_REF_HDF5
    if _DREAMS_AVAILABLE is True:
        return True
    if _DREAMS_AVAILABLE is False:
        return False
    try:
        from core.dreams_adapter import (
            spectra_to_mgf as _spectra_to_mgf,
            run_dreams_search as _run_dreams_search,
            list_reference_libraries as _list_reference_libraries,
            DEFAULT_REF_HDF5 as _default_ref_hdf5,
        )
        dreams_spectra_to_mgf = _spectra_to_mgf
        run_dreams_search = _run_dreams_search
        list_reference_libraries = _list_reference_libraries
        DEFAULT_REF_HDF5 = _default_ref_hdf5
        _DREAMS_AVAILABLE = True
        return True
    except Exception as _e:
        _DREAMS_AVAILABLE = False
        _DREAMS_IMPORT_ERROR = str(_e)
        print(f"[DreaMS] Not available: {_e}")
        return False


def _list_dreams_reference_libraries_light():
    libs = []
    if _LOCAL_DREAMS_REF_DIR.exists():
        libs.extend(_LOCAL_DREAMS_REF_DIR.glob("*.hdf5"))
    return sorted(str(p) for p in libs)

from uic import main_v4 as main

# ---------------- Paths ----------------
DATA_DIR = "data"
MODEL_DIR = "model"
STRU_DIR = "Stru"

INDEX_POS = os.path.join(DATA_DIR, "references_index_positive_spec2vec.bin")
INDEX_NEG = os.path.join(DATA_DIR, "references_index_negative_spec2vec.bin")

REF_POS = os.path.join(DATA_DIR, "references_spectrums_positive.pickle")
REF_NEG = os.path.join(DATA_DIR, "references_spectrums_negative.pickle")

W2V_POS = os.path.join(MODEL_DIR, "spec2vec_spectraverse_positive.model")
W2V_NEG = os.path.join(MODEL_DIR, "spec2vec_spectraverse_negative.model")

STRUCT_DB = os.path.join(STRU_DIR, "FlavonoidStructureDB-dedup.csv")

# Keep CPU headroom for the Qt UI and external model subprocesses.
MAX_WORKERS = min(max(os.cpu_count() or 1, 2), 4)

# Negative-ion public-library-similarity reference loading and the
# candidate-vs-library scoring function moved to
# core/annomass_algorithm.py (the only remaining caller) -- see
# annomass_algorithm._load_public_library_reference /
# annomass_algorithm._public_library_similarity.


def require_file(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")

def fmt_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


LIGHT_FILE_DIALOG_STYLE = """
    QFileDialog, QFileDialog QWidget {
        background: #f7f7f7;
        color: #202020;
        font-family: "Segoe UI","DejaVu Sans","Arial";
        font-size: 12px;
    }
    QFileDialog QLineEdit, QFileDialog QComboBox, QFileDialog QListView,
    QFileDialog QTreeView, QFileDialog QTableView, QFileDialog QAbstractItemView {
        background: #ffffff;
        color: #202020;
        border: 1px solid #b8b8b8;
        selection-background-color: #2f7dd1;
        selection-color: #ffffff;
    }
    QFileDialog QLabel {
        color: #202020;
    }
    QFileDialog QPushButton {
        background: #ffffff;
        color: #202020;
        border: 1px solid #a8a8a8;
        border-radius: 3px;
        padding: 4px 10px;
    }
    QFileDialog QPushButton:hover {
        background: #eaf3ff;
        border-color: #2f7dd1;
    }
"""


LIGHT_DIALOG_STYLE = """
    QDialog, QDialog QWidget {
        background: #f7f7f7;
        color: #202020;
        font-family: "Segoe UI","DejaVu Sans","Arial";
        font-size: 12px;
    }
    QDialog QLineEdit, QDialog QComboBox, QDialog QComboBox QAbstractItemView,
    QDialog QPlainTextEdit, QDialog QTextEdit {
        background: #ffffff;
        color: #202020;
        border: 1px solid #b8b8b8;
        selection-background-color: #2f7dd1;
        selection-color: #ffffff;
    }
    QDialog QLabel { color: #202020; background: transparent; }
    QDialog QGroupBox { color: #202020; }
    QDialog QCheckBox { color: #202020; }
    QDialog QPushButton {
        background: #ffffff;
        color: #202020;
        border: 1px solid #a8a8a8;
        border-radius: 3px;
        padding: 4px 10px;
    }
    QDialog QPushButton:hover {
        background: #eaf3ff;
        border-color: #2f7dd1;
    }
"""


_LAST_SAVE_DIR = None


def light_get_save_file_name(parent, caption, suggested, file_filter):
    """Styled non-native Save dialog; always returns an absolute path.

    `suggested` may be a bare filename ("annomass_results.xlsx") or a full
    path; bare names are resolved against the remembered last save directory
    (falling back to the current directory).  The dialog previously received
    the suggested name in QFileDialog's `directory` argument, which anchored
    the dialog to the app's working directory and made the user's browsed
    save location not take effect.
    """
    global _LAST_SAVE_DIR
    suggested = str(suggested or "")
    if os.sep in suggested or (os.altsep and os.altsep in suggested):
        start_dir, file_name = os.path.split(os.path.abspath(suggested))
    else:
        start_dir = _LAST_SAVE_DIR or os.getcwd()
        file_name = suggested
    dlg = QtWidgets.QFileDialog(parent, caption, start_dir, file_filter)
    dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setStyleSheet(LIGHT_FILE_DIALOG_STYLE)
    if file_name:
        dlg.selectFile(file_name)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return "", ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if path:
        if not os.path.isabs(path):
            path = os.path.join(dlg.directory().absolutePath(), path)
        path = os.path.abspath(path)
        _LAST_SAVE_DIR = os.path.dirname(path)
    return path, dlg.selectedNameFilter()


def light_get_open_file_name(parent, caption, directory, file_filter):
    dlg = QtWidgets.QFileDialog(parent, caption, directory, file_filter)
    dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
    dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setStyleSheet(LIGHT_FILE_DIALOG_STYLE)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return "", ""
    files = dlg.selectedFiles()
    return (files[0] if files else ""), dlg.selectedNameFilter()


def light_get_existing_directory(parent, caption, start_dir):
    """Styled non-native folder picker; returns '' when cancelled."""
    dlg = QtWidgets.QFileDialog(parent, caption, start_dir or os.getcwd())
    dlg.setFileMode(QtWidgets.QFileDialog.Directory)
    dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setStyleSheet(LIGHT_FILE_DIALOG_STYLE)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return ""
    files = dlg.selectedFiles()
    return files[0] if files else ""


class AnnoMassNavigationToolbar(NavigationToolbar):
    """Matplotlib toolbar with a dependable project-styled Save As dialog."""

    def save_figure(self, *args):
        filters = (
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg);;"
            "JPEG image (*.jpg *.jpeg);;TIFF image (*.tif *.tiff);;All files (*)"
        )
        path, selected_filter = light_get_save_file_name(
            self,
            "Save spectrum image as",
            "annomass_spectrum",
            filters,
        )
        if not path:
            return

        suffix = Path(path).suffix.lower()
        if not suffix:
            extension_by_filter = {
                "PNG image (*.png)": ".png",
                "PDF document (*.pdf)": ".pdf",
                "SVG image (*.svg)": ".svg",
                "JPEG image (*.jpg *.jpeg)": ".jpg",
                "TIFF image (*.tif *.tiff)": ".tif",
            }
            path += extension_by_filter.get(selected_filter, ".png")
        try:
            self.canvas.figure.savefig(path, bbox_inches="tight")
        except Exception as exc:
            light_message_box(
                self,
                QtWidgets.QMessageBox.Warning,
                "Save image",
                f"Could not save image:\n{exc}",
            )


LIGHT_MESSAGE_BOX_STYLE = """
    QMessageBox {
        background: #f6f6f6;
        color: #202020;
    }
    QMessageBox QLabel {
        color: #202020;
        background: transparent;
        min-width: 360px;
    }
    QMessageBox QPushButton {
        background: #ffffff;
        color: #202020;
        border: 1px solid #a8a8a8;
        border-radius: 3px;
        padding: 5px 18px;
        min-width: 72px;
    }
    QMessageBox QPushButton:hover {
        background: #eaf3ff;
        border-color: #2f7dd1;
    }
"""


GLOBAL_MENU_STYLE = """
    QMenu {
        background: #ffffff;
        color: #202020;
        border: 1px solid #b8b8b8;
        padding: 4px 0px;
    }
    QMenu::item {
        padding: 5px 24px 5px 20px;
        background: transparent;
        color: #202020;
    }
    QMenu::item:selected {
        background: #2f7dd1;
        color: #ffffff;
    }
    QMenu::item:disabled {
        color: #a0a0a0;
    }
    QMenu::separator {
        height: 1px;
        background: #d8d8d8;
        margin: 4px 8px;
    }
"""


def light_message_box(parent, icon, title, text):
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(str(title))
    msg.setText(str(text))
    msg.setStyleSheet(LIGHT_MESSAGE_BOX_STYLE)
    msg.exec_()


def light_confirm_box(parent, title, text):
    """Styled Yes/No question; returns True only on Yes."""
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Question)
    msg.setWindowTitle(str(title))
    msg.setText(str(text))
    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    msg.setDefaultButton(QtWidgets.QMessageBox.No)
    msg.setStyleSheet(LIGHT_MESSAGE_BOX_STYLE)
    return msg.exec_() == QtWidgets.QMessageBox.Yes


def _make_scrollbar(track, handle, handle_hover):
    return f"""
        QScrollBar:vertical {{ background: {track}; width: 8px; border-radius: 4px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {handle}; border-radius: 4px; min-height: 24px; }}
        QScrollBar::handle:vertical:hover {{ background: {handle_hover}; }}
        QScrollBar:horizontal {{ background: {track}; height: 8px; border-radius: 4px; margin: 0; }}
        QScrollBar::handle:horizontal {{ background: {handle}; border-radius: 4px; min-width: 24px; }}
        QScrollBar::handle:horizontal:hover {{ background: {handle_hover}; }}
        QScrollBar::add-line, QScrollBar::sub-line,
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; border: none; }}
    """

def _make_checkbox(border, checked_bg, checked_border, text_color):
    return f"""
        QCheckBox {{ color: {text_color}; spacing: 6px; }}
        QCheckBox::indicator {{
            width: 15px; height: 15px;
            border: 1.5px solid {border}; border-radius: 4px; background: transparent;
        }}
        QCheckBox::indicator:checked {{
            background: {checked_bg}; border-color: {checked_border};
            image: none;
        }}
        QCheckBox::indicator:hover {{ border-color: {checked_bg}; }}
    """

THEMES = {
    # ── 1. VS Dark+ ──────────────────────────────────────────────────────────
    "VS Dark+": """
        QMainWindow, QDialog { background: #1E1E1E; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Consolas","Arial";
            font-size: 12px; color: #CCCCCC; background: transparent;
        }
        /* Activity Bar */
        QFrame#activityBar {
            background: #333333;
            border-right: 1px solid #252526;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #333333; color: #CCCCCC;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #858585; font-size: 11px; font-weight: 500;
            padding: 0; margin: 0;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(255,255,255,12);
            color: #CCCCCC; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #007ACC; color: #FFFFFF;
            background: rgba(255,255,255,20);
        }
        QFrame#activityBar QPushButton:disabled {
            color: #555555;
        }
        QFrame#activityBar QFrame { background: #444444; }
        /* Sidebar */
        QFrame#leftPanel {
            background: #252526;
            border-right: none;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #252526; color: #BBBBBB;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #252526; border: none; outline: none;
            padding: 0; color: #CCCCCC;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #094771; color: #FFFFFF;
            border-left: 3px solid #007ACC;
        }
        QListWidget#list_spectrum::item:hover {
            background: #2A2D2E;
        }
        /* Main panel */
        QWidget#mainPanel { background: #1E1E1E; }
        /* Tab bar - VS Code editor tabs */
        QTabWidget::pane { border: none; background: #1E1E1E; top: 0; }
        QTabBar { background: #2D2D2D; }
        QTabBar::tab {
            background: #2D2D2D; color: #9D9D9D;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #252526;
            padding: 8px 16px; min-width: 80px;
            font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #1E1E1E; color: #FFFFFF;
            border-top: 2px solid #007ACC;
        }
        QTabBar::tab:!selected:hover {
            background: #2A2D2E; color: #CCCCCC;
        }
        /* GroupBox as panel sections */
        QGroupBox {
            background: #1E1E1E; border: 1px solid #3C3C3C;
            border-radius: 0px; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #9D9D9D; font-weight: 700; font-size: 10px;
            background: #1E1E1E; letter-spacing: 1px;
        }
        /* Sidebar GroupBoxes */
        QFrame#leftPanel QGroupBox {
            background: #252526; border: none;
            border-top: 1px solid #3C3C3C;
            margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title {
            background: #252526; color: #BBBBBB;
        }
        QLabel { color: #CCCCCC; background: transparent; }
        QLabel#lbl_time {
            background: #252526; color: #9D9D9D;
            border: none; border-top: 1px solid #3C3C3C;
            padding: 6px 12px; font-size: 10px;
        }
        /* Buttons */
        QPushButton {
            background: #0E639C; border: 1px solid #1177BB;
            padding: 5px 14px; border-radius: 2px;
            color: #FFFFFF; font-weight: 500; font-size: 12px;
        }
        QPushButton:hover { background: #1177BB; }
        QPushButton:pressed { background: #0D5A8E; }
        QPushButton:disabled { background: #3C3C3C; color: #6B6B6B; border-color: #3C3C3C; }
        /* Inputs */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #3C3C3C; border: 1px solid #555555;
            border-radius: 2px; padding: 4px 8px;
            color: #CCCCCC; selection-background-color: #264F78;
            font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #007ACC;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #3C3C3C; border: 1px solid #555555;
            selection-background-color: #094771; color: #CCCCCC; outline: none;
        }
        /* Tables */
        QTableWidget {
            background: #1E1E1E; border: none;
            gridline-color: #2D2D2D; color: #CCCCCC;
            alternate-background-color: #252526;
        }
        QHeaderView::section {
            background: #252526; color: #9D9D9D;
            border: none; border-bottom: 1px solid #3C3C3C;
            border-right: 1px solid #3C3C3C;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #252526; border: 1px solid #3C3C3C; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #264F78; color: #FFFFFF; }
        QTableWidget::item:hover { background: #2A2D2E; }
        /* Progress / Status bar */
        QFrame#bottomStrip { background: #007ACC; }
        QProgressBar {
            background: transparent; border: none;
            color: #FFFFFF; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: rgba(255,255,255,30); }
        QStatusBar { background: #1E1E1E; color: #CCCCCC; font-size: 11px; border-top: 1px solid #333333; }
        QStatusBar::item { border: none; }
        /* Splitter */
        QSplitter::handle { background: #3C3C3C; }
        QSplitter#splitter_main::handle { background: #007ACC; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #1F8AD3; }
        QToolTip {
            background: #252526; color: #CCCCCC;
            border: 1px solid #454545; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #1E1E1E; border: 1px solid #3C3C3C;
            color: #CCCCCC; selection-background-color: #264F78;
        }
        QScrollBar:vertical { background: #1E1E1E; width: 10px; }
        QScrollBar::handle:vertical { background: #424242; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #686868; }
        QScrollBar:horizontal { background: #1E1E1E; height: 10px; }
        QScrollBar::handle:horizontal { background: #424242; min-width: 20px; border-radius: 5px; }
        QScrollBar::handle:horizontal:hover { background: #686868; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #CCCCCC; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555555; background: transparent; }
        QCheckBox::indicator:checked { background: #007ACC; border-color: #007ACC; }
    """,

    # ── 2. VS Light+ ─────────────────────────────────────────────────────────
    "VS Light+": """
        QMainWindow, QDialog { background: #FFFFFF; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #3B3B3B; background: transparent;
        }
        QFrame#activityBar {
            background: #FFFFFF;
            border-right: 1px solid #E0E0E0;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #FFFFFF; color: #007ACC;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #555555; font-size: 11px; font-weight: 500;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(0,0,0,6);
            color: #222222; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #007ACC; color: #007ACC;
            background: rgba(0,122,204,8);
        }
        QFrame#activityBar QPushButton:disabled { color: #AAAAAA; }
        QFrame#activityBar QFrame { background: #E0E0E0; }
        QFrame#leftPanel {
            background: #F0F0F0;
            border-right: none;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #F0F0F0; color: #616161;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #F0F0F0; border: none; outline: none;
            padding: 0; color: #3B3B3B;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #D6EBFF; color: #3B3B3B;
            border-left: 3px solid #007ACC;
        }
        QListWidget#list_spectrum::item:hover { background: #E8E8E8; }
        QWidget#mainPanel { background: #FFFFFF; }
        QTabWidget::pane { border: none; background: #FFFFFF; top: 0; }
        QTabBar { background: #ECECEC; }
        QTabBar::tab {
            background: #ECECEC; color: #7A7A7A;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #D8D8D8;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #FFFFFF; color: #333333;
            border-top: 2px solid #007ACC;
        }
        QTabBar::tab:!selected:hover { background: #E3E3E3; color: #3B3B3B; }
        QGroupBox {
            background: #FFFFFF; border: 1px solid #E7E7E7;
            border-radius: 0; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #616161; font-weight: 700; font-size: 10px;
            background: #FFFFFF; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #F0F0F0; border: none;
            border-top: 1px solid #D0D0D0; margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title { background: #F0F0F0; color: #616161; }
        QLabel { color: #3B3B3B; background: transparent; }
        QLabel#lbl_time {
            background: #F0F0F0; color: #616161;
            border: none; border-top: 1px solid #D0D0D0;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #007ACC; border: 1px solid #006BB3;
            padding: 5px 14px; border-radius: 2px;
            color: #FFFFFF; font-weight: 500; font-size: 12px;
        }
        QPushButton:hover { background: #006BB3; }
        QPushButton:pressed { background: #005A99; }
        QPushButton:disabled { background: #CCCCCC; color: #999999; border-color: #CCCCCC; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #FFFFFF; border: 1px solid #CECECE;
            border-radius: 2px; padding: 4px 8px;
            color: #3B3B3B; selection-background-color: #ADD6FF; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #007ACC; outline: none;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #FFFFFF; border: 1px solid #CECECE;
            selection-background-color: #D6EBFF; color: #3B3B3B; outline: none;
        }
        QTableWidget {
            background: #FFFFFF; border: none;
            gridline-color: #F0F0F0; color: #3B3B3B;
            alternate-background-color: #F8F8F8;
        }
        QHeaderView::section {
            background: #F3F3F3; color: #7A7A7A;
            border: none; border-bottom: 1px solid #E7E7E7;
            border-right: 1px solid #E7E7E7;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #F3F3F3; border: 1px solid #E7E7E7; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #D6EBFF; color: #3B3B3B; }
        QTableWidget::item:hover { background: #E8E8E8; }
        QFrame#bottomStrip { background: #007ACC; }
        QProgressBar {
            background: transparent; border: none;
            color: #FFFFFF; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: rgba(255,255,255,30); }
        QStatusBar { background: #F3F3F3; color: #3B3B3B; font-size: 11px; border-top: 1px solid #E0E0E0; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #E7E7E7; }
        QSplitter#splitter_main::handle { background: #007ACC; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #1F8AD3; }
        QToolTip {
            background: #F3F3F3; color: #3B3B3B;
            border: 1px solid #D4D4D4; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #FFFFFF; border: 1px solid #E7E7E7;
            color: #3B3B3B; selection-background-color: #ADD6FF;
        }
        QScrollBar:vertical { background: #F3F3F3; width: 10px; }
        QScrollBar::handle:vertical { background: #C1C1C1; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #A8A8A8; }
        QScrollBar:horizontal { background: #F3F3F3; height: 10px; }
        QScrollBar::handle:horizontal { background: #C1C1C1; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #3B3B3B; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #CECECE; background: transparent; }
        QCheckBox::indicator:checked { background: #007ACC; border-color: #007ACC; }
    """,

    # ── 3. One Dark Pro ───────────────────────────────────────────────────────
    "One Dark Pro": """
        QMainWindow, QDialog { background: #282C34; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #ABB2BF; background: transparent;
        }
        QFrame#activityBar {
            background: #21252B;
            border-right: 1px solid #181A1F;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #21252B; color: #61AFEF;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #636D83; font-size: 11px; font-weight: 500;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(255,255,255,8);
            color: #ABB2BF; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #61AFEF; color: #61AFEF;
            background: rgba(97,175,239,10);
        }
        QFrame#activityBar QPushButton:disabled { color: #3E4452; }
        QFrame#activityBar QFrame { background: #3E4452; }
        QFrame#leftPanel {
            background: #21252B;
            border-right: none;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #21252B; color: #636D83;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #21252B; border: none; outline: none;
            padding: 0; color: #ABB2BF;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #2C313A; color: #61AFEF;
            border-left: 3px solid #61AFEF;
        }
        QListWidget#list_spectrum::item:hover { background: #2C313C; }
        QWidget#mainPanel { background: #282C34; }
        QTabWidget::pane { border: none; background: #282C34; top: 0; }
        QTabBar { background: #21252B; }
        QTabBar::tab {
            background: #21252B; color: #636D83;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #181A1F;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #282C34; color: #ABB2BF;
            border-top: 2px solid #61AFEF;
        }
        QTabBar::tab:!selected:hover { background: #2C313A; color: #ABB2BF; }
        QGroupBox {
            background: #282C34; border: 1px solid #3E4452;
            border-radius: 0; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #636D83; font-weight: 700; font-size: 10px;
            background: #282C34; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #21252B; border: none;
            border-top: 1px solid #3E4452; margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title { background: #21252B; color: #636D83; }
        QLabel { color: #ABB2BF; background: transparent; }
        QLabel#lbl_time {
            background: #21252B; color: #636D83;
            border: none; border-top: 1px solid #3E4452;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #61AFEF; border: none;
            padding: 5px 14px; border-radius: 4px;
            color: #282C34; font-weight: 600; font-size: 12px;
        }
        QPushButton:hover { background: #528BBA; }
        QPushButton:pressed { background: #4070A0; }
        QPushButton:disabled { background: #3E4452; color: #636D83; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #1D2026; border: 1px solid #3E4452;
            border-radius: 4px; padding: 4px 8px;
            color: #ABB2BF; selection-background-color: #3E4452; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #61AFEF;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #21252B; border: 1px solid #3E4452;
            selection-background-color: #2C313A; color: #ABB2BF; outline: none;
        }
        QTableWidget {
            background: #282C34; border: none;
            gridline-color: #3E4452; color: #ABB2BF;
            alternate-background-color: #2C313A;
        }
        QHeaderView::section {
            background: #21252B; color: #636D83;
            border: none; border-bottom: 1px solid #3E4452;
            border-right: 1px solid #3E4452;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #21252B; border: 1px solid #3E4452; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #2C313A; color: #61AFEF; }
        QTableWidget::item:hover { background: #2C313C; }
        QFrame#bottomStrip { background: #21252B; border-top: 1px solid #181A1F; }
        QProgressBar {
            background: transparent; border: none;
            color: #636D83; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: #61AFEF; }
        QStatusBar { background: #21252B; color: #ABB2BF; font-size: 11px; border-top: 1px solid #181A1F; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #3E4452; }
        QSplitter#splitter_main::handle { background: #61AFEF; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #74C0F4; }
        QToolTip {
            background: #21252B; color: #ABB2BF;
            border: 1px solid #3E4452; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #1D2026; border: 1px solid #3E4452;
            color: #ABB2BF; selection-background-color: #3E4452;
        }
        QScrollBar:vertical { background: #21252B; width: 10px; }
        QScrollBar::handle:vertical { background: #4B5263; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #636D83; }
        QScrollBar:horizontal { background: #21252B; height: 10px; }
        QScrollBar::handle:horizontal { background: #4B5263; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #ABB2BF; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #4B5263; background: transparent; }
        QCheckBox::indicator:checked { background: #61AFEF; border-color: #61AFEF; }
    """,

    # ── 4. Monokai ────────────────────────────────────────────────────────────
    "Monokai": """
        QMainWindow, QDialog { background: #272822; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #F8F8F2; background: transparent;
        }
        QFrame#activityBar {
            background: #1E1F1C;
            border-right: 1px solid #1A1B18;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #1E1F1C; color: #A6E22E;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #75715E; font-size: 11px; font-weight: 500;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(255,255,255,8);
            color: #F8F8F2; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #A6E22E; color: #A6E22E;
            background: rgba(166,226,46,10);
        }
        QFrame#activityBar QPushButton:disabled { color: #49483E; }
        QFrame#activityBar QFrame { background: #49483E; }
        QFrame#leftPanel {
            background: #1E1F1C;
            border-right: none;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #1E1F1C; color: #75715E;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #1E1F1C; border: none; outline: none;
            padding: 0; color: #F8F8F2;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #49483E; color: #A6E22E;
            border-left: 3px solid #A6E22E;
        }
        QListWidget#list_spectrum::item:hover { background: #3E3D32; }
        QWidget#mainPanel { background: #272822; }
        QTabWidget::pane { border: none; background: #272822; top: 0; }
        QTabBar { background: #1E1F1C; }
        QTabBar::tab {
            background: #1E1F1C; color: #75715E;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #1A1B18;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #272822; color: #F8F8F2;
            border-top: 2px solid #A6E22E;
        }
        QTabBar::tab:!selected:hover { background: #3E3D32; color: #F8F8F2; }
        QGroupBox {
            background: #272822; border: 1px solid #3E3D32;
            border-radius: 0; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #75715E; font-weight: 700; font-size: 10px;
            background: #272822; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #1E1F1C; border: none;
            border-top: 1px solid #3E3D32; margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title { background: #1E1F1C; color: #75715E; }
        QLabel { color: #F8F8F2; background: transparent; }
        QLabel#lbl_time {
            background: #1E1F1C; color: #75715E;
            border: none; border-top: 1px solid #3E3D32;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #A6E22E; border: none;
            padding: 5px 14px; border-radius: 4px;
            color: #272822; font-weight: 700; font-size: 12px;
        }
        QPushButton:hover { background: #8EB82A; }
        QPushButton:pressed { background: #7AA024; }
        QPushButton:disabled { background: #3E3D32; color: #75715E; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #1E1F1C; border: 1px solid #49483E;
            border-radius: 4px; padding: 4px 8px;
            color: #F8F8F2; selection-background-color: #49483E; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #A6E22E;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #1E1F1C; border: 1px solid #49483E;
            selection-background-color: #49483E; color: #F8F8F2; outline: none;
        }
        QTableWidget {
            background: #272822; border: none;
            gridline-color: #3E3D32; color: #F8F8F2;
            alternate-background-color: #2D2E27;
        }
        QHeaderView::section {
            background: #1E1F1C; color: #75715E;
            border: none; border-bottom: 1px solid #49483E;
            border-right: 1px solid #49483E;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #1E1F1C; border: 1px solid #49483E; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #49483E; color: #A6E22E; }
        QTableWidget::item:hover { background: #3E3D32; }
        QFrame#bottomStrip { background: #1E1F1C; border-top: 1px solid #1A1B18; }
        QProgressBar {
            background: transparent; border: none;
            color: #75715E; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: #A6E22E; }
        QStatusBar { background: #1E1F1C; color: #F8F8F2; font-size: 11px; border-top: 1px solid #1A1B18; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #3E3D32; }
        QSplitter#splitter_main::handle { background: #A6E22E; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #C3F56C; }
        QToolTip {
            background: #1E1F1C; color: #F8F8F2;
            border: 1px solid #49483E; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #1E1F1C; border: 1px solid #49483E;
            color: #F8F8F2; selection-background-color: #49483E;
        }
        QScrollBar:vertical { background: #1E1F1C; width: 10px; }
        QScrollBar::handle:vertical { background: #49483E; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #75715E; }
        QScrollBar:horizontal { background: #1E1F1C; height: 10px; }
        QScrollBar::handle:horizontal { background: #49483E; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #F8F8F2; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #49483E; background: transparent; }
        QCheckBox::indicator:checked { background: #A6E22E; border-color: #A6E22E; }
    """,

    # ── 5. Solarized Light ────────────────────────────────────────────────────
    "Solarized Light": """
        QMainWindow, QDialog { background: #FDF6E3; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #657B83; background: transparent;
        }
        QFrame#activityBar {
            background: #FDF6E3;
            border-right: 1px solid #D3CBB8;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #FDF6E3; color: #268BD2;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #586E75; font-size: 11px; font-weight: 500;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(0,0,0,6);
            color: #2E4057; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #268BD2; color: #268BD2;
            background: rgba(38,139,210,8);
        }
        QFrame#activityBar QPushButton:disabled { color: #B0AFA8; }
        QFrame#activityBar QFrame { background: #D3CBB8; }
        QFrame#leftPanel {
            background: #EEE8D5;
            border-right: none;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #EEE8D5; color: #93A1A1;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #EEE8D5; border: none; outline: none;
            padding: 0; color: #657B83;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #D0C8B0; color: #268BD2;
            border-left: 3px solid #268BD2;
        }
        QListWidget#list_spectrum::item:hover { background: #E2DBD0; }
        QWidget#mainPanel { background: #FDF6E3; }
        QTabWidget::pane { border: none; background: #FDF6E3; top: 0; }
        QTabBar { background: #EEE8D5; }
        QTabBar::tab {
            background: #EEE8D5; color: #93A1A1;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #D3CBB8;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #FDF6E3; color: #586E75;
            border-top: 2px solid #268BD2;
        }
        QTabBar::tab:!selected:hover { background: #E2DBD0; color: #586E75; }
        QGroupBox {
            background: #FDF6E3; border: 1px solid #D3CBB8;
            border-radius: 0; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #93A1A1; font-weight: 700; font-size: 10px;
            background: #FDF6E3; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #EEE8D5; border: none;
            border-top: 1px solid #D3CBB8; margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title { background: #EEE8D5; color: #93A1A1; }
        QLabel { color: #657B83; background: transparent; }
        QLabel#lbl_time {
            background: #EEE8D5; color: #93A1A1;
            border: none; border-top: 1px solid #D3CBB8;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #268BD2; border: none;
            padding: 5px 14px; border-radius: 2px;
            color: #FDF6E3; font-weight: 500; font-size: 12px;
        }
        QPushButton:hover { background: #2176AE; }
        QPushButton:pressed { background: #1A6090; }
        QPushButton:disabled { background: #D3CBB8; color: #93A1A1; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #EEE8D5; border: 1px solid #D3CBB8;
            border-radius: 2px; padding: 4px 8px;
            color: #657B83; selection-background-color: #D0C8B0; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #268BD2;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #EEE8D5; border: 1px solid #D3CBB8;
            selection-background-color: #D0C8B0; color: #657B83; outline: none;
        }
        QTableWidget {
            background: #FDF6E3; border: none;
            gridline-color: #EEE8D5; color: #657B83;
            alternate-background-color: #F5EDDB;
        }
        QHeaderView::section {
            background: #EEE8D5; color: #93A1A1;
            border: none; border-bottom: 1px solid #D3CBB8;
            border-right: 1px solid #D3CBB8;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #EEE8D5; border: 1px solid #D3CBB8; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #D0C8B0; color: #268BD2; }
        QTableWidget::item:hover { background: #E2DBD0; }
        QFrame#bottomStrip { background: #268BD2; }
        QProgressBar {
            background: transparent; border: none;
            color: #FDF6E3; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: rgba(255,255,255,30); }
        QStatusBar { background: #EEE8D5; color: #657B83; font-size: 11px; border-top: 1px solid #D3CBB8; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #D3CBB8; }
        QSplitter#splitter_main::handle { background: #268BD2; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #3EA8E0; }
        QToolTip {
            background: #073642; color: #93A1A1;
            border: 1px solid #002B36; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #FDF6E3; border: 1px solid #D3CBB8;
            color: #657B83; selection-background-color: #D0C8B0;
        }
        QScrollBar:vertical { background: #EEE8D5; width: 10px; }
        QScrollBar::handle:vertical { background: #C2BAA6; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #A8A090; }
        QScrollBar:horizontal { background: #EEE8D5; height: 10px; }
        QScrollBar::handle:horizontal { background: #C2BAA6; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #657B83; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #D3CBB8; background: transparent; }
        QCheckBox::indicator:checked { background: #268BD2; border-color: #268BD2; }
    """,

    # ── 6. Mint Green ─────────────────────────────────────────────────────────
    "Mint Green": """
        QMainWindow, QDialog { background: #F0FAF5; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #2D4A3E; background: transparent;
        }
        QFrame#activityBar {
            background: #F0FAF5;
            border-right: 1px solid #C3E6CB;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #F0FAF5; color: #2D6A4F;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #40916C; font-size: 11px; font-weight: 500;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(0,0,0,5);
            color: #1B4332; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #52B788; color: #2D6A4F;
            background: rgba(82,183,136,10);
        }
        QFrame#activityBar QPushButton:disabled { color: #AACFBC; }
        QFrame#activityBar QFrame { background: #C3E6CB; }
        QFrame#leftPanel {
            background: #F0FAF5;
            border-right: 1px solid #B7E4C7;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #F0FAF5; color: #40916C;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #F0FAF5; border: none; outline: none;
            padding: 0; color: #2D4A3E;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #B7E4C7; color: #1B4332;
            border-left: 3px solid #52B788;
        }
        QListWidget#list_spectrum::item:hover { background: #C8EDD4; }
        QWidget#mainPanel { background: #F0FAF5; }
        QTabWidget::pane { border: none; background: #F0FAF5; top: 0; }
        QTabBar { background: #D8F3DC; }
        QTabBar::tab {
            background: #D8F3DC; color: #52796F;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #B7E4C7;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #F0FAF5; color: #1B4332;
            border-top: 2px solid #52B788;
        }
        QTabBar::tab:!selected:hover { background: #C8EDD4; color: #2D4A3E; }
        QGroupBox {
            background: #F0FAF5; border: 1px solid #B7E4C7;
            border-radius: 0; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #52796F; font-weight: 700; font-size: 10px;
            background: #F0FAF5; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #F0FAF5; border: none;
            border-top: 1px solid #B7E4C7; margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title { background: #F0FAF5; color: #40916C; }
        QLabel { color: #2D4A3E; background: transparent; }
        QLabel#lbl_time {
            background: #F0FAF5; color: #52796F;
            border: none; border-top: 1px solid #B7E4C7;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #52B788; border: none;
            padding: 5px 14px; border-radius: 4px;
            color: #FFFFFF; font-weight: 600; font-size: 12px;
        }
        QPushButton:hover { background: #40916C; }
        QPushButton:pressed { background: #2D6A4F; }
        QPushButton:disabled { background: #B7E4C7; color: #74C69D; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #FFFFFF; border: 1px solid #B7E4C7;
            border-radius: 4px; padding: 4px 8px;
            color: #2D4A3E; selection-background-color: #D8F3DC; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #52B788;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #FFFFFF; border: 1px solid #B7E4C7;
            selection-background-color: #D8F3DC; color: #2D4A3E; outline: none;
        }
        QTableWidget {
            background: #F0FAF5; border: none;
            gridline-color: #D8F3DC; color: #2D4A3E;
            alternate-background-color: #E8F5EC;
        }
        QHeaderView::section {
            background: #D8F3DC; color: #40916C;
            border: none; border-bottom: 1px solid #B7E4C7;
            border-right: 1px solid #B7E4C7;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #D8F3DC; border: 1px solid #B7E4C7; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #B7E4C7; color: #1B4332; }
        QTableWidget::item:hover { background: #C8EDD4; }
        QFrame#bottomStrip { background: #52B788; }
        QProgressBar {
            background: transparent; border: none;
            color: #FFFFFF; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: rgba(255,255,255,30); }
        QStatusBar { background: #F0FAF5; color: #2D4A3E; font-size: 11px; border-top: 1px solid #C3E6CB; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #B7E4C7; }
        QSplitter#splitter_main::handle { background: #52B788; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #6ECFA1; }
        QToolTip {
            background: #D8F3DC; color: #2D4A3E;
            border: 1px solid #B7E4C7; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #F0FAF5; border: 1px solid #B7E4C7;
            color: #2D4A3E; selection-background-color: #D8F3DC;
        }
        QScrollBar:vertical { background: #D8F3DC; width: 10px; }
        QScrollBar::handle:vertical { background: #95D5B2; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #74C69D; }
        QScrollBar:horizontal { background: #D8F3DC; height: 10px; }
        QScrollBar::handle:horizontal { background: #95D5B2; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #2D4A3E; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #B7E4C7; background: transparent; }
        QCheckBox::indicator:checked { background: #52B788; border-color: #52B788; }
    """,

    # ── 7. Soft Purple ────────────────────────────────────────────────────────
    "Soft Purple": """
        QMainWindow, QDialog { background: #F8F5FF; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #3D2C6E; background: transparent;
        }
        QFrame#activityBar {
            background: #F8F5FF;
            border-right: 1px solid #DDD6FE;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #F8F5FF; color: #6D28D9;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #5B21B6; font-size: 11px; font-weight: 500;
        }
        QFrame#activityBar QPushButton:hover {
            background: rgba(0,0,0,5);
            color: #3D2C6E; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #8B5CF6; color: #6D28D9;
            background: rgba(139,92,246,10);
        }
        QFrame#activityBar QPushButton:disabled { color: #C4B5FD; }
        QFrame#activityBar QFrame { background: #DDD6FE; }
        QFrame#leftPanel {
            background: #F8F5FF;
            border-right: 1px solid #DDD6FE;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #F8F5FF; color: #7C3AED;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #F8F5FF; border: none; outline: none;
            padding: 0; color: #3D2C6E;
        }
        QListWidget#list_spectrum::item {
            padding: 5px 12px; border-left: 3px solid transparent;
        }
        QListWidget#list_spectrum::item:selected {
            background: #DDD6FE; color: #2D1B69;
            border-left: 3px solid #8B5CF6;
        }
        QListWidget#list_spectrum::item:hover { background: #E5DEFF; }
        QWidget#mainPanel { background: #F8F5FF; }
        QTabWidget::pane { border: none; background: #F8F5FF; top: 0; }
        QTabBar { background: #EDE9FE; }
        QTabBar::tab {
            background: #EDE9FE; color: #7C5CBF;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #DDD6FE;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #F8F5FF; color: #2D1B69;
            border-top: 2px solid #8B5CF6;
        }
        QTabBar::tab:!selected:hover { background: #E5DEFF; color: #3D2C6E; }
        QGroupBox {
            background: #F8F5FF; border: 1px solid #DDD6FE;
            border-radius: 0; margin-top: 20px; padding: 8px 6px 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 6px;
            color: #7C5CBF; font-weight: 700; font-size: 10px;
            background: #F8F5FF; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #F8F5FF; border: none;
            border-top: 1px solid #DDD6FE; margin-top: 0; border-radius: 0;
        }
        QFrame#leftPanel QGroupBox::title { background: #F8F5FF; color: #7C3AED; }
        QLabel { color: #3D2C6E; background: transparent; }
        QLabel#lbl_time {
            background: #F8F5FF; color: #7C5CBF;
            border: none; border-top: 1px solid #DDD6FE;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #8B5CF6; border: none;
            padding: 5px 14px; border-radius: 4px;
            color: #FFFFFF; font-weight: 600; font-size: 12px;
        }
        QPushButton:hover { background: #7C3AED; }
        QPushButton:pressed { background: #6D28D9; }
        QPushButton:disabled { background: #DDD6FE; color: #A78BFA; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #FFFFFF; border: 1px solid #DDD6FE;
            border-radius: 4px; padding: 4px 8px;
            color: #3D2C6E; selection-background-color: #EDE9FE; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #8B5CF6;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #FFFFFF; border: 1px solid #DDD6FE;
            selection-background-color: #EDE9FE; color: #3D2C6E; outline: none;
        }
        QTableWidget {
            background: #F8F5FF; border: none;
            gridline-color: #EDE9FE; color: #3D2C6E;
            alternate-background-color: #F2EEFF;
        }
        QHeaderView::section {
            background: #EDE9FE; color: #7C5CBF;
            border: none; border-bottom: 1px solid #DDD6FE;
            border-right: 1px solid #DDD6FE;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #EDE9FE; border: 1px solid #DDD6FE; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #DDD6FE; color: #2D1B69; }
        QTableWidget::item:hover { background: #E5DEFF; }
        QFrame#bottomStrip { background: #8B5CF6; }
        QProgressBar {
            background: transparent; border: none;
            color: #FFFFFF; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: rgba(255,255,255,30); }
        QStatusBar { background: #F8F5FF; color: #3D2C6E; font-size: 11px; border-top: 1px solid #DDD6FE; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #DDD6FE; }
        QSplitter#splitter_main::handle { background: #8B5CF6; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #A985FA; }
        QToolTip {
            background: #EDE9FE; color: #3D2C6E;
            border: 1px solid #DDD6FE; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #F8F5FF; border: 1px solid #DDD6FE;
            color: #3D2C6E; selection-background-color: #EDE9FE;
        }
        QScrollBar:vertical { background: #EDE9FE; width: 10px; }
        QScrollBar::handle:vertical { background: #C4B5FD; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #A78BFA; }
        QScrollBar:horizontal { background: #EDE9FE; height: 10px; }
        QScrollBar::handle:horizontal { background: #C4B5FD; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #3D2C6E; spacing: 6px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #DDD6FE; background: transparent; }
        QCheckBox::indicator:checked { background: #8B5CF6; border-color: #8B5CF6; }
    """,

    # ── 8. Sky (OmicOS-inspired) ─────────────────────────────────────────────
    # Light, airy canvas + soft blue accent + generously rounded ("pill")
    # buttons/inputs/badges, modeled after a reference screenshot of OmicOS
    # (a separate Windows app the user liked the look of) -- same selector
    # coverage as the other themes, just softer radii/spacing/palette.
    "Sky": """
        QMainWindow, QDialog { background: #F7F8FA; }
        QWidget {
            font-family: "Segoe UI","DejaVu Sans","Arial";
            font-size: 12px; color: #1F2430; background: transparent;
        }
        QFrame#activityBar {
            background: #FFFFFF;
            border-right: 1px solid #ECEEF2;
        }
        QFrame#activityBar QLabel#label_logo {
            background: #FFFFFF; color: #4A7FE8;
            font-size: 13px; font-weight: 700;
        }
        QFrame#activityBar QPushButton {
            background: transparent; border: none;
            border-left: 3px solid transparent;
            color: #6B7280; font-size: 11px; font-weight: 500;
            border-radius: 0;
        }
        QFrame#activityBar QPushButton:hover {
            background: #EEF3FF; color: #333A45; border-left: 3px solid transparent;
        }
        QFrame#activityBar QPushButton:pressed {
            border-left: 3px solid #4A7FE8; color: #4A7FE8; background: #E7EFFF;
        }
        QFrame#activityBar QPushButton:disabled { color: #B7BCC6; }
        QFrame#activityBar QFrame { background: #ECEEF2; }
        QFrame#leftPanel {
            background: #F7F8FA;
            border-right: 1px solid #ECEEF2;
        }
        QLabel#label_list_title, QLabel#label_spectra_title {
            background: #F7F8FA; color: #8A8F98;
            font-size: 10px; font-weight: 700; letter-spacing: 1px;
            padding: 8px 12px 4px 12px;
        }
        QListWidget#list_spectrum {
            background: #F7F8FA; border: none; outline: none;
            padding: 4px; color: #1F2430;
        }
        QListWidget#list_spectrum::item {
            padding: 6px 12px; margin: 1px 4px; border-radius: 8px;
            border-left: none;
        }
        QListWidget#list_spectrum::item:selected {
            background: #E7EFFF; color: #2A56C6;
        }
        QListWidget#list_spectrum::item:hover { background: #EEF3FF; }
        /* Agent session list: OmicOS-style grouped history (see
           _agent_rebuild_session_list) -- group-header rows are added with
           Qt.NoItemFlags, which Qt maps to the :disabled item pseudo-state
           below, so they read as small gray section labels rather than
           clickable rows. */
        QListWidget#agent_session_list {
            background: #F7F8FA; border: none; outline: none; padding: 2px;
        }
        QListWidget#agent_session_list::item {
            padding: 6px 10px; margin: 1px 2px; border-radius: 8px; color: #1F2430;
        }
        QListWidget#agent_session_list::item:selected { background: #E7EFFF; color: #2A56C6; }
        QListWidget#agent_session_list::item:hover:!disabled { background: #EEF3FF; }
        QListWidget#agent_session_list::item:disabled {
            color: #8A8F98; font-size: 10px; font-weight: 700;
            padding-top: 10px; padding-bottom: 2px;
        }
        QWidget#mainPanel { background: #FFFFFF; }
        QTabWidget::pane { border: none; background: #FFFFFF; top: 0; }
        QTabBar { background: #F7F8FA; }
        QTabBar::tab {
            background: #F7F8FA; color: #8A8F98;
            border: none; border-top: 2px solid transparent;
            border-right: 1px solid #ECEEF2;
            padding: 8px 16px; min-width: 80px; font-size: 12px;
        }
        QTabBar::tab:selected {
            background: #FFFFFF; color: #1F2430;
            border-top: 2px solid #4A7FE8;
        }
        QTabBar::tab:!selected:hover { background: #EEF3FF; color: #1F2430; }
        QGroupBox {
            background: #FFFFFF; border: 1px solid #ECEEF2;
            border-radius: 10px; margin-top: 20px; padding: 10px 8px 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px; padding: 0 6px;
            color: #8A8F98; font-weight: 700; font-size: 10px;
            background: #FFFFFF; letter-spacing: 1px;
        }
        QFrame#leftPanel QGroupBox {
            background: #FFFFFF; border: 1px solid #ECEEF2;
            margin: 8px; border-radius: 10px;
        }
        QFrame#leftPanel QGroupBox::title { background: #FFFFFF; color: #8A8F98; }
        QLabel { color: #1F2430; background: transparent; }
        QLabel#lbl_time {
            background: #F7F8FA; color: #8A8F98;
            border: none; border-top: 1px solid #ECEEF2;
            padding: 6px 12px; font-size: 10px;
        }
        QPushButton {
            background: #4A7FE8; border: 1px solid #4A7FE8;
            padding: 6px 16px; border-radius: 14px;
            color: #FFFFFF; font-weight: 600; font-size: 12px;
        }
        QPushButton:hover { background: #3F6FD1; border-color: #3F6FD1; }
        QPushButton:pressed { background: #35609C; border-color: #35609C; }
        QPushButton:disabled { background: #E5E8EF; color: #AEB3BD; border-color: #E5E8EF; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #FFFFFF; border: 1px solid #E5E8EF;
            border-radius: 10px; padding: 5px 10px;
            color: #1F2430; selection-background-color: #CBDCFF; font-size: 12px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #4A7FE8; outline: none;
        }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView {
            background: #FFFFFF; border: 1px solid #E5E8EF; border-radius: 8px;
            selection-background-color: #E7EFFF; color: #1F2430; outline: none;
        }
        QTableWidget {
            background: #FFFFFF; border: none;
            gridline-color: #F1F3F7; color: #1F2430;
            alternate-background-color: #FAFBFD;
        }
        QHeaderView::section {
            background: #FAFBFD; color: #8A8F98;
            border: none; border-bottom: 1px solid #ECEEF2;
            border-right: 1px solid #ECEEF2;
            padding: 5px 8px; font-weight: 700; font-size: 11px;
        }
        QTableView QTableCornerButton::section { background: #FAFBFD; border: 1px solid #ECEEF2; }
        QTableWidget::item { padding: 4px 8px; }
        QTableWidget::item:selected { background: #E7EFFF; color: #1F2430; }
        QTableWidget::item:hover { background: #F3F6FC; }
        QFrame#bottomStrip { background: #4A7FE8; }
        QProgressBar {
            background: transparent; border: none;
            color: #FFFFFF; font-size: 11px; font-weight: 500;
            text-align: left; min-height: 20px;
        }
        QProgressBar::chunk { background: rgba(255,255,255,35); }
        QStatusBar { background: #FFFFFF; color: #6B7280; font-size: 11px; border-top: 1px solid #ECEEF2; }
        QStatusBar::item { border: none; }
        QSplitter::handle { background: #ECEEF2; }
        QSplitter#splitter_main::handle { background: #4A7FE8; width: 6px;}
        QSplitter#splitter_main::handle:hover { background: #3F6FD1; }
        QToolTip {
            background: #1F2430; color: #F7F8FA;
            border: 1px solid #1F2430; border-radius: 6px; padding: 4px 8px; font-size: 11px;
        }
        QPlainTextEdit, QTextEdit {
            background: #FFFFFF; border: 1px solid #E5E8EF; border-radius: 10px;
            color: #1F2430; selection-background-color: #CBDCFF;
        }
        QScrollBar:vertical { background: #F7F8FA; width: 10px; }
        QScrollBar::handle:vertical { background: #D7DBE3; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #B7BCC6; }
        QScrollBar:horizontal { background: #F7F8FA; height: 10px; }
        QScrollBar::handle:horizontal { background: #D7DBE3; min-width: 20px; border-radius: 5px; }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        QCheckBox { color: #1F2430; spacing: 6px; }
        QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px; border: 1.5px solid #C7CCD6; background: transparent; }
        QCheckBox::indicator:checked { background: #4A7FE8; border-color: #4A7FE8; }
    """,
}

CODEX_DARK_THEME = """
    QMainWindow, QDialog { background: #09090B; }
    QWidget {
        font-family: "Inter","Segoe UI","DejaVu Sans","Arial";
        font-size: 12px; color: #E4E4E7; background: transparent;
    }
    QFrame#activityBar {
        background: #0C0C0F;
        border-right: 1px solid #242428;
    }
    QFrame#activityBar QLabel#label_logo {
        background: #0C0C0F; color: #FAFAFA;
        font-size: 13px; font-weight: 700;
    }
    QFrame#activityBar QPushButton {
        background: transparent; border: none;
        border-left: 2px solid transparent;
        color: #A1A1AA; font-size: 11px; font-weight: 500;
        padding: 6px 4px;
    }
    QFrame#activityBar QPushButton:hover {
        background: #18181B; color: #FAFAFA;
    }
    QFrame#activityBar QPushButton:pressed {
        background: #202024; color: #FAFAFA; border-left: 2px solid #F4F4F5;
    }
    QFrame#activityBar QPushButton:disabled { color: #52525B; }
    QFrame#activityBar QFrame { background: #27272A; }
    QFrame#leftPanel {
        background: #0F0F12;
        border-right: 1px solid #27272A;
    }
    QLabel#label_list_title, QLabel#label_spectra_title {
        background: #0F0F12; color: #71717A;
        font-size: 10px; font-weight: 700; letter-spacing: 1px;
        padding: 8px 12px 4px 12px;
    }
    QListWidget#list_spectrum {
        background: #0F0F12; border: none; outline: none;
        padding: 4px; color: #D4D4D8;
    }
    QListWidget#list_spectrum::item {
        padding: 6px 10px; margin: 1px 3px; border-radius: 6px;
    }
    QListWidget#list_spectrum::item:selected {
        background: #27272A; color: #FAFAFA;
    }
    QListWidget#list_spectrum::item:hover { background: #18181B; }
    QWidget#mainPanel { background: #09090B; }
    QTabWidget::pane { border: none; background: #09090B; top: 0; }
    QTabBar { background: #0F0F12; }
    QTabBar::tab {
        background: #0F0F12; color: #A1A1AA;
        border: none; border-top: 2px solid transparent;
        border-right: 1px solid #27272A;
        padding: 8px 16px; min-width: 80px; font-size: 12px;
    }
    QTabBar::tab:selected {
        background: #09090B; color: #FAFAFA;
        border-top: 2px solid #FAFAFA;
    }
    QTabBar::tab:!selected:hover { background: #18181B; color: #E4E4E7; }
    QGroupBox {
        background: #0F0F12; border: 1px solid #27272A;
        border-radius: 8px; margin-top: 20px; padding: 10px 8px 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; left: 10px; padding: 0 6px;
        color: #A1A1AA; font-weight: 700; font-size: 10px;
        background: #0F0F12; letter-spacing: 1px;
    }
    QFrame#leftPanel QGroupBox {
        background: #141417; border: 1px solid #27272A;
        margin: 8px; border-radius: 8px;
    }
    QFrame#leftPanel QGroupBox::title { background: #141417; color: #A1A1AA; }
    QLabel { color: #E4E4E7; background: transparent; }
    QLabel#lbl_time {
        background: #0F0F12; color: #A1A1AA;
        border: none; border-top: 1px solid #27272A;
        padding: 6px 12px; font-size: 10px;
    }
    QPushButton {
        background: #FAFAFA; border: 1px solid #FAFAFA;
        padding: 6px 14px; border-radius: 6px;
        color: #09090B; font-weight: 600; font-size: 12px;
    }
    QPushButton:hover { background: #E4E4E7; border-color: #E4E4E7; }
    QPushButton:pressed { background: #D4D4D8; border-color: #D4D4D8; }
    QPushButton:disabled { background: #27272A; color: #71717A; border-color: #27272A; }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background: #141417; border: 1px solid #3F3F46;
        border-radius: 6px; padding: 5px 9px;
        color: #FAFAFA; selection-background-color: #3F3F46; font-size: 12px;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border-color: #A1A1AA; outline: none;
    }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView {
        background: #18181B; border: 1px solid #3F3F46;
        selection-background-color: #27272A; color: #FAFAFA; outline: none;
    }
    QTableWidget {
        background: #09090B; border: none;
        gridline-color: #27272A; color: #E4E4E7;
        alternate-background-color: #111114;
    }
    QHeaderView::section {
        background: #141417; color: #A1A1AA;
        border: none; border-bottom: 1px solid #27272A;
        border-right: 1px solid #27272A;
        padding: 5px 8px; font-weight: 700; font-size: 11px;
    }
    QTableView QTableCornerButton::section { background: #141417; border: 1px solid #27272A; }
    QTableWidget::item { padding: 4px 8px; }
    QTableWidget::item:selected { background: #27272A; color: #FAFAFA; }
    QTableWidget::item:hover { background: #18181B; }
    QFrame#bottomStrip { background: #18181B; border-top: 1px solid #27272A; }
    QProgressBar {
        background: transparent; border: none;
        color: #D4D4D8; font-size: 11px; font-weight: 500;
        text-align: left; min-height: 20px;
    }
    QProgressBar::chunk { background: #3F3F46; }
    QStatusBar { background: #09090B; color: #A1A1AA; font-size: 11px; border-top: 1px solid #27272A; }
    QStatusBar::item { border: none; }
    QSplitter::handle { background: #27272A; }
    QSplitter#splitter_main::handle { background: #3F3F46; width: 6px;}
    QSplitter#splitter_main::handle:hover { background: #52525B; }
    QToolTip {
        background: #18181B; color: #FAFAFA;
        border: 1px solid #3F3F46; border-radius: 6px; padding: 4px 8px; font-size: 11px;
    }
    QPlainTextEdit, QTextEdit {
        background: #141417; border: 1px solid #3F3F46; border-radius: 8px;
        color: #FAFAFA; selection-background-color: #3F3F46;
    }
    QScrollBar:vertical { background: #09090B; width: 10px; }
    QScrollBar::handle:vertical { background: #3F3F46; min-height: 20px; border-radius: 5px; }
    QScrollBar::handle:vertical:hover { background: #52525B; }
    QScrollBar:horizontal { background: #09090B; height: 10px; }
    QScrollBar::handle:horizontal { background: #3F3F46; min-width: 20px; border-radius: 5px; }
    QScrollBar::handle:horizontal:hover { background: #52525B; }
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
    QCheckBox, QRadioButton { color: #E4E4E7; spacing: 6px; }
    QCheckBox::indicator, QRadioButton::indicator {
        width: 14px; height: 14px; border: 1px solid #52525B; background: #141417;
    }
    QCheckBox::indicator { border-radius: 4px; }
    QRadioButton::indicator { border-radius: 7px; }
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {
        background: #FAFAFA; border-color: #FAFAFA;
    }
"""


LIGHTWEIGHT_WORKSPACE_CHROME = """
    QGroupBox {
        border: 1px solid rgba(128,128,128,42);
        border-radius: 6px;
        margin-top: 14px;
        padding: 7px 6px 6px;
        font-weight: 500;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 5px;
        font-size: 11px;
        font-weight: 600;
    }
    QGroupBox#groupBox_spectrum,
    QGroupBox#groupBox_structures,
    QGroupBox#groupBoxAnno,
    QGroupBox#groupBoxRef {
        border-color: rgba(128,128,128,28);
        border-radius: 4px;
        margin-top: 12px;
        padding: 5px 5px 5px;
    }
    QTableWidget, QTableView {
        border: 0;
        gridline-color: transparent;
        selection-background-color: rgba(74,127,232,42);
    }
    QHeaderView::section {
        border: 0;
        border-bottom: 1px solid rgba(128,128,128,42);
        padding: 5px 8px;
        font-size: 11px;
        font-weight: 600;
    }
    QTableWidget::item, QTableView::item {
        border: 0;
        padding: 4px 7px;
    }
    QTabWidget::pane {
        border: 0;
        top: -1px;
    }
    QTabBar::tab {
        min-height: 22px;
        padding: 5px 10px;
        margin: 0 1px;
        border-radius: 5px;
        font-size: 11px;
    }
    QSplitter::handle {
        background: rgba(128,128,128,28);
    }
    QSplitter::handle:hover {
        background: rgba(128,128,128,70);
    }
    QProgressBar {
        border: 0;
        border-radius: 4px;
        min-height: 7px;
        max-height: 8px;
        text-align: center;
    }
"""

CODEX_LIGHT_THEME = """
    QMainWindow, QDialog { background: #FFFFFF; }
    QWidget {
        font-family: "Inter","Segoe UI","DejaVu Sans","Arial";
        font-size: 12px; color: #18181B; background: transparent;
    }
    QFrame#activityBar {
        background: #FAFAFA;
        border-right: 1px solid #E4E4E7;
    }
    QFrame#activityBar QLabel#label_logo {
        background: #FAFAFA; color: #18181B;
        font-size: 13px; font-weight: 700;
    }
    QFrame#activityBar QPushButton {
        background: transparent; border: none;
        border-left: 2px solid transparent;
        color: #71717A; font-size: 11px; font-weight: 500;
        padding: 6px 4px;
    }
    QFrame#activityBar QPushButton:hover {
        background: #F4F4F5; color: #18181B;
    }
    QFrame#activityBar QPushButton:pressed {
        background: #E4E4E7; color: #18181B; border-left: 2px solid #18181B;
    }
    QFrame#activityBar QPushButton:disabled { color: #D4D4D8; }
    QFrame#activityBar QFrame { background: #E4E4E7; }
    QFrame#leftPanel {
        background: #FAFAFA;
        border-right: 1px solid #E4E4E7;
    }
    QLabel#label_list_title, QLabel#label_spectra_title {
        background: #FAFAFA; color: #71717A;
        font-size: 10px; font-weight: 700; letter-spacing: 1px;
        padding: 8px 12px 4px 12px;
    }
    QListWidget#list_spectrum {
        background: #FAFAFA; border: none; outline: none;
        padding: 4px; color: #18181B;
    }
    QListWidget#list_spectrum::item {
        padding: 6px 10px; margin: 1px 3px; border-radius: 6px;
    }
    QListWidget#list_spectrum::item:selected {
        background: #E4E4E7; color: #09090B;
    }
    QListWidget#list_spectrum::item:hover { background: #F4F4F5; }
    QWidget#mainPanel { background: #FFFFFF; }
    QTabWidget::pane { border: none; background: #FFFFFF; top: 0; }
    QTabBar { background: #FAFAFA; }
    QTabBar::tab {
        background: #FAFAFA; color: #71717A;
        border: none; border-top: 2px solid transparent;
        border-right: 1px solid #E4E4E7;
        padding: 8px 16px; min-width: 80px; font-size: 12px;
    }
    QTabBar::tab:selected {
        background: #FFFFFF; color: #18181B;
        border-top: 2px solid #18181B;
    }
    QTabBar::tab:!selected:hover { background: #F4F4F5; color: #18181B; }
    QGroupBox {
        background: #FFFFFF; border: 1px solid #E4E4E7;
        border-radius: 8px; margin-top: 20px; padding: 10px 8px 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin; left: 10px; padding: 0 6px;
        color: #71717A; font-weight: 700; font-size: 10px;
        background: #FFFFFF; letter-spacing: 1px;
    }
    QFrame#leftPanel QGroupBox {
        background: #FFFFFF; border: 1px solid #E4E4E7;
        margin: 8px; border-radius: 8px;
    }
    QFrame#leftPanel QGroupBox::title { background: #FFFFFF; color: #71717A; }
    QLabel { color: #18181B; background: transparent; }
    QLabel#lbl_time {
        background: #FAFAFA; color: #71717A;
        border: none; border-top: 1px solid #E4E4E7;
        padding: 6px 12px; font-size: 10px;
    }
    QPushButton {
        background: #18181B; border: 1px solid #18181B;
        padding: 6px 14px; border-radius: 6px;
        color: #FFFFFF; font-weight: 600; font-size: 12px;
    }
    QPushButton:hover { background: #3F3F46; border-color: #3F3F46; }
    QPushButton:pressed { background: #52525B; border-color: #52525B; }
    QPushButton:disabled { background: #E4E4E7; color: #A1A1AA; border-color: #E4E4E7; }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background: #FFFFFF; border: 1px solid #D4D4D8;
        border-radius: 6px; padding: 5px 9px;
        color: #18181B; selection-background-color: #E4E4E7; font-size: 12px;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border-color: #71717A; outline: none;
    }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView {
        background: #FFFFFF; border: 1px solid #D4D4D8;
        selection-background-color: #E4E4E7; color: #18181B; outline: none;
    }
    QTableWidget {
        background: #FFFFFF; border: none;
        gridline-color: #E4E4E7; color: #18181B;
        alternate-background-color: #FAFAFA;
    }
    QHeaderView::section {
        background: #FAFAFA; color: #71717A;
        border: none; border-bottom: 1px solid #E4E4E7;
        border-right: 1px solid #E4E4E7;
        padding: 5px 8px; font-weight: 700; font-size: 11px;
    }
    QTableView QTableCornerButton::section { background: #FAFAFA; border: 1px solid #E4E4E7; }
    QTableWidget::item { padding: 4px 8px; }
    QTableWidget::item:selected { background: #E4E4E7; color: #18181B; }
    QTableWidget::item:hover { background: #F4F4F5; }
    QFrame#bottomStrip { background: #F4F4F5; border-top: 1px solid #E4E4E7; }
    QProgressBar {
        background: transparent; border: none;
        color: #52525B; font-size: 11px; font-weight: 500;
        text-align: left; min-height: 20px;
    }
    QProgressBar::chunk { background: #D4D4D8; }
    QStatusBar { background: #FFFFFF; color: #71717A; font-size: 11px; border-top: 1px solid #E4E4E7; }
    QStatusBar::item { border: none; }
    QSplitter::handle { background: #E4E4E7; }
    QSplitter#splitter_main::handle { background: #D4D4D8; width: 6px;}
    QSplitter#splitter_main::handle:hover { background: #A1A1AA; }
    QToolTip {
        background: #18181B; color: #FFFFFF;
        border: 1px solid #18181B; border-radius: 6px; padding: 4px 8px; font-size: 11px;
    }
    QPlainTextEdit, QTextEdit {
        background: #FFFFFF; border: 1px solid #D4D4D8; border-radius: 8px;
        color: #18181B; selection-background-color: #E4E4E7;
    }
    QScrollBar:vertical { background: #FFFFFF; width: 10px; }
    QScrollBar::handle:vertical { background: #D4D4D8; min-height: 20px; border-radius: 5px; }
    QScrollBar::handle:vertical:hover { background: #A1A1AA; }
    QScrollBar:horizontal { background: #FFFFFF; height: 10px; }
    QScrollBar::handle:horizontal { background: #D4D4D8; min-width: 20px; border-radius: 5px; }
    QScrollBar::handle:horizontal:hover { background: #A1A1AA; }
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
    QCheckBox, QRadioButton { color: #18181B; spacing: 6px; }
    QCheckBox::indicator, QRadioButton::indicator {
        width: 14px; height: 14px; border: 1px solid #A1A1AA; background: #FFFFFF;
    }
    QCheckBox::indicator { border-radius: 4px; }
    QRadioButton::indicator { border-radius: 7px; }
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {
        background: #18181B; border-color: #18181B;
    }
"""

TRANSLUCENT_LIGHT_THEME = CODEX_LIGHT_THEME + """
    QMainWindow, QDialog { background: rgba(255,255,255,242); }
    QFrame#activityBar, QFrame#leftPanel {
        background: rgba(250,250,250,190);
        border-color: rgba(228,228,231,190);
    }
    QWidget#mainPanel { background: rgba(255,255,255,232); }
    QTabBar { background: rgba(250,250,250,180); }
    QTabBar::tab { background: rgba(250,250,250,145); }
    QTabBar::tab:selected { background: rgba(255,255,255,230); }
    QGroupBox {
        background: rgba(255,255,255,205);
        border-color: rgba(228,228,231,190);
    }
    QFrame#leftPanel QGroupBox {
        background: rgba(255,255,255,175);
        border-color: rgba(228,228,231,180);
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
        background: rgba(255,255,255,210);
        border-color: rgba(212,212,216,190);
    }
    QTableWidget {
        background: rgba(255,255,255,205);
        alternate-background-color: rgba(250,250,250,135);
    }
    QHeaderView::section {
        background: rgba(250,250,250,175);
        border-bottom-color: rgba(228,228,231,190);
        border-right-color: rgba(228,228,231,160);
    }
    QFrame#bottomStrip, QStatusBar { background: rgba(250,250,250,175); }
"""


def _codex_accent_theme(accent, hover, pressed, on_accent="#09090B"):
    return CODEX_DARK_THEME + f"""
        QFrame#activityBar QPushButton:pressed {{
            background: rgba(255,255,255,8); color: {accent}; border-left: 2px solid {accent};
        }}
        QListWidget#list_spectrum::item:selected {{
            background: rgba(255,255,255,8); color: {accent};
        }}
        QTabBar::tab:selected {{
            background: #09090B; color: #FAFAFA;
            border-top: 2px solid {accent};
        }}
        QPushButton {{
            background: {accent}; border: 1px solid {accent};
            color: {on_accent}; font-weight: 700;
        }}
        QPushButton:hover {{ background: {hover}; border-color: {hover}; }}
        QPushButton:pressed {{ background: {pressed}; border-color: {pressed}; }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {accent};
        }}
        QTableWidget::item:selected {{ background: rgba(255,255,255,8); color: {accent}; }}
        QFrame#bottomStrip {{ background: #18181B; border-top: 2px solid {accent}; }}
        QSplitter#splitter_main::handle {{ background: {accent}; width: 6px; }}
        QSplitter#splitter_main::handle:hover {{ background: {hover}; }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background: {accent}; border-color: {accent};
        }}
    """


PASTEL_THEME_PALETTES = {
    "Ocean Light":    dict(bg="#F2F9FF", surface="#FFFFFF", panel="#EAF6FF", soft="#DFF0FF", border="#C9E2F4", text="#14324A", muted="#5E7B91", accent="#2F80B7", hover="#256D9F", pressed="#1E5A84"),
    "Aurora Light":   dict(bg="#F8F4FF", surface="#FFFFFF", panel="#F1EAFE", soft="#E9DDFD", border="#D9C8F4", text="#33224D", muted="#74618D", accent="#7A5AB8", hover="#684AA4", pressed="#563E8C"),
    "Rose Light":     dict(bg="#FFF5F7", surface="#FFFFFF", panel="#FFECEF", soft="#FFE1E7", border="#F5C8D2", text="#4A1F2B", muted="#8B5E68", accent="#C6536D", hover="#AC405A", pressed="#92354D"),
    "Peach Light":    dict(bg="#FFF8F1", surface="#FFFFFF", panel="#FFF0DE", soft="#FFE4C7", border="#F0CCAA", text="#472B16", muted="#86664A", accent="#C66B2E", hover="#A95722", pressed="#8E461B"),
    "Mint Light":     dict(bg="#F2FBF6", surface="#FFFFFF", panel="#E7F7EE", soft="#DDF2E8", border="#C0E2D0", text="#183A2A", muted="#5C806B", accent="#3B9270", hover="#2F7A5D", pressed="#27644D"),
    "Gold Light":     dict(bg="#FFFBEF", surface="#FFFFFF", panel="#FFF4D7", soft="#FCE9B6", border="#EAD39A", text="#3D2E12", muted="#806F48", accent="#A8761E", hover="#8E6217", pressed="#755113"),
    "Lavender Light": dict(bg="#F6F5FF", surface="#FFFFFF", panel="#EDEBFF", soft="#E4E1FF", border="#CECAF3", text="#26264D", muted="#65658C", accent="#6367B1", hover="#5458A0", pressed="#464A89"),
    "Teal Light":     dict(bg="#EFFBF9", surface="#FFFFFF", panel="#E0F6F2", soft="#CCF0EA", border="#B4DDD6", text="#123A36", muted="#557D78", accent="#2C8B80", hover="#23756C", pressed="#1D6059"),
}


def _codex_light_accent_theme(accent, hover, pressed, tint="#F8FAFC", soft="#EEF2FF", on_accent="#FFFFFF",
                              bg=None, surface="#FFFFFF", panel=None, border="#D8E2EE",
                              text="#1F2937", muted="#64748B"):
    bg = bg or tint
    panel = panel or tint
    return CODEX_LIGHT_THEME + f"""
        QMainWindow, QDialog {{ background: {bg}; }}
        QWidget {{ color: {text}; }}
        QWidget#mainPanel {{ background: rgba(255,255,255,216); }}
        QFrame#activityBar {{
            background: rgba(255,255,255,150);
            border-right: 1px solid {border};
        }}
        QFrame#activityBar QLabel#label_logo {{
            background: transparent; color: {accent};
        }}
        QFrame#activityBar QPushButton {{
            color: {muted};
        }}
        QFrame#activityBar QPushButton:hover {{
            background: {soft}; color: {text};
        }}
        QFrame#activityBar QPushButton:pressed {{
            background: {soft}; color: {accent}; border-left: 2px solid {accent};
        }}
        QFrame#leftPanel, QLabel#label_list_title, QLabel#label_spectra_title {{
            background: rgba(255,255,255,132);
            color: {muted};
            border-right-color: {border};
        }}
        QListWidget#list_spectrum {{
            background: rgba(255,255,255,72);
            color: {text};
        }}
        QListWidget#list_spectrum::item:selected {{
            background: {soft}; color: {accent};
        }}
        QListWidget#list_spectrum::item:hover {{ background: {soft}; }}
        QTabBar {{ background: rgba(255,255,255,120); }}
        QTabBar::tab {{ background: rgba(255,255,255,76); color: {muted}; }}
        QTabBar::tab:selected {{
            background: rgba(255,255,255,210); color: {text};
            border-top: 2px solid {accent};
        }}
        QTabBar::tab:!selected:hover {{ background: {soft}; color: {text}; }}
        QGroupBox {{
            background: rgba(255,255,255,178);
            border-color: {border};
        }}
        QFrame#leftPanel QGroupBox {{
            background: rgba(255,255,255,130);
            border-color: {border};
        }}
        QGroupBox::title {{ color: {accent}; }}
        QLabel {{ color: {text}; }}
        QLabel#lbl_time {{
            background: rgba(255,255,255,95); color: {muted}; border-top-color: {border};
        }}
        QPushButton {{
            background: {panel}; border: 1px solid {border};
            color: {accent}; font-weight: 700;
        }}
        QPushButton:hover {{ background: {soft}; border-color: {accent}; color: {hover}; }}
        QPushButton:pressed {{ background: {accent}; border-color: {accent}; color: {on_accent}; }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {accent};
        }}
        QComboBox QAbstractItemView {{
            background: {surface}; color: {text}; selection-background-color: {soft};
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
            background: rgba(255,255,255,190);
            border-color: {border};
            color: {text};
            selection-background-color: {soft};
        }}
        QTableWidget {{
            background: rgba(255,255,255,170);
            color: {text};
            gridline-color: {border};
            alternate-background-color: rgba(255,255,255,92);
        }}
        QTableWidget::item:selected {{ background: {soft}; color: {accent}; }}
        QTableWidget::item:hover {{ background: {panel}; }}
        QHeaderView::section {{
            background: rgba(255,255,255,128);
            color: {muted};
            border-bottom-color: {border};
            border-right-color: {border};
        }}
        QFrame#bottomStrip {{ background: rgba(255,255,255,105); border-top: 2px solid {accent}; }}
        QStatusBar {{ background: rgba(255,255,255,125); color: {muted}; border-top-color: {border}; }}
        QSplitter#splitter_main::handle {{ background: {accent}; width: 6px; }}
        QSplitter#splitter_main::handle:hover {{ background: {hover}; }}
        QToolTip {{ background: {text}; color: {surface}; border-color: {text}; }}
        QScrollBar:vertical, QScrollBar:horizontal {{ background: rgba(255,255,255,80); }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {border}; }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {muted}; }}
        QCheckBox, QRadioButton {{ color: {text}; }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background: {accent}; border-color: {accent};
        }}
    """


CODEX_THEME_ACCENTS = {
    "Graphite Dark": ("#FAFAFA", "#E4E4E7", "#D4D4D8", "#09090B"),
    "Light": ("#18181B", "#3F3F46", "#52525B", "#FFFFFF"),
    "Ocean Dark": ("#38BDF8", "#0EA5E9", "#0284C7", "#06121A"),
    "Aurora Dark": ("#A78BFA", "#8B5CF6", "#7C3AED", "#100A1E"),
    "Rose Dark": ("#FB7185", "#F43F5E", "#E11D48", "#1F0A10"),
    "Ember Dark": ("#F97316", "#EA580C", "#C2410C", "#1E0D05"),
    "Forest Dark": ("#34D399", "#10B981", "#059669", "#06140F"),
    "Gold Dark": ("#FBBF24", "#F59E0B", "#D97706", "#171003"),
    "Crimson Dark": ("#EF4444", "#DC2626", "#B91C1C", "#1A0707"),
    "Teal Dark": ("#2DD4BF", "#14B8A6", "#0D9488", "#041412"),
    "Indigo Dark": ("#818CF8", "#6366F1", "#4F46E5", "#090B1F"),
    "Lime Dark": ("#A3E635", "#84CC16", "#65A30D", "#0E1404"),
    "Ocean Light": ("#0284C7", "#0369A1", "#075985", "#FFFFFF"),
    "Aurora Light": ("#7C3AED", "#6D28D9", "#5B21B6", "#FFFFFF"),
    "Rose Light": ("#E11D48", "#BE123C", "#9F1239", "#FFFFFF"),
    "Peach Light": ("#EA580C", "#C2410C", "#9A3412", "#FFFFFF"),
    "Mint Light": ("#059669", "#047857", "#065F46", "#FFFFFF"),
    "Gold Light": ("#B45309", "#92400E", "#78350F", "#FFFFFF"),
    "Lavender Light": ("#6366F1", "#4F46E5", "#4338CA", "#FFFFFF"),
    "Teal Light": ("#0D9488", "#0F766E", "#115E59", "#FFFFFF"),
}

CODEX_COLORED_THEMES = {
    name: _codex_accent_theme(accent, hover, pressed)
    for name, (accent, hover, pressed, _bg) in CODEX_THEME_ACCENTS.items()
    if name.endswith(" Dark") and name != "Graphite Dark"
}

CODEX_LIGHT_COLORED_THEMES = {
    name: _codex_light_accent_theme(
        p["accent"], p["hover"], p["pressed"],
        tint=p["panel"], soft=p["soft"], bg=p["bg"], surface=p["surface"],
        panel=p["panel"], border=p["border"], text=p["text"], muted=p["muted"],
    )
    for name, p in PASTEL_THEME_PALETTES.items()
}

THEMES = {
    "Sky": THEMES.get("Sky", ""),
    "Light": TRANSLUCENT_LIGHT_THEME,
    **CODEX_LIGHT_COLORED_THEMES,
    "Graphite Dark": CODEX_DARK_THEME,
    **CODEX_COLORED_THEMES,
    **THEMES,
}

THEME_NAME_ALIASES = {
    "Codex Dark": "Graphite Dark",
    "Codex Light": "Light",
    "Codex Ocean": "Ocean Dark",
    "Codex Aurora": "Aurora Dark",
    "Codex Rose": "Rose Dark",
    "Codex Ember": "Ember Dark",
    "Codex Forest": "Forest Dark",
    "Codex Gold": "Gold Dark",
    "Codex Crimson": "Crimson Dark",
    "Codex Teal": "Teal Dark",
    "Codex Indigo": "Indigo Dark",
    "Codex Lime": "Lime Dark",
    "Codex Light Ocean": "Ocean Light",
    "Codex Light Aurora": "Aurora Light",
    "Codex Light Rose": "Rose Light",
    "Codex Light Peach": "Peach Light",
    "Codex Light Mint": "Mint Light",
    "Codex Light Gold": "Gold Light",
    "Codex Light Lavender": "Lavender Light",
    "Codex Light Teal": "Teal Light",
}



class AnnoMASS(QMainWindow, main.Ui_MainWindow):
    # Flavonoid Agent cross-thread bridge signals (see AgentWorker /
    # _agent_pipeline_bridge_execute / _agent_load_spectra_execute below).
    # Owned by this main-thread QMainWindow so any cross-thread emit/connect
    # involving them is correctly auto-queued by Qt.
    agent_pipeline_result_ready  = QtCore.pyqtSignal(object)   # (df, mf_cache)
    agent_pipeline_result_failed = QtCore.pyqtSignal(str)
    agent_load_spectra_ready     = QtCore.pyqtSignal(object)   # {"n_spectra":..., "titles":[...]}
    agent_load_spectra_failed    = QtCore.pyqtSignal(str)
    agent_plot_ready             = QtCore.pyqtSignal(object)   # {"spectrum_index" or "smiles": ...}
    agent_plot_failed            = QtCore.pyqtSignal(str)
    agent_file_ready             = QtCore.pyqtSignal(object)   # {"file_path": ...}
    agent_file_failed            = QtCore.pyqtSignal(str)
    agent_preview_save_ready     = QtCore.pyqtSignal(object)   # {"file_path": ..., "verified": True}
    agent_preview_save_failed    = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._prime_startup_background()
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._make_workspace_splitters_interactive()

        self._splitter_init_done = False

        self._loaded_files = []

        self._tune_results_tabs()
        self.setWindowTitle("AnnoMASS (Flavonoid Annotation)")

        try:
            self.setWindowIcon(QtGui.QIcon("icon/favicon.ico"))
        except Exception:
            pass

        # temp
        try:
            shutil.rmtree("temp")
        except Exception:
            pass
        os.makedirs("temp", exist_ok=True)

        # Logo assets are resolved from the application directory so launching
        # from a desktop shortcut (whose working directory can be elsewhere)
        # does not silently leave the top-left logo blank.
        self._logo_source_pixmap = QPixmap(
            str(Path(__file__).resolve().parent / "ui" / "assets" / "annomass_logo.png")
        )
        if not self._logo_source_pixmap.isNull():
            self.label_logo.setScaledContents(False)
            self.label_logo.installEventFilter(self)
            self._refresh_header_logo()

        # sidebar: theme switcher + settings + ETA
        self._init_theme_switcher()
        self._create_left_settings_and_time()

        # holders
        self.pp = None
        self.pn = None
        self.reference_positive = None
        self.reference_negative = None

        self.spectrums = pd.DataFrame(columns=["title", "spectrum"])
        self.current_spectrum = None
        self.current_reference = None

        self.default_database = None
        self.deepmass_positive = None
        self.deepmass_negative = None
        self._indices_refs_ready = False
        self._structure_db_ready = False
        self._models_ready = False

        # bind actions
        self.butt_open.setDisabled(True)

        self.butt_open.clicked.connect(self.load_spectrums)

        # Allow sidebar to show full filenames via horizontal scroll
        self.list_spectrum.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_spectrum.setWordWrap(False)
        # Clicking a loaded spectrum previews it in the Results tab's own
        # embedded spectrum canvas (_pipeline_ax/_pipeline_canvas) -- the
        # same one the results table's row-click uses (_pipeline_plot_selected)
        # -- one merged spectrum viewer, not a separate display per entry point.
        self.list_spectrum.itemClicked.connect(self._on_spectrum_list_clicked)

        # Pipeline results tab
        self._init_pipeline_tab()

        # Flavonoid Agent standalone tab
        self._init_agent_tab()
        self._apply_lightweight_workspace_chrome()

        # Pipeline button
        self.butt_pipeline.clicked.connect(self.run_pipeline)
        if _DREAMS_AVAILABLE is False:
            self.butt_pipeline.setEnabled(False)
            self.butt_pipeline.setToolTip("DreaMS not available – run disabled")

        # threads
        self.Thread_LoadIndexPositive = None
        self.Thread_LoadIndexNegative = None
        self.Thread_LoadReference = None
        self.Thread_LoadModels = None
        self.Thread_LoadStructureDB = None
        self._load_watcher = None
        self.Thread_Pipeline = None

        # check required files
        try:
            require_file(INDEX_POS, "positive HNSW index")
            require_file(INDEX_NEG, "negative HNSW index")
            require_file(REF_POS, "positive reference pickle")
            require_file(REF_NEG, "negative reference pickle")
            require_file(W2V_POS, "positive Word2Vec model")
            require_file(W2V_NEG, "negative Word2Vec model")
            require_file(STRUCT_DB, "structure DB CSV")
        except Exception as e:
            self.ErrorMsg(str(e))
            return

        # Load heavyweight resources in the background. The window can show
        # immediately; run/open buttons stay disabled until everything needed
        # for identification is ready.
        self.progressBar.setValue(0)
        self.progressBar.setFormat("Loading resources...")
        self.lbl_time.setText("Loading...\nPlease wait")
        self.load_structure_db_async()
        self.load_resources_parallel()
        self.load_models_async()

    def _refresh_header_logo(self):
        """Fit the wordmark into the activity bar without distorting it."""
        source = getattr(self, "_logo_source_pixmap", QPixmap())
        label = getattr(self, "label_logo", None)
        if label is None or source.isNull():
            return
        # The current brand asset is a wide wordmark.  The label already owns
        # its safe-area margins, so fit against its contents rectangle without
        # subtracting a second padding that would make the mark look clipped.
        target = label.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        label.setPixmap(source.scaled(
            target,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        ))

    def eventFilter(self, watched, event):
        if watched is getattr(self, "label_logo", None) and event.type() == QtCore.QEvent.Resize:
            self._refresh_header_logo()
        return super().eventFilter(watched, event)

    # ---------- showEvent: set initial splitter sizes when window first shown ----------
    def _show_workspace_switch_cover(self, text="Switching workspace...", duration_ms=220):
        """Cover Manual/Agent relayout so Qt never exposes a blank black frame."""
        cover = getattr(self, "_workspace_switch_cover", None)
        if cover is None:
            cover = QtWidgets.QWidget(self)
            cover.setObjectName("workspace_switch_cover")
            cover.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            lay = QtWidgets.QVBoxLayout(cover)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addStretch(1)
            card = QtWidgets.QFrame(cover)
            card.setObjectName("workspace_switch_card")
            card_lay = QtWidgets.QHBoxLayout(card)
            card_lay.setContentsMargins(22, 14, 22, 14)
            dot = QtWidgets.QLabel(card)
            dot.setFixedSize(10, 10)
            dot.setObjectName("workspace_switch_dot")
            label = QtWidgets.QLabel(text, card)
            label.setObjectName("workspace_switch_label")
            card_lay.addWidget(dot)
            card_lay.addWidget(label)
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            row.addWidget(card)
            row.addStretch(1)
            lay.addLayout(row)
            lay.addStretch(1)
            self._workspace_switch_cover = cover
            self._workspace_switch_cover_label = label
            self._workspace_switch_cover_seq = 0

        is_light = _theme_is_light(getattr(self, "_current_theme_name", "Graphite Dark"))
        cover.setStyleSheet("""
            QWidget#workspace_switch_cover {{
                background: {cover_bg};
            }}
            QFrame#workspace_switch_card {{
                background: {card_bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel#workspace_switch_label {{
                color: {text_color};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#workspace_switch_dot {{
                background: {accent};
                border-radius: 5px;
            }}
        """.format(
            cover_bg="#F8FBFF" if is_light else "#111827",
            card_bg="#FFFFFF" if is_light else "#18181B",
            border="#DCE7F7" if is_light else "#2F343F",
            text_color="#111827" if is_light else "#F8FAFC",
            accent=_theme_accent(getattr(self, "_current_theme_name", "Graphite Dark")),
        ))
        if hasattr(self, "_workspace_switch_cover_label"):
            self._workspace_switch_cover_label.setText(text)
        cover.setGeometry(self.rect())
        cover.show()
        cover.raise_()
        self._workspace_switch_cover_seq = getattr(self, "_workspace_switch_cover_seq", 0) + 1
        seq = self._workspace_switch_cover_seq
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        for delay in (0, 40, 100):
            QtCore.QTimer.singleShot(delay, cover.raise_)
        QtCore.QTimer.singleShot(
            duration_ms,
            lambda: cover.hide()
            if getattr(self, "_workspace_switch_cover_seq", 0) == seq else None
        )

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitter_init_done:
            self._splitter_init_done = True
            self._make_workspace_splitters_interactive()
            self._apply_lightweight_workspace_chrome()
            if hasattr(self, "splitter_main"):
                w = self.splitter_main.width()
                act_w  = 190                               # wide AnnoMASS wordmark
                left_w = min(220, max(100, int(w * 0.18))) # leftPanel
                main_w = max(100, w - act_w - left_w)      # mainPanel
                self.splitter_main.setSizes([act_w, left_w, main_w])

    def _make_workspace_splitters_interactive(self):
        """Keep manual workspace panes draggable even near the window minimum.

        The Agent/Manual mode row adds vertical content above the result tabs.
        Without relaxing child minimum sizes, Qt can consume all available
        space at the default window size, leaving no slack for splitter handles
        until the user maximizes the window.
        """
        splitter_specs = (
            ("splitter_main", QtCore.Qt.Horizontal),
        )
        for name, orientation in splitter_specs:
            sp = getattr(self, name, None)
            if sp is None:
                continue
            sp.setHandleWidth(10)
            sp.setOpaqueResize(True)
            sp.setChildrenCollapsible(True)
            sp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            for i in range(sp.count()):
                sp.setCollapsible(i, True)
                child = sp.widget(i)
                if child is None:
                    continue
                child.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                if orientation == QtCore.Qt.Horizontal:
                    child.setMinimumWidth(0)
                else:
                    child.setMinimumHeight(0)
            for i in range(1, sp.count()):
                handle = sp.handle(i)
                if handle is not None:
                    handle.setCursor(
                        QtCore.Qt.SplitHCursor
                        if orientation == QtCore.Qt.Horizontal
                        else QtCore.Qt.SplitVCursor
                    )

        # Keep truly useful lower bounds only; everything else should be able
        # to shrink so splitter handles have room to move at startup size.
        if hasattr(self, "activityBar"):
            self.activityBar.setMinimumWidth(180)
            self.activityBar.setMaximumWidth(260)
        if hasattr(self, "leftPanel"):
            self.leftPanel.setMinimumWidth(80)
        if hasattr(self, "mainPanel"):
            self.mainPanel.setMinimumWidth(260)

        flexible_widgets = ("resultsPanel", "tabWidget_results")
        for name in flexible_widgets:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            widget.setMinimumSize(0, 0)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # ---------- Tabs tuning ----------
    def _prime_startup_background(self):
        pale = QtGui.QColor("#F3F5F8")
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, pale)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#FFFFFF"))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#1F2430"))
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        for name in ("centralwidget", "workspace_panel", "activityBar", "leftPanel", "mainPanel", "bottomStrip"):
            w = self.findChild(QtWidgets.QWidget, name)
            if w is not None:
                w.setAutoFillBackground(True)
                w.setPalette(pal)

    def _tune_results_tabs(self):
        tw = self.findChild(QtWidgets.QTabWidget, "tabWidget_results")
        if tw is None:
            return
        tb = tw.tabBar()
        tb.setExpanding(False)
        tb.setUsesScrollButtons(True)
        tb.setElideMode(Qt.ElideRight)
        tb.setDocumentMode(True)

    # ---------- Left panel splitter fix ----------
    def _fix_left_panel_splitter(self):
        """Ensure the left panel's Actions/Spectra splitter is draggable."""
        left = self.findChild(QtWidgets.QWidget, "leftPanel")
        sp   = self.findChild(QtWidgets.QSplitter, "splitter_left_actions_spectra")
        if left is None or sp is None:
            return

        lay = left.layout()
        if lay is None:
            return

        # Give the splitter all available vertical stretch in the left panel layout
        idx = lay.indexOf(sp)
        if idx >= 0:
            lay.setStretch(idx, 1)

        # Keep enough height for the full horizontal wordmark and its padding.
        try:
            self.label_logo.setFixedHeight(72)
            self.label_logo.setContentsMargins(10, 8, 10, 8)
        except Exception:
            pass

        # Expand size policy on Actions groupbox and spectra panel so splitter works
        for name in ("groupBox_actions", "spectraPanel"):
            w = self.findChild(QtWidgets.QWidget, name)
            if w:
                w.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding,
                    QtWidgets.QSizePolicy.Expanding,
                )

        # Set sensible initial sizes: Actions ~200px, Spectra fills the rest
        sp.setSizes([200, 400])
        sp.setChildrenCollapsible(True)

    def _finalize_left_panel_splitter(self):
        """
        Integrate all widgets in the left panel (except logo and progressBar) into a
        vertical QSplitter so each section can be resized by dragging.
        """
        left = self.findChild(QtWidgets.QWidget, "leftPanel")
        if left is None:
            return
        lay = left.layout()
        if lay is None:
            return

        # Collect widgets to move into splitter (excluding logo and progressBar)
        logo_w = self.findChild(QtWidgets.QLabel, "label_logo")
        pb_w   = self.findChild(QtWidgets.QProgressBar, "progressBar")

        # Extract all intermediate widgets
        mid_widgets = []
        for i in range(lay.count() - 1, -1, -1):
            item = lay.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is None:
                continue
            if w is logo_w or w is pb_w:
                continue
            lay.removeWidget(w)
            mid_widgets.insert(0, w)

        if not mid_widgets:
            return

        # Build main vertical splitter
        master_sp = QtWidgets.QSplitter(left)
        master_sp.setOrientation(QtCore.Qt.Vertical)
        master_sp.setHandleWidth(8)
        master_sp.setChildrenCollapsible(True)   # Allow collapsing to zero
        master_sp.setObjectName("splitter_left_master")

        for w in mid_widgets:
            w.setParent(master_sp)
            master_sp.addWidget(w)
            w.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            w.setMinimumHeight(0)   # Allow shrinking to minimum

        # Re-insert: logo -> master splitter -> progressBar
        # Find current positions of logo and progressBar
        logo_idx = lay.indexOf(logo_w) if logo_w else -1
        pb_idx   = lay.indexOf(pb_w)   if pb_w   else -1

        if pb_idx >= 0:
            lay.insertWidget(pb_idx, master_sp)
        else:
            lay.addWidget(master_sp)

        # Make master splitter fill the remaining space
        master_sp_idx = lay.indexOf(master_sp)
        if master_sp_idx >= 0:
            lay.setStretch(master_sp_idx, 1)
        if logo_w:
            lay.setStretch(lay.indexOf(logo_w), 0)
        if pb_w:
            lay.setStretch(lay.indexOf(pb_w), 0)

        # Initial sizes: Actions/Spectra splitter larger, others ~120px each
        sizes = []
        for i in range(master_sp.count()):
            w = master_sp.widget(i)
            if w and w.objectName() == "splitter_left_actions_spectra":
                sizes.append(300)
            else:
                sizes.append(120)
        master_sp.setSizes(sizes)

    # ---------- Theme ----------
    # matplotlib face/edge colours that match each theme
    _THEME_MPL = {
        "Graphite Dark":   dict(fig="#09090B", ax="#0F0F12", text="#E4E4E7", grid="#27272A", spine="#3F3F46"),
        "Light":           dict(fig="#FFFFFF", ax="#FAFAFA", text="#18181B", grid="#E4E4E7", spine="#D4D4D8"),
        "VS Dark+":        dict(fig="#1E1E1E", ax="#252526", text="#D4D4D4", grid="#3E3E3E", spine="#555555"),
        "VS Light+":       dict(fig="#F3F3F3", ax="#FFFFFF", text="#333333", grid="#E0E0E0", spine="#BBBBBB"),
        "One Dark Pro":    dict(fig="#282C34", ax="#21252B", text="#ABB2BF", grid="#3E4451", spine="#4B5263"),
        "Monokai":         dict(fig="#272822", ax="#1E1F1C", text="#F8F8F2", grid="#3B3A32", spine="#49483E"),
        "Solarized Light": dict(fig="#EEE8D5", ax="#FDF6E3", text="#657B83", grid="#DDD6C1", spine="#C3BAA3"),
        "Mint Green":      dict(fig="#F0FAF5", ax="#FFFFFF", text="#2D4A3E", grid="#D8F3DC", spine="#B7E4C7"),
        "Soft Purple":     dict(fig="#F8F5FF", ax="#FFFFFF", text="#3D2C6E", grid="#EDE9FE", spine="#DDD6FE"),
    }

    def _apply_theme(self, name: str):
        name = THEME_NAME_ALIASES.get(name, name)
        qss = THEMES.get(name, "").replace("{{", "{").replace("}}", "}")
        # GLOBAL_MENU_STYLE is also applied at the QApplication level (see
        # __main__ below), but every theme's blanket "QWidget { background:
        # transparent; ... }" rule wins over that once it's set here on the
        # main window itself -- a widget-level stylesheet overrides an
        # ancestor QApplication-level one for any selector that also
        # matches (QMenu IS-A QWidget), regardless of which one is more
        # specific. `transparent` on a QMenu popup (an independent top-level
        # window, not backed by the app's own compositing) then renders as
        # a solid black box instead of see-through -- this was the actual
        # cause of every "right-click menu is black" report. Appending it
        # here, in the SAME stylesheet application, guarantees it wins.
        self.setStyleSheet(qss + LIGHTWEIGHT_WORKSPACE_CHROME + GLOBAL_MENU_STYLE)
        self._apply_lightweight_workspace_chrome()
        # update matplotlib figures to match theme
        try:
            import matplotlib
            mpl = self._THEME_MPL.get(name, {})
            if not mpl and name in CODEX_THEME_ACCENTS:
                accent, _hover, _pressed, bg = CODEX_THEME_ACCENTS[name]
                is_light = _theme_is_light(name)
                mpl = dict(
                    fig="#FFFFFF" if is_light else bg,
                    ax="#FAFAFA" if is_light else "#0F0F12",
                    text="#18181B" if is_light else "#E4E4E7",
                    grid="#E4E4E7" if is_light else "#27272A",
                    spine=accent,
                )
            if mpl:
                matplotlib.rcParams.update({
                    "figure.facecolor":  mpl["fig"],
                    "axes.facecolor":    mpl["ax"],
                    "axes.edgecolor":    mpl["spine"],
                    "axes.labelcolor":   mpl["text"],
                    "xtick.color":       mpl["text"],
                    "ytick.color":       mpl["text"],
                    "text.color":        mpl["text"],
                    "grid.color":        mpl["grid"],
                    "legend.facecolor":  mpl["ax"],
                    "legend.edgecolor":  mpl["spine"],
                    "legend.labelcolor": mpl["text"],
                    "savefig.facecolor": mpl["fig"],
                })
                # redraw open canvases (Results tab's embedded spectrum +
                # cluster-map plots -- plain FigureCanvas, not the MakeFigure
                # wrapper, so use .figure not .fig)
                for canvas in [getattr(self, "_pipeline_canvas", None),
                                getattr(self, "_pipeline_cluster_canvas", None)]:
                    if canvas is not None:
                        try:
                            fig = canvas.figure
                            fig.set_facecolor(mpl["fig"])
                            for ax in fig.get_axes():
                                ax.set_facecolor(mpl["ax"])
                                ax.tick_params(colors=mpl["text"])
                                ax.xaxis.label.set_color(mpl["text"])
                                ax.yaxis.label.set_color(mpl["text"])
                                for spine in ax.spines.values():
                                    spine.set_edgecolor(mpl["spine"])
                            canvas.draw()
                        except Exception:
                            pass
        except Exception:
            pass
        self._current_theme_name = name
        # Re-apply theme-aware colours to widgets whose creation-time styles
        # were hardcoded (section-title labels, parameter bar), so they stay
        # readable after a theme switch instead of inheriting #555/#f5f5f5.
        try:
            muted = _theme_muted(name)
            for lbl in self.findChildren(QtWidgets.QLabel):
                if lbl.objectName() == "sectionTitle":
                    cur = lbl.styleSheet() or ""
                    m = re.search(r"font-size:\s*([\d.]+)px", cur)
                    fs = m.group(1) if m else "11"
                    lbl.setStyleSheet(f"color: {muted}; font-size: {fs}px;")
            if hasattr(self, "_param_bar"):
                bar_bg = "#FFFFFF" if _theme_is_light(name) else "#18181B"
                self._param_bar.setStyleSheet(f"QFrame {{ background: {bar_bg}; border-radius: 4px; }}")
            # matplotlib navigation toolbar icons ship as black bitmaps; tint
            # them to the theme text colour so they stay visible on dark themes.
            icon_color = "#1F2937" if _theme_is_light(name) else "#E4E4E7"
            for tb in self.findChildren(NavigationToolbar):
                for act in tb.actions():
                    ic = act.icon()
                    if ic.isNull():
                        continue
                    tinted = QtGui.QIcon()
                    for sz in (QtCore.QSize(16, 16), QtCore.QSize(20, 20), QtCore.QSize(24, 24), QtCore.QSize(32, 32)):
                        pm = ic.pixmap(sz)
                        if pm.isNull():
                            continue
                        out = QtGui.QPixmap(pm.size())
                        out.fill(QtCore.Qt.transparent)
                        p = QtGui.QPainter(out)
                        p.drawPixmap(0, 0, pm)
                        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
                        p.fillRect(out.rect(), QtGui.QColor(icon_color))
                        p.end()
                        tinted.addPixmap(out)
                    if not tinted.isNull():
                        act.setIcon(tinted)
        except Exception:
            pass
        if hasattr(self, "_agent_panel"):
            try:
                self._agent_panel.setStyleSheet(_agent_panel_style_for_theme(name))
                self._agent_chat_log.setStyleSheet(_agent_chat_browser_style_for_theme(name))
                self._agent_chat_log.document().setDefaultStyleSheet(_agent_chat_document_style_for_theme(name))
                if hasattr(self, "_agent_preview_panel"):
                    self._agent_preview_panel.set_embedded_dark(not _theme_is_light(name))
                self._agent_render_session(self._agent_current_session)
            except Exception:
                pass
        try:
            s = QtCore.QSettings("AnnoMASS", "AnnoMASS")
            s.setValue("theme", name)
        except Exception:
            pass

    def _apply_lightweight_workspace_chrome(self):
        """Runtime polish that QSS alone cannot reliably enforce."""
        for table in self.findChildren(QtWidgets.QTableWidget):
            try:
                table.setShowGrid(False)
                table.setAlternatingRowColors(False)
                table.setFrameShape(QtWidgets.QFrame.NoFrame)
                table.setWordWrap(False)
                table.setFocusPolicy(QtCore.Qt.StrongFocus)
                table.verticalHeader().setVisible(False)
                table.verticalHeader().setDefaultSectionSize(24)
                table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                table.horizontalHeader().setHighlightSections(False)
                table.horizontalHeader().setMinimumSectionSize(48)
            except Exception:
                pass

        for tab in self.findChildren(QtWidgets.QTabWidget):
            try:
                tab.setDocumentMode(True)
                tab.setElideMode(Qt.ElideRight)
                tab.tabBar().setExpanding(False)
                tab.tabBar().setUsesScrollButtons(True)
            except Exception:
                pass

        for toolbar in self.findChildren(NavigationToolbar):
            try:
                toolbar.setIconSize(QtCore.QSize(16, 16))
                toolbar.setMaximumHeight(28)
            except Exception:
                pass

    def _init_theme_switcher(self):
        left = self.findChild(QtWidgets.QWidget, "leftPanel")
        if left is None or left.layout() is None:
            return

        gb = QtWidgets.QGroupBox("Appearance", left)
        lay = QtWidgets.QVBoxLayout(gb)
        lay.setContentsMargins(8, 8, 8, 8)

        cb = QtWidgets.QComboBox(gb)
        cb.addItems(list(THEMES.keys()))
        lay.addWidget(cb)

        idx = left.layout().indexOf(self.progressBar) if hasattr(self, "progressBar") else -1
        if idx >= 0:
            left.layout().insertWidget(idx, gb)
        else:
            left.layout().addWidget(gb)

        theme = "Sky"
        try:
            s = QtCore.QSettings("AnnoMASS", "AnnoMASS")
            theme = s.value("theme", theme)
            theme = THEME_NAME_ALIASES.get(theme, theme)
        except Exception:
            pass
        if theme in THEMES and not _theme_is_light(theme):
            theme = "Sky"
        if theme not in THEMES:
            theme = "Sky"

        cb.setCurrentText(theme)
        cb.currentTextChanged.connect(self._apply_theme)
        self._apply_theme(theme)

    # ---------- Left settings + ETA ----------
    def _create_left_settings_and_time(self):
        left = self.findChild(QtWidgets.QWidget, "leftPanel")
        if left is None or left.layout() is None:
            self.le_allowed_elem = QtWidgets.QLineEdit(self)
            self.spin_maxcand = QtWidgets.QSpinBox(self)
            self.lbl_time = QtWidgets.QLabel("Ready", self)
            return

        self.gb_settings = QtWidgets.QGroupBox("Settings", left)
        form = QtWidgets.QFormLayout(self.gb_settings)
        form.setContentsMargins(8, 8, 8, 8)

        self.le_allowed_elem = QtWidgets.QLineEdit(self.gb_settings)
        self.le_allowed_elem.setPlaceholderText("e.g. CHO/CHON (empty=no filter)")
        self.le_allowed_elem.setText("")
        form.addRow("Allowed:", self.le_allowed_elem)

        self.spin_maxcand = QtWidgets.QSpinBox(self.gb_settings)
        self.spin_maxcand.setRange(50, 5000)
        self.spin_maxcand.setSingleStep(50)
        self.spin_maxcand.setValue(500)
        form.addRow("Max cand:", self.spin_maxcand)

        self.lbl_time = QtWidgets.QLabel("Ready", left)
        self.lbl_time.setObjectName("lbl_time")
        self.lbl_time.setWordWrap(True)
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setMinimumHeight(0)

        lay = left.layout()
        idx = lay.indexOf(self.progressBar) if hasattr(self, "progressBar") else -1
        if idx >= 0:
            lay.insertWidget(idx, self.lbl_time)
            lay.insertWidget(idx, self.gb_settings)
        else:
            lay.addWidget(self.gb_settings)
            lay.addWidget(self.lbl_time)

        self.progressBar.setFormat(f"Ready{self._workers_suffix()}")

    def _workers_suffix(self):
        """' | Workers N' is a backend implementation detail (parallel
        worker-thread count) that's meaningless to someone chatting with the
        Agent -- suppressed on the bottom status bar while Agent Mode is
        active. Guarded with getattr since this can be called before
        _init_agent_tab() has built _mode_agent_radio (early in __init__)."""
        agent_mode = getattr(self, "_mode_agent_radio", None)
        if agent_mode is not None and agent_mode.isChecked():
            return ""
        return f" | Workers {MAX_WORKERS}"

    def _collect_runtime_settings(self):
        allowed = (self.le_allowed_elem.text() or "").strip().upper()
        try:
            max_cand = int(self.spin_maxcand.value())
        except Exception:
            max_cand = 500
        return allowed, max_cand

    def _apply_settings_to_spectra(self, spectrums):
        allowed, max_cand = self._collect_runtime_settings()
        # FIDDLE is an optional formula pre-filter owned by the Pipeline tab's
        # own checkbox (see _init_pipeline_tab) -- purely a candidate
        # pre-filter feeding core.main.identify_unknown, never merged into
        # rank_key. Guarded with getattr since the Pipeline tab widgets may
        # not exist yet the very first time this runs.
        use_fiddle = bool(getattr(self, "_pl_fiddle_chk", None) and self._pl_fiddle_chk.isChecked())
        fiddle_instrument = (
            self._pl_fiddle_instrument.currentText()
            if getattr(self, "_pl_fiddle_instrument", None) else "Orbitrap"
        )
        for sp in spectrums:
            sp.set("only_flavonoids", False)
            sp.set("flav_classes", [])
            sp.set("allowed_elements", allowed)
            sp.set("max_candidates", max_cand)
            sp.set("use_fiddle", use_fiddle)
            sp.set("fiddle_instrument", fiddle_instrument)
            sp.set("fiddle_elements", allowed)

    def _add_widget_to_groupbox(self, gb: QtWidgets.QGroupBox, w: QtWidgets.QWidget):
        lay = gb.layout()
        if lay is None:
            lay = QtWidgets.QVBoxLayout(gb)
            gb.setLayout(lay)
        lay.addWidget(w)

    # ---------- ETA ----------
    def _start_eta(self, total):
        self._run_start_time = time.time()
        self._done_count = 0
        self._total_count = int(total)
        self._timer.start()
        self._update_time_display()

    def _stop_eta(self):
        self._timer.stop()
        self._update_time_display(final=True)

    def _update_time_display(self, final=False):
        if self._run_start_time is None:
            return
        elapsed = time.time() - self._run_start_time
        done = int(self._done_count)
        total = int(self._total_count)

        if final:
            eta_str = "00:00"
        else:
            if done > 0 and total > 0:
                remaining = (elapsed / done) * max(total - done, 0)
                eta_str = fmt_seconds(remaining)
            else:
                eta_str = "--:--"

        pct = int(100 * done / total) if total > 0 else 0
        self.progressBar.setValue(pct)
        self.progressBar.setFormat(f"{done}/{total} | Elapsed {fmt_seconds(elapsed)} | ETA {eta_str}{self._workers_suffix()}")
        self.lbl_time.setText(f"Progress: {done}/{total}\nElapsed: {fmt_seconds(elapsed)}\nETA: {eta_str}")

    # ---------- dialogs ----------
    # These 3 are used everywhere in the app (both Manual and Agent mode --
    # every ErrorMsg/WarnMsg/InforMsg call, including all Agent-panel error
    # messages) and never had an explicit stylesheet, so they inherited
    # whatever dark system/Qt theme is active and rendered unreadable --
    # same root cause already fixed for AgentSettingsDialog/FileViewerWidget.
    def WarnMsg(self, text):
        msg = QtWidgets.QMessageBox(self)
        msg.setStyleSheet(LIGHT_DIALOG_STYLE)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Warning")
        msg.setText(str(text))
        msg.exec_()

    def ErrorMsg(self, text):
        msg = QtWidgets.QMessageBox(self)
        msg.setStyleSheet(LIGHT_DIALOG_STYLE)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(str(text))
        msg.exec_()

    def InforMsg(self, text):
        msg = QtWidgets.QMessageBox(self)
        msg.setStyleSheet(LIGHT_DIALOG_STYLE)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setWindowTitle("Info")
        msg.setText(str(text))
        msg.exec_()

    # ---------- resource loading ----------
    def _set_startup_controls_enabled(self, enabled):
        self.butt_open.setEnabled(enabled)
        if hasattr(self, "butt_pipeline") and _DREAMS_AVAILABLE is not False:
            self.butt_pipeline.setEnabled(enabled)

    def load_structure_db_async(self):
        self.Thread_LoadStructureDB = Thread_LoadStructureDB(STRUCT_DB)
        self.Thread_LoadStructureDB.loaded.connect(self._on_structure_db_loaded)
        self.Thread_LoadStructureDB.failed.connect(self._on_structure_db_failed)
        self.Thread_LoadStructureDB.start()

    def load_models_async(self):
        self.Thread_LoadModels = Thread_LoadModels(W2V_POS, W2V_NEG)
        self.Thread_LoadModels.loaded.connect(self._on_models_loaded)
        self.Thread_LoadModels.failed.connect(self._on_models_failed)
        self.Thread_LoadModels.start()

    def load_resources_parallel(self):
        self._set_startup_controls_enabled(False)

        self.Thread_LoadIndexPositive = Thread_LoadIndex(INDEX_POS)
        self.Thread_LoadIndexPositive._index.connect(self._set_index_positive)
        self.Thread_LoadIndexPositive.start()

        self.Thread_LoadIndexNegative = Thread_LoadIndex(INDEX_NEG)
        self.Thread_LoadIndexNegative._index.connect(self._set_index_negative)
        self.Thread_LoadIndexNegative.start()

        self.Thread_LoadReference = Thread_LoadReference(REF_POS, REF_NEG)
        self.Thread_LoadReference._reference_positive.connect(self._set_reference_positive)
        self.Thread_LoadReference._reference_negative.connect(self._set_reference_negative)
        self.Thread_LoadReference.start()

        self._load_watcher = Thread_LoadWatcher(
            self.Thread_LoadIndexPositive,
            self.Thread_LoadIndexNegative,
            self.Thread_LoadReference,
        )
        self._load_watcher._done.connect(self._on_load_all_finished)
        self._load_watcher.start()

    def _on_load_all_finished(self):
        self._indices_refs_ready = True
        self._update_resource_loading_state()

    def _on_structure_db_loaded(self, df):
        self.default_database = df
        self._structure_db_ready = True
        try:
            th = threading.Thread(target=prebuild_database_cache, args=(df,), daemon=True)
            th.start()
        except Exception as e:
            print("prebuild_database_cache failed:", e)
        self._update_resource_loading_state()

    def _on_structure_db_failed(self, message):
        self.progressBar.setFormat("Structure DB failed")
        self.ErrorMsg(f"Missing/invalid structure DB: {STRUCT_DB}\n\n{message}")

    def _on_models_loaded(self, positive, negative):
        self.deepmass_positive = positive
        self.deepmass_negative = negative
        self._models_ready = True
        self._update_resource_loading_state()

    def _on_models_failed(self, message):
        self.progressBar.setFormat("Model loading failed")
        self.ErrorMsg(f"Cannot load Word2Vec models.\n{message}")

    def _update_resource_loading_state(self):
        ready_parts = sum(bool(x) for x in (
            self._indices_refs_ready, self._structure_db_ready, self._models_ready
        ))
        self.progressBar.setValue(int(ready_parts / 3 * 100))
        if not self._indices_refs_ready:
            msg = "Loading indices and references..."
        elif not self._structure_db_ready:
            msg = "Loading structure database..."
        elif not self._models_ready:
            msg = "Loading AnnoMASS structure models..."
        else:
            msg = f"Ready{self._workers_suffix()}"
        self.progressBar.setFormat(msg)
        self.lbl_time.setText("Ready\nResources loaded" if ready_parts == 3 else "Loading...\nResources")
        if ready_parts == 3:
            self._set_startup_controls_enabled(True)

    def _set_index_positive(self, msg):
        self.pp = msg

    def _set_index_negative(self, msg):
        self.pn = msg

    def _set_reference_positive(self, msg):
        self.reference_positive = msg

    def _set_reference_negative(self, msg):
        self.reference_negative = msg

    # ---------- load spectra ----------
    def load_spectrums(self):
        dlg = QtWidgets.QFileDialog(self, "Load Spectrum Files", "",
                                    "MS Files (*.mgf *.mat *.msp)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.setOptions(QtWidgets.QFileDialog.DontUseNativeDialog)
        dlg.setStyleSheet("""
            QWidget            { background: #FFFFFF; color: #222222; font-size: 12px; }
            QLabel             { color: #222222; background: transparent; }
            QPushButton        { background: #E8E8E8; color: #222222;
                                 border: 1px solid #C8C8C8; padding: 4px 14px;
                                 border-radius: 3px; }
            QPushButton:hover  { background: #D4D4D4; }
            QPushButton:pressed{ background: #BEBEBE; }
            QLineEdit          { background: #FFFFFF; color: #222222;
                                 border: 1px solid #C8C8C8; padding: 3px 6px;
                                 border-radius: 2px; }
            QTreeView, QListView { background: #FFFFFF; color: #222222;
                                   border: 1px solid #E0E0E0; outline: none; }
            QTreeView::item:selected, QListView::item:selected
                               { background: #D6EBFF; color: #222222; }
            QTreeView::item:hover, QListView::item:hover
                               { background: #EEF6FF; }
            QHeaderView::section { background: #F3F3F3; color: #555555;
                                   border: none; border-bottom: 1px solid #E0E0E0;
                                   padding: 4px 8px; }
            QComboBox          { background: #FFFFFF; color: #222222;
                                 border: 1px solid #C8C8C8; padding: 3px 6px; }
            QComboBox QAbstractItemView { background: #FFFFFF; color: #222222; }
            QToolBar           { background: #F5F5F5; border: none; }
            QToolButton        { background: transparent; color: #444444; border: none; }
            QToolButton:hover  { background: #E8E8E8; }
            QSplitter::handle  { background: #E0E0E0; }
            QScrollBar:vertical          { background: #F5F5F5; width: 10px; }
            QScrollBar::handle:vertical  { background: #C0C0C0; border-radius: 5px;
                                           min-height: 20px; }
            QScrollBar:horizontal        { background: #F5F5F5; height: 10px; }
            QScrollBar::handle:horizontal{ background: #C0C0C0; border-radius: 5px;
                                           min-width: 20px; }
            QScrollBar::add-line, QScrollBar::sub-line,
            QScrollBar::add-page, QScrollBar::sub-page { background: none; border: none; }
        """)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        fileNames = dlg.selectedFiles()
        if not fileNames:
            return

        self._loaded_files = fileNames

        try:
            spectrums = load_from_files(fileNames)
        except Exception as e:
            self.ErrorMsg(
                "Failed to load file(s).\n\n"
                "Tip: MGF with 'Num Peaks: 0' may crash parser.\n"
                "Please filter them out first.\n\n"
                f"Error:\n{e}"
            )
            return

        titles = []
        for s in spectrums:
            titles.append(s.metadata.get("compound_name", "") or s.metadata.get("title", "Unknown"))

        self.spectrums = pd.DataFrame({"title": titles, "spectrum": spectrums})
        self.set_list_spectrums()

    def set_list_spectrums(self):
        self.list_spectrum.clear()
        if len(self.spectrums) == 0:
            return
        for i in self.spectrums.index:
            title = str(self.spectrums.loc[i, "title"])
            item = QtWidgets.QListWidgetItem(title)
            item.setData(Qt.UserRole, int(i))
            self.list_spectrum.addItem(item)
        self.list_spectrum.setCurrentRow(0)

    def _plot_spectrum_object(self, sp):
        """Single shared entry point for showing a raw spectrum -- draws into
        the Results tab's own embedded canvas (self._pipeline_ax), the same
        one row-clicks on the results table use (_pipeline_plot_selected),
        so there is one spectrum viewer, not a separate standalone panel.
        Used by both the loaded-spectrum list click and the Agent's
        show_spectrum_plot tool."""
        self.current_spectrum = sp
        ax = getattr(self, "_pipeline_ax", None)
        canvas = getattr(self, "_pipeline_canvas", None)
        if ax is None or canvas is None:
            return
        try:
            ax.cla()
            mz  = np.asarray(sp.peaks.mz, float)
            ints = np.asarray(sp.peaks.intensities, float)
            if len(ints) > 0 and ints.max() > 0:
                ints = ints / ints.max()
            ax.vlines(mz, 0, ints, colors="#e74c3c", lw=1.2, label="Observed")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_ylim(0, 1.15)
            if len(mz):
                pad = max(10, (mz.max() - mz.min()) * 0.05)
                ax.set_xlim(max(0, mz.min() - pad), mz.max() + pad)
            ax.set_xlabel("m/z", fontsize=8)
            ax.set_ylabel("Relative Intensity", fontsize=8)
            ax.tick_params(labelsize=7)
            title = str(sp.get("compound_name") or sp.get("title") or "")
            ax.set_title(title, fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            canvas.draw()
            if hasattr(self, "_pipeline_spec_lbl"):
                self._pipeline_spec_lbl.setText(f"Loaded spectrum: {title or 'unknown'}")
        except Exception:
            self.WarnMsg("Cannot plot spectrum")

    def _on_spectrum_list_clicked(self, item):
        idx = item.data(Qt.UserRole)
        if idx is None:
            return
        try:
            sp = self.spectrums.loc[int(idx), "spectrum"]
        except Exception:
            return
        self._plot_spectrum_object(sp)

    # ---------- run ----------


    def plot_mol(self, smiles_override=None, name_hint=None):
        """Render a structure image to a temp PNG. Only caller left is the
        Agent's show_structure_plot tool (_agent_show_structure_execute),
        which always passes smiles_override and shows the PNG in its own
        embedded preview pane -- there is no manual-mode QLabel to draw into
        anymore (the redundant "Structure View" viewer-panel box, which
        duplicated the Pipeline results tab's own per-candidate structure
        preview, was removed)."""
        try:
            smi_anno = smiles_override
            mol_anno = Chem.MolFromSmiles(smi_anno)
            if mol_anno is None:
                return

            fn = _agent_temp_basename(name_hint)
            Draw.MolToFile(mol_anno, f"temp/{fn}.png", wedgeBonds=False)
            # Exposed so the Agent's show_structure_plot tool can hand the
            # exact rendered file back for the chat's clickable-link preview
            # (see _agent_show_structure_execute).
            self._last_structure_png_path = str(Path(f"temp/{fn}.png").resolve())
        except Exception:
            pass

    # ── Spectrum Predictor ───────────────────────────────────────────────────────────────

    _MF_COLS = ["#", "Compound", "Formula", "SMILES", "Pred Cosine",
                "Obs Peaks", "Pred Peaks", "Note"]





# ===================== Threads =====================
class Thread_LoadReference(QThread):
    _reference_positive = QtCore.pyqtSignal(list)
    _reference_negative = QtCore.pyqtSignal(list)

    def __init__(self, positive, negative):
        super().__init__()
        self.positive = positive
        self.negative = negative

    def run(self):
        with open(self.positive, "rb") as f:
            ref_pos = pickle.load(f)
        with open(self.negative, "rb") as f:
            ref_neg = pickle.load(f)
        self._reference_positive.emit(list(ref_pos))
        self._reference_negative.emit(list(ref_neg))

class Thread_LoadStructureDB(QThread):
    loaded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            df = pd.read_csv(self.csv_path, low_memory=False)
            self.loaded.emit(df)
        except Exception as e:
            self.failed.emit(str(e))

class Thread_LoadModels(QThread):
    loaded = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, positive_path, negative_path):
        super().__init__()
        self.positive_path = positive_path
        self.negative_path = negative_path

    def run(self):
        try:
            positive = Word2Vec.load(self.positive_path)
            negative = Word2Vec.load(self.negative_path)
            self.loaded.emit(positive, negative)
        except Exception as e:
            self.failed.emit(str(e))

class Thread_LoadIndex(QThread):
    _index = QtCore.pyqtSignal(hnswlib.Index)

    def __init__(self, spec_path):
        super().__init__()
        self.spec_path = spec_path

    def run(self):
        dim = 300 if "spec2vec" in self.spec_path.lower() else 200
        idx = Index(space="l2", dim=dim)
        idx.load_index(self.spec_path)
        self._index.emit(idx)

class Thread_LoadWatcher(QThread):
    _done = QtCore.pyqtSignal()

    def __init__(self, *threads):
        super().__init__()
        self.threads = threads

    def run(self):
        for th in self.threads:
            th.wait()
        self._done.emit()

# ===================== Plot Canvas =====================
class MakeFigure(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(top=0.95, bottom=0.25, left=0.12, right=0.98)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.tick_params(width=0.8, labelsize=7)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def PlotSpectrum(self, spectrum, reference=None, loss=False):
        self.axes.cla()
        if spectrum is None:
            self.draw()
            return

        mz, abunds = spectrum.peaks.mz, spectrum.peaks.intensities

        if loss:
            try:
                spectrum2 = msfilters.add_parent_mass(spectrum)
                spectrum2 = msfilters.add_losses(spectrum2, loss_mz_from=10.0, loss_mz_to=2000.0)
                mz, abunds = spectrum2.losses.mz, spectrum2.losses.intensities
            except Exception:
                return

        if len(abunds) > 0 and np.max(abunds) > 0:
            abunds = abunds / np.max(abunds)

        self.axes.vlines(mz, ymin=0, ymax=abunds, color="r", lw=1.2)
        self.axes.axhline(y=0, color="black", lw=0.6)
        if len(mz) > 0:
            self.axes.set_xlim(max(0, mz.min() - 10), mz.max() + 10)
        self.axes.set_ylim(-0.05, 1.15)
        self.axes.set_xlabel("m/z", fontsize=8)
        self.axes.set_ylabel("Relative Intensity", fontsize=8)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw()

def closeEvent(self, event):
    """Stop any running threads before closing to prevent QThread abort crash."""
    if hasattr(self, "_agent_save_sessions"):
        self._agent_save_sessions()
    proc = getattr(self, "_agent_terminal_process", None)
    notifier = getattr(self, "_agent_terminal_notifier", None)
    if notifier is not None:
        notifier.setEnabled(False)
    if proc is not None:
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    master_fd = getattr(self, "_agent_terminal_master_fd", None)
    if master_fd is not None:
        try:
            os.close(master_fd)
        except OSError:
            pass
    for attr in ("Thread_LoadIndexPositive", "Thread_LoadIndexNegative",
                 "Thread_LoadReference", "Thread_LoadModels", "Thread_LoadStructureDB",
                 "Thread_Pipeline", "Thread_Agent"):
        t = getattr(self, attr, None)
        if t is not None and t.isRunning():
            if hasattr(t, "requestInterruption"):
                t.requestInterruption()
            t.quit()
            if not t.wait(2000):
                # Still running -- e.g. blocked inside a synchronous
                # network call (Thread_Agent's LLM request) that never
                # checks isInterruptionRequested(). Left alone, this
                # process would linger in the background (invisible,
                # near-0% CPU) until that call eventually times out --
                # Qt's QThread destructor blocks on it during interpreter
                # shutdown. Force it down so the process actually exits.
                t.terminate()
                t.wait(1000)
    event.accept()




# ============================================================================
# Pipeline Worker Thread
# ============================================================================

class PipelineWorker(QtCore.QThread):
    """Staged pipeline: DreaMS (A) -> DeepMASS + library spectrum (B) ->
    DeepMASS + Spectrum Predictor (C) -> DeepMet CLM (D).

    Thin QThread wrapper around core.annomass_algorithm.run_annomass_pipeline,
    the single implementation of the AnnoMASS algorithm (shared with
    run_pipeline_headless.py and public_pip_ablation.py). This class only
    adapts that Qt-free function to Qt's signal/slot and thread-interruption
    conventions -- no algorithm logic lives here anymore.
    """
    progress  = QtCore.pyqtSignal(str)   # step description
    finished_with_result = QtCore.pyqtSignal(object)  # (DataFrame, mf_cache, denovo_df)
    failed    = QtCore.pyqtSignal(str)

    def __init__(self, spectrums, pp, pn, deepmass_pos, deepmass_neg,
                 ref_positive, ref_negative, default_database,
                 ref_hdf5, parent=None, **param_kwargs):
        """*param_kwargs* are AnnoMassParams fields (core/annomass_algorithm.py);
        options of the retired orthogonal-cascade architecture are ignored."""
        super().__init__(parent)
        self.spectrums         = spectrums
        self.pp                = pp
        self.pn                = pn
        self.deepmass_pos      = deepmass_pos
        self.deepmass_neg      = deepmass_neg
        self.ref_positive      = ref_positive
        self.ref_negative      = ref_negative
        self.default_database  = default_database
        self.ref_hdf5          = ref_hdf5
        param_kwargs.setdefault("use_deepmet_clm", True)
        self.params = annomass_algorithm.AnnoMassParams.from_kwargs(**param_kwargs)

    def run(self):
        try:
            result_df, mf_cache, denovo_df = annomass_algorithm.run_annomass_pipeline(
                self.spectrums, self.pp, self.pn, self.deepmass_pos, self.deepmass_neg,
                self.ref_positive, self.ref_negative, self.default_database, self.ref_hdf5,
                self.params,
                progress_cb=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
            self.finished_with_result.emit((result_df, mf_cache, denovo_df))
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


# ============================================================================
# Background thread: Spectrum Predictor prediction for pipeline row viewer
# ============================================================================

class _PipelineMFThread(QtCore.QThread):
    """Run mf_predict in background. Signal only carries int to avoid cross-thread
    object marshalling issues; results are stored as instance attributes."""
    done = QtCore.pyqtSignal(int)   # emits row_idx when finished

    def __init__(self, smiles, precursor_mz, adduct, nce, row_idx, parent=None):
        super().__init__(parent)
        self.smiles       = smiles
        self.precursor_mz = precursor_mz
        self.adduct       = adduct
        self.nce          = nce
        self.row_idx      = row_idx
        # Results written by run(), read by main thread after signal
        self.pred_mz      = []
        self.pred_int     = []
        self.mf_score     = None
        # Per-peak fragment atom-bitmask (parallel to pred_mz/pred_int) and
        # the canonical SMILES it was computed against -- feeds the
        # "click a predicted peak -> show fragment structure" viewer.
        self.frag_ids          = []
        self.frag_bits         = 0
        self.canonical_smiles  = ""

    def run(self):
        try:
            score, pred_mz, pred_int, frag_ids, frag_bits, canonical_smiles = mf_predict(
                smiles=self.smiles,
                query_mz=[],
                query_intensity=[],
                precursor_mz=self.precursor_mz,
                adduct=self.adduct,
                nce=self.nce,
                tol_da=0.5,
            )
            self.pred_mz     = list(pred_mz)
            self.pred_int    = list(pred_int)
            self.mf_score    = float(score)
            self.frag_ids    = list(frag_ids)
            self.frag_bits   = int(frag_bits)
            self.canonical_smiles = canonical_smiles
        except Exception as e:
            print(f"[Spectrum Predictor] prediction error: {e}")
        finally:
            self.done.emit(self.row_idx)


# ============================================================================
# Flavonoid Agent Worker Thread (standalone tab -- see core/agent_tools.py,
# core/agent_llm_client.py, core/agent_config.py)
# ============================================================================

class _MainThreadCallAwaiter(QtCore.QObject):
    """Generic "wait for one main-thread result" primitive shared by the
    run_pipeline and load_spectra_file cross-thread bridges. Must be
    constructed INSIDE the calling thread (e.g. inside AgentWorker.run()),
    so its slots have that thread's affinity -- Qt then auto-queues delivery
    into it when a main-thread signal fires. The queued delivery of
    on_ready/on_failed is what unblocks wait()'s QEventLoop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.error = None
        self._loop = QtCore.QEventLoop()

    @QtCore.pyqtSlot(object)
    def on_ready(self, payload):
        self.result = payload
        self._loop.quit()

    @QtCore.pyqtSlot(str)
    def on_failed(self, message):
        self.error = message
        self._loop.quit()

    def wait(self, timeout_ms):
        QtCore.QTimer.singleShot(timeout_ms, self._loop.quit)
        self._loop.exec_()
        if self.result is None and self.error is None:
            self.error = "Timed out waiting for the main thread to respond."


class AgentWorker(QtCore.QThread):
    """Runs the Flavonoid Agent's tool-calling loop off the GUI thread.
    core/agent_tools.py has the pure loop logic + tool registry;
    core/agent_llm_client.py has the LLM client. The two "bridge" tools
    (run_pipeline, load_spectra_file) touch main-thread-owned GUI state, so
    they're implemented here as methods that request the main thread do the
    work (via request_run_pipeline/request_load_spectra, wired up by the
    caller before .start()) and block only THIS thread until it responds."""
    progress = QtCore.pyqtSignal(str)
    usage_updated = QtCore.pyqtSignal(object)
    finished_with_result = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    request_run_pipeline = QtCore.pyqtSignal()
    request_load_spectra = QtCore.pyqtSignal(object)  # file_path string or file_paths list
    request_show_plot = QtCore.pyqtSignal(int)      # spectrum_index
    request_show_structure = QtCore.pyqtSignal(str, str)  # smiles, compound_name
    request_open_file = QtCore.pyqtSignal(str)      # file_path
    request_save_preview = QtCore.pyqtSignal(object)  # {destination_path, overwrite}
    request_reveal_file = QtCore.pyqtSignal(str)    # file_path -- fire-and-forget,
    request_tool_approval = QtCore.pyqtSignal(object)  # dict payload, handled on GUI thread
    # unlike the bridge signals above: any tool whose result includes a
    # "file_path" (export_pipeline_report, plot_pipeline_results,
    # write_pipeline_word_report, ...) gets it previewed + linked in chat
    # via this, without each such tool needing its own dedicated
    # request_*/_bridge_call plumbing (see _dispatch_tool).

    INITIAL_TOOL_ITERATIONS = 16
    APPROVAL_REQUIRED_TOOLS = {
        "write_project_file",
        "apply_project_patch",
        "run_project_tests",
        "run_project_command",
    }
    FAST_MODE_EXCLUDED_TOOLS = {
        "read_project_file",
        "list_project_files",
        "search_project",
        "write_project_file",
        "apply_project_patch",
        "get_project_diff",
        "summarize_project_changes",
        "review_project_code",
        "run_project_tests",
        "run_project_command",
    }

    def __init__(self, main_window, messages, provider_cfg, tool_log=None, work_mode="fast", parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.messages = messages
        self.provider_cfg = provider_cfg
        self.work_mode = str(work_mode or "fast").lower()
        # Shared list (same object as session["tool_log"]) that run_agent_loop
        # appends full {"name","args","result","duration_s","status"} records
        # to -- the chat log only ever shows a one-line "✓ name (Ns)"
        # summary; clicking that line looks the record up here by id (see
        # _agent_show_tool_detail).
        self.tool_log = tool_log if tool_log is not None else []

    def run(self):
        from core import agent_config as ac
        from core import agent_llm_client as llm
        from core import agent_memory
        from core import agent_skills
        from core import agent_tools as at
        try:
            client = llm.build_llm_client(self.provider_cfg)
            protocol = self.provider_cfg.get("protocol", "openai_compatible")
            tool_names = None
            if self.work_mode == "fast":
                tool_names = [
                    name for name in at.AGENT_TOOLS.keys()
                    if name not in self.FAST_MODE_EXCLUDED_TOOLS
                ]
            tool_schemas = llm.translate_tool_schemas(at.canonical_tool_schemas(tool_names), protocol)
            # Loaded fresh every run (not cached) so a remember_fact call made
            # earlier in the same conversation is visible on the next turn too.
            memory_text = agent_memory.load_memory()
            skills_text = agent_skills.load_skills_prompt()
            # Thinking-effort setting is also loaded fresh every run (cheap
            # file read) so changing it in Settings takes effect on the very
            # next message, no restart needed.
            effort = ac.get_thinking_effort()
            system_prompt = (
                at.SYSTEM_PROMPT
                + ("\n\n[Current GUI work mode]\nFast mode is active: answer and use scientific/result tools, but do not inspect, edit, or validate project code. Ask the user to switch to Code mode for code changes.\n"
                   if self.work_mode == "fast"
                   else "\n\n[Current GUI work mode]\nCode mode is active: project code tools are available, with GUI permission dialogs for writes and validation commands.\n")
                + at.EFFORT_INSTRUCTIONS.get(effort, "")
                + (f"\n\n[Remembered from earlier conversations]\n{memory_text}" if memory_text else "")
                + (f"\n\n[Installed Agent Skills]\n{skills_text}" if skills_text else "")
            )
            text = at.run_agent_loop(
                client, tool_schemas, self.messages, self._dispatch_tool,
                system_prompt=system_prompt, progress_cb=self.progress.emit,
                max_iters=self.INITIAL_TOOL_ITERATIONS,
                context_window=self.provider_cfg.get("context_window"),
                tool_log=self.tool_log,
                should_stop=self.isInterruptionRequested,
                usage_cb=self.usage_updated.emit,
            )
            self.finished_with_result.emit(text)
        except llm.LLMClientError as e:
            self.failed.emit(str(e))
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())

    def _dispatch_tool(self, name, args):
        from core import agent_tools as at
        if name == "run_pipeline":
            return self._tool_run_pipeline()
        if name == "load_spectra_file":
            return self._tool_load_spectra_file(args.get("file_paths") or args.get("file_path", ""))
        if name == "show_spectrum_plot":
            return self._tool_show_spectrum_plot(int(args.get("spectrum_index", -1)))
        if name == "show_structure_plot":
            return self._tool_show_structure_plot(args.get("smiles", ""), args.get("compound_name", ""))
        if name == "open_file":
            return self._tool_open_file(args.get("file_path", ""))
        if name == "save_current_preview":
            return self._tool_save_current_preview(
                args.get("destination_path", ""), bool(args.get("overwrite", False))
            )
        spec = at.AGENT_TOOLS.get(name)
        if spec is None or spec.fn is None:
            return {"error": f"Unknown tool '{name}'."}
        if name in self.APPROVAL_REQUIRED_TOOLS and not self._request_tool_approval(name, args):
            return {"error": f"Permission denied by user for tool '{name}'."}
        result = spec.fn(self.main_window, **args)
        # "the file you asked for is right there" --
        # generated-file tools (export_pipeline_report, plot_pipeline_results,
        # write_pipeline_word_report, ...) don't need their own bridge
        # signal/awaiter for this: a bare file_path in the result is enough
        # to have it previewed + linked in chat automatically, same as the
        # dedicated open_file tool already does.
        if isinstance(result, dict) and result.get("file_path"):
            self.request_reveal_file.emit(str(result["file_path"]))
        return result

    def _request_tool_approval(self, name, args):
        event = threading.Event()
        payload = {
            "name": name,
            "args": args,
            "approved": False,
            "event": event,
        }
        self.progress.emit(f"Requesting permission: {name}")
        self.request_tool_approval.emit(payload)
        while not event.wait(0.1):
            if self.isInterruptionRequested():
                return False
        return bool(payload.get("approved"))

    def _bridge_call(self, ready_signal, failed_signal, emit_request, timeout_ms):
        """Shared plumbing for the 5 bridge tools (run_pipeline, load_spectra_file,
        show_spectrum_plot, show_structure_plot, open_file): connect a fresh
        awaiter to the given ready/failed signal pair, fire the request, block
        this thread until a response arrives, then disconnect."""
        awaiter = _MainThreadCallAwaiter()
        ready_signal.connect(awaiter.on_ready)
        failed_signal.connect(awaiter.on_failed)
        emit_request()
        awaiter.wait(timeout_ms)
        ready_signal.disconnect(awaiter.on_ready)
        failed_signal.disconnect(awaiter.on_failed)
        return awaiter

    def _tool_run_pipeline(self):
        from core import agent_tools as at
        awaiter = _MainThreadCallAwaiter()
        self.main_window.agent_pipeline_result_ready.connect(awaiter.on_ready)
        self.main_window.agent_pipeline_result_failed.connect(awaiter.on_failed)
        self.progress.emit("Requesting AnnoMASS run on the main thread...")
        self.request_run_pipeline.emit()
        awaiter.wait(30 * 60 * 1000)  # full pipeline can be slow
        self.main_window.agent_pipeline_result_ready.disconnect(awaiter.on_ready)
        self.main_window.agent_pipeline_result_failed.disconnect(awaiter.on_failed)
        if awaiter.error is not None:
            return {"error": awaiter.error}
        if isinstance(awaiter.result, tuple):
            df, mf_cache = awaiter.result[0], (awaiter.result[1] if len(awaiter.result) > 1 else {})
        else:
            df, mf_cache = awaiter.result, {}
        if df is None or len(df) == 0:
            return {"error": "AnnoMASS ran but produced no candidates."}
        records = at._safe_records(df, at._PIPELINE_RESULT_COLUMNS, limit=20)
        return {"cascade_explanation": at.CASCADE_EXPLANATION_TEXT, "n_candidates": int(len(df)), "candidates": records}

    def _tool_load_spectra_file(self, file_path):
        awaiter = _MainThreadCallAwaiter()
        self.main_window.agent_load_spectra_ready.connect(awaiter.on_ready)
        self.main_window.agent_load_spectra_failed.connect(awaiter.on_failed)
        self.progress.emit(f"Requesting file load on the main thread: {file_path}")
        self.request_load_spectra.emit(file_path)
        awaiter.wait(2 * 60 * 1000)  # just file parsing, should be fast
        self.main_window.agent_load_spectra_ready.disconnect(awaiter.on_ready)
        self.main_window.agent_load_spectra_failed.disconnect(awaiter.on_failed)
        if awaiter.error is not None:
            return {"error": awaiter.error}
        return awaiter.result

    def _tool_show_spectrum_plot(self, spectrum_index):
        self.progress.emit(f"Requesting spectrum plot on the main thread: index {spectrum_index}")
        awaiter = self._bridge_call(
            self.main_window.agent_plot_ready, self.main_window.agent_plot_failed,
            lambda: self.request_show_plot.emit(spectrum_index),
            30 * 1000,  # just repopulating a plot, should be quick
        )
        if awaiter.error is not None:
            return {"error": awaiter.error}
        return awaiter.result

    def _tool_show_structure_plot(self, smiles, compound_name=""):
        self.progress.emit(f"Requesting structure plot on the main thread: {smiles}")
        awaiter = self._bridge_call(
            self.main_window.agent_plot_ready, self.main_window.agent_plot_failed,
            lambda: self.request_show_structure.emit(smiles, str(compound_name or "")),
            30 * 1000,
        )
        if awaiter.error is not None:
            return {"error": awaiter.error}
        return awaiter.result

    def _tool_open_file(self, file_path):
        self.progress.emit(f"Requesting file viewer open on the main thread: {file_path}")
        awaiter = self._bridge_call(
            self.main_window.agent_file_ready, self.main_window.agent_file_failed,
            lambda: self.request_open_file.emit(file_path),
            30 * 1000,
        )
        if awaiter.error is not None:
            return {"error": awaiter.error}
        return awaiter.result

    def _tool_save_current_preview(self, destination_path, overwrite=False):
        self.progress.emit(f"Saving current Agent preview to: {destination_path}")
        awaiter = self._bridge_call(
            self.main_window.agent_preview_save_ready,
            self.main_window.agent_preview_save_failed,
            lambda: self.request_save_preview.emit({
                "destination_path": destination_path,
                "overwrite": overwrite,
            }),
            30 * 1000,
        )
        if awaiter.error is not None:
            return {"error": awaiter.error}
        return awaiter.result


class AgentTestConnectionWorker(QtCore.QThread):
    """Used only by the Settings dialog's "Test Connection" button."""
    progress = QtCore.pyqtSignal(str)
    finished_with_result = QtCore.pyqtSignal(object)  # (ok: bool, message: str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, provider_cfg, parent=None):
        super().__init__(parent)
        self.provider_cfg = provider_cfg

    def run(self):
        try:
            from core import agent_llm_client as llm
            self.progress.emit("Testing connection...")
            ok, message = llm.test_connection(self.provider_cfg)
            self.finished_with_result.emit((ok, message))
        except Exception as e:
            self.failed.emit(str(e))




# ============================================================================
# Pipeline Tab  (added dynamically to tabWidget_results)
# ============================================================================

class _StructurePreviewLabel(QtWidgets.QLabel):
    """QLabel that re-renders its content (full candidate structure or a
    clicked fragment's SVG) at the widget's *current* size on every resize,
    instead of keeping one fixed-resolution pixmap from whenever it was
    first drawn -- so dragging the splitter to enlarge the panel actually
    gives a bigger/sharper picture instead of the same small pixmap sitting
    in a bigger empty box."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = None      # None | "structure" | "fragment"
        self._smiles = ""
        self._svg = ""
        self.setText("No structure")

    def show_structure(self, smiles):
        self._mode = "structure"
        self._smiles = str(smiles or "")
        self._render()

    def show_fragment(self, svg):
        self._mode = "fragment"
        self._svg = svg or ""
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        w = max(60, self.width())
        h = max(60, self.height())
        if self._mode == "structure":
            self._render_structure(w, h)
        elif self._mode == "fragment":
            self._render_fragment(w, h)

    def _render_structure(self, w, h):
        smi = self._smiles.strip()
        if not smi or smi.lower() in ("nan", "none", "null"):
            self.setText("No SMILES")
            return
        try:
            from io import BytesIO
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                self.setText("Invalid SMILES")
                return
            img = Draw.MolToImage(mol, size=(w, h), wedgeBonds=True)
            buf = BytesIO()
            img.save(buf, format="PNG")
            px = QtGui.QPixmap()
            px.loadFromData(buf.getvalue(), "PNG")
            self.setPixmap(px)
        except Exception:
            self.setText("Structure failed")

    def _render_fragment(self, w, h):
        if not self._svg:
            self.setText("No fragment")
            return
        try:
            from PyQt5 import QtSvg
            renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(self._svg.encode("utf-8")))
            if not renderer.isValid():
                self.setText("Fragment render failed")
                return
            px = QtGui.QPixmap(w, h)
            px.fill(Qt.white)
            painter = QtGui.QPainter(px)
            renderer.render(painter)
            painter.end()
            self.setPixmap(px)
        except Exception:
            self.setText("Fragment render failed")


class _NullRow:
    """Swallows addWidget/addStretch: the parameter widgets are built in _init_pipeline_tab but laid out
    in the Settings dialog (see _build_param_settings) instead of a long, truncated bar."""
    def setSpacing(self, *a, **k): pass
    def addWidget(self, *a, **k): pass
    def addStretch(self, *a, **k): pass


def _build_param_settings(self, bar_layout, tw):
    """Settings page (a tab of the results tab widget, opened by the Settings button) holding every run
    parameter with full labels, plus a slim bar with the Settings button and a one-line summary. The widgets
    stay the same self._pl_* objects, so run_pipeline reads them exactly as before; only where they are shown
    changed. The page is added to `tw` by _init_pipeline_tab (after the Results tab)."""
    page = QtWidgets.QWidget()
    page.setObjectName("tab_settings_page")
    outer = QtWidgets.QVBoxLayout(page)
    outer.setContentsMargins(10, 8, 10, 8)

    def _goto_results():
        for i in range(tw.count()):
            if tw.tabText(i) == "Results":
                tw.setCurrentIndex(i)
                return

    head = QtWidgets.QHBoxLayout()
    btn_back = QtWidgets.QPushButton("←  Back to Results")
    btn_back.clicked.connect(_goto_results)
    head.addWidget(btn_back)
    title = QtWidgets.QLabel("<b>Settings</b>")
    title.setStyleSheet("font-size: 15px;")
    head.addWidget(title)
    head.addStretch()
    btn_reset = QtWidgets.QPushButton("Restore defaults")
    head.addWidget(btn_reset)
    outer.addLayout(head)

    cols = QtWidgets.QHBoxLayout()
    left = QtWidgets.QVBoxLayout()
    right = QtWidgets.QVBoxLayout()
    cols.addLayout(left, 1)
    cols.addLayout(right, 1)
    outer.addLayout(cols)
    outer.addStretch(1)

    def _group(side, title_, rows):
        box = QtWidgets.QGroupBox(title_)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setHorizontalSpacing(16)
        for label, widget in rows:
            if label is None:
                form.addRow(widget)
            else:
                form.addRow(label, widget)
        side.addWidget(box)

    def _pair(w1, w2):
        h = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(h)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(w1)
        lay.addWidget(w2)
        lay.addStretch()
        return h

    # the old bar used narrow fixed widths to fit one row; there is room here, so show full values
    for _w in (self._pl_dreams_sim_thr, self._pl_mf_nce, self._pl_top_frags, self._pl_ice_primary_thr,
               self._pl_ice_secondary_thr, self._pl_ice_peak_min_pct, self._pl_ice_peak_top_k, self._pl_dm_min_support):
        _w.setFixedWidth(96)
    for _w in (self._pl_mf_instrument, self._pl_fiddle_instrument):
        _w.setFixedWidth(130)
    self._pl_abcort_checkpoint.setFixedWidth(190)

    # Level weights/thresholds are fixed calibrated values (core.annomass_algorithm.
    # AnnoMassParams); the retired DreaMS-threshold and DeepMASS-only-support
    # widgets are kept only so saved settings still load.
    self._pl_dreams_sim_thr.setVisible(False)
    self._pl_dm_min_support.setVisible(False)
    _group(left, "Run modules (untick to skip a level)", [
        (None, self._pl_mod_chks["A"]),
        (None, self._pl_mod_chks["B"]),
        (None, self._pl_mod_chks["C"]),
        (None, self._pl_deepmet_clm_chk),
    ])
    _group(left, "Library match", [
        (None, self._pl_cho_chk),
    ])
    _group(left, "Fragment simulation", [
        ("Collision energy (NCE)", self._pl_mf_nce),
        ("Instrument", self._pl_mf_instrument),
        ("Fragments shown on the predicted spectrum", self._pl_top_frags),
    ])
    _group(left, "Level D thresholds (ICE entropy / cosine)", [
        ("Primary threshold", self._pl_ice_primary_thr),
        ("Secondary threshold", self._pl_ice_secondary_thr),
    ])
    left.addStretch(1)
    _group(right, "ICE peak cleaning", [
        ("Minimum relative intensity (%)", self._pl_ice_peak_min_pct),
        ("Strongest peaks kept", self._pl_ice_peak_top_k),
    ])
    _group(right, "Optional refinements", [
        (None, self._pl_annosm_chk),
        (None, _pair(self._pl_abcort_chk, self._pl_abcort_checkpoint)),
        (None, _pair(self._pl_fiddle_chk, self._pl_fiddle_instrument)),
    ])
    right.addStretch(1)

    # defaults = the values the widgets have right now (set above in _init_pipeline_tab)
    def _val(w):
        if isinstance(w, QtWidgets.QAbstractButton):
            return w.isChecked()
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentIndex()
        return w.value()
    _widgets = [self._pl_dreams_sim_thr, self._pl_cho_chk, self._pl_mf_nce, self._pl_mf_instrument, self._pl_top_frags,
                self._pl_ice_primary_thr, self._pl_ice_secondary_thr, self._pl_ice_peak_min_pct, self._pl_ice_peak_top_k,
                self._pl_annosm_chk, self._pl_abcort_chk, self._pl_abcort_checkpoint, self._pl_fiddle_chk,
                self._pl_fiddle_instrument, self._pl_deepmet_clm_chk, self._pl_dm_min_support,
                *self._pl_mod_chks.values()]
    _defaults = [(w, _val(w)) for w in _widgets]

    def _restore():
        for w, v in _defaults:
            if not w.isEnabled():
                continue
            if isinstance(w, QtWidgets.QAbstractButton):
                w.setChecked(bool(v))
            elif isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(int(v))
            else:
                w.setValue(v)
    btn_reset.clicked.connect(_restore)
    self._settings_page = page

    # slim bar: Settings button (jumps to the Settings page) + one-line summary of the current parameters
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(10)
    btn = QtWidgets.QPushButton("Settings")
    btn.setToolTip("Open the Settings page (library match, fragment simulation, ICE thresholds, optional refinements).")
    btn.clicked.connect(lambda: tw.setCurrentWidget(page))
    row.addWidget(btn)
    summary = QtWidgets.QLabel()
    summary.setStyleSheet("color: #555555;")
    row.addWidget(summary, 1)
    bar_layout.addLayout(row)
    self._param_summary_lbl = summary

    def _refresh(*_a):
        _mods = "".join(k for k, c in (*self._pl_mod_chks.items(), ("D", self._pl_deepmet_clm_chk)) if c.isChecked())
        parts = ["Modules " + ("→".join(_mods) if _mods else "none") + (", CHO only" if self._pl_cho_chk.isChecked() else ""),
                 f"NCE {self._pl_mf_nce.value():g}, {self._pl_mf_instrument.currentText()}",
                 f"D thresholds {self._pl_ice_primary_thr.value():.2f} / {self._pl_ice_secondary_thr.value():.2f}"]
        opts = [name for chk, name in ((self._pl_annosm_chk, "substituent check"), (self._pl_abcort_chk, "RT prediction"),
                                       (self._pl_fiddle_chk, "formula pre-filter"))
                if chk.isChecked()]
        parts.append("Options: " + (", ".join(opts) if opts else "none"))
        summary.setText("   |   ".join(parts))

    for w in _widgets:
        for sig in ("valueChanged", "toggled", "currentIndexChanged"):
            if hasattr(w, sig):
                getattr(w, sig).connect(_refresh)
    _refresh()


def _is_category_d(df):
    """Boolean mask of category-D (DeepMet CLM, library-external) rows."""
    src = df.get("candidate_source", pd.Series("", index=df.index)).fillna("").astype(str)
    lvl = df.get("confidence_level", pd.Series("", index=df.index)).fillna("").astype(str)
    return src.eq("deepmet_clm") | lvl.eq("D")


class _ItemBackgroundDelegate(QtWidgets.QStyledItemDelegate):
    """Paint Qt.BackgroundRole ourselves.

    The theme stylesheets (plus LIGHTWEIGHT_WORKSPACE_CHROME) define
    ``QTableWidget::item`` rules; once those exist Qt draws the item through the
    style sheet and silently ignores ``QTableWidgetItem.setBackground``, so the
    Result-group colours never showed.  Filling the cell here, before the normal
    painting, makes the colours independent of the active theme.  Selected rows
    keep the theme's selection colour.
    """

    def paint(self, painter, option, index):
        brush = index.data(QtCore.Qt.BackgroundRole)
        if brush is not None and not (option.state & QtWidgets.QStyle.State_Selected):
            painter.save()
            painter.fillRect(option.rect, brush)
            painter.restore()
        super().paint(painter, option, index)


def _init_pipeline_tab(self):
    """Create the AnnoMASS results tab with spectrum viewer."""
    tw = self.findChild(QtWidgets.QTabWidget, "tabWidget_results")
    if tw is None:
        return

    page = QtWidgets.QWidget()
    page.setObjectName("tab_pipeline_page")
    vlay = QtWidgets.QVBoxLayout(page)
    vlay.setContentsMargins(6, 6, 6, 6)
    vlay.setSpacing(4)

    # ── Parameter bar ────────────────────────────────────────────────────────
    # Two stacked rows (not one long QHBoxLayout) so nothing overlaps/gets
    # clipped on a narrow window -- a single row of ~20 widgets doesn't wrap
    # in Qt and overflowed on small screens. Labels describe what each
    # setting actually does rather than which internal technique implements
    # it (DreaMS/Spectrum Predictor/AnnoSM/ABCoRT/FIDDLE) -- those names still show
    # up in tooltips for anyone who wants the technical detail.
    self._param_bar = QtWidgets.QFrame()
    self._param_bar.setObjectName("paramBar")
    self._param_bar.setFrameShape(QtWidgets.QFrame.StyledPanel)
    self._param_bar.setStyleSheet("QFrame { background: #f5f5f5; border-radius: 4px; }")
    param_vlay = QtWidgets.QVBoxLayout(self._param_bar)
    param_vlay.setContentsMargins(8, 4, 8, 4)
    param_vlay.setSpacing(4)

    row1 = _NullRow()
    row2 = _NullRow()

    def _vsep():
        s = QtWidgets.QFrame()
        s.setFrameShape(QtWidgets.QFrame.VLine)
        s.setFrameShadow(QtWidgets.QFrame.Sunken)
        return s

    # Row 1: core evidence-gathering settings (spectral library match, MS/MS
    # fragment simulation).
    row1.addWidget(QtWidgets.QLabel("<b>Library match</b>"))
    row1.addWidget(QtWidgets.QLabel("Min similarity:"))
    self._pl_dreams_sim_thr = QtWidgets.QDoubleSpinBox()
    self._pl_dreams_sim_thr.setRange(0.0, 1.0)
    self._pl_dreams_sim_thr.setDecimals(2)
    self._pl_dreams_sim_thr.setSingleStep(0.05)
    self._pl_dreams_sim_thr.setValue(0.70)
    self._pl_dreams_sim_thr.setFixedWidth(68)
    self._pl_dreams_sim_thr.setToolTip(
        "Minimum spectral-library similarity threshold (0-1, recommended 0.70; "
        "engine: DreaMS embedding search).\n"
        "Top-100 hits are recalled from the library; only hits with similarity >= this value are kept.\n"
        "The highest-scoring compound per spectrum is shown in the final results."
    )
    row1.addWidget(self._pl_dreams_sim_thr)

    self._pl_cho_chk = QtWidgets.QCheckBox("CHO only")
    self._pl_cho_chk.setChecked(True)
    self._pl_cho_chk.setToolTip("Search only CHO compounds (filter out heteroatom-containing structures)")
    row1.addWidget(self._pl_cho_chk)

    row1.addWidget(_vsep())

    # Adduct is auto-detected per spectrum (see
    # core.annomass_algorithm._spectrum_adduct -- reads the MGF's own
    # adduct/precursor_type field, falling back to an ionmode-based guess),
    # not a manual per-run setting: no dropdown here on purpose. NCE (collision
    # energy) is a real instrument setting, independent of adduct, so it stays.
    row1.addWidget(QtWidgets.QLabel("<b>Fragment simulation</b>"))
    row1.addWidget(QtWidgets.QLabel("Collision energy (NCE):"))
    self._pl_mf_nce = QtWidgets.QDoubleSpinBox()
    self._pl_mf_nce.setRange(0.0, 200.0)
    self._pl_mf_nce.setDecimals(1)
    self._pl_mf_nce.setSingleStep(5.0)
    self._pl_mf_nce.setValue(30.0)
    self._pl_mf_nce.setFixedWidth(72)
    self._pl_mf_nce.setToolTip("Normalized Collision Energy for simulated MS/MS matching (e.g. 20 / 40 / 60; engine: ICEBERG/ms-pred)")
    row1.addWidget(self._pl_mf_nce)

    row1.addWidget(QtWidgets.QLabel("Instrument:"))
    self._pl_mf_instrument = QtWidgets.QComboBox()
    self._pl_mf_instrument.addItems(["Orbitrap", "QTOF", "IT-FT", "Unknown"])
    self._pl_mf_instrument.setCurrentText("Unknown")
    self._pl_mf_instrument.setFixedWidth(88)
    self._pl_mf_instrument.setToolTip(
        "Instrument type for fragment simulation (engine: ICEBERG/ms-pred).\n"
        "Orbitrap = Orbitrap HCD, IT-FT = Orbitrap CID, QTOF = quadrupole "
        "time-of-flight.\n"
        "These are the only categories the model was trained on; anything "
        "else falls back to Orbitrap."
    )
    row1.addWidget(self._pl_mf_instrument)

    row1.addWidget(QtWidgets.QLabel("ICE primary:"))
    self._pl_ice_primary_thr = QtWidgets.QDoubleSpinBox()
    self._pl_ice_primary_thr.setRange(0.0, 1.0)
    self._pl_ice_primary_thr.setDecimals(2)
    self._pl_ice_primary_thr.setSingleStep(0.05)
    self._pl_ice_primary_thr.setValue(0.50)
    self._pl_ice_primary_thr.setFixedWidth(68)
    self._pl_ice_primary_thr.setToolTip(
        "Level D (DeepMet CLM) threshold: one of the ICEBERG entropy / cosine "
        "similarities must reach this primary threshold. Default: 0.50."
    )
    row1.addWidget(self._pl_ice_primary_thr)

    row1.addWidget(QtWidgets.QLabel("ICE secondary:"))
    self._pl_ice_secondary_thr = QtWidgets.QDoubleSpinBox()
    self._pl_ice_secondary_thr.setRange(0.0, 1.0)
    self._pl_ice_secondary_thr.setDecimals(2)
    self._pl_ice_secondary_thr.setSingleStep(0.05)
    self._pl_ice_secondary_thr.setValue(0.40)
    self._pl_ice_secondary_thr.setFixedWidth(68)
    self._pl_ice_secondary_thr.setToolTip(
        "Level D (DeepMet CLM) threshold: the other ICEBERG similarity must reach this "
        "companion threshold (either metric may be the primary one). Default: 0.40."
    )
    row1.addWidget(self._pl_ice_secondary_thr)

    row1.addWidget(QtWidgets.QLabel("Top fragments:"))
    self._pl_top_frags = QtWidgets.QSpinBox()
    self._pl_top_frags.setRange(1, 100)
    self._pl_top_frags.setValue(100)
    self._pl_top_frags.setFixedWidth(52)
    self._pl_top_frags.setToolTip(
        "How many of the highest-intensity simulated fragment peaks to show "
        "(and label with their m/z) on the predicted spectrum below.\n"
        "Default 100 shows the complete sparse Spectrum Predictor output. "
        "Applies to the plot only -- does not affect scoring or ranking."
    )
    self._pl_top_frags.valueChanged.connect(lambda _v: _pipeline_plot_selected(self))
    row1.addWidget(self._pl_top_frags)

    self._pl_dm_min_support = QtWidgets.QSpinBox()
    self._pl_dm_min_support.setRange(1, 20)
    self._pl_dm_min_support.setValue(3)
    self._pl_dm_min_support.setFixedWidth(52)
    self._pl_dm_min_support.setToolTip(
        "Minimum number of supporting spectra for a candidate whose only evidence is\n"
        "DeepMASS (no library match, no ICEBERG pass). Default 3 (GNPS-522 benchmark rule).\n"
        "Each spectrum is ranked on its own here, so the support count is 1: for a single\n"
        "injection / unrelated spectra set this to 1 to keep DeepMASS-only candidates.\n"
        "Lower values report more candidates, including weakly supported ones."
    )
    row1.addWidget(QtWidgets.QLabel("Min support (DeepMASS-only):"))
    row1.addWidget(self._pl_dm_min_support)
    row1.addStretch()

    # Row 2: ICEBERG peak cleaning and optional refinement toggles.
    row2.addWidget(QtWidgets.QLabel("<b>ICE peak cleaning:</b>"))
    row2.addWidget(QtWidgets.QLabel("Min relative intensity (%):"))
    self._pl_ice_peak_min_pct = QtWidgets.QDoubleSpinBox()
    self._pl_ice_peak_min_pct.setRange(0.0, 100.0)
    self._pl_ice_peak_min_pct.setDecimals(1)
    self._pl_ice_peak_min_pct.setSingleStep(0.5)
    self._pl_ice_peak_min_pct.setValue(1.0)
    self._pl_ice_peak_min_pct.setFixedWidth(68)
    self._pl_ice_peak_min_pct.setToolTip(
        "Before ICEBERG entropy/cosine scoring, normalize each spectrum to its "
        "own base peak and discard weaker peaks. Default 1.0% removes low-level "
        "noise without using instrument-specific absolute intensities."
    )
    row2.addWidget(self._pl_ice_peak_min_pct)

    row2.addWidget(QtWidgets.QLabel("Strongest peaks:"))
    self._pl_ice_peak_top_k = QtWidgets.QSpinBox()
    self._pl_ice_peak_top_k.setRange(0, 1000)
    self._pl_ice_peak_top_k.setValue(80)
    self._pl_ice_peak_top_k.setFixedWidth(62)
    self._pl_ice_peak_top_k.setSpecialValueText("All")
    self._pl_ice_peak_top_k.setToolTip(
        "Maximum number of strongest peaks retained on both experimental and "
        "predicted spectra for ICEBERG scoring. Default 80; 0 keeps all peaks "
        "that pass the relative-intensity threshold."
    )
    row2.addWidget(self._pl_ice_peak_top_k)
    row2.addWidget(_vsep())

    row2.addWidget(QtWidgets.QLabel("<b>Optional refinements:</b>"))

    # AnnoSM evidence rerank (see core/annosm_bridge.py)
    from core import annosm_bridge as _ab_check
    self._pl_annosm_chk = QtWidgets.QCheckBox("Substituent-position check")
    self._pl_annosm_chk.setChecked(True)
    self._pl_annosm_chk.setToolTip(
        "Re-examines isomer candidates for diagnostic fragment evidence of a\n"
        "different substitution mode (engine: AnnoSM, extended library).\n"
        "On the 791-spectrum flavonoid benchmark this moves Top1 651 -> 656\n"
        "(only promotes a same-formula, same-evidence-tier challenger over the\n"
        "current rank-1; adds fragment-matching time per candidate spectrum)."
    )
    if not _ab_check.is_available():
        self._pl_annosm_chk.setChecked(False)
        self._pl_annosm_chk.setEnabled(False)
        self._pl_annosm_chk.setToolTip(f"Substituent-position check unavailable: {_ab_check.unavailable_reason()}")
    row2.addWidget(self._pl_annosm_chk)

    # ABCoRT predicted retention time (see core/abcort_bridge.py)
    from core import abcort_bridge as _abc_check
    self._pl_abcort_chk = QtWidgets.QCheckBox("Predict retention time")
    self._pl_abcort_chk.setChecked(False)
    self._pl_abcort_chk.setToolTip(
        "Predict retention time (GNN model, SMRT-pretrained or your own\n"
        "fine-tuned checkpoint; engine: ABCoRT) for each final candidate.\n"
        "Purely informational: adds a 'predicted_rt_min' column, never\n"
        "affects ranking or filtering."
    )
    row2.addWidget(self._pl_abcort_chk)

    self._pl_abcort_checkpoint = QtWidgets.QComboBox()
    for label, path in _abc_check.list_checkpoints():
        self._pl_abcort_checkpoint.addItem(label, path)
    self._pl_abcort_checkpoint.setFixedWidth(150)
    self._pl_abcort_checkpoint.setToolTip("Which retention-time model checkpoint to use for prediction.")
    row2.addWidget(self._pl_abcort_checkpoint)

    if not _abc_check.is_available():
        self._pl_abcort_chk.setChecked(False)
        self._pl_abcort_chk.setEnabled(False)
        self._pl_abcort_checkpoint.setEnabled(False)
        self._pl_abcort_chk.setToolTip(f"Retention-time prediction unavailable: {_abc_check.unavailable_reason()}")

    # FIDDLE formula pre-filter (see core/fiddle_adapter.py)
    from core import fiddle_adapter as _fiddle_check
    self._pl_fiddle_chk = QtWidgets.QCheckBox("Formula pre-filter")
    self._pl_fiddle_chk.setChecked(False)
    self._pl_fiddle_chk.setToolTip(
        "Deep-learning molecular-formula prediction (engine: FIDDLE).\n"
        "Pre-filters database candidates to those matching the predicted\n"
        "formula before scoring. A candidate pre-filter, not part of ranking."
    )
    row2.addWidget(self._pl_fiddle_chk)

    self._pl_fiddle_instrument = QtWidgets.QComboBox()
    self._pl_fiddle_instrument.addItems(["Orbitrap", "Q-TOF"])
    self._pl_fiddle_instrument.setFixedWidth(90)
    self._pl_fiddle_instrument.setToolTip("MS instrument type (selects the pretrained formula-prediction model).")
    row2.addWidget(self._pl_fiddle_instrument)

    if not _fiddle_check.is_available("Orbitrap"):
        self._pl_fiddle_chk.setChecked(False)
        self._pl_fiddle_chk.setEnabled(False)
        self._pl_fiddle_instrument.setEnabled(False)
        self._pl_fiddle_chk.setToolTip("Formula pre-filter unavailable (model files not found).")

    # Parallel library-external route backed by DeepMet's flavonoid CLM.
    from core import deepmet_clm_adapter as _clm_check
    self._pl_deepmet_clm_chk = QtWidgets.QCheckBox("DeepMet CLM")
    self._pl_deepmet_clm_chk.setChecked(True)
    self._pl_deepmet_clm_chk.setToolTip(
        "Run every spectrum through the DeepMet CLM candidate pool in parallel\n"
        "with the AnnoMASS library route. Candidates already in the active\n"
        "library are removed; external candidates are scored by ICEBERG."
    )
    if not _clm_check.is_available():
        self._pl_deepmet_clm_chk.setChecked(False)
        self._pl_deepmet_clm_chk.setEnabled(False)
        self._pl_deepmet_clm_chk.setToolTip("DeepMet CLM candidate pool unavailable.")
    row2.addWidget(self._pl_deepmet_clm_chk)

    # Module selection (levels A-D of the staged waterfall, core/staged_levels.py).  A skipped module is not computed and
    # never passes; D is the CLM checkbox above.  Without A, the B score is renormalised over DeepMASS + library cosine.
    _mod_tips = {
        "A": ("A  DreaMS match (MSI 2)",
              "Level A: DreaMS similarity of the spectrum to the library, combined with the cosine to the matched library\n"
              "spectrum (0.8 / 0.2); passes at >= 0.70.  Result: MSI Level 2."),
        "B": ("B  DeepMASS library (MSI 2)",
              "Level B: structure-database candidate that also has a library spectrum. DeepMASS score + library cosine\n"
              "+ DreaMS similarity (0.5 / 0.2 / 0.3); passes at >= 0.97.  Result: MSI Level 2."),
        "C": ("C  ICEBERG prediction (MSI 3)",
              "Level C: structure-database candidate without a library spectrum. Level-B score carried forward plus the\n"
              "ICEBERG predicted-spectrum cosine (0.3 / 0.7); passes at >= 0.12.  Result: MSI Level 3."),
    }
    self._pl_mod_chks = {}
    for _k, (_lab, _tip) in _mod_tips.items():
        _c = QtWidgets.QCheckBox(_lab)
        _c.setChecked(True)
        _c.setToolTip(_tip)
        self._pl_mod_chks[_k] = _c
    self._pl_deepmet_clm_chk.setText("D  DeepMet CLM (MSI 4)")
    self._pl_deepmet_clm_chk.setToolTip(
        "Level D: generative candidates (DeepMet CLM) that are not in the active library; both the ICEBERG and the\n"
        "entropy score must pass (>= 0.5).  Result: MSI Level 4.\n"
        "Run every spectrum through the CLM candidate pool in parallel with the library route; candidates already in\n"
        "the active library are removed."
    )

    row2.addStretch()

    _build_param_settings(self, param_vlay, tw)

    vlay.addWidget(self._param_bar)

    # Status label
    self._pipeline_status_lbl = QtWidgets.QLabel(
        "Not run yet. Click [Run] in the toolbar to start."
    )
    self._pipeline_status_lbl.setWordWrap(True)
    vlay.addWidget(self._pipeline_status_lbl)

    # ── Splitter: table (top) + spectrum viewer (bottom) ─────────────────
    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

    # Results table
    self._pipeline_result_table = QtWidgets.QTableWidget()
    self._pipeline_result_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    self._pipeline_result_table.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
    self._pipeline_result_table.horizontalHeader().setStretchLastSection(True)
    self._pipeline_result_table.setAlternatingRowColors(True)
    self._pipeline_result_table.setItemDelegate(
        _ItemBackgroundDelegate(self._pipeline_result_table)
    )
    splitter.addWidget(self._pipeline_result_table)

    # Spectrum viewer panel
    spec_panel = QtWidgets.QWidget()
    spec_vlay  = QtWidgets.QVBoxLayout(spec_panel)
    spec_vlay.setContentsMargins(2, 2, 2, 2)
    spec_vlay.setSpacing(2)

    spec_lbl = QtWidgets.QLabel(
        "← Click a row to view its spectrum (red=observed, blue=predicted or best-match library spectrum ↓). "
        "Click a blue peak to see its fragment structure in the panel on the right."
    )
    spec_lbl.setObjectName("sectionTitle")
    spec_lbl.setStyleSheet("font-size: 11px;")
    self._pipeline_spec_lbl = spec_lbl
    spec_vlay.addWidget(spec_lbl)

    preview_row = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

    spec_box = QtWidgets.QWidget()
    spec_box_lay = QtWidgets.QVBoxLayout(spec_box)
    spec_box_lay.setContentsMargins(0, 0, 0, 0)

    # Canvas and fragment table share a vertical splitter (not a fixed
    # layout) so the user can drag the divider to resize either one.
    spec_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

    canvas_box = QtWidgets.QWidget()
    canvas_box_lay = QtWidgets.QVBoxLayout(canvas_box)
    canvas_box_lay.setContentsMargins(0, 0, 0, 0)
    self._pipeline_fig  = Figure(figsize=(6, 2.2), dpi=90)
    self._pipeline_ax   = self._pipeline_fig.add_subplot(111)
    self._pipeline_canvas = FigureCanvas(self._pipeline_fig)
    self._pipeline_canvas.setMinimumHeight(160)
    self._pipeline_canvas.mpl_connect("button_press_event", lambda ev: _pipeline_on_spectrum_click(self, ev))
    self._pipeline_canvas.mpl_connect("scroll_event", lambda ev: _pipeline_on_spectrum_scroll(self, ev))
    # Standard matplotlib zoom-to-rectangle / pan / home toolbar, plus
    # scroll-wheel zoom-at-cursor (wired up above) for quick back-and-forth
    # zooming without needing to pick a tool first.
    self._pipeline_toolbar = AnnoMassNavigationToolbar(self._pipeline_canvas, canvas_box)
    self._pipeline_toolbar.setIconSize(QtCore.QSize(16, 16))
    canvas_box_lay.addWidget(self._pipeline_toolbar)
    canvas_box_lay.addWidget(self._pipeline_canvas)
    spec_splitter.addWidget(canvas_box)

    table_box = QtWidgets.QWidget()
    table_box_lay = QtWidgets.QVBoxLayout(table_box)
    table_box_lay.setContentsMargins(0, 0, 0, 0)
    frag_table_lbl = QtWidgets.QLabel("Predicted fragment peaks (click a row, or a blue peak above, to see its structure)")
    frag_table_lbl.setObjectName("sectionTitle")
    frag_table_lbl.setStyleSheet("font-size: 10px;")
    table_box_lay.addWidget(frag_table_lbl)
    self._pipeline_frag_table = QtWidgets.QTableWidget()
    self._pipeline_frag_table.setColumnCount(2)
    self._pipeline_frag_table.setHorizontalHeaderLabels(["m/z", "Intensity (%)"])
    self._pipeline_frag_table.horizontalHeader().setStretchLastSection(True)
    self._pipeline_frag_table.verticalHeader().setDefaultSectionSize(18)
    self._pipeline_frag_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    self._pipeline_frag_table.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
    self._pipeline_frag_table.setMinimumHeight(60)
    self._pipeline_frag_table.cellClicked.connect(lambda row, col: _pipeline_on_frag_table_click(self, row))
    table_box_lay.addWidget(self._pipeline_frag_table)
    spec_splitter.addWidget(table_box)

    spec_splitter.setStretchFactor(0, 3)
    spec_splitter.setStretchFactor(1, 2)
    spec_box_lay.addWidget(spec_splitter)
    preview_row.addWidget(spec_box)

    struct_box = QtWidgets.QWidget()
    struct_lay = QtWidgets.QVBoxLayout(struct_box)
    struct_lay.setContentsMargins(6, 0, 6, 0)
    struct_lay.setSpacing(4)
    struct_title_row = QtWidgets.QHBoxLayout()
    struct_title_row.setSpacing(4)
    self._pipeline_struct_title = QtWidgets.QLabel("Structure")
    self._pipeline_struct_title.setObjectName("sectionTitle")
    self._pipeline_struct_title.setStyleSheet("font-size: 11px;")
    struct_title_row.addWidget(self._pipeline_struct_title, stretch=1)
    self._pipeline_struct_reset_btn = QtWidgets.QPushButton("Full structure")
    self._pipeline_struct_reset_btn.setToolTip("Return to the full candidate structure")
    self._pipeline_struct_reset_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 1px 6px; }")
    self._pipeline_struct_reset_btn.setVisible(False)
    self._pipeline_struct_reset_btn.clicked.connect(lambda: _pipeline_reset_structure_preview(self))
    struct_title_row.addWidget(self._pipeline_struct_reset_btn)
    self._pipeline_fragment_save_btn = QtWidgets.QPushButton("Save fragment")
    self._pipeline_fragment_save_btn.setToolTip("Save the currently displayed ICEBERG fragment structure")
    self._pipeline_fragment_save_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 1px 6px; }")
    self._pipeline_fragment_save_btn.setVisible(False)
    self._pipeline_fragment_save_btn.clicked.connect(lambda: _pipeline_save_current_fragment(self))
    struct_title_row.addWidget(self._pipeline_fragment_save_btn)
    self._pipeline_struct_preview = _StructurePreviewLabel()
    self._pipeline_struct_preview.setAlignment(Qt.AlignCenter)
    self._pipeline_struct_preview.setMinimumSize(180, 160)
    self._pipeline_struct_preview.setStyleSheet(
        "QLabel { background: white; border: 1px solid #ccc; border-radius: 4px; }"
    )
    struct_lay.addLayout(struct_title_row)
    struct_lay.addWidget(self._pipeline_struct_preview, stretch=1)
    preview_row.addWidget(struct_box)

    cluster_box = QtWidgets.QWidget()
    cluster_lay = QtWidgets.QVBoxLayout(cluster_box)
    cluster_lay.setContentsMargins(0, 0, 0, 0)
    cluster_lay.setSpacing(4)
    cluster_title = QtWidgets.QLabel("DreaMS embedding map")
    cluster_title.setObjectName("sectionTitle")
    cluster_title.setStyleSheet("font-size: 11px;")
    self._pipeline_cluster_fig = Figure(figsize=(5.2, 2.0), dpi=90)
    self._pipeline_cluster_ax = self._pipeline_cluster_fig.add_subplot(111)
    self._pipeline_cluster_canvas = FigureCanvas(self._pipeline_cluster_fig)
    self._pipeline_cluster_canvas.setMinimumHeight(150)
    cluster_lay.addWidget(cluster_title)
    cluster_lay.addWidget(self._pipeline_cluster_canvas, stretch=1)
    preview_row.addWidget(cluster_box)
    preview_row.setStretchFactor(0, 4)
    preview_row.setStretchFactor(1, 2)
    preview_row.setStretchFactor(2, 3)
    spec_vlay.addWidget(preview_row, stretch=1)

    splitter.addWidget(spec_panel)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)
    vlay.addWidget(splitter, stretch=1)
    # ── Bottom row: Top-K filter + save button ───────────────────────────
    btn_row = QtWidgets.QHBoxLayout()

    btn_row.addWidget(QtWidgets.QLabel("Top-K per spectrum:"))
    self._pl_topk_spin = QtWidgets.QSpinBox()
    self._pl_topk_spin.setRange(0, 10000)
    self._pl_topk_spin.setSpecialValueText("All")
    self._pl_topk_spin.setValue(3)
    self._pl_topk_spin.setFixedWidth(58)
    self._pl_topk_spin.setToolTip(
        "Keep exactly the top-K candidates per spectrum in the result table and exports.\n"
        "E.g. 12 spectra with K=1 → 12 rows (rank-1 per spectrum).\n"
        "Choose All to retain every valid candidate; click [Apply] after changing."
    )
    btn_row.addWidget(self._pl_topk_spin)

    btn_apply_topk = QtWidgets.QPushButton("Apply Filter")
    btn_apply_topk.setMinimumWidth(104)
    btn_apply_topk.setToolTip("Filter current results by Top-K (no rerun needed)")
    btn_apply_topk.clicked.connect(lambda: _apply_pipeline_topk(self))
    btn_row.addWidget(btn_apply_topk)


    # Category D (DeepMet CLM, library-external structures) is hidden unless this button is pressed.
    self._pl_show_d_btn = QtWidgets.QPushButton("Show D (CLM)")
    self._pl_show_d_btn.setCheckable(True)
    self._pl_show_d_btn.setChecked(False)
    self._pl_show_d_btn.setMinimumWidth(104)
    self._pl_show_d_btn.setToolTip(
        "Show or hide category-D rows (library-external structures generated by the DeepMet CLM).\n"
        "Off by default. The table, the Top-K filter and the exports all follow what is shown."
    )
    self._pl_show_d_btn.toggled.connect(
        lambda _c: _apply_pipeline_topk(self)
        if getattr(self, "_pipeline_last_df_full", None) is not None else None
    )
    btn_row.addWidget(self._pl_show_d_btn)

    btn_row.addStretch()
    vlay.addLayout(btn_row)

    # Session action buttons (Save Results / Save Project / Load Project /
    # Clear) live directly under the Run button in the activity bar.
    _bar = getattr(self, "vl_activity", None)
    if _bar is not None and hasattr(self, "butt_pipeline"):
        _act_buttons = []

        btn_save = QtWidgets.QPushButton("Save Results (Excel)")
        btn_save.clicked.connect(self._pipeline_save_results)
        _act_buttons.append(btn_save)

        btn_save_figures = QtWidgets.QPushButton("Save Figures")
        btn_save_figures.setToolTip(
            "Choose a folder and export the spectrum/fragment figures for the "
            "current result view:\n- figures/spectra_comparison  experimental vs "
            "predicted mirror plots\n- figures/iceberg_fragments  matched "
            "fragment structure sheets\nIndependent of [Save Results (Excel)]; "
            "no figures folder is auto-created beside the workbook."
        )
        btn_save_figures.clicked.connect(self._pipeline_save_figures)
        _act_buttons.append(btn_save_figures)

        btn_save_project = QtWidgets.QPushButton("Save Project")
        btn_save_project.setToolTip(
            "Save the complete current session (full result table, loaded spectra "
            "snapshot and predicted-spectrum cache) into one .annomass project "
            "file.\nLoad it later with [Load Project] to browse every detail "
            "without rerunning the pipeline."
        )
        btn_save_project.clicked.connect(self._pipeline_save_project)
        _act_buttons.append(btn_save_project)

        btn_load_project = QtWidgets.QPushButton("Load Project")
        btn_load_project.setToolTip(
            "Restore a previously saved .annomass project file: result table, "
            "experimental spectra and predicted-spectrum mirror plots come back "
            "exactly as after the original run (no recomputation)."
        )
        btn_load_project.clicked.connect(self._pipeline_load_project)
        _act_buttons.append(btn_load_project)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.setToolTip(
            "Discard the current result table and cached session state.\n"
            "Save a project file first if you may want these results back."
        )
        btn_clear.clicked.connect(self._pipeline_clear_results)
        _act_buttons.append(btn_clear)

        btn_unload_spectra = QtWidgets.QPushButton("Unload Spectra")
        btn_unload_spectra.setToolTip(
            "Remove the loaded spectrum files from the workspace sidebar.\n"
            "Already computed results keep their own spectra snapshot and "
            "stay browsable."
        )
        btn_unload_spectra.clicked.connect(self._pipeline_unload_spectra)
        _act_buttons.append(btn_unload_spectra)

        _insert_at = _bar.indexOf(self.butt_pipeline) + 1
        for _btn in _act_buttons:
            _btn.setFixedHeight(48)
            _bar.insertWidget(_insert_at, _btn)
            _insert_at += 1

        # Fit the bar to the widest label instead of fixing button widths,
        # so no button text is ever truncated.
        _needed = max(
            [_btn.sizeHint().width() for _btn in _act_buttons]
            + [self.butt_pipeline.sizeHint().width()]
        ) + 22
        _needed = min(_needed, 260)
        if hasattr(self, "activityBar"):
            _needed = max(180, _needed)
            self.activityBar.setMinimumWidth(_needed)
            self.activityBar.setMaximumWidth(max(260, _needed))

    self._pipeline_last_df      = None
    self._pipeline_last_df_full = None   # unfiltered full result
    self._pipeline_spectrums    = []   # snapshot of loaded spectra at pipeline run time
    self._pipeline_mf_pool     = []   # keep threads alive until finished (prevent GC crash)
    self._pipeline_start_time  = None

    # Elapsed-time ticker: fires every second while pipeline is running
    self._pipeline_clock_timer = QtCore.QTimer()
    self._pipeline_clock_timer.setInterval(1000)
    self._pipeline_clock_timer.timeout.connect(self._pipeline_tick_clock)

    # Debounce timer: only plot after user stops moving for 250 ms
    self._pipeline_row_timer = QtCore.QTimer()
    self._pipeline_row_timer.setSingleShot(True)
    self._pipeline_row_timer.timeout.connect(self._pipeline_plot_selected)

    # currentRowChanged fires only when the row index changes, not on every scroll
    self._pipeline_result_table.currentItemChanged.connect(
        lambda *_: self._pipeline_row_timer.start(250)
    )
    self._pipeline_result_table.itemClicked.connect(
        lambda *_: self._pipeline_row_timer.start(50)
    )

    tw.addTab(page, "Results")
    tw.addTab(self._settings_page, "Settings")

    # Keep confirmed DeepMet CLM structures in their own visible result page.
    # These rows are intentionally independent of the library-only Results
    # table.
    clm_page = QtWidgets.QWidget()
    # The CLM page is no longer added as a tab, but the CLM helpers still write into its widgets.
    # Keep a reference: an unparented local QWidget is destroyed (with its children) when this
    # function returns, leaving self._pipeline_denovo_table pointing at a deleted C++ object.
    self._pipeline_clm_page = clm_page
    clm_vlay = QtWidgets.QVBoxLayout(clm_page)
    clm_vlay.setContentsMargins(8, 8, 8, 8)
    clm_note = QtWidgets.QLabel(
        "Confirmed library-external structures generated by DeepMet CLM "
        "(dual Top-1; entropy and cosine ≥ 0.5). These remain visible even "
        "when the same spectrum also has a reportable library candidate."
    )
    clm_note.setWordWrap(True)
    clm_vlay.addWidget(clm_note)
    clm_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    self._pipeline_denovo_table = QtWidgets.QTableWidget()
    self._pipeline_denovo_table.setEditTriggers(
        QtWidgets.QAbstractItemView.NoEditTriggers
    )
    self._pipeline_denovo_table.setSelectionBehavior(
        QtWidgets.QTableView.SelectRows
    )
    self._pipeline_denovo_table.setAlternatingRowColors(True)
    self._pipeline_denovo_table.horizontalHeader().setStretchLastSection(True)
    clm_splitter.addWidget(self._pipeline_denovo_table)

    # CLM rows used to stop at the table.  Give them the same two pieces of
    # visual evidence as library rows: the generated molecular structure and
    # an observed-vs-predicted mirror spectrum.  These are separate widgets
    # because both result pages must retain their current selection/view.
    clm_detail = QtWidgets.QWidget()
    clm_detail_lay = QtWidgets.QHBoxLayout(clm_detail)
    clm_detail_lay.setContentsMargins(2, 2, 2, 2)
    clm_detail_lay.setSpacing(6)

    clm_spec_box = QtWidgets.QWidget()
    clm_spec_lay = QtWidgets.QVBoxLayout(clm_spec_box)
    clm_spec_lay.setContentsMargins(0, 0, 0, 0)
    self._pipeline_clm_spec_lbl = QtWidgets.QLabel(
        "Select a CLM row to compare the observed and predicted spectra."
    )
    self._pipeline_clm_spec_lbl.setObjectName("sectionTitle")
    self._pipeline_clm_spec_lbl.setStyleSheet("font-size: 11px;")
    self._pipeline_clm_fig = Figure(figsize=(6, 2.2), dpi=90)
    self._pipeline_clm_ax = self._pipeline_clm_fig.add_subplot(111)
    self._pipeline_clm_canvas = FigureCanvas(self._pipeline_clm_fig)
    self._pipeline_clm_canvas.setMinimumHeight(180)
    self._pipeline_clm_toolbar = AnnoMassNavigationToolbar(
        self._pipeline_clm_canvas, clm_spec_box
    )
    self._pipeline_clm_toolbar.setIconSize(QtCore.QSize(16, 16))
    clm_spec_lay.addWidget(self._pipeline_clm_spec_lbl)
    clm_spec_lay.addWidget(self._pipeline_clm_toolbar)
    clm_spec_lay.addWidget(self._pipeline_clm_canvas, stretch=1)
    clm_detail_lay.addWidget(clm_spec_box, stretch=4)

    clm_struct_box = QtWidgets.QWidget()
    clm_struct_lay = QtWidgets.QVBoxLayout(clm_struct_box)
    clm_struct_lay.setContentsMargins(4, 0, 4, 0)
    clm_struct_title = QtWidgets.QLabel("CLM compound structure")
    clm_struct_title.setObjectName("sectionTitle")
    clm_struct_title.setStyleSheet("font-size: 11px;")
    self._pipeline_clm_struct_preview = _StructurePreviewLabel()
    self._pipeline_clm_struct_preview.setAlignment(Qt.AlignCenter)
    self._pipeline_clm_struct_preview.setMinimumSize(220, 180)
    self._pipeline_clm_struct_preview.setStyleSheet(
        "QLabel { background: white; border: 1px solid #ccc; border-radius: 4px; }"
    )
    clm_struct_lay.addWidget(clm_struct_title)
    clm_struct_lay.addWidget(self._pipeline_clm_struct_preview, stretch=1)
    clm_detail_lay.addWidget(clm_struct_box, stretch=2)
    clm_splitter.addWidget(clm_detail)
    clm_splitter.setStretchFactor(0, 3)
    clm_splitter.setStretchFactor(1, 2)
    clm_vlay.addWidget(clm_splitter, stretch=1)

    self._pipeline_clm_row_timer = QtCore.QTimer()
    self._pipeline_clm_row_timer.setSingleShot(True)
    self._pipeline_clm_row_timer.timeout.connect(
        lambda: _pipeline_plot_selected_clm(self)
    )
    self._pipeline_denovo_table.currentItemChanged.connect(
        lambda *_: self._pipeline_clm_row_timer.start(150)
    )
    self._pipeline_denovo_table.itemClicked.connect(
        lambda *_: self._pipeline_clm_row_timer.start(30)
    )
    # The dedicated "CLM Structures" tab was removed: category-D rows are part of the
    # main Results table.  The widgets above are still built so the CLM helpers keep working.

    tw.tabBar().setVisible(True)


def run_pipeline(self):
    """Run the full identification pipeline."""
    if not _ensure_dreams_adapter():
        self.ErrorMsg(f"DreaMS is not available. AnnoMASS requires DreaMS.\n{_DREAMS_IMPORT_ERROR or ''}")
        return
    if len(self.spectrums) == 0:
        self.ErrorMsg("Please load spectra first.")
        return
    for attr in ("pp", "pn", "deepmass_positive", "deepmass_negative",
                 "reference_positive", "reference_negative", "default_database"):
        if getattr(self, attr, None) is None:
            self.ErrorMsg("Resources not fully loaded yet. Please wait.")
            return

    # Single reference library / MS1 tolerance -- no per-module picker; this
    # is the one flavonoid reference set the algorithm is validated against.
    ref_hdf5 = str(DEFAULT_REF_HDF5)
    mz_tol   = 10.0

    cho_only = self._pl_cho_chk.isChecked()

    # Spectrum Predictor adduct is auto-detected per spectrum inside the algorithm
    # (core.annomass_algorithm._spectrum_adduct reads the MGF's own
    # adduct/precursor_type field, falling back to an ionmode-based guess).
    # This constant is only the last-resort fallback for a spectrum with
    # neither an adduct field nor a recognizable ionmode.
    adduct = "[M+H]+"

    try:
        nce = float(self._pl_mf_nce.value())
    except Exception:
        nce = 40.0
    instrument = self._pl_mf_instrument.currentText() if hasattr(self, "_pl_mf_instrument") else "Unknown"
    ice_primary = float(self._pl_ice_primary_thr.value())
    ice_secondary = float(self._pl_ice_secondary_thr.value())
    ice_peak_min_rel = float(self._pl_ice_peak_min_pct.value()) / 100.0
    ice_peak_top_k = int(self._pl_ice_peak_top_k.value())
    if ice_primary < ice_secondary:
        self.ErrorMsg("ICE primary threshold must be greater than or equal to ICE secondary threshold.")
        return

    _clm_on = self._pl_deepmet_clm_chk.isChecked() if hasattr(self, "_pl_deepmet_clm_chk") else False
    _modules = "".join(k for k, c in getattr(self, "_pl_mod_chks", {}).items() if c.isChecked()) + ("D" if _clm_on else "")
    if not _modules:
        self.ErrorMsg("No module is selected. Tick at least one of A / B / C / D in Settings → Run modules.")
        return

    spectrums = list(self.spectrums["spectrum"])
    self._apply_settings_to_spectra(spectrums)
    # Save snapshot for row-click spectrum viewer
    if hasattr(self, "_pipeline_spectrums"):
        self._pipeline_spectrums = spectrums
    # Save Spectrum Predictor params for on-demand prediction in row-click
    self._pipeline_mf_adduct     = adduct
    self._pipeline_mf_nce        = nce
    self._pipeline_mf_instrument = instrument

    self.butt_pipeline.setEnabled(False)
    self.progressBar.setRange(0, 0)
    self.progressBar.setFormat("Running...")

    # Start elapsed-time clock
    import time as _time
    self._pipeline_start_time = _time.time()
    if hasattr(self, "_pipeline_clock_timer"):
        self._pipeline_clock_timer.start()

    if hasattr(self, "_pipeline_status_lbl"):
        self._pipeline_status_lbl.setText("Started...")

    # The worker returns every ranked candidate so the result-level Top-K can
    # be changed without rerunning.  The selected K is applied consistently
    # to the table, Agent context and exports in _apply_pipeline_topk().
    _topn_per_spec = 0
    _enable_annosm = self._pl_annosm_chk.isChecked() if hasattr(self, "_pl_annosm_chk") else False
    _enable_abcort = self._pl_abcort_chk.isChecked() if hasattr(self, "_pl_abcort_chk") else False
    _abcort_checkpoint = (
        self._pl_abcort_checkpoint.currentData() if hasattr(self, "_pl_abcort_checkpoint") else None
    )
    _enable_deepmet_clm = self._pl_deepmet_clm_chk.isChecked() if hasattr(self, "_pl_deepmet_clm_chk") else False
    self.Thread_Pipeline = PipelineWorker(
        spectrums            = spectrums,
        pp                   = self.pp,
        pn                   = self.pn,
        deepmass_pos         = self.deepmass_positive,
        deepmass_neg         = self.deepmass_negative,
        ref_positive         = self.reference_positive,
        ref_negative         = self.reference_negative,
        default_database     = self.default_database,
        ref_hdf5             = ref_hdf5,
        mz_tol_ppm           = mz_tol,
        cho_only             = cho_only,
        mf_adduct            = adduct,
        mf_nce               = nce,
        mf_instrument        = instrument,
        iceberg_peak_min_rel_intensity = ice_peak_min_rel,
        iceberg_peak_top_k   = ice_peak_top_k,
        top_n_per_spectrum   = _topn_per_spec,
        dedup_per_spectrum   = True,
        enable_annosm_rerank = _enable_annosm,
        enable_abcort_predict = _enable_abcort,
        abcort_checkpoint_path = _abcort_checkpoint,
        use_deepmet_clm      = _enable_deepmet_clm,
        clm_primary_min      = ice_primary,
        clm_secondary_min    = ice_secondary,
        enabled_levels       = _modules,
    )
    self.Thread_Pipeline.progress.connect(self._on_pipeline_progress)
    self.Thread_Pipeline.finished_with_result.connect(self._on_pipeline_done)
    self.Thread_Pipeline.failed.connect(self._on_pipeline_failed)
    self.Thread_Pipeline.start()


def _update_deepmet_clm_panel(self, denovo_df):
    """Populate the separate DeepMet CLM table when present."""
    table = getattr(self, "_pipeline_denovo_table", None)
    if table is None:
        return
    if denovo_df is None or len(denovo_df) == 0:
        self._pipeline_clm_display_df = None
        table.setRowCount(0)
        table.setColumnCount(0)
        if hasattr(self, "_pipeline_clm_struct_preview"):
            self._pipeline_clm_struct_preview.show_structure("")
        if hasattr(self, "_pipeline_clm_ax"):
            self._pipeline_clm_ax.cla()
            self._pipeline_clm_canvas.draw_idle()
        return

    denovo_df = _pipeline_normalize_result_columns(denovo_df).reset_index(drop=True)
    self._pipeline_clm_display_df = denovo_df
    cols = [c for c in ["deepmet_rank", "spectrum_title", "spectrum_index",
                        "compound_name", "formula",
                        "formula_in_structure_library", "structure_library_formula_count", "smiles",
                        "inchikey", "support_spectra_count", "iceberg_entropy_similarity",
                        "iceberg_cosine_similarity", "iceberg_dual_threshold_pass",
                        "deepmet_sampling_frequency", "candidate_source"]
            if c in denovo_df.columns]
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels(cols)
    table.setRowCount(len(denovo_df))
    for r, (_, row) in enumerate(denovo_df[cols].iterrows()):
        for c, col in enumerate(cols):
            val = row[col]
            text = f"{val:.4f}" if isinstance(val, float) else str(val)
            table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
    table.resizeColumnsToContents()
    if len(denovo_df):
        table.setCurrentCell(0, 0)
        if hasattr(self, "_pipeline_clm_row_timer"):
            self._pipeline_clm_row_timer.start(30)


def _pipeline_plot_selected_clm_impl(self):
    """Draw the selected CLM structure and observed/predicted mirror plot.

    The dedicated CLM result is aggregated by structure.  Its representative
    row points at the supporting spectrum with the strongest ICEBERG evidence
    and carries ``_mf_cache_row`` so the exact prediction used for validation
    can be restored from a saved project.
    """
    df = getattr(self, "_pipeline_clm_display_df", None)
    table = getattr(self, "_pipeline_denovo_table", None)
    if df is None or table is None or not len(df):
        return
    row_idx = table.currentRow()
    if row_idx < 0 or row_idx >= len(df):
        return

    record = df.iloc[row_idx]
    smiles = str(record.get("smiles", "") or "")
    cname = str(record.get("compound_name", "") or "")
    struct = getattr(self, "_pipeline_clm_struct_preview", None)
    if struct is not None:
        struct.setToolTip(cname or smiles)
        struct.show_structure(smiles)

    try:
        sp_idx = int(record.get("spectrum_index", -1))
    except (TypeError, ValueError):
        sp_idx = -1
    sp = None
    spectrums = getattr(self, "_pipeline_spectrums", [])
    if 0 <= sp_idx < len(spectrums):
        sp = spectrums[sp_idx]
    if sp is None and hasattr(self, "spectrums") and len(self.spectrums) > 0:
        if 0 <= sp_idx < len(self.spectrums):
            sp = self.spectrums["spectrum"].iloc[sp_idx]

    cached = None
    raw_cache_key = record.get("_mf_cache_row", None)
    try:
        if raw_cache_key is not None and not pd.isna(raw_cache_key):
            cached = getattr(self, "_pipeline_mf_cache", {}).get(int(raw_cache_key))
    except (TypeError, ValueError):
        cached = None

    ax = self._pipeline_clm_ax
    ax.cla()
    all_mz = []
    obs_plotted = False
    pred_plotted = False
    if sp is not None:
        try:
            obs_mz = np.asarray(sp.peaks.mz, dtype=float)
            obs_int = np.asarray(sp.peaks.intensities, dtype=float)
            if len(obs_int) and float(obs_int.max()) > 0:
                obs_int = obs_int / obs_int.max()
            ax.vlines(obs_mz, 0, obs_int, colors="#e74c3c", lw=1.2,
                      label="Observed ↑")
            all_mz.extend(obs_mz.tolist())
            obs_plotted = True
        except Exception:
            pass
    if cached is not None:
        pred_mz = np.asarray(cached.get("mz", []), dtype=float)
        pred_int = np.asarray(cached.get("intensity", []), dtype=float)
        if len(pred_int) and float(pred_int.max()) > 0:
            top_n = int(self._pl_top_frags.value()) if hasattr(self, "_pl_top_frags") else 100
            keep = np.argsort(pred_int)[-min(top_n, len(pred_int)):]
            keep = keep[np.argsort(pred_mz[keep])]
            shown_mz = pred_mz[keep]
            shown_int = pred_int[keep] / pred_int.max()
            ax.vlines(shown_mz, 0, -shown_int, colors="#2980b9", lw=1.2,
                      label="Predicted spectrum ↓")
            all_mz.extend(shown_mz.tolist())
            pred_plotted = True

    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-1.15, 1.15)
    if all_mz:
        lo, hi = min(all_mz), max(all_mz)
        pad = max(10.0, (hi - lo) * 0.05)
        ax.set_xlim(max(0.0, lo - pad), hi + pad)
    ax.set_xlabel("m/z", fontsize=8)
    ax.set_ylabel("Relative Intensity", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"], fontsize=7)
    entropy = record.get("iceberg_entropy_similarity", None)
    cosine = record.get("iceberg_cosine_similarity", None)
    title_parts = [cname] if cname and cname not in ("nan", "None") else []
    try:
        title_parts.append(f"Entropy={float(entropy):.3f}")
    except (TypeError, ValueError):
        pass
    try:
        title_parts.append(f"Cosine={float(cosine):.3f}")
    except (TypeError, ValueError):
        pass
    ax.set_title("  |  ".join(title_parts) or "CLM spectrum", fontsize=8)
    if obs_plotted or pred_plotted:
        ax.legend(fontsize=7, loc="upper right")
    if hasattr(self, "_pipeline_clm_toolbar"):
        self._pipeline_clm_toolbar.update()
    self._pipeline_clm_canvas.draw_idle()

    details = []
    if obs_plotted:
        details.append("red=Observed ↑")
    if pred_plotted:
        details.append("blue=Predicted spectrum ↓")
    elif cached is None:
        details.append("prediction cache unavailable; rerun to regenerate it")
    self._pipeline_clm_spec_lbl.setText(
        f"CLM row {row_idx + 1}: {cname or 'unknown'}  |  " + "  |  ".join(details)
    )


def _pipeline_plot_selected_clm(self):
    """Keep a malformed/restored CLM row from terminating the Qt process."""
    try:
        return _pipeline_plot_selected_clm_impl(self)
    except Exception:
        import traceback
        traceback.print_exc()
        label = getattr(self, "_pipeline_clm_spec_lbl", None)
        if label is not None:
            label.setText(
                "CLM detail view error (the result table is still available): "
                + traceback.format_exc().strip().splitlines()[-1]
            )



def _on_pipeline_progress(self, msg):
    self.progressBar.setFormat(msg)
    if hasattr(self, "_pipeline_status_lbl"):
        self._pipeline_status_lbl.setText(msg)


_LEVEL_TOOLTIP = (
    "Staged annotation level. Each spectrum stops at the first level with a passing "
    "candidate and only that level is shown: A = DreaMS library match (DreaMS similarity "
    "+ library-spectrum cosine); B = structure candidate or "
    "DreaMS hit that failed A, with library evidence (structure score + library cosine + "
    "carried DreaMS score); C = any candidate, including those that failed B (carried B "
    "score + ICEBERG predicted spectrum); D = DeepMet CLM "
    "library-external structure (ICEBERG entropy + cosine dual Top-1). MSI levels: A/B = 2, "
    "C = 3, D = 4. Spectra that pass no level are not annotated."
)


def _pipeline_normalize_result_columns(df):
    """Return a copy with pipeline/DeepMet CLM column names normalized."""
    out = df.copy()
    rename_map = {
        "SMILES": "smiles",
        "CanonicalSMILES": "smiles",
        "MolecularFormula": "formula",
        "Formula": "formula",
        "InChIKey": "inchikey",
        "sampling_frequency": "deepmet_sampling_frequency",
        "best_mf_score": "best_spectrum_predictor_score",
        "mf_score": "spectrum_predictor_score",
        "mf_adduct": "spectrum_predictor_adduct",
    }
    applied = {}
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            applied[old] = new
    if applied:
        out = out.rename(columns=applied)
    return out


def _on_pipeline_done(self, payload):
    self.butt_pipeline.setEnabled(True)
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(100)

    # Stop elapsed clock and show final time
    if hasattr(self, "_pipeline_clock_timer"):
        self._pipeline_clock_timer.stop()
    if hasattr(self, "_pipeline_start_time") and self._pipeline_start_time is not None:
        import time as _time
        total_s = int(_time.time() - self._pipeline_start_time)
        m, s = divmod(total_s, 60)
        self._pipeline_start_time = None
        _final_time_str = f"Total time: {m:02d}:{s:02d}"
    else:
        _final_time_str = ""

    # payload is (DataFrame, mf_cache, denovo_df); older 2-tuple / bare
    # DataFrame forms kept for backward compat.
    denovo_df = None
    if isinstance(payload, tuple):
        if len(payload) >= 3:
            df, mf_cache, denovo_df = payload[0], payload[1], payload[2]
        elif len(payload) == 2:
            df, mf_cache = payload
        else:
            df, mf_cache = payload[0], {}
    else:
        df, mf_cache = payload, {}

    if df is not None:
        df = _pipeline_normalize_result_columns(df)
    if denovo_df is not None:
        denovo_df = _pipeline_normalize_result_columns(denovo_df)

    self._pipeline_last_df_full = df       # full result, never filtered
    self._pipeline_mf_cache     = mf_cache
    self._pipeline_last_denovo_df = denovo_df
    _update_deepmet_clm_panel(self, denovo_df)

    if df is None or (hasattr(df, "empty") and df.empty):
        msg = f"Done – no candidates found.  {_final_time_str}"
        self.progressBar.setFormat(msg)
        if hasattr(self, "_pipeline_status_lbl"):
            self._pipeline_status_lbl.setText(msg)
        return

    # Apply Top-K per spectrum filter, then display.
    _apply_pipeline_topk(self, _final_time_str=_final_time_str)
    return


def _apply_pipeline_topk(self, _final_time_str=""):
    """Apply optional ICE display filtering and exact per-spectrum Top-K."""
    df = getattr(self, "_pipeline_last_df_full", None)
    if df is None or (hasattr(df, "empty") and df.empty):
        return

    k = int(self._pl_topk_spin.value()) if hasattr(self, "_pl_topk_spin") else 3
    df = _pipeline_normalize_result_columns(df)
    if "_mf_cache_row" not in df.columns:
        df["_mf_cache_row"] = np.arange(len(df), dtype=int)

    show_d = getattr(self, "_pl_show_d_btn", None)
    d_hidden = show_d is not None and not show_d.isChecked()
    n_d_hidden = 0
    if d_hidden:
        _d_mask = _is_category_d(df)
        n_d_hidden = int(_d_mask.sum())
        df = df[~_d_mask].copy()

    if "spectrum_index" in df.columns:
        _rank_col = "spectrum_rank" if "spectrum_rank" in df.columns else "rank"
        # Top-K per spectrum.  A spectrum only carries candidates of the one
        # level it was annotated at (A, B, C or D), so one quota fits all.
        df_filtered = _select_topk_per_spectrum(
            df, k, order_col=_rank_col, ascending=True
        )
    else:
        _ordered = df.sort_values("rank", ascending=True).copy()
        df_filtered = (_ordered if k == 0 else _ordered.head(k)).reset_index(drop=True)

    # "rank" is kept as the original overall-quality rank (not renumbered),
    # since rows are now displayed grouped by spectrum rather than sorted by
    # rank -- renumbering 1..N after that reorder would turn "rank" into a
    # meaningless row-position counter instead of true relative quality.

    df_filtered = _pipeline_normalize_result_columns(df_filtered)
    self._pipeline_last_df = df_filtered

    n_spec = df_filtered["spectrum_index"].nunique() if "spectrum_index" in df_filtered.columns else "?"
    if hasattr(self, "_pipeline_status_lbl"):
        status = (
            f"Done: {len(getattr(self, '_pipeline_last_df_full', df))} candidates, "
            f"{f'D hidden ({n_d_hidden} rows), ' if d_hidden and n_d_hidden else ''}"
            f"{'All candidates' if k == 0 else f'Top-{k} per spectrum'} → "
            f"{len(df_filtered)} rows / {n_spec} spectra"
        )
        if _final_time_str:
            status += f"  |  {_final_time_str}"
        if len(df_filtered):
            status += (
                f"\nTop compound: {df_filtered.iloc[0]['compound_name']}  "
                f"({df_filtered.iloc[0].get('confidence_level', '')})"
                "\nLevels: A DreaMS -> B structure + library spectrum -> C structure + "
                "predicted spectrum -> D DeepMet CLM; each spectrum stops at its first passing level."
            )
        self._pipeline_status_lbl.setText(status)
    _topk_label = "All" if k == 0 else f"Top-{k}/spectrum"
    self.progressBar.setFormat(f"Done: {len(df_filtered)} rows ({_topk_label})")

    # Fill results table (batch fill for speed)
    t = self._pipeline_result_table
    t.setUpdatesEnabled(False)          # suppress repaints during fill
    t.clearContents()
    t.setRowCount(0)

    cols = ["spectrum_title", "spectrum_index", "spectrum_rank",
            "confidence_level", "msi_level", "level_score",
            "compound_name", "formula", "smiles", "inchikey",
            "support_spectra_count", "support_precursor_mz",
            "DreaMS_similarity", "library_cosine", "deepmass_score",
            "spectrum_predictor_score",
            "iceberg_entropy_similarity", "iceberg_cosine_similarity",
            "predicted_rt_min",
            "flavonoid_class", "query_adduct",
            "ms1_error_ppm"]
    cols = [c for c in cols if c in df_filtered.columns]
    # Column headers describe what each score/state measures, not which
    # internal engine implements it (DreaMS/DeepMASS/AnnoSM/Spectrum
    # Predictor/...) -- same convention as the parameter bar above. Engine
    # names still show up in the header tooltips for anyone who wants the
    # technical detail.
    _display_labels = {
        "spectrum_title": "Spectrum",
        "spectrum_rank": "Rank in spectrum",
        "rank": "Overall rank",
        "DreaMS_similarity": "Library-match similarity",
        "library_cosine": "Library-spectrum cosine",
        "deepmass_score": "Structure score",
        "spectrum_predictor_score": "Spectrum-prediction score",
        "confidence_level": "Level",
        "msi_level": "MSI level",
        "level_score": "Level score",
        "deepmet_sampling_frequency": "DeepMet CLM sampling frequency",
        "support_spectra_count": "Support spectra",
        "support_precursor_mz": "Support precursor m/z",
        "ms1_error_ppm": "MS1 error (ppm)",
        "ms2_peaks": "MS2 peaks",
        "annosm_state": "Substituent-check state",
        "annosm_score": "Substituent-check score",
        "annosm_promoted": "Substituent-check promoted",
        "predicted_rt_min": "Predicted RT (min)",
    }
    t.setColumnCount(len(cols))
    t.setHorizontalHeaderLabels([_display_labels.get(c, c) for c in cols])
    t.setRowCount(len(df_filtered))
    for ci, col in enumerate(cols):
        header_item = t.horizontalHeaderItem(ci)
        if header_item is not None and col == "confidence_level":
            header_item.setToolTip(_LEVEL_TOOLTIP)
        elif header_item is not None and col == "msi_level":
            header_item.setToolTip(
                "Metabolomics Standards Initiative confidence (Schymanski et al. 2014): "
                "A and B = Level 2 (experimental reference spectrum), C = Level 3 "
                "(in-silico prediction + structure prior), D = Level 4 (generated "
                "structure). Level 1 needs an in-house authentic standard."
            )
        elif header_item is not None and col == "level_score":
            header_item.setToolTip(
                "Weighted score of the spectrum's level (A: DreaMS + library cosine; "
                "B: structure score + library cosine; C: structure score + "
                "spectrum-prediction score). Candidates are ordered by it."
            )
        elif header_item is not None and col == "DreaMS_similarity":
            header_item.setToolTip("Level A component (engine: DreaMS embedding search).")
        elif header_item is not None and col == "library_cosine":
            header_item.setToolTip("Levels A/B component: modified cosine between the query and the "
                                   "matched experimental library spectrum.")
        elif header_item is not None and col == "deepmass_score":
            header_item.setToolTip("Levels B/C component: structure-retrieval score (engine: DeepMASS).")
        elif header_item is not None and col == "spectrum_predictor_score":
            header_item.setToolTip("Level C component (engine: ICEBERG/ms-pred fragment simulation).")
        elif header_item is not None and col == "deepmet_sampling_frequency":
            header_item.setToolTip("Frequency of this structure in the DeepMet CLM sampling output.")
        elif header_item is not None and col in ("annosm_state", "annosm_score", "annosm_promoted"):
            header_item.setToolTip(
                "Extended-library gated rerank for isomer substituent position "
                "(engine: AnnoSM; optional, toggle in the parameter bar above). "
                "'Substituent-check promoted' = this candidate was promoted "
                "over the original rank-1 of the same level for its "
                "spectrum because of stronger CFIL fragment evidence for a "
                "different substitution mode, within a small structure-score gap."
            )
        elif header_item is not None and col == "predicted_rt_min":
            header_item.setToolTip(
                "GNN-predicted retention time (engine: ABCoRT; optional, toggle in the "
                "parameter bar above). Informational only -- not used to rank "
                "or filter candidates."
            )

    # Fixed column widths – much faster than resizeColumnsToContents()
    _col_w = {"compound_name": 220, "rank": 80, "spectrum_rank": 100, "confidence_level": 70,
              "msi_level": 80, "level_score": 90, "library_cosine": 110,
              "candidate_source": 120, "deepmet_sampling_frequency": 130,
              "support_spectra_count": 90, "spectrum_ids": 150,
              "support_precursor_mz": 180,
              "best_DreaMS_similarity": 120, "best_deepmass_score": 120,
              "best_public_library_similarity": 130,
              "best_spectrum_predictor_score": 100,
              "annosm_state": 130, "annosm_score": 90, "annosm_promoted": 100,
              "predicted_rt_min": 120,
              "spectrum_title": 220, "precursor_mz": 90,
              "query_adduct": 90, "ms1_error_ppm": 100, "ms2_peaks": 70,
              "flavonoid_class": 120, "formula": 90,
              "DreaMS_similarity": 110, "DreaMS_match": 100,
              "deepmass_score": 110, "spectrum_predictor_score": 100, "spectrum_predictor_adduct": 95,
              "smiles": 200,
              "inchikey": 180, "support_spectrum_indices": 160,
              "support_spectrum_titles": 240, "source": 100}
    for ci, col in enumerate(cols):
        t.setColumnWidth(ci, _col_w.get(col, 100))
    t.horizontalHeader().setStretchLastSection(True)

    _CONFIDENCE_COLOR = {
        "A": QtGui.QColor(0, 155, 70, 175),
        "B": QtGui.QColor(90, 170, 70, 175),
        "C": QtGui.QColor(230, 150, 60, 175),
        "D": QtGui.QColor(155, 115, 215, 165),
    }
    for r, (_, row) in enumerate(df_filtered.iterrows()):
        for c, col in enumerate(cols):
            val  = row.get(col, "")
            text = f"{val:.4f}" if isinstance(val, float) and val == val else str(val)
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            if col == "confidence_level":
                item.setToolTip(_LEVEL_TOOLTIP)
                color = _CONFIDENCE_COLOR.get(str(val))
                if color is not None:
                    item.setBackground(color)
            t.setItem(r, c, item)

    t.setUpdatesEnabled(True)
    if len(df_filtered) > 0:
        t.setCurrentCell(0, 0)
        self._pipeline_row_timer.start(50)

    # Switch to Pipeline tab
    tw = self.tabWidget_results
    for i in range(tw.count()):
        if tw.tabText(i) == "Results":
            tw.setCurrentIndex(i)
            break


def _on_pipeline_failed(self, msg):
    self.butt_pipeline.setEnabled(True)
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(0)
    if hasattr(self, "_pipeline_clock_timer"):
        self._pipeline_clock_timer.stop()
    self._pipeline_start_time = None
    if "Stopped by user" in str(msg):
        self.progressBar.setFormat("Stopped")
        if hasattr(self, "_pipeline_status_lbl"):
            self._pipeline_status_lbl.setText("Stopped by user.")
        return
    self.progressBar.setFormat("Failed")
    if hasattr(self, "_pipeline_status_lbl"):
        self._pipeline_status_lbl.setText(f"Failed:\n{msg[:300]}")
    self.ErrorMsg(f"Failed:\n{msg[:800]}")


def _pipeline_tick_clock(self):
    """Called every second while pipeline is running — updates elapsed display."""
    if not hasattr(self, "_pipeline_start_time") or self._pipeline_start_time is None:
        return
    import time as _time
    elapsed = int(_time.time() - self._pipeline_start_time)
    m, s = divmod(elapsed, 60)
    cur = self._pipeline_status_lbl.text()
    # Replace or append the elapsed line
    lines = [l for l in cur.splitlines() if not l.startswith("⏱")]
    lines.append(f"⏱ Elapsed: {m:02d}:{s:02d}")
    self._pipeline_status_lbl.setText("\n".join(lines))


def _pipeline_update_cluster_map(self, selected_spectrum_index=None):
    """Draw DreaMS embedding projection for the spectra used by pipeline."""
    canvas = getattr(self, "_pipeline_cluster_canvas", None)
    ax = getattr(self, "_pipeline_cluster_ax", None)
    df = getattr(self, "_pipeline_last_df", None)
    if canvas is None or ax is None:
        return
    ax.cla()
    if df is None or getattr(df, "empty", True):
        ax.text(0.5, 0.5, "No embedding map", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        canvas.draw()
        return
    needed = {"dreams_umap_x", "dreams_umap_y", "spectrum_index"}
    if not needed <= set(df.columns):
        ax.text(0.5, 0.5, "No DreaMS coordinates", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        canvas.draw()
        return
    plot_df = df.dropna(subset=["dreams_umap_x", "dreams_umap_y"]).copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No DreaMS coordinates", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        canvas.draw()
        return
    plot_df = plot_df.sort_values("rank", ascending=True) if "rank" in plot_df.columns else plot_df
    plot_df = plot_df.drop_duplicates(subset=["spectrum_index"], keep="first")
    x = pd.to_numeric(plot_df["dreams_umap_x"], errors="coerce")
    y = pd.to_numeric(plot_df["dreams_umap_y"], errors="coerce")
    ax.set_facecolor("#fbfbfb")
    ax.grid(True, color="#d8d8d8", linewidth=0.65, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#c8c8c8")
        spine.set_linewidth(0.8)
    def _major_class(v):
        s = str(v or "").lower()
        if "flavonol" in s:
            return "Flavonol"
        if "isoflavone" in s or "isoflavan" in s:
            return "Isoflavonoid"
        if "flavanone" in s or "dihydroflavonol" in s:
            return "Flavanone / Dihydroflavonol"
        if "flavan3ol" in s or "flavan" in s:
            return "Flavan / Flavanol"
        if "chalcone" in s:
            return "Chalcone"
        if "aurone" in s:
            return "Aurone"
        if "anthocyanidin" in s:
            return "Anthocyanidin"
        if "pterocarpan" in s:
            return "Pterocarpan"
        if "flavone" in s:
            return "Flavone"
        return "Other flavonoids"
    plot_df["_major_flav_class"] = plot_df.get("flavonoid_class", "").apply(_major_class)
    palette = {
        "Flavone": "#2f6fdd",
        "Flavonol": "#32a89a",
        "Isoflavonoid": "#f19cab",
        "Flavanone / Dihydroflavonol": "#a98ad6",
        "Flavan / Flavanol": "#f2c45d",
        "Chalcone": "#e68a3a",
        "Aurone": "#7aa64b",
        "Anthocyanidin": "#cf5aa8",
        "Pterocarpan": "#8c6d5c",
        "Other flavonoids": "#b9b9c8",
    }
    for cls, sub in plot_df.groupby("_major_flav_class", sort=False):
        ax.scatter(
            pd.to_numeric(sub["dreams_umap_x"], errors="coerce"),
            pd.to_numeric(sub["dreams_umap_y"], errors="coerce"),
            s=18, alpha=0.82, color=palette.get(cls, "#b9b9c8"),
            edgecolors="none", linewidths=0, label=cls,
        )
    title = f"DreaMS UMAP by flavonoid class  n={len(plot_df)}"
    if selected_spectrum_index is not None:
        try:
            sid = int(selected_spectrum_index)
            sel = plot_df[plot_df["spectrum_index"].astype(int) == sid]
            if not sel.empty:
                sx = float(sel.iloc[0]["dreams_umap_x"])
                sy = float(sel.iloc[0]["dreams_umap_y"])
                cls = sel.iloc[0].get("_major_flav_class", "Other flavonoids")
                title = f"DreaMS UMAP  {cls}"
                ax.scatter([sx], [sy], s=54, color="#e53935", edgecolors="none", linewidths=0, zorder=6)
        except Exception:
            pass
    if len(x) > 0 and len(y) > 0:
        xpad = max((float(x.max()) - float(x.min())) * 0.08, 0.5)
        ypad = max((float(y.max()) - float(y.min())) * 0.08, 0.5)
        ax.set_xlim(float(x.min()) - xpad, float(x.max()) + xpad)
        ax.set_ylim(float(y.min()) - ypad, float(y.max()) + ypad)
    ax.set_xlabel("UMAP-1", fontsize=7, color="#555")
    ax.set_ylabel("UMAP-2", fontsize=7, color="#555")
    ax.tick_params(labelsize=6, colors="#666", length=2)
    ax.set_title(title, fontsize=8, fontweight="bold", color="#222")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles[:8], labels[:8],
            loc="upper right", fontsize=6, frameon=True,
            facecolor="white", edgecolor="#cccccc", framealpha=0.9,
            markerscale=1.2, borderpad=0.35, labelspacing=0.3,
        )
    try:
        self._pipeline_cluster_fig.tight_layout(pad=0.5)
    except Exception:
        pass
    canvas.draw()


def _parse_library_peaks(text):
    """"mz:intensity;mz:intensity" -> (mz array, intensity array scaled to 1)."""
    mz, it = [], []
    if isinstance(text, str) and text:
        for part in text.split(";"):
            try:
                a, b = part.split(":")
                mz.append(float(a))
                it.append(float(b))
            except ValueError:
                continue
    mz = np.asarray(mz, dtype=float)
    it = np.asarray(it, dtype=float)
    if len(it) and it.max() > 0:
        it = it / it.max()
    return mz, it


def _pipeline_plot_selected_impl(self):
    """Debounced: draw observed + cached Spectrum Predictor prediction instantly."""
    if not hasattr(self, "_pipeline_last_df") or self._pipeline_last_df is None:
        return
    df      = self._pipeline_last_df
    row_idx = self._pipeline_result_table.currentRow()
    if row_idx < 0 or row_idx >= len(df):
        return

    record = df.iloc[row_idx]
    sp_idx = int(record.get("spectrum_index", -1))
    cname  = str(record.get("compound_name", "") or "")
    smiles = str(record.get("smiles", "") or "")
    dm_sim = record.get("DreaMS_similarity", 0.0)
    spectrum_predictor_score = record.get("spectrum_predictor_score", record.get("mf_score", None))
    self._pipeline_update_structure_preview(smiles, cname)
    self._pipeline_update_cluster_map(sp_idx)

    # Fetch observed spectrum
    sp = None
    spectrums = getattr(self, "_pipeline_spectrums", [])
    if 0 <= sp_idx < len(spectrums):
        sp = spectrums[sp_idx]
    if sp is None and hasattr(self, "spectrums") and len(self.spectrums) > 0:
        if 0 <= sp_idx < len(self.spectrums):
            sp = self.spectrums["spectrum"].iloc[sp_idx]

    # Fetch cached Spectrum Predictor prediction (computed during pipeline run):
    # {"mz", "intensity", "frag_ids", "frag_bits", "canonical_smiles"} or None
    mf_cache   = getattr(self, "_pipeline_mf_cache", {})
    try:
        cache_row = int(record.get("_mf_cache_row", row_idx))
    except Exception:
        cache_row = row_idx
    cached     = mf_cache.get(cache_row)
    # Cleared/refreshed below so the click handler always reflects what's
    # actually drawn for the currently selected row.
    self._pipeline_click_ctx = None
    self._pipeline_obs_click_ctx = None
    self._pipeline_click_annotation = None  # ax.cla() below already destroys it

    ax = self._pipeline_ax
    ax.cla()

    # ── Observed spectrum (red, upward) ──────────────────────────────────
    all_mz = []
    obs_plotted = False
    if sp is not None:
        try:
            obs_mz  = np.asarray(sp.peaks.mz, float)
            obs_int = np.asarray(sp.peaks.intensities, float)
            if len(obs_int) > 0 and obs_int.max() > 0:
                obs_int = obs_int / obs_int.max()
            ax.vlines(obs_mz, 0, obs_int, colors="#e74c3c", lw=1.2, label="Observed ↑")
            all_mz.extend(obs_mz.tolist())
            obs_plotted = True
            # Experimental peaks get the same click interaction as predicted
            # ones: click a red peak to label its m/z.
            self._pipeline_obs_click_ctx = {"obs_mz": obs_mz, "obs_int": obs_int}
        except Exception:
            pass

    # ── Predicted spectrum (blue, downward) ────────────────────────────────
    pred_plotted = False
    if cached is not None:
        pred_int_full  = np.asarray(cached.get("intensity", []), float)
        pred_mz_full   = np.asarray(cached.get("mz", []), float)
        frag_ids_full  = np.asarray(cached.get("frag_ids", []), np.int64)
        if len(pred_int_full) > 0 and pred_int_full.max() > 0:
            top_n = int(self._pl_top_frags.value()) if hasattr(self, "_pl_top_frags") else 20
            top_n = min(top_n, len(pred_int_full))
            keep_idx = np.argsort(pred_int_full)[-top_n:]

            # Always show the precursor/molecular-ion peak, even when it
            # doesn't rank in the top-N by intensity -- at higher collision
            # energies most of the precursor population fragments away, so
            # a low (but real) precursor peak getting silently cut from the
            # top-N view read as "the precursor is just missing."
            prec_mz = float(sp.get("precursor_mz") or 0.0) if sp is not None else 0.0
            if prec_mz > 0:
                nearest = int(np.argmin(np.abs(pred_mz_full - prec_mz)))
                if abs(pred_mz_full[nearest] - prec_mz) <= 0.02 and nearest not in keep_idx:
                    keep_idx = np.append(keep_idx, nearest)

            keep_idx = keep_idx[np.argsort(pred_mz_full[keep_idx])]  # left-to-right

            pred_arr    = pred_int_full[keep_idx] / pred_int_full.max()
            pred_mz_arr = pred_mz_full[keep_idx]
            frag_ids_arr = (
                frag_ids_full[keep_idx] if len(frag_ids_full) == len(pred_int_full)
                else np.array([], dtype=np.int64)
            )

            ax.vlines(pred_mz_arr, 0, -pred_arr, colors="#2980b9", lw=1.2,
                      label="Predicted spectrum ↓")
            all_mz.extend(pred_mz_arr.tolist())
            pred_plotted = True
            self._pipeline_click_ctx = {
                "row_idx": row_idx,
                "pred_mz": pred_mz_arr,
                "pred_int": pred_arr,
                "frag_ids": frag_ids_arr,
                "canonical_smiles": cached.get("canonical_smiles", ""),
            }
            _pipeline_fill_frag_table(self, pred_mz_arr, pred_arr)

    if not pred_plotted and hasattr(self, "_pipeline_frag_table"):
        self._pipeline_frag_table.setRowCount(0)

    # ── Library spectrum (blue, downward) for level A / B rows ──────────────
    # A and B rows have no predicted spectrum; the best-matching experimental
    # library spectrum is drawn in its place.
    lib_plotted = False
    if not pred_plotted:
        lib_mz, lib_int = _parse_library_peaks(record.get("library_peaks", ""))
        if len(lib_mz):
            ax.vlines(lib_mz, 0, -lib_int, colors="#2980b9", lw=1.2,
                      label="Library spectrum ↓ (best match)")
            all_mz.extend(lib_mz.tolist())
            lib_plotted = True

    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-1.15, 1.15)
    if all_mz:
        lo, hi = min(all_mz), max(all_mz)
        pad = max(10, (hi - lo) * 0.05)
        ax.set_xlim(max(0, lo - pad), hi + pad)
    ax.set_xlabel("m/z", fontsize=8)
    ax.set_ylabel("Relative Intensity", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"], fontsize=7)

    title_parts = []
    if cname and cname not in ("nan", "None"):
        title_parts.append(cname)
    if dm_sim:
        title_parts.append(f"Lib={float(dm_sim):.3f}")
    if spectrum_predictor_score and str(spectrum_predictor_score) not in ("None", "nan"):
        title_parts.append(f"Pred={float(spectrum_predictor_score):.3f}")
    ax.set_title("  |  ".join(title_parts) if title_parts else "Spectrum", fontsize=8)
    if obs_plotted or pred_plotted or lib_plotted:
        ax.legend(fontsize=7, loc="upper right")

    if hasattr(self, "_pipeline_toolbar"):
        # Reset the zoom/pan history so "Home" resets to *this* spectrum's
        # full view, not a stale view left over from whichever row/zoom
        # level was showing before this one.
        self._pipeline_toolbar.update()
    self._pipeline_canvas.draw()

    lbl_parts = []
    if obs_plotted:
        lbl_parts.append("red=Observed ↑ (click a peak for its m/z)")
    if pred_plotted:
        lbl_parts.append("blue=Predicted spectrum ↓ (click a peak for its fragment)")
    elif lib_plotted:
        lbl_parts.append("blue=Library spectrum ↓ (best-matching experimental library spectrum)")
    elif cached is None and spectrum_predictor_score and str(spectrum_predictor_score) not in ("None", "nan"):
        lbl_parts.append("(Predicted spectrum not cached – rerun pipeline)")
    if hasattr(self, "_pipeline_spec_lbl"):
        self._pipeline_spec_lbl.setText(
            f"Row {row_idx+1}: {cname or 'unknown'}  "
            + ("  |  ".join(lbl_parts) if lbl_parts else "")
        )


def _pipeline_plot_selected(self):
    """Error-proof wrapper around _pipeline_plot_selected_impl.

    This slot fires from the row timer ~50ms after every pipeline finishes
    (and on every results-table row click / top-frags spin change). An
    unhandled Python exception raised here propagates into PyQt5's C++
    layer, which calls qFatal() -> abort() -> the whole process dies with
    SIGABRT right after a successful run, losing all results -- exactly the
    "runs fine, then the window vanishes" crash. Catch, log, and keep the
    window alive instead."""
    try:
        return _pipeline_plot_selected_impl(self)
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            if hasattr(self, "_pipeline_status_lbl"):
                self._pipeline_status_lbl.setText(
                    "Result view error (results are still in the table): "
                    + traceback.format_exc().strip().splitlines()[-1]
                )
        except Exception:
            pass


# ============================================================================
# Fragment structure viewer: click a predicted (blue, downward) peak in the
# pipeline spectrum plot to see the substructure Spectrum Predictor attributes it to.
# ============================================================================

class _FragmentRenderThread(QtCore.QThread):
    """Render one fragment SVG in the background (subprocess call into
    ms-pred's own venv -- see core/icefrag_adapter.render_fragment_svg)."""
    done = QtCore.pyqtSignal()

    def __init__(self, smiles, frag_id, observed_mz, observed_int, parent=None):
        super().__init__(parent)
        self.smiles       = smiles
        self.frag_id      = frag_id
        self.observed_mz  = observed_mz
        self.observed_int = observed_int
        self.result        = None
        self.error         = None

    def run(self):
        try:
            resp = mf_render_fragment_svg(self.smiles, self.frag_id, observed_mz=self.observed_mz)
            if resp and "error" in resp:
                self.error = resp["error"]
            else:
                self.result = resp
        except Exception as e:
            self.error = str(e)
        finally:
            self.done.emit()


def _pipeline_fill_frag_table(self, pred_mz_arr, pred_int_arr):
    """Populate the fragment-peak table, ranked by intensity (highest
    first) -- same top-N set currently drawn on the spectrum plot."""
    table = getattr(self, "_pipeline_frag_table", None)
    if table is None:
        return
    order = np.argsort(pred_int_arr)[::-1]
    table.setUpdatesEnabled(False)
    table.setRowCount(len(order))
    for row, idx in enumerate(order):
        mz_item = QtWidgets.QTableWidgetItem(f"{pred_mz_arr[idx]:.4f}")
        int_item = QtWidgets.QTableWidgetItem(f"{pred_int_arr[idx] * 100:.2f}")
        mz_item.setData(Qt.UserRole, int(idx))
        int_item.setData(Qt.UserRole, int(idx))
        table.setItem(row, 0, mz_item)
        table.setItem(row, 1, int_item)
    table.setUpdatesEnabled(True)


def _pipeline_show_fragment_for_index(self, idx):
    """Label the clicked predicted peak's m/z; when per-peak fragment data
    is available, also kick off a background render of that fragment --
    shared by both the plot-click handler and the fragment-table click
    handler so clicking either one behaves identically."""
    ctx = getattr(self, "_pipeline_click_ctx", None)
    if not ctx:
        return
    pred_mz  = ctx.get("pred_mz")
    pred_int = ctx.get("pred_int")
    frag_ids = ctx.get("frag_ids")
    canonical_smiles = ctx.get("canonical_smiles") or ""
    if pred_mz is None or pred_int is None or idx >= len(pred_mz):
        return
    observed_mz  = float(pred_mz[idx])
    observed_int = float(pred_int[idx])

    # The m/z label is unconditional -- clicking a blue peak must always
    # show its number, exactly like the red experimental peaks, even when
    # the fragment metadata (ids / canonical SMILES) is missing from a
    # restored session.  Fragment rendering below additionally requires
    # that metadata.
    ax = getattr(self, "_pipeline_ax", None)
    if ax is not None:
        old = getattr(self, "_pipeline_click_annotation", None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        self._pipeline_click_annotation = ax.annotate(
            f"{observed_mz:.4f}",
            xy=(observed_mz, -observed_int), xytext=(0, -8),
            textcoords="offset points", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1a5276",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2980b9", alpha=0.9),
        )
        if hasattr(self, "_pipeline_canvas"):
            self._pipeline_canvas.draw()

    has_frag_meta = (
        frag_ids is not None and idx < len(frag_ids) and canonical_smiles
    )
    if not has_frag_meta or getattr(self, "_pipeline_fragment_busy", False):
        if hasattr(self, "_pipeline_spec_lbl"):
            _why = (
                "fragment metadata not available in this session"
                if not has_frag_meta else "previous fragment render still busy"
            )
            self._pipeline_spec_lbl.setText(
                f"m/z {observed_mz:.4f} — {_why}"
            )
        return
    frag_id = int(frag_ids[idx])

    self._pipeline_fragment_busy = True
    if hasattr(self, "_pipeline_spec_lbl"):
        self._pipeline_prev_spec_lbl_text = self._pipeline_spec_lbl.text()
        self._pipeline_spec_lbl.setText(f"Rendering fragment for m/z {observed_mz:.4f} …")

    thread = _FragmentRenderThread(canonical_smiles, frag_id, observed_mz, observed_int, parent=self)
    thread.done.connect(lambda: _pipeline_fragment_render_done(self))
    self._pipeline_fragment_thread = thread   # keep a reference alive
    thread.start()


def _pipeline_on_spectrum_scroll(self, event):
    """Scroll-wheel zoom centered on the cursor's m/z position -- zooms the
    x-axis only (the y-axis is already a fixed -1..1 relative-intensity
    scale, so there's nothing to gain by zooming it); works regardless of
    which NavigationToolbar tool (if any) is currently selected."""
    ax = getattr(self, "_pipeline_ax", None)
    if ax is None or event.inaxes is not ax or event.xdata is None:
        return
    xlim = ax.get_xlim()
    span = xlim[1] - xlim[0]
    if span <= 0:
        return
    zoom_factor = 0.8 if event.button == "up" else 1.25  # up = zoom in
    new_span = max(span * zoom_factor, 0.5)  # never collapse past 0.5 Da
    # Keep the point under the cursor fixed while zooming.
    frac = (event.xdata - xlim[0]) / span
    new_lo = event.xdata - frac * new_span
    new_hi = new_lo + new_span
    ax.set_xlim(new_lo, new_hi)
    if hasattr(self, "_pipeline_canvas"):
        self._pipeline_canvas.draw_idle()


def _pipeline_show_observed_peak_mz(self, x_click):
    """Label the m/z of the clicked experimental (red) peak.

    Mirror of the predicted-peak interaction: one large number, only for the
    peak actually clicked.  Experimental peaks carry no fragment ids, so no
    fragment rendering follows -- the label replaces any previous annotation.
    """
    ctx = getattr(self, "_pipeline_obs_click_ctx", None)
    if not ctx:
        return
    obs_mz  = ctx.get("obs_mz")
    obs_int = ctx.get("obs_int")
    if obs_mz is None or len(obs_mz) == 0:
        return
    ax = getattr(self, "_pipeline_ax", None)
    if ax is None:
        return
    xlim = ax.get_xlim()
    tol = max(0.3, (xlim[1] - xlim[0]) * 0.01)
    idx = int(np.argmin(np.abs(obs_mz - x_click)))
    if abs(obs_mz[idx] - x_click) > tol:
        return
    mz_val  = float(obs_mz[idx])
    old = getattr(self, "_pipeline_click_annotation", None)
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    self._pipeline_click_annotation = ax.annotate(
        f"{mz_val:.4f}",
        xy=(mz_val, float(obs_int[idx])), xytext=(0, 8),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#922b21",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e74c3c", alpha=0.9),
    )
    if hasattr(self, "_pipeline_canvas"):
        self._pipeline_canvas.draw()


def _pipeline_on_spectrum_click(self, event):
    """Matplotlib button_press_event on the pipeline spectrum canvas.
    Clicks on the red experimental (y>0) trace label the peak's m/z; clicks
    on the blue Spectrum-Predictor trace (y<0) near an actual predicted peak
    additionally render that peak's fragment structure."""
    toolbar = getattr(self, "_pipeline_toolbar", None)
    if toolbar is not None and getattr(toolbar, "mode", ""):
        return  # a zoom-rect/pan tool is active -- this click is for that, not us
    if event.inaxes is not getattr(self, "_pipeline_ax", None):
        return
    if event.xdata is None or event.ydata is None:
        return
    if event.ydata > 0:
        _pipeline_show_observed_peak_mz(self, event.xdata)
        return
    if event.ydata >= 0:
        return
    ctx = getattr(self, "_pipeline_click_ctx", None)
    if not ctx:
        return
    pred_mz = ctx.get("pred_mz")
    if pred_mz is None or len(pred_mz) == 0:
        return

    idx = int(np.argmin(np.abs(pred_mz - event.xdata)))
    xlim = self._pipeline_ax.get_xlim()
    tol = max(0.3, (xlim[1] - xlim[0]) * 0.01)
    if abs(pred_mz[idx] - event.xdata) > tol:
        return

    _pipeline_show_fragment_for_index(self, idx)


def _pipeline_on_frag_table_click(self, row):
    """Row click in the fragment-peak table -- shows the same fragment
    view as clicking the corresponding peak on the plot."""
    table = getattr(self, "_pipeline_frag_table", None)
    if table is None:
        return
    item = table.item(row, 0)
    if item is None:
        return
    idx = item.data(Qt.UserRole)
    if idx is None:
        return
    _pipeline_show_fragment_for_index(self, int(idx))


def _pipeline_fragment_render_done(self):
    """Render the clicked peak's fragment directly into the structure
    preview panel (in place of the full candidate structure) -- no popup.
    Use the "Full structure" button next to the panel title to go back."""
    self._pipeline_fragment_busy = False
    thread = self.sender()
    if hasattr(self, "_pipeline_spec_lbl") and hasattr(self, "_pipeline_prev_spec_lbl_text"):
        self._pipeline_spec_lbl.setText(self._pipeline_prev_spec_lbl_text)
    if thread is None:
        return
    if thread.error or not thread.result:
        light_message_box(
            self, QtWidgets.QMessageBox.Warning, "Fragment structure",
            f"Could not render fragment: {thread.error or 'unknown error'}",
        )
        return

    svg = thread.result.get("svg", "")
    formula = thread.result.get("formula", "")
    h_shift = thread.result.get("h_shift", 0)
    observed_mz  = thread.observed_mz
    observed_int = thread.observed_int

    label = getattr(self, "_pipeline_struct_preview", None)
    if label is None:
        return
    if not svg:
        light_message_box(self, QtWidgets.QMessageBox.Warning, "Fragment structure", "Could not render fragment SVG.")
        return
    label.show_fragment(svg)   # re-renders itself on later resize too

    info_parts = [f"m/z {observed_mz:.4f}", f"Intensity {observed_int * 100:.2f}%"]
    if formula:
        info_parts.append(formula)
    if h_shift:
        sign = "+" if h_shift > 0 else ""
        info_parts.append(f"H{sign}{h_shift}")
    info_text = "  |  ".join(info_parts)
    self._pipeline_current_fragment_svg = svg
    self._pipeline_current_fragment_info = info_text
    label.setToolTip(f"Fragment: {info_text}")
    if hasattr(self, "_pipeline_struct_title"):
        self._pipeline_struct_title.setText(f"Fragment: {info_text}")
    if hasattr(self, "_pipeline_struct_reset_btn"):
        self._pipeline_struct_reset_btn.setVisible(True)
    if hasattr(self, "_pipeline_fragment_save_btn"):
        self._pipeline_fragment_save_btn.setVisible(True)


def _pipeline_save_current_fragment(self):
    """Save the fragment currently shown in the shared structure panel."""
    svg = getattr(self, "_pipeline_current_fragment_svg", "")
    if not svg:
        return
    default_name = "iceberg_fragment.svg"
    info = str(getattr(self, "_pipeline_current_fragment_info", "")).replace("/", "_")
    if info:
        default_name = "iceberg_fragment_" + info.split("  |  ")[0].replace(" ", "_") + ".svg"
    path, _ = light_get_save_file_name(
        self, "Save fragment structure", default_name,
        "SVG image (*.svg);;All files (*)"
    )
    if not path:
        return
    if not path.lower().endswith(".svg"):
        path += ".svg"
    try:
        Path(path).write_text(svg, encoding="utf-8")
        if hasattr(self, "_pipeline_spec_lbl"):
            self._pipeline_spec_lbl.setText(f"Fragment saved: {path}")
    except Exception as exc:
        light_message_box(self, QtWidgets.QMessageBox.Warning, "Save fragment", str(exc))


def _pipeline_on_row_selected(self):
    """Legacy stub kept for binding compatibility."""
    pass


def _pipeline_update_structure_preview(self, smiles, title=""):
    """Render the selected pipeline candidate's full structure in the shared
    preview panel (also used for the fragment-click view -- see
    _pipeline_fragment_render_done / _pipeline_reset_structure_preview)."""
    # Remembered so the "Full structure" reset button can restore this after
    # a fragment is shown, without needing to reselect the row.
    self._pipeline_full_structure_smiles = smiles
    self._pipeline_full_structure_title  = title
    self._pipeline_current_fragment_svg = ""
    self._pipeline_current_fragment_info = ""

    if hasattr(self, "_pipeline_struct_title"):
        self._pipeline_struct_title.setText("Structure")
    if hasattr(self, "_pipeline_struct_reset_btn"):
        self._pipeline_struct_reset_btn.setVisible(False)
    if hasattr(self, "_pipeline_fragment_save_btn"):
        self._pipeline_fragment_save_btn.setVisible(False)

    label = getattr(self, "_pipeline_struct_preview", None)
    if label is None:
        return
    smi = str(smiles or "").strip()
    label.setToolTip(str(title or smi))
    label.show_structure(smi)   # re-renders itself on later resize too


def _pipeline_reset_structure_preview(self):
    """"Full structure" button: restore the candidate structure after a
    fragment was shown, without touching the row selection/spectrum plot."""
    smi   = getattr(self, "_pipeline_full_structure_smiles", "")
    title = getattr(self, "_pipeline_full_structure_title", "")
    _pipeline_update_structure_preview(self, smi, title)
    if hasattr(self, "_pipeline_fragment_save_btn"):
        self._pipeline_fragment_save_btn.setVisible(False)


def _pipeline_mf_done(self, row_idx):
    """Called (in main thread via queued connection) when Spectrum Predictor finishes."""
    # If user has moved away, skip drawing
    if self._pipeline_result_table.currentRow() != row_idx:
        return

    # Retrieve results from the sender thread object
    thread = self.sender()
    if thread is None:
        return
    pred_mz  = thread.pred_mz
    pred_int = thread.pred_int
    mf_score = thread.mf_score

    if not pred_int or max(pred_int) <= 0:
        if hasattr(self, "_pipeline_spec_lbl"):
            self._pipeline_spec_lbl.setText(
                self._pipeline_spec_lbl.text().replace(" (computing...)", " (no prediction)")
            )
        return

    ax       = self._pipeline_ax
    pred_arr = np.asarray(pred_int, float) / max(pred_int)
    pred_mz_arr = np.asarray(pred_mz, float)

    ax.vlines(pred_mz_arr, 0, -pred_arr, colors="#2980b9", lw=1.2,
              label="Predicted spectrum ↓")

    # Widen X range if predicted peaks go beyond observed
    cur_xlim = ax.get_xlim()
    if len(pred_mz_arr) > 0:
        new_xmin = min(cur_xlim[0], pred_mz_arr.min() - 10)
        new_xmax = max(cur_xlim[1], pred_mz_arr.max() + 10)
        ax.set_xlim(new_xmin, new_xmax)

    ax.legend(fontsize=7, loc="upper right")

    # Update title with predicted-spectrum score
    df      = self._pipeline_last_df
    record  = df.iloc[row_idx]
    cname   = str(record.get("compound_name", "") or "")
    dm_sim  = record.get("DreaMS_similarity", 0.0)
    title_parts = []
    if cname and cname not in ("nan", "None"):
        title_parts.append(cname)
    if dm_sim:
        title_parts.append(f"Lib={float(dm_sim):.3f}")
    if mf_score is not None:
        title_parts.append(f"Pred={mf_score:.3f}")
    ax.set_title("  |  ".join(title_parts) if title_parts else "Spectrum", fontsize=8)

    self._pipeline_canvas.draw()

    if hasattr(self, "_pipeline_spec_lbl"):
        self._pipeline_spec_lbl.setText(
            f"Row {row_idx+1}: {cname or 'unknown'}  "
            f"[red=observed (↑) | blue=Predicted spectrum (↓)]"
        )


def _pipeline_save_results(self):
    _library_df = self._pipeline_last_df
    _clm_df = getattr(self, "_pipeline_last_denovo_df", None)
    # Category D (DeepMet CLM) is exported only while "Show D (CLM)" is pressed.
    _show_d = getattr(self, "_pl_show_d_btn", None)
    if _show_d is not None and not _show_d.isChecked():
        _clm_df = None
        if _library_df is not None and hasattr(_library_df, "empty") and not _library_df.empty:
            _library_df = _library_df[~_is_category_d(_library_df)].copy()
    _no_library = _library_df is None or (
        hasattr(_library_df, "empty") and _library_df.empty
    )
    _no_clm = _clm_df is None or (hasattr(_clm_df, "empty") and _clm_df.empty)
    if _no_library and _no_clm:
        _full = getattr(self, "_pipeline_last_df_full", None)
        _hidden_d = (_show_d is not None and not _show_d.isChecked() and _full is not None
                     and hasattr(_full, "empty") and not _full.empty and bool(_is_category_d(_full).any()))
        light_message_box(
            self,
            QtWidgets.QMessageBox.Information,
            "No Results",
            "No results to save yet."
            + ("\nOnly category-D rows exist and they are hidden; press [Show D (CLM)] to include them." if _hidden_d else ""),
        )
        return
    path, _ = light_get_save_file_name(
        self, "Save Results", "annomass_results.xlsx",
        "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
    )
    if not path:
        return
    # The user's chosen location is authoritative: resolve it to an absolute
    # path up front and create the target folder when the dialog allowed
    # typing a not-yet-existing directory name.
    path = os.path.abspath(path)
    _save_parent = os.path.dirname(path)
    if _save_parent:
        os.makedirs(_save_parent, exist_ok=True)
    try:
        export_df = _pipeline_export_dataframe(
            _library_df if not _no_library else pd.DataFrame()
        )
        if path.lower().endswith(".csv"):
            export_df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            _pipeline_save_excel_with_structures(export_df, path)
        # Library and library-external results are deliberately separate.
        _deepmet_sidecar = None
        if not _no_clm:
            _stem, _ext = os.path.splitext(path)
            _deepmet_sidecar = _stem + "_generated_structure_inference.xlsx"
            _pipeline_save_excel_with_structures(
                _pipeline_export_dataframe(_clm_df), _deepmet_sidecar
            )
        light_message_box(
            self,
            QtWidgets.QMessageBox.Information,
            "Saved",
            f"Saved to:\n{path}"
            + (f"\n\nSelected generated-structure inferences:\n{_deepmet_sidecar}" if _deepmet_sidecar else ""),
        )
    except Exception as e:
        light_message_box(self, QtWidgets.QMessageBox.Critical, "Error", str(e))


def _pipeline_save_figures(self):
    """Export mirror spectra + fragment structure sheets to a chosen folder.

    Deliberately separate from [Save Results (Excel)]: the table and the
    figure/fragment export have different destinations, and the user picks
    this one explicitly instead of getting a figures folder auto-created
    next to the workbook."""
    rows = getattr(self, "_pipeline_last_df", None)
    if rows is None or (hasattr(rows, "empty") and rows.empty):
        light_message_box(
            self,
            QtWidgets.QMessageBox.Information,
            "No Results",
            "No results to export figures from yet.",
        )
        return
    out_dir = light_get_existing_directory(
        self,
        "Choose Folder for Spectrum & Fragment Figures",
        _LAST_SAVE_DIR or os.getcwd(),
    )
    if not out_dir:
        return
    out_dir = os.path.abspath(out_dir)
    try:
        from core.result_figure_exporter import export_result_figures
        made = export_result_figures(
            rows,
            getattr(self, "_pipeline_spectrums", []),
            getattr(self, "_pipeline_mf_cache", {}),
            out_dir,
            full_results=self._pipeline_last_df,
            progress=lambda msg: print(msg, flush=True),
        )
    except Exception as e:
        light_message_box(
            self, QtWidgets.QMessageBox.Critical, "Error", f"Figure export failed:\n{e}"
        )
        return
    light_message_box(
        self,
        QtWidgets.QMessageBox.Information,
        "Figures Saved",
        f"Generated {len(made)} figure files in:\n{os.path.join(out_dir, 'figures')}\n\n"
        "(spectra_comparison = experimental/predicted mirror plots,\n"
        "iceberg_fragments = matched fragment structure sheets)",
    )


_PROJECT_FORMAT = "annomass-project"
_PROJECT_VERSION = 2


def _project_json_default(value):
    import numpy as _np
    if isinstance(value, _np.generic):
        return value.item()
    return str(value)


def _pipeline_write_project(path, payload):
    """Write the GUI session state as a portable zip container.

    Layout (version 2, no pickle anywhere):
      manifest.json        format/version/counts
      results_full.csv     complete legacy candidate table
      denovo.csv           DeepMet CLM diagnostic rows (when present)
      spectra.npz          experimental spectra peak arrays
      spectra_meta.json    per-spectrum metadata (title, precursor_mz, ...)
      predictions.npz      cached predicted-spectrum peak arrays
      predictions_meta.json  canonical SMILES per cached row

    Plain CSV/npz/JSON keeps the file loadable across machines and conda
    environments; only numpy/pandas themselves are required to read it.
    """
    import io
    import json
    import zipfile

    spectrums = list(payload.get("spectrums") or [])
    spec_arrays = {}
    spec_meta = []
    for i, sp in enumerate(spectrums):
        spec_arrays[f"mz_{i}"] = np.asarray(sp.peaks.mz, dtype=float)
        spec_arrays[f"int_{i}"] = np.asarray(sp.peaks.intensities, dtype=float)
        spec_meta.append(dict(getattr(sp, "metadata", None) or {}))

    mf_cache = payload.get("mf_cache") or {}
    pred_arrays = {}
    pred_meta = {}
    for key, cached in mf_cache.items():
        row = f"row_{key}"
        pred_arrays[f"mz_{row}"] = np.asarray(cached.get("mz", []), dtype=float)
        pred_arrays[f"int_{row}"] = np.asarray(cached.get("intensity", []), dtype=float)
        pred_arrays[f"frag_{row}"] = np.asarray(cached.get("frag_ids", []), dtype=np.int64)
        if cached.get("frag_bits") is not None:
            pred_arrays[f"bits_{row}"] = np.asarray(cached.get("frag_bits"))
        pred_meta[str(key)] = {"canonical_smiles": str(cached.get("canonical_smiles", "") or "")}

    def _npz_bytes(arrays):
        buf = io.BytesIO()
        if arrays:
            np.savez(buf, **arrays)
        return buf.getvalue()

    denovo_df = payload.get("denovo_df")
    has_denovo = denovo_df is not None and hasattr(denovo_df, "empty") and not denovo_df.empty
    manifest = {
        "format": _PROJECT_FORMAT,
        "version": _PROJECT_VERSION,
        "saved_at": payload.get("saved_at", ""),
        "n_spectra": len(spectrums),
        "n_result_rows": int(len(payload.get("df_full", []))),
        "mf_cache_rows": sorted(str(k) for k in mf_cache.keys()),
        "has_denovo": bool(has_denovo),
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        zf.writestr("results_full.csv", payload.get("df_full").to_csv(index=False))
        if has_denovo:
            zf.writestr("denovo.csv", denovo_df.to_csv(index=False))
        zf.writestr("spectra.npz", _npz_bytes(spec_arrays))
        zf.writestr("spectra_meta.json", json.dumps(spec_meta, default=_project_json_default))
        zf.writestr("predictions.npz", _npz_bytes(pred_arrays))
        zf.writestr(
            "predictions_meta.json", json.dumps(pred_meta, default=_project_json_default)
        )


def _read_project_v1_pickle(path):
    """Legacy version-1 project files (plain pickle)."""
    import pickle
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("format") != _PROJECT_FORMAT:
        raise ValueError("Not an AnnoMASS project file.")
    return payload


def _pipeline_read_project(path):
    """Read a project file (zip v2, or legacy pickle v1) after validation."""
    import io
    import json
    import zipfile

    from core.Spectrum import Spectrum as _ProjectSpectrum

    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] != b"PK":
        payload = _read_project_v1_pickle(path)
        version = int(payload.get("version", 0))
    else:
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            version = int(manifest.get("version", 0))
            if int(manifest.get("n_result_rows", 0)) and "results_full.csv" in zf.namelist():
                df_full = pd.read_csv(io.StringIO(zf.read("results_full.csv").decode("utf-8")))
            else:
                df_full = pd.DataFrame()
            if manifest.get("has_denovo") and "denovo.csv" in zf.namelist():
                denovo_df = pd.read_csv(io.StringIO(zf.read("denovo.csv").decode("utf-8")))
            else:
                denovo_df = None

            spectrums = []
            spec_meta = json.loads(zf.read("spectra_meta.json").decode("utf-8"))
            n_spectra = int(manifest.get("n_spectra", 0))
            if n_spectra:
                spec_npz = np.load(io.BytesIO(zf.read("spectra.npz")))
                for i in range(n_spectra):
                    spectrums.append(_ProjectSpectrum(
                        mz=spec_npz[f"mz_{i}"],
                        intensities=spec_npz[f"int_{i}"],
                        metadata=dict(spec_meta[i]) if i < len(spec_meta) else None,
                    ))

            mf_cache = {}
            rows = manifest.get("mf_cache_rows", [])
            if rows:
                pred_npz = np.load(io.BytesIO(zf.read("predictions.npz")))
                pred_meta = json.loads(zf.read("predictions_meta.json").decode("utf-8"))
                for row in rows:
                    key = int(row)
                    cached = {
                        "mz": pred_npz[f"mz_row_{row}"],
                        "intensity": pred_npz[f"int_row_{row}"],
                        "frag_ids": pred_npz[f"frag_row_{row}"],
                        "canonical_smiles": pred_meta.get(row, {}).get("canonical_smiles", ""),
                    }
                    if f"bits_row_{row}" in pred_npz.files:
                        cached["frag_bits"] = pred_npz[f"bits_row_{row}"]
                    mf_cache[key] = cached
        payload = {
            "format": _PROJECT_FORMAT,
            "version": version,
            "saved_at": manifest.get("saved_at", ""),
            "df_full": df_full,
            "mf_cache": mf_cache,
            "spectrums": spectrums,
            "denovo_df": denovo_df,
        }

    if payload.get("format") != _PROJECT_FORMAT:
        raise ValueError("Not an AnnoMASS project file.")
    if version > _PROJECT_VERSION:
        raise ValueError(
            f"Project file version {version} is newer than this app supports "
            f"(version {_PROJECT_VERSION}); please update AnnoMASS."
        )
    return payload


def _pipeline_save_project(self):
    """Save the current result session as a reloadable project file."""
    df = getattr(self, "_pipeline_last_df_full", None)
    if df is None or (hasattr(df, "empty") and df.empty):
        light_message_box(
            self,
            QtWidgets.QMessageBox.Information,
            "No Results",
            "Run the pipeline (or load a project) before saving a project.",
        )
        return
    path, _ = light_get_save_file_name(
        self, "Save AnnoMASS Project", "annomass_project.annomass",
        "AnnoMASS Project (*.annomass);;All Files (*)",
    )
    if not path:
        return
    path = os.path.abspath(path)
    if not path.lower().endswith(".annomass"):
        path += ".annomass"
    _save_parent = os.path.dirname(path)
    if _save_parent:
        os.makedirs(_save_parent, exist_ok=True)
    import time as _time
    payload = {
        "format": _PROJECT_FORMAT,
        "version": _PROJECT_VERSION,
        "saved_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "df_full": df,
        "mf_cache": getattr(self, "_pipeline_mf_cache", {}) or {},
        "spectrums": list(getattr(self, "_pipeline_spectrums", []) or []),
        "denovo_df": getattr(self, "_pipeline_last_denovo_df", None),
    }
    try:
        _pipeline_write_project(path, payload)
    except Exception as e:
        light_message_box(
            self, QtWidgets.QMessageBox.Critical, "Error", f"Failed to save project:\n{e}"
        )
        return
    light_message_box(
        self,
        QtWidgets.QMessageBox.Information,
        "Project Saved",
        f"Saved to:\n{path}\n\nUse [Load Project] later to restore the result "
        "table, spectra and predicted-spectrum cache without rerunning.",
    )


def _pipeline_load_project(self):
    """Restore a saved project session: results, spectra and detail views."""
    if not self.butt_pipeline.isEnabled():
        light_message_box(
            self,
            QtWidgets.QMessageBox.Warning,
            "Pipeline Running",
            "Wait for the current run to finish before loading a project.",
        )
        return
    path, _ = light_get_open_file_name(
        self, "Open AnnoMASS Project", _LAST_SAVE_DIR or os.getcwd(),
        "AnnoMASS Project (*.annomass);;All Files (*)",
    )
    if not path:
        return
    _pipeline_load_project_file(self, path)


def _pipeline_load_project_file(self, path):
    """Load a project from an explicit path (file dialog or launch argument)."""
    path = os.path.abspath(str(path))
    if not os.path.isfile(path):
        light_message_box(
            self, QtWidgets.QMessageBox.Critical, "Error", f"Project file not found:\n{path}"
        )
        return False
    try:
        payload = _pipeline_read_project(path)
    except Exception as e:
        light_message_box(
            self, QtWidgets.QMessageBox.Critical, "Error", f"Failed to open project:\n{e}"
        )
        return False
    df = payload.get("df_full")
    if df is None or (hasattr(df, "empty") and df.empty):
        light_message_box(
            self,
            QtWidgets.QMessageBox.Warning,
            "Empty Project",
            "The project file contains no result rows.",
        )
        return False
    self._pipeline_last_df_full = _pipeline_normalize_result_columns(df)
    self._pipeline_mf_cache = payload.get("mf_cache") or {}
    self._pipeline_spectrums = list(payload.get("spectrums") or [])
    self._pipeline_last_denovo_df = payload.get("denovo_df")
    self._pipeline_mf_pool = []
    self._pipeline_start_time = None
    try:
        _update_deepmet_clm_panel(self, self._pipeline_last_denovo_df)
    except Exception:
        pass
    # Rebuild the result table through the standard display path so Top-K,
    # ICE filtering and row-click detail views behave exactly like a fresh run.
    _apply_pipeline_topk(self, _final_time_str=f"project {payload.get('saved_at', '')}".strip())
    tw = self.tabWidget_results
    for i in range(tw.count()):
        if tw.tabText(i) == "Results":
            tw.setCurrentIndex(i)
            break
    return True


def _pipeline_clear_results(self):
    """Discard the current result session (table, caches, detail views)."""
    if not self.butt_pipeline.isEnabled():
        light_message_box(
            self,
            QtWidgets.QMessageBox.Warning,
            "Pipeline Running",
            "Wait for the current run to finish before clearing.",
        )
        return
    if (
        getattr(self, "_pipeline_last_df_full", None) is None
        and getattr(self, "_pipeline_last_df", None) is None
    ):
        light_message_box(
            self,
            QtWidgets.QMessageBox.Information,
            "Nothing to Clear",
            "No results are currently loaded.",
        )
        return
    if not light_confirm_box(
        self,
        "Clear Results",
        "Discard the current result table and cached spectra?\n"
        "Save a project file first if you may want these results back.",
    ):
        return
    self._pipeline_last_df = None
    self._pipeline_last_df_full = None
    self._pipeline_last_denovo_df = None
    self._pipeline_mf_cache = {}
    self._pipeline_spectrums = []
    self._pipeline_click_ctx = None
    self._pipeline_click_annotation = None

    t = getattr(self, "_pipeline_result_table", None)
    if t is not None:
        t.setUpdatesEnabled(False)
        t.clearContents()
        t.setRowCount(0)
        t.setUpdatesEnabled(True)
    frag = getattr(self, "_pipeline_frag_table", None)
    if frag is not None:
        frag.setRowCount(0)
    clm_table = getattr(self, "_pipeline_denovo_table", None)
    if clm_table is not None:
        clm_table.setRowCount(0)
    self._pipeline_clm_display_df = None
    clm_struct = getattr(self, "_pipeline_clm_struct_preview", None)
    if clm_struct is not None:
        clm_struct.show_structure("")
    try:
        self._pipeline_clm_ax.cla()
        self._pipeline_clm_canvas.draw_idle()
    except Exception:
        pass
    try:
        self._pipeline_update_structure_preview("", "")
    except Exception:
        pass
    try:
        self._pipeline_ax.cla()
        self._pipeline_canvas.draw()
    except Exception:
        pass
    if hasattr(self, "_pipeline_status_lbl"):
        self._pipeline_status_lbl.setText(
            "Results cleared. Load a project or run the pipeline to start again."
        )
    if hasattr(self, "_pipeline_spec_lbl"):
        self._pipeline_spec_lbl.setText("")
    self.progressBar.setRange(0, 100)
    self.progressBar.setValue(0)
    self.progressBar.setFormat("")


def _pipeline_unload_spectra(self):
    """Unload the loaded spectrum files (sidebar list + spectrum store)."""
    if not self.butt_pipeline.isEnabled():
        light_message_box(
            self,
            QtWidgets.QMessageBox.Warning,
            "Pipeline Running",
            "Wait for the current run to finish before unloading spectra.",
        )
        return
    n_loaded = len(self.spectrums) if self.spectrums is not None else 0
    if n_loaded == 0:
        light_message_box(
            self,
            QtWidgets.QMessageBox.Information,
            "No Spectra",
            "No spectrum files are currently loaded.",
        )
        return
    if not light_confirm_box(
        self,
        "Unload Spectra",
        f"Remove the {n_loaded} loaded spectra from the workspace?\n"
        "Saved result sessions keep their own spectra snapshot, so already "
        "computed results stay browsable.",
    ):
        return
    self.spectrums = pd.DataFrame(columns=["title", "spectrum"])
    if hasattr(self, "current_spectrum"):
        self.current_spectrum = None
    try:
        self.set_list_spectrums()
    except Exception:
        pass
    try:
        self._pipeline_ax.cla()
        self._pipeline_canvas.draw()
    except Exception:
        pass


def _pipeline_export_dataframe(df):
    """Return the shared public result schema (nine data columns)."""
    from core.result_table_export import compact_result_table
    return compact_result_table(_pipeline_normalize_result_columns(df))


def _pipeline_save_excel_with_structures(df, path):
    """Export pipeline results to Excel and embed one structure image per row."""
    import openpyxl
    from io import BytesIO
    from openpyxl.drawing.image import Image as XLImage

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "AnnoMASS Results"
    ws.freeze_panes = "A2"

    cols = list(df.columns)
    struct_col = len(cols) + 1
    for ci, col in enumerate(cols, start=1):
        ws.cell(row=1, column=ci, value=col)
    ws.cell(row=1, column=struct_col, value="Structure")

    img_buffers = []
    img_w, img_h = 150, 120
    row_h = img_h * 0.75
    smiles_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else None)

    for ri, (_, row) in enumerate(df.iterrows(), start=2):
        for ci, col in enumerate(cols, start=1):
            val = row.get(col, "")
            if pd.isna(val):
                val = ""
            elif isinstance(val, (np.integer, np.floating)):
                val = val.item()
            ws.cell(row=ri, column=ci, value=val)

        smi = str(row.get(smiles_col, "") if smiles_col else "").strip()
        if smi and smi.lower() not in ("nan", "none", "null"):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                buf = BytesIO()
                Draw.MolToImage(mol, size=(img_w, img_h), wedgeBonds=True).save(buf, format="PNG")
                buf.seek(0)
                img_buffers.append(buf)
                img = XLImage(buf)
                img.width = img_w
                img.height = img_h
                cell = f"{openpyxl.utils.get_column_letter(struct_col)}{ri}"
                ws.add_image(img, cell)
                ws.row_dimensions[ri].height = row_h

    for ci, col in enumerate(cols, start=1):
        width = min(max(len(str(col)) + 2, 10), 42)
        if col in ("smiles", "spectrum_title", "SMILES", "Spectrum IDs"):
            width = 34
        elif col in ("compound_name", "inchikey", "Compound name", "InChIKey"):
            width = 26
        ws.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = width
    ws.column_dimensions[openpyxl.utils.get_column_letter(struct_col)].width = 24
    ws.auto_filter.ref = ws.dimensions
    from openpyxl.styles import Alignment, Font, PatternFill
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="234E70")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    if "Annotation basis" in cols:
        ws.column_dimensions[openpyxl.utils.get_column_letter(cols.index("Annotation basis") + 1)].width = 64
    wb.save(path)










# ── bind Pipeline methods to AnnoMASS ────────────────────────────────────
AnnoMASS._init_pipeline_tab      = _init_pipeline_tab
AnnoMASS.run_pipeline            = run_pipeline
AnnoMASS._on_pipeline_progress   = _on_pipeline_progress
AnnoMASS._on_pipeline_done       = _on_pipeline_done
AnnoMASS._on_pipeline_failed     = _on_pipeline_failed
AnnoMASS._apply_pipeline_topk    = _apply_pipeline_topk
AnnoMASS._pipeline_save_results     = _pipeline_save_results
AnnoMASS._pipeline_save_figures     = _pipeline_save_figures
AnnoMASS._pipeline_save_project     = _pipeline_save_project
AnnoMASS._pipeline_load_project     = _pipeline_load_project
AnnoMASS._pipeline_load_project_file = _pipeline_load_project_file
AnnoMASS._pipeline_clear_results    = _pipeline_clear_results
AnnoMASS._pipeline_unload_spectra   = _pipeline_unload_spectra
AnnoMASS._pipeline_update_structure_preview = _pipeline_update_structure_preview
AnnoMASS._pipeline_update_cluster_map = _pipeline_update_cluster_map
AnnoMASS._pipeline_plot_selected    = _pipeline_plot_selected
AnnoMASS._pipeline_on_spectrum_click     = _pipeline_on_spectrum_click
AnnoMASS._pipeline_on_spectrum_scroll    = _pipeline_on_spectrum_scroll
AnnoMASS._pipeline_fragment_render_done  = _pipeline_fragment_render_done
AnnoMASS._pipeline_save_current_fragment = _pipeline_save_current_fragment
AnnoMASS._pipeline_on_row_selected  = _pipeline_on_row_selected
AnnoMASS._pipeline_mf_done          = _pipeline_mf_done
AnnoMASS._pipeline_tick_clock       = _pipeline_tick_clock


# ============================================================================
# Flavonoid Agent cross-thread bridge (main-thread side) -- see AgentWorker
# above and core/agent_tools.py for the tool-calling loop.
# ============================================================================

def _agent_pipeline_bridge_execute(self):
    """Main-thread execution of the agent's run_pipeline tool request.
    Mirrors run_pipeline()'s own precondition guard verbatim, plus one
    addition the manual path doesn't need (its button is simply disabled
    while running): no Thread_Pipeline already in progress."""
    if not _ensure_dreams_adapter():
        self.agent_pipeline_result_failed.emit(
            f"DreaMS is not available; AnnoMASS cannot run. {_DREAMS_IMPORT_ERROR or ''}"
        )
        return
    if len(self.spectrums) == 0:
        self.agent_pipeline_result_failed.emit(
            "No spectra are loaded. Load a spectra file first (e.g. via the "
            "load_spectra_file tool), then retry."
        )
        return
    for attr in ("pp", "pn", "deepmass_positive", "deepmass_negative",
                 "reference_positive", "reference_negative", "default_database"):
        if getattr(self, attr, None) is None:
            self.agent_pipeline_result_failed.emit("Reference resources are still loading. Wait, then retry.")
            return
    if getattr(self, "Thread_Pipeline", None) is not None and self.Thread_Pipeline.isRunning():
        self.agent_pipeline_result_failed.emit("A Pipeline run is already in progress; wait for it to finish.")
        return

    self.run_pipeline()
    if getattr(self, "Thread_Pipeline", None) is None:
        self.agent_pipeline_result_failed.emit("Failed to start.")
        return
    # Relay THIS run's completion to the agent, additively -- run_pipeline()'s
    # own _on_pipeline_done/_on_pipeline_failed connections are untouched, so
    # the Pipeline tab's table/status updates exactly as a manual click would.
    self.Thread_Pipeline.finished_with_result.connect(self._agent_relay_pipeline_ready)
    self.Thread_Pipeline.failed.connect(self._agent_relay_pipeline_failed)


def _agent_relay_pipeline_ready(self, payload):
    self.agent_pipeline_result_ready.emit(payload)


def _agent_relay_pipeline_failed(self, message):
    self.agent_pipeline_result_failed.emit(message)


def _agent_load_spectra_execute(self, file_path):
    """Main-thread execution of the agent's load_spectra_file tool request.
    Mirrors load_spectrums()'s body (the "Open" button handler) minus the
    file dialog -- takes explicit path(s) instead of prompting the user."""
    if isinstance(file_path, (list, tuple)):
        file_paths = [str(p) for p in file_path if str(p or "").strip()]
    else:
        file_paths = [str(file_path)] if str(file_path or "").strip() else []
    if not file_paths:
        self.agent_load_spectra_failed.emit("No file paths were provided.")
        return
    missing = [p for p in file_paths if not Path(p).exists()]
    if missing:
        shown = "\n".join(missing[:10])
        extra = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        self.agent_load_spectra_failed.emit(f"File not found:\n{shown}{extra}")
        return
    self._loaded_files = file_paths
    try:
        spectrums = load_from_files(file_paths)
    except Exception as e:
        self.agent_load_spectra_failed.emit(
            "Failed to load file(s). Tip: MGF with 'Num Peaks: 0' may crash the parser; "
            f"filter those out first. Error: {e}"
        )
        return
    titles = []
    for s in spectrums:
        titles.append(s.metadata.get("compound_name", "") or s.metadata.get("title", "Unknown"))
    self.spectrums = pd.DataFrame({"title": titles, "spectrum": spectrums})
    self.set_list_spectrums()
    self.agent_load_spectra_ready.emit({
        "n_files": len(file_paths),
        "files": file_paths[:50],
        "n_spectra": len(spectrums),
        "titles": titles[:50],
    })


def _agent_temp_basename(name_hint=None):
    """Build a human-readable temp/ basename (no extension) from a compound
    name / spectrum title, e.g. "Quercetin" or "Quercetin_2" if that name is
    already in use this session. Falls back to a random string when no hint
    is given (or it sanitizes away to nothing) so callers without compound
    context still get a valid, collision-free file."""
    Path("temp").mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name_hint or "").strip()).strip("._")
    safe = safe[:60].rstrip("._")
    if not safe or safe.lower() in ("nan", "unknown", "none"):
        return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    candidate = safe
    n = 2
    while Path(f"temp/{candidate}.png").exists():
        candidate = f"{safe}_{n}"
        n += 1
    return candidate


def _agent_export_figure_png(fig, name_hint=None):
    """Save a matplotlib Figure to a fresh temp PNG, same temp/ convention
    plot_mol already uses for structure images. Returns the resolved path."""
    fn = _agent_temp_basename(name_hint)
    Path("temp").mkdir(exist_ok=True)
    path = str(Path(f"temp/{fn}.png").resolve())
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _agent_show_plot_execute(self, spectrum_index):
    """Main-thread execution of the agent's show_spectrum_plot tool request.
    Finds the list_spectrum row whose Qt.UserRole data matches spectrum_index
    (row position isn't guaranteed to equal the index), selects it, and plots
    the raw loaded spectrum directly from self.spectrums -- independent of
    whether the unified algorithm has been run yet (it used to require
    checking self.identified_spectrums, populated only by the old manual
    identify workflow, which no longer exists). The rendered spectrum is then
    exported to a PNG and shown in the Agent panel's embedded preview pane
    (Agent Mode hides the manual Spectrum View/Structure View splitter pane,
    see _apply_mode in _init_agent_tab) rather than a separate window."""
    target_row = -1
    for row in range(self.list_spectrum.count()):
        item = self.list_spectrum.item(row)
        if item is not None and item.data(Qt.UserRole) == spectrum_index:
            target_row = row
            break
    if target_row < 0:
        self.agent_plot_failed.emit(f"No loaded spectrum found with index {spectrum_index}.")
        return
    try:
        sp = self.spectrums["spectrum"].iloc[target_row]
    except Exception:
        self.agent_plot_failed.emit(f"Could not load spectrum {spectrum_index}.")
        return
    self.list_spectrum.setCurrentRow(target_row)
    self._plot_spectrum_object(sp)
    try:
        name_hint = str(sp.get("compound_name") or sp.get("title") or "")
        png_path = _agent_export_figure_png(self._pipeline_fig, name_hint=name_hint)
        self._get_file_viewer().load_file(png_path)
        self._agent_show_preview_panel()
        self._agent_update_preview_title(png_path)
        self._agent_append_display("file_link", png_path)
    except Exception:
        pass  # the plot itself succeeded; a failed export/preview shouldn't fail the tool call
    self.agent_plot_ready.emit({"spectrum_index": spectrum_index})


def _agent_show_structure_execute(self, smiles, compound_name=""):
    """Main-thread execution of the agent's show_structure_plot tool request
    -- renders an arbitrary SMILES independent of whatever spectrum/
    candidate row is currently selected (see plot_mol's smiles_override
    parameter), then shows the exact rendered PNG in the embedded preview
    pane (see _agent_show_plot_execute's docstring for why)."""
    if not smiles or not smiles.strip():
        self.agent_plot_failed.emit("No SMILES provided.")
        return
    if Chem.MolFromSmiles(smiles) is None:
        self.agent_plot_failed.emit(f"RDKit could not parse SMILES: {smiles}")
        return
    name_hint = str(compound_name or "").strip()
    if not name_hint:
        # The Agent didn't pass a name -- fall back to looking the SMILES up
        # in the current AnnoMASS results so the saved file still gets a
        # readable name instead of a random one.
        df = getattr(self, "_pipeline_last_df", None)
        try:
            if df is not None and "smiles" in df.columns:
                hit = df.loc[df["smiles"].astype(str) == str(smiles), "compound_name"]
                if len(hit):
                    name_hint = str(hit.iloc[0] or "")
        except Exception:
            pass
    self.plot_mol(smiles_override=smiles, name_hint=name_hint)
    png_path = getattr(self, "_last_structure_png_path", None)
    if png_path:
        try:
            self._get_file_viewer().load_file(png_path)
            self._agent_show_preview_panel()
            self._agent_update_preview_title(png_path)
            self._agent_append_display("file_link", png_path)
        except Exception:
            pass
    self.agent_plot_ready.emit({"smiles": smiles})


def _agent_open_file_execute(self, file_path):
    """Main-thread execution of the agent's open_file tool request -- opens
    the file in the embedded preview panel (see core/file_viewer_widget.py)
    and leaves a clickable link in the chat so the user can re-open it
    later without asking the agent again."""
    from pathlib import Path as _Path
    if not file_path or not _Path(file_path).exists():
        self.agent_file_failed.emit(f"File not found: {file_path}")
        return
    try:
        self._get_file_viewer().load_file(file_path)
        self._agent_show_preview_panel()
        self._agent_update_preview_title(file_path)
        self._agent_append_display("file_link", file_path)
    except Exception as e:
        self.agent_file_failed.emit(f"Failed to open file: {e}")
        return
    self.agent_file_ready.emit({"file_path": file_path})


def _agent_save_current_preview_execute(self, payload):
    """Main-thread bridge for conversational Save As requests from the Agent."""
    payload = payload if isinstance(payload, dict) else {}
    destination = str(payload.get("destination_path") or "").strip()
    overwrite = bool(payload.get("overwrite", False))
    if not destination:
        self.agent_preview_save_failed.emit("A destination path and file name are required.")
        return
    try:
        saved_path = self._get_file_viewer().save_current_to(destination, overwrite=overwrite)
    except Exception as exc:
        self.agent_preview_save_failed.emit(str(exc))
        return
    self._agent_update_preview_title()
    self._agent_append_display("file_link", saved_path)
    self.agent_preview_save_ready.emit({
        "file_path": saved_path,
        "verified": Path(saved_path).is_file(),
    })


def _get_file_viewer(self):
    """Returns the embedded right-side preview panel built in _init_agent_tab
    (core/file_viewer_widget.py, embedded=True -- NOT a separate OS window)."""
    return self._agent_preview_panel


def _agent_show_preview_panel(self):
    """Un-collapses the right-side preview panel (it starts hidden until
    something is shown)."""
    self._agent_preview_panel_container.setVisible(True)


def _agent_update_preview_title(self, file_path=None):
    if not hasattr(self, "_agent_preview_title_lbl"):
        return
    path = file_path or getattr(self._get_file_viewer(), "_current_path", None)
    if path:
        p = Path(path)
        self._agent_preview_title_lbl.setText(f"File Viewer - {p.name}")
        self._agent_preview_title_lbl.setToolTip(str(p.resolve()))


def _agent_use_current_preview_file(self):
    path = getattr(self._get_file_viewer(), "_current_path", None)
    if not path:
        self._agent_view_file_clicked()
        path = getattr(self._get_file_viewer(), "_current_path", None)
    if path:
        self._agent_input_edit.setPlainText(f"请基于当前文件回答：{Path(path).resolve()}")
        self._agent_input_edit.setFocus()


def _agent_view_file_clicked(self):
    viewer = self._get_file_viewer()
    self._agent_show_preview_panel()
    if getattr(viewer, "_current_path", None) is None:
        viewer._on_open_clicked()
    self._agent_update_preview_title()


def _agent_file_tree_clicked(self, index):
    """Double-click on a previewable file in the sidebar file browser opens
    it in the embedded preview panel."""
    from core.file_viewer_widget import PREVIEWABLE_EXTENSIONS
    path = self._agent_fs_model.filePath(index)
    p = Path(path)
    if p.is_dir() or p.suffix.lower() not in PREVIEWABLE_EXTENSIONS:
        return
    viewer = self._get_file_viewer()
    try:
        viewer.load_file(path)
    except Exception as e:
        self.ErrorMsg(f"Failed to open {path}: {e}")
        return
    self._agent_show_preview_panel()
    self._agent_update_preview_title(path)


def _agent_file_tree_context_menu(self, pos):
    """Right-click menu for the Agent file browser.

    The file tree intentionally opens files only on double-click, so common
    single-click actions live here: copy paths, preview files, and change the
    visible root without risking accidental file opens.
    """
    from core.i18n import tr
    index = self._agent_file_tree.indexAt(pos)
    if not index.isValid():
        return
    path = self._agent_fs_model.filePath(index)
    if not path:
        return
    p = Path(path)

    menu = QtWidgets.QMenu(self._agent_file_tree)
    menu.setStyleSheet(GLOBAL_MENU_STYLE)

    copy_abs = menu.addAction("Copy absolute path")
    copy_rel = menu.addAction("Copy path relative to project")
    copy_name = menu.addAction("Copy file name" if p.is_file() else "Copy folder name")
    insert_path = menu.addAction(tr("agent.insert_path"))
    ask_about_file = None
    if p.is_file():
        ask_about_file = menu.addAction(tr("agent.ask_about_file"))
    menu.addSeparator()

    open_preview = None
    if p.is_file():
        open_preview = menu.addAction("Open in preview")
    set_root = None
    if p.is_dir():
        set_root = menu.addAction("Use this folder as root")
    reveal = menu.addAction("Open containing folder")
    reset_root = menu.addAction("Back to project root")
    menu.addSeparator()
    delete_path = menu.addAction("Delete folder..." if p.is_dir() else "Delete file...")

    chosen = menu.exec_(self._agent_file_tree.viewport().mapToGlobal(pos))
    if chosen is None:
        return
    if chosen == copy_abs:
        self._agent_copy_text_to_clipboard(str(p.resolve()), "Copied absolute path.")
    elif chosen == copy_rel:
        self._agent_copy_text_to_clipboard(_agent_relative_project_path(p), "Copied relative path.")
    elif chosen == copy_name:
        self._agent_copy_text_to_clipboard(p.name, "Copied name.")
    elif chosen == insert_path:
        self._agent_insert_text_at_cursor(str(p.resolve()))
    elif ask_about_file is not None and chosen == ask_about_file:
        self._agent_input_edit.setPlainText(f"请查看并解释这个文件：{p.resolve()}")
        self._agent_input_edit.setFocus()
    elif open_preview is not None and chosen == open_preview:
        self._agent_open_file_execute(str(p))
    elif set_root is not None and chosen == set_root:
        self._agent_set_file_browser_root(str(p))
    elif chosen == reveal:
        target = p if p.is_dir() else p.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))
    elif chosen == reset_root:
        self._agent_set_file_browser_root(str(Path(__file__).resolve().parent))
    elif chosen == delete_path:
        self._agent_delete_file_tree_path(p)


def _agent_delete_file_tree_path(self, path):
    p = Path(path)
    try:
        target = p.resolve()
    except Exception:
        target = p
    project_root = Path(__file__).resolve().parent
    current_root = Path(self._agent_fs_model.rootPath()).resolve() if hasattr(self, "_agent_fs_model") else project_root

    if target in (project_root, current_root):
        self.WarnMsg("This folder is the current file-browser root and cannot be deleted here.")
        return
    if not target.exists():
        self.WarnMsg(f"Path no longer exists:\n{target}")
        self._agent_refresh_file_tree()
        return

    kind = "folder and all of its contents" if target.is_dir() else "file"
    msg = QtWidgets.QMessageBox(self)
    msg.setIcon(QtWidgets.QMessageBox.Warning)
    msg.setWindowTitle("Delete from project")
    msg.setText(f"Delete this {kind}?")
    msg.setInformativeText(str(target))
    msg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes)
    msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
    msg.button(QtWidgets.QMessageBox.Yes).setText("Delete")
    msg.setStyleSheet(LIGHT_MESSAGE_BOX_STYLE)
    if msg.exec_() != QtWidgets.QMessageBox.Yes:
        return

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except Exception as e:
        self.ErrorMsg(f"Failed to delete:\n{target}\n\n{e}")
        return

    self._agent_refresh_file_tree()
    if hasattr(self, "_agent_status_lbl"):
        self._agent_status_lbl.setText(f"Deleted: {target}")


def _agent_copy_text_to_clipboard(self, text, status_text=None):
    QtWidgets.QApplication.clipboard().setText(str(text))
    if status_text and hasattr(self, "_agent_status_lbl"):
        self._agent_status_lbl.setText(status_text)


def _agent_insert_text_at_cursor(self, text):
    edit = self._agent_input_edit
    cursor = edit.textCursor()
    prefix = "" if not edit.toPlainText().strip() else " "
    cursor.insertText(prefix + str(text))
    edit.setTextCursor(cursor)
    edit.setFocus()


def _agent_apply_file_filter(self, text):
    text = (text or "").strip()
    if not hasattr(self, "_agent_fs_model"):
        return
    if text:
        self._agent_fs_model.setNameFilters([f"*{text}*"])
    else:
        self._agent_fs_model.setNameFilters([])


def _agent_relative_project_path(path):
    p = Path(path).resolve()
    root = Path(__file__).resolve().parent
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _agent_set_file_browser_root(self, root):
    root = str(Path(root).resolve())
    self._agent_fs_model.setRootPath(root)
    self._agent_file_tree.setRootIndex(self._agent_fs_model.index(root))
    if hasattr(self, "_agent_status_lbl"):
        self._agent_status_lbl.setText(f"File browser root: {root}")


def _agent_refresh_file_tree(self):
    """Manual refresh for the file tree (⟳ button): QFileSystemModel only
    watches directories the tree has actually expanded/rendered, and on
    WSL9p its filesystem-change notifications aren't always reliable, so a
    file a tool just wrote (e.g. an exported report) can be invisible until
    something re-triggers a scan. Re-applying the current root path does
    that -- same mechanism _agent_set_file_browser_root already uses when
    switching roots, just applied to the existing one instead of a new one."""
    self._agent_set_file_browser_root(self._agent_fs_model.rootPath())


def _agent_chat_link_clicked(self, url):
    """A link in the chat log was clicked -- either a file/image link (see
    _agent_entry_html's "file_link" kind), opened in the embedded preview
    panel instead of letting Qt try to launch an external program, or an
    "agenttool://session_idx/tool_log_id" link (the "tool" kind's clickable
    checklist line), which shows that call's full input/output JSON."""
    raw = url.toString()
    if raw.startswith("agenttool://"):
        try:
            session_idx_str, tool_id_str = raw[len("agenttool://"):].split("/", 1)
            self._agent_show_tool_detail(int(session_idx_str), int(tool_id_str))
        except (ValueError, IndexError):
            pass
        return

    if raw.startswith("quicktask:"):
        # Welcome-screen quick-task pill (see _agent_welcome_html) -- fills
        # the input box, same as clicking one in the web version does, so
        # the user can still edit it before sending.
        # No "//" after the scheme deliberately: a "scheme://<unicode>"
        # link makes Qt's QUrl parse the payload as a HOSTNAME, which it
        # then silently IDNA/punycode-encodes (confirmed live: clicking a
        # Chinese quick-task pill filled the input with "xn--..." garbage
        # instead of the task text). "scheme:<payload>" (opaque form, no
        # authority component) sidesteps hostname parsing entirely.
        import urllib.parse
        task_text = urllib.parse.unquote(raw[len("quicktask:"):])
        self._agent_input_edit.setPlainText(task_text)
        self._agent_input_edit.setFocus()
        return

    path = url.toLocalFile() or raw
    if not path:
        return
    viewer = self._get_file_viewer()
    try:
        viewer.load_file(path)
    except Exception as e:
        self.ErrorMsg(f"Failed to open {path}: {e}")
        return
    self._agent_show_preview_panel()


def _agent_show_tool_detail(self, session_idx, tool_log_id):
    """The 'Agent execution timeline' detail popup -- full input/output JSON
    for one tool call, opened by clicking its checklist line in the chat."""
    if session_idx < 0 or session_idx >= len(self._agent_sessions):
        return
    tool_log = self._agent_sessions[session_idx].get("tool_log") or []
    record = next((r for r in tool_log if r["id"] == tool_log_id), None)
    if record is None:
        return

    import json
    dlg = QtWidgets.QDialog(self)
    dlg.setWindowTitle(f"Tool call: {record['name']}")
    dlg.setStyleSheet(LIGHT_DIALOG_STYLE)
    dlg.resize(680, 520)
    lay = QtWidgets.QVBoxLayout(dlg)
    status_color = "#2e7d32" if record["status"] == "completed" else "#c62828"
    lay.addWidget(QtWidgets.QLabel(
        f"<b>{record['name']}</b> — "
        f"<span style='color:{status_color};'>{record['status']}</span> — "
        f"{record['duration_s']}s"
    ))
    mono = QtGui.QFont("Consolas")
    mono.setStyleHint(QtGui.QFont.Monospace)

    tabs = QtWidgets.QTabWidget()
    for title, payload in (
        ("Arguments", record.get("args")),
        ("Result", record.get("result")),
        ("Raw", record),
    ):
        body = QtWidgets.QPlainTextEdit()
        body.setReadOnly(True)
        body.setFont(mono)
        body.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        tabs.addTab(body, title)
    lay.addWidget(tabs, stretch=1)
    close_btn = QtWidgets.QPushButton("Close")
    close_btn.clicked.connect(dlg.accept)
    lay.addWidget(close_btn)
    dlg.exec_()


def _agent_browse_file_root(self):
    dlg = QtWidgets.QFileDialog(self, "Choose Folder", self._agent_fs_model.rootPath())
    dlg.setFileMode(QtWidgets.QFileDialog.Directory)
    dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setStyleSheet(LIGHT_FILE_DIALOG_STYLE)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return
    selected = dlg.selectedFiles()
    root = selected[0] if selected else ""
    if root:
        self._agent_set_file_browser_root(root)


# ============================================================================
# Flavonoid Agent standalone tab
# ============================================================================

def _wsl_windows_clipboard_text():
    """Read the Windows host clipboard via powershell.exe interop.

    Under WSLg, this app runs its Qt platform plugin as xcb (X11 via
    XWayland) so fcitx5 Chinese input works -- but XWayland's CLIPBOARD
    sync with the Windows host is flaky (unlike native Wayland clients),
    so Ctrl+V/paste can silently see empty clipboard content here even
    though the Windows source really did copy something. Falling back to
    the Windows clipboard directly sidesteps that XWayland bridge.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.rstrip("\r\n")
    except Exception:
        pass
    return None


class _ClipboardBridgeLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that tops up an empty Qt clipboard from the Windows host
    before deferring to Qt's normal paste handling (see
    _wsl_windows_clipboard_text). Ctrl+V and the right-click Paste action
    both go through Qt's own codepath afterward, unchanged."""

    def _prime_clipboard_from_windows(self):
        cb = QtWidgets.QApplication.clipboard()
        if not cb.text():
            text = _wsl_windows_clipboard_text()
            if text:
                cb.setText(text)

    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Paste):
            self._prime_clipboard_from_windows()
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        self._prime_clipboard_from_windows()
        super().contextMenuEvent(event)


class AgentSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from core.i18n import tr
        self.setObjectName("agent_settings_dialog")
        self.setWindowTitle(tr("agent.settings_title"))
        self.setMinimumSize(680, 420)
        self.setStyleSheet(LIGHT_DIALOG_STYLE + """
            QDialog#agent_settings_dialog {
                background: #F8FAFC;
            }
            QDialog#agent_settings_dialog QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                margin-top: 18px;
                padding: 14px 12px 12px 12px;
                font-weight: 700;
                color: #0F172A;
            }
            QDialog#agent_settings_dialog QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background: #FFFFFF;
                color: #334155;
            }
            QDialog#agent_settings_dialog QLineEdit,
            QDialog#agent_settings_dialog QComboBox {
                min-height: 28px;
                border-radius: 6px;
                border: 1px solid #CBD5E1;
                padding: 3px 8px;
                background: #FFFFFF;
            }
            QDialog#agent_settings_dialog QLineEdit:focus,
            QDialog#agent_settings_dialog QComboBox:focus {
                border-color: #2563EB;
            }
            QDialog#agent_settings_dialog QPushButton {
                min-height: 28px;
                border-radius: 6px;
            }
            QDialog#agent_settings_dialog QLabel#agent_settings_status {
                color: #475569;
            }
        """)
        from core import agent_config as ac
        self._ac = ac
        self.cfg = ac.load_config()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        settings_box = QtWidgets.QGroupBox(tr("agent.settings_connection"))
        settings_lay = QtWidgets.QVBoxLayout(settings_box)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.provider_combo = QtWidgets.QComboBox()
        for key, preset in ac.PROVIDER_PRESETS.items():
            self.provider_combo.addItem(preset["label"], key)
        form.addRow(tr("agent.settings_provider"), self.provider_combo)

        self.protocol_combo = QtWidgets.QComboBox()
        self.protocol_combo.addItem(tr("agent.settings_protocol_openai"), "openai_compatible")
        self.protocol_combo.addItem(tr("agent.settings_protocol_anthropic"), "anthropic")
        form.addRow(tr("agent.settings_protocol"), self.protocol_combo)

        # Base URL/Model are ALWAYS plain text fields (never a fixed dropdown
        # of model names) -- Doubao/Volcengine Ark identifies deployments by
        # an opaque endpoint ID, not a friendly model name, and applying that
        # rule uniformly avoids a provider-specific special case here.
        self.base_url_edit = _ClipboardBridgeLineEdit()
        self.base_url_edit.setClearButtonEnabled(True)
        form.addRow(tr("agent.settings_base_url"), self.base_url_edit)

        # Editable combo: suggests common models per provider (e.g. DeepSeek's
        # deepseek-chat/deepseek-reasoner) but stays free-text so a newer
        # model id, or Doubao's opaque endpoint ID, can always be typed in.
        self.model_edit = QtWidgets.QComboBox()
        self.model_edit.setEditable(True)
        self.model_edit.setLineEdit(_ClipboardBridgeLineEdit())
        self.model_edit.lineEdit().setClearButtonEnabled(True)
        form.addRow(tr("agent.settings_model"), self.model_edit)

        key_row = QtWidgets.QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        self.api_key_edit = _ClipboardBridgeLineEdit()
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_edit.setClearButtonEnabled(True)
        show_chk = QtWidgets.QCheckBox(tr("agent.settings_show"))
        show_chk.toggled.connect(
            lambda checked: self.api_key_edit.setEchoMode(
                QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
            )
        )
        key_row.addWidget(self.api_key_edit, stretch=1)
        key_row.addWidget(show_chk)
        form.addRow(tr("agent.settings_api_key"), key_row)

        settings_lay.addLayout(form)
        layout.addWidget(settings_box)

        # Thinking effort: a global, provider-independent behavior lever
        # (see core/agent_config.py's THINKING_EFFORT_LEVELS / core/
        # agent_tools.py's EFFORT_INSTRUCTIONS) -- not tied to any one
        # provider's settings, but kept in this same dialog since it's the
        # only Agent-wide settings surface that exists.
        self.effort_combo = QtWidgets.QComboBox()
        self.effort_combo.addItem(tr("agent.effort_low"), "low")
        self.effort_combo.addItem(tr("agent.effort_medium"), "medium")
        self.effort_combo.addItem(tr("agent.effort_high"), "high")
        self.effort_combo.setToolTip(tr("agent.settings_thinking_effort_tooltip"))
        idx = self.effort_combo.findData(ac.get_thinking_effort(self.cfg))
        if idx >= 0:
            self.effort_combo.setCurrentIndex(idx)

        behavior_box = QtWidgets.QGroupBox(tr("agent.settings_behavior"))
        behavior_lay = QtWidgets.QFormLayout(behavior_box)
        behavior_lay.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        behavior_lay.setHorizontalSpacing(12)
        behavior_lay.setVerticalSpacing(10)
        behavior_lay.addRow(tr("agent.settings_thinking_effort"), self.effort_combo)
        layout.addWidget(behavior_box)

        test_row = QtWidgets.QHBoxLayout()
        self.test_btn = QtWidgets.QPushButton(tr("agent.settings_test_connection"))
        self.test_status_lbl = QtWidgets.QLabel("")
        self.test_status_lbl.setObjectName("agent_settings_status")
        self.test_status_lbl.setWordWrap(True)
        test_row.addWidget(self.test_btn)
        test_row.addWidget(self.test_status_lbl, stretch=1)
        layout.addLayout(test_row)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_save)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.provider_combo.currentIndexChanged.connect(lambda *_: self._apply_preset())
        self.test_btn.clicked.connect(self._on_test_connection)
        self._test_thread = None

        active_key, active_cfg = ac.get_active_provider_config(self.cfg)
        if active_key:
            idx = self.provider_combo.findData(active_key)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
            self._populate_model_choices(active_key)
            self.base_url_edit.setText(active_cfg["base_url"])
            self.model_edit.setCurrentText(active_cfg["model"])
            self.api_key_edit.setText(active_cfg["api_key"])
            self._set_protocol(active_cfg.get("protocol") or ac.PROVIDER_PRESETS.get(active_key, {}).get("protocol"))
        else:
            self._apply_preset()

    def _current_provider_key(self):
        return self.provider_combo.currentData()

    def _set_protocol(self, protocol):
        idx = self.protocol_combo.findData(protocol or "openai_compatible")
        self.protocol_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _populate_model_choices(self, key):
        preset = self._ac.PROVIDER_PRESETS.get(key, {})
        self.model_edit.blockSignals(True)
        self.model_edit.clear()
        self.model_edit.addItems(preset.get("model_choices", []))
        self.model_edit.blockSignals(False)
        if key in ("doubao", "volcengine_ark"):
            # Deliberately left English -- this describes a Volcengine Ark
            # console concept, not app UI chrome; not worth a translation key.
            self.model_edit.lineEdit().setPlaceholderText(
                "Ark model name or endpoint ID from your Volcengine console"
            )
        else:
            self.model_edit.lineEdit().setPlaceholderText("")

    def _apply_preset(self):
        key = self._current_provider_key()
        preset = self._ac.PROVIDER_PRESETS.get(key, {})
        stored = self.cfg.get("providers", {}).get(key, {})
        self._populate_model_choices(key)
        self._set_protocol(stored.get("protocol") or preset.get("protocol", "openai_compatible"))
        self.base_url_edit.setText(stored.get("base_url") or preset.get("base_url", ""))
        self.model_edit.setCurrentText(stored.get("model") or preset.get("model", ""))
        self.api_key_edit.setText(stored.get("api_key", ""))

    def _build_provider_cfg(self):
        return {
            "protocol": self.protocol_combo.currentData() or "openai_compatible",
            "base_url": self.base_url_edit.text().strip(),
            "model": self.model_edit.currentText().strip(),
            "api_key": self.api_key_edit.text().strip(),
        }

    def _on_test_connection(self):
        from core.i18n import tr
        provider_cfg = self._build_provider_cfg()
        if not provider_cfg["base_url"] or not provider_cfg["model"] or not provider_cfg["api_key"]:
            self.test_status_lbl.setText(tr("agent.settings_fill_first"))
            return
        self.test_btn.setEnabled(False)
        self.test_status_lbl.setText(tr("agent.settings_testing"))
        self._test_thread = AgentTestConnectionWorker(provider_cfg)
        self._test_thread.finished_with_result.connect(self._on_test_done)
        self._test_thread.failed.connect(lambda msg: self._on_test_done((False, msg)))
        self._test_thread.start()

    def _on_test_done(self, result):
        from core.i18n import tr
        ok, message = result
        self.test_btn.setEnabled(True)
        prefix = tr("agent.settings_connected") if ok else tr("agent.settings_failed")
        self.test_status_lbl.setText(prefix + str(message))

    def _on_save(self):
        key = self._current_provider_key()
        preset = self._ac.PROVIDER_PRESETS.get(key, {})
        self._ac.set_provider_config(
            self.cfg, key,
            base_url=self.base_url_edit.text().strip(),
            model=self.model_edit.currentText().strip(),
            api_key=self.api_key_edit.text().strip(),
            protocol=self.protocol_combo.currentData() or preset.get("protocol"),
        )
        self._ac.set_thinking_effort(self.cfg, self.effort_combo.currentData())
        self._ac.save_config(self.cfg)
        self.accept()


class AgentSkillsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from core import agent_skills
        self._skills = agent_skills
        self.setWindowTitle("Agent Skills")
        self.setMinimumSize(760, 520)
        self.setStyleSheet(LIGHT_DIALOG_STYLE)

        layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 4, 0)
        self.skill_list = QtWidgets.QListWidget()
        self.skill_list.currentItemChanged.connect(lambda *_: self._load_selected())
        left_lay.addWidget(QtWidgets.QLabel("Installed skills"))
        left_lay.addWidget(self.skill_list, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.install_file_btn = QtWidgets.QPushButton("Install file")
        self.install_file_btn.clicked.connect(self._install_file)
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        row.addWidget(self.install_file_btn)
        row.addWidget(self.delete_btn)
        left_lay.addLayout(row)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(4, 0, 0, 0)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("skill_name")
        self.content_edit = QtWidgets.QPlainTextEdit()
        mono = QtGui.QFont("Consolas")
        mono.setStyleHint(QtGui.QFont.Monospace)
        self.content_edit.setFont(mono)
        self.content_edit.setPlaceholderText("# Skill title\n\nReusable instructions for the Agent...")
        right_lay.addWidget(QtWidgets.QLabel("Name"))
        right_lay.addWidget(self.name_edit)
        right_lay.addWidget(QtWidgets.QLabel("Markdown instructions"))
        right_lay.addWidget(self.content_edit, stretch=1)
        save_row = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save / Update")
        self.save_btn.clicked.connect(self._save_current)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.refresh_btn)
        save_row.addStretch()
        save_row.addWidget(close_btn)
        right_lay.addLayout(save_row)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, stretch=1)
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)
        self._refresh()

    def _refresh(self):
        self.skill_list.clear()
        for row in self._skills.list_skills():
            item = QtWidgets.QListWidgetItem(f"{row['name']}  ({row['chars']} chars)")
            item.setData(QtCore.Qt.UserRole, row["name"])
            item.setToolTip(str(row.get("path", "")))
            self.skill_list.addItem(item)
        self.status_lbl.setText(f"Skill directory: {self._skills.SKILL_DIR}")

    def _current_name(self):
        item = self.skill_list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else ""

    def _load_selected(self):
        name = self._current_name()
        if not name:
            return
        result = self._skills.read_skill(name)
        if result.get("error"):
            self.status_lbl.setText(result["error"])
            return
        self.name_edit.setText(result["name"])
        self.content_edit.setPlainText(result["content"])
        self.status_lbl.setText(str(result["path"]))

    def _save_current(self):
        name = self.name_edit.text().strip()
        content = self.content_edit.toPlainText().strip()
        result = self._skills.install_skill(name=name, content=content)
        self.status_lbl.setText(result.get("error") or f"Saved: {result.get('path')}")
        self._refresh()

    def _install_file(self):
        path, _ = light_get_open_file_name(
            self,
            "Install Agent Skill",
            "",
            "Markdown (*.md);;Text (*.txt);;All files (*)",
        )
        if not path:
            return
        name = Path(path).stem
        result = self._skills.install_skill(name=name, source_path=path)
        self.status_lbl.setText(result.get("error") or f"Installed: {result.get('path')}")
        self._refresh()

    def _delete_selected(self):
        name = self._current_name()
        if not name:
            return
        confirm = QtWidgets.QMessageBox.question(
            self, "Delete skill", f"Delete skill '{name}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        result = self._skills.delete_skill(name)
        self.status_lbl.setText(result.get("error") or f"Deleted: {name}")
        self.name_edit.clear()
        self.content_edit.clear()
        self._refresh()


# Shared by the file tree's icon provider (_AgentFileIconProvider) and
# _agent_entry_html's "file_link" chat cards -- one mapping so both surfaces
# agree on what a given file type looks like.
#
# The file tree prefers real, colored mimetype icons from Ubuntu's
# "Humanity" icon theme (already installed system-wide at
# /usr/share/icons/Humanity -- this is the same icon set a normal desktop
# file manager uses, so a .pdf/.png/.csv/... look like the familiar
# official app icons instead of a generic glyph). Qt's own default
# QFileIconProvider can't be trusted to find these on its own here (no
# desktop icon theme is configured for this bare WSL Qt install), so the
# theme's SVGs are loaded by explicit path. Falls back to a drawn emoji
# glyph (still distinct per type, just not the "real" icon) if that theme
# ever isn't present -- e.g. a different machine -- so the tree never
# regresses to Qt's identical-for-everything default.
_AGENT_ICON_THEME_DIR = Path("/usr/share/icons/Humanity/mimes/48")
_AGENT_FILE_TYPE_ICON_FILE = {
    ".pdf": "application-pdf.svg",
    ".png": "image-png.svg", ".jpg": "image-jpeg.svg", ".jpeg": "image-jpeg.svg",
    ".gif": "image-gif.svg", ".bmp": "image-bmp.svg", ".svg": "svg.svg", ".webp": "image-x-generic.svg",
    ".csv": "application-vnd.ms-excel.svg", ".tsv": "application-vnd.ms-excel.svg",
    ".xlsx": "application-vnd.openxmlformats-officedocument.spreadsheetml.sheet.svg",
    ".xls": "application-vnd.ms-excel.svg",
    ".py": "text-x-python.svg",
    ".json": "text-x-generic.svg",
    ".md": "text-x-generic.svg", ".txt": "text-plain.svg", ".log": "text-plain.svg",
    ".mgf": "application-octet-stream.svg", ".mzml": "application-octet-stream.svg",
    ".mzxml": "application-octet-stream.svg", ".hdf5": "application-octet-stream.svg", ".h5": "application-octet-stream.svg",
    ".zip": "application-zip.svg", ".tar": "application-x-compressed-tar.svg", ".gz": "application-x-gzip.svg",
    ".yaml": "text-x-generic.svg", ".yml": "text-x-generic.svg", ".ini": "text-x-generic.svg",
    ".cfg": "text-x-generic.svg", ".toml": "text-x-generic.svg",
    ".sh": "application-x-shellscript.svg", ".js": "application-x-javascript.svg", ".ts": "text-x-javascript.svg",
    ".html": "text-html.svg", ".css": "text-css.svg",
}
_AGENT_FILE_DEFAULT_ICON_FILE = "text-x-generic.svg"

# Emoji fallback -- same extensions, used only if _AGENT_ICON_THEME_DIR
# (or the specific file) doesn't exist on the machine running this.
_AGENT_FILE_TYPE_EMOJI = {
    ".pdf": "📕",
    ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".gif": "🖼", ".bmp": "🖼", ".svg": "🖼", ".webp": "🖼",
    ".csv": "📊", ".tsv": "📊", ".xlsx": "📊", ".xls": "📊",
    ".py": "🐍",
    ".json": "🧩",
    ".md": "📝", ".txt": "📝", ".log": "📝",
    ".mgf": "🧪", ".mzml": "🧪", ".mzxml": "🧪", ".hdf5": "🧪", ".h5": "🧪",
    ".zip": "🗜", ".tar": "🗜", ".gz": "🗜",
    ".yaml": "⚙", ".yml": "⚙", ".ini": "⚙", ".cfg": "⚙", ".toml": "⚙",
    ".sh": "💻", ".js": "💻", ".ts": "💻", ".html": "💻", ".css": "💻",
}
_AGENT_FILE_DEFAULT_EMOJI = "📄"


def _agent_file_type_emoji(path):
    return _AGENT_FILE_TYPE_EMOJI.get(Path(path).suffix.lower(), _AGENT_FILE_DEFAULT_EMOJI)


class _AgentFileIconProvider(QtWidgets.QFileIconProvider):
    """Real, colored per-type mimetype icons (see _AGENT_FILE_TYPE_ICON_FILE
    above) for the Agent tab's file tree, with a drawn-emoji fallback."""

    def __init__(self):
        super().__init__()
        self._cache = {}

    def icon(self, arg):
        if isinstance(arg, QtCore.QFileInfo):
            if arg.isDir():
                return super().icon(arg)
            return self._icon_for(arg.filePath())
        return super().icon(arg)

    def _icon_for(self, path):
        ext = Path(path).suffix.lower()
        icon_file = _AGENT_FILE_TYPE_ICON_FILE.get(ext, _AGENT_FILE_DEFAULT_ICON_FILE)
        cached = self._cache.get(icon_file)
        if cached is not None:
            return cached
        theme_path = _AGENT_ICON_THEME_DIR / icon_file
        icon = QtGui.QIcon(str(theme_path)) if theme_path.is_file() else None
        if icon is None or icon.isNull():
            icon = self._emoji_icon(_AGENT_FILE_TYPE_EMOJI.get(ext, _AGENT_FILE_DEFAULT_EMOJI))
        self._cache[icon_file] = icon
        return icon

    def _emoji_icon(self, emoji, size=18):
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(int(size * 0.62))
        painter.setFont(font)
        painter.drawText(pix.rect(), QtCore.Qt.AlignCenter, emoji)
        painter.end()
        return QtGui.QIcon(pix)


def _agent_html_escape(text):
    import html
    return html.escape(str(text)).replace("\n", "<br>")


def _agent_compact_display_text(text):
    """Normalize model text for Qt chat rendering only.

    LLMs sometimes emit many blank lines around SMILES, names, and file paths.
    QTextDocument.setMarkdown renders each blank run as visible paragraph
    space, which makes the Agent panel look broken. Keep intentional line
    breaks, but cap blank runs to one empty line.
    """
    import re
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _agent_is_compact_literal_line(text):
    """Detect one-line technical values that should not become big prose."""
    value = str(text or "").strip()
    if not value or "\n" in value or len(value) > 220:
        return False
    if value.startswith(("/", "~")) and ("/" in value or "." in value):
        return True
    if any(ch in value for ch in ("=", "#", "@")) and any(ch.isdigit() for ch in value):
        return True
    smiles_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#@+\\-/%.")
    return (
        len(value) >= 18
        and set(value) <= smiles_chars
        and any(ch in value for ch in "()=#")
        and any(ch.isdigit() for ch in value)
    )


def _agent_markdown_to_html(text):
    """Renders markdown (headers/bold/italic/lists/tables/code) via Qt's own
    QTextDocument.setMarkdown() -- no external 'markdown' package needed,
    PyQt5 5.14+ ships this natively. Returns just the <body> inner content
    so it can be inserted as a fragment into the chat log's own document
    instead of replacing it wholesale."""
    import re
    from PyQt5 import QtGui
    doc = QtGui.QTextDocument()
    doc.setDefaultStyleSheet("""
        p { margin-top: 2px; margin-bottom: 5px; }
        ul, ol { margin-top: 3px; margin-bottom: 5px; }
        li { margin-bottom: 2px; }
        pre { margin-top: 4px; margin-bottom: 6px; }
    """)
    display_text = _agent_compact_display_text(text)
    if _agent_is_compact_literal_line(display_text):
        return f"<p><code>{_agent_html_escape(display_text)}</code></p>"
    doc.setMarkdown(display_text)
    html = doc.toHtml()
    m = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL)
    return m.group(1) if m else _agent_html_escape(text)


def _theme_accent(theme_name):
    if theme_name == "Sky":
        return "#4A7FE8"
    return CODEX_THEME_ACCENTS.get(theme_name or "Graphite Dark", CODEX_THEME_ACCENTS["Graphite Dark"])[0]


def _theme_is_light(theme_name):
    return theme_name in ("Sky", "Light") or str(theme_name or "").endswith(" Light")


def _light_theme_palette(theme_name):
    if theme_name == "Sky":
        return dict(
            bg="#F3F5F8", surface="#FFFFFF", panel="#EEF3FF", soft="#E7EFFF",
            border="#D9E4F7", text="#1F2430", muted="#6B7280",
            accent="#4A7FE8", hover="#3F6FD1", pressed="#35609C",
        )
    return PASTEL_THEME_PALETTES.get(theme_name, dict(
        bg="#F8FAFC", surface="#FFFFFF", panel="#F1F5F9", soft="#E2E8F0",
        border="#CBD5E1", text="#1F2937", muted="#64748B",
        accent=_theme_accent(theme_name), hover="#334155", pressed="#1E293B",
    ))


def _theme_muted(theme_name):
    """Secondary-text colour with adequate contrast on the active theme.

    Used for section-title labels so they stay readable (and slightly de-
    emphasised) on both light and dark themes, instead of a hardcoded #555
    that vanishes on dark backgrounds."""
    if _theme_is_light(theme_name):
        return _light_theme_palette(theme_name)["muted"]
    return "#A1A1AA"


def _contrast_text(hex_color):
    color = str(hex_color or "#FAFAFA").lstrip("#")
    if len(color) != 6:
        return "#09090B"
    try:
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    except Exception:
        return "#09090B"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#09090B" if luminance > 0.62 else "#FFFFFF"


def _agent_entry_html(kind, text, session_idx=None, tool_log_id=None, theme_name=None):
    """One chat-log entry's HTML, shared by _agent_append_display (paint as
    it happens) and _agent_render_session (repaint from history on session
    switch) so the two never drift out of sync."""
    display_text = _agent_compact_display_text(text)
    accent = _theme_accent(theme_name)
    is_light = _theme_is_light(theme_name)
    if is_light:
        p = _light_theme_palette(theme_name)
        accent = p["accent"]
        text_color = p["text"]
        muted = p["muted"]
        chip_bg = p["panel"]
        chip_border = p["border"]
        file_bg = "rgba(255,255,255,0.70)"
        file_border = p["border"]
        user_bg = p["soft"]
        user_text = p["pressed"]
    else:
        text_color = "#E4E4E7"
        muted = "#A1A1AA"
        chip_bg = "#18181B"
        chip_border = "#27272A"
        file_bg = "#141417"
        file_border = "#27272A"
        user_bg = accent
        user_text = _contrast_text(accent)

    def row(inner_html, align="left", max_width="76%"):
        # Qt's rich-text engine doesn't honor CSS "display:inline-block" +
        # "max-width" on a <div> (silently falls back to full block width),
        # nor a percentage "width" on a <table> nested inside another
        # table's <td> (resolves against the unconstrained outer cell, so
        # it ends up full-width too). A single top-level <table> using its
        # own HTML "align"/"width" attributes is what Qt's renderer actually
        # honors, so use that directly instead of nesting/CSS.
        align_attr = "right" if align == "right" else "left"
        return (
            f"<table width='{max_width}' align='{align_attr}' cellspacing='0' cellpadding='0' "
            "style='margin:8px 0;'><tr><td align='left'>"
            f"{inner_html}"
            "</td></tr></table>"
        )

    def label(text, color=None):
        color = color or muted
        return f"<div style='font-size:10px; color:{color}; margin:0 0 5px 1px; font-weight:600; letter-spacing:0.2px;'>{text}</div>"

    if kind == "user":
        body = (
            f"<div style='background:{user_bg}; color:{user_text}; "
            "border-radius:10px; padding:7px 11px; line-height:1.42; font-weight:500;'>"
            f"{_agent_html_escape(display_text)}</div>"
        )
        return row(body, align="right", max_width="58%")
    if kind == "agent":
        # Flat, borderless text (assistant turns read as
        # part of the document, not a boxed "chat bubble" card) -- only the
        # small label above anchors where the turn starts.
        body = (
            f"<div style='color:{text_color}; padding:2px 2px 0 2px; line-height:1.48;'>"
            f"{_agent_markdown_to_html(display_text)}</div>"
        )
        return row(body, align="left", max_width="96%")
    if kind == "tool":
        # Checklist-style tool-call trace (see core/agent_tools.py's
        # run_agent_loop): "✓ name (Ns)" on success, "✗ name (Ns) -- error:
        # ..." on failure, anything else (e.g. "Compacting conversation
        # history...") is a neutral status line, each rendered as a small
        # pill/chip. When a tool_log_id is attached, the chip is a clickable
        # link -- the "Agent execution timeline" detail view
        # (_agent_show_tool_detail) shows that call's full input/output
        # JSON on click.
        color = "#22C55E" if display_text.startswith("✓") else ("#F87171" if display_text.startswith("✗") else muted)
        chip = f"display:inline-block; background:{chip_bg}; border:1px solid {chip_border}; border-radius:999px; padding:2px 10px; font-size:10px; color:{color};"
        if tool_log_id is not None and session_idx is not None:
            url = f"agenttool://{session_idx}/{tool_log_id}"
            return (
                "<div style='text-align:left; margin:2px 0 2px 4px;'>"
                f"<div style='{chip}'><a href=\"{url}\" style='color:{color}; text-decoration:none;'>"
                f"{_agent_html_escape(display_text)} details</a></div></div>"
            )
        if display_text.startswith("✓") or display_text.startswith("✗"):
            return f"<div style='text-align:left; margin:2px 0 2px 4px;'><div style='{chip}'>{_agent_html_escape(display_text)}</div></div>"
        return f"<div style='text-align:left; margin:2px 0 2px 4px;'><div style='{chip}'><i>{_agent_html_escape(display_text)}</i></div></div>"
    if kind == "file_link":
        # text is a local file path -- rendered as a clickable link that
        # opens in the embedded preview panel (_agent_chat_link_clicked),
        # not a new window.
        from PyQt5.QtCore import QUrl
        p = Path(text)
        url = QUrl.fromLocalFile(str(p)).toString()
        icon = _agent_file_type_emoji(p)
        return row(
            f"<div style='background:{file_bg}; border:1px solid {file_border}; border-radius:10px; "
            f"padding:7px 10px; color:{text_color};'>{icon} <a href=\"{url}\" style='color:{accent}; text-decoration:none;'>{_agent_html_escape(p.name)}</a></div>",
            align="left",
            max_width="58%",
        )
    return row(f"<i>{_agent_html_escape(text)}</i>", align="left")


def _agent_fill_provider_label(self):
    from core import agent_config as ac
    from core.i18n import tr
    cfg = ac.load_config()
    key, provider_cfg = ac.get_active_provider_config(cfg)
    if key:
        label = ac.PROVIDER_PRESETS.get(key, {}).get("label", key)
        self._agent_provider_lbl.setText(tr("agent.provider_label", label=label, model=provider_cfg.get('model', '')))
        ready = bool(provider_cfg.get("api_key"))
    else:
        self._agent_provider_lbl.setText(tr("agent.provider_not_configured"))
        ready = False
    if hasattr(self, "_agent_send_btn"):
        self._agent_send_btn.setEnabled(ready)
    return ready


def _new_agent_session(name):
    # display_log: [(kind, text, tool_log_id_or_None), ...]
    # tool_log: [{"id","name","args","result","duration_s","status"}, ...] --
    # the "Agent execution timeline" detail behind each clickable "✓ name"
    # chat line (see _agent_show_tool_detail).
    # pinned/created_at: session-list grouping (see _agent_rebuild_session_list).
    now = time.time()
    return {"name": name, "messages": [], "display_log": [], "tool_log": [],
            "last_usage": None,
            "pinned": False, "created_at": now, "updated_at": now}


def _agent_sessions_path():
    return Path.home() / ".annomass" / "agent_sessions.json"


def _agent_sanitize_session(session, fallback_name):
    now = time.time()
    if not isinstance(session, dict):
        return _new_agent_session(fallback_name)
    clean = _new_agent_session(str(session.get("name") or fallback_name))
    clean["messages"] = session.get("messages") if isinstance(session.get("messages"), list) else []
    clean["display_log"] = session.get("display_log") if isinstance(session.get("display_log"), list) else []
    clean["tool_log"] = session.get("tool_log") if isinstance(session.get("tool_log"), list) else []
    clean["last_usage"] = session.get("last_usage") if isinstance(session.get("last_usage"), dict) else None
    clean["pinned"] = bool(session.get("pinned"))
    clean["created_at"] = float(session.get("created_at") or now)
    clean["updated_at"] = float(session.get("updated_at") or clean["created_at"])
    return clean


def _agent_load_sessions(self):
    from core.i18n import tr
    import json
    path = _agent_sessions_path()
    if not path.exists():
        return [_new_agent_session(tr("agent.chat_default_name", n=1))], 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_sessions = data.get("sessions", [])
        sessions = [
            _agent_sanitize_session(s, tr("agent.chat_default_name", n=i + 1))
            for i, s in enumerate(raw_sessions)
            if isinstance(s, dict)
        ]
        if not sessions:
            raise ValueError("No saved sessions.")
        current = int(data.get("current_session", 0))
        current = max(0, min(current, len(sessions) - 1))
        return sessions, current
    except Exception:
        return [_new_agent_session(tr("agent.chat_default_name", n=1))], 0


def _agent_save_sessions(self):
    import json
    path = _agent_sessions_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "current_session": int(getattr(self, "_agent_current_session", 0)),
            "sessions": getattr(self, "_agent_sessions", []),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        # Persistence is helpful, but a failed save must not break chat.
        pass


AGENT_PANEL_STYLE = """
    #agent_panel { background: #09090B; }

    #agent_panel QWidget#agent_toolbar {
        background: #0F0F12; border-bottom: 1px solid #27272A;
    }
    #agent_panel QWidget#agent_toolbar QLabel { color: #A1A1AA; font-size: 12px; }
    #agent_panel QLabel#agent_provider_lbl { color: #FAFAFA; font-weight: 600; }

    #agent_panel QWidget#agent_sidebar {
        background: #0F0F12; border: 1px solid #27272A; border-radius: 10px;
    }

    #agent_panel QWidget#agent_composer_toolbar QCheckBox { color: #A1A1AA; font-size: 11px; }

    /* Default: flat "ghost" buttons -- most Agent-tab actions (Skills,
    Settings, Retry, Copy Chat, Export, Clear Chat, ...) are secondary,
    so only [variant="primary"]/[variant="danger"] below stand out. */
    #agent_panel QPushButton {
        background: transparent; border: 1px solid transparent;
        color: #D4D4D8; padding: 5px 12px; border-radius: 7px;
        font-weight: 500; font-size: 12px;
    }
    #agent_panel QPushButton:hover { background: #18181B; color: #FAFAFA; border-color: #27272A; }
    #agent_panel QPushButton:pressed { background: #27272A; }
    #agent_panel QPushButton:disabled { color: #52525B; }

    #agent_panel QPushButton[variant="primary"] {
        background: #FAFAFA; border: 1px solid #FAFAFA;
        color: #09090B; font-weight: 600; padding: 6px 18px; border-radius: 8px;
    }
    #agent_panel QPushButton[variant="primary"]:hover { background: #E4E4E7; border-color: #E4E4E7; }
    #agent_panel QPushButton[variant="primary"]:pressed { background: #D4D4D8; border-color: #D4D4D8; }
    #agent_panel QPushButton[variant="primary"]:disabled { background: #27272A; color: #71717A; border-color: #27272A; }

    #agent_panel QPushButton[variant="danger"] { color: #F87171; }
    #agent_panel QPushButton[variant="danger"]:hover { background: #2A1215; color: #FCA5A5; border-color: #3F1D22; }
    #agent_panel QPushButton[variant="danger"]:disabled { color: #C2C7D0; background: transparent; }

    #agent_panel QLineEdit {
        background: #141417; border: 1px solid #27272A; border-radius: 8px;
        padding: 5px 10px; color: #FAFAFA; font-size: 12px;
    }
    #agent_panel QLineEdit:focus { border-color: #71717A; }

    #agent_panel QTreeView {
        background: transparent; border: none; outline: none; color: #D4D4D8;
    }
    #agent_panel QTreeView::item { padding: 3px 2px; border-radius: 6px; }
    #agent_panel QTreeView::item:selected { background: #27272A; color: #FAFAFA; }
    #agent_panel QTreeView::item:hover { background: #18181B; }
    #agent_panel QListWidget#agent_session_list {
        background: transparent; border: none; outline: none; padding: 2px;
    }
    #agent_panel QListWidget#agent_session_list::item {
        padding: 6px 10px; margin: 1px 2px; border-radius: 7px; color: #D4D4D8;
    }
    #agent_panel QListWidget#agent_session_list::item:selected { background: #27272A; color: #FAFAFA; }
    #agent_panel QListWidget#agent_session_list::item:hover:!disabled { background: #18181B; }
    #agent_panel QListWidget#agent_session_list::item:disabled {
        color: #71717A; font-size: 10px; font-weight: 700;
        padding-top: 10px; padding-bottom: 2px;
    }
    #agent_panel QPlainTextEdit {
        background: #141417; border: 1px solid #27272A; border-radius: 10px;
        color: #FAFAFA; selection-background-color: #3F3F46; padding: 6px 8px;
    }
    #agent_panel QPlainTextEdit:focus { border-color: #71717A; }
    #agent_panel QLabel { color: #D4D4D8; }
    #agent_panel QSplitter::handle { background: #27272A; }
"""


SKY_AGENT_PANEL_STYLE = """
    #agent_panel { background: rgba(243,245,248,230); }
    #agent_panel QWidget#agent_toolbar {
        background: rgba(255,255,255,0.52); border-bottom: 1px solid rgba(226,232,240,0.58);
    }
    #agent_panel QWidget#agent_toolbar QLabel { color: #6B7280; font-size: 12px; }
    #agent_panel QLabel#agent_provider_lbl { color: #1F2430; font-weight: 600; }
    #agent_panel QWidget#agent_sidebar {
        background: rgba(255,255,255,0.42);
        border: 1px solid rgba(226,232,240,0.55);
        border-radius: 12px;
    }
    #agent_panel QWidget#agent_composer_toolbar QCheckBox { color: #8A8F98; font-size: 11px; }
    #agent_panel QPushButton {
        background: transparent; border: 1px solid transparent;
        color: #4B5563; padding: 5px 12px; border-radius: 8px;
        font-weight: 500; font-size: 12px;
    }
    #agent_panel QPushButton:hover { background: #F0F2F5; color: #1F2430; }
    #agent_panel QPushButton:pressed { background: #E5E8EF; }
    #agent_panel QPushButton:checked { background: #E7EFFF; color: #2A56C6; border-color: #D9E4F7; }
    #agent_panel QPushButton:disabled { color: #C2C7D0; }
    #agent_panel QPushButton[variant="primary"] {
        background: #4A7FE8; border: 1px solid #4A7FE8;
        color: #FFFFFF; font-weight: 600; padding: 6px 18px; border-radius: 16px;
    }
    #agent_panel QPushButton[variant="primary"]:hover { background: #3F6FD1; border-color: #3F6FD1; }
    #agent_panel QPushButton[variant="primary"]:pressed { background: #35609C; border-color: #35609C; }
    #agent_panel QPushButton[variant="primary"]:disabled { background: #E5E8EF; color: #AEB3BD; border-color: #E5E8EF; }
    #agent_panel QPushButton[variant="danger"] { color: #DC2626; }
    #agent_panel QPushButton[variant="danger"]:hover { background: #FEF2F2; color: #B91C1C; }
    #agent_panel QPushButton[variant="danger"]:disabled { color: #C2C7D0; background: transparent; }
    #agent_panel QLineEdit {
        background: rgba(255,255,255,0.68); border: 1px solid rgba(229,232,239,0.72); border-radius: 10px;
        padding: 5px 10px; color: #1F2430; font-size: 12px;
    }
    #agent_panel QLineEdit:focus { border-color: #4A7FE8; }
    #agent_panel QTreeView {
        background: transparent; border: none; outline: none; color: #1F2430;
    }
    #agent_panel QTreeView::item { padding: 3px 2px; border-radius: 6px; }
    #agent_panel QTreeView::item:selected { background: rgba(74,127,232,0.16); color: #2A56C6; }
    #agent_panel QTreeView::item:hover { background: rgba(74,127,232,0.08); }
    #agent_panel QListWidget#agent_session_list {
        background: transparent; border: none; outline: none; padding: 2px;
    }
    #agent_panel QListWidget#agent_session_list::item {
        padding: 6px 10px; margin: 1px 2px; border-radius: 8px; color: #1F2430;
    }
    #agent_panel QListWidget#agent_session_list::item:selected { background: #E7EFFF; color: #2A56C6; }
    #agent_panel QListWidget#agent_session_list::item:hover:!disabled { background: #EEF3FF; }
    #agent_panel QListWidget#agent_session_list::item:disabled {
        color: #8A8F98; font-size: 10px; font-weight: 700;
        padding-top: 10px; padding-bottom: 2px;
    }
    #agent_panel QPlainTextEdit {
        background: rgba(255,255,255,0.70); border: 1px solid rgba(229,232,239,0.72); border-radius: 10px;
        color: #1F2430; selection-background-color: #CBDCFF; padding: 6px 8px;
    }
    #agent_panel QPlainTextEdit:focus { border-color: #4A7FE8; }
    #agent_panel QLabel { color: #1F2430; }
    #agent_panel QSplitter::handle { background: #ECEEF2; }
"""


def _agent_panel_style_for_theme(theme_name):
    if theme_name == "Sky":
        return SKY_AGENT_PANEL_STYLE
    accent = _theme_accent(theme_name)
    hover = CODEX_THEME_ACCENTS.get(theme_name or "", ("#FAFAFA", "#E4E4E7", "#D4D4D8", "#09090B"))[1]
    if _theme_is_light(theme_name):
        p = _light_theme_palette(theme_name)
        return SKY_AGENT_PANEL_STYLE + f"""
            #agent_panel {{ background: {p['bg']}; }}
            #agent_panel QWidget#agent_toolbar {{
                background: rgba(255,255,255,0.48); border-bottom-color: {p['border']};
            }}
            #agent_panel QWidget#agent_toolbar QLabel {{ color: {p['muted']}; }}
            #agent_panel QLabel#agent_provider_lbl {{ color: {p['text']}; }}
            #agent_panel QWidget#agent_sidebar {{
                background: rgba(255,255,255,0.38); border-color: {p['border']};
            }}
            #agent_panel QPushButton {{
                color: {p['accent']};
            }}
            #agent_panel QPushButton:hover {{
                background: {p['soft']}; color: {p['hover']}; border-color: {p['border']};
            }}
            #agent_panel QPushButton[variant="primary"] {{
                background: {p['panel']}; border-color: {p['border']}; color: {p['accent']}; border-radius: 8px;
            }}
            #agent_panel QPushButton[variant="primary"]:hover {{ background: {p['soft']}; border-color: {p['accent']}; color: {p['hover']}; }}
            #agent_panel QPushButton[variant="primary"]:pressed {{ background: {p['accent']}; border-color: {p['accent']}; color: #FFFFFF; }}
            #agent_panel QPushButton:checked {{
                background: {p['soft']}; border-color: {p['border']}; color: {p['pressed']};
            }}
            #agent_panel QLineEdit, #agent_panel QPlainTextEdit {{
                background: rgba(255,255,255,0.62); border-color: {p['border']}; color: {p['text']};
            }}
            #agent_panel QLineEdit:focus, #agent_panel QPlainTextEdit:focus {{ border-color: {p['accent']}; }}
            #agent_panel QLabel {{ color: {p['text']}; }}
            #agent_panel QTreeView {{ color: {p['text']}; }}
            #agent_panel QTreeView::item:selected {{
                background: {p['soft']}; color: {p['accent']};
            }}
            #agent_panel QTreeView::item:hover {{ background: {p['panel']}; }}
            #agent_panel QListWidget#agent_session_list::item {{ color: {p['text']}; }}
            #agent_panel QListWidget#agent_session_list::item:selected {{
                background: {p['soft']}; color: {p['accent']};
            }}
            #agent_panel QListWidget#agent_session_list::item:hover:!disabled {{
                background: {p['panel']};
            }}
            #agent_panel QListWidget#agent_session_list::item:disabled {{
                color: {p['muted']};
            }}
        """
    return AGENT_PANEL_STYLE + f"""
        #agent_panel QPushButton[variant="primary"] {{
            background: {accent}; border-color: {accent}; color: {_contrast_text(accent)};
        }}
        #agent_panel QPushButton[variant="primary"]:hover {{
            background: {hover}; border-color: {hover};
        }}
        #agent_panel QLineEdit:focus, #agent_panel QPlainTextEdit:focus {{ border-color: {accent}; }}
        #agent_panel QPushButton:checked {{
            background: rgba(255,255,255,10); border-color: #3F3F46; color: {accent};
        }}
        #agent_panel QListWidget#agent_session_list::item:selected {{
            background: rgba(255,255,255,8); color: {accent};
        }}
        #agent_panel QTreeView::item:selected {{
            background: rgba(255,255,255,8); color: {accent};
        }}
    """


def _agent_chat_document_style_for_theme(theme_name):
    accent = _theme_accent(theme_name)
    if _theme_is_light(theme_name):
        p = _light_theme_palette(theme_name)
        accent = p["accent"]
        return f"""
            body {{
                background: transparent;
                color: {p['text']};
                font-family: "Segoe UI", "DejaVu Sans", Arial;
                font-size: 12px;
                line-height: 1.42;
            }}
            p {{ margin: 2px 0 6px 0; }}
            h1, h2, h3 {{ margin: 8px 0 5px 0; font-weight: 600; }}
            a {{ color: {accent}; text-decoration: none; }}
            pre {{
                background: #0F172A; color: #E5E7EB; border-radius: 6px;
                padding: 8px; font-family: "DejaVu Sans Mono", "Consolas", monospace;
                font-size: 11px; white-space: pre-wrap;
                margin: 5px 0 7px 0;
            }}
            code {{
                background: {p['panel']}; color: {p['text']}; border-radius: 4px; padding: 1px 4px;
                font-family: "DejaVu Sans Mono", "Consolas", monospace; font-size: 11px;
            }}
            ul, ol {{ margin-top: 3px; margin-bottom: 6px; }}
            li {{ margin-bottom: 2px; }}
            table {{ border-collapse: collapse; margin: 6px 0; }}
            th, td {{ border: 1px solid {p['border']}; padding: 3px 6px; }}
        """
    return f"""
        body {{
            background: transparent;
            color: #E4E4E7;
            font-family: "Segoe UI", "DejaVu Sans", Arial;
            font-size: 12px;
            line-height: 1.42;
        }}
        p {{ margin: 2px 0 6px 0; }}
        h1, h2, h3 {{ margin: 8px 0 5px 0; font-weight: 600; }}
        a {{ color: {accent}; text-decoration: none; }}
        pre {{
            background: #141417; color: #E4E4E7; border: 1px solid #27272A;
            border-radius: 6px; padding: 8px; margin: 5px 0 7px 0;
            font-family: "DejaVu Sans Mono", "Consolas", monospace;
            font-size: 11px; white-space: pre-wrap;
        }}
        code {{
            background: #18181B; color: #FAFAFA; border-radius: 4px; padding: 1px 4px;
            font-family: "DejaVu Sans Mono", "Consolas", monospace; font-size: 11px;
        }}
        ul, ol {{ margin-top: 3px; margin-bottom: 6px; }}
        li {{ margin-bottom: 2px; }}
        table {{ border-collapse: collapse; margin: 6px 0; }}
        th, td {{ border: 1px solid #27272A; padding: 3px 6px; }}
    """


def _agent_chat_browser_style_for_theme(theme_name):
    if _theme_is_light(theme_name):
        return (
            "QTextBrowser { background: rgba(255,255,255,0.58); "
            "border: 1px solid rgba(226,232,240,0.58); "
            "border-radius: 10px; padding: 14px 18px; }"
        )
    return (
        "QTextBrowser { background: #0C0C0F; border: 1px solid #27272A; "
        "border-radius: 10px; padding: 14px 18px; }"
    )


class _AgentPtyTerminal(QtWidgets.QPlainTextEdit):
    sendBytes = QtCore.pyqtSignal(bytes)

    _KEY_BYTES = {
        QtCore.Qt.Key_Return: b"\r",
        QtCore.Qt.Key_Enter: b"\r",
        QtCore.Qt.Key_Backspace: b"\x7f",
        QtCore.Qt.Key_Tab: b"\t",
        QtCore.Qt.Key_Left: b"\x1b[D",
        QtCore.Qt.Key_Right: b"\x1b[C",
        QtCore.Qt.Key_Up: b"\x1b[A",
        QtCore.Qt.Key_Down: b"\x1b[B",
        QtCore.Qt.Key_Home: b"\x1b[H",
        QtCore.Qt.Key_End: b"\x1b[F",
        QtCore.Qt.Key_Delete: b"\x1b[3~",
        QtCore.Qt.Key_PageUp: b"\x1b[5~",
        QtCore.Qt.Key_PageDown: b"\x1b[6~",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setUndoRedoEnabled(False)
        self.setTabChangesFocus(False)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        font = QtGui.QFont("DejaVu Sans Mono")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(10)
        self.setFont(font)

    def keyPressEvent(self, event):
        mods = event.modifiers()
        key = event.key()
        if mods & QtCore.Qt.ControlModifier:
            if key == QtCore.Qt.Key_C:
                self.sendBytes.emit(b"\x03")
                return
            if key == QtCore.Qt.Key_D:
                self.sendBytes.emit(b"\x04")
                return
            if key == QtCore.Qt.Key_L:
                self.clear()
                self.sendBytes.emit(b"\x0c")
                return
        payload = self._KEY_BYTES.get(key)
        if payload is not None:
            self.sendBytes.emit(payload)
            return
        text = event.text()
        if text:
            self.sendBytes.emit(text.encode("utf-8", errors="ignore"))
            return
        super().keyPressEvent(event)

    def insert_terminal_text(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.setTextCursor(cursor)
        self.insertPlainText(text)
        self.moveCursor(QtGui.QTextCursor.End)


def _init_agent_tab(self):
    """Builds the Flavonoid Agent panel as a STANDALONE sibling of
    tabWidget_results (not one of its tabs) inside resultsPanel/vl_results,
    plus the Manual/Agent mode switch above both. This lets Agent Mode hide
    the entire tab bar while the agent's own UI stays visible -- if the
    agent were just another tab, hiding tabWidget_results would hide it too."""
    from core.i18n import tr, get_language, set_language, LANG_EN, LANG_ZH

    results_panel = self.findChild(QtWidgets.QWidget, "resultsPanel")
    vl_results = getattr(self, "vl_results", None)
    tw = self.findChild(QtWidgets.QTabWidget, "tabWidget_results")
    if results_panel is None or vl_results is None or tw is None:
        return

    # ── Mode switch (Manual / Agent), sits above both panels ────────────────
    mode_row = QtWidgets.QWidget(results_panel)
    mode_hlay = QtWidgets.QHBoxLayout(mode_row)
    mode_hlay.setContentsMargins(6, 4, 6, 4)
    mode_hlay.addWidget(QtWidgets.QLabel(tr("agent.mode_label")))
    self._mode_manual_radio = QtWidgets.QRadioButton(tr("agent.mode_manual"))
    self._mode_agent_radio = QtWidgets.QRadioButton(tr("agent.mode_agent"))
    self._mode_manual_radio.setChecked(True)
    mode_hlay.addWidget(self._mode_manual_radio)
    mode_hlay.addWidget(self._mode_agent_radio)
    mode_hlay.addStretch()
    vl_results.insertWidget(0, mode_row)

    # ── Standalone Agent panel ───────────────────────────────────────────────
    page = QtWidgets.QWidget(results_panel)
    page.setObjectName("agent_panel")
    page.setStyleSheet(_agent_panel_style_for_theme(getattr(self, "_current_theme_name", "Graphite Dark")))
    vlay = QtWidgets.QVBoxLayout(page)
    vlay.setContentsMargins(0, 0, 0, 0)
    vlay.setSpacing(0)

    toolbar = QtWidgets.QWidget()
    toolbar.setObjectName("agent_toolbar")
    top_row = QtWidgets.QHBoxLayout(toolbar)
    top_row.setContentsMargins(10, 8, 8, 8)
    top_row.setSpacing(6)
    self._agent_focus_btn = QtWidgets.QPushButton("☰")
    self._agent_focus_btn.setCheckable(True)
    self._agent_focus_btn.setFixedWidth(34)
    self._agent_focus_btn.setToolTip("Focus mode: hide side panels and keep only the chat column.")
    self._agent_focus_btn.toggled.connect(lambda checked: self._agent_set_focus_mode(checked))
    top_row.addWidget(self._agent_focus_btn)

    self._agent_right_panel_btn = QtWidgets.QPushButton("◨")
    self._agent_right_panel_btn.setCheckable(True)
    self._agent_right_panel_btn.setChecked(True)
    self._agent_right_panel_btn.setFixedWidth(34)
    self._agent_right_panel_btn.setToolTip("Show or hide the right preview/terminal panel.")
    self._agent_right_panel_btn.toggled.connect(lambda checked: self._agent_toggle_right_panel(checked))
    top_row.addWidget(self._agent_right_panel_btn)

    self._agent_provider_lbl = QtWidgets.QLabel(tr("agent.provider_not_configured"))
    self._agent_provider_lbl.setObjectName("agent_provider_lbl")
    top_row.addWidget(self._agent_provider_lbl, stretch=1)

    self._agent_context_meter = _AgentContextMeter()
    top_row.addWidget(self._agent_context_meter)

    top_row.addWidget(QtWidgets.QLabel(tr("agent.language_label")))
    self._agent_lang_combo = QtWidgets.QComboBox()
    self._agent_lang_combo.addItem("English", LANG_EN)
    self._agent_lang_combo.addItem("中文", LANG_ZH)
    self._agent_lang_combo.setCurrentIndex(self._agent_lang_combo.findData(get_language()))

    def _on_lang_changed():
        chosen = self._agent_lang_combo.currentData()
        if chosen != get_language():
            set_language(chosen)
            self.InforMsg(tr("agent.restart_required"))
    self._agent_lang_combo.currentIndexChanged.connect(lambda *_: _on_lang_changed())
    top_row.addWidget(self._agent_lang_combo)

    view_file_btn = QtWidgets.QPushButton(tr("agent.view_file_button"))
    view_file_btn.clicked.connect(lambda: self._agent_view_file_clicked())
    top_row.addWidget(view_file_btn)

    skills_btn = QtWidgets.QPushButton("Skills")
    skills_btn.clicked.connect(lambda: self._agent_open_skills())
    top_row.addWidget(skills_btn)

    settings_btn = QtWidgets.QPushButton(tr("agent.settings_button"))
    settings_btn.clicked.connect(lambda: self._agent_open_settings())
    top_row.addWidget(settings_btn)
    vlay.addWidget(toolbar)

    body_margin = QtWidgets.QWidget()
    body_margin_lay = QtWidgets.QVBoxLayout(body_margin)
    body_margin_lay.setContentsMargins(8, 8, 8, 8)
    body_margin_lay.setSpacing(4)
    vlay.addWidget(body_margin, stretch=1)
    vlay = body_margin_lay

    # Left column: session list (top) + file browser (bottom), stacked in a
    # resizable splitter -- both are "navigation" panels, kept together so
    # the chat column stays the visually dominant area.
    # The sidebar itself is now also draggable (part of body_splitter below)
    # rather than a fixed 220px width, per explicit user ask.
    sidebar = QtWidgets.QWidget()
    sidebar.setObjectName("agent_sidebar")
    sidebar.setMinimumWidth(120)
    self._agent_sidebar = sidebar
    side_vlay = QtWidgets.QVBoxLayout(sidebar)
    side_vlay.setContentsMargins(6, 6, 6, 6)
    side_vlay.setSpacing(6)

    new_chat_btn = QtWidgets.QPushButton(tr("agent.new_chat"))
    new_chat_btn.setProperty("variant", "primary")
    new_chat_btn.clicked.connect(lambda: self._agent_new_chat())
    self._agent_session_search = QtWidgets.QLineEdit()
    self._agent_session_search.setPlaceholderText(tr("agent.search_sessions"))
    self._agent_session_search.textChanged.connect(lambda *_: self._agent_rebuild_session_list())
    self._agent_session_list = QtWidgets.QListWidget()
    self._agent_session_list.setObjectName("agent_session_list")
    # itemClicked (not currentRowChanged) -- the list now has non-selectable
    # group-header rows (see _agent_rebuild_session_list) interspersed with
    # real session rows, so raw row position no longer equals a session
    # index; every real item's session index is stashed on it via
    # setData(Qt.UserRole, ...) instead, and looked up from there.
    self._agent_session_list.itemClicked.connect(lambda item: self._agent_session_item_clicked(item))
    # Double-click (QListWidget's default edit trigger) to rename a session
    # -- itemChanged fires on the user's edit and syncs the new name back
    # into self._agent_sessions[idx]["name"].
    self._agent_session_list.itemChanged.connect(lambda item: self._agent_session_renamed(item))
    # Right-click context menu: Rename / Delete Chat.
    self._agent_session_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    self._agent_session_list.customContextMenuRequested.connect(
        lambda pos: self._agent_session_context_menu(pos)
    )

    session_box = QtWidgets.QWidget()
    session_vlay = QtWidgets.QVBoxLayout(session_box)
    session_vlay.setContentsMargins(0, 0, 0, 0)
    session_vlay.addWidget(new_chat_btn)
    session_vlay.addWidget(self._agent_session_search)
    session_vlay.addWidget(self._agent_session_list, stretch=1)

    # File browser: double-click a previewable file to open it in the embedded
    # preview panel. Single-click only selects, avoiding accidental opens while
    # navigating a project tree.
    file_box = QtWidgets.QWidget()
    file_vlay = QtWidgets.QVBoxLayout(file_box)
    file_vlay.setContentsMargins(0, 0, 0, 0)
    file_header_row = QtWidgets.QHBoxLayout()
    file_header_row.addWidget(QtWidgets.QLabel(tr("agent.file_browser_title")))
    refresh_root_btn = QtWidgets.QPushButton("⟳")
    refresh_root_btn.setFixedWidth(28)
    refresh_root_btn.setToolTip(tr("agent.file_browser_refresh"))
    refresh_root_btn.clicked.connect(lambda: self._agent_refresh_file_tree())
    browse_root_btn = QtWidgets.QPushButton("…")
    browse_root_btn.setFixedWidth(28)
    browse_root_btn.setToolTip(tr("agent.file_browser_browse"))
    browse_root_btn.clicked.connect(lambda: self._agent_browse_file_root())
    file_header_row.addStretch()
    file_header_row.addWidget(refresh_root_btn)
    file_header_row.addWidget(browse_root_btn)
    file_vlay.addLayout(file_header_row)
    self._agent_file_filter = QtWidgets.QLineEdit()
    self._agent_file_filter.setPlaceholderText(tr("agent.search_files"))
    self._agent_file_filter.textChanged.connect(lambda text: self._agent_apply_file_filter(text))
    file_vlay.addWidget(self._agent_file_filter)

    project_root = str(Path(__file__).resolve().parent)
    self._agent_fs_model = QtWidgets.QFileSystemModel()
    self._agent_fs_model.setIconProvider(_AgentFileIconProvider())
    self._agent_fs_model.setRootPath(project_root)
    self._agent_fs_model.setNameFilterDisables(False)
    self._agent_file_tree = QtWidgets.QTreeView()
    self._agent_file_tree.setModel(self._agent_fs_model)
    self._agent_file_tree.setRootIndex(self._agent_fs_model.index(project_root))
    for col in (1, 2, 3):  # keep only the Name column -- Size/Type/Date don't fit a 220px sidebar
        self._agent_file_tree.setColumnHidden(col, True)
    self._agent_file_tree.setHeaderHidden(True)
    self._agent_file_tree.doubleClicked.connect(lambda idx: self._agent_file_tree_clicked(idx))
    self._agent_file_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    self._agent_file_tree.customContextMenuRequested.connect(
        lambda pos: self._agent_file_tree_context_menu(pos)
    )
    file_vlay.addWidget(self._agent_file_tree, stretch=1)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    splitter.addWidget(session_box)
    splitter.addWidget(file_box)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 2)
    side_vlay.addWidget(splitter)

    # Chat column
    chat_col = QtWidgets.QWidget()
    self._agent_chat_col = chat_col
    chat_vlay = QtWidgets.QVBoxLayout(chat_col)
    chat_vlay.setContentsMargins(0, 0, 0, 0)
    self._agent_chat_layout = chat_vlay

    # QTextBrowser (not plain QTextEdit) -- needed for clickable links
    # (anchorClicked signal / setOpenLinks) so files/images the agent opens
    # or generates (see _agent_entry_html's "file_link" kind) can be clicked
    # to reopen. Otherwise API-compatible with QTextEdit (.append/.clear/etc).
    self._agent_chat_log = QtWidgets.QTextBrowser()
    self._agent_chat_log.setReadOnly(True)
    self._agent_chat_log.setFrameShape(QtWidgets.QFrame.NoFrame)
    self._agent_chat_log.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
    self._agent_chat_log.document().setDocumentMargin(6)
    self._agent_chat_log.setStyleSheet(
        _agent_chat_browser_style_for_theme(getattr(self, "_current_theme_name", "Graphite Dark"))
    )
    self._agent_chat_log.document().setDefaultStyleSheet(
        _agent_chat_document_style_for_theme(getattr(self, "_current_theme_name", "Graphite Dark"))
    )
    # Disable Qt's default link handling (which would try to launch an
    # external program) and route clicks into the embedded preview panel.
    self._agent_chat_log.setOpenLinks(False)
    self._agent_chat_log.anchorClicked.connect(lambda url: self._agent_chat_link_clicked(url))
    chat_vlay.addWidget(self._agent_chat_log, stretch=1)

    chat_vlay.addSpacing(6)

    # Composer: input + primary Send/Stop action on their own row (the
    # dominant, always-visible controls) -- everything else (Retry, Copy,
    # Export, Clear, the process-visibility toggle) is secondary and moves
    # to a quieter strip underneath instead of competing on the same row.
    composer_row = QtWidgets.QHBoxLayout()
    composer_row.setSpacing(6)
    self._agent_input_edit = _AgentInputEdit()
    self._agent_input_edit.setPlaceholderText(tr("agent.input_placeholder"))
    self._agent_input_edit.sendRequested.connect(lambda: self._agent_send_message())
    self._agent_send_btn = QtWidgets.QPushButton(tr("agent.send"))
    self._agent_send_btn.setProperty("variant", "primary")
    self._agent_send_btn.clicked.connect(lambda: self._agent_send_message())
    self._agent_stop_btn = QtWidgets.QPushButton(tr("agent.stop"))
    self._agent_stop_btn.setProperty("variant", "danger")
    self._agent_stop_btn.setEnabled(False)
    self._agent_stop_btn.clicked.connect(lambda: self._agent_stop_current_run())
    composer_row.addWidget(self._agent_input_edit, stretch=1)
    composer_row.addWidget(self._agent_send_btn)
    composer_row.addWidget(self._agent_stop_btn)
    chat_vlay.addLayout(composer_row)

    composer_toolbar = QtWidgets.QWidget()
    composer_toolbar.setObjectName("agent_composer_toolbar")
    toolbar_row = QtWidgets.QHBoxLayout(composer_toolbar)
    toolbar_row.setContentsMargins(0, 2, 0, 0)
    toolbar_row.addWidget(QtWidgets.QLabel("Mode"))
    self._agent_work_mode_combo = QtWidgets.QComboBox()
    self._agent_work_mode_combo.addItem("Fast", "fast")
    self._agent_work_mode_combo.addItem("Code", "code")
    self._agent_work_mode_combo.setToolTip("Fast: fewer tools for quicker chat. Code: enable project editing tools with permission prompts.")
    self._agent_work_mode_combo.setFixedWidth(88)
    toolbar_row.addWidget(self._agent_work_mode_combo)
    self._agent_show_process_chk = QtWidgets.QCheckBox("Show process")
    self._agent_show_process_chk.setChecked(True)
    self._agent_show_process_chk.setToolTip(
        "Show or hide the Agent execution timeline: thinking status, tool calls, "
        "and clickable tool input/output details."
    )
    self._agent_show_process_chk.toggled.connect(
        lambda *_: self._agent_render_session(self._agent_current_session)
    )
    retry_btn = QtWidgets.QPushButton(tr("agent.retry"))
    retry_btn.clicked.connect(lambda: self._agent_retry_last_message())
    copy_btn = QtWidgets.QPushButton(tr("agent.copy_chat"))
    copy_btn.clicked.connect(lambda: self._agent_copy_current_chat())
    export_btn = QtWidgets.QPushButton(tr("agent.export_chat"))
    export_btn.clicked.connect(lambda: self._agent_export_current_chat())
    clear_btn = QtWidgets.QPushButton(tr("agent.clear_chat"))
    clear_btn.setProperty("variant", "danger")
    clear_btn.setToolTip(tr("agent.clear_chat_tooltip"))
    clear_btn.clicked.connect(lambda: self._agent_clear_chat())
    toolbar_row.addWidget(self._agent_show_process_chk)
    toolbar_row.addStretch()
    toolbar_row.addWidget(retry_btn)
    toolbar_row.addWidget(copy_btn)
    toolbar_row.addWidget(export_btn)
    toolbar_row.addWidget(clear_btn)
    chat_vlay.addWidget(composer_toolbar)

    self._agent_status_lbl = QtWidgets.QLabel("")
    self._agent_status_lbl.setWordWrap(True)
    chat_vlay.addWidget(self._agent_status_lbl)

    for sequence, callback in [
        ("Ctrl+N", self._agent_new_chat),
        ("Ctrl+F", lambda: self._agent_session_search.setFocus()),
        ("Ctrl+L", lambda: self._agent_input_edit.setFocus()),
        ("Ctrl+B", lambda: self._agent_focus_btn.toggle()),
        ("Esc", self._agent_stop_current_run),
        ("Ctrl+Shift+C", self._agent_copy_current_chat),
    ]:
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), page)
        shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        shortcut.activated.connect(callback)

    # Right-side preview panel (embedded, NOT a separate OS window -- see
    # core/file_viewer_widget.py's embedded=True mode). Shows PDFs/images/
    # text the agent opens or generates (show_spectrum_plot/
    # show_structure_plot/open_file), replacing the standalone popup window
    # used in the first version of this feature, per explicit feedback that
    # a separate window wasn't wanted. It stays visible by default as a
    # Codex-like artifact/preview lane for plots, structures, and opened files.
    from core.file_viewer_widget import FileViewerWidget
    preview_container = QtWidgets.QWidget()
    preview_vlay = QtWidgets.QVBoxLayout(preview_container)
    preview_vlay.setContentsMargins(4, 0, 0, 0)
    preview_header = QtWidgets.QHBoxLayout()
    self._agent_preview_title_lbl = QtWidgets.QLabel(tr("agent.file_viewer_title"))
    preview_header.addWidget(self._agent_preview_title_lbl, stretch=1)
    preview_btn = QtWidgets.QPushButton("Preview")
    preview_btn.setCheckable(True)
    preview_btn.setChecked(True)
    terminal_btn = QtWidgets.QPushButton("Terminal")
    terminal_btn.setCheckable(True)
    use_current_btn = QtWidgets.QPushButton(tr("agent.use_current_file"))
    use_current_btn.clicked.connect(lambda: self._agent_use_current_preview_file())
    preview_close_btn = QtWidgets.QPushButton("✕")
    preview_close_btn.setFixedWidth(24)
    preview_close_btn.clicked.connect(lambda: self._agent_toggle_right_panel(False))
    preview_header.addWidget(preview_btn)
    preview_header.addWidget(terminal_btn)
    preview_header.addWidget(use_current_btn)
    preview_header.addWidget(preview_close_btn)
    preview_vlay.addLayout(preview_header)

    right_stack = QtWidgets.QStackedWidget(preview_container)
    self._agent_preview_panel = FileViewerWidget(preview_container, embedded=True)
    self._agent_preview_panel.set_embedded_dark(
        not _theme_is_light(getattr(self, "_current_theme_name", "Graphite Dark"))
    )
    right_stack.addWidget(self._agent_preview_panel)

    terminal_page = QtWidgets.QWidget(preview_container)
    terminal_lay = QtWidgets.QVBoxLayout(terminal_page)
    terminal_lay.setContentsMargins(0, 0, 0, 0)
    terminal_lay.setSpacing(6)
    self._agent_terminal_output = _AgentPtyTerminal(terminal_page)
    terminal_lay.addWidget(self._agent_terminal_output, stretch=1)
    right_stack.addWidget(terminal_page)
    self._agent_right_stack = right_stack
    preview_vlay.addWidget(right_stack, stretch=1)
    self._agent_preview_panel_container = preview_container

    def _set_right_view(index):
        right_stack.setCurrentIndex(index)
        preview_btn.setChecked(index == 0)
        terminal_btn.setChecked(index == 1)
        use_current_btn.setVisible(index == 0)
    preview_btn.clicked.connect(lambda: _set_right_view(0))
    terminal_btn.clicked.connect(lambda: _set_right_view(1))
    _set_right_view(0)
    self._agent_start_terminal_process()

    chat_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    chat_splitter.addWidget(chat_col)
    chat_splitter.addWidget(preview_container)
    chat_splitter.setStretchFactor(0, 2)
    chat_splitter.setStretchFactor(1, 1)
    preview_container.setVisible(True)
    chat_splitter.setSizes([680, 320])
    self._agent_chat_splitter = chat_splitter

    body_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    body_splitter.addWidget(sidebar)
    body_splitter.addWidget(chat_splitter)
    body_splitter.setStretchFactor(0, 0)
    body_splitter.setStretchFactor(1, 1)
    body_splitter.setSizes([220, 700])
    self._agent_body_splitter = body_splitter
    self._agent_focus_restore_preview_visible = False
    vlay.addWidget(body_splitter, stretch=1)

    vl_results.addWidget(page)
    page.setVisible(False)  # Manual mode is the default
    self._agent_panel = page

    self._agent_sessions, self._agent_current_session = self._agent_load_sessions()
    self.Thread_Agent = None
    self._agent_rebuild_session_list()
    self._agent_render_session(self._agent_current_session)

    ready = self._agent_fill_provider_label()
    if not ready:
        self._agent_status_lbl.setText(tr("agent.no_provider_configured"))

    def _apply_mode():
        agent_selected = self._mode_agent_radio.isChecked()
        if getattr(self, "_agent_mode_initialized", False):
            self._show_workspace_switch_cover(
                "Opening Agent..." if agent_selected else "Restoring Manual workspace...",
                260 if not agent_selected else 210,
            )
        tw.setVisible(not agent_selected)
        self._agent_panel.setVisible(agent_selected)
        # Manual-mode-only controls (buttons + spectra list/settings) hide
        # too in Agent Mode, for a clean chat-only workspace.
        # show_spectrum_plot/show_structure_plot/open_file render into the
        # embedded right-side preview panel instead (exported to an image
        # file, then shown via a clickable chat link).
        if hasattr(self, "activityBar"):
            self.activityBar.setVisible(not agent_selected)
        if hasattr(self, "leftPanel"):
            self.leftPanel.setVisible(not agent_selected)
        if agent_selected:
            if hasattr(self, "_agent_show_process_chk"):
                self._agent_show_process_chk.blockSignals(True)
                self._agent_show_process_chk.setChecked(True)
                self._agent_show_process_chk.blockSignals(False)
            if hasattr(self, "_agent_focus_btn"):
                self._agent_focus_btn.blockSignals(True)
                self._agent_focus_btn.setChecked(False)
                self._agent_focus_btn.blockSignals(False)
            self._agent_set_focus_mode(False)
            self._agent_toggle_right_panel(True)
            if hasattr(self, "_agent_body_splitter"):
                self._agent_body_splitter.setSizes([220, max(1, self._agent_body_splitter.width() - 220)])
            if hasattr(self, "_agent_chat_splitter"):
                self._agent_chat_splitter.setSizes([
                    max(1, int(self._agent_chat_splitter.width() * 0.68)),
                    max(1, int(self._agent_chat_splitter.width() * 0.32)),
                ])
            self._agent_render_session(self._agent_current_session)
        if not agent_selected:
            def _restore_manual_splitters():
                self._make_workspace_splitters_interactive()
                if hasattr(self, "activityBar"):
                    self.activityBar.setVisible(True)
                    self.activityBar.updateGeometry()
                if hasattr(self, "leftPanel"):
                    self.leftPanel.setVisible(True)
                    self.leftPanel.updateGeometry()

                if hasattr(self, "splitter_main"):
                    w = max(1, self.splitter_main.width())
                    act_w = 190
                    left_w = min(240, max(120, int(w * 0.18)))
                    main_w = max(320, w - act_w - left_w)
                    self.splitter_main.setSizes([act_w, left_w, main_w])

            QtCore.QTimer.singleShot(0, _restore_manual_splitters)
            QtCore.QTimer.singleShot(120, _restore_manual_splitters)
        # The bottom status bar's "| Workers N" suffix (a backend thread-pool
        # detail, meaningless in Agent Mode) is only re-evaluated when its
        # text is next set -- refresh it immediately on mode switch instead
        # of waiting for the next unrelated progress update.
        if hasattr(self, "progressBar"):
            current = self.progressBar.format()
            if agent_selected and " | Workers" in current:
                self.progressBar.setFormat(current.split(" | Workers")[0])
            elif not agent_selected and " | Workers" not in current:
                self.progressBar.setFormat(current + self._workers_suffix())
    self._mode_manual_radio.toggled.connect(lambda *_: _apply_mode())
    self._mode_agent_radio.toggled.connect(lambda *_: _apply_mode())
    _apply_mode()
    self._agent_mode_initialized = True


def _agent_open_settings(self):
    from core.i18n import tr
    dlg = AgentSettingsDialog(self)
    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        if self._agent_fill_provider_label():
            self._agent_append_display("system", tr("agent.provider_configured"))


def _agent_open_skills(self):
    dlg = AgentSkillsDialog(self)
    dlg.exec_()


def _agent_toggle_right_panel(self, visible):
    panel = getattr(self, "_agent_preview_panel_container", None)
    if panel is not None:
        panel.setVisible(bool(visible))
    if hasattr(self, "_agent_right_panel_btn"):
        self._agent_right_panel_btn.blockSignals(True)
        self._agent_right_panel_btn.setChecked(bool(visible))
        self._agent_right_panel_btn.setToolTip(
            "Hide the right preview/terminal panel." if visible
            else "Show the right preview/terminal panel."
        )
        self._agent_right_panel_btn.blockSignals(False)
    if hasattr(self, "_agent_chat_splitter"):
        if visible:
            self._agent_chat_splitter.setSizes([680, 320])
        else:
            self._agent_chat_splitter.setSizes([max(1, self._agent_chat_splitter.width()), 0])


def _agent_start_terminal_process(self):
    if getattr(self, "_agent_terminal_process", None) is not None:
        return
    output = getattr(self, "_agent_terminal_output", None)
    if output is None:
        return
    try:
        master_fd, slave_fd = pty.openpty()
        fcntl.fcntl(master_fd, fcntl.F_SETFL, os.O_NONBLOCK)
        winsz = struct.pack("HHHH", 32, 100, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsz)
        env = os.environ.copy()
        env.update({
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "PROMPT_COMMAND": "",
            "PS1": "\\w $ ",
        })
        proc = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-i"],
            cwd=str(Path(__file__).resolve().parent),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
        os.close(slave_fd)
    except Exception as e:
        output.insert_terminal_text(f"Failed to start PTY terminal: {e}\n")
        return

    ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def _read_pty():
        chunks = []
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            return
        text = b"".join(chunks).decode("utf-8", errors="replace")
        text = ansi_re.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
        output.insert_terminal_text(text)

    def _send_pty(data):
        try:
            os.write(master_fd, data)
        except OSError:
            output.insert_terminal_text("\n[terminal is not running]\n")

    notifier = QtCore.QSocketNotifier(master_fd, QtCore.QSocketNotifier.Read, self)
    notifier.activated.connect(lambda *_: _read_pty())
    output.sendBytes.connect(_send_pty)
    self._agent_terminal_process = proc
    self._agent_terminal_master_fd = master_fd
    self._agent_terminal_notifier = notifier


def _agent_set_focus_mode(self, enabled):
    sidebar = getattr(self, "_agent_sidebar", None)
    preview = getattr(self, "_agent_preview_panel_container", None)
    if enabled:
        if preview is not None:
            self._agent_focus_restore_preview_visible = preview.isVisible()
            preview.setVisible(False)
        if sidebar is not None:
            sidebar.setVisible(False)
        if hasattr(self, "_agent_chat_layout"):
            self._agent_chat_layout.setContentsMargins(28, 0, 28, 0)
        if hasattr(self, "_agent_body_splitter"):
            self._agent_body_splitter.setSizes([0, max(1, self._agent_body_splitter.width())])
        if hasattr(self, "_agent_chat_splitter"):
            self._agent_chat_splitter.setSizes([max(1, self._agent_chat_splitter.width()), 0])
        if hasattr(self, "_agent_focus_btn"):
            self._agent_focus_btn.setToolTip("Restore side panels.")
        return

    if sidebar is not None:
        sidebar.setVisible(True)
    if hasattr(self, "_agent_chat_layout"):
        self._agent_chat_layout.setContentsMargins(0, 0, 0, 0)
    if preview is not None and getattr(self, "_agent_focus_restore_preview_visible", False):
        preview.setVisible(True)
    if hasattr(self, "_agent_body_splitter"):
        self._agent_body_splitter.setSizes([220, max(1, self._agent_body_splitter.width() - 220)])
    if hasattr(self, "_agent_chat_splitter"):
        if preview is not None and preview.isVisible():
            self._agent_chat_splitter.setSizes([max(1, int(self._agent_chat_splitter.width() * 0.68)), max(1, int(self._agent_chat_splitter.width() * 0.32))])
        else:
            self._agent_chat_splitter.setSizes([max(1, self._agent_chat_splitter.width()), 0])
    if hasattr(self, "_agent_focus_btn"):
        self._agent_focus_btn.setToolTip("Focus mode: hide side panels and keep only the chat column.")


def _agent_current_session_dict(self):
    return self._agent_sessions[self._agent_current_session]


def _agent_append_display(self, kind, text, tool_log_id=None):
    """Append one entry to the current session's re-renderable transcript
    AND paint it into the chat log widget immediately."""
    session = self._agent_current_session_dict()
    was_empty = not session["display_log"]
    session["display_log"].append((kind, text, tool_log_id))
    session["updated_at"] = time.time()
    if was_empty:
        # The chat log was showing the welcome screen (setHtml, see
        # _agent_render_session) -- clear it first, or this first real
        # entry would just get appended underneath the welcome content
        # instead of replacing it. Also re-groups the sidebar entry (e.g.
        # a session opened yesterday that just got its first message today
        # moves into "Today").
        self._agent_chat_log.clear()
        self._agent_rebuild_session_list()
    show_process = (
        not hasattr(self, "_agent_show_process_chk")
        or self._agent_show_process_chk.isChecked()
    )
    if kind != "tool" or show_process:
        self._agent_chat_log.append(
            _agent_entry_html(
                kind, text, self._agent_current_session, tool_log_id,
                getattr(self, "_current_theme_name", "Graphite Dark"),
            )
        )
    self._agent_save_sessions()


_AGENT_QUICK_TASKS = [
    "解释一个鉴定结果的证据等级",
    "帮我理解 AnnoSM 同分异构判断逻辑",
    "结构检索分数和 DreaMS 谱库分数分别代表什么",
    "怎么判断一张谱图质量够不够用",
    "ABCoRT 预测的保留时间可信吗",
    "帮我梳理这套鉴定流程的每一步",
]
_AGENT_QUICK_TASKS_EN = [
    "Explain the evidence level of an identification result",
    "Help me understand how AnnoSM separates isomers",
    "What do the structure-retrieval score and the DreaMS library score mean?",
    "How do I tell whether a spectrum is of good enough quality?",
    "Can I trust the retention time predicted by ABCoRT?",
    "Walk me through every step of the identification workflow",
]


def _agent_quick_tasks():
    from core.i18n import get_language, LANG_EN
    return _AGENT_QUICK_TASKS_EN if get_language() == LANG_EN else _AGENT_QUICK_TASKS


class _AgentContextMeter(QtWidgets.QWidget):
    """Compact circular context usage meter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._payload = None
        self.setFixedSize(46, 46)
        self.setToolTip("Context usage: unknown")

    def set_usage(self, payload):
        self._payload = payload if isinstance(payload, dict) else None
        self._refresh_tooltip()
        self.update()

    def _numbers(self):
        payload = self._payload or {}
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        total = int(usage.get("total_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        window = int(payload.get("context_window") or 0)
        ratio = payload.get("ratio")
        if ratio is None and window > 0:
            ratio = total / window
        return input_tokens, output_tokens, total, window, ratio

    def _refresh_tooltip(self):
        input_tokens, output_tokens, total, window, ratio = self._numbers()
        if not self._payload:
            self.setToolTip("Context usage: unknown")
            return
        pct = f"{ratio * 100:.0f}%" if ratio is not None else "unknown"
        window_text = f"{window:,}" if window > 0 else "unknown"
        self.setToolTip(
            "Context usage\n"
            f"Total: {total:,} / {window_text} tokens ({pct})\n"
            f"Input: {input_tokens:,} tokens\n"
            f"Output: {output_tokens:,} tokens"
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect().adjusted(7, 7, -7, -7)
        pen_width = 4
        base_pen = QtGui.QPen(QtGui.QColor("#27272A"), pen_width)
        base_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(base_pen)
        painter.drawArc(rect, 0, 360 * 16)

        _, _, total, window, ratio = self._numbers()
        if self._payload and ratio is not None and window > 0:
            ratio = max(0.0, min(float(ratio), 1.0))
            if ratio >= 0.90:
                color = "#F87171"
            elif ratio >= 0.75:
                color = "#F59E0B"
            else:
                color = "#FAFAFA"
            arc_pen = QtGui.QPen(QtGui.QColor(color), pen_width)
            arc_pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(arc_pen)
            painter.drawArc(rect, 90 * 16, int(-360 * 16 * ratio))
            text = f"{ratio * 100:.0f}"
        elif self._payload and total > 0:
            text = "tok"
            color = "#A1A1AA"
        else:
            text = "--"
            color = "#71717A"

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(color))
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, text)


class _AgentInputEdit(QtWidgets.QPlainTextEdit):
    """Plain-text chat input with stable Linux/WSLg input-method behavior."""
    sendRequested = QtCore.pyqtSignal()
    _fcitx_activated = False
    _MIN_HEIGHT = 42
    _MAX_HEIGHT = 168
    _RESIZE_MARGIN = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setTabChangesFocus(True)
        self.setMinimumHeight(self._MIN_HEIGHT)
        self.setMaximumHeight(self._MAX_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setAttribute(QtCore.Qt.WA_InputMethodEnabled, True)
        self.setInputMethodHints(QtCore.Qt.ImhNone)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._manual_height = None
        self._resizing_height = False
        self._resize_start_y = 0
        self._resize_start_height = self._MIN_HEIGHT
        self.setToolTip("Drag the top edge to resize. Double-click the top edge to reset.")
        self.document().contentsChanged.connect(self._fit_to_contents)
        QtCore.QTimer.singleShot(0, self._fit_to_contents)

    def _apply_height(self, height):
        height = max(self._MIN_HEIGHT, min(self._MAX_HEIGHT, int(height)))
        if self.height() != height:
            self.setFixedHeight(height)
            self.updateGeometry()
        doc_height = self.document().size().height()
        need_scroll = height >= self._MAX_HEIGHT or doc_height > max(1, height - self.fontMetrics().lineSpacing())
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded if need_scroll else QtCore.Qt.ScrollBarAlwaysOff
        )

    def _fit_to_contents(self):
        margins = self.contentsMargins()
        doc_margin = self.document().documentMargin()
        doc_height = self.document().size().height()
        target = int(doc_height + doc_margin * 2 + margins.top() + margins.bottom() + 14)
        if self._manual_height is not None:
            target = max(target, self._manual_height)
        self._apply_height(target)

    def _is_resize_zone(self, pos):
        return 0 <= pos.y() <= self._RESIZE_MARGIN

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._is_resize_zone(event.pos()):
            self._resizing_height = True
            self._resize_start_y = event.globalPos().y()
            self._resize_start_height = self.height()
            self.viewport().setCursor(QtCore.Qt.SizeVerCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing_height:
            delta = self._resize_start_y - event.globalPos().y()
            self._manual_height = max(self._MIN_HEIGHT, min(self._MAX_HEIGHT, self._resize_start_height + delta))
            self._apply_height(self._manual_height)
            event.accept()
            return
        self.viewport().setCursor(QtCore.Qt.SizeVerCursor if self._is_resize_zone(event.pos()) else QtCore.Qt.IBeamCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing_height and event.button() == QtCore.Qt.LeftButton:
            self._resizing_height = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._is_resize_zone(event.pos()):
            self._manual_height = None
            self._fit_to_contents()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not (modifiers & QtCore.Qt.ShiftModifier):
            self.sendRequested.emit()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._activate_fcitx_pinyin_once()

    def _activate_fcitx_pinyin_once(self):
        if self._fcitx_activated:
            return
        self._fcitx_activated = True
        try:
            import subprocess
            subprocess.run(["fcitx5-remote", "-s", "pinyin"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=0.5)
            subprocess.run(["fcitx5-remote", "-o"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=0.5)
        except Exception:
            pass

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            paths = [url.toLocalFile() for url in mime.urls() if url.toLocalFile()]
            if paths:
                self.textCursor().insertText(" ".join(paths))
                event.acceptProposedAction()
                return
        if mime.hasText():
            self.textCursor().insertText(mime.text())
            event.acceptProposedAction()
            return
        super().dropEvent(event)


def _agent_welcome_html(self):
    from core.i18n import tr
    import urllib.parse
    theme_name = getattr(self, "_current_theme_name", "Graphite Dark")
    accent = _theme_accent(theme_name)
    if _theme_is_light(theme_name):
        p = _light_theme_palette(theme_name)
        accent = p["accent"]
        title_color, sub_color = p["text"], p["muted"]
        card_bg, card_border, card_text = "rgba(255,255,255,0.64)", p["border"], p["text"]
    else:
        title_color, sub_color = "#FAFAFA", "#A1A1AA"
        card_bg, card_border, card_text = "#141417", "#27272A", "#E4E4E7"
    rows = "".join(
        f"<p style='margin:6px 0;'>"
        f"<a href='quicktask:{urllib.parse.quote(task)}' "
        f"style='display:block; padding:7px 12px; border:1px solid {card_border}; "
        f"border-radius:8px; color:{card_text}; text-decoration:none; background:{card_bg};'>"
        f"{task}</a></p>"
        for task in _agent_quick_tasks()
    )
    return (
        "<div style='text-align:center; padding:36px 30px 18px;'>"
        f"<div style='font-size:17px; font-weight:600; color:{title_color};'>{tr('agent.welcome_title')}</div>"
        f"<div style='font-size:12px; color:{sub_color}; margin-top:6px;'>{tr('agent.welcome_subtitle')}</div>"
        f"<div style='width:34px; height:2px; background:{accent}; margin:14px auto 0;'></div>"
        "</div>"
        f"<div style='max-width:420px; margin:0 auto;'>{rows}</div>"
    )


def _agent_render_session(self, index):
    self._agent_chat_log.clear()
    self._agent_update_context_meter(self._agent_sessions[index].get("last_usage"))
    display_log = self._agent_sessions[index]["display_log"]
    if not display_log:
        self._agent_chat_log.setHtml(self._agent_welcome_html())
        return
    show_process = (
        not hasattr(self, "_agent_show_process_chk")
        or self._agent_show_process_chk.isChecked()
    )
    for entry in display_log:
        # Tolerate the older 2-tuple shape (kind, text) too, in case a
        # session was created before tool_log_id existed.
        kind, text = entry[0], entry[1]
        if kind == "tool" and not show_process:
            continue
        tool_log_id = entry[2] if len(entry) > 2 else None
        self._agent_chat_log.append(
            _agent_entry_html(
                kind, text, index, tool_log_id,
                getattr(self, "_current_theme_name", "Graphite Dark"),
            )
        )


def _agent_rebuild_session_list(self):
    """Fully repaints the sidebar session list, grouped by recency
    (pinned first, then Today/Yesterday/Previous 7 Days/Older -- OmicOS-
    style grouping) instead of a flat list. Called after any add/delete/
    rename/pin change rather than doing incremental inserts, since the
    group-header rows make raw list-widget row position meaningless as a
    session index; every real session's actual index into
    self._agent_sessions is stashed on its item via
    setData(Qt.UserRole, ...) and read back from there instead (see
    _agent_session_item_clicked/_agent_session_renamed/
    _agent_session_context_menu)."""
    from core.i18n import tr
    lst = self._agent_session_list
    lst.blockSignals(True)
    lst.clear()
    search = ""
    if hasattr(self, "_agent_session_search"):
        search = self._agent_session_search.text().strip().lower()

    def matches_search(session):
        if not search:
            return True
        haystack = [session.get("name", "")]
        for kind, text, *_ in session.get("display_log", [])[:12]:
            haystack.append(str(text))
        return search in "\n".join(haystack).lower()

    def add_header(text):
        h = QtWidgets.QListWidgetItem(text)
        h.setFlags(QtCore.Qt.NoItemFlags)
        lst.addItem(h)

    def add_session_item(idx, session):
        it = QtWidgets.QListWidgetItem(session["name"])
        it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
        it.setData(QtCore.Qt.UserRole, idx)
        lst.addItem(it)
        if idx == self._agent_current_session:
            lst.setCurrentItem(it)

    pinned = [(i, s) for i, s in enumerate(self._agent_sessions) if s.get("pinned") and matches_search(s)]
    if pinned:
        add_header(tr("agent.sessions_pinned"))
        for i, s in pinned:
            add_session_item(i, s)

    now = time.time()
    groups = {"today": [], "yesterday": [], "week": [], "older": []}
    for i, s in enumerate(self._agent_sessions):
        if s.get("pinned"):
            continue
        if not matches_search(s):
            continue
        age_days = (now - s.get("updated_at", s.get("created_at", now))) / 86400.0
        key = "today" if age_days < 1 else "yesterday" if age_days < 2 else "week" if age_days < 7 else "older"
        groups[key].append((i, s))

    for key, label_key in (("today", "agent.sessions_today"), ("yesterday", "agent.sessions_yesterday"),
                            ("week", "agent.sessions_week"), ("older", "agent.sessions_older")):
        if groups[key]:
            add_header(tr(label_key))
            for i, s in groups[key]:
                add_session_item(i, s)

    lst.blockSignals(False)


def _agent_session_item_clicked(self, item):
    idx = item.data(QtCore.Qt.UserRole)
    if idx is None:  # a group-header row, not a real session
        return
    self._agent_switch_session(idx)


def _agent_session_renamed(self, item):
    idx = item.data(QtCore.Qt.UserRole)
    if idx is None or not (0 <= idx < len(self._agent_sessions)):
        return
    new_name = item.text().strip()
    if not new_name:
        # Revert a blank rename without re-triggering this same handler.
        self._agent_session_list.blockSignals(True)
        item.setText(self._agent_sessions[idx]["name"])
        self._agent_session_list.blockSignals(False)
        return
    self._agent_sessions[idx]["name"] = new_name
    self._agent_sessions[idx]["updated_at"] = time.time()
    self._agent_save_sessions()


def _agent_session_context_menu(self, pos):
    from core.i18n import tr
    item = self._agent_session_list.itemAt(pos)
    if item is None:
        return
    idx = item.data(QtCore.Qt.UserRole)
    if idx is None:
        return
    session = self._agent_sessions[idx]
    menu = QtWidgets.QMenu(self._agent_session_list)
    rename_action = menu.addAction(tr("agent.rename_chat"))
    pin_action = menu.addAction(tr("agent.unpin_chat") if session.get("pinned") else tr("agent.pin_chat"))
    duplicate_action = menu.addAction(tr("agent.duplicate_chat"))
    copy_action = menu.addAction(tr("agent.copy_chat"))
    export_action = menu.addAction(tr("agent.export_chat"))
    delete_action = menu.addAction(tr("agent.delete_chat"))
    chosen = menu.exec_(self._agent_session_list.mapToGlobal(pos))
    if chosen is rename_action:
        self._agent_session_list.editItem(item)
    elif chosen is pin_action:
        session["pinned"] = not session.get("pinned")
        self._agent_rebuild_session_list()
        self._agent_save_sessions()
    elif chosen is duplicate_action:
        self._agent_duplicate_session(idx)
    elif chosen is copy_action:
        QtWidgets.QApplication.clipboard().setText(_agent_chat_as_markdown(session))
        self._agent_status_lbl.setText(tr("agent.chat_copied"))
    elif chosen is export_action:
        self._agent_export_session(idx)
    elif chosen is delete_action:
        self._agent_delete_session(idx)


def _agent_delete_session(self, idx):
    """Deletes a chat session (list entry + its stored messages/display_log/
    tool_log). Always keeps at least one session -- deleting the only
    remaining one just resets it to a fresh empty chat instead of leaving
    zero sessions."""
    from core.i18n import tr
    if not (0 <= idx < len(self._agent_sessions)):
        return
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        self.ErrorMsg(tr("agent.err_still_working_session"))
        return
    name = self._agent_sessions[idx]["name"]
    confirm = QtWidgets.QMessageBox(self)
    confirm.setStyleSheet(LIGHT_DIALOG_STYLE)
    confirm.setIcon(QtWidgets.QMessageBox.Question)
    confirm.setWindowTitle(tr("agent.delete_chat"))
    confirm.setText(tr("agent.delete_chat_confirm", name=name))
    confirm.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    confirm.setDefaultButton(QtWidgets.QMessageBox.No)
    if confirm.exec_() != QtWidgets.QMessageBox.Yes:
        return

    if len(self._agent_sessions) == 1:
        # Last session -- reset in place rather than leaving zero sessions.
        self._agent_sessions[0] = _new_agent_session(tr("agent.chat_default_name", n=1))
        self._agent_current_session = 0
        self._agent_rebuild_session_list()
        self._agent_render_session(0)
        self._agent_save_sessions()
        return

    self._agent_sessions.pop(idx)
    if self._agent_current_session == idx:
        self._agent_current_session = min(idx, len(self._agent_sessions) - 1)
        self._agent_render_session(self._agent_current_session)
    elif self._agent_current_session > idx:
        self._agent_current_session -= 1
    self._agent_rebuild_session_list()
    self._agent_save_sessions()


def _agent_duplicate_session(self, idx):
    from core.i18n import tr
    import copy
    if not (0 <= idx < len(self._agent_sessions)):
        return
    new_session = copy.deepcopy(self._agent_sessions[idx])
    new_session["name"] = f"{new_session.get('name', tr('agent.chat_default_name', n=idx + 1))} copy"
    now = time.time()
    new_session["created_at"] = now
    new_session["updated_at"] = now
    new_session["pinned"] = False
    self._agent_sessions.append(new_session)
    self._agent_current_session = len(self._agent_sessions) - 1
    self._agent_rebuild_session_list()
    self._agent_render_session(self._agent_current_session)
    self._agent_save_sessions()


def _agent_new_chat(self):
    from core.i18n import tr
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        self.ErrorMsg(tr("agent.err_still_working_session"))
        return
    n = len(self._agent_sessions) + 1
    name = tr("agent.chat_default_name", n=n)
    self._agent_sessions.append(_new_agent_session(name))
    self._agent_current_session = len(self._agent_sessions) - 1
    self._agent_rebuild_session_list()
    self._agent_render_session(self._agent_current_session)
    self._agent_save_sessions()


def _agent_switch_session(self, index):
    from core.i18n import tr
    if index < 0 or index >= len(self._agent_sessions):
        return
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        # Snap the list selection back so it doesn't silently look switched.
        self._agent_rebuild_session_list()
        self.ErrorMsg(tr("agent.err_still_working_session"))
        return
    self._agent_current_session = index
    self._agent_render_session(index)
    self._agent_status_lbl.setText("")
    self._agent_save_sessions()


def _agent_auto_title_from_text(text):
    one_line = re.sub(r"\s+", " ", (text or "").strip())
    return one_line[:28] + ("..." if len(one_line) > 28 else "") or "Chat"


def _agent_start_worker(self, session):
    from core.i18n import tr
    from core import agent_config as ac
    from core import agent_tools as at
    cfg = ac.load_config()
    _, provider_cfg = ac.get_active_provider_config(cfg)
    if not provider_cfg or not provider_cfg.get("api_key"):
        self.ErrorMsg(tr("agent.err_no_provider"))
        return
    session["messages"] = at.sanitize_persistent_messages(session.get("messages", []))
    self._agent_send_btn.setEnabled(False)
    if hasattr(self, "_agent_stop_btn"):
        self._agent_stop_btn.setEnabled(True)
    self._agent_status_lbl.setText(tr("agent.thinking"))

    work_mode = "fast"
    if hasattr(self, "_agent_work_mode_combo"):
        work_mode = self._agent_work_mode_combo.currentData() or "fast"
    self.Thread_Agent = AgentWorker(
        self, session["messages"], provider_cfg,
        tool_log=session["tool_log"],
        work_mode=work_mode,
    )
    self.Thread_Agent.request_run_pipeline.connect(self._agent_pipeline_bridge_execute)
    self.Thread_Agent.request_load_spectra.connect(self._agent_load_spectra_execute)
    self.Thread_Agent.request_show_plot.connect(self._agent_show_plot_execute)
    self.Thread_Agent.request_show_structure.connect(self._agent_show_structure_execute)
    self.Thread_Agent.request_open_file.connect(self._agent_open_file_execute)
    self.Thread_Agent.request_save_preview.connect(self._agent_save_current_preview_execute)
    self.Thread_Agent.request_reveal_file.connect(self._agent_open_file_execute)
    self.Thread_Agent.request_tool_approval.connect(self._agent_request_tool_approval)
    self.Thread_Agent.progress.connect(self._agent_on_progress)
    self.Thread_Agent.usage_updated.connect(self._agent_on_usage_updated)
    self.Thread_Agent.finished_with_result.connect(self._agent_on_finished)
    self.Thread_Agent.failed.connect(self._agent_on_failed)
    self.Thread_Agent.start()


def _agent_send_message(self):
    from core import agent_config as ac
    from core.i18n import tr
    text = self._agent_input_edit.toPlainText().strip()
    if not text:
        return
    cfg = ac.load_config()
    _, provider_cfg = ac.get_active_provider_config(cfg)
    if not provider_cfg or not provider_cfg.get("api_key"):
        self.ErrorMsg(tr("agent.err_no_provider"))
        return
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        queue = getattr(self, "_agent_pending_messages", None)
        if queue is None:
            queue = []
            self._agent_pending_messages = queue
        queue.append(text)
        self._agent_append_display("user", text)
        self._agent_input_edit.clear()
        self._agent_append_display("system", "New instruction received. Stopping the current task, then I will continue with this message.")
        self._agent_save_sessions()
        self._agent_stop_current_run()
        return

    session = self._agent_current_session_dict()
    if not session.get("messages") and re.match(r"^(Chat|对话)\s+\d+$", session.get("name", "")):
        session["name"] = _agent_auto_title_from_text(text)
    self._agent_append_display("user", text)
    self._agent_input_edit.clear()
    session["messages"].append({"role": "user", "content": text})
    self._agent_save_sessions()
    self._agent_rebuild_session_list()
    self._agent_start_worker(session)


def _agent_start_pending_message_if_any(self):
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        return
    queue = getattr(self, "_agent_pending_messages", None) or []
    if not queue:
        return
    text = queue.pop(0)
    self._agent_pending_messages = queue
    session = self._agent_current_session_dict()
    from core import agent_tools as at
    session["messages"] = at.sanitize_persistent_messages(session.get("messages", []))
    session["messages"].append({"role": "user", "content": text})
    session["updated_at"] = time.time()
    self._agent_status_lbl.setText("Continuing with the latest instruction...")
    self._agent_save_sessions()
    QtCore.QTimer.singleShot(0, lambda: self._agent_start_worker(session))


def _agent_request_tool_approval(self, payload):
    name = str(payload.get("name") or "")
    args = payload.get("args") or {}
    event = payload.get("event")
    try:
        preview = json.dumps(args, ensure_ascii=False, indent=2, default=str)
    except Exception:
        preview = str(args)
    if len(preview) > 1800:
        preview = preview[:1800] + "\n... [truncated]"

    approved = self._agent_show_permission_card(
        title="Agent permission",
        heading=f"Allow Agent to run {name}?",
        details=preview,
        accept_text="Allow",
        cancel_text="Cancel",
    )
    payload["approved"] = approved
    if hasattr(self, "_agent_status_lbl"):
        self._agent_status_lbl.setText(
            f"Permission {'granted' if approved else 'denied'}: {name}"
        )
    if event is not None:
        event.set()


def _agent_show_permission_card(self, title, heading, details, accept_text="Allow", cancel_text="Cancel"):
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.setObjectName("agent_permission_dialog")
    dialog.setMinimumWidth(520)
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    lay = QtWidgets.QVBoxLayout(dialog)
    lay.setContentsMargins(18, 18, 18, 18)
    lay.setSpacing(12)

    title_lbl = QtWidgets.QLabel(heading)
    title_lbl.setObjectName("agent_permission_heading")
    title_lbl.setWordWrap(True)
    lay.addWidget(title_lbl)

    detail_box = QtWidgets.QPlainTextEdit()
    detail_box.setObjectName("agent_permission_details")
    detail_box.setReadOnly(True)
    detail_box.setPlainText(str(details or ""))
    detail_box.setMinimumHeight(180)
    detail_box.setMaximumHeight(260)
    lay.addWidget(detail_box)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch()
    cancel_btn = QtWidgets.QPushButton(cancel_text)
    allow_btn = QtWidgets.QPushButton(accept_text)
    allow_btn.setObjectName("agent_permission_allow")
    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(allow_btn)
    lay.addLayout(btn_row)

    cancel_btn.clicked.connect(dialog.reject)
    allow_btn.clicked.connect(dialog.accept)
    dialog.setStyleSheet("""
        QDialog#agent_permission_dialog {
            background: #F8FAFC;
            color: #111827;
            border-radius: 14px;
        }
        QLabel#agent_permission_heading {
            color: #111827;
            font-size: 15px;
            font-weight: 700;
        }
        QPlainTextEdit#agent_permission_details {
            background: #FFFFFF;
            border: 1px solid #D8E2F0;
            border-radius: 10px;
            color: #1F2937;
            padding: 10px;
            font-family: "DejaVu Sans Mono", "Consolas", monospace;
            font-size: 11px;
        }
        QPushButton {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            color: #111827;
            padding: 7px 16px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #EEF2FF;
            border-color: #93A4C7;
        }
        QPushButton#agent_permission_allow {
            background: #111827;
            border-color: #111827;
            color: #FFFFFF;
        }
        QPushButton#agent_permission_allow:hover {
            background: #1F2937;
        }
    """)
    return dialog.exec_() == QtWidgets.QDialog.Accepted


def _agent_stop_current_run(self):
    from core.i18n import tr
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        self.Thread_Agent.requestInterruption()
        self._agent_cancel_running_computer_tasks()
        self._agent_status_lbl.setText(tr("agent.stopping"))
        if hasattr(self, "_agent_stop_btn"):
            self._agent_stop_btn.setEnabled(False)


def _agent_cancel_running_computer_tasks(self):
    """Best-effort cancellation for Agent-started local work.

    QThread.requestInterruption() only works at checkpoints. Some Agent tools
    block while a pipeline thread or subprocess is still running, so the Stop
    button also tears down those concrete tasks and wakes bridge awaiters.
    """
    stop_message = "Stopped by user."

    proc = getattr(self, "_agent_active_subprocess", None)
    if proc is not None:
        try:
            if proc.poll() is None:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                def _kill_if_still_running(p=proc):
                    try:
                        if p.poll() is None:
                            if os.name == "posix":
                                os.killpg(p.pid, signal.SIGKILL)
                            else:
                                p.kill()
                    except Exception:
                        pass
                QtCore.QTimer.singleShot(900, _kill_if_still_running)
        except Exception:
            pass

    pipeline = getattr(self, "Thread_Pipeline", None)
    if pipeline is not None and pipeline.isRunning():
        try:
            pipeline.requestInterruption()
            for sig_name in ("progress", "finished_with_result", "failed"):
                sig = getattr(pipeline, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except Exception:
                        pass
            if not pipeline.wait(250):
                pipeline.terminate()
                pipeline.wait(1200)
        except Exception:
            pass
        try:
            self.butt_pipeline.setEnabled(True)
        except Exception:
            pass
        try:
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
            self.progressBar.setFormat("Stopped")
        except Exception:
            pass
        if hasattr(self, "_pipeline_clock_timer"):
            self._pipeline_clock_timer.stop()
        self._pipeline_start_time = None
        if hasattr(self, "_pipeline_status_lbl"):
            self._pipeline_status_lbl.setText("Force-stopped by user.")
        self.Thread_Pipeline = None

    for signal_name in (
        "agent_pipeline_result_failed",
        "agent_load_spectra_failed",
        "agent_plot_failed",
        "agent_file_failed",
    ):
        sig = getattr(self, signal_name, None)
        if sig is not None:
            try:
                sig.emit(stop_message)
            except Exception:
                pass


def _agent_retry_last_message(self):
    from core.i18n import tr
    if self.Thread_Agent is not None and self.Thread_Agent.isRunning():
        self.ErrorMsg(tr("agent.err_still_working"))
        return
    session = self._agent_current_session_dict()
    last_user_log = None
    for i in range(len(session.get("display_log", [])) - 1, -1, -1):
        entry = session["display_log"][i]
        if entry and entry[0] == "user":
            last_user_log = i
            break
    if last_user_log is None:
        self._agent_status_lbl.setText(tr("agent.nothing_to_retry"))
        return

    last_user_text = session["display_log"][last_user_log][1]
    last_user_msg = None
    for i in range(len(session.get("messages", [])) - 1, -1, -1):
        msg = session["messages"][i]
        if msg.get("role") == "user" and msg.get("content") == last_user_text:
            last_user_msg = i
            break
    if last_user_msg is None:
        self._agent_status_lbl.setText(tr("agent.nothing_to_retry"))
        return

    session["display_log"] = session["display_log"][:last_user_log + 1]
    session["messages"] = session["messages"][:last_user_msg + 1]
    session["updated_at"] = time.time()
    self._agent_render_session(self._agent_current_session)
    self._agent_save_sessions()
    self._agent_start_worker(session)


def _agent_chat_as_markdown(session):
    lines = [f"# {session.get('name', 'Chat')}", ""]
    for entry in session.get("display_log", []):
        kind = entry[0] if entry else ""
        text = str(entry[1]) if len(entry) > 1 else ""
        if kind == "user":
            lines.extend(["## You", text, ""])
        elif kind == "agent":
            lines.extend(["## Flavonoid Agent", text, ""])
        elif kind == "tool":
            lines.extend([f"> {text}", ""])
        elif kind == "file_link":
            lines.extend([f"[File] {text}", ""])
        elif kind == "system":
            lines.extend([f"> System: {text}", ""])
    return "\n".join(lines).strip() + "\n"


def _agent_copy_current_chat(self):
    from core.i18n import tr
    text = _agent_chat_as_markdown(self._agent_current_session_dict())
    QtWidgets.QApplication.clipboard().setText(text)
    self._agent_status_lbl.setText(tr("agent.chat_copied"))


def _agent_export_current_chat(self):
    return _agent_export_session(self, self._agent_current_session)


def _agent_export_session(self, idx):
    from core.i18n import tr
    if not (0 <= idx < len(self._agent_sessions)):
        return
    session = self._agent_sessions[idx]
    default_name = re.sub(r"[^A-Za-z0-9._-]+", "_", session.get("name", "chat")).strip("_") or "chat"
    out_dir = Path.home() / ".annomass" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_path = str(out_dir / f"{default_name}.md")
    path, _ = light_get_save_file_name(self, "Export Agent chat", default_path, "Markdown (*.md);;Text (*.txt)")
    if not path:
        return
    Path(path).write_text(_agent_chat_as_markdown(session), encoding="utf-8")
    self._agent_status_lbl.setText(tr("agent.chat_exported", path=path))


# Compact candidate-table view for the embedded preview panel -- a subset of
# _PIPELINE_RESULT_COLUMNS (core/agent_tools.py) that fits a ~440px pane,
# with the same display labels the Pipeline tab itself uses for consistency.
_AGENT_CANDIDATE_TABLE_COLUMNS = [
    "rank", "spectrum_rank", "compound_name", "evidence_level", "formula",
    "best_deepmass_score", "best_DreaMS_similarity",
    "best_public_library_similarity", "best_spectrum_predictor_score", "predicted_rt_min",
]
_AGENT_CANDIDATE_TABLE_LABELS = {
    "rank": "Rank", "spectrum_rank": "Spectrum rank", "compound_name": "Compound", "evidence_level": "Evidence",
    "best_deepmass_score": "Structure", "best_DreaMS_similarity": "Library match",
    "best_public_library_similarity": "Public lib.", "best_spectrum_predictor_score": "Spectrum pred.",
    "predicted_rt_min": "RT (min)",
}
_AGENT_CANDIDATE_TOOLS = {"explain_pipeline_results", "run_pipeline"}
_AGENT_EXPORT_TOOLS = {"export_pipeline_report", "write_pipeline_word_report", "plot_pipeline_results"}




def _agent_on_progress(self, message):
    self._agent_status_lbl.setText(message)
    if message.startswith("Thinking"):
        self._agent_append_display("tool", message)
    elif message.startswith("Calling tool:"):
        self._agent_append_display("tool", "Calling " + message[len("Calling tool:"):].strip())
    elif message.startswith("Requesting "):
        self._agent_append_display("tool", message)
    # Finished tool calls get a clickable checklist-style line. The preceding
    # "Calling ..." / "Requesting ..." lines above are intentionally visible
    # too, so Agent Mode has an execution timeline without exposing
    # hidden chain-of-thought.
    if message.startswith("✓") or message.startswith("✗"):
        # run_agent_loop appends to tool_log BEFORE calling progress_cb (see
        # its docstring), so the record for THIS message is always the last
        # one -- correlate by id so the chat line can link to its full
        # input/output (see _agent_show_tool_detail).
        session = self._agent_current_session_dict()
        tool_log = session.get("tool_log") or []
        record = tool_log[-1] if tool_log else None
        record_id = record["id"] if record else None
        self._agent_append_display("tool", message, tool_log_id=record_id)

        # Candidate ranking table: explain_pipeline_results/run_pipeline
        # return {"candidates": [...]} -- show it as a real table in the
        # embedded preview panel instead of leaving the reader to parse it
        # out of the agent's prose/markdown reply. This runs on the main
        # thread (progress signals are queued cross-thread by Qt), so
        # touching the table widget here is safe.
        if (record is not None and record.get("name") in _AGENT_CANDIDATE_TOOLS
                and record["status"] == "completed"
                and isinstance(record.get("result"), dict)
                and record["result"].get("candidates")):
            try:
                self._get_file_viewer().load_table(
                    record["result"]["candidates"],
                    columns=_AGENT_CANDIDATE_TABLE_COLUMNS,
                    column_labels=_AGENT_CANDIDATE_TABLE_LABELS,
                )
                self._agent_show_preview_panel()
            except Exception:
                pass  # the tool call itself succeeded; a failed table render shouldn't be treated as an error
        if (record is not None and record.get("name") in _AGENT_EXPORT_TOOLS
                and record["status"] == "completed"
                and isinstance(record.get("result"), dict)
                and record["result"].get("file_path")):
            try:
                file_path = record["result"]["file_path"]
                self._get_file_viewer().load_file(file_path)
                self._agent_show_preview_panel()
                self._agent_update_preview_title(file_path)
                self._agent_append_display("file_link", file_path)
            except Exception:
                pass
    elif message.startswith("Compacting"):
        self._agent_append_display("tool", message)


def _agent_update_context_meter(self, payload):
    if hasattr(self, "_agent_context_meter"):
        self._agent_context_meter.set_usage(payload)


def _agent_on_usage_updated(self, payload):
    session = self._agent_current_session_dict()
    if isinstance(payload, dict):
        session["last_usage"] = payload
        session["updated_at"] = time.time()
        self._agent_update_context_meter(payload)
        self._agent_save_sessions()


def _agent_on_finished(self, text):
    from core.i18n import tr
    from core import agent_tools as at
    self._agent_send_btn.setEnabled(True)
    if hasattr(self, "_agent_stop_btn"):
        self._agent_stop_btn.setEnabled(False)
    if text.strip() == "[stopped by user]":
        self._agent_status_lbl.setText(tr("agent.stopped"))
        self._agent_append_display("system", tr("agent.stopped"))
    else:
        self._agent_status_lbl.setText(tr("agent.ready"))
        self._agent_append_display("agent", text)
        session = self._agent_current_session_dict()
        session["messages"] = at.sanitize_persistent_messages(session.get("messages", []))
        if text.strip():
            session["messages"].append({"role": "assistant", "content": text})
    self.Thread_Agent = None
    self._agent_save_sessions()
    self._agent_start_pending_message_if_any()


def _agent_on_failed(self, message):
    from core.i18n import tr
    from core import agent_tools as at
    self._agent_send_btn.setEnabled(True)
    if hasattr(self, "_agent_stop_btn"):
        self._agent_stop_btn.setEnabled(False)
    self._agent_status_lbl.setText(tr("agent.failed_prefix", message=message))
    self._agent_append_display("system", tr("agent.error_prefix", message=message))
    # Roll back the user's turn so the next message doesn't build on a
    # broken exchange.
    session = self._agent_current_session_dict()
    session["messages"] = at.sanitize_persistent_messages(session.get("messages", []))
    if session["messages"] and session["messages"][-1].get("role") == "user":
        session["messages"].pop()
    self.Thread_Agent = None
    self._agent_save_sessions()
    self._agent_start_pending_message_if_any()


def _agent_clear_chat(self):
    session = self._agent_current_session_dict()
    session["messages"] = []
    session["display_log"] = []
    session["tool_log"] = []
    session["last_usage"] = None
    session["updated_at"] = time.time()
    self._agent_render_session(self._agent_current_session)
    self._agent_status_lbl.setText("")
    self._agent_save_sessions()


# ── bind Flavonoid Agent methods to AnnoMASS ────────────────────────────────
AnnoMASS._agent_pipeline_bridge_execute = _agent_pipeline_bridge_execute
AnnoMASS._agent_relay_pipeline_ready    = _agent_relay_pipeline_ready
AnnoMASS._agent_relay_pipeline_failed   = _agent_relay_pipeline_failed
AnnoMASS._agent_load_spectra_execute    = _agent_load_spectra_execute
AnnoMASS._agent_show_plot_execute       = _agent_show_plot_execute
AnnoMASS._agent_show_structure_execute  = _agent_show_structure_execute
AnnoMASS._agent_open_file_execute       = _agent_open_file_execute
AnnoMASS._agent_save_current_preview_execute = _agent_save_current_preview_execute
AnnoMASS._get_file_viewer               = _get_file_viewer
AnnoMASS._agent_show_preview_panel      = _agent_show_preview_panel
AnnoMASS._agent_update_preview_title    = _agent_update_preview_title
AnnoMASS._agent_use_current_preview_file = _agent_use_current_preview_file
AnnoMASS._agent_view_file_clicked       = _agent_view_file_clicked
AnnoMASS._agent_file_tree_clicked       = _agent_file_tree_clicked
AnnoMASS._agent_file_tree_context_menu  = _agent_file_tree_context_menu
AnnoMASS._agent_delete_file_tree_path   = _agent_delete_file_tree_path
AnnoMASS._agent_copy_text_to_clipboard  = _agent_copy_text_to_clipboard
AnnoMASS._agent_insert_text_at_cursor   = _agent_insert_text_at_cursor
AnnoMASS._agent_apply_file_filter       = _agent_apply_file_filter
AnnoMASS._agent_set_file_browser_root   = _agent_set_file_browser_root
AnnoMASS._agent_refresh_file_tree       = _agent_refresh_file_tree
AnnoMASS._agent_chat_link_clicked       = _agent_chat_link_clicked
AnnoMASS._agent_show_tool_detail        = _agent_show_tool_detail
AnnoMASS._agent_browse_file_root        = _agent_browse_file_root
AnnoMASS._agent_load_sessions           = _agent_load_sessions
AnnoMASS._agent_save_sessions           = _agent_save_sessions
AnnoMASS._agent_start_worker            = _agent_start_worker
AnnoMASS._agent_start_pending_message_if_any = _agent_start_pending_message_if_any
AnnoMASS._agent_request_tool_approval   = _agent_request_tool_approval
AnnoMASS._agent_show_permission_card    = _agent_show_permission_card
AnnoMASS._agent_stop_current_run        = _agent_stop_current_run
AnnoMASS._agent_cancel_running_computer_tasks = _agent_cancel_running_computer_tasks
AnnoMASS._agent_retry_last_message      = _agent_retry_last_message
AnnoMASS._agent_copy_current_chat       = _agent_copy_current_chat
AnnoMASS._agent_export_current_chat     = _agent_export_current_chat
AnnoMASS._agent_export_session          = _agent_export_session
AnnoMASS._agent_duplicate_session       = _agent_duplicate_session
AnnoMASS._init_agent_tab                = _init_agent_tab
AnnoMASS._agent_fill_provider_label     = _agent_fill_provider_label
AnnoMASS._agent_open_settings           = _agent_open_settings
AnnoMASS._agent_open_skills             = _agent_open_skills
AnnoMASS._agent_toggle_right_panel      = _agent_toggle_right_panel
AnnoMASS._agent_start_terminal_process  = _agent_start_terminal_process
AnnoMASS._agent_set_focus_mode          = _agent_set_focus_mode
AnnoMASS._agent_current_session_dict    = _agent_current_session_dict
AnnoMASS._agent_append_display          = _agent_append_display
AnnoMASS._agent_render_session          = _agent_render_session
AnnoMASS._agent_welcome_html            = _agent_welcome_html
AnnoMASS._agent_rebuild_session_list     = _agent_rebuild_session_list
AnnoMASS._agent_session_item_clicked     = _agent_session_item_clicked
AnnoMASS._agent_session_renamed         = _agent_session_renamed
AnnoMASS._agent_session_context_menu    = _agent_session_context_menu
AnnoMASS._agent_delete_session          = _agent_delete_session
AnnoMASS._agent_new_chat                = _agent_new_chat
AnnoMASS._agent_switch_session          = _agent_switch_session
AnnoMASS._agent_send_message            = _agent_send_message
AnnoMASS._agent_on_progress             = _agent_on_progress
AnnoMASS._agent_update_context_meter    = _agent_update_context_meter
AnnoMASS._agent_on_usage_updated        = _agent_on_usage_updated
AnnoMASS._agent_on_finished             = _agent_on_finished
AnnoMASS._agent_on_failed               = _agent_on_failed
AnnoMASS._agent_clear_chat              = _agent_clear_chat


# Bind methods to AnnoMASS
AnnoMASS.closeEvent            = closeEvent


def _install_hard_exit_watchdog(app):
    """Belt-and-suspenders against leaked background processes (see
    closeEvent): once Qt starts quitting, force the whole process down a
    few seconds later no matter what. closeEvent already stops every
    thread it knows about, but this is a hard backstop against any thread
    -- present or added later -- that we didn't anticipate keeping the
    process alive in the background (invisible, near-0% CPU) indefinitely."""
    def _watchdog():
        time.sleep(5)
        os._exit(0)
    app.aboutToQuit.connect(lambda: threading.Thread(target=_watchdog, daemon=True).start())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _install_hard_exit_watchdog(app)
    # Applied at the QApplication level (not per-dialog) since QMenu popups
    # (e.g. a QLineEdit's right-click Cut/Copy/Paste menu) are top-level
    # windows that don't inherit a parent widget's setStyleSheet(); without
    # this they fell back to an unreadable native/system menu style.
    app.setStyleSheet(GLOBAL_MENU_STYLE)
    ui = AnnoMASS()
    ui.show()
    sys.exit(app.exec_())
