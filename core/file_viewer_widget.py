"""PDF/image/text file viewer for the Flavonoid Agent panel.

Standalone top-level window (not embedded in tabWidget_results or the Agent
panel itself), so it's reachable from both Manual and Agent mode without
duplicating layout. See annomass_app.py's "View File…" button and the
open_file agent tool (AgentWorker._tool_open_file / _agent_open_file_execute).
"""

from pathlib import Path
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from PyQt5 import QtSvg
except ImportError:  # pragma: no cover - environment dependent
    QtSvg = None

PDF_EXTENSIONS = {".pdf"}
SVG_EXTENSIONS = {".svg"}
# .tif/.tiff: Qt's own QImageReader plugin covers this on most Linux/conda
# Qt builds, but isn't guaranteed everywhere -- _load_raster_image() falls
# back to Pillow (if installed) when Qt's own load comes back null, so TIFF
# still works either way.
RASTER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
IMAGE_EXTENSIONS = RASTER_IMAGE_EXTENSIONS | SVG_EXTENSIONS
# .mgf/.msp/.mat: this project's own spectra formats -- plain text, so they
# preview the same way .txt does, no special parsing needed.
CSV_EXTENSIONS = {".csv", ".tsv"}
CODE_EXTENSIONS = {".py", ".sh", ".bash", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".xml", ".yml", ".yaml"}
TEXT_EXTENSIONS = {".txt", ".log", ".md", ".mgf", ".msp", ".mat", ".json"} | CODE_EXTENSIONS
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
WORD_EXTENSIONS = {".docx"}
# .doc (legacy pre-2007 binary format) is deliberately NOT in here --
# python-docx (the library used for .docx below) can't read it, and there's
# no other docx/doc parser already in this project's dependencies. Handled
# as a clear "not supported, please save as .docx" message instead of
# silently failing or crashing (see _load_word).
LEGACY_WORD_EXTENSIONS = {".doc"}
PREVIEWABLE_EXTENSIONS = (
    PDF_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS | CSV_EXTENSIONS
    | EXCEL_EXTENSIONS | WORD_EXTENSIONS | LEGACY_WORD_EXTENSIONS
)

ZOOM_MIN = 0.1
ZOOM_MAX = 8.0
ZOOM_STEP = 1.25

# Duplicated (not imported) from annomass_app.py's LIGHT_DIALOG_STYLE /
# LIGHT_FILE_DIALOG_STYLE to avoid a circular import -- this module must not
# import annomass_app.py. Same colors, kept visually consistent by hand.
# Without this, both this window and the native QFileDialog it opens inherit
# whatever dark system/Qt theme is active and render unreadable (dark text
# on dark background) -- same root cause already fixed once for
# AgentSettingsDialog.
_LIGHT_WINDOW_STYLE = """
    QWidget { background: rgba(247,247,247,230); color: #202020;
              font-family: "Segoe UI","DejaVu Sans","Arial"; font-size: 12px; }
    QLabel { color: #202020; background: transparent; }
    QPlainTextEdit, QScrollArea { background: rgba(255,255,255,205); color: #202020;
                                    border: 1px solid rgba(184,184,184,190); }
    QTableWidget {
        background: rgba(255,255,255,205); color: #202020;
        border: 1px solid rgba(184,184,184,190);
        alternate-background-color: rgba(247,247,247,150);
    }
    QPushButton { background: rgba(255,255,255,190); color: #202020; border: 1px solid rgba(168,168,168,190);
                  border-radius: 3px; padding: 4px 10px; }
    QPushButton:hover { background: #eaf3ff; border-color: #2f7dd1; }
    QPushButton:disabled { color: #a0a0a0; }
"""

_DARK_EMBEDDED_STYLE = """
    QWidget { background: #0C0C0F; color: #E4E4E7;
              font-family: "Segoe UI","DejaVu Sans","Arial"; font-size: 12px; }
    QLabel { color: #D4D4D8; background: transparent; }
    QPlainTextEdit, QScrollArea {
        background: #141417; color: #E4E4E7;
        border: 1px solid #27272A; border-radius: 8px;
        selection-background-color: #3F3F46;
    }
    QTableWidget {
        background: #141417; color: #E4E4E7;
        border: 1px solid #27272A; border-radius: 8px;
        gridline-color: #27272A; alternate-background-color: #0F0F12;
        selection-background-color: #27272A; selection-color: #FAFAFA;
    }
    QHeaderView::section {
        background: #18181B; color: #A1A1AA;
        border: none; border-bottom: 1px solid #27272A;
        border-right: 1px solid #27272A; padding: 5px 8px;
        font-weight: 700; font-size: 11px;
    }
    QPushButton {
        background: transparent; color: #D4D4D8; border: 1px solid transparent;
        border-radius: 7px; padding: 4px 10px;
    }
    QPushButton:hover { background: #18181B; color: #FAFAFA; border-color: #27272A; }
    QPushButton:pressed { background: #27272A; }
    QPushButton:disabled { color: #52525B; }
    QComboBox {
        background: #141417; color: #FAFAFA; border: 1px solid #27272A;
        border-radius: 7px; padding: 4px 8px;
    }
    QComboBox QAbstractItemView {
        background: #18181B; color: #FAFAFA; border: 1px solid #3F3F46;
        selection-background-color: #27272A;
    }
    QScrollBar:vertical { background: #0C0C0F; width: 10px; }
    QScrollBar::handle:vertical { background: #3F3F46; min-height: 20px; border-radius: 5px; }
    QScrollBar::handle:vertical:hover { background: #52525B; }
    QScrollBar:horizontal { background: #0C0C0F; height: 10px; }
    QScrollBar::handle:horizontal { background: #3F3F46; min-width: 20px; border-radius: 5px; }
    QScrollBar::handle:horizontal:hover { background: #52525B; }
    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
        background: none; border: none;
    }
"""

_LIGHT_FILE_DIALOG_STYLE = """
    QFileDialog, QFileDialog QWidget { background: #f7f7f7; color: #202020;
        font-family: "Segoe UI","DejaVu Sans","Arial"; font-size: 12px; }
    QFileDialog QLineEdit, QFileDialog QComboBox, QFileDialog QListView,
    QFileDialog QTreeView, QFileDialog QTableView, QFileDialog QAbstractItemView {
        background: #ffffff; color: #202020; border: 1px solid #b8b8b8;
        selection-background-color: #2f7dd1; selection-color: #ffffff; }
    QFileDialog QLabel { color: #202020; }
    QFileDialog QPushButton { background: #ffffff; color: #202020;
        border: 1px solid #a8a8a8; border-radius: 3px; padding: 4px 10px; }
    QFileDialog QPushButton:hover { background: #eaf3ff; border-color: #2f7dd1; }
"""


class FileViewerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, embedded=False):
        """embedded=True: a normal child widget placed inside a layout (the
        Agent panel's right-side preview pane) -- no window chrome, no
        Qt.Window flag. embedded=False (default): a standalone top-level
        window, kept for any future non-Agent-panel use of this widget."""
        from core.i18n import tr
        if embedded:
            super().__init__(parent)
        else:
            super().__init__(parent, QtCore.Qt.Window)
            self.setWindowTitle(tr("agent.file_viewer_title"))
            self.resize(800, 900)
        self.embedded = embedded
        self.setStyleSheet(_DARK_EMBEDDED_STYLE if embedded else _LIGHT_WINDOW_STYLE)

        self._pdf_doc = None
        self._pdf_page_index = 0
        self._current_path = None
        # What's currently loaded into image_label, so _apply_zoom() knows
        # how to re-render it: "raster" (scale the cached base QPixmap),
        # "svg" (re-render the QSvgRenderer at the new target size -- crisp
        # at any zoom, unlike raster scaling), "pdf" (re-render the current
        # page at a zoom-scaled matrix), or None (nothing zoomable loaded).
        self._content_kind = None
        self._base_pixmap = None      # raster images
        self._svg_renderer = None     # SVG
        self._zoom = 1.0

        vlay = QtWidgets.QVBoxLayout(self)

        toolbar = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton(tr("agent.file_viewer_open"))
        self.open_btn.clicked.connect(self._on_open_clicked)
        self.save_as_btn = QtWidgets.QPushButton("Save As…")
        self.save_as_btn.setToolTip("Choose a folder and file name for the current preview")
        self.save_as_btn.setEnabled(False)
        self.save_as_btn.clicked.connect(self._on_save_as_clicked)
        self.prev_btn = QtWidgets.QPushButton(tr("agent.file_viewer_prev"))
        self.prev_btn.clicked.connect(lambda: self._change_page(-1))
        self.next_btn = QtWidgets.QPushButton(tr("agent.file_viewer_next"))
        self.next_btn.clicked.connect(lambda: self._change_page(1))
        self.page_lbl = QtWidgets.QLabel("")
        toolbar.addWidget(self.open_btn)
        toolbar.addWidget(self.save_as_btn)
        toolbar.addStretch()
        self.zoom_out_btn = QtWidgets.QPushButton("−")
        self.zoom_out_btn.setFixedWidth(28)
        self.zoom_out_btn.setToolTip(tr("agent.file_viewer_zoom_out"))
        self.zoom_out_btn.clicked.connect(lambda: self._zoom_by(1 / ZOOM_STEP))
        self.zoom_lbl = QtWidgets.QLabel("100%")
        self.zoom_lbl.setFixedWidth(44)
        self.zoom_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.zoom_lbl.setToolTip(tr("agent.file_viewer_zoom_reset"))
        self.zoom_lbl.mousePressEvent = lambda _ev: self._set_zoom(1.0)
        self.zoom_in_btn = QtWidgets.QPushButton("+")
        self.zoom_in_btn.setFixedWidth(28)
        self.zoom_in_btn.setToolTip(tr("agent.file_viewer_zoom_in"))
        self.zoom_in_btn.clicked.connect(lambda: self._zoom_by(ZOOM_STEP))
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.zoom_lbl)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addStretch()
        # Sheet selector -- only shown for a multi-sheet Excel workbook (see
        # _load_excel); single-sheet files and every other format leave it
        # hidden.
        self.sheet_combo = QtWidgets.QComboBox()
        self.sheet_combo.setMinimumWidth(120)
        self.sheet_combo.currentIndexChanged.connect(self._on_sheet_changed)
        self.sheet_combo.setVisible(False)
        toolbar.addWidget(self.sheet_combo)
        toolbar.addWidget(self.prev_btn)
        toolbar.addWidget(self.page_lbl)
        toolbar.addWidget(self.next_btn)
        vlay.addLayout(toolbar)
        self._set_zoom_controls_visible(False)

        self.stack = QtWidgets.QStackedWidget()
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        # Structured data page -- populated by load_table() when the agent's
        # explain_pipeline_results/run_pipeline tools return candidate rows,
        # so results are scannable as a real table instead of only prose/
        # markdown in the chat (see annomass_app.py's _agent_on_progress).
        self.table_widget = QtWidgets.QTableWidget()
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.table_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table_widget.setShowGrid(True)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSortingEnabled(True)
        copy_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Copy, self.table_widget)
        copy_shortcut.activated.connect(self._copy_table_selection)
        self.stack.addWidget(self.image_scroll)  # index 0: PDF page image
        self.stack.addWidget(self.text_edit)      # index 1: plain text
        self.stack.addWidget(self.table_widget)   # index 2: candidate/structured-data table
        vlay.addWidget(self.stack, stretch=1)

        self._set_pdf_controls_visible(False)

    def set_embedded_dark(self, enabled):
        if self.embedded:
            self.setStyleSheet(_DARK_EMBEDDED_STYLE if enabled else _LIGHT_WINDOW_STYLE)

    def _copy_table_selection(self):
        ranges = self.table_widget.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        lines = []
        for row in range(r.topRow(), r.bottomRow() + 1):
            vals = []
            for col in range(r.leftColumn(), r.rightColumn() + 1):
                item = self.table_widget.item(row, col)
                vals.append("" if item is None else item.text())
            lines.append("\t".join(vals))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def load_table(self, records, columns=None, column_labels=None):
        """records: list of dicts (e.g. Pipeline candidate rows). columns:
        optional explicit column order (defaults to the union of keys found,
        in first-seen order). column_labels: optional {col: display_label}."""
        self._pdf_doc = None
        self._current_path = None
        self.save_as_btn.setEnabled(False)
        self.sheet_combo.setVisible(False)
        self._set_pdf_controls_visible(False)
        self._set_zoom_controls_visible(False)
        self.stack.setCurrentWidget(self.table_widget)
        self._populate_table(records, columns, column_labels)

    def _populate_table(self, records, columns=None, column_labels=None):
        """Shared by load_table() (candidate/structured-data rows from the
        agent's own tools) and _load_excel_sheet() (a real .xlsx/.xls file
        opened from disk) -- both just need a QTableWidget filled in, the
        difference is only in what happens to _current_path/sheet_combo
        around the call, which load_table() and _load_excel_sheet() each
        handle themselves."""
        if not records:
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)
            return

        if columns is None:
            columns = []
            for r in records:
                for k in r.keys():
                    if k not in columns:
                        columns.append(k)
        column_labels = column_labels or {}

        self.table_widget.setSortingEnabled(False)
        self.table_widget.setColumnCount(len(columns))
        self.table_widget.setHorizontalHeaderLabels([column_labels.get(c, c) for c in columns])
        self.table_widget.setRowCount(len(records))
        for row_idx, record in enumerate(records):
            for col_idx, col in enumerate(columns):
                val = record.get(col)
                text = "" if val is None else (f"{val:.4g}" if isinstance(val, float) else str(val))
                self.table_widget.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(text))
        self.table_widget.resizeColumnsToContents()
        self.table_widget.setSortingEnabled(True)

    def _set_pdf_controls_visible(self, visible):
        self.prev_btn.setVisible(visible)
        self.next_btn.setVisible(visible)
        self.page_lbl.setVisible(visible)

    def _set_zoom_controls_visible(self, visible):
        self.zoom_out_btn.setVisible(visible)
        self.zoom_lbl.setVisible(visible)
        self.zoom_in_btn.setVisible(visible)

    def _on_open_clicked(self):
        dlg = QtWidgets.QFileDialog(
            self, "Open File", "",
            "Documents (*.pdf *.png *.jpg *.jpeg *.bmp *.gif *.svg *.tif *.tiff "
            "*.txt *.log *.md *.py *.sh *.json *.csv *.tsv *.mgf *.msp *.mat *.xlsx *.xls *.docx *.doc)",
        )
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setStyleSheet(_LIGHT_FILE_DIALOG_STYLE)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        files = dlg.selectedFiles()
        if files:
            self.load_file(files[0])

    def _on_save_as_clicked(self):
        """Save the current Agent preview to a user-selected path and name."""
        if not self._current_path:
            return
        source = Path(self._current_path)
        is_raster = source.suffix.lower() in RASTER_IMAGE_EXTENSIONS
        if is_raster:
            file_filter = (
                "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;"
                "TIFF image (*.tif *.tiff);;BMP image (*.bmp);;All files (*)"
            )
            default_name = source.stem
        else:
            suffix_pattern = f"*{source.suffix}" if source.suffix else "*"
            file_filter = f"Current file type ({suffix_pattern});;All files (*)"
            default_name = source.name

        dlg = QtWidgets.QFileDialog(self, "Save preview as", default_name, file_filter)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setStyleSheet(_LIGHT_FILE_DIALOG_STYLE)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        files = dlg.selectedFiles()
        if not files:
            return
        destination = Path(files[0])

        if is_raster and not destination.suffix:
            extension_by_filter = {
                "PNG image (*.png)": ".png",
                "JPEG image (*.jpg *.jpeg)": ".jpg",
                "TIFF image (*.tif *.tiff)": ".tif",
                "BMP image (*.bmp)": ".bmp",
            }
            destination = destination.with_suffix(
                extension_by_filter.get(dlg.selectedNameFilter(), source.suffix or ".png")
            )
        try:
            destination = Path(self.save_current_to(destination, overwrite=True))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save preview", str(exc))
            return

        self.save_as_btn.setToolTip(f"Saved to {destination}")

    def save_current_to(self, destination, overwrite=False):
        """Save/copy the current preview to an explicit path for UI or Agent use."""
        if not self._current_path:
            raise ValueError("There is no file in the Agent preview to save.")
        source = Path(self._current_path)
        destination = Path(destination).expanduser()
        if not destination.name:
            raise ValueError("The destination must include a file name.")
        if not destination.suffix:
            destination = destination.with_suffix(source.suffix or ".png")
        destination = destination.resolve()
        if destination.exists() and source.resolve() != destination and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.suffix.lower() in RASTER_IMAGE_EXTENSIONS:
            pixmap = self._base_pixmap
            if pixmap is None or pixmap.isNull():
                pixmap = QtGui.QPixmap(str(source))
            if pixmap.isNull() or not pixmap.save(str(destination)):
                raise OSError(f"Could not save image as {destination.suffix or 'the requested format'}.")
        elif source.resolve() != destination:
            shutil.copy2(str(source), str(destination))
        return str(destination)

    def load_file(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        self.save_as_btn.setEnabled(False)
        self._current_path = str(p)
        suffix = p.suffix.lower()
        if suffix in PDF_EXTENSIONS:
            self._load_pdf(str(p))
        elif suffix in SVG_EXTENSIONS:
            self._load_svg(str(p))
        elif suffix in RASTER_IMAGE_EXTENSIONS:
            self._load_raster_image(str(p))
        elif suffix in CSV_EXTENSIONS:
            self._load_csv(str(p))
        elif suffix in EXCEL_EXTENSIONS:
            self._load_excel(str(p))
        elif suffix in WORD_EXTENSIONS:
            self._load_word(str(p))
        elif suffix in LEGACY_WORD_EXTENSIONS:
            self._load_legacy_word(str(p))
        else:
            self._load_text(str(p))
        self.save_as_btn.setEnabled(True)

    # ---------- zoom (shared by raster images / SVG / PDF pages) ----------

    def _zoom_by(self, factor):
        self._set_zoom(self._zoom * factor)

    def _set_zoom(self, value):
        self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, value))
        self.zoom_lbl.setText(f"{round(self._zoom * 100)}%")
        self._apply_zoom()

    def _apply_zoom(self):
        if self._content_kind == "raster" and self._base_pixmap is not None:
            size = self._base_pixmap.size() * self._zoom
            self.image_label.setPixmap(self._base_pixmap.scaled(
                size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            ))
        elif self._content_kind == "svg" and self._svg_renderer is not None:
            self.image_label.setPixmap(self._render_svg_pixmap())
        elif self._content_kind == "pdf" and self._pdf_doc is not None:
            self._render_pdf_page()

    def _reset_zoom_for_new_content(self, kind):
        self._content_kind = kind
        self._zoom = 1.0
        self.zoom_lbl.setText("100%")
        self._set_zoom_controls_visible(True)

    # ---------- raster images (PNG/JPG/BMP/GIF/TIFF) ----------

    def _load_raster_image(self, path):
        self._pdf_doc = None
        self._svg_renderer = None
        self._set_pdf_controls_visible(False)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.image_scroll)

        pixmap = QtGui.QPixmap(path)
        if pixmap.isNull():
            # Qt's own QImageReader plugin set may not include TIFF on every
            # build -- fall back to Pillow, which reads it directly, then
            # hand the decoded RGB(A) buffer to Qt as a QImage.
            pixmap = self._load_via_pillow(path)
        self._base_pixmap = pixmap
        self._reset_zoom_for_new_content("raster")
        self._apply_zoom()

    def _load_via_pillow(self, path):
        try:
            from PIL import Image
        except ImportError:
            return QtGui.QPixmap()  # null pixmap -- image_label just shows nothing
        try:
            img = Image.open(path)
            img = img.convert("RGBA") if img.mode not in ("RGB", "RGBA") else img
            qimage = QtGui.QImage(
                img.tobytes(), img.width, img.height,
                QtGui.QImage.Format_RGBA8888 if img.mode == "RGBA" else QtGui.QImage.Format_RGB888,
            )
            return QtGui.QPixmap.fromImage(qimage.copy())  # .copy(): detach from img's buffer
        except Exception:
            return QtGui.QPixmap()

    # ---------- SVG ----------

    def _load_svg(self, path):
        self._pdf_doc = None
        self._set_pdf_controls_visible(False)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.image_scroll)

        if QtSvg is None:
            # No QtSvg module available in this PyQt5 install -- degrade to
            # showing nothing rather than crashing; open_btn/other formats
            # still work.
            self._svg_renderer = None
            self.image_label.setText("(SVG support not available: PyQt5.QtSvg not installed)")
            self._set_zoom_controls_visible(False)
            return
        self._svg_renderer = QtSvg.QSvgRenderer(path)
        self._reset_zoom_for_new_content("svg")
        self._apply_zoom()

    def _render_svg_pixmap(self):
        natural = self._svg_renderer.defaultSize()
        if natural.isEmpty():
            natural = QtCore.QSize(512, 512)
        target = natural * self._zoom
        pixmap = QtGui.QPixmap(target)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        self._svg_renderer.render(painter)
        painter.end()
        return pixmap

    # ---------- PDF ----------

    def _load_pdf(self, path):
        import fitz
        self._svg_renderer = None
        self._pdf_doc = fitz.open(path)
        self._pdf_page_index = 0
        self._set_pdf_controls_visible(True)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.image_scroll)
        self._reset_zoom_for_new_content("pdf")
        self._render_pdf_page()

    def _render_pdf_page(self):
        import fitz
        from core.i18n import tr
        page = self._pdf_doc[self._pdf_page_index]
        # Base 2x render for readability at 100% zoom (the default 72dpi
        # render is blurry once scaled up in the QLabel), scaled further by
        # the current zoom level -- re-rendered at the target resolution
        # (not a raster QPixmap.scale()) so zoomed-in PDF text stays crisp.
        render_scale = 2 * self._zoom
        pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale))
        image = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(image))
        self.page_lbl.setText(tr(
            "agent.file_viewer_page_label",
            current=self._pdf_page_index + 1, total=self._pdf_doc.page_count,
        ))
        self.prev_btn.setEnabled(self._pdf_page_index > 0)
        self.next_btn.setEnabled(self._pdf_page_index < self._pdf_doc.page_count - 1)

    def _change_page(self, delta):
        if self._pdf_doc is None:
            return
        new_index = self._pdf_page_index + delta
        if 0 <= new_index < self._pdf_doc.page_count:
            self._pdf_page_index = new_index
            self._render_pdf_page()

    def _load_text(self, path):
        self._pdf_doc = None
        self._svg_renderer = None
        self._content_kind = None
        self._set_pdf_controls_visible(False)
        self._set_zoom_controls_visible(False)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.text_edit)
        if Path(path).suffix.lower() in CODE_EXTENSIONS:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            font.setPointSize(10)
            self.text_edit.setFont(font)
        else:
            self.text_edit.setFont(QtGui.QFont())
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            self.text_edit.setPlainText(f.read())

    # ---------- CSV/TSV ----------

    def _load_csv(self, path):
        self._pdf_doc = None
        self._svg_renderer = None
        self._content_kind = None
        self._set_pdf_controls_visible(False)
        self._set_zoom_controls_visible(False)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.table_widget)

        try:
            import pandas as pd
            sep = "\t" if Path(path).suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, low_memory=False)
        except Exception as e:
            self._populate_table([{"Error": f"Could not open table: {e}"}])
            return
        self._populate_dataframe(df)

    # ---------- Excel (.xlsx/.xls) ----------

    def _load_excel(self, path):
        self._pdf_doc = None
        self._svg_renderer = None
        self._content_kind = None
        self._set_pdf_controls_visible(False)
        self._set_zoom_controls_visible(False)
        self.stack.setCurrentWidget(self.table_widget)

        try:
            import pandas as pd
            sheet_names = pd.ExcelFile(path).sheet_names
        except Exception as e:
            self.sheet_combo.setVisible(False)
            self._populate_table([{"Error": f"Could not open workbook: {e}"}])
            return

        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.addItems([str(s) for s in sheet_names])
        self.sheet_combo.blockSignals(False)
        self.sheet_combo.setVisible(len(sheet_names) > 1)
        self._load_excel_sheet(sheet_names[0])

    def _on_sheet_changed(self, index):
        if index < 0 or self._current_path is None:
            return
        p = Path(self._current_path)
        if p.suffix.lower() not in EXCEL_EXTENSIONS:
            return
        self._load_excel_sheet(self.sheet_combo.itemText(index))

    def _load_excel_sheet(self, sheet_name):
        import pandas as pd
        try:
            df = pd.read_excel(self._current_path, sheet_name=sheet_name)
        except Exception as e:
            self._populate_table([{"Error": f"Could not read sheet '{sheet_name}': {e}"}])
            return
        self._populate_dataframe(df)

    def _populate_dataframe(self, df):
        import pandas as pd
        df = df.where(pd.notna(df), None)
        df.columns = [str(c) for c in df.columns]
        self._populate_table(df.to_dict(orient="records"), columns=list(df.columns))

    # ---------- Word (.docx) ----------

    def _load_word(self, path):
        self._pdf_doc = None
        self._svg_renderer = None
        self._content_kind = None
        self._set_pdf_controls_visible(False)
        self._set_zoom_controls_visible(False)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.text_edit)

        try:
            import docx
        except ImportError:
            self.text_edit.setPlainText(
                "(Word document support not available: the 'python-docx' package isn't installed.)"
            )
            return

        try:
            document = docx.Document(path)
        except Exception as e:
            self.text_edit.setPlainText(f"(Could not open Word document: {e})")
            return

        lines = []
        for para in document.paragraphs:
            text = para.text
            if not text.strip():
                lines.append("")
                continue
            style_name = (para.style.name or "") if para.style else ""
            style_lower = style_name.lower()
            if style_lower.startswith("heading") or style_lower == "title":
                digits = "".join(ch for ch in style_lower if ch.isdigit())
                lines.append(f"{'#' * (int(digits) if digits else 1)} {text}")
            else:
                lines.append(text)

        for ti, table in enumerate(document.tables, start=1):
            lines.append("")
            lines.append(f"[Table {ti}]")
            for row in table.rows:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))

        self.text_edit.setPlainText("\n".join(lines) if lines else "(This document has no readable text.)")

    def _load_legacy_word(self, path):
        self._pdf_doc = None
        self._svg_renderer = None
        self._content_kind = None
        self._set_pdf_controls_visible(False)
        self._set_zoom_controls_visible(False)
        self.sheet_combo.setVisible(False)
        self.stack.setCurrentWidget(self.text_edit)
        # Legacy .doc is a pre-2007 binary format that python-docx (used for
        # .docx above) can't read, and this project has no other doc/antiword
        # parser dependency -- rather than crash or silently show nothing,
        # say so plainly with an actionable next step.
        self.text_edit.setPlainText(
            "(This is a legacy .doc file (the pre-2007 Word binary format), "
            "which isn't supported.\n\nOpen it in Word or WPS and use "
            "\"Save As\" → .docx, then open the .docx file here instead.)"
        )
