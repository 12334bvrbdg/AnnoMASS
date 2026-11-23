#!/usr/bin/env python3
"""Splash-screen bootstrap for the AnnoMASS desktop GUI.

WHY THIS FILE EXISTS (not just splash code inside annomass_app.py):
annomass_app.py's own module-level imports (torch, rdkit, gensim,
DreaMS, ...) take many seconds. Python must finish executing a module's
top-level code before any of ITS OWN `if __name__ == "__main__":` code can
run -- so by the time annomass_app.py's own main block starts, the slow
part is already over, and nothing inside that file could show a splash
screen before it. This is a separate, minimal entry point instead: it
imports only PyQt5 (fast) and shows a splash immediately, then imports
annomass_app itself (the slow part) on a background thread while the
splash stays responsive on the main thread, and only constructs the real
AnnoMASS() QMainWindow (which must happen on the main/GUI thread -- Qt
widgets can't be created off-thread) once that import finishes.

Run this instead of `python annomass_app.py` directly (see run_gui.sh).
"""
import os
import sys
import traceback

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from PyQt5 import QtCore, QtGui, QtWidgets

APP_DIR = os.path.dirname(os.path.abspath(__file__))

STARTUP_STEPS = [
    "Starting interface...",
    "Loading analysis modules...",
    "Preparing workspace...",
]


class _BackgroundImport(QtCore.QObject):
    """Runs the slow `import annomass_app` on a worker thread. Plain
    QThread + signals (not the queue/timer workaround used in
    webapi/resources.py) is fine here -- unlike that FastAPI/uvicorn case,
    THIS process's main thread already runs a live Qt event loop
    (app.exec_(), started right after this worker launches), so ordinary
    queued-connection signal delivery works as Qt intends."""
    step = QtCore.pyqtSignal(int)  # index into STARTUP_STEPS
    done = QtCore.pyqtSignal(object)  # the imported module
    failed = QtCore.pyqtSignal(str)

    def run(self):
        try:
            self.step.emit(0)
            import annomass_app as appmod  # the slow import
            self.step.emit(1)
            # Touch a couple of lazily-resolved module attributes so any
            # deferred heavy sub-imports they trigger also happen here, on
            # the background thread, not later during AnnoMASS.__init__
            # on the GUI thread.
            _ = appmod.AnnoMASS
            self.step.emit(2)
            self.done.emit(appmod)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SplashWidget(QtWidgets.QWidget):
    """Small, lightweight splash window shown only while `annomass_app` is
    being imported. Closes once the handoff card has been painted."""

    def __init__(self):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#F8FBFF"))
        self.setPalette(pal)
        self.setStyleSheet("SplashWidget { background: #F8FBFF; }")
        self.setFixedSize(520, 300)
        self._apply_rounded_mask()
        self._center_on_screen()

        self._logo = QtGui.QPixmap(
            os.path.join(APP_DIR, "ui", "assets", "annomass_logo.png")
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._status_lbl = None  # painted manually, see paintEvent
        self._status_text = STARTUP_STEPS[0]
        self._progress = 0.15
        self._error_text = None

        self._pulse = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(80)

    def _center_on_screen(self):
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move(
            screen.x() + (screen.width() - self.width()) // 2,
            screen.y() + (screen.height() - self.height()) // 2,
        )

    def _apply_rounded_mask(self):
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect()), 14, 14)
        region = QtGui.QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

    def _on_tick(self):
        self._pulse += 0.04
        self.update()

    def set_step(self, index: int):
        self._status_text = STARTUP_STEPS[index]
        self._progress = (index + 1) / (len(STARTUP_STEPS) + 0.5)
        self.update()

    def set_ready(self):
        self._status_text = "Ready"
        self._progress = 1.0
        self.update()

    def set_error(self, message: str):
        self._error_text = message
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#F8FBFF"))

        path = QtGui.QPainterPath()
        card_rect = QtCore.QRectF(rect).adjusted(1, 1, -1, -1)
        path.addRoundedRect(card_rect, 16, 16)
        painter.setClipPath(path)

        grad = QtGui.QLinearGradient(card_rect.topLeft(), card_rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 210))
        grad.setColorAt(0.58, QtGui.QColor(247, 250, 255, 188))
        grad.setColorAt(1.0, QtGui.QColor(239, 246, 255, 176))
        painter.fillRect(card_rect, QtGui.QBrush(grad))

        painter.setPen(QtGui.QPen(QtGui.QColor(214, 226, 242, 118), 1))
        painter.drawRoundedRect(card_rect, 16, 16)

        glow_r = 72 + 3 * abs(((self._pulse % 2) - 1))
        glow_center = QtCore.QPointF(rect.width() - 88, 76)
        glow = QtGui.QRadialGradient(glow_center, glow_r)
        glow_color = QtGui.QColor(74, 127, 232, 24)
        glow.setColorAt(0.0, glow_color)
        glow.setColorAt(1.0, QtGui.QColor(74, 127, 232, 0))
        painter.setBrush(QtGui.QBrush(glow))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(glow_center, glow_r, glow_r)

        # Full horizontal wordmark.  It owns the whole header row so the
        # separate splash title can never overlap or obscure its lettering.
        if not self._logo.isNull():
            logo_bounds = QtCore.QRect(42, 26, rect.width() - 84, 98)
            scaled_logo = self._logo.scaled(
                logo_bounds.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            logo_rect = QtCore.QRect(QtCore.QPoint(0, 0), scaled_logo.size())
            logo_rect.moveCenter(logo_bounds.center())
            painter.drawPixmap(logo_rect, scaled_logo)

        subtitle_font = QtGui.QFont("Segoe UI", 9)
        painter.setFont(subtitle_font)
        painter.setPen(QtGui.QColor("#64748B"))
        painter.drawText(QtCore.QRect(42, 124, rect.width() - 84, 20), QtCore.Qt.AlignCenter,
                          "Flavonoid MS/MS workspace")

        tagline_font = QtGui.QFont("Segoe UI", 9)
        painter.setFont(tagline_font)
        painter.setPen(QtGui.QColor("#334155"))
        painter.drawText(QtCore.QRect(42, 150, rect.width() - 84, 20), QtCore.Qt.AlignLeft,
                          "Turning complex spectra into clear flavonoid calls.")
        painter.setPen(QtGui.QColor("#64748B"))
        painter.drawText(QtCore.QRect(42, 174, rect.width() - 84, 20), QtCore.Qt.AlignLeft,
                          "Preparing local models and data resources")

        # Progress bar + status (or error state)
        bar_rect = QtCore.QRect(42, 204, rect.width() - 84, 3)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(226, 232, 240, 210))
        painter.drawRoundedRect(bar_rect, 2, 2)

        if self._error_text is None:
            fill_w = int(bar_rect.width() * min(self._progress, 1.0))
            fill_grad = QtGui.QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
            fill_grad.setColorAt(0, QtGui.QColor("#3B82F6"))
            fill_grad.setColorAt(1, QtGui.QColor("#60A5FA"))
            painter.setBrush(QtGui.QBrush(fill_grad))
            painter.drawRoundedRect(QtCore.QRect(bar_rect.x(), bar_rect.y(), fill_w, bar_rect.height()), 1, 1)

            painter.setFont(QtGui.QFont("Segoe UI", 9))
            painter.setPen(QtGui.QColor("#64748B"))
            painter.drawText(QtCore.QRect(42, 220, rect.width() - 84, 22), QtCore.Qt.AlignLeft, self._status_text)
        else:
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            painter.setPen(QtGui.QColor("#F87171"))
            err_rect = QtCore.QRect(42, 220, rect.width() - 84, 44)
            painter.drawText(err_rect, QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                              f"Startup failed: {self._error_text.strip().splitlines()[-1][:120]}")

        painter.setFont(QtGui.QFont("Segoe UI", 8))
        painter.setPen(QtGui.QColor(148, 163, 184, 160))
        painter.drawText(QtCore.QRect(rect.width() - 90, rect.height() - 26, 76, 16),
                          QtCore.Qt.AlignRight, "v0.1.0")


class HandoffCard(QtWidgets.QWidget):
    """Large handoff card used only while the real main window first paints."""

    def __init__(self, geometry):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setGeometry(geometry.adjusted(-28, -28, 28, 28))
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#09090B"))
        self.setPalette(pal)
        self.setStyleSheet("HandoffCard { background: #09090B; }")
        self._pulse = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(80)

    def _on_tick(self):
        self._pulse += 0.04
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#09090B"))

        card_w = min(760, rect.width() - 96)
        card_h = min(420, rect.height() - 96)
        card_rect = QtCore.QRectF(
            (rect.width() - card_w) / 2,
            (rect.height() - card_h) / 2,
            card_w,
            card_h,
        )

        path = QtGui.QPainterPath()
        path.addRoundedRect(card_rect, 18, 18)
        painter.setClipPath(path)

        grad = QtGui.QLinearGradient(card_rect.topLeft(), card_rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor("#18181B"))
        grad.setColorAt(0.58, QtGui.QColor("#111113"))
        grad.setColorAt(1.0, QtGui.QColor("#0B0B0D"))
        painter.fillRect(card_rect, QtGui.QBrush(grad))

        glow_r = 140 + 6 * abs(((self._pulse % 2) - 1))
        glow_center = QtCore.QPointF(card_rect.right() - 120, card_rect.top() + 100)
        glow = QtGui.QRadialGradient(glow_center, glow_r)
        glow.setColorAt(0.0, QtGui.QColor(84, 130, 255, 34))
        glow.setColorAt(0.62, QtGui.QColor(84, 130, 255, 10))
        glow.setColorAt(1.0, QtGui.QColor(84, 130, 255, 0))
        painter.setBrush(QtGui.QBrush(glow))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(glow_center, glow_r, glow_r)

        painter.setClipping(False)
        painter.setPen(QtGui.QPen(QtGui.QColor("#27272A"), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(card_rect, 18, 18)

        x = int(card_rect.left()) + 54
        y = int(card_rect.top()) + 58
        logo_rect = QtCore.QRect(x, y, 62, 62)
        painter.setBrush(QtGui.QColor("#FAFAFA"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(logo_rect, 14, 14)
        painter.setFont(QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#09090B"))
        painter.drawText(logo_rect, QtCore.Qt.AlignCenter, "FM")

        painter.setFont(QtGui.QFont("Segoe UI", 25, QtGui.QFont.DemiBold))
        painter.setPen(QtGui.QColor("#FAFAFA"))
        painter.drawText(QtCore.QRect(x, y + 94, card_w - 108, 42), QtCore.Qt.AlignLeft, "AnnoMASS Studio")

        painter.setFont(QtGui.QFont("Segoe UI", 11))
        painter.setPen(QtGui.QColor("#A1A1AA"))
        painter.drawText(QtCore.QRect(x, y + 144, card_w - 108, 24), QtCore.Qt.AlignLeft,
                          "Opening the main workspace")

        bar_rect = QtCore.QRect(x, int(card_rect.bottom()) - 72, card_w - 108, 4)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#27272A"))
        painter.drawRoundedRect(bar_rect, 2, 2)
        offset = int((bar_rect.width() + 120) * ((self._pulse * 0.7) % 1.0)) - 120
        chunk = QtCore.QRect(bar_rect.x() + offset, bar_rect.y(), 120, bar_rect.height())
        painter.setBrush(QtGui.QColor("#FAFAFA"))
        painter.drawRoundedRect(chunk.intersected(bar_rect), 2, 2)


def _preferred_main_window_geometry():
    screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
    width = min(max(1100, int(screen.width() * 0.86)), 1440)
    height = min(max(700, int(screen.height() * 0.84)), 920)
    x = screen.x() + (screen.width() - width) // 2
    y = screen.y() + (screen.height() - height) // 2
    return QtCore.QRect(x, y, width, height)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#EFF6FF"))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#F8FBFF"))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#F1F7FF"))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#14324A"))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#14324A"))
    app.setPalette(pal)

    splash = SplashWidget()
    splash.show()
    splash.repaint()
    app.processEvents()

    thread = QtCore.QThread()
    worker = _BackgroundImport()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    state = {"ui": None, "handoff": None}

    def on_step(index):
        splash.set_step(index)

    def on_done(appmod):
        splash.set_ready()
        QtCore.QTimer.singleShot(80, lambda: _show_main_window(appmod))

    def _show_main_window(appmod):
        try:
            app.setStyleSheet(appmod.GLOBAL_MENU_STYLE)
            appmod._install_hard_exit_watchdog(app)

            ui = appmod.AnnoMASS()
            state["ui"] = ui  # keep a reference alive
            target_geometry = _preferred_main_window_geometry()
            ui.setGeometry(target_geometry)

            handoff = HandoffCard(target_geometry)
            state["handoff"] = handoff
            handoff.show()
            handoff.repaint()
            handoff.raise_()
            app.processEvents(QtCore.QEventLoop.AllEvents, 120)

            splash.close()
            ui.show()
            ui.repaint()

            handoff.raise_()
            handoff.activateWindow()
            for delay in (0, 80, 180, 360, 700, 1100):
                QtCore.QTimer.singleShot(delay, handoff.raise_)

            def finish_handoff():
                handoff.close()
                state["handoff"] = None
                ui.raise_()
                ui.activateWindow()

            QtCore.QTimer.singleShot(2200, finish_handoff)
        except Exception:
            splash.set_error(traceback.format_exc())
        thread.quit()

    def on_failed(message):
        splash.set_error(message)
        thread.quit()

    worker.step.connect(on_step)
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
    thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
