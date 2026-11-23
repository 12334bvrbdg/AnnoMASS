# -*- coding: utf-8 -*-
# Hand-written replacement for main_v4.py
# Structure:
#   vl_root (QVBoxLayout on centralwidget)
#     workspace_panel (QWidget)
#       hl_workspace (QHBoxLayout)
#         splitter_main (QSplitter Horizontal)
#           activityBar  (QFrame, resizable, 1st pane)
#           leftPanel    (QFrame, 2nd pane)
#           mainPanel    (QFrame, 3rd pane)
#     bottomStrip (QFrame, 22px)

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(1100, 700))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # ── root vertical layout ──────────────────────────────────────────────
        self.vl_root = QtWidgets.QVBoxLayout(self.centralwidget)
        self.vl_root.setContentsMargins(0, 0, 0, 0)
        self.vl_root.setSpacing(0)
        self.vl_root.setObjectName("vl_root")

        # ── workspace panel ───────────────────────────────────────────────────
        self.workspace_panel = QtWidgets.QWidget(self.centralwidget)
        self.workspace_panel.setObjectName("workspace_panel")
        self.hl_workspace = QtWidgets.QHBoxLayout(self.workspace_panel)
        self.hl_workspace.setContentsMargins(0, 0, 0, 0)
        self.hl_workspace.setSpacing(0)
        self.hl_workspace.setObjectName("hl_workspace")

        # ── main horizontal splitter (activityBar | leftPanel | mainPanel) ────
        self.splitter_main = QtWidgets.QSplitter(self.workspace_panel)
        self.splitter_main.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_main.setHandleWidth(6)
        self.splitter_main.setChildrenCollapsible(False)
        self.splitter_main.setObjectName("splitter_main")

        # ACTIVITY BAR ── first pane, resizable
        self.activityBar = QtWidgets.QFrame(self.splitter_main)
        self.activityBar.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.activityBar.setMinimumWidth(180)
        self.activityBar.setMaximumWidth(260)
        self.activityBar.setObjectName("activityBar")

        self.vl_activity = QtWidgets.QVBoxLayout(self.activityBar)
        self.vl_activity.setContentsMargins(0, 0, 0, 0)
        self.vl_activity.setSpacing(0)
        self.vl_activity.setObjectName("vl_activity")

        self.label_logo = QtWidgets.QLabel(self.activityBar)
        self.label_logo.setAlignment(QtCore.Qt.AlignCenter)
        self.label_logo.setFixedHeight(72)
        self.label_logo.setContentsMargins(10, 8, 10, 8)
        self.label_logo.setObjectName("label_logo")
        self.vl_activity.addWidget(self.label_logo)

        self.act_div0 = QtWidgets.QFrame(self.activityBar)
        self.act_div0.setFrameShape(QtWidgets.QFrame.HLine)
        self.act_div0.setFrameShadow(QtWidgets.QFrame.Plain)
        self.act_div0.setMaximumHeight(1)
        self.act_div0.setObjectName("act_div0")
        self.vl_activity.addWidget(self.act_div0)

        for obj_name in ("butt_open",):
            btn = QtWidgets.QPushButton(self.activityBar)
            btn.setFixedHeight(48)
            btn.setObjectName(obj_name)
            self.vl_activity.addWidget(btn)
            setattr(self, obj_name, btn)

        self.act_div1 = QtWidgets.QFrame(self.activityBar)
        self.act_div1.setFrameShape(QtWidgets.QFrame.HLine)
        self.act_div1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.act_div1.setMaximumHeight(1)
        self.act_div1.setObjectName("act_div1")
        self.vl_activity.addWidget(self.act_div1)

        for obj_name in ("butt_pipeline",):
            btn = QtWidgets.QPushButton(self.activityBar)
            btn.setFixedHeight(48)
            btn.setObjectName(obj_name)
            self.vl_activity.addWidget(btn)
            setattr(self, obj_name, btn)

        self.vl_activity.addStretch(1)

        # LEFT PANEL ── sidebar with spectrum list
        self.leftPanel = QtWidgets.QFrame(self.splitter_main)
        self.leftPanel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.leftPanel.setMinimumWidth(100)
        self.leftPanel.setObjectName("leftPanel")

        self.vl_left = QtWidgets.QVBoxLayout(self.leftPanel)
        self.vl_left.setContentsMargins(0, 0, 0, 0)
        self.vl_left.setSpacing(0)
        self.vl_left.setObjectName("vl_left")

        self.label_list_title = QtWidgets.QLabel(self.leftPanel)
        self.label_list_title.setObjectName("label_list_title")
        self.vl_left.addWidget(self.label_list_title)

        self.list_spectrum = QtWidgets.QListWidget(self.leftPanel)
        self.list_spectrum.setObjectName("list_spectrum")
        self.vl_left.addWidget(self.list_spectrum)

        # MAIN PANEL ── editor area (right of sidebar)
        self.mainPanel = QtWidgets.QFrame(self.splitter_main)
        self.mainPanel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mainPanel.setObjectName("mainPanel")

        self.vl_main = QtWidgets.QVBoxLayout(self.mainPanel)
        self.vl_main.setContentsMargins(0, 0, 0, 0)
        self.vl_main.setSpacing(0)
        self.vl_main.setObjectName("vl_main")

        # Results panel fills the whole main panel now. There used to be a
        # vertical splitter here (results tabs on top, a separate "Spectrum
        # View" + "Structure View" viewer panel on the bottom) -- removed:
        # the results tab has its own per-candidate spectrum/structure/
        # cluster-map preview already, so the bottom viewer panel was a
        # redundant second copy of the same information.
        self.resultsPanel = QtWidgets.QWidget(self.mainPanel)
        self.resultsPanel.setObjectName("resultsPanel")
        self.vl_results = QtWidgets.QVBoxLayout(self.resultsPanel)
        self.vl_results.setContentsMargins(0, 0, 0, 0)
        self.vl_results.setObjectName("vl_results")

        self.tabWidget_results = QtWidgets.QTabWidget(self.resultsPanel)
        self.tabWidget_results.setObjectName("tabWidget_results")

        self.vl_results.addWidget(self.tabWidget_results)

        self.vl_main.addWidget(self.resultsPanel)

        # ── assemble workspace ────────────────────────────────────────────────
        self.hl_workspace.addWidget(self.splitter_main, 1)      # splitter fills all space

        # ── add workspace to root ─────────────────────────────────────────────
        self.vl_root.addWidget(self.workspace_panel, 1)

        # ── bottom status strip ───────────────────────────────────────────────
        self.bottomStrip = QtWidgets.QFrame(self.centralwidget)
        self.bottomStrip.setFixedHeight(22)
        self.bottomStrip.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.bottomStrip.setObjectName("bottomStrip")
        self.hl_bottom = QtWidgets.QHBoxLayout(self.bottomStrip)
        self.hl_bottom.setContentsMargins(8, 1, 8, 1)
        self.hl_bottom.setObjectName("hl_bottom")
        self.progressBar = QtWidgets.QProgressBar(self.bottomStrip)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.hl_bottom.addWidget(self.progressBar)

        self.vl_root.addWidget(self.bottomStrip, 0)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _t = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_t("MainWindow", "AnnoMASS"))
        self.label_logo.setText(_t("MainWindow", "FM"))
        self.butt_open.setText(_t("MainWindow", "Open"))
        self.butt_open.setToolTip(_t("MainWindow", "Open spectrum file"))
        self.butt_pipeline.setText(_t("MainWindow", "Run"))
        self.butt_pipeline.setToolTip(_t("MainWindow", "Run AnnoMASS flavonoid identification"))
        self.label_list_title.setText(_t("MainWindow", "SPECTRA"))

        self.progressBar.setFormat(_t("MainWindow", "Ready"))
