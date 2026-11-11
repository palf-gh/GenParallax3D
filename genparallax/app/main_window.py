"""Main application window."""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QLibraryInfo, QSettings, Qt, QTimer, QTranslator
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QScrollArea, QSplitter, QToolBar,
)

from genparallax import __version__
from genparallax.app import i18n
from genparallax.app.i18n import tr, tr_message
from genparallax.app.panels import LayerPanel, PropertiesPanel
from genparallax.app.viewport import Viewport, configure_default_surface_format
from genparallax.app.workers import BackgroundTask, DepthEstimationJob, LayerOrderJob
from genparallax.document.psd_loader import PsdDocument
from genparallax.export.ae import write_ae_metadata
from genparallax.export.glb import export_scene
from genparallax.project import cache_dir_for, load_project, reload_document, save_project
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.constraints import (
    adopt_snapshot, apply_resolution, check_layer_order, resolve_layer_order, snapshot_for_resolution,
)
from genparallax.reconstruction.depth import load_depth_image
from genparallax.reconstruction.estimator import MODEL_ID
from genparallax.reconstruction.meshing import update_unit_mesh
from genparallax.scene import ReconstructionMode, Scene

APP_NAME = "GenParallax3D"
DISPLAY_MODES = [
    ("Original", "original"), ("Reference", "reference"), ("Free Camera", "free"), ("Depth", "depth"),
    ("Difference", "difference"), ("Depth Difference", "depth_difference"), ("Wireframe", "wireframe"),
    ("Parallax", "parallax"),
]
ALPHA_MODES = [("Split Opaque / Translucent (recommended)", "split"), ("Blend Everything", "blend")]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.doc: PsdDocument | None = None
        self.scene: Scene | None = None
        self.project_path: Path | None = None
        self.settings = QSettings(APP_NAME, APP_NAME)
        self.depth_job = DepthEstimationJob(self)
        self.depth_job.finished.connect(self._on_depth_estimated)
        self.depth_job.failed.connect(self._on_depth_failed)
        # One-off heavy work (open, reload, depth, export) runs in a thread; editing is
        # locked meanwhile, but the window and the viewport keep responding.
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_task_finished)
        self.task.failed.connect(self._on_task_failed)
        self._task_callbacks: dict = {}
        # PSD change tracking (Photoshop saves via a temporary file and a rename)
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_psd_changed)
        self._reload_timer = QTimer(self, singleShot=True, interval=800)
        self._reload_timer.timeout.connect(self._auto_reload)
        self._reload_attempts = 0
        # Layer order is kept in the background after edits.
        self._conflict_timer = QTimer(self, singleShot=True, interval=300)
        self._conflict_timer.timeout.connect(self._update_conflicts)
        self.conflict_label = QLabel()
        self.order_job = LayerOrderJob(self)
        self.order_job.finished.connect(self._on_order_resolved)
        self.order_job.failed.connect(self._on_order_failed)
        self._edit_generation = 0  # bumped by structural changes: pending results are dropped
        self._unit_edits: dict[str, int] = {}  # per-unit edit counters: pending results skip these units
        self._order_pending = False
        self._order_in_flight = False  # until its result has been handled on this thread

        self.viewport = Viewport()
        self.layers = LayerPanel()
        self.props = PropertiesPanel()
        props_scroll = QScrollArea()
        props_scroll.setWidgetResizable(True)
        props_scroll.setWidget(self.props)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.layers)
        splitter.addWidget(self.viewport)
        splitter.addWidget(props_scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 900, 360])
        self.setCentralWidget(splitter)

        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(2000)
        log_dock = QDockWidget(tr("Status / Log"), self)
        log_dock.setObjectName("log")
        log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        self.progress = QProgressBar(maximumWidth=180, textVisible=False)
        self.progress.hide()
        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.conflict_label)
        self.statusBar().addPermanentWidget(self.progress)

        self._build_actions(log_dock)
        self._connect()
        self._update_title()
        self.resize(1500, 900)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    # ---------------------------------------------------------------- UI
    def _build_actions(self, log_dock) -> None:
        mb = self.menuBar()
        self.file_menu = file_menu = mb.addMenu(tr("&File"))
        self._add(file_menu, "Open PSD/PSB…", self.open_psd, QKeySequence.StandardKey.Open)
        self._add(file_menu, "Open Project…", self.open_project, "Ctrl+Shift+O")
        self._add(file_menu, "Reload PSD", lambda: self.reload_psd(), "Ctrl+R")
        self.auto_reload_action = QAction(tr("Reload PSD Automatically When Saved"), self, checkable=True)
        self.auto_reload_action.setChecked(self.settings.value("auto_reload", True, type=bool))
        self.auto_reload_action.toggled.connect(lambda v: self.settings.setValue("auto_reload", v))
        file_menu.addAction(self.auto_reload_action)
        file_menu.addSeparator()
        self._add(file_menu, "Save Project", self.save_project, QKeySequence.StandardKey.Save)
        self._add(file_menu, "Save Project As…", lambda: self.save_project(ask=True), QKeySequence.StandardKey.SaveAs)
        file_menu.addSeparator()
        self._add(file_menu, "Export GLB (Combined Scene)…", lambda: self.export_glb("combined"), "Ctrl+E")
        self._add(file_menu, "Export GLB (Separate Objects)…", lambda: self.export_glb("separate"), "Ctrl+Shift+E")
        alpha_menu = file_menu.addMenu(tr("Transparency in Exported GLB"))
        self.alpha_group = QActionGroup(self)
        self.alpha_actions = {}
        for label, mode in ALPHA_MODES:
            act = QAction(tr(label), self, checkable=True)
            act.triggered.connect(lambda _=False, m=mode: self._set_alpha_mode(m))
            self.alpha_group.addAction(act)
            alpha_menu.addAction(act)
            self.alpha_actions[mode] = act
        self.alpha_actions["split"].setChecked(True)
        file_menu.addSeparator()
        self._add(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        self.depth_menu = depth_menu = mb.addMenu(tr("&Depth"))
        self._add(depth_menu, "Load Depth Map…", self.load_depth_map, "Ctrl+D")
        self.estimate_action = self._add(depth_menu, "Estimate Depth (Depth Anything V2 Small)", self.estimate_depth,
                                         "Ctrl+Shift+D")
        self._add(depth_menu, "Clear Depth Map", self.clear_depth)
        depth_menu.addSeparator()
        self._add(depth_menu, "Assign Depths from Depth Map", self.assign_depths, "Ctrl+Shift+A")
        self._add(depth_menu, "Resolve Layer Order", self.resolve_order, "Ctrl+L")
        self.auto_order_action = QAction(tr("Keep Layer Order Automatically"), self, checkable=True, checked=True)
        self.auto_order_action.setToolTip(tr("After each edit, push lower layers back where they would show in front "
                                             "of the layers above them in the PSD"))
        self.auto_order_action.toggled.connect(self._on_auto_order)
        depth_menu.addAction(self.auto_order_action)
        self._add(depth_menu, "Reset Default Layer Depths", self.reset_depths)

        view_menu = mb.addMenu(tr("&View"))
        self.mode_group = QActionGroup(self)
        self.mode_actions = {}
        for i, (label, mode) in enumerate(DISPLAY_MODES):
            act = self._add(view_menu, label, lambda _=False, m=mode: self.set_mode(m), f"{i + 1}")
            act.setCheckable(True)
            self.mode_group.addAction(act)
            self.mode_actions[mode] = act
        view_menu.addSeparator()
        self._add(view_menu, "Reset Preview Camera", self.viewport.reset_camera, "F")
        log_action = log_dock.toggleViewAction()
        view_menu.addAction(log_action)
        language_menu = view_menu.addMenu(tr("Language"))
        language_group = QActionGroup(self)
        for code, name in i18n.LANGUAGES.items():
            act = QAction(name, self, checkable=True, checked=code == i18n.language())
            act.triggered.connect(lambda _=False, c=code: self._set_language(c))
            language_group.addAction(act)
            language_menu.addAction(act)

        help_menu = mb.addMenu(tr("&Help"))
        self._add(help_menu, "Viewport Controls", self._show_controls)
        self._add(help_menu, "About", lambda: QMessageBox.about(
            self, APP_NAME, f"{APP_NAME} {__version__}\n" + tr("Layer-aware 3D parallax reconstruction from PSD/PSB.")))

        self.toolbar = tb = QToolBar(tr("Main"), self)
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.toolbar_actions = []
        for label, slot in (("Open PSD", self.open_psd), ("Load Depth", self.load_depth_map),
                            ("Estimate Depth", self.estimate_depth), ("Export GLB", lambda: self.export_glb(None))):
            self.toolbar_actions.append(tb.addAction(tr(label), slot))
        tb.addSeparator()
        tb.addWidget(QLabel(tr(" Display: ")))
        self.mode_combo = QComboBox()
        for label, mode in DISPLAY_MODES:
            self.mode_combo.addItem(tr(label), mode)
        self.mode_combo.currentIndexChanged.connect(lambda: self.set_mode(self.mode_combo.currentData()))
        tb.addWidget(self.mode_combo)
        self.set_mode("reference")

    def _add(self, menu, text, slot, shortcut=None) -> QAction:
        act = QAction(tr(text), self)
        if shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _connect(self) -> None:
        self.layers.selectionChangedUnits.connect(self._on_units_selected)
        self.layers.visibilityChanged.connect(self._on_visibility)
        self.layers.mergeRequested.connect(self.merge_selected)
        self.layers.splitRequested.connect(self.split_selected)
        self.props.unitParamChanged.connect(self._on_param)
        self.props.renameRequested.connect(self._on_rename)
        self.props.compositionChanged.connect(self._on_composition_changed)
        self.props.parallaxChanged.connect(self._on_parallax)
        self.props.exportScaleChanged.connect(self._on_export_scale)
        self.props.depthRangeChanged.connect(self._on_depth_range)
        self.viewport.unitPicked.connect(self._on_pick)
        self.viewport.statusMessage.connect(self.status)

    # ------------------------------------------------------------ helpers
    def status(self, text: str) -> None:
        self.status_label.setText(text)

    def log(self, text: str, level: str = "info") -> None:
        prefix = {"warning": "⚠ ", "error": "✖ "}.get(level, "")
        self.log_view.appendPlainText(prefix + text)
        self.status(text)

    def _error(self, title: str, message: str, details: str = "") -> None:
        logging.error("%s: %s\n%s", title, message, details)
        self.log(f"{title}: {message}", "error")
        box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _update_title(self) -> None:
        name = self.project_path.name if self.project_path else (self.doc.path.name if self.doc else tr("Untitled"))
        self.setWindowTitle(f"{name} — {APP_NAME}")

    def _last_dir(self) -> str:
        return str(self.settings.value("last_dir", str(Path.home())))

    def _remember_dir(self, path: str | Path) -> None:
        self.settings.setValue("last_dir", str(Path(path).parent))

    def _require_scene(self) -> bool:
        if self.task.running:
            self.status(tr("Busy — please wait until the current task finishes."))
            return False
        if self.scene is None:
            QMessageBox.information(self, APP_NAME, tr("Open a PSD/PSB file first."))
            return False
        return True

    def set_mode(self, mode: str) -> None:
        self.viewport.set_mode(mode)
        self.mode_actions[mode].setChecked(True)
        idx = self.mode_combo.findData(mode)
        if idx != self.mode_combo.currentIndex():
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)

    def _set_language(self, code: str) -> None:
        self.settings.setValue("language", code)
        QMessageBox.information(self, APP_NAME, tr("Language changes take effect after restarting the application."))

    # ---------------------------------------------------- background work
    def _run_task(self, label: str, work, on_done, on_fail=None, lock: bool = True) -> bool:
        """Run ``work()`` in the background thread; ``on_done(result)`` runs on the GUI thread."""
        if self.task.running:
            self.status(tr("Busy — please wait until the current task finishes."))
            return False
        token = object()
        self._task_callbacks[token] = (on_done, on_fail)
        if lock:
            self._lock_editing(True)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status(label)
        self.task.start(work, token)
        return True

    def _finish_task(self, token):
        self._lock_editing(False)
        self.progress.hide()
        return self._task_callbacks.pop(token, (None, None))

    def _on_task_finished(self, token, result) -> None:
        on_done, _ = self._finish_task(token)
        if on_done is not None:
            on_done(result)

    def _on_task_failed(self, token, message: str, details: str) -> None:
        _, on_fail = self._finish_task(token)
        if on_fail is not None:
            on_fail(message, details)
        else:
            self._error(APP_NAME, message, details)

    def _lock_editing(self, locked: bool) -> None:
        for w in (self.layers, self.props, self.file_menu, self.depth_menu):
            w.setEnabled(not locked)
        for act in self.toolbar_actions:
            act.setEnabled(not locked)

    def _run_on_snapshot(self, label: str, operate, on_done) -> bool:
        """Run ``operate(snapshot)`` on a copy of the scene in the background, then take
        the copy over. Editing is locked meanwhile, so nothing can diverge."""
        snapshot = snapshot_for_resolution(self.scene)
        scene = self.scene

        def done(result):
            if self.scene is not scene:  # another document was opened meanwhile
                return
            adopt_snapshot(scene, snapshot)
            self._after_scene_change()
            on_done(result)

        return self._run_task(label, lambda: operate(snapshot), done)

    # ------------------------------------------------------------- files
    def open_psd(self, path: str | None = None) -> None:
        if self.task.running:
            return self.status(tr("Busy — please wait until the current task finishes."))
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, tr("Open PSD/PSB"), self._last_dir(),
                                                  tr("Photoshop (*.psd *.psb);;All files (*)"))
            if not path:
                return
        self._remember_dir(path)

        def work():
            doc = PsdDocument(path)
            return doc, unitlib.scene_from_document(doc)

        def done(result):
            doc, scene = result
            self.project_path = None
            self._set_document(doc, scene)
            self.log(tr("Loaded {name}: {w}×{h}{psb}, {layers} layers, {units} reconstruction units.",
                        name=doc.path.name, w=doc.width, h=doc.height, psb=" (PSB)" if doc.is_psb else "",
                        layers=len(doc.leaves()), units=len(scene.units)))
            for w in doc.warnings:
                self.log(tr_message(w), "warning")

        self._run_task(tr("Loading {name}…", name=Path(path).name), work, done,
                       lambda m, d: self._error(tr("Could not open PSD"), m, d))

    def _set_document(self, doc: PsdDocument, scene: Scene) -> None:
        self._touch()
        self._unit_edits.clear()
        self.doc, self.scene = doc, scene
        self.layers.populate(doc, scene)
        self.props.set_scene(scene)
        self.viewport.set_scene(scene, doc.composite)
        self._update_title()
        self._watch(doc.path)
        self.auto_order_action.setChecked(scene.auto_layer_order)
        self.alpha_actions.get(scene.export_settings.alpha_mode, self.alpha_actions["split"]).setChecked(True)
        self._conflict_timer.start()

    def open_project(self, path: str | None = None) -> None:
        if self.task.running:
            return self.status(tr("Busy — please wait until the current task finishes."))
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, tr("Open Project"), self._last_dir(),
                                                  tr("GenParallax3D project (*.gp3d)"))
            if not path:
                return
        self._remember_dir(path)

        def done(result):
            scene, doc, warnings = result
            self.project_path = Path(path)
            self._set_document(doc, scene)
            self.log(tr("Opened project {name}.", name=Path(path).name))
            for w in warnings + doc.warnings:
                self.log(tr_message(w), "warning")

        self._run_task(tr("Opening project…"), lambda: load_project(path), done,
                       lambda m, d: self._error(tr("Could not open project"), m, d))

    def save_project(self, ask: bool = False) -> None:
        if not self._require_scene():
            return
        path = self.project_path
        if ask or path is None:
            default = str(Path(self._last_dir()) / (self.doc.path.stem + ".gp3d"))
            chosen, _ = QFileDialog.getSaveFileName(self, tr("Save Project"), default,
                                                    tr("GenParallax3D project (*.gp3d)"))
            if not chosen:
                return
            path = Path(chosen)
        try:
            save_project(path, self.scene, self.doc)
        except Exception as exc:  # noqa: BLE001
            return self._error(tr("Could not save project"), str(exc))
        self.project_path = path
        self._remember_dir(path)
        self._update_title()
        self.log(tr("Saved project to {path}.", path=path))

    def _set_alpha_mode(self, mode: str) -> None:
        if self.scene is not None:
            self.scene.export_settings.alpha_mode = mode

    def export_glb(self, mode: str | None) -> None:
        if not self._require_scene():
            return
        mode = mode or self.scene.export_settings.mode
        default = self.scene.export_settings.output_path or str(Path(self._last_dir()) / (self.doc.path.stem + ".glb"))
        path, _ = QFileDialog.getSaveFileName(self, tr("Export GLB ({mode})", mode=tr(mode)), default,
                                              tr("glTF binary (*.glb)"))
        if not path:
            return
        self._remember_dir(path)
        self.scene.export_settings.mode = mode
        self.scene.export_settings.output_path = path
        scene, doc = self.scene, self.doc

        def work():
            result = export_scene(scene, path, mode=mode)
            meta_dir = Path(path).parent / Path(path).stem if mode == "separate" else Path(path).parent
            meta = write_ae_metadata(meta_dir / f"{Path(path).stem}_ae.json", scene, result, doc)
            return result, meta

        def done(payload):
            result, meta = payload
            self.log(tr("Exported {n} GLB file(s) and {meta} to {folder}.", n=len(result.files), meta=meta.name,
                        folder=meta.parent))

        self._run_task(tr("Exporting…"), work, done, lambda m, d: self._error(tr("Export failed"), m, d))

    # ------------------------------------------------------------- depth
    def load_depth_map(self) -> None:
        if not self._require_scene():
            return
        path, _ = QFileDialog.getOpenFileName(self, tr("Load Depth Map"), self._last_dir(),
                                              tr("Depth maps (*.png *.tif *.tiff *.jpg *.npy)"))
        if not path:
            return
        invert = QMessageBox.question(
            self, tr("Depth Map Convention"),
            tr("Is white NEAR in this depth map?\n\nYes: white = near (inverse depth, Depth Anything style)\n"
               "No: white = far (distance)"),
        ) == QMessageBox.StandardButton.No
        try:
            nearness = load_depth_image(path, (self.scene.width, self.scene.height), white_is_near=not invert)
        except Exception as exc:  # noqa: BLE001
            return self._error(tr("Could not load depth map"), str(exc))
        self._apply_global_depth(nearness, path)

    def estimate_depth(self) -> None:
        if not self._require_scene() or self.depth_job.running:
            return
        cache = str(cache_dir_for(self.project_path)) if self.project_path else \
            str(Path(tempfile.gettempdir()) / "genparallax3d_cache")
        self.estimate_action.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.log(tr("Estimating depth with {model} in a worker process (first run downloads the model)…",
                    model=MODEL_ID))
        self.depth_job.start(self.doc.composite, cache)

    def _on_depth_estimated(self, nearness) -> None:
        self.progress.hide()
        self.estimate_action.setEnabled(True)
        if self.scene is None or nearness.shape != (self.scene.height, self.scene.width):
            return self.log(tr("Depth result does not match the current document; ignored."), "warning")
        self._apply_global_depth(nearness, "depth-anything-v2-small")

    def _on_depth_failed(self, message: str) -> None:
        self.progress.hide()
        self.estimate_action.setEnabled(True)
        self.log(tr("Depth estimation failed: {message}", message=message), "error")
        QMessageBox.warning(self, tr("Depth estimation failed"),
                            tr("{message}\n\nInstall the AI extras with:  pip install torch transformers",
                               message=message))

    def _apply_global_depth(self, nearness, source: str) -> None:
        if self.task.running:  # e.g. a depth estimate arriving during an export
            QTimer.singleShot(500, lambda: self._apply_global_depth(nearness, source))
            return

        def done(moved):
            self.log(tr("Applied depth map ({name}); base depth and strength taken from it.",
                        name=Path(source).name))
            self._report_moved(moved)

        self._run_on_snapshot(tr("Building depth meshes…"),
                              lambda snap: unitlib.set_global_depth(snap, nearness, source, assign=True), done)

    def assign_depths(self) -> None:
        if not self._require_scene():
            return
        if self.scene.global_depth is None:
            return self.log(tr("Load or estimate a depth map first."), "warning")

        def done(moved):
            self.log(tr("Re-assigned base depth and strength from the depth map."))
            self._report_moved(moved)

        self._run_on_snapshot(tr("Building depth meshes…"), unitlib.assign_depths_from_map, done)

    def resolve_order(self) -> None:
        if not self._require_scene():
            return

        def done(moved):
            if not moved:
                self.log(tr("No layer-order conflicts."))
            self._report_moved(moved)

        self._run_on_snapshot(tr("Updating layer order…"), resolve_layer_order, done)

    def _report_moved(self, moved) -> None:
        for c in moved:
            unit = self.scene.unit_by_id(c.lower_id)
            if unit is None:
                continue
            uppers = "、".join(self.scene.unit_by_id(i).name for i in c.upper_ids if self.scene.unit_by_id(i))
            self.log(tr("'{name}' moved back {px} px to stay behind {uppers}.", name=unit.name,
                        px=f"{c.depth_px:.1f}", uppers=uppers))

    def clear_depth(self) -> None:
        if not self._require_scene():
            return
        self._run_on_snapshot(tr("Building depth meshes…"),
                              lambda snap: unitlib.set_global_depth(snap, None, None),
                              lambda _r: self.log(tr("Cleared depth map; units are flat.")))

    def reset_depths(self) -> None:
        if not self._require_scene():
            return
        unitlib.assign_default_depths(self.scene)
        for u in self.scene.units:
            update_unit_mesh(u, self.scene.reference_camera)
        self._after_scene_change()

    def _on_depth_range(self, ratio: float) -> None:
        if self.scene is None:
            return
        if self.scene.global_depth is None:
            self.scene.depth_far_ratio = ratio
            return

        def operate(snap):
            snap.depth_far_ratio = ratio
            unitlib.refresh_all(snap, mesh=False, place_overlays=False)
            return unitlib.assign_depths_from_map(snap)

        if not self._run_on_snapshot(tr("Building depth meshes…"), operate, self._report_moved):
            self.props.set_scene(self.scene)  # busy: show the value actually in use

    # ------------------------------------------------------- layer order
    def _update_conflicts(self) -> None:
        if self.scene is None:
            self.conflict_label.clear()
            return
        if self.task.running:  # a locked operation will take the scene over; check after it
            self._conflict_timer.start()
            return
        if not self.scene.auto_layer_order:
            self._show_conflicts(check_layer_order(self.scene))
            return
        # Resolve on a snapshot in the background; the viewport keeps working meanwhile.
        if self._order_in_flight:
            self._order_pending = True  # run again once the current pass finishes
            return
        self._order_pending = False
        token = (id(self.scene), self._edit_generation, dict(self._unit_edits))
        self._order_in_flight = self.order_job.start(self.scene, token)
        self.conflict_label.setText(tr("Updating layer order…"))

    @property
    def layer_order_busy(self) -> bool:
        """True while a layer-order pass is queued, running or waiting to be applied."""
        return self._order_in_flight or self._conflict_timer.isActive()

    def _on_order_resolved(self, payload) -> None:
        self._order_in_flight = False
        (scene_id, generation, edits), snapshot, conflicts = payload
        if self.scene is None or scene_id != id(self.scene) or generation != self._edit_generation:
            self._conflict_timer.start()  # the scene was restructured meanwhile: start over
            return
        # Units edited while the pass ran keep their live state; everything else is applied
        # now, and the next pass only redoes what depends on the edited units.
        edited = {uid for uid, n in self._unit_edits.items() if edits.get(uid, 0) != n}
        apply_resolution(self.scene, snapshot, skip=edited)
        self.viewport.update()
        if edited or self._order_pending:
            self._conflict_timer.start()
        else:
            self._show_conflicts(conflicts)

    def _on_order_failed(self, message: str) -> None:
        self._order_in_flight = False
        self.log(tr("Layer-order update failed: {message}", message=message), "error")
        self.conflict_label.setText(tr("⚠ Layer order not updated"))

    def _show_conflicts(self, conflicts) -> None:
        if conflicts:
            self.conflict_label.setText(tr("⚠ {n} layer-order conflict(s) — Depth ▸ Resolve Layer Order",
                                           n=len(conflicts)))
            self.conflict_label.setToolTip("\n".join(
                tr("{name}: {px} px in front of an upper layer", name=self.scene.unit_by_id(c.lower_id).name,
                   px=f"{c.depth_px:.1f}")
                for c in conflicts if self.scene.unit_by_id(c.lower_id)))
        else:
            self.conflict_label.setText(tr("Layer order OK"))
            self.conflict_label.setToolTip("")

    def _touch(self) -> None:
        """A structural change: background layer-order results for the old state are dropped."""
        self._edit_generation += 1

    def _touch_units(self, units) -> None:
        """Units whose settings changed: background results skip them (the rest still applies)."""
        for unit in units:
            self._unit_edits[unit.id] = self._unit_edits.get(unit.id, 0) + 1

    def _on_auto_order(self, enabled: bool) -> None:
        if self.scene is not None:
            self.scene.auto_layer_order = enabled
            self._conflict_timer.start()

    # ------------------------------------------------------------ reload
    def _watch(self, path: Path) -> None:
        if self.watcher.files():
            self.watcher.removePaths(self.watcher.files())
        if path.exists():
            self.watcher.addPath(str(path))

    def _on_psd_changed(self, path: str) -> None:
        if not self.auto_reload_action.isChecked():
            self.log(tr("The PSD changed on disk (File ▸ Reload PSD to update)."))
            if Path(path).exists():
                self.watcher.addPath(path)  # re-arm after an atomic save
            return
        self._reload_attempts = 0
        self._reload_timer.start()

    def _auto_reload(self) -> None:
        self._reload_attempts += 1
        if not self.reload_psd(quiet_failure=self._reload_attempts < 5):
            self._reload_timer.start(1000)  # busy: try again shortly

    def reload_psd(self, quiet_failure: bool = False) -> bool:
        """Start reloading the PSD in the background; False if it could not start now."""
        if self.scene is None or self.doc is None:
            return True
        if self.task.running:
            return False
        scene, doc = self.scene, self.doc
        selected = self.props.unit_ids

        def done(result):
            new_scene, new_doc, warnings = result
            self._set_document(new_doc, new_scene)
            self.layers.select_units({i for i in selected if new_scene.unit_by_id(i)})
            self.log(tr("Reloaded {name}; settings kept for {n} units.", name=new_doc.path.name, n=len(new_scene.units)))
            for w in warnings:
                self.log(tr_message(w), "warning")

        def failed(message, details):
            if quiet_failure:  # probably still being written: retry
                self._reload_timer.start(1000)
            else:
                self._error(tr("Could not reload PSD"), message, details)
                self._watch(doc.path)

        return self._run_task(tr("Loading {name}…", name=doc.path.name),
                              lambda: reload_document(scene, doc), done, failed)

    # -------------------------------------------------------------- units
    def _after_scene_change(self, ids: list[str] | None = None) -> None:
        self._touch()
        self.layers.refresh_units()
        self.props.set_units(self.scene, ids if ids is not None else self.props.unit_ids)
        self.viewport.update()
        self._conflict_timer.start()

    def _on_units_selected(self, ids: list[str]) -> None:
        self.props.set_units(self.scene, ids)
        self.viewport.set_selected(set(ids))

    def _on_pick(self, uid, additive: bool) -> None:
        if additive:
            ids = set(self.props.unit_ids) ^ ({uid} if uid else set())
        else:
            ids = {uid} if uid else set()
        self.layers.select_units(ids)

    def _on_visibility(self, uid: str, visible: bool) -> None:
        unit = self.scene.unit_by_id(uid)
        if unit:
            self._touch()  # changes which units take part in the layer order
            unit.visible = visible
            self.viewport.update()
            self._conflict_timer.start()

    def _on_param(self, ids: list[str], attr: str, value, relief: bool) -> None:
        units = [u for u in (self.scene.unit_by_id(i) for i in ids) if u is not None]
        if not units:
            return
        self._touch_units(units)
        # With several units selected, depth edits are relative to the first unit, so the
        # selection moves together instead of collapsing onto one plane.
        first = units[0]
        ratio = value / first.base_depth if attr == "base_depth" and first.base_depth else None
        delta = value - getattr(first, attr) if attr in ("depth_strength", "depth_offset") else None
        for unit in units:
            if len(units) > 1 and ratio is not None:
                unit.base_depth *= ratio
            elif len(units) > 1 and delta is not None:
                setattr(unit, attr, getattr(unit, attr) + delta)
            else:
                setattr(unit, attr, value)
            if attr == "mode" and value == ReconstructionMode.MANUAL_DEPTH and not unit.custom_depth_path:
                self.log(tr("{name}: Manual Depth uses a Custom Depth Map (falls back to the global map).",
                            name=unit.name), "warning")
            if relief:
                unitlib.refresh_relief(self.scene, unit)
            update_unit_mesh(unit, self.scene.reference_camera)
        if attr in ("mode", "mesh_density", "adaptive_mesh", "mesh_detail", "edge_inset", "occlusion_inset",
                    "conform", "depth_offset"):
            self.props.set_units(self.scene, ids)
        self.viewport.update()
        # Throttle rather than debounce: while a slider is dragged the pass still runs
        # every few hundred milliseconds, so the layers below follow the drag.
        if not self._conflict_timer.isActive():
            self._conflict_timer.start()

    def _on_rename(self, uid: str, name: str) -> None:
        unit = self.scene.unit_by_id(uid)
        if unit:
            unit.name = name
            self.layers.refresh_units()

    def merge_selected(self) -> None:
        if self.scene is None:
            return
        units = [self.scene.unit_by_id(i) for i in self.layers.selected_unit_ids()]
        units = [u for u in units if u]
        if len(units) < 2:
            return self.log(tr("Select layers from at least two units to merge."), "warning")
        merged = unitlib.merge_units(self.scene, self.doc, units)
        if merged:
            self.log(tr("Merged {n} units into '{name}'.", n=len(units), name=merged.name))
            self._after_scene_change([merged.id])
            self.layers.select_units({merged.id})

    def split_selected(self) -> None:
        if self.scene is None:
            return
        ids = self.layers.selected_unit_ids()
        new_ids = []
        for uid in ids:
            unit = self.scene.unit_by_id(uid)
            if unit:
                new_ids += [u.id for u in unitlib.split_unit(self.scene, self.doc, unit)]
        self._after_scene_change(new_ids)
        self.layers.select_units(set(new_ids))

    # ------------------------------------------------------------- camera
    def _on_composition_changed(self) -> None:
        """Lens, composition size or image scale changed: keep the layout and rebuild meshes."""
        if self.scene is None:
            return
        ids = list(self.props.unit_ids)
        self.scene.apply_composition()  # rescales the unit depths: same relative layout
        for u in self.scene.units:
            update_unit_mesh(u, self.scene.reference_camera)
        self.props.set_scene(self.scene)
        self.viewport.reset_camera()
        self._after_scene_change(ids)

    def _on_parallax(self, amplitude: float, period: float) -> None:
        self.viewport.parallax_amplitude = amplitude
        self.viewport.parallax_period = period

    def _on_export_scale(self, scale: float) -> None:
        if self.scene is not None:
            self.scene.export_settings.scale = scale

    def _show_controls(self) -> None:
        QMessageBox.information(self, tr("Viewport Controls"), tr(
            "Free Camera / Depth / Wireframe:\n"
            "  Left drag: orbit    Middle or Shift+Left drag: pan\n"
            "  Right drag / wheel: dolly    W A S D Q E: move    F: reset\n\n"
            "Click: select unit    Ctrl+Click: add / remove from selection\n"
            "1–8: display modes (Original, Reference, Free, Depth, Difference, Depth Difference, Wireframe, Parallax)\n\n"
            "Depth Difference: red = mesh nearer than the depth map, blue = further"))

    def closeEvent(self, event) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.depth_job.shutdown()
        self.order_job.shutdown()
        self.task.shutdown()
        self.viewport.release_gl()
        super().closeEvent(event)


def install_language(app: QApplication) -> None:
    """Pick the UI language (saved setting, Japanese by default) and Qt's own translations."""
    code = str(QSettings(APP_NAME, APP_NAME).value("language", "ja"))
    i18n.set_language(code)
    if code == "ja":
        translator = QTranslator(app)
        if translator.load("qtbase_ja", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
            app.installTranslator(translator)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    configure_default_surface_format()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    install_language(app)
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        target = argv[1]
        if target.lower().endswith(".gp3d"):
            win.open_project(target)
        else:
            win.open_psd(target)
    return app.exec()
