"""Left (PSD layer tree) and right (unit / camera properties) panels."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QStyle, QToolButton, QTreeWidget,
    QScrollArea, QTabWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from genparallax.app.i18n import tr, tr_message
from genparallax.document.psd_loader import PsdDocument
from genparallax.scene import CAMERA_PRESETS_MM, FILM_MEASURES, MODEL_UNITS, PX_TO_MM, ReconstructionMode, Scene

KEY_ROLE = Qt.ItemDataRole.UserRole
MEASURE_LABELS = {"horizontal": "Horizontal", "vertical": "Vertical", "diagonal": "Diagonal"}

#: Names as in After Effects' Model Units menu.
MODEL_UNIT_LABELS = {"mm": "mm", "cm": "Centimetres", "m": "Metres", "km": "Kilometres",
                     "in": "Inches", "ft": "Feet", "yd": "Yards", "mi": "Miles"}


class LayerPanel(QWidget):
    selectionChangedUnits = Signal(list)  # unit ids
    visibilityChanged = Signal(str, bool)  # unit id, visible
    mergeRequested = Signal()
    splitRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([tr("Layer"), tr("Unit")])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemChanged.connect(self._on_item_changed)
        merge = QPushButton(tr("Merge into Unit"))
        merge.setToolTip(tr("Combine the selected layers into a single reconstruction unit"))
        merge.clicked.connect(self.mergeRequested)
        split = QPushButton(tr("Split Unit"))
        split.setToolTip(tr("Split the selected unit back into one unit per layer"))
        split.clicked.connect(self.splitRequested)
        buttons = QHBoxLayout()
        buttons.addWidget(merge)
        buttons.addWidget(split)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        layout.addLayout(buttons)
        self.doc: PsdDocument | None = None
        self.scene: Scene | None = None
        self._items: dict[str, QTreeWidgetItem] = {}
        self._updating = False

    def populate(self, doc: PsdDocument | None, scene: Scene | None) -> None:
        self.doc, self.scene = doc, scene
        self._updating = True
        self.tree.clear()
        self._items.clear()
        if doc is not None:
            # Show Photoshop order: top layer first.
            for info in reversed(doc.root):
                self.tree.addTopLevelItem(self._make_item(info))
            self.tree.expandAll()
        self._updating = False
        self.refresh_units()

    def _make_item(self, info) -> QTreeWidgetItem:
        label = ("↳ " if info.clipping else "") + info.name
        item = QTreeWidgetItem([label, ""])
        item.setData(0, KEY_ROLE, info.key)
        tip = [tr("{kind} — {blend}, opacity {opacity}", kind=info.kind, blend=info.blend_mode, opacity=info.opacity),
               tr("bbox {bbox}", bbox=info.bbox)]
        if info.warnings:
            item.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
            tip += [tr_message(w) for w in info.warnings]
        if not info.visible:
            item.setForeground(0, Qt.GlobalColor.gray)
            tip.append(tr("Hidden in the PSD"))
        item.setToolTip(0, "\n".join(tip))
        self._items[info.key] = item
        for child in reversed(info.children):
            item.addChild(self._make_item(child))
        return item

    def refresh_units(self) -> None:
        if self.scene is None:
            return
        self._updating = True
        for key, item in self._items.items():
            unit = self.scene.unit_for_layer(key)
            item.setText(1, unit.name if unit else "")
            if unit is not None and key == unit.source_layers[0]:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if unit.visible else Qt.CheckState.Unchecked)
            else:
                item.setData(0, Qt.ItemDataRole.CheckStateRole, None)
        self._updating = False

    def selected_layer_keys(self) -> list[str]:
        keys = []
        for item in self.tree.selectedItems():
            keys.append(item.data(0, KEY_ROLE))
            if self.doc and self.doc.layers[keys[-1]].is_group:  # a group selects its leaves
                keys += [i.key for i in self.doc.iter_layers(self.doc.layers[keys[-1]].children)]
        return list(dict.fromkeys(keys))

    def selected_unit_ids(self) -> list[str]:
        if self.scene is None:
            return []
        ids = []
        for key in self.selected_layer_keys():
            unit = self.scene.unit_for_layer(key)
            if unit and unit.id not in ids:
                ids.append(unit.id)
        return ids

    def select_units(self, ids: set[str]) -> None:
        if self.scene is None:
            return
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        for uid in ids:
            unit = self.scene.unit_by_id(uid)
            if unit:
                for key in unit.source_layers:
                    if key in self._items:
                        self._items[key].setSelected(True)
                        self.tree.scrollToItem(self._items[key])
        self.tree.blockSignals(False)
        self.selectionChangedUnits.emit(list(ids))

    def _on_selection(self) -> None:
        self.selectionChangedUnits.emit(self.selected_unit_ids())

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0 or self.scene is None:
            return
        unit = self.scene.unit_for_layer(item.data(0, KEY_ROLE))
        state = item.data(0, Qt.ItemDataRole.CheckStateRole)
        if unit is not None and state is not None:
            self.visibilityChanged.emit(unit.id, item.checkState(0) == Qt.CheckState.Checked)


class SliderSpin(QWidget):
    """A double spin box paired with a slider."""

    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, decimals: int = 1, step: float = 1.0, parent=None):
        super().__init__(parent)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 3)
        layout.addWidget(self.spin, 2)
        self.set_range(minimum, maximum)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._guard = False

    def set_range(self, minimum: float, maximum: float) -> None:
        self._min, self._max = minimum, maximum
        self.spin.blockSignals(True)
        self.spin.setRange(minimum, maximum)
        self.spin.blockSignals(False)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, value: float) -> None:
        self._guard = True
        self.spin.setValue(value)
        self.slider.setValue(self._to_slider(value))
        self._guard = False

    def _to_slider(self, v: float) -> int:
        return int(round(1000 * (v - self._min) / max(self._max - self._min, 1e-9)))

    def _from_slider(self, s: int) -> None:
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(self._min + (self._max - self._min) * s / 1000.0)
        self._guard = False
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v: float) -> None:
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(self._to_slider(v))
        self._guard = False
        self.valueChanged.emit(v)


class CollapsibleSection(QWidget):
    """A titled section whose body can be folded away (used for the advanced settings)."""

    def __init__(self, title: str, body: QWidget, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(False)
        self.button.setArrowType(Qt.ArrowType.RightArrow)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; padding: 2px 0; }")
        self.button.toggled.connect(self._toggle)
        self.body = body
        self.body.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.button)
        layout.addWidget(self.body)

    def _toggle(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


def _scroll_page(widgets: list[QWidget]) -> QScrollArea:
    """A vertically scrolling tab page; it is never narrower than its contents."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(4, 4, 4, 4)
    for w in widgets:
        layout.addWidget(w)
    layout.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(page)
    return scroll


class PropertiesPanel(QWidget):
    #: (unit ids, attribute, value, needs_relief_refresh)
    unitParamChanged = Signal(list, str, object, bool)
    renameRequested = Signal(str, str)
    compositionChanged = Signal()
    parallaxChanged = Signal(float, float)  # amplitude fraction, period seconds
    modelUnitChanged = Signal(str)  # After Effects Model Units the GLB is written for
    depthRangeChanged = Signal(float)  # far / near ratio

    def __init__(self, parent=None):
        super().__init__(parent)
        self.unit_ids: list[str] = []
        self.scene: Scene | None = None
        self._loading = False

        # --- unit properties (basic) -------------------------------------
        unit_box = QGroupBox(tr("Reconstruction Unit"))
        form = QFormLayout(unit_box)
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._rename)
        self.info_label = QLabel(tr("No unit selected"))
        self.info_label.setWordWrap(True)
        self.mode_combo = QComboBox()
        for m in ReconstructionMode:
            self.mode_combo.addItem(tr(m.label), m)
        self.mode_combo.currentIndexChanged.connect(
            lambda: self._emit("mode", self.mode_combo.currentData(), True))
        self.z_position = QDoubleSpinBox()
        self.z_position.setDecimals(1)
        self.z_position.setRange(-1.0e7, 1.0e7)
        self.z_position.setSingleStep(10.0)
        self.z_position.setSuffix(" px")
        self.z_position.setKeyboardTracking(False)
        self.z_position.setToolTip(tr("After Effects z: 0 is the composition plane, positive is away from the camera"))
        self.z_position.valueChanged.connect(self._on_z_position)
        self.distance = QDoubleSpinBox()
        self.distance.setDecimals(1)
        self.distance.setRange(0.1, 1.0e7)
        self.distance.setSingleStep(10.0)
        self.distance.setSuffix(" px")
        self.distance.setKeyboardTracking(False)
        self.distance.setToolTip(tr("Distance from the camera in composition pixels (Z position + Zoom)"))
        self.distance.valueChanged.connect(self._on_distance)
        self.strength = SliderSpin(-5000, 5000, 1, 5)
        self.strength.valueChanged.connect(lambda v: self._emit("depth_strength", self._to_internal(v)))
        self.conform = QCheckBox(tr("Lay onto layers below (overlay)"))
        self.conform.setToolTip(tr("For light rays, haze, scattered petals…: take depth from the layers underneath "
                                   "instead of the depth map, which only shows what is behind a translucent layer"))
        self.conform.toggled.connect(lambda v: self._emit("conform", bool(v), True))
        form.addRow(tr("Name"), self.name_edit)
        form.addRow(self.info_label)
        form.addRow(tr("Reconstruction Mode"), self.mode_combo)
        form.addRow(tr("Z Position (px)"), self.z_position)
        form.addRow(tr("Depth Strength (px)"), self.strength)
        form.addRow(self.conform)

        # --- unit properties (advanced) ----------------------------------
        advanced_unit = QWidget()
        dform = QFormLayout(advanced_unit)
        dform.setContentsMargins(0, 0, 0, 0)
        self.offset = SliderSpin(-5000, 5000, 1, 5)
        self.offset.valueChanged.connect(lambda v: self._emit("depth_offset", self._to_internal(v)))
        self.smoothing = SliderSpin(0, 200, 1, 1.0)
        self.smoothing.setToolTip(tr("Gaussian σ in image pixels; large values smooth thin lattice layers such as towers"))
        self._smooth_timer = QTimer(self, singleShot=True, interval=200)
        self._smooth_timer.timeout.connect(lambda: self._emit("depth_smoothing", self.smoothing.value(), True))
        self.smoothing.valueChanged.connect(lambda _v: self._smooth_timer.start())
        self.density = QSpinBox()
        self.density.setRange(1, 128)
        self.density.setSuffix(" px")
        self.density.setKeyboardTracking(False)
        self.density.valueChanged.connect(lambda v: self._emit("mesh_density", int(v)))
        self.density.setToolTip(tr("Grid step; with an adaptive mesh this is the smallest cell"))
        self.adaptive = QCheckBox(tr("Adaptive"))
        self.adaptive.setToolTip(tr("Large cells where the surface is flat, fine cells at silhouettes and depth changes"))
        self.adaptive.toggled.connect(lambda v: self._emit("adaptive_mesh", bool(v)))
        self.detail = QDoubleSpinBox()
        self.detail.setRange(0.001, 1.0)
        self.detail.setDecimals(3)
        self.detail.setSingleStep(0.01)
        self.detail.setKeyboardTracking(False)
        self.detail.setToolTip(tr("Adaptive tolerance in relief units: smaller = more triangles"))
        self.detail.valueChanged.connect(lambda v: self._emit("mesh_detail", float(v)))
        self.edge_inset = QSpinBox()
        self.edge_inset.setRange(0, 32)
        self.edge_inset.setSuffix(" px")
        self.edge_inset.setKeyboardTracking(False)
        self.edge_inset.setToolTip(tr("Depth near the silhouette is taken from this far inside, so blurred AI depth "
                                      "at the outline does not curl the edge"))
        self.edge_inset.valueChanged.connect(lambda v: self._emit("edge_inset", int(v), True))
        self.occlusion_inset = QSpinBox()
        self.occlusion_inset.setRange(0, 64)
        self.occlusion_inset.setSuffix(" px")
        self.occlusion_inset.setKeyboardTracking(False)
        self.occlusion_inset.setToolTip(tr("Depth this close to an opaque layer above is not trusted and is filled in "
                                           "from further away, so the layer does not grow spikes towards that layer"))
        self.occlusion_inset.valueChanged.connect(lambda v: self._emit("occlusion_inset", int(v), True))
        density_row = QHBoxLayout()
        density_row.addWidget(self.density)
        density_row.addWidget(self.adaptive)
        self.custom_depth = QLineEdit()
        self.custom_depth.setReadOnly(True)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_custom_depth)
        clear = QPushButton("×")
        clear.setFixedWidth(28)
        clear.clicked.connect(lambda: self._emit("custom_depth_path", None, True))
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_depth)
        custom_row.addWidget(browse)
        custom_row.addWidget(clear)
        dform.addRow(tr("Camera Distance (px)"), self.distance)
        dform.addRow(tr("Depth Offset (px)"), self.offset)
        dform.addRow(tr("Depth Smoothing (σ px)"), self.smoothing)
        dform.addRow(tr("Edge Inset"), self.edge_inset)
        dform.addRow(tr("Occlusion Inset"), self.occlusion_inset)
        dform.addRow(tr("Custom Depth Map"), custom_row)
        unit_section = CollapsibleSection(tr("Depth Settings"), advanced_unit)
        mesh_body = QWidget()
        self.mesh_form = mform = QFormLayout(mesh_body)
        mform.setContentsMargins(0, 0, 0, 0)
        mform.addRow(tr("Mesh Density"), density_row)
        mform.addRow(tr("Mesh Detail"), self.detail)
        mesh_section = CollapsibleSection(tr("Mesh"), mesh_body)

        # --- composition and camera (basic) ------------------------------
        comp_box = QGroupBox(tr("Composition"))
        cform = QFormLayout(comp_box)
        self.comp_width = QSpinBox()
        self.comp_width.setRange(1, 100000)
        self.comp_width.setSuffix(" px")
        self.comp_width.setKeyboardTracking(False)
        self.comp_width.valueChanged.connect(lambda _v: self._on_composition_size())
        self.comp_height = QSpinBox()
        self.comp_height.setRange(1, 100000)
        self.comp_height.setSuffix(" px")
        self.comp_height.setKeyboardTracking(False)
        self.comp_height.valueChanged.connect(lambda _v: self._on_composition_size())
        size_row = QHBoxLayout()
        size_row.addWidget(self.comp_width)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.comp_height)
        self.image_scale = QDoubleSpinBox()
        self.image_scale.setRange(0.1, 10000.0)
        self.image_scale.setDecimals(1)
        self.image_scale.setSingleStep(1.0)
        self.image_scale.setSuffix(" %")
        self.image_scale.setKeyboardTracking(False)
        self.image_scale.setToolTip(tr("Composition pixels per image pixel; the image stays centred"))
        self.image_scale.valueChanged.connect(self._on_image_scale)
        fit = QPushButton(tr("Fit"))
        fit.setToolTip(tr("Scale the image so that all of it fits in the composition"))
        fit.clicked.connect(lambda: self._set_image_scale(cover=False))
        fill = QPushButton(tr("Fill"))
        fill.setToolTip(tr("Scale the image so that it covers the composition"))
        fill.clicked.connect(lambda: self._set_image_scale(cover=True))
        actual = QPushButton(tr("100%"))
        actual.clicked.connect(lambda: self._set_image_scale(None))
        scale_row = QHBoxLayout()
        scale_row.addWidget(self.image_scale, 2)
        scale_row.addWidget(fit)
        scale_row.addWidget(fill)
        scale_row.addWidget(actual)
        self.preset = QComboBox()
        for mm in CAMERA_PRESETS_MM:
            self.preset.addItem(f"{mm:g} mm", mm)
        self.preset.addItem(tr("Custom"), None)
        self.preset.setToolTip(tr("After Effects focal-length presets for a 36 mm film"))
        self.preset.currentIndexChanged.connect(self._on_preset)
        self.focal = QDoubleSpinBox()
        self.focal.setRange(0.1, 10000.0)
        self.focal.setDecimals(2)
        self.focal.setSingleStep(1.0)
        self.focal.setSuffix(" mm")
        self.focal.setKeyboardTracking(False)
        self.focal.valueChanged.connect(self._on_focal)
        cform.addRow(tr("Composition Size (px)"), size_row)
        cform.addRow(tr("Image Scale (%)"), scale_row)
        cform.addRow(tr("Lens Preset"), self.preset)
        cform.addRow(tr("Focal Length (mm)"), self.focal)

        # --- composition and camera (advanced) ---------------------------
        advanced_camera = QWidget()
        cam_form = QFormLayout(advanced_camera)
        cam_form.setContentsMargins(0, 0, 0, 0)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(0.1, 179.0)
        self.angle.setDecimals(2)
        self.angle.setSuffix("°")
        self.angle.setKeyboardTracking(False)
        self.angle.setToolTip(tr("Editing Zoom or Angle of View changes the focal length, as in After Effects"))
        self.angle.valueChanged.connect(self._on_angle)
        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(0.1, 1.0e7)
        self.zoom.setDecimals(1)
        self.zoom.setKeyboardTracking(False)
        self.zoom.setToolTip(tr("Editing Zoom or Angle of View changes the focal length, as in After Effects"))
        self.zoom.valueChanged.connect(self._on_zoom)
        self.film_size = QDoubleSpinBox()
        self.film_size.setRange(0.1, 1000.0)
        self.film_size.setDecimals(2)
        self.film_size.setSuffix(" mm")
        self.film_size.setKeyboardTracking(False)
        self.film_size.valueChanged.connect(self._on_film_size)
        self.measure = QComboBox()
        for value in FILM_MEASURES:
            self.measure.addItem(tr(MEASURE_LABELS[value]), value)
        self.measure.currentIndexChanged.connect(self._on_measure)
        self.units = QComboBox()
        self.units.addItem(tr("px"), "px")
        self.units.addItem(tr("mm"), "mm")
        self.units.setToolTip(tr("Unit used for Zoom and the composition size readout; layer Z stays in pixels"))
        self.units.currentIndexChanged.connect(self._on_units_changed)
        self.measured = QLabel("—")
        self.measured.setToolTip(tr("Size of the composition along the measured film dimension"))
        self.depth_range = QDoubleSpinBox()
        self.depth_range.setRange(1.05, 100.0)
        self.depth_range.setDecimals(2)
        self.depth_range.setSingleStep(0.25)
        self.depth_range.setPrefix("× ")
        self.depth_range.setKeyboardTracking(False)
        self.depth_range.setToolTip(tr("Farthest depth-map value lands at this multiple of the nearest (Zoom) "
                                       "distance.\nChanging it re-assigns all depths from the depth map."))
        self.depth_range.valueChanged.connect(lambda v: None if self._loading else self.depthRangeChanged.emit(v))
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.0, 0.5)
        self.amplitude.setSingleStep(0.01)
        self.amplitude.setValue(0.05)
        self.period = QDoubleSpinBox()
        self.period.setRange(0.5, 30.0)
        self.period.setValue(4.0)
        self.period.setSuffix(" s")
        for w in (self.amplitude, self.period):
            w.valueChanged.connect(lambda _v: self.parallaxChanged.emit(self.amplitude.value(), self.period.value()))
        self.model_unit = QComboBox()
        for value in MODEL_UNITS:
            self.model_unit.addItem(tr(MODEL_UNIT_LABELS[value]), value)
        self.model_unit.setCurrentIndex(list(MODEL_UNITS).index("m"))
        self.model_unit.setToolTip(tr("Pick the same Model Units in After Effects' Model Settings and keep Object "
                                      "Scale at 100 %: one composition pixel is then one pixel in After Effects"))
        self.model_unit.currentIndexChanged.connect(
            lambda _i: None if self._loading else self.modelUnitChanged.emit(self.model_unit.currentData()))
        cam_form.addRow(tr("Angle of View (°)"), self.angle)
        cam_form.addRow(tr("Zoom"), self.zoom)
        cam_form.addRow(tr("Film Size (mm)"), self.film_size)
        cam_form.addRow(tr("Film Size Measured"), self.measure)
        cam_form.addRow(tr("Units"), self.units)
        cam_form.addRow(tr("Measured Composition Size"), self.measured)
        camera_section = CollapsibleSection(tr("Camera Settings"), advanced_camera)
        scene_body = QWidget()
        sform = QFormLayout(scene_body)
        sform.setContentsMargins(0, 0, 0, 0)
        sform.addRow(tr("Scene Depth Range"), self.depth_range)
        sform.addRow(tr("Parallax Amplitude (× width)"), self.amplitude)
        sform.addRow(tr("Parallax Period"), self.period)
        scene_section = CollapsibleSection(tr("Depth Range and Preview"), scene_body)
        export_body = QWidget()
        eform = QFormLayout(export_body)
        eform.setContentsMargins(0, 0, 0, 0)
        eform.addRow(tr("Export Model Units (AE)"), self.model_unit)
        export_section = CollapsibleSection(tr("Export"), export_body)

        # Two tabs: the selected layer, and the scene as a whole. In each, the group box
        # holds what most users need; the folded sections hold the fine adjustments.
        self.tabs = QTabWidget()
        self.tabs.addTab(_scroll_page([unit_box, unit_section, mesh_section]), tr("Layer"))
        self.tabs.addTab(_scroll_page([comp_box, camera_section, scene_section, export_section]), tr("Scene"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self._unit_widgets = [unit_box, unit_section, mesh_section]
        self._composition_widgets = [comp_box, camera_section, scene_section, export_section]
        self.set_scene(None)

    # ------------------------------------------------------------------
    def minimumSizeHint(self):  # noqa: N802 (Qt API)
        """Wide enough for the widest page and its expanded sections, plus a scroll bar."""
        hint = super().minimumSizeHint()
        width = 0
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i).widget()
            width = max(width, page.sizeHint().width())
            margins = page.layout().contentsMargins()
            for section in page.findChildren(CollapsibleSection):  # folded ones count as open
                width = max(width, section.body.sizeHint().width() + margins.left() + margins.right())
        width += self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent) + 8
        hint.setWidth(max(hint.width(), width))
        return hint

    def set_scene(self, scene: Scene | None) -> None:
        self.scene = scene
        for w in self._composition_widgets:
            w.setEnabled(scene is not None)
        if scene is not None:
            comp = scene.composition
            zoom = max(comp.zoom_px, 1.0)
            for w in (self.strength, self.offset):
                w.set_range(round(-2.0 * zoom, 1), round(2.0 * zoom, 1))
            self._loading = True
            self.comp_width.setValue(comp.width)
            self.comp_height.setValue(comp.height)
            self.image_scale.setValue(comp.image_scale * 100.0)
            self.film_size.setValue(comp.film_size_mm)
            self.measure.setCurrentIndex(FILM_MEASURES.index(comp.measure) if comp.measure in FILM_MEASURES else 0)
            self._refresh_lens_views()
            self.depth_range.setValue(scene.depth_far_ratio)
            self.model_unit.setCurrentIndex(list(MODEL_UNITS).index(scene.export_settings.model_unit))
            self._loading = False
        self.set_units(scene, self.unit_ids if scene is not None else [])

    def set_units(self, scene: Scene | None, ids: list[str]) -> None:
        self.scene = scene or self.scene
        units = [u for u in (self.scene.unit_by_id(i) for i in ids) if u] if self.scene else []
        self.unit_ids = [u.id for u in units]
        for w in self._unit_widgets:
            w.setEnabled(bool(units))
        if not units:
            self.info_label.setText(tr("No unit selected"))
            self.name_edit.clear()
            return
        u = units[0]
        self._loading = True
        self.name_edit.setText(u.name if len(units) == 1 else tr("{n} units", n=len(units)))
        self.name_edit.setEnabled(len(units) == 1)
        l, t, r, b = u.bbox
        verts = len(u.mesh.vertices) if u.mesh is not None else 0
        self.info_label.setText(
            tr("{layers} layer(s), {w}×{h} px at ({x}, {y}), {verts} vertices", layers=len(u.source_layers),
               w=r - l, h=b - t, x=l, y=t, verts=verts)
            + ("" if len(units) == 1 else "\n" + tr("Editing {n} units together", n=len(units)))
        )
        self.mode_combo.setCurrentIndex(list(ReconstructionMode).index(u.mode))
        self.z_position.setValue(self.scene.to_ae_z(u.effective_base_depth))
        self.distance.setValue(u.effective_base_depth * self.scene.composition.image_scale)
        self.strength.set_value(self._to_composition(u.depth_strength))
        self.offset.set_value(self._to_composition(u.depth_offset))
        self.smoothing.set_value(u.depth_smoothing)
        self.density.setValue(u.mesh_density)
        self.adaptive.setChecked(u.adaptive_mesh)
        self.detail.setValue(u.mesh_detail)
        self.detail.setEnabled(u.adaptive_mesh)
        self.edge_inset.setValue(u.edge_inset)
        self.occlusion_inset.setValue(u.occlusion_inset)
        self.conform.setChecked(u.conform)
        for w in (self.z_position, self.distance, self.strength):  # derived from the layers below for overlays
            w.setEnabled(not u.conform)
        self.custom_depth.setText(u.custom_depth_path or "")
        self._loading = False

    # ---------------------------------------------------------- composition
    def _scale(self) -> float:
        return self.scene.composition.image_scale if self.scene is not None else 1.0

    def _to_internal(self, value: float) -> float:
        return value / self._scale()

    def _to_composition(self, value: float) -> float:
        return value * self._scale()

    def _refresh_lens_views(self) -> None:
        if self.scene is None:
            return
        comp = self.scene.composition
        self.focal.setValue(comp.focal_length_mm)
        self.angle.setValue(comp.angle_of_view_deg)
        self._refresh_zoom()
        self._refresh_measured()
        self._show_preset(comp.focal_length_mm)

    def _refresh_zoom(self) -> None:
        if self.scene is None:
            return
        zoom = self.scene.composition.zoom_px
        if self.units.currentData() == "mm":
            self.zoom.setSuffix(" mm")
            self.zoom.setValue(zoom * PX_TO_MM)
        else:
            self.zoom.setSuffix(" px")
            self.zoom.setValue(zoom)

    def _refresh_measured(self) -> None:
        if self.scene is None:
            return
        px = self.scene.composition.measured_px
        self.measured.setText(f"{px * PX_TO_MM:.1f} mm" if self.units.currentData() == "mm" else f"{px:.1f} px")

    def _on_units_changed(self) -> None:
        if self.scene is None:
            return
        self._loading = True
        self._refresh_zoom()
        self._refresh_measured()
        self._loading = False

    def _show_preset(self, focal_mm: float) -> None:
        index = next((i for i in range(self.preset.count())
                      if (data := self.preset.itemData(i)) is not None and abs(data - focal_mm) < 0.01),
                     self.preset.count() - 1)
        self.preset.setCurrentIndex(index)

    def _on_composition_size(self) -> None:
        if self._loading or self.scene is None:
            return
        self.scene.composition.width = int(self.comp_width.value())
        self.scene.composition.height = int(self.comp_height.value())
        self._loading = True
        self._refresh_lens_views()
        self._loading = False
        self.compositionChanged.emit()

    def _on_image_scale(self, percent: float) -> None:
        if self._loading or self.scene is None:
            return
        self.scene.composition.image_scale = max(percent, 0.1) / 100.0
        self.compositionChanged.emit()

    def _set_image_scale(self, cover: bool | None) -> None:
        if self.scene is None:
            return
        comp = self.scene.composition
        comp.image_scale = 1.0 if cover is None else comp.fit_scale(self.scene.width, self.scene.height, cover=cover)
        self._loading = True
        self.image_scale.setValue(comp.image_scale * 100.0)
        self._loading = False
        self.compositionChanged.emit()

    def _on_preset(self) -> None:
        if self._loading or self.scene is None:
            return
        mm = self.preset.currentData()
        if mm is not None:
            self._on_focal(float(mm))

    def _on_focal(self, mm: float) -> None:
        if self._loading or self.scene is None:
            return
        self.scene.composition.focal_length_mm = mm
        self._loading = True
        self._refresh_lens_views()
        self._loading = False
        self.compositionChanged.emit()

    def _on_angle(self, degrees: float) -> None:
        if self._loading or self.scene is None:
            return
        self.scene.composition.angle_of_view_deg = degrees
        self._loading = True
        self._refresh_lens_views()
        self._loading = False
        self.compositionChanged.emit()

    def _on_zoom(self, value: float) -> None:
        if self._loading or self.scene is None:
            return
        zoom_px = value / PX_TO_MM if self.units.currentData() == "mm" else value
        self.scene.composition.zoom_px = zoom_px
        self._loading = True
        self._refresh_lens_views()
        self._loading = False
        self.compositionChanged.emit()

    def _on_film_size(self, mm: float) -> None:
        if self._loading or self.scene is None:
            return
        self.scene.composition.film_size_mm = mm
        self._loading = True
        self._refresh_lens_views()
        self._loading = False
        self.compositionChanged.emit()

    def _on_measure(self) -> None:
        if self._loading or self.scene is None:
            return
        self.scene.composition.measure = self.measure.currentData()
        self._loading = True
        self._refresh_lens_views()
        self._loading = False
        self.compositionChanged.emit()

    # ---------------------------------------------------------- unit edits
    def _on_z_position(self, z: float) -> None:
        if self._loading or self.scene is None or not self.unit_ids:
            return
        unit = self.scene.unit_by_id(self.unit_ids[0])
        if unit is None:
            return
        base = z / self._scale() + self.scene.reference_camera.focal_px - unit.depth_offset
        self._loading = True
        self.distance.setValue(z + self.scene.composition.zoom_px)
        self._loading = False
        self._emit("base_depth", max(base, 1.0))

    def _on_distance(self, distance: float) -> None:
        if self._loading or self.scene is None or not self.unit_ids:
            return
        unit = self.scene.unit_by_id(self.unit_ids[0])
        if unit is None:
            return
        base = distance / self._scale() - unit.depth_offset
        self._loading = True
        self.z_position.setValue(distance - self.scene.composition.zoom_px)
        self._loading = False
        self._emit("base_depth", max(base, 1.0))

    def _emit(self, attr: str, value, relief: bool = False) -> None:
        if self._loading or not self.unit_ids:
            return
        self.unitParamChanged.emit(list(self.unit_ids), attr, value, relief)

    def _rename(self) -> None:
        if len(self.unit_ids) == 1 and self.name_edit.text().strip():
            self.renameRequested.emit(self.unit_ids[0], self.name_edit.text().strip())

    def _browse_custom_depth(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Custom Depth Map"), "",
                                              tr("Depth maps (*.png *.tif *.tiff *.jpg *.npy)"))
        if path:
            self.custom_depth.setText(path)
            self._emit("custom_depth_path", path, True)
            if self.mode_combo.currentData() != ReconstructionMode.MANUAL_DEPTH:
                self.mode_combo.setCurrentIndex(list(ReconstructionMode).index(ReconstructionMode.MANUAL_DEPTH))
