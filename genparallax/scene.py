"""Internal scene data model shared by the preview renderer and the exporters.

Coordinate conventions (used everywhere in this package):

* Image space: pixel corners, ``u`` to the right in ``[0, width]``, ``v`` downwards
  in ``[0, height]``. Pixel ``(i, j)`` covers ``[i, i+1] x [j, j+1]``.
* Camera / world space follows glTF and OpenGL: right-handed, ``+Y`` up, and
  the reference camera sits at the origin looking down ``-Z``.
* A point at image position ``(u, v)`` with camera depth ``Z > 0`` lies at::

      X =  (u - cx) * Z / f
      Y = -(v - cy) * Z / f
      world = (X, Y, -Z)

* Units are pixels at the reference plane: a surface at depth ``Z == f`` maps
  one world unit to one image pixel. ``f`` is the reference focal length in
  pixels, which is also After Effects' camera *Zoom* value.
"""

from __future__ import annotations

import enum
import itertools
import math
import uuid
from dataclasses import dataclass, field

import numpy as np


class ReconstructionMode(str, enum.Enum):
    FLAT = "flat"
    DEPTH_SURFACE = "depth_surface"
    SMOOTHED_SURFACE = "smoothed_surface"
    MANUAL_DEPTH = "manual_depth"
    EXCLUDED = "excluded"

    @property
    def label(self) -> str:
        return {
            "flat": "Flat",
            "depth_surface": "Depth Surface",
            "smoothed_surface": "Smoothed Surface",
            "manual_depth": "Manual Depth",
            "excluded": "Excluded",
        }[self.value]


@dataclass
class Camera:
    """A perspective camera described by position / target / up.

    ``fov_y_deg`` is the vertical field of view; ``width``/``height`` are the
    image dimensions in pixels and define the aspect ratio and the principal
    point (always the image centre).
    """

    width: int
    height: int
    fov_y_deg: float = 39.6
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    near: float = 1.0
    far: float = 1.0e6
    projection: str = "perspective"

    # --- intrinsics -----------------------------------------------------
    @property
    def focal_px(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.fov_y_deg) / 2.0)

    @focal_px.setter
    def focal_px(self, value: float) -> None:
        self.fov_y_deg = math.degrees(2.0 * math.atan((self.height / 2.0) / value))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def focal_length_mm(self, film_width_mm: float = 36.0) -> float:
        """Focal length for a horizontal film size (After Effects default 36 mm)."""
        return self.focal_px * film_width_mm / self.width

    # --- matrices -------------------------------------------------------
    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def projection_matrix(self, aspect: float | None = None) -> np.ndarray:
        return perspective(self.fov_y_deg, aspect or self.aspect, self.near, self.far)

    def copy(self) -> "Camera":
        return Camera(
            width=self.width,
            height=self.height,
            fov_y_deg=self.fov_y_deg,
            position=self.position.copy(),
            target=self.target.copy(),
            up=self.up.copy(),
            near=self.near,
            far=self.far,
            projection=self.projection,
        )

    # --- geometry -------------------------------------------------------
    def back_project(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Image coordinates plus camera depth to camera-space points (N, 3).

        Only valid for the reference camera (origin, looking down -Z).
        """
        f = self.focal_px
        x = (u - self.cx) * z / f
        y = -(v - self.cy) * z / f
        return np.stack([x, y, -z], axis=-1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Camera-space points (N, 3) to image coordinates (N, 2)."""
        f = self.focal_px
        z = -points[:, 2]
        u = points[:, 0] * f / z + self.cx
        v = -points[:, 1] * f / z + self.cy
        return np.stack([u, v], axis=-1)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    side = np.cross(fwd, up)
    n = np.linalg.norm(side)
    if n < 1e-9:  # looking straight along up
        side = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(side)
    side /= n
    true_up = np.cross(side, fwd)
    m = np.identity(4)
    m[0, :3] = side
    m[1, :3] = true_up
    m[2, :3] = -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    t = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float32, camera / world space
    triangles: np.ndarray  # (M, 3) uint32
    uvs: np.ndarray  # (N, 2) float32, glTF convention (origin top-left)
    normals: np.ndarray | None = None
    material_ref: str | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.triangles) == 0

    def compute_normals(self) -> None:
        v = self.vertices.astype(np.float64)
        t = self.triangles
        n = np.zeros_like(v)
        if len(t):
            fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
            for k in range(3):
                np.add.at(n, t[:, k], fn)
        length = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(length > 1e-12, n / np.maximum(length, 1e-12), [0.0, 0.0, 1.0])
        self.normals = n.astype(np.float32)


_revisions = itertools.count(1)


def next_revision() -> int:
    """A revision number unique across all units and threads.

    Background passes work on copies of units; unique numbers guarantee that a copy's
    revision can never be mistaken for an unrelated edit of the live unit.
    """
    return next(_revisions)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class ReconstructionUnit:
    """One reconstruction target, built from one or more PSD layers."""

    name: str
    source_layers: list[str]  # PSD layer keys
    texture: np.ndarray  # (h, w, 4) uint8 RGBA, straight alpha, cropped to bbox
    offset: tuple[int, int]  # (left, top) of ``texture`` on the canvas
    id: str = field(default_factory=_new_id)
    base_depth: float = 1000.0  # B_i, pixels along the camera axis
    depth_strength: float = 0.0  # S_i, relief scale in pixels
    depth_offset: float = 0.0  # added to B_i, convenient for nudging
    depth_smoothing: float = 2.0  # Gaussian sigma in pixels
    mesh_density: int = 4  # grid step in pixels (minimum cell size when adaptive)
    adaptive_mesh: bool = True  # quadtree cells, refined at silhouettes and where the relief varies
    mesh_detail: float = 0.03  # adaptive refinement tolerance, in units of the relief R
    edge_inset: int = 2  # px; depth near the silhouette is taken from this far inside
    #: px; depth this close to an opaque layer above is not trusted (see ``trusted_pixels``)
    occlusion_inset: int = 6
    #: Overlay layers (light rays, scattered flowers, haze…) take their depth from the
    #: layers below them instead of the depth map, which only shows what is underneath.
    conform: bool = False
    mode: ReconstructionMode = ReconstructionMode.DEPTH_SURFACE
    custom_depth_path: str | None = None
    visible: bool = True
    order: int = 0  # PSD stacking order, 0 = bottom
    relief: np.ndarray | None = None  # R_i, (h, w) float32, + = further away
    relief_revision: int = 0  # bumped whenever ``relief`` is recomputed
    #: Local push-back (px) from the layer-order constraints; derived, not saved.
    depth_push: np.ndarray | None = None
    #: Depth statistics measured from the depth map: median Z and Z spread (px).
    depth_stats: tuple[float, float] | None = None
    mesh: Mesh | None = None
    transform: np.ndarray = field(default_factory=lambda: np.identity(4))
    # Cache owned by the meshing module (grid topology), not persisted.
    grid_cache: dict | None = field(default=None, repr=False)
    #: Inputs the last layer-order resolution of this unit was computed from.
    resolve_signature: tuple | None = field(default=None, repr=False)
    #: (revision, per-pixel mesh depth) cache owned by the meshing module.
    depth_cache: tuple | None = field(default=None, repr=False)
    revision: int = 0  # bumped whenever the mesh changes

    @property
    def alpha(self) -> np.ndarray:
        return self.texture[:, :, 3]

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.texture.shape[:2]
        return w, h

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        w, h = self.size
        return self.offset[0], self.offset[1], self.offset[0] + w, self.offset[1] + h

    @property
    def effective_base_depth(self) -> float:
        return self.base_depth + self.depth_offset


#: After Effects converts pixels to millimetres at 72 dpi.
PX_TO_MM = 25.4 / 72.0
FILM_MEASURES = ("horizontal", "vertical", "diagonal")
#: After Effects' camera presets (focal length in mm for a 36 mm film).
CAMERA_PRESETS_MM = (15.0, 20.0, 24.0, 28.0, 35.0, 50.0, 80.0, 135.0, 200.0)


@dataclass
class Composition:
    """The After Effects composition and camera the scene is built for.

    Mirrors AE's Camera Settings dialog. Lengths are composition pixels unless a name
    ends in ``_mm``. ``Zoom`` is the distance at which a layer at 100 % scale fills
    exactly the composition; in AE that plane is ``z = 0`` and the default camera sits
    at ``z = -zoom``. The image is placed on that plane, centred, at ``image_scale``.
    """

    width: int
    height: int
    film_size_mm: float = 36.0
    focal_length_mm: float = 50.0
    measure: str = "horizontal"  # which composition dimension the film size spans
    image_scale: float = 1.0  # composition pixels per image pixel

    @property
    def measured_px(self) -> float:
        if self.measure == "vertical":
            return float(self.height)
        if self.measure == "diagonal":
            return math.hypot(self.width, self.height)
        return float(self.width)

    @property
    def zoom_px(self) -> float:
        return self.measured_px * self.focal_length_mm / self.film_size_mm

    @zoom_px.setter
    def zoom_px(self, value: float) -> None:
        self.focal_length_mm = max(value, 1e-3) * self.film_size_mm / self.measured_px

    @property
    def angle_of_view_deg(self) -> float:
        """Angle across the measured dimension (AE's "Angle of View")."""
        return math.degrees(2.0 * math.atan(self.film_size_mm / (2.0 * self.focal_length_mm)))

    @angle_of_view_deg.setter
    def angle_of_view_deg(self, value: float) -> None:
        value = min(max(value, 0.1), 179.0)
        self.focal_length_mm = self.film_size_mm / (2.0 * math.tan(math.radians(value) / 2.0))

    @property
    def fov_y_deg(self) -> float:
        """Vertical field of view of the composition camera."""
        return math.degrees(2.0 * math.atan(self.height / 2.0 / self.zoom_px))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def fit_scale(self, image_width: int, image_height: int, cover: bool = False) -> float:
        sx, sy = self.width / image_width, self.height / image_height
        return max(sx, sy) if cover else min(sx, sy)

    def copy(self) -> "Composition":
        return Composition(self.width, self.height, self.film_size_mm, self.focal_length_mm, self.measure,
                           self.image_scale)


#: After Effects (checked in 27.0) maps 1 metre of an imported 3D model to 512 composition
#: pixels at Model Units "Metres" and Object Scale 100 %, whatever the comp or model size.
#: The model's origin goes to the layer position (the comp centre, z = 0 by default), glTF
#: +Y becomes AE up (-Y) and glTF -Z becomes AE +Z (away from the camera).
AE_PX_PER_METRE = 512.0
#: The Model Units of After Effects' Model Settings dialog, in metres.
MODEL_UNITS: dict[str, float] = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
}


def units_per_px(model_unit: str) -> float:
    """glTF units per composition pixel for a model AE reads in ``model_unit`` at 100 %."""
    return 1.0 / (AE_PX_PER_METRE * MODEL_UNITS[model_unit])


@dataclass
class ExportSettings:
    mode: str = "combined"  # "combined" or "separate"
    #: The unit to pick in After Effects' Model Settings (Object Scale 100 %) so that one
    #: composition pixel here is one pixel in AE.
    model_unit: str = "m"
    origin: str = "reference_plane"  # "reference_plane" or "camera"
    include_camera: bool = True
    output_path: str | None = None
    #: "split": opaque texels as an alpha-tested (MASK) primitive and only translucent
    #: edges / overlays blended; external renderers then cannot hide layers behind
    #: translucent ones by sorting. "blend": one blended primitive per unit.
    alpha_mode: str = "split"

    def __post_init__(self):
        if self.model_unit not in MODEL_UNITS:
            self.model_unit = "m"

    @property
    def scale(self) -> float:
        """World units per composition pixel."""
        return units_per_px(self.model_unit)


@dataclass
class Scene:
    width: int
    height: int
    reference_camera: Camera
    preview_camera: Camera
    composition: Composition = None  # type: ignore[assignment]  (always set by ``create``)
    units: list[ReconstructionUnit] = field(default_factory=list)
    global_depth: np.ndarray | None = None  # (H, W) float32 nearness in [0, 1], 1 = near
    depth_source: str | None = None  # path of the depth map or "depth-anything-v2-small"
    #: Depth-map nearness 1 maps to Z = f and nearness 0 to Z = depth_far_ratio * f (linear in 1/Z).
    depth_far_ratio: float = 3.0
    #: Re-apply the layer-order constraints automatically after edits.
    auto_layer_order: bool = True
    colour_management: str = "sRGB"  # textures are stored and rendered as sRGB-encoded values
    export_settings: ExportSettings = field(default_factory=ExportSettings)

    @classmethod
    def create(cls, width: int, height: int, fov_y_deg: float | None = None,
               composition: Composition | None = None) -> "Scene":
        """A scene for an image of ``width x height`` pixels.

        By default the composition has the image's size with AE's default 50 mm camera.
        ``fov_y_deg`` instead gives the vertical field of view over the image.
        """
        if composition is None:
            composition = Composition(width, height)
            if fov_y_deg is not None:
                composition.zoom_px = (height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
        ref = Camera(width, height)
        preview = ref.copy()
        scene = cls(width, height, ref, preview, composition)
        scene.apply_composition(rescale_units=False)
        preview.fov_y_deg, preview.near, preview.far = ref.fov_y_deg, ref.near, ref.far
        preview.target = np.array([0.0, 0.0, -ref.focal_px])
        return scene

    # ---------------------------------------------------------- composition
    def apply_composition(self, rescale_units: bool = True) -> float:
        """Update the reference camera from the composition; returns the focal ratio.

        Internal lengths are *image* pixels: the reference focal length is
        ``zoom / image_scale``. With ``rescale_units`` all unit depths are scaled by
        the same ratio, which keeps the layout (and the reference view) unchanged.
        Meshes are not rebuilt here.
        """
        ref = self.reference_camera
        old = ref.focal_px
        ref.focal_px = self.composition.zoom_px / self.composition.image_scale
        ref.near, ref.far = ref.focal_px * 0.05, ref.focal_px * 200.0
        ratio = ref.focal_px / old
        if rescale_units and abs(ratio - 1.0) > 1e-12:
            for u in self.units:
                u.base_depth *= ratio
                u.depth_strength *= ratio
                u.depth_offset *= ratio
                if u.depth_push is not None:
                    u.depth_push = u.depth_push * ratio
                if u.depth_stats is not None:
                    u.depth_stats = (u.depth_stats[0] * ratio, u.depth_stats[1] * ratio)
        return ratio

    def to_ae_z(self, depth: float) -> float:
        """Camera depth (image px) to After Effects z (composition px; 0 = the 100 % plane)."""
        return (depth - self.reference_camera.focal_px) * self.composition.image_scale

    def from_ae_z(self, z: float) -> float:
        return z / self.composition.image_scale + self.reference_camera.focal_px

    def image_rect_in_composition(self) -> tuple[float, float, float, float]:
        """(left, top, width, height) of the image on the z = 0 plane, in composition px."""
        s = self.composition.image_scale
        w, h = self.width * s, self.height * s
        return (self.composition.width - w) / 2.0, (self.composition.height - h) / 2.0, w, h

    @property
    def depth_range(self) -> tuple[float, float]:
        zs = [u.effective_base_depth for u in self.units if u.mode != ReconstructionMode.EXCLUDED]
        if not zs:
            f = self.reference_camera.focal_px
            return f, f
        return min(zs), max(zs)

    @property
    def depth_map_range(self) -> tuple[float, float]:
        """(Z_near, Z_far) used to convert depth-map nearness into camera depth."""
        f = self.reference_camera.focal_px
        return f, f * max(self.depth_far_ratio, 1.0001)

    def nearness_to_z(self, nearness: np.ndarray) -> np.ndarray:
        zn, zf = self.depth_map_range
        return 1.0 / (1.0 / zf + np.asarray(nearness, dtype=np.float64) * (1.0 / zn - 1.0 / zf))

    def z_to_nearness(self, z: np.ndarray) -> np.ndarray:
        zn, zf = self.depth_map_range
        return (1.0 / np.asarray(z, dtype=np.float64) - 1.0 / zf) / (1.0 / zn - 1.0 / zf)

    def unit_by_id(self, unit_id: str) -> ReconstructionUnit | None:
        return next((u for u in self.units if u.id == unit_id), None)

    def unit_for_layer(self, layer_key: str) -> ReconstructionUnit | None:
        return next((u for u in self.units if layer_key in u.source_layers), None)

    def active_units(self) -> list[ReconstructionUnit]:
        return [
            u
            for u in self.units
            if u.mode != ReconstructionMode.EXCLUDED and u.mesh is not None and not u.mesh.is_empty
        ]
