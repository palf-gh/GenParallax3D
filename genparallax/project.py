"""Project file (``.gp3d``, JSON) save/load, and reloading an edited PSD.

Only settings are stored. Layer images are re-read from the PSD, and large data
(AI depth maps) live in a cache folder next to the project, so a re-edited PSD
keeps its depth settings: units are matched to layers by layer id, then by path.
The same matching is used when the PSD changes while the application is open.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from genparallax import __version__
from genparallax.document.psd_loader import PsdDocument
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.constraints import is_rigid, resolve_layer_order
from genparallax.reconstruction.depth import load_depth_image
from genparallax.scene import Composition, ExportSettings, ReconstructionMode, Scene

FORMAT = "genparallax3d-project"
FORMAT_VERSION = 2  # v2: adaptive meshing, edge inset, depth-map range

UNIT_DEFAULTS = {
    "depth_offset": 0.0,
    "depth_smoothing": 2.0,
    "mesh_density": 8,  # v1 projects used uniform 8 px grids
    "adaptive_mesh": False,
    "mesh_detail": 0.03,
    # Projects from before simplification get it too: it keeps the look and the
    # reference view, and only removes vertices.
    "mesh_simplify": "standard",
    "edge_inset": 0,
    "occlusion_inset": 6,
    "visible": True,
}


def cache_dir_for(project_path: str | Path) -> Path:
    p = Path(project_path)
    return p.parent / ".gp3d_cache" / p.stem


def _relative_to(base: Path | None):
    def rel(p: str | Path | None) -> str | None:
        if p is None:
            return None
        if base is None:
            return str(Path(p).resolve())
        try:
            return str(Path(p).resolve().relative_to(base.resolve()))
        except ValueError:
            return str(Path(p).resolve())
    return rel


def _resolver(base: Path | None):
    def resolve(p: str | None) -> Path | None:
        if p is None:
            return None
        q = Path(p)
        return q if q.is_absolute() or base is None else (base / q)
    return resolve


def scene_to_dict(scene: Scene, doc: PsdDocument, base: Path | None = None,
                  depth_cache: Path | None = None) -> dict:
    """Serialisable settings. Paths are relative to ``base`` when given."""
    rel = _relative_to(base)
    ref = scene.reference_camera
    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "generator": f"GenParallax3D {__version__}",
        "psd": rel(doc.path),
        "image_size": [scene.width, scene.height],
        "camera": {"fov_y_deg": ref.fov_y_deg, "near": ref.near, "far": ref.far},
        "composition": dataclasses.asdict(scene.composition),
        "depth": {"source": scene.depth_source, "cache": rel(depth_cache), "far_ratio": scene.depth_far_ratio,
                  "auto_layer_order": scene.auto_layer_order},
        "colour_management": scene.colour_management,
        "export": vars(scene.export_settings),
        "units": [
            {
                "id": u.id,
                "name": u.name,
                "layers": [
                    {"key": k, "layer_id": doc.layers[k].layer_id, "path": doc.layers[k].path}
                    for k in u.source_layers if k in doc.layers
                ],
                "base_depth": u.base_depth,
                "depth_strength": u.depth_strength,
                "depth_offset": u.depth_offset,
                "depth_smoothing": u.depth_smoothing,
                "mesh_density": u.mesh_density,
                "adaptive_mesh": u.adaptive_mesh,
                "mesh_detail": u.mesh_detail,
                "mesh_simplify": u.mesh_simplify,
                "edge_inset": u.edge_inset,
                "occlusion_inset": u.occlusion_inset,
                "conform": u.conform,
                "mode": u.mode.value,
                "custom_depth_path": rel(u.custom_depth_path),
                "visible": u.visible,
            }
            for u in scene.units
        ],
    }


def scene_from_dict(data: dict, doc: PsdDocument, base: Path | None = None,
                    global_depth: np.ndarray | None = None) -> tuple[Scene, list[str]]:
    """Rebuild a scene for ``doc`` from saved settings. Returns ``(scene, warnings)``."""
    resolve = _resolver(base)
    warnings: list[str] = []
    if [doc.width, doc.height] != data.get("image_size"):
        warnings.append(f"PSD size changed from {data.get('image_size')} to {[doc.width, doc.height]}.")

    if "composition" in data:
        scene = Scene.create(doc.width, doc.height, composition=Composition(**data["composition"]))
    else:  # projects from before compositions: the image filled the frame
        scene = Scene.create(doc.width, doc.height, data["camera"]["fov_y_deg"])
    scene.colour_management = data.get("colour_management", "sRGB")
    export = dict(data.get("export", {}))
    # Before the After Effects check the scale was a free number (0.01 per px by default),
    # which AE showed 5.12 times too large; such projects now export in metres.
    export.pop("scale", None)
    scene.export_settings = ExportSettings(**export)

    depth = data.get("depth", {})
    scene.depth_far_ratio = depth.get("far_ratio", 3.0)
    scene.auto_layer_order = depth.get("auto_layer_order", True)
    if global_depth is not None:
        scene.global_depth = global_depth
        scene.depth_source = depth.get("source")
    else:
        cache_file = resolve(depth.get("cache"))
        source = resolve(depth.get("source"))
        if cache_file and cache_file.exists():
            scene.global_depth = np.load(cache_file)
            scene.depth_source = depth.get("source")
        elif source and source.exists():
            scene.global_depth = load_depth_image(source, (doc.width, doc.height))
            scene.depth_source = depth["source"]
        elif depth.get("source"):
            warnings.append("Cached depth map is missing; run depth estimation again.")
    if scene.global_depth is not None and scene.global_depth.shape != (doc.height, doc.width):
        warnings.append("The depth map does not match the PSD size and was discarded.")
        scene.global_depth = None

    claimed: set[str] = set()
    for entry in data["units"]:
        keys = []
        for ref in entry["layers"]:
            info = doc.find_layer(ref.get("layer_id"), ref.get("path"))
            if info is None:
                warnings.append(f"Layer '{ref.get('path')}' no longer exists in the PSD.")
            elif info.key not in claimed:
                keys.append(info.key)
        if not keys:
            continue
        unit = unitlib.make_unit(doc, keys, entry["name"])
        if unit is None:
            continue
        unit.id = entry["id"]
        unit.base_depth = entry["base_depth"]
        unit.depth_strength = entry["depth_strength"]
        for attr, default in UNIT_DEFAULTS.items():
            setattr(unit, attr, entry.get(attr, default))
        unit.conform = entry.get("conform", unit.conform)  # older projects: auto-detected
        unit.mode = ReconstructionMode(entry.get("mode", "depth_surface"))
        cdp = resolve(entry.get("custom_depth_path"))
        unit.custom_depth_path = str(cdp) if cdp else None
        claimed.update(unit.source_layers)
        scene.units.append(unit)

    # Layers added to the PSD since the settings were saved get default units at the
    # front; the layer-order pass below pushes them behind the layers above them.
    f = scene.reference_camera.focal_px
    new_units = []
    for info in doc.leaves():
        if info.key in claimed or not info.reconstructable or not doc.effectively_visible(info.key):
            continue
        unit = unitlib.make_unit(doc, [info.key], info.name)
        if unit is None:
            continue
        unit.base_depth = f
        scene.units.append(unit)
        new_units.append(unit)
        warnings.append(f"New layer '{info.path}' added.")

    unitlib.refresh_all(scene)
    unitlib.conform_overlays(scene)
    for unit in new_units:
        if unit.depth_stats is not None:  # take the new layer's depth from the depth map
            unit.base_depth, unit.depth_strength = unit.depth_stats
            unitlib.update_unit_mesh(unit, scene.reference_camera)
    # Push fields are derived from the settings and not saved, so rebuild them on every
    # load, whatever the auto-order setting (this also places new layers).
    known = {u.id for u in scene.units} - {u.id for u in new_units}
    for c in resolve_layer_order(scene):
        unit = scene.unit_by_id(c.lower_id)
        if unit.id not in known or is_rigid(unit):
            warnings.append(f"'{unit.name}' was moved back {c.depth_px:.1f} px to keep the PSD layer order.")
    return scene, warnings


def save_project(path: str | Path, scene: Scene, doc: PsdDocument) -> Path:
    path = Path(path)
    depth_file = None
    if scene.global_depth is not None:
        cache = cache_dir_for(path)
        cache.mkdir(parents=True, exist_ok=True)
        depth_file = cache / "global_depth.npy"
        np.save(depth_file, scene.global_depth)
    data = scene_to_dict(scene, doc, path.parent, depth_file)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_project(path: str | Path) -> tuple[Scene, PsdDocument, list[str]]:
    """Returns ``(scene, document, warnings)``."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a GenParallax3D project")
    if data.get("version", 1) > FORMAT_VERSION:
        raise ValueError(f"{path} was saved by a newer version of GenParallax3D")
    doc = PsdDocument(_resolver(path.parent)(data["psd"]))
    scene, warnings = scene_from_dict(data, doc, path.parent)
    return scene, doc, warnings


def reload_document(scene: Scene, doc: PsdDocument, path: str | Path | None = None
                    ) -> tuple[Scene, PsdDocument, list[str]]:
    """Re-read the PSD (after it was edited) and carry the current settings over.

    The depth map is kept when the size is unchanged; it is flagged as possibly
    stale when the merged image changed.
    """
    data = scene_to_dict(scene, doc)
    new_doc = PsdDocument(path or doc.path)
    keep_depth = scene.global_depth is not None and scene.global_depth.shape == (new_doc.height, new_doc.width)
    new_scene, warnings = scene_from_dict(data, new_doc, None, scene.global_depth if keep_depth else None)
    if keep_depth and not np.array_equal(new_doc.composite, doc.composite):
        warnings.append("The image changed; the depth map may be out of date (re-run depth estimation if needed).")
    new_scene.preview_camera = scene.preview_camera
    return new_scene, new_doc, warnings
