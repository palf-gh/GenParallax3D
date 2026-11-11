"""A cache of the derived scene state, so large projects open quickly.

Loading a project recomputes everything that is not saved: each unit's relief from the
depth map, the overlay placement, the layer-order push fields and the meshes. For a
large PSD that takes minutes, yet it depends only on the inputs: the layer images, the
depth map, the composition and the unit settings. This module stores the result next
to the project (``.gp3d_cache/<project>/scene_cache.npz``) under a hash of those inputs,
and restores it when the hash matches. Any change to the inputs, to the cache format or
(when running from source) to the reconstruction code gives a new hash, and the scene is
then computed as before.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from genparallax import __version__
from genparallax.scene import Mesh, ReconstructionUnit, Scene, next_revision

CACHE_FILE = "scene_cache.npz"
CACHE_VERSION = 1

#: Unit settings that change the derived state (``base_depth`` and ``depth_strength``
#: are inputs for surfaces and outputs for overlays; both cases are covered).
_SETTINGS = ("name", "order", "offset", "base_depth", "depth_strength", "depth_offset", "depth_smoothing",
             "mesh_density", "adaptive_mesh", "mesh_detail", "mesh_simplify", "edge_inset", "occlusion_inset",
             "conform", "visible")


def _code_fingerprint() -> bytes:
    """Source of the modules that compute the derived state, when it is available.

    Packaged builds have no source files; their version number stands in for it.
    """
    from genparallax import scene
    from genparallax.reconstruction import constraints, depth, meshing, units

    h = hashlib.blake2b(digest_size=16)
    for module in (scene, depth, meshing, constraints, units):
        path = Path(getattr(module, "__file__", "") or "")
        if path.suffix == ".py" and path.is_file():
            h.update(path.read_bytes())
    return h.digest()


def scene_key(scene: Scene) -> str:
    """A hash of everything the derived unit state depends on."""
    h = hashlib.blake2b(digest_size=20)

    def add(value) -> None:
        h.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))

    comp = scene.composition
    add([CACHE_VERSION, __version__, scene.width, scene.height, scene.depth_far_ratio,
         scene.reference_camera.focal_px, comp.width, comp.height, comp.image_scale])
    h.update(_code_fingerprint())
    if scene.global_depth is None:
        add(None)
    else:
        depth = np.ascontiguousarray(scene.global_depth)
        add([str(depth.dtype), depth.shape])
        h.update(depth.tobytes())
    for unit in scene.units:
        add([unit.id] + [getattr(unit, name) for name in _SETTINGS] + [unit.mode.value])
        texture = np.ascontiguousarray(unit.texture)
        add(texture.shape)
        h.update(texture.tobytes())
        custom = unit.custom_depth_path
        if custom and Path(custom).is_file():
            stat = Path(custom).stat()
            add([custom, stat.st_size, stat.st_mtime_ns])
        else:
            add(custom)
    return h.hexdigest()


def save_scene_cache(cache_dir: str | Path, scene: Scene, warnings: list[str] | None = None,
                     key: str | None = None) -> Path | None:
    """Store the derived state of every unit. Returns the file, or None on failure.

    ``key`` is :func:`scene_key` of the inputs. It defaults to the scene as it is now,
    which is right when saving a project (the settings written are the next inputs);
    after a load, pass the key taken before the load changed any unit.
    """
    arrays: dict[str, np.ndarray] = {}
    meta = {"key": key or scene_key(scene), "warnings": list(warnings or []), "units": {}}
    for i, unit in enumerate(scene.units):
        entry = {
            "id": unit.id,
            "base_depth": unit.base_depth,
            "depth_strength": unit.depth_strength,
            "depth_stats": list(unit.depth_stats) if unit.depth_stats is not None else None,
            "mesh": unit.mesh is not None,
        }
        for name in ("relief", "depth_push"):
            value = getattr(unit, name)
            entry[name] = value is not None
            if value is not None:
                # Push fields are float64 sums; float32 halves the file and differs by
                # far less than a pixel (restored meshes are stored exactly anyway).
                arrays[f"{i}_{name}"] = value.astype(np.float32) if name == "depth_push" else value
        if unit.mesh is not None:
            arrays[f"{i}_vertices"] = unit.mesh.vertices
            arrays[f"{i}_triangles"] = unit.mesh.triangles
            arrays[f"{i}_uvs"] = unit.mesh.uvs
        meta["units"][str(i)] = entry
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    path = Path(cache_dir) / CACHE_FILE
    tmp = path.with_name(path.name + ".tmp.npz")  # np.savez keeps an .npz suffix as given
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, **arrays)
        tmp.replace(path)  # never leave a half-written cache behind
    except OSError:
        tmp.unlink(missing_ok=True)
        return None
    return path


def load_scene_cache(cache_dir: str | Path | None, scene: Scene, key: str | None = None) -> list[str] | None:
    """Restore the derived state when the cache matches the scene's inputs.

    Returns the warnings stored with it, or None (scene untouched) when there is no
    usable cache.
    """
    if cache_dir is None:
        return None
    path = Path(cache_dir) / CACHE_FILE
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            if meta.get("key") != (key or scene_key(scene)) or len(meta["units"]) != len(scene.units):
                return None
            restored = []
            for i, unit in enumerate(scene.units):
                entry = meta["units"][str(i)]
                if entry["id"] != unit.id:
                    return None
                restored.append((unit, entry, {name: data[f"{i}_{name}"] for name in
                                               (["relief"] if entry["relief"] else [])
                                               + (["depth_push"] if entry["depth_push"] else [])
                                               + (["vertices", "triangles", "uvs"] if entry["mesh"] else [])}))
    except (OSError, ValueError, KeyError):
        return None
    for unit, entry, arrays in restored:
        _restore_unit(unit, entry, arrays)
    from genparallax.reconstruction.constraints import mark_resolved

    mark_resolved(scene)
    return list(meta.get("warnings", []))


def _restore_unit(unit: ReconstructionUnit, entry: dict, arrays: dict[str, np.ndarray]) -> None:
    unit.base_depth = entry["base_depth"]
    unit.depth_strength = entry["depth_strength"]
    unit.depth_stats = tuple(entry["depth_stats"]) if entry["depth_stats"] is not None else None
    unit.relief = arrays.get("relief")
    push = arrays.get("depth_push")
    unit.depth_push = None if push is None else push.astype(np.float64)
    unit.relief_revision = next_revision()
    unit.grid_cache = None  # rebuilt on the first edit
    unit.depth_cache = None
    unit.mesh = (Mesh(vertices=arrays["vertices"], triangles=arrays["triangles"], uvs=arrays["uvs"],
                      material_ref=unit.id) if "vertices" in arrays else None)
    unit.revision = next_revision()
