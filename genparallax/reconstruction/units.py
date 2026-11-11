"""Creating, merging and splitting reconstruction units, and assigning depth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from genparallax.document.psd_loader import PsdDocument
from genparallax.reconstruction import depth as depthlib
from genparallax.reconstruction.meshing import update_unit_mesh
from genparallax.scene import ReconstructionMode, ReconstructionUnit, Scene
from genparallax.scene import next_revision

#: Default spacing between successive layers, as a fraction of the focal length.
DEFAULT_LAYER_SPACING = 0.08
#: Extra smoothing (sigma, px) used by the Smoothed Surface mode.
SMOOTHED_SURFACE_SIGMA = 12.0


def scene_from_document(doc: PsdDocument, fov_y_deg: float | None = None) -> Scene:
    scene = Scene.create(doc.width, doc.height, fov_y_deg)
    scene.units = build_default_units(doc)
    assign_default_depths(scene)
    for unit in scene.units:
        update_unit_mesh(unit, scene.reference_camera)
    return scene


def build_default_units(doc: PsdDocument) -> list[ReconstructionUnit]:
    """One unit per visible, reconstructable leaf layer (clipping layers fold into their base)."""
    units = []
    for info in doc.leaves():
        if not info.reconstructable or not doc.effectively_visible(info.key):
            continue
        unit = make_unit(doc, [info.key], info.name)
        if unit is not None:
            units.append(unit)
    return units


def make_unit(doc: PsdDocument, keys: list[str], name: str) -> ReconstructionUnit | None:
    keys = _with_clip_layers(doc, keys)
    composed = doc.compose_layers(keys)
    if composed is None:
        return None
    rgba, left, top = composed
    rgba = bleed_edges(rgba)
    order = min(doc.layers[k].order for k in keys if doc.layers[k].order >= 0)
    return ReconstructionUnit(name=name, source_layers=keys, texture=rgba, offset=(left, top), order=order,
                              conform=is_overlay(rgba))


#: A layer whose painted pixels are mostly see-through is treated as an overlay.
OVERLAY_SOLID_FRACTION = 0.3


def is_overlay(rgba: np.ndarray) -> bool:
    """True when few painted pixels are fully opaque (light rays, haze, soft flowers…)."""
    a = rgba[:, :, 3]
    painted = a > 0
    if not painted.any():
        return False
    return float(np.mean(a[painted] >= 250)) < OVERLAY_SOLID_FRACTION


def bleed_edges(rgba: np.ndarray) -> np.ndarray:
    """Copy edge colours into fully transparent texels (alpha unchanged).

    Filtering and mipmapping in the preview and in After Effects mix the RGB of
    transparent texels into silhouettes; without this they show dark or white fringes.
    """
    visible = rgba[:, :, 3] > 0
    if visible.all() or not visible.any():
        return rgba
    out = rgba.copy()
    out[:, :, :3] = depthlib.fill_outside_mask(rgba[:, :, :3], visible)
    return out


def _with_clip_layers(doc: PsdDocument, keys: list[str]) -> list[str]:
    out = list(dict.fromkeys(keys))
    for k in list(out):
        for clip in doc.clip_layers_of(k):
            if clip.key not in out:
                out.append(clip.key)
    return out


def assign_default_depths(scene: Scene) -> None:
    """Bottom layer furthest away, top layer at the reference plane (Z = f)."""
    f = scene.reference_camera.focal_px
    ordered = sorted(scene.units, key=lambda u: u.order)
    n = len(ordered)
    for rank, unit in enumerate(ordered):
        unit.base_depth = f * (1.0 + DEFAULT_LAYER_SPACING * (n - 1 - rank))
        unit.depth_offset = 0.0
        unit.depth_strength = 0.0


def merge_units(scene: Scene, doc: PsdDocument, units: list[ReconstructionUnit], name: str | None = None) -> ReconstructionUnit | None:
    if len(units) < 2:
        return None
    keys = [k for u in sorted(units, key=lambda u: u.order) for k in u.source_layers]
    first = min(units, key=lambda u: u.order)
    merged = make_unit(doc, keys, name or " + ".join(u.name for u in units))
    if merged is None:
        return None
    _copy_params(first, merged)
    idx = min(scene.units.index(u) for u in units)
    for u in units:
        scene.units.remove(u)
    scene.units.insert(idx, merged)
    refresh_unit(scene, merged)
    return merged


def split_unit(scene: Scene, doc: PsdDocument, unit: ReconstructionUnit) -> list[ReconstructionUnit]:
    keys = [k for k in unit.source_layers if not doc.layers[k].clipping]
    if len(keys) < 2:
        return [unit]
    idx = scene.units.index(unit)
    scene.units.remove(unit)
    new_units = []
    for k in keys:
        u = make_unit(doc, [k], doc.layers[k].name)
        if u is None:
            continue
        _copy_params(unit, u)
        new_units.append(u)
    for i, u in enumerate(new_units):
        scene.units.insert(idx + i, u)
        refresh_unit(scene, u)
    return new_units


def _copy_params(src: ReconstructionUnit, dst: ReconstructionUnit) -> None:
    for attr in ("base_depth", "depth_strength", "depth_offset", "depth_smoothing", "mesh_density", "mode",
                 "custom_depth_path", "visible", "adaptive_mesh", "mesh_detail", "mesh_simplify", "edge_inset",
                 "conform", "occlusion_inset"):
        setattr(dst, attr, getattr(src, attr))


def set_global_depth(scene: Scene, nearness: np.ndarray | None, source: str | None, assign: bool = True) -> list:
    """Install a scene depth map (nearness, 1 = near) and rebuild the reliefs.

    With ``assign`` the units' base depth and strength are taken from the map and the
    PSD stacking order is enforced afterwards; the resolved conflicts are returned.
    """
    scene.global_depth = nearness
    scene.depth_source = source
    assigning = assign and nearness is not None
    # When assigning, the layer-order pass lays the overlays anyway, with the layers
    # above taken into account, so do not lay them twice.
    refresh_all(scene, mesh=not assigning, place_overlays=not assigning)
    if assigning:
        return assign_depths_from_map(scene)
    return []


def visibility_masks(scene: Scene, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Per unit (cropped to its bbox): pixels where the unit itself shows in the composite.

    A depth map holds the depth of whatever is on top, so a lower layer's depth may
    only be measured where upper layers do not cover it.
    """
    coverage = np.zeros((scene.height, scene.width), dtype=np.float32)
    masks = {}
    for unit in sorted(scene.units, key=lambda u: u.order, reverse=True):
        l, t, r, b = unit.bbox
        a = unit.alpha.astype(np.float32) / 255.0
        region = coverage[t:b, l:r]
        masks[unit.id] = (a > threshold) & (region < threshold)
        region[...] = 1.0 - (1.0 - region) * (1.0 - a)
    return masks


#: A lower layer's depth is checked against occluders within this many pixels.
OCCLUDER_RADIUS = 64
#: ... and distrusted when it is not at least this much (relative) behind the occluder.
OCCLUDER_DEPTH_MARGIN = 0.15
#: A layer's depth is distrusted when it reaches (within this margin) the depth of the
#: lower layer visible beside it.
BEHIND_DEPTH_MARGIN = 0.03
#: Robust outlier removal against the unit's own smoothed depth (sigma in px, MAD multiples).
OUTLIER_SIGMA = 24.0
OUTLIER_SIGMAS = 3.0
OUTLIER_ROUNDS = 3


def trusted_pixels(scene: Scene, unit: ReconstructionUnit, known: np.ndarray, nearness: np.ndarray) -> np.ndarray:
    """Drop pixels of ``known`` whose depth-map value really belongs to another layer.

    A depth estimate is blurred across silhouettes and does not resolve small gaps or
    thin parts: the sky seen through a lattice gets the lattice's depth, and a thin bar
    of the lattice gets the sky's. Taken at face value, the lower layer grows spikes
    that reach up to the layer in front, and the upper one spikes that reach back.
    A pixel is therefore distrusted when, close to another layer's visible area, the
    map does not place it clearly behind the layers above / in front of the layers
    below, or when it lies within ``unit.occlusion_inset`` px of an opaque layer above.
    Distrusted pixels are filled smoothly from the trusted ones.
    """
    from scipy import ndimage

    l, t, r, b = unit.bbox
    pad = OCCLUDER_RADIUS
    x0, y0, x1, y1 = max(l - pad, 0), max(t - pad, 0), min(r + pad, scene.width), min(b + pad, scene.height)
    shape = (y1 - y0, x1 - x0)
    above, below = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    for other in scene.units:
        if other is unit or other.conform or other.mode == ReconstructionMode.EXCLUDED:
            continue  # overlays are see-through: the map shows what is behind them
        ol, ot, orr, ob = other.bbox
        cx0, cy0, cx1, cy1 = max(ol, x0), max(ot, y0), min(orr, x1), min(ob, y1)
        if cx1 > cx0 and cy1 > cy0:
            target = above if other.order > unit.order else below
            target[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] |=                 other.alpha[cy0 - ot : cy1 - ot, cx0 - ol : cx1 - ol] >= 128
    own = (slice(t - y0, b - y0), slice(l - x0, r - x0))
    solid = np.zeros(shape, dtype=bool)
    solid[own] = unit.alpha >= 128
    below &= ~above & ~solid  # where a lower layer is what the depth map shows
    z = scene.nearness_to_z(nearness[y0:y1, x0:x1])
    inset = max(0, int(unit.occlusion_inset))

    def depth_of_nearest(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to ``region`` and its depth, read a little inside it (rims are blurred)."""
        distance = ndimage.distance_transform_edt(~region)
        core = ndimage.binary_erosion(region, iterations=2)
        if not core.any():
            core = region
        _, (iy, ix) = ndimage.distance_transform_edt(~core, return_indices=True)
        return distance, z[iy, ix]

    suspect = np.zeros(shape, dtype=bool)
    rim = np.zeros(shape, dtype=bool)
    near_above = near_below = None
    if above.any():
        distance, z_above = depth_of_nearest(above)
        near_above = (distance <= OCCLUDER_RADIUS)[own]
        not_behind = (distance <= OCCLUDER_RADIUS) & (z <= z_above * (1.0 + OCCLUDER_DEPTH_MARGIN))
        suspect |= ndimage.binary_dilation(not_behind, iterations=inset) if inset else not_behind
        rim = distance <= inset
    if below.any():
        distance, z_below = depth_of_nearest(below)
        near_below = (distance <= OCCLUDER_RADIUS)[own]
        suspect |= (distance <= OCCLUDER_RADIUS) & (z >= z_below * (1.0 - BEHIND_DEPTH_MARGIN))
    if near_above is None and near_below is None:
        return known
    trusted = known
    for candidate in (known & ~suspect[own] & ~rim[own], known & ~rim[own]):
        if candidate.sum() >= max(16, 0.02 * known.sum()):
            trusted = candidate
            break
    # Partly blended values (between the two layers) pass the tests above. Remove what
    # still sticks out of the unit's own smooth surface towards the neighbouring layer.
    z_own = z[own].astype(np.float32)
    for _ in range(OUTLIER_ROUNDS):
        weight = trusted.astype(np.float64)
        num, den = depthlib._blur_pair(z_own * weight, weight, OUTLIER_SIGMA)  # coarse blur: fast
        smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), z_own)
        residual = z_own - smooth
        scale = 1.4826 * float(np.median(np.abs(residual[trusted]))) + 0.002 * float(np.median(z_own[trusted]))
        outlier = np.zeros_like(trusted)
        if near_below is not None:
            outlier |= near_below & (residual > OUTLIER_SIGMAS * scale)
        if near_above is not None:
            outlier |= near_above & (residual < -OUTLIER_SIGMAS * scale)
        remaining = trusted & ~outlier
        if remaining.sum() == trusted.sum() or remaining.sum() < max(16, 0.02 * known.sum()):
            break
        trusted = remaining
    return trusted


def refresh_all(scene: Scene, mesh: bool = True, place_overlays: bool = True) -> None:
    vis = visibility_masks(scene) if scene.global_depth is not None else None
    for unit in scene.units:
        refresh_relief(scene, unit, vis, place_overlays)
        if mesh:
            update_unit_mesh(unit, scene.reference_camera)


def refresh_relief(scene: Scene, unit: ReconstructionUnit, visibility: dict[str, np.ndarray] | None = None,
                   place_overlays: bool = True) -> None:
    """Recompute ``R_i`` and the depth statistics from the global or custom depth map.

    Overlays are laid onto the layers below instead (unless ``place_overlays`` is
    False, when the caller runs the layer-order pass, which lays them anyway).
    """
    unit.relief_revision = next_revision()
    unit.grid_cache = None  # adaptive meshes depend on the relief
    unit.depth_push = None  # recomputed by the layer-order pass
    unit.resolve_signature = None
    if unit.mode in (ReconstructionMode.FLAT, ReconstructionMode.EXCLUDED):
        unit.relief = None
        unit.depth_stats = None
        return
    if unit.conform:
        if place_overlays:
            conform_unit(scene, unit)
        return
    solid = unit.alpha >= 8
    custom = unit.mode == ReconstructionMode.MANUAL_DEPTH and unit.custom_depth_path         and Path(unit.custom_depth_path).exists()
    if custom:
        source = depthlib.load_depth_image(unit.custom_depth_path, (scene.width, scene.height))
        known = solid  # a per-unit map describes this unit everywhere
    elif scene.global_depth is not None:
        source = scene.global_depth
        if visibility is None:
            visibility = visibility_masks(scene)
        known = visibility.get(unit.id, solid)
    else:
        unit.relief = None
        unit.depth_stats = None
        return
    known = known & depthlib.interior(solid, unit.edge_inset)
    sigma = unit.depth_smoothing
    if unit.mode == ReconstructionMode.SMOOTHED_SURFACE:
        sigma = max(sigma, SMOOTHED_SURFACE_SIGMA)
    z = scene.nearness_to_z(depthlib.crop(source, unit.bbox))
    if not custom:
        known = trusted_pixels(scene, unit, known, source)
    unit.relief, median, spread = depthlib.relief_from_z(z, known, smoothing=sigma)
    unit.depth_stats = (median, spread) if known.any() else None


def assign_depths_from_map(scene: Scene, resolve_order: bool = True) -> list:
    """Set ``B = median Z`` and ``S = Z spread`` per unit from the depth map, then
    (optionally) push lower layers back where they would poke through upper ones.

    Returns the layer-order conflicts that were resolved.
    """
    from genparallax.reconstruction.constraints import resolve_layer_order

    for unit in scene.units:
        if unit.depth_stats is None or unit.conform:
            continue
        median, spread = unit.depth_stats
        unit.base_depth = median
        unit.depth_offset = 0.0
        unit.depth_strength = spread if unit.relief is not None else 0.0
    for unit in scene.units:
        update_unit_mesh(unit, scene.reference_camera)
    if resolve_order:
        return resolve_layer_order(scene)  # also lays the overlays
    conform_overlays(scene)
    return []


def conform_unit(scene: Scene, unit: ReconstructionUnit) -> bool:
    """Lay an overlay onto the layers below it (see :func:`constraints.place_overlay`)."""
    from genparallax.reconstruction.constraints import place_overlay

    return place_overlay(scene, unit)


def conform_overlays(scene: Scene) -> None:
    """Re-lay every overlay unit, bottom-up, so stacked overlays follow each other."""
    for unit in sorted(scene.units, key=lambda u: u.order):
        if unit.conform and unit.mode != ReconstructionMode.EXCLUDED:
            conform_unit(scene, unit)


def refresh_unit(scene: Scene, unit: ReconstructionUnit, relief: bool = True) -> None:
    if relief:
        refresh_relief(scene, unit)
    update_unit_mesh(unit, scene.reference_camera)
