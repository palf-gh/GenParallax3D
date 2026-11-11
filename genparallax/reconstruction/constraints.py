"""Inter-layer depth constraints.

Where two units overlap on the canvas, the upper PSD layer must stay in front of
the lower one, otherwise the reference view no longer reproduces the PSD (a lower
layer would poke through). Units that do not overlap may sit at any depth.

Conflicts are resolved from the top layer down:

* Surfaces (units with relief) are pushed back *locally*, only where they overlap
  an upper layer, by a smooth push field ``P``: ``Z = B + S * R + P``. The overlap
  is mostly the part hidden behind the upper layer, so the visible shape and the
  AI depth layout are kept, and what is revealed at the edges in a parallax move
  correctly lies behind the occluder.
* Planes (Flat mode, or no depth map) are shifted back as a whole, staying flat.

The check works per pixel and widens each surface by an error bound, because the
mesh interpolates linearly between vertices and can deviate from the per-pixel
depth inside a cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from genparallax.reconstruction.meshing import (ADAPTIVE_MAX_FACTOR, SIMPLIFY_MAX_FACTOR, mesh_depth_map,
                                                 simplify_tolerance, update_unit_mesh)
from genparallax.scene import ReconstructionMode, ReconstructionUnit, Scene
from genparallax.scene import next_revision

#: Minimum separation between overlapping layers, relative to depth.
MARGIN_RATIO = 0.002
#: Violations smaller than this (px) are rounding noise.
TOLERANCE_PX = 1e-4
#: Alpha below this (of 255) is too faint for the stacking order to be visible.
OVERLAP_ALPHA = 8
#: Feather radius of the push field, in mesh cells.
FEATHER_CELLS = 2.0

RELIEF_MODES = (ReconstructionMode.DEPTH_SURFACE, ReconstructionMode.SMOOTHED_SURFACE,
                ReconstructionMode.MANUAL_DEPTH)


@dataclass
class LayerConflict:
    lower_id: str
    upper_ids: list[str]
    depth_px: float  # largest distance the lower unit had to move back (or would have to)


def is_rigid(unit: ReconstructionUnit) -> bool:
    """Planes are shifted as a whole instead of being bent."""
    return unit.mode == ReconstructionMode.FLAT or unit.relief is None


def surface_offsets(unit: ReconstructionUnit) -> np.ndarray | float:
    """``S * R + P`` over the unit's bbox (0 for a plane)."""
    rel: np.ndarray | float = 0.0
    if unit.relief is not None and unit.depth_strength != 0.0 and unit.mode in RELIEF_MODES:
        rel = unit.depth_strength * unit.relief.astype(np.float64)
    if unit.depth_push is not None and unit.mode in RELIEF_MODES:
        rel = rel + unit.depth_push
    return rel


def unit_depth_map(unit: ReconstructionUnit, camera=None) -> np.ndarray:
    """Per-pixel camera depth over the unit's bbox.

    With a mesh this is the mesh itself rasterised from the reference camera (what the
    GPU depth test sees); pixels the mesh does not cover use ``B + S * R + P``.
    """
    mesh = mesh_depth_map(unit, camera) if camera is not None and unit.mesh is not None else None
    if mesh is not None and unit.depth_cache is not None and len(unit.depth_cache) > 2:
        return unit.depth_cache[2]  # gap-filled copy cached with the mesh depth
    analytic = None
    if mesh is None or np.isnan(mesh).any():
        analytic = np.full(unit.alpha.shape, unit.effective_base_depth, dtype=np.float64) + surface_offsets(unit)
    if mesh is None:
        return analytic
    filled = mesh if analytic is None else np.where(np.isnan(mesh), analytic, mesh)
    unit.depth_cache = (*unit.depth_cache[:2], filled)
    return filled


def interpolation_bound(unit: ReconstructionUnit) -> np.ndarray | float:
    """Upper bound of |mesh depth − pixel depth| around each pixel."""
    rel = surface_offsets(unit)
    if np.isscalar(rel):
        return 0.0

    def local_range(size: int) -> np.ndarray:
        size = max(2, int(size) + 1)
        return ndimage.maximum_filter(rel, size) - ndimage.minimum_filter(rel, size)

    step = max(1, int(unit.mesh_density))
    fine = local_range(step)
    tolerance = simplify_tolerance(unit)
    if tolerance is not None:  # simplified cells stay within the tolerance of S * R + P
        return np.minimum(local_range(step * SIMPLIFY_MAX_FACTOR), np.maximum(fine, tolerance))
    if not unit.adaptive_mesh:
        return fine
    # Large adaptive cells only exist where R varies less than ``mesh_detail``;
    # the push field is feathered over several cells, so it adds little there.
    coarse = local_range(step * ADAPTIVE_MAX_FACTOR)
    push_slack = 0.0 if unit.depth_push is None else local_range(step * 2)
    return np.minimum(coarse, np.maximum(fine, unit.mesh_detail * abs(unit.depth_strength) + push_slack))


def smooth_cover(deficit: np.ndarray, radius: float) -> np.ndarray:
    """A smooth field that is >= ``deficit`` everywhere and falls off over ``radius`` px."""
    r = max(1, int(round(radius)))
    grown = ndimage.grey_dilation(deficit, size=(2 * r + 1, 2 * r + 1))
    return np.maximum(ndimage.gaussian_filter(grown, r / 2.0, mode="nearest"), deficit)


def _participating(scene: Scene) -> list[ReconstructionUnit]:
    return [u for u in scene.units if u.mode != ReconstructionMode.EXCLUDED and u.visible]


def _overlaps_below(scene: Scene, unit: ReconstructionUnit) -> bool:
    l, t, r, b = unit.bbox
    for other in _participating(scene):
        if other.order < unit.order and other.mesh is not None:
            ol, ot, orr, ob = other.bbox
            if min(r, orr) > max(l, ol) and min(b, ob) > max(t, ot):
                return True
    return False


def is_overlay(scene: Scene, unit: ReconstructionUnit) -> bool:
    """Overlays are laid onto the layers below instead of constraining them."""
    return unit.conform and _overlaps_below(scene, unit)


def _has_depth_below(scene: Scene, unit: ReconstructionUnit) -> bool:
    """True when an overlapping layer below carries depth data to lie on."""
    l, t, r, b = unit.bbox
    for other in _participating(scene):
        if other.order >= unit.order or other.mesh is None:
            continue
        if other.depth_stats is None and (other.relief is None or other.depth_strength == 0.0):
            continue
        ol, ot, orr, ob = other.bbox
        if min(r, orr) > max(l, ol) and min(b, ob) > max(t, ot):
            return True
    return False


def _flatten_overlay(unit: ReconstructionUnit, camera) -> None:
    """Keep an overlay flat while nothing below has a depth map.

    The default depths are arbitrary planes; fitting an overlay across several of them
    bends it through the whole scene before any depth estimation has run.
    """
    unit.relief = None
    unit.depth_strength = 0.0
    unit.depth_stats = None
    unit.depth_push = None
    unit.grid_cache = None
    unit.relief_revision = next_revision()
    update_unit_mesh(unit, camera)


#: Relative slack for GPU depth precision when comparing exact mesh depths.
GPU_EPSILON = 1e-5


def depth_span(unit: ReconstructionUnit, camera) -> tuple[np.ndarray, np.ndarray]:
    """(near, far) depth per pixel. Exact for a meshed unit (the rasterised mesh);
    otherwise the analytic surface widened by the interpolation bound."""
    if unit.mesh is not None:
        z = unit_depth_map(unit, camera)
        eps = z * GPU_EPSILON
        return z - eps, z + eps
    z = unit_depth_map(unit)
    e = interpolation_bound(unit)
    return z - e, z + e


def _feather(unit: ReconstructionUnit) -> float:
    return FEATHER_CELLS * max(1, unit.mesh_density) * (2 if unit.adaptive_mesh else 1)


def _push_behind(unit: ReconstructionUnit, camera, required: np.ndarray, overlap: np.ndarray,
                 rigid: bool = False) -> tuple[float, np.ndarray]:
    """Move the unit back until its near side is behind ``required`` on ``overlap``.

    Surfaces grow a smooth local push field; planes (``rigid``) shift as a whole.
    Returns (largest move, mask of pixels that were in front).
    """
    moved = 0.0
    offending = np.zeros(overlap.shape, dtype=bool)
    if not overlap.any():
        return moved, offending
    for _ in range(12):
        near, _ = depth_span(unit, camera)
        deficit = np.where(overlap, required - near, 0.0)
        worst = float(deficit.max())
        if worst <= TOLERANCE_PX:
            break
        offending |= deficit > TOLERANCE_PX
        if rigid:
            unit.base_depth += worst * 1.001 + TOLERANCE_PX
            moved += worst
        else:
            add = smooth_cover(np.maximum(deficit, 0.0) * 1.1 + np.where(deficit > 0, 10 * TOLERANCE_PX, 0.0),
                               _feather(unit))
            unit.depth_push = add if unit.depth_push is None else unit.depth_push + add
            moved = max(moved, float(unit.depth_push.max()))
        update_unit_mesh(unit, camera)
    else:
        # Not converged (a coarse mesh samples the push field sparsely): shift the
        # small remainder as a whole so the order is guaranteed.
        # (Shifting the vertices does not move the perspective-interpolated depth by
        # exactly the same amount, so repeat until it holds.)
        for _ in range(8):
            near, _ = depth_span(unit, camera)
            worst = float(np.where(overlap, required - near, 0.0).max())
            if worst <= TOLERANCE_PX:
                break
            unit.base_depth += worst * 1.5 + TOLERANCE_PX
            moved += worst
            update_unit_mesh(unit, camera)
    return moved, offending


def check_layer_order(scene: Scene, apply: bool = False, margin_ratio: float = MARGIN_RATIO) -> list[LayerConflict]:
    """Find (and with ``apply``, fix) layers that break the PSD stacking order.

    Surfaces and planes must stay behind every overlapping non-overlay layer above
    them. Overlays (``unit.conform``) do not push anything back; with ``apply`` they
    are laid just in front of the layers below them, and kept behind the layers above.
    Depths are the meshes rasterised from the reference camera, so the check matches
    what the GPU draws. With ``apply`` the result depends only on the current settings:
    a unit is re-resolved from scratch when its settings or the layers above it changed,
    and otherwise keeps the push field it got for exactly those inputs.
    """
    camera = scene.reference_camera
    upper_far = np.full((scene.height, scene.width), -np.inf, dtype=np.float64)
    owner = np.full((scene.height, scene.width), -1, dtype=np.int32)
    units = sorted(_participating(scene), key=lambda u: u.order, reverse=True)
    conflicts: list[LayerConflict] = []
    ceilings: dict[str, tuple[np.ndarray, tuple]] = {}
    processed: list[ReconstructionUnit] = []  # non-overlay units above, in pass order
    for index, unit in enumerate(units):
        l, t, r, b = unit.bbox
        overlay = is_overlay(scene, unit)
        deps = _deps(unit, processed, margin_ratio)
        if apply and not overlay:
            signature = (_own_signature(unit, overlay), deps)
            if unit.resolve_signature == signature:
                # Same settings and same layers above as when it was last resolved:
                # the stored push field is still the answer.
                _accumulate(unit, camera, upper_far, owner, index)
                processed.append(unit)
                continue
            if unit.depth_push is not None:
                unit.depth_push = None
                update_unit_mesh(unit, camera)
        region = upper_far[t:b, l:r]
        overlap = (unit.alpha >= OVERLAP_ALPHA) & np.isfinite(region)
        required = np.where(overlap, region * (1.0 + margin_ratio), -np.inf)
        if overlay:
            ceilings[unit.id] = (required.copy(), deps)
            if not apply:  # an overlay may sit anywhere behind the layers above it
                near, _ = depth_span(unit, camera)
                deficit = np.where(overlap, required - near, 0.0)
                if overlap.any() and deficit.max() > TOLERANCE_PX:
                    culprits = np.unique(owner[t:b, l:r][deficit > TOLERANCE_PX])
                    conflicts.append(LayerConflict(unit.id, [units[i].id for i in culprits if i >= 0],
                                                   float(deficit.max())))
            continue  # overlays never constrain the layers below them
        if apply:
            moved, offending = _push_behind(unit, camera, required, overlap, rigid=is_rigid(unit))
        else:
            near, _ = depth_span(unit, camera)
            deficit = np.where(overlap, required - near, 0.0)
            offending = deficit > TOLERANCE_PX
            moved = float(deficit.max()) if offending.any() else 0.0
        if offending.any():
            culprits = [int(i) for i in np.unique(owner[t:b, l:r][offending]) if i >= 0]
            conflicts.append(LayerConflict(unit.id, [units[i].id for i in culprits], moved))
        if apply:
            unit.resolve_signature = (_own_signature(unit, False), deps)
        _accumulate(unit, camera, upper_far, owner, index)
        processed.append(unit)
    if apply:
        for unit in reversed(units):  # bottom-up, so stacked overlays follow each other
            if unit.id not in ceilings:
                continue
            ceiling, above = ceilings[unit.id]
            below = tuple((o.id, o.revision) for o in units
                          if o.order < unit.order and o.mesh is not None and _bbox_overlap(unit, o))
            signature = (_own_signature(unit, True), above, below)
            if unit.resolve_signature == signature:
                continue
            place_overlay(scene, unit, ceiling)
            unit.resolve_signature = signature
    return conflicts


def _bbox_overlap(a: ReconstructionUnit, b: ReconstructionUnit) -> bool:
    al, at, ar, ab = a.bbox
    bl, bt, br, bb = b.bbox
    return min(ar, br) > max(al, bl) and min(ab, bb) > max(at, bt)


def _deps(unit: ReconstructionUnit, above: list[ReconstructionUnit], margin_ratio: float) -> tuple:
    """What a unit's resolution depends on besides its own settings: the overlapping
    layers above it, by mesh revision."""
    return (margin_ratio,) + tuple((o.id, o.revision) for o in above if _bbox_overlap(unit, o))


def _own_signature(unit: ReconstructionUnit, overlay: bool) -> tuple:
    common = (unit.mode, unit.conform, unit.mesh_density, unit.adaptive_mesh, unit.mesh_detail, unit.mesh_simplify,
              id(unit.texture), unit.offset, unit.depth_offset)
    if overlay:  # B, S and R are outputs of the placement, not inputs
        return common + (unit.depth_smoothing, unit.edge_inset)
    return common + (unit.base_depth, unit.depth_strength, unit.relief_revision)


def _accumulate(unit: ReconstructionUnit, camera, upper_far: np.ndarray, owner: np.ndarray, index: int) -> None:
    """Add a resolved unit's far side to the running 'layers above' map."""
    l, t, r, b = unit.bbox
    region = upper_far[t:b, l:r]
    _, far = depth_span(unit, camera)
    own = (unit.alpha > 0) & (far > region)
    region[own] = far[own]
    owner[t:b, l:r][own] = index


def resolve_layer_order(scene: Scene, margin_ratio: float = MARGIN_RATIO) -> list[LayerConflict]:
    return check_layer_order(scene, apply=True, margin_ratio=margin_ratio)


#: How far in front of the surface below an overlay sits, relative to depth.
CONFORM_GAP_RATIO = 0.001


def place_overlay(scene: Scene, unit: ReconstructionUnit, ceiling: np.ndarray | None = None) -> bool:
    """Lay an overlay just in front of whatever lies below it in the PSD.

    ``B``, ``S`` and ``R`` are fitted to the lower layers' meshes, so the overlay
    follows the surface it is painted on; a local push field then pulls it in front
    of them where the smooth fit falls behind, and ``ceiling`` (the far side of the
    layers above, per pixel) keeps it behind those. ``depth_offset`` stays a user
    nudge. Returns False when nothing lies below.
    """
    from genparallax.reconstruction.depth import relief_from_z

    camera = scene.reference_camera
    if not _has_depth_below(scene, unit):
        _flatten_overlay(unit, camera)
        return False
    l, t, r, b = unit.bbox
    below = np.full((b - t, r - l), np.inf)
    for other in _participating(scene):
        if other.order >= unit.order or other.mesh is None:
            continue
        ol, ot, orr, ob = other.bbox
        x0, y0, x1, y1 = max(l, ol), max(t, ot), min(r, orr), min(b, ob)
        if x1 <= x0 or y1 <= y0:
            continue
        near, _ = depth_span(other, camera)
        near = near[y0 - ot : y1 - ot, x0 - ol : x1 - ol]
        present = other.alpha[y0 - ot : y1 - ot, x0 - ol : x1 - ol] >= OVERLAP_ALPHA
        sub = below[y0 - t : y1 - t, x0 - l : x1 - l]
        np.minimum(sub, np.where(present, near, np.inf), out=sub)
    known = np.isfinite(below) & (unit.alpha > 0)
    unit.relief_revision = next_revision()
    unit.grid_cache = None
    unit.depth_push = None
    if not known.any():
        update_unit_mesh(unit, camera)
        return False
    fill = float(np.median(below[known]))
    limit = np.where(np.isfinite(below), below, fill) * (1.0 - CONFORM_GAP_RATIO)
    has_ceiling = ceiling is not None and np.isfinite(ceiling).any()
    target = np.where(np.isfinite(ceiling), np.maximum(limit, ceiling), limit) if has_ceiling else limit
    relief, median, spread = relief_from_z(target - unit.depth_offset, known,
                                           smoothing=max(unit.depth_smoothing, 1.0))
    unit.base_depth, unit.depth_strength = median, spread
    unit.relief = relief if spread > 0 and unit.mode in RELIEF_MODES else None
    unit.depth_stats = (median, spread)
    unit.relief_revision = next_revision()
    update_unit_mesh(unit, camera)
    if unit.mode not in RELIEF_MODES:
        return True
    # Pull forward wherever the smooth fit lies behind the surface below …
    in_front_of = known & (unit.alpha >= OVERLAP_ALPHA)
    if has_ceiling:  # … unless a layer above leaves no room there
        in_front_of &= ~(np.isfinite(ceiling) & (ceiling >= limit))
    for _ in range(12):
        _, far = depth_span(unit, camera)
        excess = np.where(in_front_of, far - limit, 0.0)
        worst = float(excess.max()) if in_front_of.any() else 0.0
        if worst <= TOLERANCE_PX:
            break
        pull = smooth_cover(np.maximum(excess, 0.0) * 1.1 + np.where(excess > 0, 10 * TOLERANCE_PX, 0.0),
                            _feather(unit))
        unit.depth_push = -pull if unit.depth_push is None else unit.depth_push - pull
        update_unit_mesh(unit, camera)
    if has_ceiling:  # the layers above always win
        _push_behind(unit, camera, ceiling, np.isfinite(ceiling) & (unit.alpha >= OVERLAP_ALPHA))
    return True


# --------------------------------------------------------------- background use
#: Unit fields written by the layer-order pass (copied back from a snapshot).
RESOLVED_FIELDS = ("base_depth", "depth_strength", "relief", "relief_revision", "depth_stats", "depth_push",
                   "grid_cache", "mesh", "revision", "depth_cache", "resolve_signature")


def snapshot_for_resolution(scene: Scene) -> Scene:
    """A copy of the scene that the layer-order pass may modify on another thread.

    Units are shallow copies: the pass only rebinds attributes (it never writes into
    shared arrays), so the original scene stays untouched and usable meanwhile.
    """
    import copy

    snap = copy.copy(scene)
    snap.units = [copy.copy(u) for u in scene.units]
    return snap


def resolve_in_snapshot(snapshot: Scene) -> list[LayerConflict]:
    """Resolve the snapshot and return the conflicts that remain (normally none)."""
    resolve_layer_order(snapshot)
    return check_layer_order(snapshot)


def apply_resolution(scene: Scene, snapshot: Scene, skip: set[str] | frozenset = frozenset()) -> None:
    """Copy the resolved state back into the live scene.

    ``skip`` lists units edited after the snapshot was taken; they keep their live state.
    Units resolved against an edited unit's old geometry recorded its old revision in
    their signatures, so the next pass redoes exactly those.
    """
    by_id = {u.id: u for u in snapshot.units}
    for unit in scene.units:
        resolved = by_id.get(unit.id)
        if resolved is not None and unit.id not in skip:
            for name in RESOLVED_FIELDS:
                setattr(unit, name, getattr(resolved, name))


def adopt_snapshot(scene: Scene, snapshot: Scene) -> None:
    """Take over everything a background operation changed in a snapshot (all unit fields
    and the scene's depth map), keeping the live objects so open views stay valid."""
    import dataclasses

    by_id = {u.id: u for u in snapshot.units}
    for unit in scene.units:
        copy = by_id.get(unit.id)
        if copy is not None:
            for f in dataclasses.fields(unit):
                setattr(unit, f.name, getattr(copy, f.name))
    for name in ("global_depth", "depth_source", "depth_far_ratio"):
        setattr(scene, name, getattr(snapshot, name))
