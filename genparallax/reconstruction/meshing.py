"""Alpha-aware meshing and reference-camera back-projection.

The mesh is split into two stages so that parameter edits stay interactive:

1. Topology and UVs – :func:`build_grid` (uniform) or :func:`build_adaptive_grid`
   (quadtree). Rebuilt only when the density, the texture or (adaptive only) the
   relief change.
2. :func:`update_unit_mesh` – per-vertex depth ``Z = B + S * R`` and back-projection
   through the reference camera. Cheap; run on every slider change.

Because each vertex is placed on the ray through its source pixel, the reference
camera always sees the original image regardless of ``B`` and ``S``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from genparallax.scene import Camera, Mesh, ReconstructionMode, ReconstructionUnit
from genparallax.scene import next_revision

ALPHA_THRESHOLD = 1  # any non-zero alpha keeps a cell, so antialiased edges survive
#: Largest adaptive cell, as a multiple of the minimum cell size (a power of two).
ADAPTIVE_MAX_FACTOR = 8


def build_grid(alpha: np.ndarray, step: int) -> dict:
    """Uniform grid over the texture, keeping only cells touching visible alpha.

    Returns pixel-corner coordinates ``xy`` (N, 2, in texture pixels), ``faces``
    (M, 3) and ``uv`` (N, 2, glTF convention: origin top-left).
    """
    h, w = alpha.shape
    step = max(1, int(step))
    xs = np.unique(np.append(np.arange(0, w, step), w))
    ys = np.unique(np.append(np.arange(0, h, step), h))
    nx, ny = len(xs), len(ys)

    visible = alpha >= ALPHA_THRESHOLD
    # Cell occupancy: any visible pixel inside [xs[i], xs[i+1]) x [ys[j], ys[j+1]).
    row_any = np.logical_or.reduceat(visible, ys[:-1], axis=0)
    cells = np.logical_or.reduceat(row_any, xs[:-1], axis=1)  # (ny-1, nx-1)

    jj, ii = np.nonzero(cells)
    v00 = jj * nx + ii
    v10 = v00 + 1
    v01 = v00 + nx
    v11 = v01 + 1
    # Counter-clockwise when viewed from the camera (image v points down, world Y up).
    tri = np.concatenate(
        [np.stack([v00, v01, v10], axis=1), np.stack([v10, v01, v11], axis=1)], axis=0
    )

    used = np.zeros(nx * ny, dtype=bool)
    used[tri.ravel()] = True
    remap = -np.ones(nx * ny, dtype=np.int64)
    remap[used] = np.arange(used.sum())
    faces = remap[tri].astype(np.uint32)

    gx, gy = np.meshgrid(xs, ys)
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)[used].astype(np.float64)
    uv = (xy / np.array([w, h], dtype=np.float64)).astype(np.float32)
    return {"step": step, "shape": (h, w), "xy": xy, "faces": faces, "uv": uv}


def _block(a: np.ndarray, size: int, op) -> np.ndarray:
    """Reduce ``a`` over non-overlapping ``size x size`` blocks."""
    if size == 1:
        return a
    return op(op(a.reshape(a.shape[0] // size, size, a.shape[1] // size, size), axis=3), axis=1)


def build_adaptive_grid(alpha: np.ndarray, relief: np.ndarray | None, step: int, tolerance: float,
                        max_factor: int = ADAPTIVE_MAX_FACTOR) -> dict:
    """Quadtree mesh with cells of ``step * 2**k`` pixels (``2**k <= max_factor``).

    A cell is split while it straddles the silhouette or the texture edge, or while
    the relief range inside it exceeds ``tolerance``. The tree is 2:1 balanced, and a
    cell next to finer cells includes the shared edge midpoints, so there are no
    T-junctions and the surface cannot crack when the depth varies. Vertices stay on
    integer pixel corners. Returns the same structure as :func:`build_grid`.
    """
    h, w = alpha.shape
    step = max(1, int(step))
    levels = int(round(np.log2(max(1, max_factor))))
    root = step << levels
    nby, nbx = -(-h // root), -(-w // root)
    gh, gw = nby << levels, nbx << levels  # grid size in minimum cells

    vis = np.zeros((gh * step, gw * step), dtype=bool)
    vis[:h, :w] = alpha >= ALPHA_THRESHOLD
    any_vis = [_block(vis, step, np.any)]
    all_vis = [_block(vis, step, np.all)]  # False for cells crossing the texture edge
    rmax = rmin = None
    if relief is not None:
        rp = np.empty(vis.shape, dtype=np.float32)
        rp[:h, :w] = relief
        rp[:h, w:] = rp[:h, w - 1 : w]  # extend edges so padding adds no fake relief
        rp[h:, :] = rp[h - 1 : h, :]
        rmax = [_block(rp, step, np.max)]
        rmin = [_block(rp, step, np.min)]
    for k in range(1, levels + 1):
        any_vis.append(_block(any_vis[0], 1 << k, np.any))
        all_vis.append(_block(all_vis[0], 1 << k, np.all))
        if relief is not None:
            rmax.append(_block(rmax[0], 1 << k, np.max))
            rmin.append(_block(rmin[0], 1 << k, np.min))

    # Leaf level for every minimum cell (-1 = no geometry).
    level = np.full((gh, gw), -1, dtype=np.int16)
    stack = [(levels, cy, cx) for cy in range(nby) for cx in range(nbx)]
    while stack:
        k, cy, cx = stack.pop()
        if not any_vis[k][cy, cx]:
            continue
        split = not all_vis[k][cy, cx] or (
            relief is not None and rmax[k][cy, cx] - rmin[k][cy, cx] > tolerance)
        if k > 0 and split:
            stack += [(k - 1, 2 * cy + dy, 2 * cx + dx) for dy in (0, 1) for dx in (0, 1)]
        else:
            level[cy << k : (cy + 1) << k, cx << k : (cx + 1) << k] = k

    _balance(level, any_vis, levels)
    xy, faces = _triangulate(level, levels, step, w, h)
    uv = (xy / np.array([w, h], dtype=np.float64)).astype(np.float32)
    return {"step": step, "shape": (h, w), "xy": xy, "faces": faces, "uv": uv}


# ------------------------------------------------------------------ simplification
#: Depth tolerance of each simplification level, relative to the unit's depth. A depth
#: error dZ changes the parallax of a point by the fraction dZ / Z, so a relative
#: tolerance looks the same on near and far layers.
SIMPLIFY_DEPTH_RATIO = {"off": None, "standard": 0.004, "strong": 0.012}
#: Texture shift allowed inside a triangle in the reference view, in px. UVs are
#: interpolated perspective-correctly (by the GPU and in After Effects), so inside a
#: triangle whose corners lie at different depths a texel lands up to about
#: ``size * dZ / (4 Z)`` px from its screen position, although every vertex is exact.
SIMPLIFY_UV_PX = {"off": None, "standard": 0.1, "strong": 0.1}
#: Largest cell allowed across the silhouette (and across the opaque/translucent
#: boundary), in px: the overdraw into transparent texels stays below this.
SIMPLIFY_EDGE_PX = {"off": None, "standard": 8, "strong": 16}
#: Texels at least this opaque can go into the opaque (alpha-mask) part of a GLB
#: export; keep equal to ``genparallax.export.glb.OPAQUE_TEXEL``.
OPAQUE_ALPHA = 252
#: Largest simplified cell, as a multiple of the minimum cell size (a power of two).
SIMPLIFY_MAX_FACTOR = 64


def depth_bucket(depth: float) -> float:
    """``depth`` rounded *down* to a power of sqrt(2) (the tolerance's reference depth)."""
    d = max(float(depth), 1e-6)
    return float(2.0 ** (np.floor(2.0 * np.log2(d) + 1e-9) / 2.0))


def simplify_tolerance(unit: ReconstructionUnit) -> float | None:
    """The unit's depth tolerance in px along the camera axis, or None when
    simplification is off. Derived from the bucketed depth so that moving the unit
    rebuilds the mesh only when the depth crosses a bucket."""
    ratio = SIMPLIFY_DEPTH_RATIO.get(unit.mesh_simplify, SIMPLIFY_DEPTH_RATIO["standard"])
    return None if ratio is None else ratio * depth_bucket(unit.effective_base_depth)


def edge_level(step: int, edge_px: int | None) -> int:
    """Largest quadtree level whose cells (``step << k`` px) fit in ``edge_px``."""
    step = max(1, int(step))
    k = 0
    while edge_px is not None and step << (k + 1) <= edge_px:
        k += 1
    return k


def simplify_edge_level(unit: ReconstructionUnit) -> int:
    return edge_level(unit.mesh_density, SIMPLIFY_EDGE_PX.get(unit.mesh_simplify, SIMPLIFY_EDGE_PX["standard"]))


def strength_bucket(strength: float) -> float:
    """``|strength|`` rounded *up* to a power of sqrt(2).

    The topology is built for this strength, so it stays valid (the error can only
    shrink) while the relief strength is dragged below it, and the mesh is rebuilt
    only when the strength crosses a bucket.
    """
    s = abs(float(strength))
    if s == 0.0 or not np.isfinite(s):
        return 0.0
    return float(2.0 ** (np.ceil(2.0 * np.log2(s) - 1e-9) / 2.0))


#: Sample points of a cell, in the order used by the weight tables: four corners,
#: centre and edge midpoints (left, bottom, right, top), in units of the cell size.
_CELL_POINTS = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5],
                         [0, 0.5], [0.5, 1], [1, 0.5], [0.5, 0]], dtype=np.float64)
TL, TR, BL, BR, C, ML, MB, MR, MT = range(9)
#: The triangulations :func:`_triangulate` can give a leaf cell of level >= 1: two
#: triangles, or a fan from the centre with any subset of the edge midpoints. Every fan
#: triangle is one of the 4-fan's or one of the 8-fan's, so bounding those three bounds
#: every case.
_CELL_TRIANGULATIONS = (
    ((TL, BL, TR), (TR, BL, BR)),
    ((C, TL, BL), (C, BL, BR), (C, BR, TR), (C, TR, TL)),
    ((C, TL, ML), (C, ML, BL), (C, BL, MB), (C, MB, BR), (C, BR, MR), (C, MR, TR), (C, TR, MT), (C, MT, TL)),
)


def _weight_tables(points: np.ndarray, width: int, height: int) -> list[np.ndarray]:
    """For each cell triangulation, a ``(9, height * width)`` matrix mapping the nine
    sample values to the piecewise-linear interpolant at the pixel centres.

    ``points`` are the sample positions in pixels relative to the cell's top-left.
    """
    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    px, py = px.ravel(), py.ravel()
    tables = []
    for triangles in _CELL_TRIANGULATIONS:
        table = np.zeros((9, px.size), dtype=np.float32)
        done = np.zeros(px.size, dtype=bool)
        for a, b, c in triangles:
            pa, pb, pc = points[a], points[b], points[c]
            v0, v1 = pb - pa, pc - pa
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:  # degenerate after clamping to the texture
                continue
            dx, dy = px - pa[0], py - pa[1]
            b1 = (dx * v1[1] - v1[0] * dy) / den
            b2 = (v0[0] * dy - dx * v0[1]) / den
            inside = ~done & (b1 >= -1e-9) & (b2 >= -1e-9) & (1.0 - b1 - b2 >= -1e-9)
            table[a, inside] += (1.0 - b1 - b2)[inside]
            table[b, inside] += b1[inside]
            table[c, inside] += b2[inside]
            done |= inside
        tables.append(table)
    return tables


class InterpolationError:
    """Lazily evaluated interpolation error of a field, per quadtree cell.

    ``get(k, cy, cx)`` returns, for cells of ``step << k`` px, the largest
    ``|field - L|`` over the cell's pixel centres, where ``L`` is any of the
    triangulations the cell can get (see ``_CELL_TRIANGULATIONS``) through the field
    sampled at the cell's corners, centre and edge midpoints, exactly as the mesh
    vertices sample it. Cells crossing the texture edge are evaluated on their part
    inside the texture with every point clamped to it, which is how they are meshed.
    Values are cached: the relief's error is reused while only the push field changes.
    """

    def __init__(self, field: np.ndarray, step: int, levels: int):
        self.field = np.asarray(field, dtype=np.float32)
        self.step, self.levels = step, levels
        h, w = field.shape
        root = step << levels
        self.gh, self.gw = -(-h // root) * root, -(-w // root) * root
        fp = np.empty((self.gh, self.gw), dtype=np.float32)
        fp[:h, :w] = self.field
        fp[:h, w:] = fp[:h, w - 1 : w]
        fp[h:, :] = fp[h - 1 : h, :]
        self.padded = fp
        self.cache = [np.full((self.gh // (step << k), self.gw // (step << k)), np.nan, dtype=np.float32)
                      for k in range(levels + 1)]
        #: Per level, min and max of the field at the cell's nine sample points (the
        #: vertex values), filled together with ``cache`` by :meth:`get`.
        self.lo = [np.full_like(c, np.nan) for c in self.cache]
        self.hi = [np.full_like(c, np.nan) for c in self.cache]
        self.tables: dict = {}

    def get(self, k: int, cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
        cache = self.cache[k]
        missing = np.isnan(cache[cy, cx])
        if missing.any():
            self._compute(k, cy[missing], cx[missing])
        return cache[cy, cx]

    def _compute(self, k: int, cy: np.ndarray, cx: np.ndarray) -> None:
        h, w = self.field.shape
        s = self.step << k
        x0, y0 = cx * s, cy * s
        clamped = (x0 + s > w) | (y0 + s > h)
        sq = ~clamped
        if sq.any():
            if s not in self.tables:
                self.tables[s] = _weight_tables(_CELL_POINTS * s, s, s)
            pts = (np.stack([x0[sq], y0[sq]], 1)[:, None, :] + _CELL_POINTS[None] * s).reshape(-1, 2)
            values = sample_relief(self.field, pts).reshape(-1, 9).astype(np.float32)
            self.lo[k][cy[sq], cx[sq]] = values.min(axis=1)
            self.hi[k][cy[sq], cx[sq]] = values.max(axis=1)
            blocks = self.padded.reshape(self.gh // s, s, self.gw // s, s)
            err = np.zeros(int(sq.sum()), dtype=np.float32)
            for start in range(0, len(err), max(1, (1 << 22) // (s * s))):  # bounded memory
                sl = slice(start, start + max(1, (1 << 22) // (s * s)))
                pix = blocks[cy[sq][sl], :, cx[sq][sl], :].reshape(-1, s * s)
                for table in self.tables[s]:
                    np.maximum(err[sl], np.abs(pix - values[sl] @ table).max(axis=1), out=err[sl])
            self.cache[k][cy[sq], cx[sq]] = err
        for y, x in zip(y0[clamped], x0[clamped]):
            x1, y1 = min(x + s, w), min(y + s, h)
            pts = np.minimum(np.array([x, y], dtype=np.float64) + _CELL_POINTS * s, [w, h])
            values = sample_relief(self.field, pts).astype(np.float32)
            self.lo[k][y // s, x // s] = values.min()
            self.hi[k][y // s, x // s] = values.max()
            pix = self.field[y:y1, x:x1].ravel()
            err = 0.0
            local = pts - [x, y]
            key = (x1 - x, y1 - y, local.tobytes())  # the same for a whole edge row/column
            if key not in self.tables:
                self.tables[key] = _weight_tables(local, x1 - x, y1 - y)
            for table in self.tables[key]:
                err = max(err, float(np.abs(pix - values @ table).max()))
            self.cache[k][y // s, x // s] = err


def simplified_levels(alpha: np.ndarray, step: int, tolerance: float, relief_error: InterpolationError | None,
                      strength: float, push_error: InterpolationError | None, levels: int,
                      depth: float | None = None, edge: int = 0,
                      uv_tolerance: float | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Leaf level of every minimum cell for the simplified mesh (-1 = no geometry).

    A cell is kept whole when it lies entirely inside the silhouette and its triangles
    stay within ``tolerance`` px of the depth ``Z = B + S * R + P`` at every pixel
    centre, for every ``S`` in ``(strength / sqrt 2, strength]`` and ``B >= depth``:

    * ``strength * err_R + err_P`` bounds the linear-interpolation error;
    * ``dZ**2 / (4 Z_min)`` bounds the extra error of interpolating ``1/Z``
      (perspective-correct, as the GPU does) instead of ``Z`` over the cell;
    * the texture shift of perspective-correct UV interpolation,
      ``diagonal * dZ / (4 Z_min)``, must stay within ``uv_tolerance`` px.

    Cells straddling the silhouette are split down to level ``edge``, so they
    cover every visible texel with less than one edge cell of overdraw. So are cells
    mixing texels the GLB export can draw as opaque with ones it cannot, so that large
    cells do not move opaque areas into the translucent (blended) part. Evaluated
    breadth-first, so the relief error is only computed for candidate cells. The result
    is 2:1 balanced. Returns ``(level, any_vis)``.
    """
    h, w = alpha.shape
    root = step << levels
    nby, nbx = -(-h // root), -(-w // root)
    gh, gw = nby << levels, nbx << levels
    vis = np.zeros((gh * step, gw * step), dtype=bool)
    vis[:h, :w] = alpha >= ALPHA_THRESHOLD
    # Outside the texture counts as "inside" for the all-visible test: cells crossing
    # the texture edge are clamped to it, so they need not be split there.
    full = np.ones_like(vis)
    full[:h, :w] = vis[:h, :w]
    # glb.opaque_triangles looks one texel around a triangle's bounding box.
    clear = np.zeros_like(vis)
    clear[:h, :w] = ndimage.binary_dilation(alpha < OPAQUE_ALPHA, np.ones((3, 3), bool))
    solid = np.zeros_like(vis)
    solid[:h, :w] = ~clear[:h, :w]
    any_vis, all_vis = [_block(vis, step, np.any)], [_block(full, step, np.all)]
    any_clear, any_solid = [_block(clear, step, np.any)], [_block(solid, step, np.any)]
    for k in range(1, levels + 1):
        any_vis.append(_block(any_vis[0], 1 << k, np.any))
        all_vis.append(_block(all_vis[0], 1 << k, np.all))
        any_clear.append(_block(any_clear[0], 1 << k, np.any))
        any_solid.append(_block(any_solid[0], 1 << k, np.any))
    use_relief = relief_error is not None and strength > 0.0

    def error(k: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        e = np.zeros(len(ys), dtype=np.float64)
        # Depth range and lowest depth offset over the cell's vertices: the 1/Z term and
        # the UV shift depend only on the vertex depths.
        span = np.zeros(len(ys), dtype=np.float64)
        low = np.zeros(len(ys), dtype=np.float64)
        if use_relief:
            e += strength * relief_error.get(k, ys, xs)
            rlo = relief_error.lo[k][ys, xs].astype(np.float64)
            span += strength * (relief_error.hi[k][ys, xs] - rlo)
            low += np.where(rlo < 0, strength, strength / np.sqrt(2.0)) * rlo
        if push_error is not None:
            e += push_error.get(k, ys, xs)
            span += push_error.hi[k][ys, xs] - push_error.lo[k][ys, xs]
            low += push_error.lo[k][ys, xs]
        if depth is not None:
            z_min = depth + low
            safe = np.maximum(z_min, 1e-9)
            e += np.where(z_min > 0.05 * depth, span ** 2 / (4.0 * safe), np.inf)
            if uv_tolerance is not None:
                shift = np.sqrt(2.0) * (step << k) * span / (4.0 * safe)
                e = np.where(shift > uv_tolerance, np.inf, e)
        return e

    level = np.full((gh, gw), -1, dtype=np.int16)
    cy, cx = (a.ravel() for a in np.mgrid[0:nby, 0:nbx])
    for k in range(levels, -1, -1):
        keep = any_vis[k][cy, cx]
        cy, cx = cy[keep], cx[keep]
        split = np.zeros(len(cy), dtype=bool)
        if k > 0:
            mixed = ~all_vis[k][cy, cx] | (any_clear[k][cy, cx] & any_solid[k][cy, cx])
            split = mixed & (k > edge)
            check = np.nonzero(~split)[0]
            split[check] = error(k, cy[check], cx[check]) > tolerance
        leaf = np.zeros(any_vis[k].shape, dtype=bool)
        leaf[cy[~split], cx[~split]] = True
        size = 1 << k
        level[np.repeat(np.repeat(leaf, size, 0), size, 1)] = k
        cy = (2 * cy[split, None] + np.array([0, 0, 1, 1])).ravel()
        cx = (2 * cx[split, None] + np.array([0, 1, 0, 1])).ravel()

    # Balancing splits some cells that passed the test; their children have different
    # triangles, so check those too and repeat until nothing changes.
    for _ in range(4 * levels + 4):
        _balance(level, any_vis, levels)
        changed = False
        for k in range(1, levels + 1):
            size = 1 << k
            ys, xs = np.nonzero(_block(level == k, size, np.all))
            if len(ys) == 0:
                continue
            over = error(k, ys, xs) > tolerance
            for y, x in zip(ys[over], xs[over]):
                changed = True
                for dy in (0, 1):
                    for dx in (0, 1):
                        sy, sx, half = 2 * y + dy, 2 * x + dx, size // 2
                        child = k - 1 if any_vis[k - 1][sy, sx] else -1
                        level[sy * half : (sy + 1) * half, sx * half : (sx + 1) * half] = child
        if not changed:
            break
    return level, any_vis


def build_simplified_grid(alpha: np.ndarray, step: int, tolerance: float, relief: np.ndarray | None = None,
                          strength: float = 0.0, push: np.ndarray | None = None,
                          depth: float | None = None, max_factor: int = SIMPLIFY_MAX_FACTOR,
                          relief_error: InterpolationError | None = None,
                          push_error: InterpolationError | None = None, edge: int = 0,
                          uv_tolerance: float | None = None) -> dict:
    """Simplified mesh: large triangles wherever the surface is close to planar.

    Approach: an error-bounded quadtree remesh rather than edge-collapse decimation.
    Cells of ``step * 2**k`` px (up to ``max_factor``) are kept whole while their
    triangles reproduce the per-pixel depth relief within ``tolerance`` px, measured
    exactly at every pixel centre (see :func:`simplified_levels`; ``depth`` is the
    lowest base depth the mesh is used at, for the perspective term). The test is the linear-interpolation error, not the
    depth range, so smooth slopes become large cells as well as flat areas.
    Compared with edge collapse this

    * keeps every vertex on an integer pixel corner, hence on its reference ray and
      with an exact UV; inside a triangle, perspective-correct UV interpolation still
      shifts the texture where the depth varies, so that shift is bounded per cell
      (``uv_tolerance``) and the reference view keeps reproducing the PSD;
    * covers every visible texel by construction: silhouette cells are split down to
      level ``edge`` (a few px), so the overdraw into transparent texels stays slight;
    * is crack-free (2:1 balanced, edge midpoints instead of T-junctions);
    * is vectorised NumPy with no per-collapse priority queue in Python, so it is fast
      enough to rebuild while the layer-order pass updates the push field.

    The relief error depends only on the relief, so it is cached (``relief_error``)
    and reused when only the push field ``P`` changes; ``P`` gets its own error.
    """
    levels = int(round(np.log2(max(1, max_factor))))
    step = max(1, int(step))
    h, w = alpha.shape
    if relief is not None and strength > 0.0 and relief_error is None:
        relief_error = InterpolationError(relief, step, levels)
    if push is not None and push_error is None:
        push_error = InterpolationError(push, step, levels)
    level, _ = simplified_levels(alpha, step, tolerance, relief_error, strength, push_error, levels, depth,
                                 edge, uv_tolerance)
    xy, faces = _triangulate(level, levels, step, w, h)
    uv = (xy / np.array([w, h], dtype=np.float64)).astype(np.float32)
    return {"step": step, "shape": (h, w), "xy": xy, "faces": faces, "uv": uv, "level": level,
            "relief_error": relief_error, "push": push, "levels": levels, "depth": depth, "edge": edge,
            "uv_tolerance": uv_tolerance}


def _balance(level: np.ndarray, any_vis: list[np.ndarray], levels: int) -> None:
    """Split leaves until no 4-neighbour leaf is more than one level finer."""
    empty = np.int16(levels + 10)
    for _ in range(4 * levels + 4):
        lv = np.where(level < 0, empty, level)
        nb = np.full_like(lv, empty)
        nb[1:, :] = np.minimum(nb[1:, :], lv[:-1, :])
        nb[:-1, :] = np.minimum(nb[:-1, :], lv[1:, :])
        nb[:, 1:] = np.minimum(nb[:, 1:], lv[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], lv[:, 1:])
        changed = False
        for k in range(2, levels + 1):
            size = 1 << k
            leaves = _block(level == k, size, np.all)
            ys, xs = np.nonzero(leaves & (_block(nb, size, np.min) < k - 1))
            for cy, cx in zip(ys, xs):
                changed = True
                for dy in (0, 1):
                    for dx in (0, 1):
                        sy, sx = 2 * cy + dy, 2 * cx + dx
                        s = size // 2
                        level[sy * s : (sy + 1) * s, sx * s : (sx + 1) * s] = k - 1 if any_vis[k - 1][sy, sx] else -1
        if not changed:
            return


def _triangulate(level: np.ndarray, levels: int, step: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    gh, gw = level.shape

    def finer(y: int, x: int, k: int) -> bool:
        return 0 <= y < gh and 0 <= x < gw and 0 <= level[y, x] < k

    tris: list[np.ndarray] = []
    for k in range(levels + 1):
        size = 1 << k
        ys, xs = np.nonzero(_block(level == k, size, np.all))
        if len(ys) == 0:
            continue
        if k == 0:  # finest cells never need midpoints
            tl = np.stack([xs, ys], axis=1)
            bl, br, tr = tl + [0, 1], tl + [1, 1], tl + [1, 0]
            tris.append(np.stack([tl, bl, tr], axis=1))
            tris.append(np.stack([tr, bl, br], axis=1))
            continue
        half = size // 2
        for cy, cx in zip(ys, xs):
            y0, x0 = int(cy) * size, int(cx) * size
            y1, x1 = y0 + size, x0 + size
            left = finer(y0, x0 - 1, k) or finer(y0 + half, x0 - 1, k)
            bottom = finer(y1, x0, k) or finer(y1, x0 + half, k)
            right = finer(y0, x1, k) or finer(y0 + half, x1, k)
            top = finer(y0 - 1, x0, k) or finer(y0 - 1, x0 + half, k)
            if not (left or bottom or right or top):
                tris.append(np.array([[[x0, y0], [x0, y1], [x1, y0]], [[x1, y0], [x0, y1], [x1, y1]]]))
                continue
            # Boundary ring in the same winding as the uniform grid, fanned from the centre.
            ring = [(x0, y0)] + ([(x0, y0 + half)] if left else []) + [(x0, y1)] \
                + ([(x0 + half, y1)] if bottom else []) + [(x1, y1)] \
                + ([(x1, y0 + half)] if right else []) + [(x1, y0)] \
                + ([(x0 + half, y0)] if top else [])
            c = (x0 + half, y0 + half)
            tris.append(np.array([[c, ring[i], ring[(i + 1) % len(ring)]] for i in range(len(ring))]))

    if not tris:
        return np.zeros((0, 2)), np.zeros((0, 3), np.uint32)
    corners = np.concatenate(tris).reshape(-1, 2).astype(np.int64) * step
    np.minimum(corners[:, 0], w, out=corners[:, 0])  # cells crossing the texture edge
    np.minimum(corners[:, 1], h, out=corners[:, 1])
    keys = corners[:, 1] * (w + 1) + corners[:, 0]
    uniq, inverse = np.unique(keys, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    xy = np.stack([uniq % (w + 1), uniq // (w + 1)], axis=1).astype(np.float64)
    p = xy[faces]
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    faces = faces[np.abs(area) > 1e-9]  # collapsed by clamping
    used = np.zeros(len(xy), dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return xy[used], remap[faces].astype(np.uint32)


def sample_relief(relief: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample a per-pixel field at pixel-corner positions."""
    coords = np.stack([xy[:, 1] - 0.5, xy[:, 0] - 0.5])  # pixel centres sit at +0.5
    return ndimage.map_coordinates(relief, coords, order=1, mode="nearest").astype(np.float64)


def vertex_depths(unit: ReconstructionUnit, xy: np.ndarray, camera: Camera) -> np.ndarray:
    base = unit.effective_base_depth
    z = np.full(len(xy), base, dtype=np.float64)
    use_relief = unit.mode in (
        ReconstructionMode.DEPTH_SURFACE,
        ReconstructionMode.SMOOTHED_SURFACE,
        ReconstructionMode.MANUAL_DEPTH,
    )
    if use_relief and unit.relief is not None and unit.depth_strength != 0.0:
        z += unit.depth_strength * sample_relief(unit.relief, xy)
    if use_relief and unit.depth_push is not None:
        z += sample_relief(unit.depth_push, xy)
    return np.maximum(z, camera.near * 1.01)


def rasterise_coverage(xy: np.ndarray, faces: np.ndarray, shape: tuple[int, int],
                       chunk: int = 2_000_000) -> dict:
    """Which triangle covers each pixel centre, and where inside it.

    Depends only on the mesh topology and the vertices' image positions, which never
    change when depths are edited, so it is computed once per grid. Returns flat pixel
    indices, the covering triangle and the barycentric weights ``b1``, ``b2``.
    """
    h, w = shape
    tri_map = np.full(h * w, -1, dtype=np.int32)
    b1_map = np.zeros(h * w)
    b2_map = np.zeros(h * w)
    if len(faces):
        p = xy[faces]  # (M, 3, 2)
        x0 = np.clip(np.ceil(p[:, :, 0].min(1) - 0.5), 0, w).astype(np.int64)
        x1 = np.clip(np.floor(p[:, :, 0].max(1) - 0.5), -1, w - 1).astype(np.int64)
        y0 = np.clip(np.ceil(p[:, :, 1].min(1) - 0.5), 0, h).astype(np.int64)
        y1 = np.clip(np.floor(p[:, :, 1].max(1) - 0.5), -1, h - 1).astype(np.int64)
        size = np.maximum(x1 - x0, y1 - y0) + 1
        v0 = p[:, 1] - p[:, 0]
        v1 = p[:, 2] - p[:, 0]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        valid = (size > 0) & (np.abs(den) > 1e-12)
        buckets = np.ceil(np.log2(np.maximum(size, 1))).astype(np.int64)
        for k in np.unique(buckets[valid]):
            s = 1 << int(k)
            tri_all = np.nonzero(valid & (buckets == k))[0]
            oy, ox = np.mgrid[0:s, 0:s]
            ox, oy = ox.ravel(), oy.ravel()
            per = max(1, chunk // (s * s))
            for start in range(0, len(tri_all), per):
                tri = tri_all[start : start + per]
                px = x0[tri, None] + ox[None, :]
                py = y0[tri, None] + oy[None, :]
                inside = (px <= x1[tri, None]) & (py <= y1[tri, None])
                cx = px + 0.5 - p[tri, 0, 0, None]
                cy = py + 0.5 - p[tri, 0, 1, None]
                d = den[tri, None]
                b1 = (cx * v1[tri, 1, None] - v1[tri, 0, None] * cy) / d
                b2 = (v0[tri, 0, None] * cy - cx * v0[tri, 1, None]) / d
                eps = -1e-9
                inside &= (b1 >= eps) & (b2 >= eps) & (1.0 - b1 - b2 >= eps)
                flat = py[inside] * w + px[inside]
                tri_map[flat] = np.broadcast_to(tri[:, None], inside.shape)[inside]
                b1_map[flat] = b1[inside]
                b2_map[flat] = b2[inside]
    covered = np.nonzero(tri_map >= 0)[0]
    tri = tri_map[covered]
    return {"pixels": covered, "tri": tri, "b1": b1_map[covered], "b2": b2_map[covered], "shape": (h, w),
            "verts": np.ascontiguousarray(faces[tri].T) if len(faces) else np.zeros((3, 0), np.int64)}


def depth_from_coverage(coverage: dict, faces: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-pixel depth (NaN where uncovered), interpolating ``1/Z`` perspective-correctly."""
    h, w = coverage["shape"]
    out = np.full(h * w, np.nan)
    f0, f1, f2 = coverage["verts"]  # vertex indices of the covering triangle, per pixel
    iz = 1.0 / z
    b1, b2 = coverage["b1"], coverage["b2"]
    inv = iz[f0] + b1 * (iz[f1] - iz[f0]) + b2 * (iz[f2] - iz[f0])
    out[coverage["pixels"]] = 1.0 / inv
    return out.reshape(h, w)


def rasterise_depth(xy: np.ndarray, faces: np.ndarray, z: np.ndarray, shape: tuple[int, int],
                    chunk: int = 2_000_000) -> np.ndarray:
    """Depth of a mesh seen from the reference camera, sampled at pixel centres.

    ``xy`` are texture-pixel positions of the vertices, ``z`` their camera depths.
    Interpolates ``1/Z`` linearly in image space (perspective-correct), which is
    what the GPU depth test compares. Uncovered pixels are NaN.
    """
    return depth_from_coverage(rasterise_coverage(xy, faces, shape, chunk), faces, z)


def _rasterise_depth_direct(xy: np.ndarray, faces: np.ndarray, z: np.ndarray, shape: tuple[int, int],
                            chunk: int = 2_000_000) -> np.ndarray:
    """Reference implementation (one pass, no coverage table); kept for tests."""
    h, w = shape
    out = np.full((h, w), np.nan)
    if len(faces) == 0:
        return out
    p = xy[faces]  # (M, 3, 2)
    iz = 1.0 / z[faces]  # (M, 3)
    x0 = np.clip(np.ceil(p[:, :, 0].min(1) - 0.5), 0, w).astype(np.int64)
    x1 = np.clip(np.floor(p[:, :, 0].max(1) - 0.5), -1, w - 1).astype(np.int64)
    y0 = np.clip(np.ceil(p[:, :, 1].min(1) - 0.5), 0, h).astype(np.int64)
    y1 = np.clip(np.floor(p[:, :, 1].max(1) - 0.5), -1, h - 1).astype(np.int64)
    size = np.maximum(x1 - x0, y1 - y0) + 1
    v0 = p[:, 1] - p[:, 0]
    v1 = p[:, 2] - p[:, 0]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = (size > 0) & (np.abs(den) > 1e-12)
    buckets = np.ceil(np.log2(np.maximum(size, 1))).astype(np.int64)
    for k in np.unique(buckets[valid]):
        s = 1 << int(k)
        tri_all = np.nonzero(valid & (buckets == k))[0]
        oy, ox = np.mgrid[0:s, 0:s]
        ox, oy = ox.ravel(), oy.ravel()
        per = max(1, chunk // (s * s))
        for start in range(0, len(tri_all), per):
            tri = tri_all[start : start + per]
            px = x0[tri, None] + ox[None, :]
            py = y0[tri, None] + oy[None, :]
            inside = (px <= x1[tri, None]) & (py <= y1[tri, None])
            cx = px + 0.5 - p[tri, 0, 0, None]
            cy = py + 0.5 - p[tri, 0, 1, None]
            d = den[tri, None]
            b1 = (cx * v1[tri, 1, None] - v1[tri, 0, None] * cy) / d
            b2 = (v0[tri, 0, None] * cy - cx * v0[tri, 1, None]) / d
            b0 = 1.0 - b1 - b2
            eps = -1e-9
            inside &= (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
            inv = b0 * iz[tri, 0, None] + b1 * iz[tri, 1, None] + b2 * iz[tri, 2, None]
            out[py[inside], px[inside]] = 1.0 / inv[inside]
    return out


def mesh_depth_map(unit: ReconstructionUnit, camera: Camera) -> np.ndarray | None:
    """Per-pixel depth of the unit's current mesh over its bbox (cached per revision)."""
    if unit.mesh is None or unit.grid_cache is None:
        return None
    if unit.depth_cache is not None and unit.depth_cache[0] == unit.revision:
        return unit.depth_cache[1]
    grid = unit.grid_cache
    if "coverage" not in grid:  # once per topology
        grid["coverage"] = rasterise_coverage(grid["xy"], grid["faces"].astype(np.int64), unit.alpha.shape)
    z = vertex_depths(unit, grid["xy"], camera)
    depth = depth_from_coverage(grid["coverage"], grid["faces"].astype(np.int64), z)
    unit.depth_cache = (unit.revision, depth)
    return depth


def _uses_relief(unit: ReconstructionUnit) -> bool:
    return unit.mode in (ReconstructionMode.DEPTH_SURFACE, ReconstructionMode.SMOOTHED_SURFACE,
                         ReconstructionMode.MANUAL_DEPTH)


def _simplify_inputs(unit: ReconstructionUnit) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """(relief, bucketed strength, push) as :func:`vertex_depths` uses them. A negative
    strength is folded into the relief (``S * R == |S| * -R``), so the strength is >= 0."""
    if not _uses_relief(unit):
        return None, 0.0, None
    if unit.relief is None or unit.depth_strength == 0.0:
        return None, 0.0, unit.depth_push
    relief = unit.relief if unit.depth_strength > 0 else -unit.relief
    return relief, strength_bucket(unit.depth_strength), unit.depth_push


def _relief_key(unit: ReconstructionUnit) -> tuple:
    return (unit.relief_revision, unit.depth_strength > 0, max(1, int(unit.mesh_density)))


def grid_key(unit: ReconstructionUnit) -> tuple:
    h, w = unit.texture.shape[:2]
    step = max(1, int(unit.mesh_density))
    tolerance = simplify_tolerance(unit)
    if tolerance is not None:  # the push field is checked separately (see update_unit_mesh)
        relief, strength, _ = _simplify_inputs(unit)
        return (step, (h, w), "simplified", tolerance, simplify_edge_level(unit),
                depth_bucket(unit.effective_base_depth), strength, _relief_key(unit) if relief is not None else None)
    return (step, (h, w), bool(unit.adaptive_mesh), float(unit.mesh_detail),
            unit.relief_revision if unit.adaptive_mesh else None)


def _build_grid_for(unit: ReconstructionUnit, old: dict | None, push_error=None) -> dict:
    tolerance = simplify_tolerance(unit)
    if tolerance is None:
        if unit.adaptive_mesh:
            return build_adaptive_grid(unit.alpha, unit.relief, unit.mesh_density, unit.mesh_detail)
        return build_grid(unit.alpha, unit.mesh_density)
    relief, strength, push = _simplify_inputs(unit)
    relief_error = None
    if relief is not None:  # reuse the relief error while only the push field changed
        if old is not None and old.get("relief_key") == _relief_key(unit):
            relief_error = old.get("relief_error")
    grid = build_simplified_grid(unit.alpha, unit.mesh_density, tolerance, relief, strength, push,
                                 depth_bucket(unit.effective_base_depth), relief_error=relief_error,
                                 push_error=push_error, edge=simplify_edge_level(unit),
                                 uv_tolerance=SIMPLIFY_UV_PX.get(unit.mesh_simplify, SIMPLIFY_UV_PX["standard"]))
    if relief is not None:
        grid["relief_key"] = _relief_key(unit)
    return grid


def _push_still_fits(unit: ReconstructionUnit, grid: dict) -> tuple[bool, InterpolationError | None]:
    """Whether a simplified grid built for an older push field has exactly the topology
    the current push field needs (so the coverage cache stays valid), and the push
    field's interpolation error for a rebuild."""
    relief, strength, push = _simplify_inputs(unit)
    if grid.get("push") is push:
        return True, None
    push_error = InterpolationError(push, grid["step"], grid["levels"]) if push is not None else None
    level, _ = simplified_levels(unit.alpha, grid["step"], simplify_tolerance(unit),
                                 grid.get("relief_error") if relief is not None else None,
                                 strength, push_error, grid["levels"], grid["depth"], grid["edge"],
                                 grid["uv_tolerance"])
    return bool(np.array_equal(level, grid["level"])), push_error


def update_unit_mesh(unit: ReconstructionUnit, camera: Camera, force_grid: bool = False) -> Mesh | None:
    """(Re)compute the unit mesh. Rebuilds the topology only when needed."""
    if unit.mode == ReconstructionMode.EXCLUDED:
        unit.mesh = None
        unit.revision = next_revision()
        return None
    key = grid_key(unit)
    grid = unit.grid_cache
    rebuild = force_grid or grid is None or grid.get("key") != key
    push_error = None
    if not rebuild and "level" in grid and grid.get("push") is not unit.depth_push:
        fits, push_error = _push_still_fits(unit, grid)
        if fits:
            grid = dict(grid, push=unit.depth_push)  # a copy: snapshots may share the old dict
            unit.grid_cache = grid
        else:
            rebuild = True
    if rebuild:
        grid = _build_grid_for(unit, grid, push_error)
        grid["key"] = key
        unit.grid_cache = grid

    xy = grid["xy"]
    z = vertex_depths(unit, xy, camera)
    u = xy[:, 0] + unit.offset[0]
    v = xy[:, 1] + unit.offset[1]
    verts = camera.back_project(u, v, z)
    mesh = Mesh(
        vertices=verts.astype(np.float32),
        triangles=grid["faces"],
        uvs=grid["uv"],
        material_ref=unit.id,
    )
    unit.mesh = mesh  # normals are computed on export; the preview is unlit
    unit.revision = next_revision()
    return mesh
