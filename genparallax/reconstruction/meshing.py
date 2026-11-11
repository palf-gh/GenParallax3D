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


def grid_key(unit: ReconstructionUnit) -> tuple:
    h, w = unit.texture.shape[:2]
    return (max(1, int(unit.mesh_density)), (h, w), bool(unit.adaptive_mesh), float(unit.mesh_detail),
            unit.relief_revision if unit.adaptive_mesh else None)


def update_unit_mesh(unit: ReconstructionUnit, camera: Camera, force_grid: bool = False) -> Mesh | None:
    """(Re)compute the unit mesh. Rebuilds the topology only when needed."""
    if unit.mode == ReconstructionMode.EXCLUDED:
        unit.mesh = None
        unit.revision = next_revision()
        return None
    key = grid_key(unit)
    grid = unit.grid_cache
    if force_grid or grid is None or grid.get("key") != key:
        if unit.adaptive_mesh:
            grid = build_adaptive_grid(unit.alpha, unit.relief, unit.mesh_density, unit.mesh_detail)
        else:
            grid = build_grid(unit.alpha, unit.mesh_density)
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
