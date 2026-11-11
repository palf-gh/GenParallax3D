"""Mesh simplification: large triangles where the surface is planar, with the
silhouette, the reference rays, the UVs and the depth tolerance kept."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from genparallax.document.psd_loader import PsdDocument
from genparallax.project import load_project, save_project
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.constraints import check_layer_order, resolve_layer_order
from genparallax.reconstruction.meshing import (
    ALPHA_THRESHOLD, SIMPLIFY_DEPTH_RATIO, build_adaptive_grid, build_simplified_grid, rasterise_coverage,
    sample_relief, strength_bucket, update_unit_mesh,
)

from conftest import H, W


def _boundary_edges(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1], counts


def _linear_error(grid, field, shape):
    """|piecewise-linear mesh interpolant - field| at every covered pixel centre (NaN elsewhere)."""
    h, w = shape
    cov = rasterise_coverage(grid["xy"], grid["faces"].astype(np.int64), shape)
    z = sample_relief(field, grid["xy"])
    f0, f1, f2 = cov["verts"]
    lin = np.full(h * w, np.nan)
    lin[cov["pixels"]] = z[f0] + cov["b1"] * (z[f1] - z[f0]) + cov["b2"] * (z[f2] - z[f0])
    return np.abs(lin.reshape(h, w) - field)


def _smooth_field(h, w, seed=0, sigma=12.0):
    noise = np.random.default_rng(seed).normal(0, 1, (h, w))
    f = ndimage.gaussian_filter(noise, sigma)
    return (f / np.abs(f).max()).astype(np.float32)


def test_planar_relief_collapses_to_the_largest_cells():
    h, w = 300, 520
    yy, xx = np.mgrid[0:h, 0:w]
    relief = (0.002 * xx - 0.001 * yy).astype(np.float32)  # a tilted plane: big range, no curvature
    alpha = np.full((h, w), 255, np.uint8)
    simplified = build_simplified_grid(alpha, 4, 2.0, relief, strength_bucket(1000.0))
    adaptive = build_adaptive_grid(alpha, relief, 4, 0.03)
    # cells of 256 px, clamped to the texture: a 3 x 2 grid of cells. (Vertices on the
    # texture edge sample the edge pixel, half a pixel off the plane, which is within
    # the tolerance here.)
    assert len(simplified["xy"]) <= 12
    assert len(simplified["xy"]) * 50 < len(adaptive["xy"])
    err = _linear_error(simplified, 1000.0 * relief, (h, w))
    assert not np.isnan(err).any()
    assert err.max() <= 2.0


@pytest.mark.parametrize("strength,tolerance", [(300.0, 1.0), (2000.0, 2.0), (2000.0, 8.0)])
def test_depth_error_stays_within_tolerance(strength, tolerance):
    """Every pixel of a cell larger than the minimum is within the tolerance of S * R + P
    (the minimum cell is as fine as the mesh can go)."""
    h, w, step = 233, 301, 2
    relief = _smooth_field(h, w, sigma=40.0)
    push = 40.0 * _smooth_field(h, w, seed=1, sigma=30.0)
    alpha = np.full((h, w), 255, np.uint8)
    s = strength_bucket(strength)
    grid = build_simplified_grid(alpha, step, tolerance, relief, s, push)
    err = _linear_error(grid, strength * relief + push, (h, w))
    level = np.repeat(np.repeat(grid["level"], step, 0), step, 1)[:h, :w]
    coarse = level >= 1
    assert coarse.mean() > 0.5  # the test means something
    assert np.nanmax(err[coarse]) <= tolerance + 1e-3


@pytest.mark.parametrize("edge", [0, 1, 2])
def test_every_visible_texel_is_covered(edge):
    h, w = 180, 260
    yy, xx = np.mgrid[0:h, 0:w]
    alpha = np.zeros((h, w), np.uint8)
    alpha[np.hypot(xx - 120, yy - 90) < 80] = 255
    alpha[np.hypot(xx - 120, yy - 90) < 25] = 0  # a hole
    alpha[10:14, 200:250] = 3  # a faint antialiased sliver
    alpha[np.hypot(xx - 120, yy - 90) < 81] |= 1
    grid = build_simplified_grid(alpha, 4, 2.0, _smooth_field(h, w), 500.0, edge=edge)
    cov = rasterise_coverage(grid["xy"], grid["faces"].astype(np.int64), (h, w))
    covered = np.zeros(h * w, bool)
    covered[cov["pixels"]] = True
    assert covered[(alpha >= ALPHA_THRESHOLD).ravel()].all()
    # overdraw stays within one edge cell of the silhouette
    cell = 4 << edge
    near = ndimage.binary_dilation(alpha >= ALPHA_THRESHOLD, np.ones((2 * cell + 1, 2 * cell + 1), bool))
    assert not covered[~near.ravel()].any()


def test_opaque_areas_stay_in_the_opaque_export_part():
    """Large cells must not move opaque texels into the blended GLB primitive: cells
    mixing opaque and translucent texels are split like silhouette cells."""
    from genparallax.export.glb import OPAQUE_TEXEL, opaque_triangles
    from genparallax.reconstruction.meshing import OPAQUE_ALPHA

    assert OPAQUE_ALPHA == OPAQUE_TEXEL
    h, w = 400, 600
    yy, xx = np.mgrid[0:h, 0:w]
    alpha = np.full((h, w), 255, np.uint8)
    alpha[np.hypot(xx - 300, yy - 200) < 60] = 128  # a translucent window in an opaque layer
    alpha[(xx // 50 + yy // 50) % 7 == 0] = 200  # scattered translucent squares
    opaque = alpha >= OPAQUE_TEXEL

    def opaque_in_mask(grid):
        flags = opaque_triangles(grid["uv"], grid["faces"], alpha)
        cov = rasterise_coverage(grid["xy"], grid["faces"].astype(np.int64), (h, w))
        mask = np.zeros(h * w, bool)
        mask[cov["pixels"]] = flags[cov["tri"]]
        return (mask & opaque.ravel()).sum()

    reference = opaque_in_mask(build_adaptive_grid(alpha, None, 4, 0.03))
    simplified = build_simplified_grid(alpha, 4, 2.0, edge=1)
    assert opaque_in_mask(simplified) >= 0.98 * reference


@pytest.mark.parametrize("w,h,step", [(97, 61, 4), (300, 200, 2), (33, 70, 3)])
def test_simplified_mesh_is_watertight(w, h, step):
    alpha = np.full((h, w), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    relief = np.where(np.hypot(xx - w / 3, yy - h / 2) < min(w, h) / 5, 0.5, -0.5).astype(np.float32)
    relief = ndimage.gaussian_filter(relief, 2.0)
    g = build_simplified_grid(alpha, step, 1.0, relief, 200.0)
    edges, counts = _boundary_edges(g["faces"])
    assert counts.max() == 2  # manifold
    p = g["xy"][edges]
    on_border = ((p[:, :, 0] == 0).all(1) | (p[:, :, 0] == w).all(1) | (p[:, :, 1] == 0).all(1)
                 | (p[:, :, 1] == h).all(1))
    assert on_border.all()  # no T-junction cracks
    q = g["xy"][g["faces"]]
    area = 0.5 * ((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                  - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))
    assert (area < 0).all()  # consistent winding (as the uniform grid), nothing degenerate
    assert -area.sum() == pytest.approx(w * h)


@pytest.fixture
def doc(psd_path):
    return PsdDocument(psd_path)


def _depth_map():
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - W / 2, yy - H / 2) / np.hypot(W / 2, H / 2)
    return (1 - r).astype(np.float32)


def test_vertices_stay_on_reference_rays_with_exact_uvs(doc):
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, _depth_map(), "radial")
    for unit in scene.units:
        assert unit.mesh_simplify == "standard"
        unit.depth_strength *= 3
        update_unit_mesh(unit, scene.reference_camera)
        grid = unit.grid_cache
        assert "level" in grid  # built by the simplifier
        got = scene.reference_camera.project(unit.mesh.vertices.astype(np.float64))
        assert np.abs(got - (grid["xy"] + np.array(unit.offset))).max() < 2e-3
        h, w = unit.alpha.shape
        assert np.array_equal(unit.mesh.uvs, (grid["xy"] / np.array([w, h])).astype(np.float32))
        assert np.array_equal(grid["xy"], np.round(grid["xy"]))  # integer pixel corners


def test_mesh_depth_is_within_tolerance_of_the_relief(doc):
    """The mesh as the GPU draws it (``1/Z`` interpolated) stays within the unit's
    tolerance of B + S * R + P, outside the minimum cells."""
    from genparallax.reconstruction.meshing import mesh_depth_map, simplify_tolerance

    scene = unitlib.scene_from_document(doc)
    bg = next(u for u in scene.units if u.name == "background")
    h, w = bg.alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bg.relief = (_smooth_field(h, w, seed=3, sigma=30.0) + 0.8 * yy / h).astype(np.float32)
    bg.relief_revision += 1
    bg.depth_strength = 0.3 * bg.base_depth  # strong relief: the perspective term matters
    bg.depth_push = 0.05 * bg.base_depth * np.clip(1.0 - np.hypot(xx - 50, yy - 40) / 40.0, 0, None) ** 2
    for level in ("standard", "strong"):
        bg.mesh_simplify = level
        update_unit_mesh(bg, scene.reference_camera)
        depth = mesh_depth_map(bg, scene.reference_camera)
        target = bg.effective_base_depth + bg.depth_strength * bg.relief + bg.depth_push
        cells = np.repeat(np.repeat(bg.grid_cache["level"], bg.grid_cache["step"], 0), bg.grid_cache["step"], 1)
        coarse = cells[:h, :w] >= 1
        assert coarse.mean() > 0.1
        assert np.abs(depth - target)[coarse].max() <= simplify_tolerance(bg) * 1.001


def test_simplification_reduces_vertices(doc):
    """A smooth, ground-like relief: the background becomes a few large cells."""
    scene = unitlib.scene_from_document(doc)
    bg = next(u for u in scene.units if u.name == "background")
    yy, xx = np.mgrid[0:H, 0:W]
    bg.relief = ((yy / H) ** 2 + 0.3 * xx / W).astype(np.float32)
    bg.relief_revision += 1
    bg.depth_strength = 40.0
    counts = {}
    for level in ("off", "standard", "strong"):
        bg.mesh_simplify = level
        update_unit_mesh(bg, scene.reference_camera)
        counts[level] = len(bg.mesh.vertices)
    assert counts["standard"] < 0.5 * counts["off"]
    assert counts["strong"] <= counts["standard"]


@pytest.mark.parametrize("level", ["standard", "strong"])
def test_layer_order_resolves_without_shifting_surfaces(doc, level):
    """The radial map puts the background centre in front of the upper layers. With
    simplification the push field must still resolve it locally, as without."""
    results = {}
    for simplify in ("off", level):
        scene = unitlib.scene_from_document(doc)
        for u in scene.units:
            u.mesh_simplify = simplify
        unitlib.set_global_depth(scene, _depth_map(), "radial")
        for u in scene.units:
            u.depth_strength *= 3
            unitlib.refresh_unit(scene, u, relief=False)
        resolve_layer_order(scene)
        assert check_layer_order(scene) == []
        results[simplify] = {u.name: u.effective_base_depth for u in scene.units}
    for name, base in results["off"].items():
        assert results[level][name] == pytest.approx(base, rel=0.01), name


def test_push_change_reuses_topology_when_it_fits(doc):
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, _depth_map(), "radial")
    bg = next(u for u in scene.units if u.name == "background")
    update_unit_mesh(bg, scene.reference_camera)
    grid = bg.grid_cache
    bg.depth_push = np.full(bg.alpha.shape, 5.0)  # a constant push adds no interpolation error
    update_unit_mesh(bg, scene.reference_camera)
    assert bg.grid_cache["faces"] is grid["faces"]
    assert bg.grid_cache["push"] is bg.depth_push
    yy, xx = np.mgrid[0 : bg.alpha.shape[0], 0 : bg.alpha.shape[1]]
    bg.depth_push = 200.0 * np.exp(-((xx - 80.0) ** 2 + (yy - 60.0) ** 2) / 50.0)  # a sharp bump
    update_unit_mesh(bg, scene.reference_camera)
    assert len(bg.grid_cache["faces"]) > len(grid["faces"])


def test_setting_survives_project_round_trip(doc, tmp_path):
    scene = unitlib.scene_from_document(doc)
    scene.units[0].mesh_simplify = "off"
    scene.units[1].mesh_simplify = "strong"
    proj = save_project(tmp_path / "p.gp3d", scene, doc)
    scene2, _, _ = load_project(proj)
    assert [u.mesh_simplify for u in scene2.units] == [u.mesh_simplify for u in scene.units]
    assert set(SIMPLIFY_DEPTH_RATIO) == {"off", "standard", "strong"}  # the panel's choices


@pytest.mark.parametrize("level", ["standard", "strong"])
def test_overlay_texels_stay_in_front_of_the_surface_below(haze_psd, level):
    """An overlay sits only a hair in front of the layer it is laid on; the simplified
    meshes of both must keep every drawn overlay texel in front, or the depth test
    hides it in the reference view."""
    from genparallax.reconstruction.constraints import depth_span

    doc = PsdDocument(haze_psd, force_composite=True)
    scene = unitlib.scene_from_document(doc)
    for u in scene.units:
        u.mesh_simplify = level
    haze, ground = (next(u for u in scene.units if u.name == n) for n in ("haze", "ground"))
    yy, xx = np.mgrid[0:H, 0:W]
    bumpy = 0.4 + 0.3 * yy / H + 0.15 * np.sin(xx / 13.0) * np.cos(yy / 17.0)
    unitlib.set_global_depth(scene, bumpy.astype(np.float32), "bumpy", assign=True)
    cam = scene.reference_camera
    _, haze_far = depth_span(haze, cam)
    g_near, _ = depth_span(ground, cam)
    hl, ht, hr, hb = haze.bbox
    drawn = haze.alpha > 0
    assert (haze_far[drawn] < g_near[ht:hb, hl:hr][drawn]).all()


@pytest.mark.parametrize("level", ["standard", "strong"])
def test_texture_stays_in_place_in_the_reference_view(doc, level):
    """UVs are interpolated perspective-correctly, so inside a triangle spanning a depth
    range the texture slides away from the screen position. Each pixel centre must
    sample the texture within the level's UV tolerance of itself."""
    from genparallax.reconstruction.meshing import SIMPLIFY_UV_PX, vertex_depths

    scene = unitlib.scene_from_document(doc)
    bg = next(u for u in scene.units if u.name == "background")
    h, w = bg.alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bg.relief = (1.5 * yy / h + 0.5 * xx / w).astype(np.float32)  # a steep, planar-in-Z slope
    bg.relief_revision += 1
    bg.depth_strength = 0.2 * bg.base_depth
    bg.mesh_simplify = level
    update_unit_mesh(bg, scene.reference_camera)
    grid = bg.grid_cache
    faces = grid["faces"].astype(np.int64)
    cov = rasterise_coverage(grid["xy"], faces, (h, w))
    iz = 1.0 / vertex_depths(bg, grid["xy"], scene.reference_camera)
    f0, f1, f2 = cov["verts"]
    b1, b2 = cov["b1"], cov["b2"]
    weights = np.stack([(1 - b1 - b2) * iz[f0], b1 * iz[f1], b2 * iz[f2]])
    weights /= weights.sum(0)
    sampled = sum(weights[i, :, None] * grid["xy"][f] for i, f in enumerate((f0, f1, f2)))
    centre = np.stack([cov["pixels"] % w + 0.5, cov["pixels"] // w + 0.5], 1)
    shift = np.abs(sampled - centre).max(axis=1)
    # the minimum cell is as fine as the mesh goes; every larger cell keeps the bound
    cells = np.repeat(np.repeat(grid["level"], grid["step"], 0), grid["step"], 1)[:h, :w].ravel()
    coarse = cells[cov["pixels"]] >= 1
    assert coarse.mean() > 0.2
    assert shift[coarse].max() <= SIMPLIFY_UV_PX[level]
