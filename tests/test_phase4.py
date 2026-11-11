"""Phase 4: depth assignment, layer-order constraints, adaptive meshing, contours, reload."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from genparallax.document.psd_loader import PsdDocument
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.constraints import check_layer_order, resolve_layer_order
from genparallax.reconstruction.depth import inpaint, load_depth_image
from genparallax.reconstruction.meshing import build_adaptive_grid, build_grid, update_unit_mesh
from genparallax.scene import ReconstructionMode

from conftest import H, W


def _unit(scene, name):
    return next(u for u in scene.units if u.name == name)


def _layered_depth(doc, values: dict[str, float]) -> np.ndarray:
    """A depth map painted like an AI result: each visible layer region gets its nearness."""
    n = np.zeros((H, W), np.float32)
    for info in doc.leaves():
        if info.name not in values or not doc.effectively_visible(info.key):
            continue
        rgba, l, t = doc.layer_rgba(info.key)
        region = n[t : t + rgba.shape[0], l : l + rgba.shape[1]]
        a = rgba[:, :, 3] > 127
        region[a] = values[info.name]
    return n


@pytest.fixture
def doc(psd_path):
    return PsdDocument(psd_path)


def test_depth_assignment_uses_only_visible_pixels(doc):
    """The background is covered by the house and the character; their nearness must not
    leak into the background's depth or relief."""
    scene = unitlib.scene_from_document(doc)
    n = _layered_depth(doc, {"background": 0.1, "wall": 0.5, "roof": 0.6, "character": 0.9})
    unitlib.set_global_depth(scene, n, "painted", assign=True)
    bg = _unit(scene, "background")
    z_bg = float(scene.nearness_to_z(0.1))
    assert bg.base_depth == pytest.approx(z_bg, rel=1e-3)
    assert bg.depth_strength == pytest.approx(0.0, abs=1e-6)  # flat once occluders are ignored
    zs = {u.name: u.effective_base_depth for u in scene.units}
    assert zs["character"] < zs["roof"] < zs["wall"] < zs["background"]


def test_nearness_mapping_round_trip(doc):
    scene = unitlib.scene_from_document(doc)
    f = scene.reference_camera.focal_px
    assert scene.nearness_to_z(1.0) == pytest.approx(f)
    assert scene.nearness_to_z(0.0) == pytest.approx(scene.depth_far_ratio * f)
    n = np.linspace(0, 1, 11)
    assert np.allclose(scene.z_to_nearness(scene.nearness_to_z(n)), n)


def test_contradicting_depth_map_is_resolved(doc):
    """Depth map says the roof (upper layer) is behind the wall it overlaps."""
    scene = unitlib.scene_from_document(doc)
    n = _layered_depth(doc, {"background": 0.0, "wall": 0.8, "roof": 0.2, "character": 0.5})
    unitlib.set_global_depth(scene, n, "painted", assign=False)
    unitlib.assign_depths_from_map(scene, resolve_order=False)
    conflicts = check_layer_order(scene)
    wall = _unit(scene, "wall")
    assert any(c.lower_id == wall.id for c in conflicts)
    base_before = wall.effective_base_depth
    resolve_layer_order(scene)
    assert check_layer_order(scene) == []
    # the wall is a surface: pushed back locally under the roof, its own depth is kept
    assert wall.effective_base_depth == pytest.approx(base_before)
    assert wall.depth_push is not None
    roof = _unit(scene, "roof")
    rl, rt, rr, rb = roof.bbox
    wl, wt, wr, wb = wall.bbox
    under_roof = np.zeros(wall.alpha.shape, bool)
    under_roof[max(rt - wt, 0) : rb - wt, max(rl - wl, 0) : rr - wl] = True
    far_from_roof = np.zeros(wall.alpha.shape, bool)
    far_from_roof[-10:, :] = True  # bottom of the wall, well away from the roof
    peak = wall.depth_push[under_roof].max()
    assert peak > 0
    assert wall.depth_push[far_from_roof].max() < 0.1 * peak  # only feathered tail far away
    # the character does not overlap the house, so its AI depth is kept
    char = _unit(scene, "character")
    assert char.effective_base_depth == pytest.approx(float(scene.nearness_to_z(0.5)), rel=1e-3)
    assert char.depth_push is None


def test_flat_units_shift_as_a_whole(doc):
    scene = unitlib.scene_from_document(doc)
    n = _layered_depth(doc, {"background": 0.0, "wall": 0.8, "roof": 0.2, "character": 0.5})
    unitlib.set_global_depth(scene, n, "painted", assign=False)
    unitlib.assign_depths_from_map(scene, resolve_order=False)
    wall = _unit(scene, "wall")
    wall.mode = ReconstructionMode.FLAT
    unitlib.refresh_unit(scene, wall)
    resolve_layer_order(scene)
    assert check_layer_order(scene) == []
    assert wall.depth_push is None and np.ptp(wall.mesh.vertices[:, 2]) == 0
    assert wall.effective_base_depth > _unit(scene, "roof").effective_base_depth


def test_edge_inset_ignores_silhouette_bleed(doc):
    """AI depth blurs across silhouettes; the outer band must not pull the edge back."""
    scene = unitlib.scene_from_document(doc)
    n = _layered_depth(doc, {"background": 0.0, "wall": 0.3, "roof": 0.3, "character": 0.9})
    char = _unit(scene, "character")
    l, t, r, b = char.bbox
    band = (char.alpha > 0) & (np.pad(char.alpha > 0, 1)[2:, 1:-1] == 0)  # pixels on the lower edge
    sub = n[t:b, l:r]
    sub[band] = 0.0  # background depth bleeding into the edge
    char.edge_inset = 3
    unitlib.set_global_depth(scene, n, "bleed", assign=True)
    assert char.depth_strength == pytest.approx(0.0, abs=1e-3)


def test_inpaint_is_smooth_and_exact_on_known():
    yy, xx = np.mgrid[0:64, 0:64]
    truth = (xx + yy).astype(np.float32)
    known = np.ones_like(truth, bool)
    known[20:44, 20:44] = False
    out = inpaint(truth, known)
    assert np.array_equal(out[known], truth[known])
    assert np.abs(out[~known] - truth[~known]).max() < 12  # roughly continues the ramp
    assert np.abs(np.diff(out, axis=1)).max() < 4  # no streaks / jumps


def _boundary_edges(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1], counts


@pytest.mark.parametrize("w,h,step", [(97, 61, 4), (128, 128, 2), (33, 70, 3)])
def test_adaptive_mesh_is_watertight(w, h, step):
    """Opaque texture with a sharp relief feature: every single-use edge must lie on the
    texture border (no T-junction cracks), and the mesh must be coarser than uniform."""
    alpha = np.full((h, w), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    relief = np.where(np.hypot(xx - w / 3, yy - h / 2) < min(w, h) / 5, 0.5, -0.5).astype(np.float32)
    g = build_adaptive_grid(alpha, relief, step, 0.05)
    edges, counts = _boundary_edges(g["faces"])
    assert counts.max() == 2  # manifold
    p = g["xy"][edges]
    on_border = ((p[:, :, 0] == 0).all(1) | (p[:, :, 0] == w).all(1) | (p[:, :, 1] == 0).all(1) | (p[:, :, 1] == h).all(1))
    assert on_border.all()
    assert len(g["faces"]) < len(build_grid(alpha, step)["faces"])
    # covers the whole texture
    q = g["xy"][g["faces"]]
    area = 0.5 * np.abs((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))
    assert area.sum() == pytest.approx(w * h)


def test_adaptive_mesh_refines_silhouette(doc):
    scene = unitlib.scene_from_document(doc)
    roof = _unit(scene, "roof")
    roof.adaptive_mesh = True
    roof.mesh_density = 2
    update_unit_mesh(roof, scene.reference_camera)
    xy = roof.grid_cache["xy"]
    # every visible pixel is covered by some triangle cell: compare against uniform cell coverage
    uniform = build_grid(roof.alpha, 2)
    assert len(roof.mesh.triangles) <= len(uniform["faces"])
    # projection round trip holds on adaptive meshes too
    roof.depth_strength = 50.0
    roof.relief = np.random.default_rng(0).normal(0, 1, roof.alpha.shape).astype(np.float32)
    roof.relief_revision += 1
    update_unit_mesh(roof, scene.reference_camera)
    got = scene.reference_camera.project(roof.mesh.vertices.astype(np.float64))
    expected = roof.grid_cache["xy"] + np.array(roof.offset)
    assert np.abs(got - expected).max() < 2e-3
    assert xy.min() >= 0


def test_grid_cache_follows_relief(doc, depth_path):
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), "x")
    bg = _unit(scene, "background")
    before = bg.grid_cache
    bg.depth_smoothing = 10.0
    unitlib.refresh_unit(scene, bg)
    assert bg.grid_cache is not before
    after = bg.grid_cache
    bg.depth_strength *= 2  # strength edits keep the topology
    update_unit_mesh(bg, scene.reference_camera)
    assert bg.grid_cache is after


def test_flat_mode_has_no_relief(doc, depth_path):
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), "x")
    u = scene.units[1]
    u.mode = ReconstructionMode.FLAT
    unitlib.refresh_unit(scene, u)
    assert u.relief is None and np.ptp(u.mesh.vertices[:, 2]) == 0


def test_reload_keeps_settings(psd_path, tmp_path, depth_path):
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    from genparallax.project import reload_document

    doc = PsdDocument(psd_path)
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), "x")
    _unit(scene, "wall").base_depth = 777.0
    unitlib.merge_units(scene, doc, [_unit(scene, "wall"), _unit(scene, "roof")], "house")

    # edit the PSD: recolour the character and add a new layer
    psd = PSDImage.open(psd_path)
    psd.append(PixelLayer.frompil(Image.new("RGBA", (20, 20), (255, 0, 255, 255)), psd, "bird", top=5, left=5))
    psd.save(psd_path)

    new_scene, new_doc, warnings = reload_document(scene, doc, psd_path)
    names = [u.name for u in new_scene.units]
    assert "house" in names and "bird" in names
    house = _unit(new_scene, "house")
    assert house.base_depth == pytest.approx(777.0)
    assert new_scene.global_depth is not None
    assert any("bird" in w for w in warnings)


def test_overlay_stays_flat_until_a_depth_map_exists(haze_psd):
    """Default layer depths are arbitrary planes; an overlay must not be bent across them."""
    doc = PsdDocument(haze_psd, force_composite=True)
    scene = unitlib.scene_from_document(doc)
    haze = _unit(scene, "haze")
    assert haze.conform
    default_base = haze.base_depth
    resolve_layer_order(scene)
    assert haze.relief is None
    assert haze.depth_strength == 0.0
    assert haze.depth_push is None
    assert haze.base_depth == pytest.approx(default_base)
    assert np.ptp(haze.mesh.vertices[:, 2]) == pytest.approx(0.0)


def test_overlay_is_laid_between_neighbours(haze_psd, depth_path):
    from genparallax.reconstruction.constraints import depth_span

    doc = PsdDocument(haze_psd, force_composite=True)
    scene = unitlib.scene_from_document(doc)
    haze, ground, ball = (_unit(scene, n) for n in ("haze", "ground", "ball"))
    assert haze.conform and not ground.conform and not ball.conform
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), "radial", assign=True)
    assert check_layer_order(scene) == []
    # the translucent haze does not push the ground back where the ground is visible
    vis = unitlib.visibility_masks(scene)[ground.id]
    if ground.depth_push is not None:
        assert np.mean(ground.depth_push[vis] > 0.01 * ground.base_depth) < 0.02
    # haze lies in front of the ground wherever the ball does not cover it
    cam = scene.reference_camera
    _, haze_far = depth_span(haze, cam)
    g_near, _ = depth_span(ground, cam)
    hl, ht, hr, hb = haze.bbox
    under = g_near[ht:hb, hl:hr]
    bl, bt, br, bb = ball.bbox
    free = np.ones(haze.alpha.shape, bool)
    free[max(bt - ht, 0) : bb - ht, max(bl - hl, 0) : br - hl] = False
    sel = free & (haze.alpha >= 8)
    assert np.mean(haze_far[sel] < under[sel]) > 0.99


def test_no_spikes_where_depth_map_bleeds_across_layers(tmp_path):
    """A lattice in front of a far background. The depth map (like an AI estimate) fills
    the lattice's gaps with the lattice's depth and misses its thin bars. Neither layer
    may follow those wrong values."""
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGBA", (W, H))
    psd.append(PixelLayer.frompil(Image.new("RGBA", (W, H), (150, 170, 200, 255)), psd, "far", top=0, left=0))
    lattice = np.zeros((H, 80, 4), np.uint8)
    yy, xx = np.mgrid[0:H, 0:80]
    bars = ((xx % 20) < 8) | ((yy % 30) < 6)  # 12 px gaps between bars
    lattice[bars] = (120, 120, 110, 255)
    psd.append(PixelLayer.frompil(Image.fromarray(lattice, "RGBA"), psd, "lattice", top=0, left=40))
    path = tmp_path / "lattice.psd"
    psd.save(path)

    doc = PsdDocument(path, force_composite=True)
    scene = unitlib.scene_from_document(doc)
    n = np.full((H, W), 0.1, np.float32)
    n[:, 36:124] = 0.8  # the whole lattice area, gaps included, plus a 4 px halo
    n[40:46, 40:120] = 0.1  # one horizontal bar the estimate missed
    unitlib.set_global_depth(scene, n, "blurred", assign=True)
    far, lattice_unit = _unit(scene, "far"), _unit(scene, "lattice")
    z_far, z_near = float(scene.nearness_to_z(0.1)), float(scene.nearness_to_z(0.8))
    far_z = -far.mesh.vertices[:, 2]
    assert far_z.min() > z_far * 0.97  # no spikes towards the lattice (pushes go backwards only)
    lat_z = -lattice_unit.mesh.vertices[:, 2]
    assert lat_z.max() < z_near * 1.05  # the missed bar does not reach back to the background
    assert check_layer_order(scene) == []
