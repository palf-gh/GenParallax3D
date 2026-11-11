"""Headless GPU check: the reference-camera render must reproduce the PSD composite."""

from __future__ import annotations

import numpy as np
import pytest

from genparallax.document.psd_loader import PsdDocument
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.depth import load_depth_image
from genparallax.rendering.camera_control import OrbitController

from conftest import H, W

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - machine dependent
        pytest.skip(f"No OpenGL 3.3 context available: {exc}")
    yield c
    c.release()


def _over_white(rgba):
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    return rgba[..., :3].astype(np.float64) * a + 255.0 * (1 - a)


@pytest.mark.parametrize("adaptive,simplify", [(False, "off"), (True, "off"), (True, "standard"), (True, "strong")])
@pytest.mark.parametrize("strength_scale", [0.0, 1.0, 3.0])  # 6x drives Z through the near plane
def test_reference_render_matches_composite(ctx, psd_path, depth_path, strength_scale, adaptive, simplify):
    """After resolving the layer order the reference view reproduces the PSD, even with
    strong relief and a depth map that contradicts the stacking order (the radial map
    puts the background centre in front of the foreground layers)."""
    from genparallax.reconstruction.constraints import check_layer_order, resolve_layer_order
    from genparallax.rendering.renderer import SceneRenderer

    doc = PsdDocument(psd_path)
    scene = unitlib.scene_from_document(doc)
    for u in scene.units:
        u.mesh_density = 4
        u.adaptive_mesh = adaptive
        u.mesh_simplify = simplify
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), "test")
    for u in scene.units:
        u.depth_strength *= strength_scale
        unitlib.refresh_unit(scene, u, relief=False)
    resolve_layer_order(scene)
    assert check_layer_order(scene) == []

    r = SceneRenderer(ctx)
    r.sync(scene, doc.composite)
    img = r.read_reference_image(scene)
    diff = np.abs(_over_white(img) - _over_white(doc.composite))
    assert diff.mean() < 1.0, diff.mean()
    assert np.percentile(diff, 99) < 40  # only antialiased silhouette pixels may differ noticeably
    r.release()


def test_free_camera_shows_parallax(ctx, psd_path):
    from genparallax.rendering.renderer import SceneRenderer

    doc = PsdDocument(psd_path)
    scene = unitlib.scene_from_document(doc)
    r = SceneRenderer(ctx)
    r.sync(scene, doc.composite)
    before = r.read_reference_image(scene)
    orbit = OrbitController(scene.preview_camera)
    orbit.reset(scene.reference_camera)
    orbit.orbit(60, 0)
    fbo, _ = r.offscreen((W, H), slot="test")
    r.render_view(scene, fbo, (W, H), "free", scene.preview_camera)
    after = np.frombuffer(fbo.read(components=4), np.uint8).reshape(H, W, 4)[::-1]
    assert np.abs(after[..., :3].astype(int) - before[..., :3].astype(int)).mean() > 5
    picked = r.pick(scene, scene.reference_camera, (W, H), (0, 0, W, H), W / H, 115, 85)
    assert scene.unit_by_id(picked).name == "character"
    r.release()


def test_reference_render_with_overlay(ctx, haze_psd, depth_path):
    """A translucent overlay between two layers still composites like the PSD."""
    from genparallax.rendering.renderer import SceneRenderer

    doc = PsdDocument(haze_psd, force_composite=True)
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), "radial", assign=True)
    r = SceneRenderer(ctx)
    r.sync(scene, doc.composite)
    diff = np.abs(_over_white(r.read_reference_image(scene)) - _over_white(doc.composite))
    assert diff.mean() < 1.0, diff.mean()
    assert np.percentile(diff, 99) < 8
    r.release()
