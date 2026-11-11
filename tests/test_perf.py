"""Speed-related replacements must give exactly the same results as the originals."""

from __future__ import annotations

import numpy as np
import psd_tools.compression as compression
import pytest

from genparallax.document import psd_compat
from genparallax.reconstruction.depth import _blur_pair
from genparallax.reconstruction.meshing import (
    _rasterise_depth_direct, build_adaptive_grid, build_grid, rasterise_depth,
)


@pytest.mark.parametrize("depth,bpp", [(8, 1), (16, 2), (32, 4)])
@pytest.mark.parametrize("w,h", [(7, 5), (33, 4), (1, 3), (128, 64)])
def test_prediction_decoder_matches_psd_tools(depth, bpp, w, h):
    data = np.random.default_rng(w * h + depth).integers(0, 256, w * h * bpp, dtype=np.uint8).tobytes()
    assert psd_compat.decode_prediction(data, w, h, depth) == bytes(
        psd_compat._original_decode_prediction(data, w, h, depth))
    assert compression.decode_prediction is psd_compat.decode_prediction


@pytest.mark.parametrize("adaptive", [False, True])
def test_coverage_rasteriser_matches_direct(adaptive):
    rng = np.random.default_rng(3)
    alpha = np.zeros((61, 87), np.uint8)
    yy, xx = np.mgrid[0:61, 0:87]
    alpha[np.hypot(xx - 40, yy - 30) < 27] = 255
    relief = rng.normal(0, 1, alpha.shape).astype(np.float32)
    g = build_adaptive_grid(alpha, relief, 2, 0.2) if adaptive else build_grid(alpha, 3)
    z = 500 + 80 * rng.random(len(g["xy"]))
    faces = g["faces"].astype(np.int64)
    a = rasterise_depth(g["xy"], faces, z, alpha.shape)
    b = _rasterise_depth_direct(g["xy"], faces, z, alpha.shape)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) < 1e-9


def test_pyramid_blur_close_to_full_resolution():
    from scipy import ndimage

    rng = np.random.default_rng(5)
    num = ndimage.gaussian_filter(rng.random((300, 410)), 6)
    den = np.ones_like(num)
    for sigma in (8.0, 32.0):
        fast, _ = _blur_pair(num, den, sigma)
        exact = ndimage.gaussian_filter(num, sigma, mode="nearest")
        assert np.abs(fast - exact).max() < 0.02 * np.ptp(num) + 1e-6


def test_snapshot_resolution_matches_direct(psd_path, depth_path):
    from genparallax.document.psd_loader import PsdDocument
    from genparallax.reconstruction import units as unitlib
    from genparallax.reconstruction.constraints import (
        apply_resolution, check_layer_order, resolve_in_snapshot, resolve_layer_order, snapshot_for_resolution,
    )
    from genparallax.reconstruction.depth import load_depth_image
    from genparallax.reconstruction.meshing import update_unit_mesh

    from conftest import H, W

    def scene():
        s = unitlib.scene_from_document(PsdDocument(psd_path))
        unitlib.set_global_depth(s, load_depth_image(depth_path, (W, H)), "x", assign=True)
        for u in s.units:  # an edit that breaks the order again
            u.depth_strength *= 2.5
            update_unit_mesh(u, s.reference_camera)
        return s

    direct, threaded = scene(), scene()
    resolve_layer_order(direct)
    before = {u.id: u.mesh.vertices.copy() for u in threaded.units}
    snap = snapshot_for_resolution(threaded)
    remaining = resolve_in_snapshot(snap)
    assert remaining == []
    for u in threaded.units:  # the live scene is untouched until applied
        assert np.array_equal(u.mesh.vertices, before[u.id])
    apply_resolution(threaded, snap)
    assert check_layer_order(threaded) == []
    for a, b in zip(direct.units, threaded.units):
        assert np.allclose(a.mesh.vertices, b.mesh.vertices)


def test_unchanged_units_are_not_recomputed(psd_path, depth_path):
    from genparallax.document.psd_loader import PsdDocument
    from genparallax.reconstruction import units as unitlib
    from genparallax.reconstruction.constraints import check_layer_order, resolve_layer_order
    from genparallax.reconstruction.depth import load_depth_image
    from genparallax.reconstruction.meshing import update_unit_mesh

    from conftest import H, W

    s = unitlib.scene_from_document(PsdDocument(psd_path))
    unitlib.set_global_depth(s, load_depth_image(depth_path, (W, H)), "x", assign=True)
    revisions = {u.id: u.revision for u in s.units}
    resolve_layer_order(s)  # nothing changed: nothing is redone
    assert {u.id: u.revision for u in s.units} == revisions
    character = next(u for u in s.units if u.name == "character")  # top layer, overlaps only the background
    character.base_depth *= 0.9
    update_unit_mesh(character, s.reference_camera)
    revisions = {u.id: u.revision for u in s.units}
    resolve_layer_order(s)
    untouched = [u for u in s.units if u.name in ("wall", "roof")]  # do not overlap the character
    assert all(u.revision == revisions[u.id] for u in untouched)
    assert check_layer_order(s) == []


def test_every_ui_string_has_a_japanese_translation():
    """All literal tr("…") strings in the app, plus enum/menu labels, are in the dictionary."""
    import ast
    from pathlib import Path

    from genparallax.app import i18n
    from genparallax.app.main_window import ALPHA_MODES, DISPLAY_MODES
    from genparallax.scene import ReconstructionMode

    missing = []
    for source in Path(i18n.__file__).parent.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "tr" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value not in i18n.JA:
                    missing.append(f"{source.name}: {arg.value!r}")
    labels = [m.label for m in ReconstructionMode] + [l for l, _ in DISPLAY_MODES] + [l for l, _ in ALPHA_MODES]
    labels += ["combined", "separate"]
    missing += [l for l in labels if l not in i18n.JA]
    assert missing == []


def test_library_messages_are_translated():
    from genparallax.app import i18n

    i18n.set_language("ja")
    assert i18n.tr_message("a/b: Blend mode 'multiply' is rendered as Normal in 3D.") == \
        "a/b：描画モード「multiply」は3Dでは「通常」として描画されます。"
    assert i18n.tr_message("'遠景' was moved back 12.5 px to keep the PSD layer order.") == \
        "PSDの重なり順を保つため「遠景」を 12.5 px 奥へ移動しました。"
    assert i18n.tr_message("something unexpected") == "something unexpected"
    i18n.set_language("en")
    assert i18n.tr("Layer order OK") == "Layer order OK"
    i18n.set_language("ja")
