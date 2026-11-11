"""The scene cache restores exactly what loading would compute, and only when it may."""

from __future__ import annotations

import json

import numpy as np
import pytest

from genparallax.document.psd_loader import PsdDocument
from genparallax.project import cache_dir_for, load_project, save_project
from genparallax.reconstruction import constraints
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.depth import load_depth_image
from genparallax.scene_cache import CACHE_FILE

from conftest import H, W


@pytest.fixture
def project(psd_path, depth_path, tmp_path):
    doc = PsdDocument(psd_path)
    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (W, H)), str(depth_path))
    scene.units[-1].conform = True  # an overlay, so the overlay placement is cached too
    unitlib.refresh_all(scene)
    return save_project(tmp_path / "p.gp3d", scene, doc)


def _state(scene):
    return {u.id: (u.base_depth, u.depth_strength, u.depth_stats,
                   None if u.relief is None else u.relief.copy(),
                   None if u.depth_push is None else u.depth_push.copy(),
                   None if u.mesh is None else u.mesh.vertices.copy()) for u in scene.units}


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        for x, y in zip(a[k], b[k]):
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                assert x is not None and y is not None and x.shape == y.shape
                assert np.allclose(x, y, rtol=1e-6, atol=1e-4)  # push fields are cached as float32
            else:
                assert x == y


def _count_resolutions(monkeypatch):
    calls = {"overlay": 0, "push": 0}
    place, push = constraints.place_overlay, constraints._push_behind

    def counted_place(*a, **k):
        calls["overlay"] += 1
        return place(*a, **k)

    def counted_push(*a, **k):
        calls["push"] += 1
        return push(*a, **k)

    monkeypatch.setattr(constraints, "place_overlay", counted_place)
    monkeypatch.setattr(constraints, "_push_behind", counted_push)
    return calls


def test_second_load_restores_the_computed_state(project, monkeypatch):
    calls = _count_resolutions(monkeypatch)
    first, _, warnings1 = load_project(project)  # computes and writes the cache
    assert calls["overlay"] > 0 and calls["push"] > 0  # the test exercises both passes
    assert (cache_dir_for(project) / CACHE_FILE).is_file()
    calls.update(overlay=0, push=0)
    second, _, warnings2 = load_project(project)
    assert calls == {"overlay": 0, "push": 0}  # nothing was recomputed
    assert warnings1 == warnings2
    _assert_same(_state(first), _state(second))
    # The restored scene counts as resolved: a layer-order pass has nothing to redo.
    assert constraints.resolve_layer_order(second) == []
    assert calls == {"overlay": 0, "push": 0}
    _assert_same(_state(first), _state(second))


def test_changed_setting_misses_the_cache(project, monkeypatch):
    load_project(project)
    data = json.loads(project.read_text(encoding="utf-8"))
    data["units"][1]["depth_strength"] += 5.0
    project.write_text(json.dumps(data), encoding="utf-8")
    cached, _, _ = load_project(project)  # recomputes, then caches the new inputs
    (cache_dir_for(project) / CACHE_FILE).unlink()
    fresh, _, _ = load_project(project)
    _assert_same(_state(cached), _state(fresh))


def test_corrupt_cache_is_ignored(project):
    reference, _, _ = load_project(project)
    (cache_dir_for(project) / CACHE_FILE).write_bytes(b"not an npz file")
    scene, _, _ = load_project(project)
    _assert_same(_state(reference), _state(scene))


def test_saving_caches_the_current_state(project, monkeypatch):
    scene, doc, _ = load_project(project)
    scene.units[1].depth_offset = 12.0
    unitlib.refresh_unit(scene, scene.units[1])
    constraints.resolve_layer_order(scene)
    save_project(project, scene, doc, scene_cache=True)
    calls = _count_resolutions(monkeypatch)
    reopened, _, _ = load_project(project)
    assert calls == {"overlay": 0, "push": 0}
    _assert_same(_state(scene), _state(reopened))


def test_editing_after_a_cached_load_matches_a_full_load(project):
    load_project(project)  # writes the cache

    def edit(scene):
        unit = scene.units[1]
        unit.depth_offset = 20.0
        unitlib.refresh_unit(scene, unit)
        constraints.resolve_layer_order(scene)
        return _state(scene)

    cached_scene, _, _ = load_project(project)
    from_cache = edit(cached_scene)
    (cache_dir_for(project) / CACHE_FILE).unlink()
    fresh_scene, _, _ = load_project(project)
    _assert_same(from_cache, edit(fresh_scene))
