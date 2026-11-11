from __future__ import annotations

import json

import numpy as np
import pygltflib as gl
import pytest
from PIL import Image

from genparallax.document.psd_loader import PsdDocument
from genparallax.export.ae import write_ae_metadata
from genparallax.export.glb import export_scene, sanitise_filename
from genparallax.project import load_project, save_project
from genparallax.reconstruction import units as unitlib
from genparallax.reconstruction.depth import load_depth_image
from genparallax.reconstruction.meshing import update_unit_mesh
from genparallax.scene import ReconstructionMode

from conftest import H, W


@pytest.fixture
def doc(psd_path):
    return PsdDocument(psd_path)


@pytest.fixture
def scene(doc, depth_path):
    s = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(s, load_depth_image(depth_path, (W, H)), str(depth_path))
    return s


def test_psd_structure(doc):
    assert (doc.width, doc.height) == (W, H)
    names = [i.name for i in doc.leaves()]
    assert names == ["background", "wall", "roof", "character", "hidden"]
    grp = next(i for i in doc.root if i.is_group)
    assert [c.name for c in grp.children] == ["wall", "roof"]
    assert doc.layers[grp.children[0].key].path == "building/wall"
    assert doc.composite.shape == (H, W, 4)


def test_default_units_skip_hidden_and_keep_offsets(doc):
    units = unitlib.build_default_units(doc)
    assert [u.name for u in units] == ["background", "wall", "roof", "character"]
    roof = units[2]
    # antialiased disc is trimmed to its visible pixels
    assert roof.offset[0] >= 25 and roof.offset[1] >= 20
    assert roof.texture[:, :, 3].min() == 0 and roof.texture[:, :, 3].max() == 255


def test_default_depth_order(doc):
    """Without a depth map, layers are spaced by PSD order (with one, the map decides)."""
    scene = unitlib.scene_from_document(doc)
    zs = [u.base_depth for u in sorted(scene.units, key=lambda u: u.order)]
    assert zs == sorted(zs, reverse=True)  # bottom layer furthest
    assert zs[-1] == pytest.approx(scene.reference_camera.focal_px)


@pytest.mark.parametrize("base_scale,strength", [(1.0, 0.0), (1.7, 300.0), (0.4, -150.0)])
def test_projection_round_trip(scene, base_scale, strength):
    """Every vertex must project back onto its source pixel for any B and S."""
    cam = scene.reference_camera
    for unit in scene.units:
        unit.base_depth *= base_scale
        unit.depth_strength = strength
        update_unit_mesh(unit, cam)
        grid = unit.grid_cache
        expected = grid["xy"] + np.array(unit.offset)
        got = cam.project(unit.mesh.vertices.astype(np.float64))
        assert np.abs(got - expected).max() < 2e-3
        if strength:
            assert np.ptp(-unit.mesh.vertices[:, 2]) > 1.0  # relief actually present


def test_farther_flat_layer_is_bigger(scene):
    cam = scene.reference_camera
    unit = scene.units[1]
    unit.mode = ReconstructionMode.FLAT
    update_unit_mesh(unit, cam)
    w1 = np.ptp(unit.mesh.vertices[:, 0])
    unit.base_depth *= 2
    update_unit_mesh(unit, cam)
    assert np.ptp(unit.mesh.vertices[:, 0]) == pytest.approx(2 * w1, rel=1e-5)
    assert np.allclose(unit.mesh.vertices[:, 2], -unit.base_depth)


def test_mesh_faces_camera_and_covers_alpha(scene):
    for unit in scene.units:
        m = unit.mesh
        v = m.vertices.astype(np.float64)
        t = m.triangles
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        centre = v[t].mean(axis=1)
        assert (np.einsum("ij,ij->i", n, -centre) > 0).all()  # front faces toward camera
        assert m.uvs.min() >= 0 and m.uvs.max() <= 1


def test_merge_and_split(scene, doc):
    wall, roof = scene.units[1], scene.units[2]
    merged = unitlib.merge_units(scene, doc, [wall, roof], "building")
    assert merged is not None and len(scene.units) == 3
    assert set(merged.source_layers) == {wall.source_layers[0], roof.source_layers[0]}
    l, t, r, b = merged.bbox
    assert (l, r, b) == (20, 70, 100) and 20 <= t <= 24  # roof disc top after trimming
    parts = unitlib.split_unit(scene, doc, merged)
    assert [p.name for p in parts] == ["wall", "roof"] and len(scene.units) == 4


def test_glb_combined_round_trip(scene, tmp_path):
    out = tmp_path / "scene.glb"
    res = export_scene(scene, out, mode="combined")
    g = gl.GLTF2().load(str(out))
    names = [n.name for n in g.nodes]
    assert {"background", "wall", "roof", "character", "ReferenceCamera", "GenParallaxScene"} <= set(names)
    assert "KHR_materials_unlit" in g.extensionsUsed
    for mat in g.materials:
        assert mat.alphaMode in ("BLEND", "OPAQUE", "MASK") and mat.doubleSided and "KHR_materials_unlit" in mat.extensions
    blob = g.binary_blob()
    for img in g.images:
        view = g.bufferViews[img.bufferView]
        import io
        pil = Image.open(io.BytesIO(blob[view.byteOffset : view.byteOffset + view.byteLength]))
        assert pil.mode == "RGBA"
    # positions: reference plane at z = 0 after shift, scaled
    roof = next(n for n in g.nodes if n.name == "character")
    acc = g.accessors[g.meshes[roof.mesh].primitives[0].attributes.POSITION]
    s = scene.export_settings.scale
    unit = next(u for u in scene.units if u.name == "character")
    f = scene.reference_camera.focal_px
    assert acc.max[2] == pytest.approx((f + float(unit.mesh.vertices[:, 2].max())) * s, rel=1e-5)
    assert acc.min[2] == pytest.approx((f + float(unit.mesh.vertices[:, 2].min())) * s, rel=1e-5)
    assert res.files == [out]


def test_glb_uv_texel_alignment(doc, tmp_path):
    """The UV of the top-left vertex of a unit samples the unit's top-left texel (no V flip)."""
    scene = unitlib.scene_from_document(doc)  # flat planes, so "top-left" is unambiguous
    out = tmp_path / "scene.glb"
    export_scene(scene, out, mode="combined")
    g = gl.GLTF2().load(str(out))
    blob = g.binary_blob()
    node = next(n for n in g.nodes if n.name == "wall")
    prim = g.meshes[node.mesh].primitives[0]

    def read(acc_index, comps):
        acc = g.accessors[acc_index]
        view = g.bufferViews[acc.bufferView]
        data = np.frombuffer(blob, np.float32, acc.count * comps, view.byteOffset)
        return data.reshape(acc.count, comps)

    pos = read(prim.attributes.POSITION, 3)
    uv = read(prim.attributes.TEXCOORD_0, 2)
    top_left = np.lexsort((pos[:, 0], -pos[:, 1]))[0]  # highest Y, then lowest X
    assert np.allclose(uv[top_left], [0.0, 0.0])


def test_glb_separate(scene, tmp_path):
    scene.units[0].name = 'bad:name/背景'
    res = export_scene(scene, tmp_path / "out.glb", mode="separate")
    assert len(res.files) == len(scene.units) + 1
    assert all(p.exists() for p in res.files)
    assert sanitise_filename('bad:name/背景') == "bad_name_背景"
    meta = write_ae_metadata(tmp_path / "ae.json", scene, res)
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["camera"]["zoom_px"] == pytest.approx(scene.reference_camera.focal_px)
    assert len(data["units"]) == len(scene.units)


def test_project_round_trip(scene, doc, tmp_path):
    scene.units[1].base_depth = 1234.5
    scene.units[3].mode = ReconstructionMode.FLAT
    for u in (scene.units[1], scene.units[3]):
        unitlib.refresh_unit(scene, u)
    unitlib.merge_units(scene, doc, [scene.units[1], scene.units[2]], "building")
    from genparallax.reconstruction.constraints import resolve_layer_order

    resolve_layer_order(scene)  # what auto layer order does after edits in the app
    proj = save_project(tmp_path / "p.gp3d", scene, doc)
    scene2, doc2, warnings = load_project(proj)
    assert warnings == []
    assert [u.name for u in scene2.units] == [u.name for u in scene.units]
    assert scene2.units[1].base_depth == pytest.approx(1234.5)
    assert scene2.units[2].mode == ReconstructionMode.FLAT
    assert scene2.global_depth is not None
    for a, b in zip(scene.units, scene2.units):
        assert np.allclose(a.mesh.vertices, b.mesh.vertices)


def test_transparent_texels_bleed_edge_colour(doc):
    roof = unitlib.build_default_units(doc)[2]
    clear = roof.texture[:, :, 3] == 0
    assert clear.any()
    # roof is red (220, 40, 40): transparent texels must carry that colour, not black/white
    assert np.abs(roof.texture[clear][:, :3].astype(int) - [220, 40, 40]).max() < 40


def test_group_opacity_applied_to_children(psd_path, tmp_path):
    from psd_tools import PSDImage

    psd = PSDImage.open(psd_path)
    next(l for l in psd if l.is_group()).opacity = 128
    path = tmp_path / "translucent.psd"
    psd.save(path)
    d = PsdDocument(path, force_composite=True)
    wall = next(u for u in unitlib.build_default_units(d) if u.name == "wall")
    assert abs(int(wall.texture[:, :, 3].max()) - 128) <= 1
    assert any("Translucent group" in w for w in d.warnings)


def test_camera_only_glb_is_loadable(scene, tmp_path):
    res = export_scene(scene, tmp_path / "out.glb", mode="separate")
    for f in res.files:
        g = gl.GLTF2().load(str(f))
        assert g.scenes and g.nodes
    cam = gl.GLTF2().load(str(res.files[-1]))
    assert not cam.buffers and not cam.extensionsUsed and cam.cameras


def test_opaque_units_export_as_opaque(scene, tmp_path):
    export_scene(scene, tmp_path / "s.glb", mode="combined", alpha_mode="blend")
    g = gl.GLTF2().load(str(tmp_path / "s.glb"))
    modes = {m.name: m.alphaMode for m in g.materials}
    assert modes["background"] == "OPAQUE" and modes["roof"] == "BLEND"


def test_split_export_separates_opaque_and_translucent(scene, tmp_path):
    export_scene(scene, tmp_path / "s.glb", mode="combined")  # "split" is the default
    g = gl.GLTF2().load(str(tmp_path / "s.glb"))
    roof_node = next(n for n in g.nodes if n.name == "roof")
    prims = g.meshes[roof_node.mesh].primitives
    modes = [g.materials[p.material].alphaMode for p in prims]
    assert modes == ["MASK", "BLEND"]
    assert g.materials[prims[0].material].alphaCutoff == 0.5
    roof = next(u for u in scene.units if u.name == "roof")
    counts = [g.accessors[p.indices].count // 3 for p in prims]
    assert sum(counts) == len(roof.mesh.triangles) and counts[0] > counts[1] > 0
    # the opaque part only uses fully opaque texels
    from genparallax.export.glb import opaque_triangles
    mask = opaque_triangles(roof.mesh.uvs, roof.mesh.triangles, roof.alpha)
    uv = roof.mesh.uvs[roof.mesh.triangles[mask]].reshape(-1, 2) * np.array(roof.size)
    xs = np.clip(uv[:, 0].astype(int), 0, roof.size[0] - 1)
    ys = np.clip(uv[:, 1].astype(int), 0, roof.size[1] - 1)
    assert roof.alpha[ys, xs].min() >= 252


def test_composition_matches_after_effects_camera_dialog():
    """Numbers from AE's Camera Settings: 1920x1080, 36 mm film, 24 mm lens, horizontal."""
    from genparallax.scene import PX_TO_MM, Composition, Scene

    comp = Composition(1920, 1080, film_size_mm=36.0, focal_length_mm=24.0)
    assert comp.zoom_px == pytest.approx(1280.0)
    assert comp.zoom_px * PX_TO_MM == pytest.approx(451.56, abs=0.01)
    assert comp.measured_px * PX_TO_MM == pytest.approx(677.33, abs=0.01)
    assert comp.angle_of_view_deg == pytest.approx(73.74, abs=0.01)
    comp.angle_of_view_deg = 73.74
    assert comp.focal_length_mm == pytest.approx(24.0, abs=0.01)

    # a 1494x840 image filling the composition: 128.5 %
    comp = Composition(1920, 1080, focal_length_mm=24.0)
    comp.image_scale = comp.fit_scale(1494, 840, cover=True)
    scene = Scene.create(1494, 840, composition=comp)
    f = scene.reference_camera.focal_px
    assert f == pytest.approx(1280.0 / comp.image_scale)
    assert scene.to_ae_z(f) == pytest.approx(0.0)  # the 100 % plane is AE z = 0
    assert scene.to_ae_z(2 * f) == pytest.approx(1280.0)  # twice as far = one zoom behind
    assert scene.from_ae_z(1280.0) == pytest.approx(2 * f)


def test_glb_scale_matches_after_effects_model_import(scene, doc, tmp_path):
    """AE 27.0 imports 1 m as 512 comp px (Model Units "Metres", Object Scale 100 %),
    with the glTF origin at the layer position (comp centre, z = 0): the reference
    plane must come out at the composition size."""
    from genparallax.scene import AE_PX_PER_METRE, units_per_px

    assert AE_PX_PER_METRE == 512.0
    assert scene.export_settings.model_unit == "m"
    assert scene.export_settings.scale == pytest.approx(1.0 / 512.0)
    assert units_per_px("cm") == pytest.approx(100.0 / 512.0)
    out = tmp_path / "scene.glb"
    flat = unitlib.scene_from_document(doc)  # flat planes; the top one lies on the reference plane
    export_scene(flat, out, mode="combined")
    g = gl.GLTF2().load(str(out))
    root = next(n for n in g.nodes if n.name == "GenParallaxScene")
    assert root.extras["genparallax"]["after_effects_model_units"] == "m"
    top = max(flat.units, key=lambda u: u.order)
    node = next(n for n in g.nodes if n.name == top.name)
    acc = g.accessors[g.meshes[node.mesh].primitives[0].attributes.POSITION]
    assert acc.max[2] == pytest.approx(0.0, abs=1e-6) and acc.min[2] == pytest.approx(0.0, abs=1e-6)
    l, t, r, b = top.bbox
    scale = flat.composition.image_scale
    # in AE pixels the plane is as large as the layer is in composition pixels
    assert (acc.max[0] - acc.min[0]) * AE_PX_PER_METRE == pytest.approx((r - l) * scale, abs=1.5 * scale)
    assert (acc.max[1] - acc.min[1]) * AE_PX_PER_METRE == pytest.approx((b - t) * scale, abs=1.5 * scale)
    # and sits where it is in the image, measured from the comp centre (AE +Y is down)
    cx = (acc.max[0] + acc.min[0]) / 2 * AE_PX_PER_METRE
    cy = -(acc.max[1] + acc.min[1]) / 2 * AE_PX_PER_METRE
    assert cx == pytest.approx(((l + r) / 2 - flat.width / 2) * scale, abs=1.5 * scale)
    assert cy == pytest.approx(((t + b) / 2 - flat.height / 2) * scale, abs=1.5 * scale)


def test_legacy_project_export_scale_is_migrated(scene, doc, tmp_path):
    proj = save_project(tmp_path / "p.gp3d", scene, doc)
    data = json.loads(proj.read_text(encoding="utf-8"))
    data["export"].pop("model_unit")
    data["export"]["scale"] = 0.01  # the old free scale, 5.12x too large in AE
    proj.write_text(json.dumps(data), encoding="utf-8")
    scene2, _, _ = load_project(proj)
    assert scene2.export_settings.model_unit == "m"
    assert scene2.export_settings.scale == pytest.approx(1.0 / 512.0)

