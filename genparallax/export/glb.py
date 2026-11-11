"""GLB (glTF 2.0 binary) export written directly with pygltflib.

pygltflib is used instead of trimesh so we control node names, the
``KHR_materials_unlit`` extension, alpha blending and the reference camera.

Exported coordinates: glTF axes (+Y up, camera looks down -Z). With
``origin == "reference_plane"`` the scene is shifted so the reference plane
(Z = f) is at z = 0 and the camera is at ``(0, 0, f * scale)``.

``scale`` follows After Effects' Model Settings: with the default
``ExportSettings.model_unit`` ("m") a composition pixel is 1/512 m, which AE
imports at Model Units "Metres", Object Scale 100 % as exactly one pixel (see
:data:`genparallax.scene.AE_PX_PER_METRE`). AE puts the glTF origin at the layer
position, so with the default origin the model lines up with AE's default camera.
"""

from __future__ import annotations

import io
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygltflib as gl
from PIL import Image

from genparallax.scene import ReconstructionUnit, Scene

UNLIT = "KHR_materials_unlit"


@dataclass
class ExportResult:
    files: list[Path]
    unit_files: dict[str, Path]  # unit id -> file


def origin_shift(scene: Scene, origin: str) -> np.ndarray:
    if origin == "reference_plane":
        return np.array([0.0, 0.0, scene.reference_camera.focal_px])
    return np.zeros(3)


def export_scene(scene: Scene, path: str | Path, mode: str | None = None, scale: float | None = None,
                 origin: str | None = None, include_camera: bool | None = None,
                 alpha_mode: str | None = None) -> ExportResult:
    s = scene.export_settings
    mode = mode or s.mode
    alpha_mode = alpha_mode or getattr(s, "alpha_mode", "split")
    scale = s.scale if scale is None else scale
    origin = origin or s.origin
    include_camera = s.include_camera if include_camera is None else include_camera
    path = Path(path)
    units = [u for u in scene.active_units() if u.visible]

    if mode == "combined":
        write_glb(scene, units, path, scale, origin, include_camera, alpha_mode)
        return ExportResult([path], {u.id: path for u in units})

    out_dir = path.parent / path.stem if path.suffix else path
    out_dir.mkdir(parents=True, exist_ok=True)
    files, unit_files, used = [], {}, set()
    for i, unit in enumerate(sorted(units, key=lambda u: u.order)):
        name = _unique(f"{i:02d}_{sanitise_filename(unit.name)}", used)
        f = out_dir / f"{name}.glb"
        write_glb(scene, [unit], f, scale, origin, include_camera=False, alpha_mode=alpha_mode)
        files.append(f)
        unit_files[unit.id] = f
    if include_camera:
        cam_file = out_dir / "_camera.glb"
        write_glb(scene, [], cam_file, scale, origin, include_camera=True)
        files.append(cam_file)
    return ExportResult(files, unit_files)


def write_glb(scene: Scene, units: list[ReconstructionUnit], path: str | Path, scale: float | None = None,
              origin: str = "reference_plane", include_camera: bool = True, alpha_mode: str = "split") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = _GltfBuilder()
    shift = origin_shift(scene, origin)
    comp = scene.composition
    if scale is None:
        scale = scene.export_settings.scale
    ae_unit = scene.export_settings.model_unit if scale == scene.export_settings.scale else None
    scale = scale * comp.image_scale  # ``scale`` is per composition pixel; vertices are image pixels
    child_nodes = []

    for unit in sorted(units, key=lambda u: u.order):
        mesh = unit.mesh
        if mesh is None or mesh.is_empty:
            continue
        positions = ((mesh.vertices.astype(np.float64) + shift) * scale).astype(np.float32)
        if mesh.normals is None:
            mesh.compute_normals()
        opaque = None
        if alpha_mode == "split" and not unit.conform:  # overlays stay entirely blended
            opaque = opaque_triangles(mesh.uvs, mesh.triangles, unit.alpha)
        mesh_index = builder.add_mesh(unit.name, positions, mesh.normals, mesh.uvs, mesh.triangles, unit.texture,
                                      opaque)
        node = gl.Node(name=unit.name, mesh=mesh_index, extras={
            "genparallax": {
                "unit_id": unit.id,
                "source_layers": unit.source_layers,
                "psd_order": unit.order,
                "base_depth_px": unit.effective_base_depth,
                "depth_strength_px": unit.depth_strength,
                "mode": unit.mode.value,
            }
        })
        child_nodes.append(builder.add_node(node))

    if include_camera:
        ref = scene.reference_camera
        cam = gl.Camera(
            name="ReferenceCamera",
            type="perspective",
            perspective=gl.Perspective(
                yfov=math.radians(comp.fov_y_deg),  # the composition camera, as in After Effects
                aspectRatio=comp.aspect,
                znear=max(ref.near * scale, 1e-4),
                zfar=ref.far * scale,
            ),
        )
        builder.gltf.cameras.append(cam)
        cam_pos = (ref.position + shift) * scale
        child_nodes.append(builder.add_node(gl.Node(
            name="ReferenceCamera", camera=0, translation=[float(c) for c in cam_pos],
            extras={"zoom_px": comp.zoom_px, "composition_size": [comp.width, comp.height]},
        )))

    root = builder.add_node(gl.Node(name="GenParallaxScene", children=child_nodes, extras={
        "genparallax": {
            "image_size": [scene.width, scene.height],
            "composition_size": [comp.width, comp.height],
            "image_scale": comp.image_scale,
            "zoom_px": comp.zoom_px,
            "focal_length_mm": comp.focal_length_mm,
            "film_size_mm": comp.film_size_mm,
            "units_per_composition_pixel": scale / comp.image_scale,
            # After Effects Model Settings that import one composition pixel as one AE pixel
            "after_effects_model_units": ae_unit,
            "after_effects_object_scale_percent": 100.0 if ae_unit else None,
            "origin": origin,
            "colour_management": scene.colour_management,
        }
    }))
    builder.gltf.scenes = [gl.Scene(name="Scene", nodes=[root])]
    builder.gltf.scene = 0
    builder.finish(path)
    return path


class _GltfBuilder:
    def __init__(self):
        self.gltf = gl.GLTF2(asset=gl.Asset(generator="GenParallax3D", version="2.0"))
        self.gltf.extensionsUsed = [UNLIT]
        self.gltf.samplers = [gl.Sampler(magFilter=gl.LINEAR, minFilter=gl.LINEAR,
                                         wrapS=gl.CLAMP_TO_EDGE, wrapT=gl.CLAMP_TO_EDGE)]
        self.blob = bytearray()

    def _add_view(self, data: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        self.blob.extend(data)
        self.gltf.bufferViews.append(gl.BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return len(self.gltf.bufferViews) - 1

    def _add_accessor(self, array: np.ndarray, kind: str, component: int, target: int, minmax: bool = False) -> int:
        view = self._add_view(array.tobytes(), target)
        acc = gl.Accessor(bufferView=view, componentType=component, count=len(array), type=kind)
        if minmax:
            acc.min = [float(x) for x in array.min(axis=0)]
            acc.max = [float(x) for x in array.max(axis=0)]
        self.gltf.accessors.append(acc)
        return len(self.gltf.accessors) - 1

    def _add_material(self, name: str, texture_index: int, alpha_mode: str) -> int:
        self.gltf.materials.append(gl.Material(
            name=name,
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                baseColorTexture=gl.TextureInfo(index=texture_index),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0,
            ),
            alphaMode=alpha_mode,
            alphaCutoff=0.5 if alpha_mode == gl.MASK else None,
            doubleSided=True,
            extensions={UNLIT: {}},
        ))
        return len(self.gltf.materials) - 1

    def add_mesh(self, name, positions, normals, uvs, triangles, texture, opaque: np.ndarray | None = None) -> int:
        """One mesh per unit. With ``opaque`` (a mask over the triangles) the opaque part
        becomes an alpha-tested primitive and the rest a blended one, sharing vertices."""
        pos = self._add_accessor(np.ascontiguousarray(positions, np.float32), gl.VEC3, gl.FLOAT, gl.ARRAY_BUFFER, True)
        nrm = self._add_accessor(np.ascontiguousarray(normals, np.float32), gl.VEC3, gl.FLOAT, gl.ARRAY_BUFFER)
        tex = self._add_accessor(np.ascontiguousarray(uvs, np.float32), gl.VEC2, gl.FLOAT, gl.ARRAY_BUFFER)

        png = io.BytesIO()
        Image.fromarray(texture, "RGBA").save(png, format="PNG", optimize=False)
        img_view = self._add_view(png.getvalue())
        self.gltf.images.append(gl.Image(bufferView=img_view, mimeType="image/png", name=name))
        self.gltf.textures.append(gl.Texture(sampler=0, source=len(self.gltf.images) - 1, name=name))
        texture_index = len(self.gltf.textures) - 1

        if int(texture[:, :, 3].min()) == 255:
            parts = [(triangles, gl.OPAQUE, name)]
        elif opaque is None:
            parts = [(triangles, gl.BLEND, name)]
        else:
            parts = [(triangles[opaque], gl.MASK, f"{name} (opaque)"),
                     (triangles[~opaque], gl.BLEND, f"{name} (translucent)")]
        primitives = []
        for tris, alpha_mode, material_name in parts:
            if len(tris) == 0:
                continue
            idx = self._add_accessor(np.ascontiguousarray(tris, np.uint32).ravel(), gl.SCALAR, gl.UNSIGNED_INT,
                                     gl.ELEMENT_ARRAY_BUFFER)
            primitives.append(gl.Primitive(
                attributes=gl.Attributes(POSITION=pos, NORMAL=nrm, TEXCOORD_0=tex), indices=idx,
                material=self._add_material(material_name, texture_index, alpha_mode), mode=gl.TRIANGLES,
            ))
        self.gltf.meshes.append(gl.Mesh(name=name, primitives=primitives))
        return len(self.gltf.meshes) - 1

    def add_node(self, node: gl.Node) -> int:
        self.gltf.nodes.append(node)
        return len(self.gltf.nodes) - 1

    def finish(self, path: Path) -> None:
        while len(self.blob) % 4:
            self.blob.append(0)
        if self.blob:
            self.gltf.buffers = [gl.Buffer(byteLength=len(self.blob))]
            self.gltf.set_binary_blob(bytes(self.blob))
            self.gltf.save_binary(str(path))
            return
        # Camera-only file: pygltflib would add an empty (invalid) buffer, so write a
        # JSON-only GLB container ourselves.
        self.gltf.extensionsUsed = []
        self.gltf.samplers = []
        payload = self.gltf.to_json(separators=(",", ":")).encode("utf-8")
        payload += b" " * (-len(payload) % 4)
        header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(payload))
        path.write_bytes(header + struct.pack("<I4s", len(payload), b"JSON") + payload)


#: Texels at or above this alpha count as opaque when splitting a unit.
OPAQUE_TEXEL = 252


def opaque_triangles(uvs: np.ndarray, triangles: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Triangles whose texels (plus a one-texel margin for filtering) are all opaque."""
    h, w = alpha.shape
    clear = np.zeros((h + 1, w + 1), dtype=np.int64)  # summed-area table of non-opaque texels
    clear[1:, 1:] = np.cumsum(np.cumsum(alpha < OPAQUE_TEXEL, axis=0), axis=1)
    p = uvs[triangles] * np.array([w, h])  # (M, 3, 2) texture pixels
    x0 = np.clip(np.floor(p[:, :, 0].min(1)).astype(np.int64) - 1, 0, w)
    x1 = np.clip(np.ceil(p[:, :, 0].max(1)).astype(np.int64) + 1, 0, w)
    y0 = np.clip(np.floor(p[:, :, 1].min(1)).astype(np.int64) - 1, 0, h)
    y1 = np.clip(np.ceil(p[:, :, 1].max(1)).astype(np.int64) + 1, 0, h)
    count = clear[y1, x1] - clear[y0, x1] - clear[y1, x0] + clear[y0, x0]
    return count == 0


_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sanitise_filename(name: str) -> str:
    cleaned = _INVALID.sub("_", name).strip().rstrip(".")
    if not cleaned:
        cleaned = "unit"
    if cleaned.upper() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:80]


def _unique(name: str, used: set[str]) -> str:
    candidate, n = name, 2
    while candidate.lower() in used:
        candidate = f"{name}_{n}"
        n += 1
    used.add(candidate.lower())
    return candidate
