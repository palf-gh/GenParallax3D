"""After Effects side-car metadata (camera and layer information) as JSON.

After Effects uses pixels with +X right, +Y down and +Z away from the viewer.
The default AE camera for a comp sits at ``(W/2, H/2, -zoom)`` looking at the
comp centre, with ``zoom`` in pixels. The scene's :class:`Composition` holds the
same numbers as AE's Camera Settings dialog; the image lies on AE's ``z = 0``
plane at its scale, and all values written here are composition pixels.

Imported glTF geometry was checked in After Effects 27.0: at Model Units
"Metres" and Object Scale 100 % one metre is 512 composition pixels, the model's
origin sits at the layer position (comp centre, z = 0), glTF +Y is AE up and glTF
-Z is AE +Z. The GLB is scaled so that the unit named in ``model_space`` gives one
AE pixel per composition pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

from genparallax import __version__
from genparallax.document.psd_loader import PsdDocument
from genparallax.export.glb import ExportResult
from genparallax.scene import PX_TO_MM, Scene


def build_ae_metadata(scene: Scene, export: ExportResult | None = None, doc: PsdDocument | None = None,
                      frame_rate: float = 24.0, duration_s: float = 10.0) -> dict:
    comp = scene.composition
    cw, ch, zoom = comp.width, comp.height, comp.zoom_px
    left, top, iw, ih = scene.image_rect_in_composition()
    s = scene.export_settings
    units = []
    for u in sorted(scene.units, key=lambda u: u.order):
        layer_paths = [doc.layers[k].path for k in u.source_layers if doc and k in doc.layers]
        units.append({
            "id": u.id,
            "name": u.name,
            "psd_order": u.order,
            "source_layers": u.source_layers,
            "source_layer_paths": layer_paths,
            "mode": u.mode.value,
            "visible": u.visible,
            # composition pixels, After Effects convention (z = 0 is the 100 % plane)
            "ae_z": scene.to_ae_z(u.effective_base_depth),
            "distance_from_camera_px": u.effective_base_depth * comp.image_scale,
            "depth_strength_px": u.depth_strength * comp.image_scale,
            "bbox_image_px": list(u.bbox),
            "glb": str(export.unit_files[u.id].name) if export and u.id in export.unit_files else None,
        })
    return {
        "generator": f"GenParallax3D {__version__}",
        "composition": {
            "name": Path(doc.path).stem if doc else "GenParallax",
            "width": cw,
            "height": ch,
            "pixel_aspect": 1.0,
            "frame_rate": frame_rate,
            "duration_s": duration_s,
        },
        "image": {
            "size_px": [scene.width, scene.height],
            "scale_percent": comp.image_scale * 100.0,
            "position_in_composition": [left, top],
            "size_in_composition": [iw, ih],
        },
        "camera": {
            "name": "Reference Camera",
            "type": "two_node",
            "zoom_px": zoom,
            "zoom_mm": zoom * PX_TO_MM,
            "focal_length_mm": comp.focal_length_mm,
            "film_size_mm": comp.film_size_mm,
            "measure_film_size": comp.measure,
            "angle_of_view_deg": comp.angle_of_view_deg,
            "position": [cw / 2.0, ch / 2.0, -zoom],
            "point_of_interest": [cw / 2.0, ch / 2.0, 0.0],
        },
        "model_space": {
            "gltf_axes": "+X right, +Y up, camera looks down -Z",
            "units_per_pixel": s.scale,
            "after_effects_model_settings": {
                "model_units": s.model_unit,
                "object_scale_percent": 100.0,
                "up_axis": "Y",
                "flip_z": False,
                "position": [cw / 2.0, ch / 2.0, 0.0],
            },
            "origin": s.origin,
            "to_ae": "ae = (W/2 + x/units_per_pixel, H/2 - y/units_per_pixel, -z/units_per_pixel)"
                     if s.origin == "reference_plane" else
                     "ae = (W/2 + x/units_per_pixel, H/2 - y/units_per_pixel, -z/units_per_pixel - zoom_px)",
            "note": "W, H and pixels are those of the composition.",
            "verified_in_after_effects": "27.0",
        },
        "export": {
            "mode": s.mode,
            "files": [p.name for p in export.files] if export else [],
        },
        "units": units,
    }


def write_ae_metadata(path: str | Path, scene: Scene, export: ExportResult | None = None,
                      doc: PsdDocument | None = None) -> Path:
    path = Path(path)
    data = build_ae_metadata(scene, export, doc)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

