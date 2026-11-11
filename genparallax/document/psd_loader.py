"""PSD/PSB parsing and layer image extraction (psd-tools).

The original PSD hierarchy is kept read-only in :class:`PsdDocument`. Reconstruction
units reference layers by *key* (``"id:<layer_id>"``), so merging or splitting
units never modifies the document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage

from genparallax.document import psd_compat  # noqa: F401  (patches psd-tools on import)

log = logging.getLogger(__name__)

#: Layer kinds that can be rasterised and turned into geometry.
RASTER_KINDS = {"pixel", "shape", "type", "smartobject"}
#: Layer kinds that only modify what is below them and cannot become geometry.
ADJUSTMENT_KINDS = {
    "adjustment", "brightnesscontrast", "curves", "exposure", "levels", "vibrance",
    "huesaturation", "colorbalance", "blackandwhite", "photofilter", "channelmixer",
    "colorlookup", "invert", "posterize", "threshold", "selectivecolor", "gradientmap",
}
FILL_KINDS = {"solidcolorfill", "patternfill", "gradientfill"}


@dataclass
class LayerInfo:
    key: str
    layer_id: int
    name: str
    path: str  # "Group/Sub/Layer" — used to re-match layers when the PSD is edited
    kind: str
    bbox: tuple[int, int, int, int]
    visible: bool
    blend_mode: str
    opacity: int
    clipping: bool
    is_group: bool
    order: int  # global bottom-to-top index among leaves
    parent_key: str | None = None
    children: list["LayerInfo"] = field(default_factory=list)
    clip_base_key: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def reconstructable(self) -> bool:
        return (
            not self.is_group
            and not self.clipping
            and self.kind in RASTER_KINDS | FILL_KINDS
            and self.bbox[2] > self.bbox[0]
            and self.bbox[3] > self.bbox[1]
        )


class PsdDocument:
    def __init__(self, path: str | Path, force_composite: bool = False):
        """``force_composite`` re-renders the merged image with psd-tools instead of
        using the merged image Photoshop embedded in the file."""
        self.path = Path(path)
        self._force_composite = force_composite
        self._psd = PSDImage.open(self.path)
        self.width, self.height = self._psd.size
        self.warnings: list[str] = []
        self.layers: dict[str, LayerInfo] = {}
        self._raw: dict[str, object] = {}
        self.root: list[LayerInfo] = []
        self._leaf_counter = 0
        self.root = self._walk(self._psd, None, "")
        self.composite = self._composite_image()
        self._validate_compatibility()

    # ------------------------------------------------------------------
    @property
    def is_psb(self) -> bool:
        return self._psd.version == 2

    def _walk(self, container, parent_key: str | None, prefix: str) -> list[LayerInfo]:
        infos: list[LayerInfo] = []
        last_base: LayerInfo | None = None
        for layer in container:  # psd-tools iterates bottom to top
            # Photoshop assigns unique layer ids; files written by other tools may not.
            key = f"id:{layer.layer_id}"
            if layer.layer_id < 0 or key in self.layers:
                key = f"idx:{len(self.layers)}"
            path = f"{prefix}/{layer.name}" if prefix else layer.name
            info = LayerInfo(
                key=key,
                layer_id=layer.layer_id,
                name=layer.name,
                path=path,
                kind=layer.kind,
                bbox=tuple(int(c) for c in layer.bbox),
                visible=bool(layer.visible),
                blend_mode=_blend_name(layer),
                opacity=int(layer.opacity),
                clipping=bool(getattr(layer, "clipping_layer", False)),
                is_group=layer.is_group(),
                order=-1,
                parent_key=parent_key,
            )
            if info.clipping and last_base is not None:
                info.clip_base_key = last_base.key
            elif not info.clipping:
                last_base = info
            self.layers[key] = info
            self._raw[key] = layer
            if info.is_group:
                info.children = self._walk(layer, key, path)
            else:
                info.order = self._leaf_counter
                self._leaf_counter += 1
            infos.append(info)
        return infos

    def _composite_image(self) -> np.ndarray:
        try:
            img = self._psd.composite(force=self._force_composite)
        except Exception as exc:  # pragma: no cover - depends on file content
            self.warnings.append(f"Full composite failed ({exc}); using embedded preview.")
            img = self._psd.topil()
        if img is None:
            img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        return np.asarray(img.convert("RGBA")).copy()

    def _validate_compatibility(self) -> None:
        for info in self.iter_layers():
            if info.kind in ADJUSTMENT_KINDS:
                info.warnings.append("Adjustment layer: cannot be reconstructed in 3D (excluded).")
            if not info.is_group and info.blend_mode not in ("normal", "pass_through"):
                info.warnings.append(
                    f"Blend mode '{info.blend_mode}' is rendered as Normal in 3D."
                )
            if info.is_group and info.blend_mode not in ("normal", "pass_through"):
                info.warnings.append(
                    f"Group blend mode '{info.blend_mode}' is ignored; children are reconstructed individually."
                )
            if info.clipping and info.clip_base_key is None:
                info.warnings.append("Clipping layer without a base layer.")
            if info.clipping and info.clip_base_key and self.layers[info.clip_base_key].is_group:
                info.warnings.append("Clipped to a group: not supported yet, the layer is ignored.")
            if info.is_group and info.opacity < 255 and len([c for c in info.children if not c.is_group]) > 1:
                info.warnings.append(
                    "Translucent group: opacity is applied per layer, so overlapping children look lighter than in Photoshop."
                )
            for w in info.warnings:
                self.warnings.append(f"{info.path}: {w}")

    # ------------------------------------------------------------------
    def iter_layers(self, infos: list[LayerInfo] | None = None):
        for info in self.root if infos is None else infos:
            yield info
            if info.children:
                yield from self.iter_layers(info.children)

    def leaves(self) -> list[LayerInfo]:
        return sorted((i for i in self.iter_layers() if not i.is_group), key=lambda i: i.order)

    def effectively_visible(self, key: str) -> bool:
        info = self.layers[key]
        while info is not None:
            if not info.visible:
                return False
            info = self.layers.get(info.parent_key) if info.parent_key else None
        return True

    def clip_layers_of(self, key: str) -> list[LayerInfo]:
        return [i for i in self.iter_layers() if i.clip_base_key == key]

    def find_layer(self, layer_id: int | None = None, path: str | None = None) -> LayerInfo | None:
        if layer_id is not None and layer_id >= 0 and f"id:{layer_id}" in self.layers:
            info = self.layers[f"id:{layer_id}"]
            if path is None or info.path == path or info.name == path.rsplit("/", 1)[-1]:
                return info
        if path is not None:
            return next((i for i in self.iter_layers() if i.path == path), None)
        return None

    # ------------------------------------------------------------------
    def layer_rgba(self, key: str) -> tuple[np.ndarray, int, int] | None:
        """Rasterised layer (including masks, clipping layers and opacity).

        Returns ``(rgba, left, top)`` on the canvas, or ``None`` if empty.
        """
        layer = self._raw[key]
        info = self.layers[key]
        try:
            img = layer.composite(force=True)
        except Exception as exc:  # pragma: no cover - depends on file content
            log.warning("composite() failed for %s: %s", info.path, exc)
            img = layer.topil()
        if img is None:
            return None
        rgba = np.asarray(img.convert("RGBA")).copy()
        left, top = info.bbox[0], info.bbox[1]
        factor = self._ancestor_alpha(info, left, top, rgba.shape[1], rgba.shape[0])
        if factor is not None:
            rgba[:, :, 3] = np.round(rgba[:, :, 3] * factor).astype(np.uint8)
        return _clip_to_canvas(rgba, left, top, self.width, self.height)

    def _ancestor_alpha(self, info: LayerInfo, left: int, top: int, w: int, h: int) -> np.ndarray | float | None:
        """Combined opacity and raster masks of all enclosing groups, over the given canvas rectangle."""
        factor: np.ndarray | float = 1.0
        changed = False
        parent_key = info.parent_key
        while parent_key is not None:
            group = self.layers[parent_key]
            raw = self._raw[parent_key]
            if group.opacity < 255:
                factor = factor * (group.opacity / 255.0)
                changed = True
            mask = getattr(raw, "mask", None)
            if mask is not None and not getattr(mask, "disabled", False):
                m = _mask_values(mask, left, top, w, h)
                if m is not None:
                    factor = factor * m
                    changed = True
            parent_key = group.parent_key
        return factor if changed else None

    def compose_layers(self, keys: list[str]) -> tuple[np.ndarray, int, int] | None:
        """Alpha-composite several layers (bottom to top, Normal blend) into one RGBA image.

        Clipping layers whose base is in ``keys`` are skipped because the base
        composite already includes them.
        """
        infos = [self.layers[k] for k in keys if k in self.layers]
        infos = [i for i in infos if not (i.clipping and i.clip_base_key in keys)]
        infos = [i for i in infos if not i.is_group]
        infos.sort(key=lambda i: i.order)
        parts = [p for p in (self.layer_rgba(i.key) for i in infos) if p is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        left = min(p[1] for p in parts)
        top = min(p[2] for p in parts)
        right = max(p[1] + p[0].shape[1] for p in parts)
        bottom = max(p[2] + p[0].shape[0] for p in parts)
        canvas = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        for rgba, l, t in parts:
            layer_img = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            layer_img.paste(Image.fromarray(rgba, "RGBA"), (l - left, t - top))
            canvas = Image.alpha_composite(canvas, layer_img)
        return np.asarray(canvas).copy(), left, top


def _mask_values(mask, left: int, top: int, w: int, h: int) -> np.ndarray | None:
    try:
        img = mask.topil()
    except Exception:  # pragma: no cover - depends on file content
        return None
    out = np.full((h, w), getattr(mask, "background_color", 255) / 255.0, dtype=np.float32)
    if img is not None:
        m = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        ml, mt = mask.left, mask.top
        x0, y0 = max(ml, left), max(mt, top)
        x1, y1 = min(ml + m.shape[1], left + w), min(mt + m.shape[0], top + h)
        if x1 > x0 and y1 > y0:
            out[y0 - top : y1 - top, x0 - left : x1 - left] = m[y0 - mt : y1 - mt, x0 - ml : x1 - ml]
    return out


def _clip_to_canvas(rgba: np.ndarray, left: int, top: int, width: int, height: int):
    h, w = rgba.shape[:2]
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + w, width), min(top + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    rgba = rgba[y0 - top : y1 - top, x0 - left : x1 - left]
    # trim fully transparent borders
    ys, xs = np.nonzero(rgba[:, :, 3])
    if len(xs) == 0:
        return None
    rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return np.ascontiguousarray(rgba), x0 + int(xs.min()), y0 + int(ys.min())


def _blend_name(layer) -> str:
    mode = getattr(layer, "blend_mode", None)
    if mode is None:
        return "normal"
    name = getattr(mode, "name", str(mode))
    return name.lower()
