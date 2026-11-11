"""Monocular depth estimation with Depth Anything V2 Small (Phase 3, experimental).

Runs in a separate worker process (see :mod:`genparallax.app.workers`), so this
module must stay importable without Qt and must expose picklable top-level
functions. ``torch`` and ``transformers`` are imported lazily.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
AI_MODULES = ("torch", "transformers")


def ai_available() -> bool:
    """Whether the optional ``[ai]`` extra is installed.

    Checked without importing it: torch takes seconds to load. The standalone builds
    never bundle it, so there this is always ``False``.
    """
    from importlib.util import find_spec

    try:
        return all(find_spec(name) is not None for name in AI_MODULES)
    except (ImportError, ValueError):
        return False


def cache_key(rgba: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(rgba).tobytes())
    h.update(str(rgba.shape).encode())
    h.update(MODEL_ID.encode())
    return h.hexdigest()[:20]


def estimate_nearness(rgba: np.ndarray, cache_dir: str | None = None, device: str | None = None) -> np.ndarray:
    """Return nearness in ``[0, 1]`` (1 = near) at the input resolution."""
    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"depth_{cache_key(rgba)}.npy"
        if cache_file.exists():
            return np.load(cache_file)

    _hide_broken_torchvision()
    import torch
    from transformers import AutoModelForDepthEstimation

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    pixel_values = torch.from_numpy(preprocess(_flatten_on_white(rgba))).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(pixel_values=pixel_values).predicted_depth  # relative inverse depth, larger = nearer
    predicted = torch.nn.functional.interpolate(
        predicted.unsqueeze(1), size=rgba.shape[:2], mode="bicubic", align_corners=False
    ).squeeze().float().cpu().numpy()

    lo, hi = np.percentile(predicted, [0.5, 99.5])
    nearness = np.clip((predicted - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, nearness)
    return nearness


INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(rgb: np.ndarray, size: int = INPUT_SIZE, multiple: int = 14) -> np.ndarray:
    """Depth Anything V2 input: short side >= ``size``, aspect kept, both sides a multiple
    of 14, bicubic, ImageNet normalisation. Returns (3, H, W) float32.

    Done here rather than with transformers' image processor, which needs torchvision.
    """
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = size / min(h, w)

    def fit(x: float) -> int:
        return max(multiple, int(np.ceil(x / multiple - 1e-6)) * multiple)

    new_w, new_h = fit(w * scale), fit(h * scale)
    img = Image.fromarray(rgb, "RGB").resize((new_w, new_h), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _hide_broken_torchvision() -> None:
    """transformers imports torchvision whenever it is installed. A torchvision built
    for a different torch fails ("operator torchvision::nms does not exist") and
    breaks every model import. Depth Anything does not need torchvision, so in that
    case hide it from this (worker) process."""
    import sys

    if "torchvision" in sys.modules:
        return
    try:
        import torchvision.ops  # noqa: F401
    except Exception:  # noqa: BLE001 - any failure means it is unusable
        for name in [m for m in sys.modules if m == "torchvision" or m.startswith("torchvision.")]:
            del sys.modules[name]
        sys.modules["torchvision"] = None  # find_spec() now reports it as missing


def _flatten_on_white(rgba: np.ndarray) -> np.ndarray:
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)
    return rgb.round().astype(np.uint8)
