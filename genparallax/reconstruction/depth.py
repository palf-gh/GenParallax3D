"""Depth map loading, normalisation and per-unit relief extraction.

Depth maps are handled as *nearness* in ``[0, 1]`` (1 = closest), which is what
Depth Anything V2 produces (relative inverse depth) and what most painted depth
maps use (white = near).

Nearness is first converted to camera depth ``Z`` (linear in ``1/Z``, see
:meth:`Scene.nearness_to_z`). The per-unit relief ``R_i`` is then
``(Z - median) / spread`` over the unit's own visible pixels, positive meaning
*further* from the camera, so the final depth is ``Z_i(u, v) = B_i + S_i * R_i(u, v)``
and ``B_i = median``, ``S_i = spread`` reproduce the depth map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_depth_image(path: str | Path, size: tuple[int, int], white_is_near: bool = True) -> np.ndarray:
    """Load a greyscale (8/16-bit or float) depth image as nearness in ``[0, 1]``."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        img = Image.open(path)
        if img.mode in ("I;16", "I;16B", "I"):
            arr = np.asarray(img, dtype=np.float32)
        else:
            arr = np.asarray(img.convert("F"), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    if arr.shape[::-1] != tuple(size):
        arr = np.asarray(Image.fromarray(arr, mode="F").resize(size, Image.BILINEAR))
    arr = normalise(arr)
    return arr if white_is_near else 1.0 - arr


def normalise(depth: np.ndarray, mask: np.ndarray | None = None, lo_pct: float = 0.0, hi_pct: float = 100.0) -> np.ndarray:
    values = depth[mask] if mask is not None else depth.ravel()
    if values.size == 0:
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def fill_outside_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace values outside ``mask`` by the nearest value inside it."""
    if mask.all() or not mask.any():
        return values
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return values[iy, ix]


def masked_gaussian(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: smooth only using samples inside ``mask``."""
    if sigma <= 0:
        return values
    w = mask.astype(np.float32)
    num = ndimage.gaussian_filter(values * w, sigma, mode="nearest")
    den = ndimage.gaussian_filter(w, sigma, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), values)
    return out.astype(np.float32)


def inpaint(values: np.ndarray, known: np.ndarray, scales: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0)) -> np.ndarray:
    """Fill ``~known`` smoothly from ``known`` with progressively wider normalised convolution.

    Pure nearest-neighbour filling leaves streaks; this keeps holes smooth and only
    falls back to nearest values where no scale reaches.
    """
    if known.all() or not known.any():
        return values.astype(np.float32)
    out = values.astype(np.float64).copy()
    filled = known.copy()
    for sigma in scales:
        if filled.all():
            break
        w = filled.astype(np.float64)
        num, den = _blur_pair(np.where(filled, out, 0.0), w, sigma)
        take = ~filled & (den > 0.02)
        out[take] = num[take] / den[take]
        filled |= take
    if not filled.all():
        out = fill_outside_mask(out, filled)
    # soften the seams between the fill scales; known pixels stay exact
    softened = ndimage.gaussian_filter(out, 2.0, mode="nearest")
    out[~known] = softened[~known]
    return out.astype(np.float32)


def _blur_pair(num: np.ndarray, den: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-blur two fields. Wide blurs run on a block-averaged copy and are
    upsampled again: a sigma of 128 px at full resolution is needlessly slow."""
    factor = max(1, int(sigma // 4))
    if factor == 1 or min(num.shape) < 4 * factor:
        return (ndimage.gaussian_filter(num, sigma, mode="nearest"),
                ndimage.gaussian_filter(den, sigma, mode="nearest"))
    h, w = num.shape
    ph, pw = -(-h // factor) * factor, -(-w // factor) * factor

    def down(a: np.ndarray) -> np.ndarray:
        a = np.pad(a, ((0, ph - h), (0, pw - w)), mode="edge")
        return a.reshape(ph // factor, factor, pw // factor, factor).mean(axis=(1, 3))

    rows = _linear_upsampler(ph // factor, factor, h)
    cols = _linear_upsampler(pw // factor, factor, w)

    def up(a: np.ndarray) -> np.ndarray:
        return rows @ a @ cols.T  # separable bilinear upsampling as two small matrix products

    small_sigma = sigma / factor
    return (up(ndimage.gaussian_filter(down(num), small_sigma, mode="nearest")),
            up(ndimage.gaussian_filter(down(den), small_sigma, mode="nearest")))


def _linear_upsampler(n_small: int, factor: int, n_out: int) -> np.ndarray:
    """(n_out, n_small) matrix that linearly interpolates block averages back to pixels."""
    centres = (np.arange(n_out) + 0.5) / factor - 0.5  # output pixel centres in block units
    centres = np.clip(centres, 0.0, n_small - 1.0)
    lo = np.floor(centres).astype(np.int64)
    hi = np.minimum(lo + 1, n_small - 1)
    frac = centres - lo
    m = np.zeros((n_out, n_small))
    m[np.arange(n_out), lo] += 1.0 - frac
    m[np.arange(n_out), hi] += frac
    return m


def interior(mask: np.ndarray, inset: int) -> np.ndarray:
    """``mask`` eroded by ``inset`` pixels; falls back to ``mask`` if nothing would remain."""
    if inset <= 0 or not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, iterations=int(inset), border_value=1)
    return eroded if eroded.any() else mask


def relief_from_z(z: np.ndarray, known: np.ndarray, smoothing: float = 2.0) -> tuple[np.ndarray, float, float]:
    """Relief ``R`` and statistics from a camera-depth map cropped to one unit.

    ``known`` marks pixels whose depth belongs to this unit (visible in the composite,
    away from the silhouette). Hidden pixels are filled smoothly from them.
    ``R = (Z - median) / spread`` with ``spread`` the 2nd–98th percentile range, so the
    defaults ``B = median`` and ``S = spread`` reproduce the depth map.

    Returns ``(R, median_z, spread_z)``.
    """
    if not known.any():
        return np.zeros(z.shape, dtype=np.float32), float(np.median(z)), 0.0
    zs = masked_gaussian(z.astype(np.float32), known, smoothing)
    inside = zs[known]
    p2, med, p98 = np.percentile(inside, [2, 50, 98])
    spread = float(p98 - p2)
    if spread < 1e-6 * max(abs(med), 1.0):
        return np.zeros(z.shape, dtype=np.float32), float(med), 0.0
    filled = inpaint(zs, known)
    relief = np.clip((filled - med) / spread, -1.5, 1.5)
    return relief.astype(np.float32), float(med), spread


def crop(array: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    l, t, r, b = bbox
    return array[t:b, l:r]
