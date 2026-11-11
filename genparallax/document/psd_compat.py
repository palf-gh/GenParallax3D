"""Compatibility and speed patches for psd-tools, applied on import.

* Pillow 12 removed ``ImageMath.eval``, which psd-tools <= 1.10 calls with a fixed
  expression when un-premultiplying a transparent merged image.
* psd-tools decodes Photoshop's "ZIP with prediction" channels (common in large
  files) with pure-Python loops over every byte: seconds per layer. The prediction is
  a running sum along each row, which NumPy does in milliseconds.

Both patches only replace functions with equivalents that produce identical bytes.
"""

from __future__ import annotations

import numpy as np
from PIL import ImageMath

import psd_tools.compression as _compression

if not hasattr(ImageMath, "eval"):
    ImageMath.eval = lambda expression, **kw: ImageMath.unsafe_eval(expression, **kw)  # type: ignore[attr-defined]


_original_decode_prediction = _compression.decode_prediction


def decode_prediction(data: bytes, w: int, h: int, depth: int) -> bytes:
    """NumPy version of ``psd_tools.compression.decode_prediction`` (same output)."""
    if depth == 8:
        rows = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
        return np.cumsum(rows, axis=1, dtype=np.uint8).tobytes()  # wraps modulo 256
    if depth == 16:
        rows = np.frombuffer(data, dtype=">u2", count=w * h).reshape(h, w).astype(np.uint16)
        return np.cumsum(rows, axis=1, dtype=np.uint16).astype(">u2").tobytes()
    if depth == 32:
        # bytes are delta-coded per row of 4*w, packed as all first bytes, all second bytes …
        rows = np.frombuffer(data, dtype=np.uint8, count=4 * w * h).reshape(h, 4 * w)
        summed = np.cumsum(rows, axis=1, dtype=np.uint8)
        return np.ascontiguousarray(summed.reshape(h, 4, w).transpose(0, 2, 1)).tobytes()
    return _original_decode_prediction(data, w, h, depth)


_compression.decode_prediction = decode_prediction
