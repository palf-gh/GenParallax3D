"""Depth worker plumbing (spawned process + cross-thread signal) without downloading a model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


def test_depth_job_returns_cached_result(tmp_path):
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

    from genparallax.app.workers import DepthEstimationJob
    from genparallax.reconstruction.estimator import cache_key

    app = QCoreApplication.instance() or QCoreApplication([])
    rgba = np.zeros((12, 16, 4), np.uint8)
    expected = np.linspace(0, 1, 12 * 16, dtype=np.float32).reshape(12, 16)
    np.save(tmp_path / f"depth_{cache_key(rgba)}.npy", expected)

    job = DepthEstimationJob()
    results, errors = [], []
    loop = QEventLoop()
    job.finished.connect(lambda r: (results.append(r), loop.quit()))
    job.failed.connect(lambda e: (errors.append(e), loop.quit()))
    QTimer.singleShot(60000, loop.quit)
    job.start(rgba, str(tmp_path))
    loop.exec()
    job.shutdown()
    assert not errors, errors
    assert results and np.allclose(results[0], expected)
