"""Background work: AI depth estimation (separate process) and the layer-order pass (thread)."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal

from genparallax.reconstruction.estimator import estimate_nearness


class DepthEstimationJob(QObject):
    """Runs Depth Anything V2 in a worker process so the GUI stays responsive."""

    finished = Signal(object)  # numpy nearness map
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._executor: ProcessPoolExecutor | None = None
        self._future: Future | None = None

    @property
    def running(self) -> bool:
        return self._future is not None and not self._future.done()

    def start(self, rgba, cache_dir: str | None) -> None:
        if self.running:
            return
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        self._future = self._executor.submit(estimate_nearness, rgba, cache_dir)
        # Called from an executor thread; Qt queues the signal onto the GUI thread.
        self._future.add_done_callback(self._done)

    def _done(self, future: Future) -> None:
        try:
            self.finished.emit(future.result())
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


class LayerOrderJob(QObject):
    """Runs the layer-order pass on a scene snapshot in a worker thread.

    NumPy/SciPy release the GIL for the heavy array work, so the GUI keeps responding.
    ``finished`` carries ``(token, snapshot, conflicts)``; the caller decides whether the
    scene is still unchanged (``token``) before applying the snapshot.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="layer-order")
        self._future: Future | None = None

    @property
    def running(self) -> bool:
        return self._future is not None and not self._future.done()

    def start(self, scene, token) -> bool:
        from genparallax.reconstruction.constraints import resolve_in_snapshot, snapshot_for_resolution

        if self.running:
            return False
        snapshot = snapshot_for_resolution(scene)

        def run():
            return token, snapshot, resolve_in_snapshot(snapshot)

        self._future = self._executor.submit(run)
        self._future.add_done_callback(self._done)
        return True

    def _done(self, future: Future) -> None:
        try:
            self.finished.emit(future.result())
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class BackgroundTask(QObject):
    """Runs one-off heavy work (opening files, applying depth, exporting) in a thread.

    ``finished`` carries ``(token, result)`` and ``failed`` ``(token, message, traceback)``;
    the token tells the caller which request a result belongs to.
    """

    finished = Signal(object, object)
    failed = Signal(object, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="background")
        self._future: Future | None = None

    @property
    def running(self) -> bool:
        return self._future is not None and not self._future.done()

    def start(self, fn, token) -> bool:
        if self.running:
            return False
        self._future = self._executor.submit(fn)
        self._future.add_done_callback(lambda f: self._done(f, token))
        return True

    def _done(self, future: Future, token) -> None:
        import traceback

        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(token, f"{type(exc).__name__}: {exc}",
                             "".join(traceback.format_exception(exc, limit=4)))
            return
        self.finished.emit(token, result)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
