"""Smoke test for packaged builds: ``GenParallax3D --self-test [report.txt]``.

A windowed executable has no console (``sys.stdout`` is ``None``), so the result is
reported by the exit code (0 = passed) and, optionally, written to a report file.
It exercises every bundled dependency without user interaction:

* builds a small layered PSD with psd-tools and loads it back,
* creates reconstruction units, applies a depth map and exports a GLB,
* loads Qt's Japanese translations and builds the main window (off screen),
* creates an OpenGL 3.3 context with moderngl and renders the reference view.

Machines without OpenGL 3.3 (for example CI runners) can pass ``--no-gl`` to skip
the last step. ``--gui`` additionally shows the window for a moment, which makes
Qt create the viewport's own GL context.
"""

from __future__ import annotations

import sys
import tempfile
import time
import traceback
from pathlib import Path


def _make_psd(path: Path, w: int = 96, h: int = 64) -> None:
    import numpy as np
    from PIL import Image, ImageDraw
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGBA", (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 4), np.uint8)
    bg[..., 0] = 40 + xx
    bg[..., 1] = 60 + yy
    bg[..., 2] = 140
    bg[..., 3] = 255
    psd.append(PixelLayer.frompil(Image.fromarray(bg, "RGBA"), psd, "background", top=0, left=0))
    disc = Image.new("RGBA", (30, 30), (0, 0, 0, 0))
    ImageDraw.Draw(disc).ellipse([2, 2, 27, 27], fill=(220, 40, 40, 255))
    psd.append(PixelLayer.frompil(disc, psd, "disc", top=16, left=30))
    psd.save(path)


def _check_core(work: Path, log) -> tuple:
    import numpy as np
    import pygltflib
    from PIL import Image

    from genparallax.document.psd_loader import PsdDocument
    from genparallax.export.glb import export_scene
    from genparallax.reconstruction import units as unitlib
    from genparallax.reconstruction.depth import load_depth_image

    psd_path = work / "selftest.psd"
    _make_psd(psd_path)
    doc = PsdDocument(psd_path)
    assert len(doc.leaves()) == 2, "unexpected layer count"
    log(f"psd: {doc.width}x{doc.height}, {len(doc.leaves())} layers")

    yy, xx = np.mgrid[0:doc.height, 0:doc.width]
    r = np.hypot(xx - doc.width / 2, yy - doc.height / 2) / np.hypot(doc.width / 2, doc.height / 2)
    depth_path = work / "depth.png"
    Image.fromarray(((1 - r) * 255).astype(np.uint8), "L").save(depth_path)

    scene = unitlib.scene_from_document(doc)
    unitlib.set_global_depth(scene, load_depth_image(depth_path, (doc.width, doc.height)), "selftest")
    out = work / "selftest.glb"
    export_scene(scene, out, mode="combined")
    gltf = pygltflib.GLTF2().load(str(out))
    assert gltf.meshes, "GLB has no meshes"
    log(f"glb: {out.stat().st_size} bytes, {len(gltf.meshes)} meshes")
    return doc, scene


def _check_gl(doc, scene, log) -> None:
    import moderngl
    import numpy as np

    from genparallax.rendering.renderer import SceneRenderer

    ctx = moderngl.create_standalone_context(require=330)
    try:
        log(f"gl: {ctx.info.get('GL_RENDERER')} / {ctx.info.get('GL_VERSION')}")
        r = SceneRenderer(ctx)
        r.sync(scene, doc.composite)
        img = r.read_reference_image(scene)
        diff = np.abs(img[..., :3].astype(int) - doc.composite[..., :3].astype(int)).mean()
        log(f"gl: reference render mean difference {diff:.2f}")
        assert diff < 8.0, f"reference render differs from the composite ({diff:.2f})"
        r.release()
    finally:
        ctx.release()


def _check_gui(show: bool, log) -> None:
    from PySide6.QtWidgets import QApplication

    from genparallax.app import main_window as mw
    from genparallax.reconstruction.estimator import ai_available

    mw.configure_default_surface_format()
    QApplication.setAttribute(mw.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    from PySide6.QtCore import QLibraryInfo, QTranslator

    translations = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    assert QTranslator(app).load("qtbase_ja", translations), f"qtbase_ja.qm not found in {translations}"
    mw.install_language(app)
    log(f"qt: {app.platformName()}, Japanese Qt translations found")
    win = mw.MainWindow()
    log(f"gui: main window built; AI depth available: {ai_available()}")
    if show:
        win.show()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        log(f"gui: viewport GL context {'ok' if win.viewport.ctx is not None else 'MISSING'}")
        assert win.viewport.ctx is not None, "the viewport did not create an OpenGL context"
    # hide() rather than close(): closeEvent would save this window's geometry over the user's.
    win.hide()
    win.depth_job.shutdown()
    win.order_job.shutdown()
    win.task.shutdown()
    app.processEvents()


def run(args: list[str]) -> int:
    report = next((a for a in args if not a.startswith("--")), None)
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        if sys.stdout is not None:
            print(msg, flush=True)

    from genparallax import __version__

    log(f"GenParallax3D {__version__} self-test (frozen: {getattr(sys, 'frozen', False)})")
    ok = True
    with tempfile.TemporaryDirectory(prefix="gp3d_selftest_") as tmp:
        try:
            doc, scene = _check_core(Path(tmp), log)
            if "--no-gl" not in args:
                _check_gl(doc, scene, log)
            _check_gui("--gui" in args, log)
        except Exception:  # noqa: BLE001 - reported in the result
            ok = False
            log(traceback.format_exc())
    log("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    if report:
        Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if ok else 1
