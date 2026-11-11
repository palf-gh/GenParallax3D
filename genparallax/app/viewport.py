"""3D viewport: QOpenGLWidget hosting a ModernGL :class:`SceneRenderer`."""

from __future__ import annotations

import time

import moderngl
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from genparallax.app.i18n import tr
from genparallax.rendering.camera_control import OrbitController, parallax_pose
from genparallax.rendering.renderer import REFERENCE_MODES, SceneRenderer
from genparallax.scene import Scene


def configure_default_surface_format() -> None:
    """Must run before QApplication is created."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


class Viewport(QOpenGLWidget):
    unitPicked = Signal(object, bool)  # unit id or None, additive selection
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.ctx: moderngl.Context | None = None
        self.renderer: SceneRenderer | None = None
        self.scene: Scene | None = None
        self.composite = None
        self.mode = "reference"
        self.selected: set[str] = set()
        self.orbit: OrbitController | None = None
        self.parallax_camera = None
        self.parallax_amplitude = 0.05  # fraction of the image width
        self.parallax_period = 4.0  # seconds
        self._parallax_start = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._press_pos: QPoint | None = None
        self._last_pos: QPoint | None = None
        self._dragged = False
        self._last_rect = (0, 0, 1, 1)

    # --------------------------------------------------------------- state
    def set_scene(self, scene: Scene | None, composite=None) -> None:
        self.scene = scene
        self.composite = composite
        if scene is not None:
            self.orbit = OrbitController(scene.preview_camera)
            self.reset_camera()
            self.parallax_camera = scene.preview_camera.copy()
        self.update()

    def reset_camera(self) -> None:
        if self.scene is None or self.orbit is None:
            return
        near, far = self.scene.depth_range
        self.orbit.reset(self.scene.reference_camera, focus_depth=0.5 * (near + far))
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "parallax":
            self._parallax_start = time.perf_counter()
            self._timer.start(16)
        else:
            self._timer.stop()
        self.update()

    def set_selected(self, ids: set[str]) -> None:
        self.selected = set(ids)
        self.update()

    # ------------------------------------------------------------- OpenGL
    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.renderer = SceneRenderer(self.ctx)
        info = self.ctx.info
        self.statusMessage.emit(f"OpenGL {info.get('GL_VERSION', '?')} — {info.get('GL_RENDERER', '?')}")

    def _device_size(self) -> tuple[int, int]:
        dpr = self.devicePixelRatioF()
        return max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))

    def _active_camera(self):
        if self.scene is None:
            return None
        if self.mode in REFERENCE_MODES:
            return self.scene.reference_camera
        if self.mode == "parallax":
            ref = self.scene.reference_camera
            near, far = self.scene.depth_range
            phase = ((time.perf_counter() - self._parallax_start) / self.parallax_period) % 1.0
            parallax_pose(self.parallax_camera, ref, phase, self.parallax_amplitude * self.scene.width,
                          0.5 * (near + far))
            return self.parallax_camera
        return self.scene.preview_camera

    def paintGL(self) -> None:
        if self.ctx is None:
            return
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        size = self._device_size()
        if self.scene is None:
            fbo.use()
            self.ctx.viewport = (0, 0, *size)
            fbo.clear(*self.renderer.background, 1.0)
            return
        self.renderer.sync(self.scene, self.composite)
        self._last_rect = self.renderer.render_view(self.scene, fbo, size, self.mode, self._active_camera(),
                                                    self.selected)

    def release_gl(self) -> None:
        if self.renderer is not None:
            self.makeCurrent()
            self.renderer.release()
            self.doneCurrent()

    # -------------------------------------------------------------- input
    def mousePressEvent(self, event) -> None:
        self._press_pos = self._last_pos = event.position().toPoint()
        self._dragged = False
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None or self.orbit is None:
            return
        pos = event.position().toPoint()
        d = pos - self._last_pos
        self._last_pos = pos
        if (pos - self._press_pos).manhattanLength() > 3:
            self._dragged = True
        if not self._dragged:
            return
        if self.mode in REFERENCE_MODES or self.mode == "parallax":
            self.statusMessage.emit(tr("Switch to Free Camera, Depth or Wireframe to move the camera."))
            return
        buttons = event.buttons()
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if buttons & Qt.MouseButton.MiddleButton or (buttons & Qt.MouseButton.LeftButton and shift):
            self.orbit.pan(d.x(), d.y(), self.height())
        elif buttons & Qt.MouseButton.LeftButton:
            self.orbit.orbit(d.x(), d.y())
        elif buttons & Qt.MouseButton.RightButton:
            self.orbit.dolly(-d.y() * 0.05)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._dragged and event.button() == Qt.MouseButton.LeftButton and self.scene and self.renderer:
            self._pick(event.position().toPoint(), bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))
        self._press_pos = self._last_pos = None

    def _pick(self, pos: QPoint, additive: bool) -> None:
        if self.mode in ("original", "difference", "parallax"):
            return
        dpr = self.devicePixelRatioF()
        x, y = int(pos.x() * dpr), int(pos.y() * dpr)
        size = self._device_size()
        rect = self._last_rect
        if not (rect[0] <= x < rect[0] + rect[2] and size[1] - rect[1] - rect[3] <= y < size[1] - rect[1]):
            self.unitPicked.emit(None, additive)
            return
        camera = self._active_camera()
        aspect = self.scene.reference_camera.aspect if self.mode in REFERENCE_MODES else size[0] / size[1]
        self.makeCurrent()
        uid = self.renderer.pick(self.scene, camera, size, rect, aspect, x, y)
        self.doneCurrent()
        self.unitPicked.emit(uid, additive)

    def wheelEvent(self, event) -> None:
        if self.orbit is None or self.mode in REFERENCE_MODES:
            return
        self.orbit.dolly(event.angleDelta().y() / 120.0)
        self.update()

    def keyPressEvent(self, event) -> None:
        if self.orbit is None:
            return super().keyPressEvent(event)
        k = event.key()
        moves = {Qt.Key.Key_W: (1, 0, 0), Qt.Key.Key_S: (-1, 0, 0), Qt.Key.Key_A: (0, -1, 0),
                 Qt.Key.Key_D: (0, 1, 0), Qt.Key.Key_E: (0, 0, 1), Qt.Key.Key_Q: (0, 0, -1)}
        if k == Qt.Key.Key_F:
            self.reset_camera()
        elif k in moves and self.mode not in REFERENCE_MODES:
            self.orbit.move(*moves[k])
            self.update()
        else:
            super().keyPressEvent(event)
