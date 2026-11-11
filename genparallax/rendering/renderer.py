"""ModernGL scene renderer, independent of Qt.

Used by the Qt viewport (with the widget's framebuffer) and by headless tests
(with a standalone context). Textures are uploaded as plain RGBA8 and the
framebuffer is not sRGB, so unlit colours pass through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np

from genparallax.scene import Camera, ReconstructionUnit, Scene

MODES = ("original", "reference", "free", "depth", "difference", "depth_difference", "wireframe", "parallax")
#: Modes that are always seen through the reference camera.
REFERENCE_MODES = {"original", "reference", "difference", "depth_difference"}
#: Modes drawn inside the composition frame, as After Effects would show them.
FRAMED_MODES = REFERENCE_MODES | {"parallax"}
#: Texels at or above this alpha are drawn in the opaque (depth-writing) pass.
OPAQUE_ALPHA = 0.996

UNIT_VS = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_mv;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
out float v_depth;
void main() {
    v_uv = in_uv;
    v_depth = -(u_mv * vec4(in_pos, 1.0)).z;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

UNIT_FS = """
#version 330
uniform sampler2D u_tex;
uniform int u_mode;          // 0 texture, 1 depth, 2 flat colour, 3 id, 4 nearness
uniform vec2 u_inv_z;        // (1/Z_far, 1/Z_near - 1/Z_far) of the depth-map mapping
uniform vec3 u_colour;
uniform float u_tint;
uniform vec2 u_depth_range;
uniform vec2 u_alpha_range;  // texel alpha drawn in this pass: [min, max)
in vec2 v_uv;
in float v_depth;
out vec4 f_colour;
void main() {
    vec4 c = texture(u_tex, v_uv);
    if (u_mode >= 3) {
        if (c.a < 0.5) discard;
    } else if (c.a < u_alpha_range.x || c.a >= u_alpha_range.y) {
        discard;
    }
    if (u_mode == 0) {
        f_colour = vec4(mix(c.rgb, vec3(1.0, 0.55, 0.1), u_tint), c.a);
    } else if (u_mode == 4) {
        f_colour = vec4(clamp((1.0 / v_depth - u_inv_z.x) / u_inv_z.y, 0.0, 1.0), 0.0, 0.0, 1.0);
    } else if (u_mode == 1) {
        float d = clamp((v_depth - u_depth_range.x) / max(u_depth_range.y - u_depth_range.x, 1e-3), 0.0, 1.0);
        f_colour = vec4(vec3(1.0 - d), c.a);
    } else {
        f_colour = vec4(u_colour, 1.0);
    }
}
"""

QUAD_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

QUAD_FS = """
#version 330
uniform sampler2D u_image;      // top row first (uploaded from numpy)
uniform sampler2D u_rendered;   // OpenGL framebuffer (bottom row first)
uniform sampler2D u_depth;      // depth-map nearness, top row first
uniform int u_mode;             // 0 image over background, 1 difference, 2 depth difference
uniform vec3 u_background;
uniform float u_gain;
in vec2 v_uv;
out vec4 f_colour;
vec3 over_bg(vec4 c) { return c.rgb * c.a + u_background * (1.0 - c.a); }
void main() {
    vec4 img = texture(u_image, vec2(v_uv.x, 1.0 - v_uv.y));
    if (u_mode == 0) {
        f_colour = vec4(over_bg(img), 1.0);
    } else if (u_mode == 2) {
        vec4 r = texture(u_rendered, v_uv);
        float ai = texture(u_depth, vec2(v_uv.x, 1.0 - v_uv.y)).r;
        vec3 grey = vec3(0.25 + 0.5 * ai);
        if (r.a < 0.5) {
            f_colour = vec4(grey * 0.4, 1.0);   // no geometry here
        } else {
            float d = clamp((r.r - ai) * u_gain, -1.0, 1.0);  // > 0: mesh nearer than the map
            vec3 c = d > 0.0 ? mix(grey, vec3(1.0, 0.25, 0.15), d) : mix(grey, vec3(0.2, 0.45, 1.0), -d);
            f_colour = vec4(c, 1.0);
        }
    } else {
        vec4 r = texture(u_rendered, v_uv);
        vec3 d = abs(over_bg(img) - over_bg(r)) * u_gain;
        f_colour = vec4(clamp(d, 0.0, 1.0), 1.0);
    }
}
"""


@dataclass
class _GpuUnit:
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    vao: moderngl.VertexArray
    texture: moderngl.Texture
    revision: int
    texture_src: int
    faces_src: int
    n_indices: int
    centroid: np.ndarray

    def release(self) -> None:
        for r in (self.vao, self.vbo, self.ibo, self.texture):
            r.release()


def fit_rect(width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
    """Largest centred rectangle of ``aspect`` inside ``width x height``."""
    if width / max(height, 1) > aspect:
        w = int(round(height * aspect))
        return (width - w) // 2, 0, w, height
    h = int(round(width / aspect))
    return 0, (height - h) // 2, width, h


class SceneRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.unit_prog = ctx.program(vertex_shader=UNIT_VS, fragment_shader=UNIT_FS)
        self.quad_prog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=QUAD_FS)
        self.quad_vao = ctx.vertex_array(self.quad_prog, [])
        self.units: dict[str, _GpuUnit] = {}
        self.composite_tex: moderngl.Texture | None = None
        self._composite_src: int | None = None
        self.depth_tex: moderngl.Texture | None = None
        self._depth_src: int | None = None
        self.depth_difference_gain = 4.0
        self._offscreen: dict[str, tuple[moderngl.Framebuffer, moderngl.Texture, tuple[int, int]]] = {}
        self.background = (0.16, 0.16, 0.17)
        self.image_background = (1.0, 1.0, 1.0)
        self.difference_gain = 4.0
        self.depth_range = (1.0, 2.0)

    # ------------------------------------------------------------------ sync
    def sync(self, scene: Scene, composite: np.ndarray | None) -> None:
        if composite is not None and id(composite) != self._composite_src:
            if self.composite_tex is not None:
                self.composite_tex.release()
            self.composite_tex = self._make_texture(composite)
            self._composite_src = id(composite)

        if id(scene.global_depth) != self._depth_src:
            if self.depth_tex is not None:
                self.depth_tex.release()
                self.depth_tex = None
            if scene.global_depth is not None:
                d = np.ascontiguousarray(scene.global_depth, dtype=np.float32)
                self.depth_tex = self.ctx.texture(d.shape[::-1], 1, d.tobytes(), dtype="f4")
                self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._depth_src = id(scene.global_depth)
        zn, zf = scene.depth_map_range
        self.inv_z = (1.0 / zf, 1.0 / zn - 1.0 / zf)

        alive = set()
        zmin, zmax = np.inf, -np.inf
        for unit in scene.units:
            if unit.mesh is None or unit.mesh.is_empty:
                continue
            alive.add(unit.id)
            self._sync_unit(unit)
            z = -unit.mesh.vertices[:, 2]
            zmin, zmax = min(zmin, float(z.min())), max(zmax, float(z.max()))
        for uid in list(self.units):
            if uid not in alive:
                self.units.pop(uid).release()
        if np.isfinite(zmin):
            pad = max((zmax - zmin) * 0.02, 1.0)
            self.depth_range = (zmin - pad, zmax + pad)

    def _sync_unit(self, unit: ReconstructionUnit) -> None:
        mesh = unit.mesh
        gpu = self.units.get(unit.id)
        if gpu is not None and gpu.revision == unit.revision and gpu.texture_src == id(unit.texture):
            return
        data = np.hstack([mesh.vertices.astype("f4"), mesh.uvs.astype("f4")]).tobytes()
        if gpu is not None and gpu.texture_src == id(unit.texture) and gpu.faces_src == id(mesh.triangles) \
                and gpu.vbo.size == len(data):
            gpu.vbo.write(data)  # parameter edit: only vertex positions changed
        else:
            if gpu is not None:
                gpu.release()
            vbo = self.ctx.buffer(data)
            ibo = self.ctx.buffer(mesh.triangles.astype("u4").tobytes())
            vao = self.ctx.vertex_array(self.unit_prog, [(vbo, "3f 2f", "in_pos", "in_uv")], index_buffer=ibo,
                                        index_element_size=4)
            gpu = _GpuUnit(vbo, ibo, vao, self._make_texture(unit.texture), 0, id(unit.texture),
                           id(mesh.triangles), mesh.triangles.size, np.zeros(3))
            self.units[unit.id] = gpu
        gpu.revision = unit.revision
        gpu.centroid = mesh.vertices.mean(axis=0).astype(np.float64)

    def _make_texture(self, rgba: np.ndarray) -> moderngl.Texture:
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, np.ascontiguousarray(rgba, dtype=np.uint8).tobytes(), alignment=1)
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        try:
            tex.anisotropy = 8.0
        except Exception:  # pragma: no cover - driver dependent
            pass
        return tex

    def release(self) -> None:
        for gpu in self.units.values():
            gpu.release()
        self.units.clear()
        if self.composite_tex is not None:
            self.composite_tex.release()
            self.composite_tex = None
        if self.depth_tex is not None:
            self.depth_tex.release()
            self.depth_tex = None
        for fbo, tex, _ in self._offscreen.values():
            fbo.release()
            tex.release()
        self._offscreen.clear()

    # ---------------------------------------------------------------- render
    def draw_units(self, scene: Scene, camera: Camera, aspect: float, mode: str = "texture",
                   selected: set[str] | None = None, id_colours: dict[str, tuple] | None = None) -> None:
        view = camera.view_matrix()
        proj = camera.projection_matrix(aspect)
        mvp = (proj @ view).astype("f4")
        self.unit_prog["u_mvp"].write(mvp.T.tobytes())
        self.unit_prog["u_mv"].write(view.astype("f4").T.tobytes())
        self.unit_prog["u_depth_range"].value = self.depth_range
        self.unit_prog["u_tex"].value = 0
        code = {"texture": 0, "depth": 1, "wireframe": 2, "id": 3, "nearness": 4}[mode]
        self.unit_prog["u_mode"].value = code
        self.unit_prog["u_inv_z"].value = getattr(self, "inv_z", (0.0, 1.0))

        # PSD order (bottom to top). Where layers overlap on the canvas the layer-order
        # constraints make it the true back-to-front order, which per-object depth
        # sorting cannot guarantee for large, interleaved surfaces.
        visible = sorted((u for u in scene.units if u.visible and u.id in self.units), key=lambda u: u.order)

        ctx = self.ctx
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.wireframe = mode == "wireframe"

        def draw_all(alpha_range: tuple[float, float]) -> None:
            self.unit_prog["u_alpha_range"].value = alpha_range
            for unit in visible:
                gpu = self.units[unit.id]
                gpu.texture.use(0)
                self.unit_prog["u_tint"].value = 0.18 if selected and unit.id in selected else 0.0
                if mode == "wireframe":
                    self.unit_prog["u_colour"].value = (1.0, 0.6, 0.15) if selected and unit.id in selected \
                        else (0.75, 0.85, 1.0)
                elif mode == "id" and id_colours:
                    self.unit_prog["u_colour"].value = id_colours[unit.id]
                gpu.vao.render(moderngl.TRIANGLES)

        if mode in ("texture", "depth"):
            # 1) opaque texels with depth writes, 2) translucent texels blended on top
            #    without depth writes, so they never hide what lies behind them.
            ctx.disable(moderngl.BLEND)
            draw_all((OPAQUE_ALPHA, 2.0))
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            fbo = ctx.fbo
            fbo.depth_mask = False
            draw_all((0.004, OPAQUE_ALPHA))
            fbo.depth_mask = True
        else:
            ctx.disable(moderngl.BLEND)
            draw_all((0.0, 2.0))
        ctx.wireframe = False
        ctx.disable(moderngl.BLEND)

    def draw_image(self, mode: int = 0, rendered: moderngl.Texture | None = None) -> None:
        if self.composite_tex is None:
            return
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.composite_tex.use(0)
        self.quad_prog["u_image"].value = 0
        self.quad_prog["u_rendered"].value = 1
        self.quad_prog["u_depth"].value = 2
        if rendered is not None:
            rendered.use(1)
        if self.depth_tex is not None:
            self.depth_tex.use(2)
        self.quad_prog["u_mode"].value = mode
        self.quad_prog["u_background"].value = self.image_background
        self.quad_prog["u_gain"].value = self.depth_difference_gain if mode == 2 else self.difference_gain
        self.quad_vao.render(moderngl.TRIANGLES, vertices=3)

    def offscreen(self, size: tuple[int, int], slot: str = "reference") -> tuple[moderngl.Framebuffer, moderngl.Texture]:
        cached = self._offscreen.get(slot)
        if cached is None or cached[2] != size:
            if cached is not None:
                cached[0].release()
                cached[1].release()
            tex = self.ctx.texture(size, 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            depth = self.ctx.depth_renderbuffer(size)
            fbo = self.ctx.framebuffer(color_attachments=[tex], depth_attachment=depth)
            self._offscreen[slot] = (fbo, tex, size)
        return self._offscreen[slot][0], self._offscreen[slot][1]

    def render_reference(self, scene: Scene, max_size: int = 2048, mode: str = "texture") -> moderngl.Texture:
        """Render the scene through the reference camera into an offscreen texture (transparent background).

        ``mode="nearness"`` writes the depth-map nearness of the mesh instead of colour."""
        scale = min(1.0, max_size / max(scene.width, scene.height))
        size = (max(1, int(scene.width * scale)), max(1, int(scene.height * scale)))
        fbo, tex = self.offscreen(size)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.draw_units(scene, scene.reference_camera, scene.reference_camera.aspect, mode)
        return tex

    def read_reference_image(self, scene: Scene) -> np.ndarray:
        """Reference render as an (H, W, 4) uint8 array, top row first."""
        self.render_reference(scene, max_size=max(scene.width, scene.height))
        fbo, _, (w, h) = self._offscreen["reference"]
        data = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
        return data.reshape(h, w, 4)[::-1].copy()

    def render_view(self, scene: Scene, target: moderngl.Framebuffer, size: tuple[int, int], mode: str,
                    camera: Camera, selected: set[str] | None = None) -> tuple[int, int, int, int]:
        """Render one display mode into ``target``. Returns the image rectangle used."""
        width, height = size
        frame = None
        if mode in FRAMED_MODES:
            # The composition frame fitted into the view, and the image placed in it at
            # its scale. The image may be smaller than the frame or stick out of it.
            comp = scene.composition
            frame = fit_rect(width, height, comp.aspect)
            k = frame[2] / comp.width  # framebuffer pixels per composition pixel
            iw, ih = scene.width * comp.image_scale * k, scene.height * comp.image_scale * k
            rect = (int(round(frame[0] + (frame[2] - iw) / 2)), int(round(frame[1] + (frame[3] - ih) / 2)),
                    max(1, int(round(iw))), max(1, int(round(ih))))
            aspect = scene.reference_camera.aspect
        else:
            rect = (0, 0, width, height)
            aspect = width / max(height, 1)

        rendered = None
        if mode == "difference":
            rendered = self.render_reference(scene)
        elif mode == "depth_difference" and self.depth_tex is not None:
            rendered = self.render_reference(scene, mode="nearness")

        target.use()
        self.ctx.viewport = (0, 0, width, height)
        target.clear(*self.background, 1.0, depth=1.0)
        if frame is not None:  # everything outside the composition is cut off
            self.ctx.scissor = frame
            target.clear(*self.image_background, 1.0, depth=1.0)
        self.ctx.viewport = rect
        if mode == "original":
            self.draw_image(0)
        elif mode == "difference":
            self.draw_image(1, rendered)
        elif mode == "depth_difference":
            if rendered is not None:
                self.draw_image(2, rendered)
            else:
                self.draw_image(0)  # no depth map: show the original
        else:
            draw_mode = {"depth": "depth", "wireframe": "wireframe"}.get(mode, "texture")
            self.draw_units(scene, camera, aspect, draw_mode, selected)
        self.ctx.scissor = None
        self.ctx.viewport = (0, 0, width, height)
        return rect

    def pick(self, scene: Scene, camera: Camera, size: tuple[int, int], rect: tuple[int, int, int, int],
             aspect: float, x: int, y: int) -> str | None:
        """Unit id under the pixel ``(x, y)`` (top-left origin, framebuffer pixels)."""
        ids = {u.id: i + 1 for i, u in enumerate(scene.units)}
        colours = {uid: (((n >> 16) & 255) / 255.0, ((n >> 8) & 255) / 255.0, (n & 255) / 255.0) for uid, n in ids.items()}
        fbo, _ = self.offscreen(size, slot="pick")
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.ctx.viewport = rect
        self.draw_units(scene, camera, aspect, "id", id_colours=colours)
        gl_y = size[1] - 1 - y
        px = fbo.read(viewport=(x, gl_y, 1, 1), components=3)
        n = (px[0] << 16) | (px[1] << 8) | px[2]
        self.ctx.viewport = (0, 0, *size)
        return next((uid for uid, k in ids.items() if k == n), None)
