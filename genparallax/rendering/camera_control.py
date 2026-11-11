"""Orbit / pan / dolly controller for the free preview camera, plus parallax motion."""

from __future__ import annotations

import math

import numpy as np

from genparallax.scene import Camera


class OrbitController:
    def __init__(self, camera: Camera):
        self.camera = camera
        self.target = np.array([0.0, 0.0, -camera.focal_px])
        self.distance = camera.focal_px
        self.yaw = 0.0  # degrees, around +Y
        self.pitch = 0.0  # degrees, around camera right axis
        self.home_distance = self.distance

    def reset(self, reference: Camera, focus_depth: float | None = None) -> None:
        d = focus_depth or reference.focal_px
        self.camera.fov_y_deg = reference.fov_y_deg
        self.camera.near, self.camera.far = reference.near, reference.far
        self.target = np.array([0.0, 0.0, -d])
        self.distance = d
        self.home_distance = d
        self.yaw = self.pitch = 0.0
        self.apply()

    def _offset(self) -> np.ndarray:
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return self.distance * np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fwd = -self._offset() / max(self.distance, 1e-9)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        return fwd, right, up

    def apply(self) -> None:
        self.camera.position = self.target + self._offset()
        self.camera.target = self.target.copy()
        self.camera.up = np.array([0.0, 1.0, 0.0])

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.25
        self.pitch = float(np.clip(self.pitch + dy * 0.25, -89.0, 89.0))
        self.apply()

    def pan(self, dx: float, dy: float, viewport_height: int) -> None:
        _, right, up = self._basis()
        scale = 2.0 * self.distance * math.tan(math.radians(self.camera.fov_y_deg) / 2.0) / max(viewport_height, 1)
        self.target = self.target - right * dx * scale + up * dy * scale
        self.apply()

    def dolly(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.9 ** steps), self.home_distance * 0.02, self.home_distance * 50))
        self.apply()

    def move(self, forward: float, strafe: float, lift: float = 0.0) -> None:
        """Truck/dolly the whole rig (target moves with the camera)."""
        fwd, right, up = self._basis()
        step = self.home_distance * 0.02
        self.target = self.target + (fwd * forward + right * strafe + up * lift) * step
        self.apply()


def parallax_pose(camera: Camera, reference: Camera, phase: float, amplitude: float, focus_depth: float) -> None:
    """Move the camera on a figure-of-eight around the reference position, looking at the focus plane.

    ``phase`` in ``[0, 1)``; ``amplitude`` in pixels at the reference plane.
    """
    a = 2.0 * math.pi * phase
    camera.position = np.array([amplitude * math.sin(a), 0.35 * amplitude * math.sin(2 * a), 0.0])
    camera.target = np.array([0.0, 0.0, -focus_depth])
    camera.up = np.array([0.0, 1.0, 0.0])
    camera.fov_y_deg = reference.fov_y_deg
    camera.near, camera.far = reference.near, reference.far
