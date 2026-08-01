"""Explicit transforms between PX4 LOCAL_NED and a reconstructed 3DGS scene."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def _rotation_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be right-handed")
    return matrix


def body_to_ned_from_euler(roll, pitch, yaw):
    """Return body-FRD to NED rotation for PX4 roll, pitch and yaw."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    roll_matrix = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    pitch_matrix = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    yaw_matrix = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return yaw_matrix @ pitch_matrix @ roll_matrix


@dataclass(frozen=True)
class Px4SceneAlignment:
    scene_from_ned_rotation: np.ndarray
    scene_from_ned_translation: np.ndarray
    body_from_camera_rotation: np.ndarray
    scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "scene_from_ned_rotation", _rotation_matrix(
            self.scene_from_ned_rotation, "scene_from_ned_rotation"))
        object.__setattr__(self, "body_from_camera_rotation", _rotation_matrix(
            self.body_from_camera_rotation, "body_from_camera_rotation"))
        translation = np.asarray(self.scene_from_ned_translation, dtype=np.float64)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("scene_from_ned_translation must contain 3 finite values")
        object.__setattr__(self, "scene_from_ned_translation", translation)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        transform = config["scene_from_ned"]
        return cls(
            scene_from_ned_rotation=transform["rotation"],
            scene_from_ned_translation=transform["translation_m"],
            body_from_camera_rotation=config["body_frd_from_camera_opencv"],
            scale=float(transform.get("scale", 1.0)),
        )

    def position_scene_from_ned(self, position_ned):
        position = np.asarray(position_ned, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_ned must contain 3 finite values")
        return (
            self.scene_from_ned_translation
            + self.scale * self.scene_from_ned_rotation @ position
        )

    def vector_scene_from_ned(self, vector_ned):
        vector = np.asarray(vector_ned, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("vector_ned must contain 3 finite values")
        return self.scale * self.scene_from_ned_rotation @ vector

    def vector_ned_from_scene(self, vector_scene):
        vector = np.asarray(vector_scene, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("vector_scene must contain 3 finite values")
        return self.scene_from_ned_rotation.T @ vector / self.scale

    def camera_c2w(self, position_ned, roll=0.0, pitch=0.0, yaw=0.0):
        """Return OpenCV optical camera-to-scene matrix from PX4 telemetry."""
        ned_from_body = body_to_ned_from_euler(roll, pitch, yaw)
        scene_from_camera = (
            self.scene_from_ned_rotation
            @ ned_from_body
            @ self.body_from_camera_rotation
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = scene_from_camera
        matrix[:3, 3] = self.position_scene_from_ned(position_ned)
        return matrix
