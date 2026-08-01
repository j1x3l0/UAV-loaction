"""Offline unit tests for the read-only observation bridge.

No PX4, no ply file, no torch: the renderer is a stub and target/config
resolution is exercised through ``ReadOnlyObservationBridge.resolve_config``.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from integrations.px4_scene_alignment import Px4SceneAlignment
from integrations.read_only_observation_bridge import (
    ReadOnlyObservationBridge,
    load_telemetry,
    make_observation,
)

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "px4_gate_mid_alignment.json",
)

ANCHOR_SCENE = np.array([2.6877755542, 2.4675363013, 1.4877064383])


class _StubRenderer:
    """Returns a fixed depth image; records the requested camera pose."""

    def __init__(self, depth=None):
        self._depth = (
            depth if depth is not None
            else np.full((64, 64, 1), 5.0, dtype=np.float32)
        )
        self.last_c2w = None

    def render(self, camera_pos, camera_c2w=None):
        self.last_c2w = camera_c2w
        return self._depth, np.zeros((64, 64, 3), dtype=np.float32)


def _alignment():
    return Px4SceneAlignment.from_json(CONFIG)


class MakeObservationTest(unittest.TestCase):
    def setUp(self):
        self.alignment = _alignment()
        self.renderer = _StubRenderer()

    def test_anchor_pose_maps_to_known_scene_position(self):
        target_scene = ANCHOR_SCENE
        obs = make_observation(
            self.alignment, self.renderer,
            [0.0, 0.0, -1.0],   # position NED = anchor
            [1.0, 0.0, 0.0],    # velocity NED
            0.0, 0.0, 0.0,      # roll/pitch/yaw
            target_scene,
        )
        position_scene = self.alignment.position_scene_from_ned([0.0, 0.0, -1.0])
        np.testing.assert_allclose(position_scene, ANCHOR_SCENE, atol=1e-8)
        expected_vel_scene = self.alignment.vector_scene_from_ned([1.0, 0.0, 0.0])
        expected_vec = np.concatenate([expected_vel_scene, np.zeros(3)])
        np.testing.assert_allclose(obs["vec"], expected_vec, atol=1e-6)
        self.assertEqual(obs["depth"].shape, (64, 64, 1))
        self.assertEqual(obs["depth"].dtype, np.float32)
        np.testing.assert_allclose(obs["depth"], 5.0)
        # camera c2w must have been passed to the renderer (axis convention)
        self.assertIsNotNone(self.renderer.last_c2w)

    def test_camera_c2w_uses_attitude(self):
        make_observation(
            self.alignment, self.renderer,
            [0.0, 0.0, -1.0], [0.0, 0.0, 0.0],
            np.deg2rad(0), np.deg2rad(0), np.deg2rad(90.0),
            ANCHOR_SCENE,
        )
        expected = self.alignment.camera_c2w([0.0, 0.0, -1.0], 0.0, 0.0, np.deg2rad(90.0))
        np.testing.assert_allclose(self.renderer.last_c2w, expected, atol=1e-9)

    def test_nonfinite_pose_rejected(self):
        with self.assertRaisesRegex(ValueError, "position_ned"):
            make_observation(
                self.alignment, self.renderer,
                [0.0, np.nan, -1.0], [0.0, 0.0, 0.0],
                0.0, 0.0, 0.0, ANCHOR_SCENE)
        with self.assertRaisesRegex(ValueError, "velocity_ned"):
            make_observation(
                self.alignment, self.renderer,
                [0.0, 0.0, -1.0], [np.inf, 0.0, 0.0],
                0.0, 0.0, 0.0, ANCHOR_SCENE)

    def test_bad_renderer_shape_rejected(self):
        self.renderer = _StubRenderer(depth=np.zeros((32, 32, 1), np.float32))
        with self.assertRaisesRegex(ValueError, "depth shape"):
            make_observation(
                self.alignment, self.renderer,
                [0.0, 0.0, -1.0], [0.0, 0.0, 0.0],
                0.0, 0.0, 0.0, ANCHOR_SCENE)


class ResolveConfigTest(unittest.TestCase):
    def test_defaults_target_to_anchor(self):
        config, target_scene = ReadOnlyObservationBridge.resolve_config(CONFIG)
        self.assertEqual(config.width, 64)
        self.assertEqual(config.height, 64)
        self.assertAlmostEqual(config.fx, 97.1433427374)
        self.assertAlmostEqual(config.fy, 97.0647991673)
        np.testing.assert_allclose(target_scene, ANCHOR_SCENE, atol=1e-8)

    def test_strict_target_without_target_block_fails(self):
        with self.assertRaisesRegex(ValueError, "target not configured"):
            ReadOnlyObservationBridge.resolve_config(CONFIG, strict_target=True)

    def test_invalid_target_frame_fails(self):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False) as handle:
            with open(CONFIG, encoding="utf-8") as source:
                raw = json.load(source)
            raw["target"] = {"frame": "body", "position_m": [0, 0, 0]}
            json.dump(raw, handle)
            path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, "target.frame"):
                ReadOnlyObservationBridge.resolve_config(path)
        finally:
            os.unlink(path)


class LoadTelemetryTest(unittest.TestCase):
    def _write(self, samples):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False)
        json.dump({"schema": 1, "samples": samples}, handle)
        handle.close()
        return handle.name

    def test_empty_samples_fail(self):
        path = self._write([])
        try:
            with self.assertRaisesRegex(ValueError, "no samples"):
                load_telemetry(path)
        finally:
            os.unlink(path)

    def test_nonfinite_sample_fails_with_index(self):
        path = self._write([{
            "t_s": 0.0,
            "position_ned": [0, 0, -1],
            "velocity_ned": [np.nan, 0, 0],
            "attitude_rpy_rad": [0, 0, 0],
        }])
        try:
            with self.assertRaisesRegex(ValueError, "sample 0 velocity_ned"):
                load_telemetry(path)
        finally:
            os.unlink(path)

    def test_non_monotonic_timestamps_fail(self):
        path = self._write([
            {"t_s": 0.0, "position_ned": [0, 0, -1],
             "velocity_ned": [0, 0, 0], "attitude_rpy_rad": [0, 0, 0]},
            {"t_s": 2.0, "position_ned": [0, 0, -1],
             "velocity_ned": [0, 0, 0], "attitude_rpy_rad": [0, 0, 0]},
            {"t_s": 1.0, "position_ned": [0, 0, -1],
             "velocity_ned": [0, 0, 0], "attitude_rpy_rad": [0, 0, 0]},
        ])
        try:
            with self.assertRaisesRegex(ValueError, "not monotonic"):
                load_telemetry(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
