import os
import unittest

import numpy as np

from integrations.px4_scene_alignment import (
    Px4SceneAlignment,
    body_to_ned_from_euler,
)


CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "px4_gate_mid_alignment.json",
)


class Px4SceneAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.alignment = Px4SceneAlignment.from_json(CONFIG)

    def test_anchor_reconstructs_recorded_camera_position(self):
        position = self.alignment.position_scene_from_ned([0.0, 0.0, -1.0])
        np.testing.assert_allclose(
            position, [2.6877755542, 2.4675363013, 1.4877064383], atol=1e-8
        )

    def test_anchor_reconstructs_recorded_camera_rotation(self):
        expected = np.array([
            [-0.4128668189, -0.3027412422, 0.8590044995],
            [-0.9101836218, 0.1715923917, -0.3769904848],
            [-0.0332680689, -0.9374986887, -0.3463949252],
        ])
        np.testing.assert_allclose(
            self.alignment.camera_c2w([0, 0, -1])[:3, :3], expected, atol=1e-8
        )

    def test_vector_round_trip(self):
        vector = np.array([1.2, -0.5, 0.3])
        scene = self.alignment.vector_scene_from_ned(vector)
        np.testing.assert_allclose(
            self.alignment.vector_ned_from_scene(scene), vector, atol=1e-9
        )

    def test_yaw_rotates_body_forward_to_ned_east(self):
        rotation = body_to_ned_from_euler(0.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-9)

    def test_camera_rotation_stays_right_handed(self):
        rotation = self.alignment.camera_c2w(
            [0.2, -0.1, -1.1], roll=0.1, pitch=-0.2, yaw=0.3
        )[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
