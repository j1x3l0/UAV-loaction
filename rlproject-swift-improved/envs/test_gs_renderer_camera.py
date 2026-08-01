import unittest

import numpy as np

from envs.gs_renderer import GSplatRenderer


class GSRendererCameraTest(unittest.TestCase):
    def test_device_choice_is_validated_before_loading(self):
        with self.assertRaisesRegex(ValueError, "device"):
            GSplatRenderer("missing.ply", device="metal")

    def test_explicit_intrinsics_are_preserved(self):
        renderer = GSplatRenderer.__new__(GSplatRenderer)
        renderer.width = 64
        renderer.height = 32
        renderer.fov = 90.0
        renderer.fx = 50.0
        renderer.fy = 51.0
        renderer.cx = 31.0
        renderer.cy = 15.0
        self.assertEqual((renderer.fx, renderer.fy), (50.0, 51.0))
        self.assertEqual((renderer.cx, renderer.cy), (31.0, 15.0))

    def test_accepts_valid_camera_matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(GSplatRenderer._validate_c2w(matrix), matrix)

    def test_rejects_reflection_camera_matrix(self):
        matrix = np.eye(4)
        matrix[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "right-handed"):
            GSplatRenderer._validate_c2w(matrix)

    def test_rejects_non_orthonormal_camera_matrix(self):
        matrix = np.eye(4)
        matrix[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            GSplatRenderer._validate_c2w(matrix)


if __name__ == "__main__":
    unittest.main()
