import unittest

from scripts.validate_camera_registration import scaled_camera


class CameraScalingTest(unittest.TestCase):
    def setUp(self):
        self.transforms = {
            "w": 720,
            "h": 1280,
            "fl_x": 1092.8626057956928,
            "fl_y": 1091.978990632419,
            "cx": 360.0,
            "cy": 640.0,
        }

    def test_policy_center_crop_intrinsics(self):
        camera = scaled_camera(
            self.transforms, 64, center_crop_square=True
        )
        self.assertEqual((camera["width"], camera["height"]), (64, 64))
        self.assertEqual(camera["crop_box"], [0.0, 280.0, 720.0, 1000.0])
        self.assertAlmostEqual(camera["fx"], 97.14334273739492)
        self.assertAlmostEqual(camera["fy"], 97.06479916732614)
        self.assertAlmostEqual(camera["cx"], 32.0)
        self.assertAlmostEqual(camera["cy"], 32.0)

    def test_full_frame_preserves_aspect_ratio(self):
        camera = scaled_camera(self.transforms, 128)
        self.assertEqual((camera["width"], camera["height"]), (72, 128))
        self.assertIsNone(camera["crop_box"])


if __name__ == "__main__":
    unittest.main()
