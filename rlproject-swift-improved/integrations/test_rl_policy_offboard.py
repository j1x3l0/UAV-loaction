import unittest
from types import SimpleNamespace

import numpy as np

from integrations.rl_policy_offboard import (
    PolicySmokeSafety,
    assert_within_envelope,
    make_smoke_observation,
    ned_to_enu,
)


class PolicyOffboardSmokeTest(unittest.TestCase):
    def test_ned_to_enu(self):
        np.testing.assert_allclose(ned_to_enu([2.0, 3.0, -4.0]), [3.0, 2.0, 4.0])

    def test_observation_shape_and_live_velocity(self):
        message = SimpleNamespace(vx=2.0, vy=3.0, vz=-4.0)
        observation = make_smoke_observation(message, depth_m=5.0)
        self.assertEqual(observation["depth"].shape, (64, 64, 1))
        np.testing.assert_allclose(observation["depth"], 5.0)
        np.testing.assert_allclose(observation["vec"], [3, 2, 4, 1, 0, 0])

    def test_envelope_accepts_nominal_hover(self):
        message = SimpleNamespace(x=0.2, y=-0.1, z=-1.0)
        assert_within_envelope(message, PolicySmokeSafety())

    def test_envelope_rejects_horizontal_escape(self):
        message = SimpleNamespace(x=1.1, y=0.0, z=-1.0)
        with self.assertRaisesRegex(RuntimeError, "horizontal"):
            assert_within_envelope(message, PolicySmokeSafety())

    def test_envelope_rejects_altitude_escape(self):
        message = SimpleNamespace(x=0.0, y=0.0, z=-2.0)
        with self.assertRaisesRegex(RuntimeError, "altitude"):
            assert_within_envelope(message, PolicySmokeSafety())

    def test_duration_is_bounded(self):
        with self.assertRaises(ValueError):
            PolicySmokeSafety(duration_s=10.1)


if __name__ == "__main__":
    unittest.main()
