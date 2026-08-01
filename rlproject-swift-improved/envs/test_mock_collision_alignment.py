"""Regression test: mock visual geometry must match collision geometry."""

import unittest

import numpy as np

from envs.visual_drone_env import VisualDroneEnv


class MockCollisionAlignmentTest(unittest.TestCase):
    def test_render_spheres_match_collision_spheres(self):
        env = VisualDroneEnv(config={"renderer": "mock"})
        expected = np.column_stack([
            env.obstacles,
            np.full(len(env.obstacles), env.obstacle_radius),
        ]).astype(np.float32)
        np.testing.assert_allclose(env._base_obstacles_for_render, expected)
        np.testing.assert_allclose(env.renderer.obstacles, expected)
        env.close()


if __name__ == "__main__":
    unittest.main()
