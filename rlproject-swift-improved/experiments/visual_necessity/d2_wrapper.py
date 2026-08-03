"""D2 wrapper: fixed goal at a scene feature, target-direction vector hidden.

Wraps a VisualDroneEnv so the goal is pinned to a fixed scene point (the
gate opening) and the target-direction components of the observation vector
are zeroed. The policy must navigate using only the depth image + velocity
(+ dense distance reward). If a policy trained this way succeeds with depth
but fails when depth is removed, visual necessity is induced.

Isolated experiment: does not modify envs/ or scripts/. Delete this
directory to revert.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class D2FixedGoalWrapper(gym.Wrapper):
    """Pin the goal and hide the target-direction vector from the policy."""

    def __init__(self, env, target_scene):
        super().__init__(env)
        self.target_scene = np.asarray(target_scene, dtype=np.float32)
        if self.target_scene.shape != (3,) or not np.all(np.isfinite(self.target_scene)):
            raise ValueError("target_scene must contain three finite values")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.target_pos = self.target_scene.copy()
        return self._hide_target(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._hide_target(obs), reward, terminated, truncated, info

    def _hide_target(self, obs):
        obs = dict(obs)
        obs["vec"] = obs["vec"].copy()
        obs["vec"][3:6] = 0.0
        return obs
