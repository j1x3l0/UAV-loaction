"""Regression tests for the adaptive entropy coefficient update direction."""

import unittest

import torch

from core.ppo_agent import AdaptiveEntropyCoeff as StateEntropyCoeff
from core.visual_ppo_agent import (
    AdaptiveEntropyCoeff as VisualEntropyCoeff,
    VisualActorCritic,
)


class AdaptiveEntropyDirectionTest(unittest.TestCase):
    def _check_direction(self, coeff_class):
        high_entropy_coeff = coeff_class(
            initial_coeff=0.01, target_entropy=2.5, lr=1e-2
        )
        initial = high_entropy_coeff.get_coeff()
        high_entropy_coeff.update(torch.tensor(4.0))
        self.assertLess(
            high_entropy_coeff.get_coeff(),
            initial,
            "alpha must decrease when entropy is above the target",
        )

        low_entropy_coeff = coeff_class(
            initial_coeff=0.01, target_entropy=2.5, lr=1e-2
        )
        initial = low_entropy_coeff.get_coeff()
        low_entropy_coeff.update(torch.tensor(1.0))
        self.assertGreater(
            low_entropy_coeff.get_coeff(),
            initial,
            "alpha must increase when entropy is below the target",
        )

    def test_state_agent_direction(self):
        self._check_direction(StateEntropyCoeff)

    def test_visual_agent_direction(self):
        self._check_direction(VisualEntropyCoeff)

    def test_log_std_remains_trainable_at_upper_bound(self):
        model = VisualActorCritic()
        with torch.no_grad():
            model.log_std.fill_(0.1)

        depth = torch.zeros(2, 1, 64, 64)
        vec = torch.zeros(2, 6)
        _, std, _ = model(depth, vec)
        std.sum().backward()

        self.assertTrue(torch.all(model.log_std.grad > 0))
        model.clamp_log_std_()
        self.assertTrue(torch.all(model.log_std <= 0))


if __name__ == "__main__":
    unittest.main()
