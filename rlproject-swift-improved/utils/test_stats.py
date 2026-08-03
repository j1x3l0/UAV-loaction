"""Tests for utils.stats paired significance tests."""

import unittest

import numpy as np
from scipy.stats import binomtest, chi2

from utils.stats import mcnemar, paired_bootstrap


class McnemarTest(unittest.TestCase):
    def test_known_textbook_case(self):
        # Classic example: b=10, c=2 discordant cells.
        cond_a = np.array([1] * 10 + [0] * 2 + [1] * 18)
        cond_b = np.array([0] * 10 + [1] * 2 + [1] * 18)
        result = mcnemar(cond_a, cond_b)
        self.assertEqual(result["b"], 10)
        self.assertEqual(result["c"], 2)
        self.assertEqual(result["n_discordant"], 12)
        # Continuity-corrected chi-square: (10-2-1)^2/12 = 4.0833, df=1.
        self.assertAlmostEqual(result["p_value"], 0.0433, places=4)
        self.assertLess(mcnemar(cond_a, cond_b, exact=True)["p_value"],
                        result["p_value"])

    def test_matches_scipy_binomtest_on_random_tables(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(10, 300)
            cond_a = rng.random(n) > rng.uniform(0.3, 0.7)
            cond_b = rng.random(n) > rng.uniform(0.3, 0.7)
            b_wins = int(np.sum((cond_a == 1) & (cond_b == 0)))
            c_wins = int(np.sum((cond_a == 0) & (cond_b == 1)))
            exact_ref = float(binomtest(min(b_wins, c_wins),
                                        b_wins + c_wins, 0.5).pvalue)
            self.assertAlmostEqual(
                mcnemar(cond_a, cond_b, exact=True)["p_value"],
                exact_ref, places=9)
            statistic = (abs(b_wins - c_wins) - 1.0) ** 2 / max(b_wins + c_wins, 1)
            corrected_ref = float(chi2.sf(statistic, df=1))
            self.assertAlmostEqual(
                mcnemar(cond_a, cond_b)["p_value"], corrected_ref, places=9)

    def test_no_discordant_cells_returns_p_one(self):
        cond_a = np.array([1, 1, 0, 0, 1])
        result = mcnemar(cond_a, cond_a.copy())
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["n_discordant"], 0)

    def test_rejects_misaligned_or_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            mcnemar(np.ones(5), np.zeros(4))
        with self.assertRaisesRegex(ValueError, "0/1"):
            mcnemar(np.array([0, 2, 1]), np.array([1, 0, 1]))
        with self.assertRaisesRegex(ValueError, "1-D"):
            mcnemar(np.array([[1, 0]]), np.array([1, 0]))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            mcnemar(np.array([1]), np.array([0]))


class PairedBootstrapTest(unittest.TestCase):
    def test_identical_conditions(self):
        rng = np.random.default_rng(7)
        cond = rng.random(200) > 0.4
        result = paired_bootstrap(cond, cond.copy(), seed=7)
        self.assertAlmostEqual(result["diff"], 0.0, places=12)
        self.assertGreater(result["p_value"], 0.9)
        self.assertEqual(result["n"], 200)
        self.assertEqual(result["n_boot"], 10000)

    def test_clearly_different_conditions(self):
        rng = np.random.default_rng(7)
        cond_a = rng.random(200) > 0.4   # ~60% success
        cond_b = rng.random(200) > 0.7   # ~30% success
        result = paired_bootstrap(cond_a, cond_b, seed=7)
        self.assertGreater(result["diff"], 0.2)
        self.assertLess(result["p_value"], 0.001)
        self.assertLess(result["ci95"][0], result["diff"] < result["ci95"][1])

    def test_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            paired_bootstrap(np.ones(5), np.zeros(4))
        with self.assertRaisesRegex(ValueError, "0/1"):
            paired_bootstrap(np.array([0, 2]), np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
