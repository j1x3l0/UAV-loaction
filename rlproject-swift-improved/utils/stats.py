"""Paired significance tests for V3 formal evaluation.

Two conditions (e.g. clean vs degraded rendering) are evaluated on the
same episode seeds, giving paired binary outcomes (success/failure) per
episode. This module provides the two standard tests for that setup:

- ``paired_bootstrap`` — percentile bootstrap of the success-rate
  difference, with a two-sided p-value. Handles small n gracefully and
  makes no distributional assumption beyond the resampling itself.
- ``mcnemar`` — McNemar's exact / corrected test on the discordant
  cells. The natural test for "the two conditions differ" on paired
  binary data.

Inputs are per-episode outcome arrays of equal length, aligned by
episode index: element i of ``cond_a`` and element i of ``cond_b`` come
from the same episode seed. Outcomes may be bool or 0/1 int.

Why paired tests? Reports aggregate per-condition success rates
(e.g. ``aggregate_summary.csv``), but independent-proportion tests
ignore that the same episodes were reused across conditions. Pairing
removes that common variance, which is what makes small degradation
effects detectable at all with V3's ~200-600 episodes per level.

Reference: Dietterich (1998), "Approximate Statistical Tests for
Comparing Supervised Classification Learning Algorithms", §2.3-2.4
(paired bootstrap and McNemar are both covered there).
"""

from __future__ import annotations

import math

import numpy as np


def _as_binary(condition, name: str) -> np.ndarray:
    values = np.asarray(condition)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D (per-episode outcomes)")
    allowed = (values == 0) | (values == 1)
    if not np.all(allowed):
        raise ValueError(f"{name} must contain only 0/1 or bool outcomes")
    return values.astype(np.int64)


def _align(cond_a, cond_b):
    a = _as_binary(cond_a, "cond_a")
    b = _as_binary(cond_b, "cond_b")
    if len(a) != len(b):
        raise ValueError(
            f"conditions must be episode-aligned: len(a)={len(a)} != len(b)={len(b)}")
    if len(a) < 2:
        raise ValueError("need at least 2 paired episodes")
    return a, b


def paired_bootstrap(cond_a, cond_b, n_boot: int = 10000,
                     seed: int | None = None) -> dict:
    """Bootstrap the success-rate difference ``cond_a - cond_b``.

    Resamples episode *indices* with replacement (keeping the pairing)
    and reports the percentile interval of the rate difference, plus a
    two-sided p-value: the fraction of bootstrap differences whose
    sign disagrees with the observed difference. The p-value uses the
    standard doubled-tail construction (≈2×min(P(diff>=0), P(diff<=0))),
    which reduces to the usual bootstrap test for a difference of zero.

    Returns:
        {"diff": observed diff, "ci95": [lo, hi],
         "p_value": two-sided bootstrap p-value,
         "n": paired episodes, "n_boot": resamples used}
    """
    a, b = _align(cond_a, cond_b)
    n = len(a)
    diff_obs = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diff = (a[idx].mean(axis=1) - b[idx].mean(axis=1)).astype(np.float64)

    lo, hi = np.percentile(boot_diff, [2.5, 97.5])
    # Two-sided p: probability of observing a difference at least as
    # extreme away from 0 as the observed one, in either direction.
    if diff_obs >= 0:
        p_value = 2.0 * float(np.mean(boot_diff <= 0))
    else:
        p_value = 2.0 * float(np.mean(boot_diff >= 0))
    p_value = min(p_value, 1.0)

    return {
        "diff": float(diff_obs),
        "ci95": [float(lo), float(hi)],
        "p_value": float(p_value),
        "n": n,
        "n_boot": n_boot,
    }


def mcnemar(cond_a, cond_b, exact: bool = False) -> dict:
    """McNemar's test on paired binary outcomes.

    Cell b = episodes where A succeeded and B failed; cell c = the
    reverse. Under the null the two discordant cells are balanced
    (b == c). ``exact=True`` uses the two-sided binomial tail on
    min(b, c) with p=0.5; ``exact=False`` applies the continuity
    correction, matching SciPy's ``stats.contingency.mcnemar`` default
    for a 2x2 table.

    Returns:
        {"b": A-win count, "c": B-win count,
         "p_value": two-sided p-value, "exact": bool,
         "n_discordant": b + c}
    """
    a, b = _align(cond_a, cond_b)
    b_wins = int(np.sum((a == 1) & (b == 0)))
    c_wins = int(np.sum((a == 0) & (b == 1)))
    n_discordant = b_wins + c_wins

    if n_discordant == 0:
        return {"b": b_wins, "c": c_wins, "p_value": 1.0,
                "exact": exact, "n_discordant": 0}

    if exact:
        # Two-sided exact binomial tail on the smaller discordant cell.
        k = min(b_wins, c_wins)
        p_value = 0.0
        term = 0.5 ** n_discordant
        for i in range(k + 1):
            # C(n, i) / 2^n, summed up to the smaller cell.
            comb = float(math.comb(n_discordant, i))
            p_value += comb * term
        p_value = 2.0 * min(p_value, 1.0)
        p_value = min(p_value, 1.0)
    else:
        # Corrected chi-square: (|b-c| - 1)^2 / (b+c) ~ chi2(1).
        numerator = (abs(b_wins - c_wins) - 1.0) ** 2
        statistic = numerator / n_discordant if n_discordant else 0.0
        from scipy.stats import chi2
        p_value = float(1.0 - chi2.cdf(statistic, df=1))

    return {"b": b_wins, "c": c_wins, "p_value": float(p_value),
            "exact": exact, "n_discordant": n_discordant}


if __name__ == "__main__":
    # Smoke self-check against a constructed example.
    rng = np.random.default_rng(0)
    n_ep = 200
    # A succeeds 70%, B succeeds 55%, outcomes correlated per episode.
    base = rng.random(n_ep)
    cond_a = (base + rng.random(n_ep) * 0.5) > 0.35
    cond_b = (base + rng.random(n_ep) * 0.5) > 0.55
    print("A SR: %.1f%%  B SR: %.1f%%" % (cond_a.mean() * 100, cond_b.mean() * 100))
    print("paired_bootstrap:", paired_bootstrap(cond_a, cond_b, seed=1))
    print("mcnemar (corrected):", mcnemar(cond_a, cond_b))
    print("mcnemar (exact):", mcnemar(cond_a, cond_b, exact=True))
