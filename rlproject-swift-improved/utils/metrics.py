"""Shared evaluation metrics."""

from typing import Tuple

import numpy as np


def wilson_confidence_interval(
    success_rate: float, n: int, z: float = 1.96
) -> Tuple[float, float]:
    """Return a binomial Wilson confidence interval in percentage points."""
    if n <= 0:
        return 0.0, 0.0
    denominator = 1 + z * z / n
    center = (success_rate + z * z / (2 * n)) / denominator
    margin = (
        z
        * np.sqrt(
            success_rate * (1 - success_rate) / n
            + z * z / (4 * n * n)
        )
        / denominator
    )
    return float((center - margin) * 100), float((center + margin) * 100)
