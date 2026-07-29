"""
metrics.py — 评估指标工具函数

架构位置: utils/ (Infrastructure层)
WHY 独立模块: 多个脚本(train_visual, eval_degradation, eval_baseline)共用
"""

import math


def wilson_confidence_interval(p, n, z=1.96):
    """
    Wilson score confidence interval for binomial proportion.

    Args:
        p: proportion (successes / n), in [0, 1]
        n: sample size (number of trials)
        z: z-score (1.96 ≈ 95% confidence)
    Returns:
        (lower, upper) as proportions in [0, 1]
    """
    if n == 0:
        return 0.0, 0.0
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
