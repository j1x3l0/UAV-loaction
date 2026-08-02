#!/usr/bin/env python3
"""Paired statistics for the aligned V3 scale comparison.

The evaluation runs every model on the same episode seeds (paired data),
so we can test whether the robust model's success rate differs from the
clean baseline at each depth scale using McNemar's test and a paired
bootstrap of the success-rate difference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats


def _success_vector(detail_rows):
    return np.array([1 if row["result"] == "success" else 0
                     for row in detail_rows], dtype=np.int64)


def mcnemar(baseline: np.ndarray, model: np.ndarray):
    """Paired McNemar on success outcomes. Returns (p_value, n_discordant)."""
    b = int(np.sum((baseline == 1) & (model == 0)))  # baseline ok, model fail
    c = int(np.sum((baseline == 0) & (model == 1)))  # baseline fail, model ok
    total = b + c
    if total == 0:
        return 1.0, 0
    chi2 = (abs(b - c) - 1.0) ** 2 / total
    return float(stats.chi2.sf(chi2, df=1)), total


def paired_bootstrap(baseline: np.ndarray, model: np.ndarray,
                     n_boot: int = 10000, rng=None):
    """Paired bootstrap 95% CI for (model_sr - baseline_sr)."""
    rng = rng or np.random.default_rng(20260803)
    n = len(baseline)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(model[idx].mean() - baseline[idx].mean()))
    diffs = np.asarray(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="per-episode comparison JSON")
    parser.add_argument("--baseline", default="clean_baseline",
                        help="model label used as the baseline")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    detail = data["episodes_detail"]
    scales = data["scales"]
    baseline_label = args.baseline
    if baseline_label not in detail:
        raise ValueError(f"baseline {baseline_label!r} not in detail")
    models = [label for label in detail if label != baseline_label]

    report = {"baseline": baseline_label, "scales": scales}
    table = []
    for scale in scales:
        baseline_vec = _success_vector(detail[baseline_label][scale])
        scale_rows = []
        for label in models:
            model_vec = _success_vector(detail[label][scale])
            p_value, discordant = mcnemar(baseline_vec, model_vec)
            low, high = paired_bootstrap(
                baseline_vec, model_vec, n_boot=args.n_boot)
            bsr = float(baseline_vec.mean() * 100)
            msr = float(model_vec.mean() * 100)
            scale_rows.append({
                "scale": scale,
                "model": label,
                "baseline_sr": round(bsr, 1),
                "model_sr": round(msr, 1),
                "diff_pp": round(msr - bsr, 1),
                "bootstrap_95ci_pp": [round(low * 100, 1), round(high * 100, 1)],
                "mcnemar_p": p_value,
                "discordant_pairs": discordant,
                "significant_at_5pct": p_value < 0.05,
            })
        table.extend(scale_rows)
    report["tests"] = table

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"baseline: {baseline_label}")
    header = (f"{'scale':>6} {'model':<16} {'base%':>7} {'model%':>7} "
              f"{'diffpp':>7} {'boot95pp':>12} {'McN p':>8} {'sig':>5}")
    print(header)
    for row in table:
        print(
            f"{row['scale']:>6} {row['model']:<16} {row['baseline_sr']:>7.1f} "
            f"{row['model_sr']:>7.1f} {row['diff_pp']:>7.1f} "
            f"[{row['bootstrap_95ci_pp'][0]:.1f},{row['bootstrap_95ci_pp'][1]:.1f}] "
            f"{row['mcnemar_p']:>8.3f} {row['significant_at_5pct']!s:>5}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
