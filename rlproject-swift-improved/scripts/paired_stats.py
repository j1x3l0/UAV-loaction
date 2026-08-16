#!/usr/bin/env python3
"""Paired statistical tests for per-episode evaluation results.

Given per-episode outcome lists (same episode seeds for both conditions),
compute:
  - McNemar test (p-value) for paired binary outcomes
  - paired bootstrap 95% CI for the SR difference
  - Cohen's g effect size (|b - c| / (b + c) for discordant pairs)
  - pooled SR for each condition

Used to quantify whether baseline-vs-ablation / clean-vs-curriculum /
PPO-vs-BC differences are significant, per the ICRA reproducibility
requirements (paired tests, 95% CI, effect size).

Usage:
  python scripts/paired_stats.py --a label=results.json --b label2=results.json
    where each JSON has episodes_detail per condition (from eval scripts).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy import stats


def outcome_to_binary(detail, key="result"):
    """Convert per-episode outcomes to success bools.

    Accepts either a list of {'result': 'success'|'collision'|'timeout'}
    dicts (eval scripts) or a plain list of strings (consistency eval).
    """
    if detail and isinstance(detail[0], dict):
        return np.array([r[key] == "success" for r in detail], dtype=bool)
    return np.array([r == "success" for r in detail], dtype=bool)


def paired_test(a, b, n_boot=20000, seed=20260809):
    """Paired tests on binary success vectors a, b (same episode order)."""
    assert len(a) == len(b), "paired vectors must have same length"
    n = len(a)

    # McNemar on discordant pairs
    b_disc = int(np.sum(a & ~b))   # a success, b fail
    c_disc = int(np.sum(~a & b))   # a fail, b success
    total_disc = b_disc + c_disc
    if total_disc == 0:
        mcnemar_p = 1.0
    else:
        chi2 = (abs(b_disc - c_disc) - 1.0) ** 2 / total_disc
        mcnemar_p = float(stats.chi2.sf(chi2, df=1))

    # Paired bootstrap 95% CI for (a_rate - b_rate)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(a[idx].mean() - b[idx].mean()))
    diffs = np.asarray(diffs)
    ci = (float(np.percentile(diffs, 2.5)) * 100,
          float(np.percentile(diffs, 97.5)) * 100)

    # Cohen's g for paired nominal data (0.5 = chance, 1.0 = max effect)
    cohens_g = None
    if total_disc > 0:
        cohens_g = abs(b_disc - c_disc) / total_disc

    return {
        "n": n,
        "a_successes": int(a.sum()),
        "b_successes": int(b.sum()),
        "a_sr": float(a.mean() * 100),
        "b_sr": float(b.mean() * 100),
        "diff_pp": float((a.mean() - b.mean()) * 100),
        "mcnemar_p": mcnemar_p,
        "discordant_pairs": total_disc,
        "b_discordant": b_disc,
        "c_discordant": c_disc,
        "bootstrap_95ci_pp": [round(ci[0], 1), round(ci[1], 1)],
        "ci_includes_zero": bool(ci[0] <= 0 <= ci[1]),
        "cohens_g": round(cohens_g, 3) if cohens_g is not None else None,
        "significant": bool(mcnemar_p < 0.05 and not (ci[0] <= 0 <= ci[1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="label=path.json (condition A)")
    parser.add_argument("--b", required=True, help="label=path.json (condition B)")
    parser.add_argument("--condition-a", default=None,
                        help="key in episodes_detail for A (default: first)")
    parser.add_argument("--condition-b", default=None,
                        help="key in episodes_detail for B (default: first)")
    parser.add_argument("--n-boot", type=int, default=20000)
    args = parser.parse_args()

    def load(spec):
        label, path = spec.split("=", 1)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return label, data

    label_a, data_a = load(args.a)
    label_b, data_b = load(args.b)

    def extract(data, condition, path):
        # Case 0: model:ablation path (eval_v3_ablation nests
        # episodes_detail[model][ablation]).
        if condition and ":" in condition:
            model, ablation = condition.split(":", 1)
            if "episodes_detail" in data:
                det = data["episodes_detail"]
                if model in det and ablation in det[model]:
                    return det[model][ablation]
            raise KeyError(
                f"no episodes_detail for {condition} in {path}")
        # Case 1: condition names a top-level sub-dict that itself has
        # episodes_detail (e.g. consistency {raw: {episodes_detail: [...]}}).
        if condition and condition in data and isinstance(data[condition], dict):
            sub = data[condition]
            if "episodes_detail" in sub:
                return sub["episodes_detail"]
            if "results" in sub:
                return sub["results"]
        # Case 2: top-level episodes_detail (list or {condition: [...]}).
        if "episodes_detail" in data:
            detail = data["episodes_detail"]
            if isinstance(detail, dict):
                if condition and condition in detail:
                    return detail[condition]
                # if nested one more level (model -> ablation), take first model's first ablation
                first_model = next(iter(detail.values()))
                if isinstance(first_model, dict):
                    return next(iter(first_model.values()))
                return next(iter(detail.values()))
            return detail
        raise KeyError(
            f"no episodes_detail for condition={condition} in {path}")

    a = outcome_to_binary(extract(data_a, args.condition_a, args.a))
    b = outcome_to_binary(extract(data_b, args.condition_b, args.b))
    result = paired_test(a, b, n_boot=args.n_boot)

    print(f"=== Paired test: {label_a} vs {label_b} ===")
    print(f"  n={result['n']} | {label_a} SR={result['a_sr']:.1f}% "
          f"({result['a_successes']}) | {label_b} SR={result['b_sr']:.1f}% "
          f"({result['b_successes']})")
    print(f"  diff = {result['diff_pp']:+.1f}pp")
    print(f"  McNemar p={result['mcnemar_p']:.4f} "
          f"(discordant={result['discordant_pairs']}, "
          f"b={result['b_discordant']}, c={result['c_discordant']})")
    print(f"  bootstrap 95% CI = {result['bootstrap_95ci_pp']}pp "
          f"(includes 0: {result['ci_includes_zero']})")
    print(f"  Cohen's g = {result['cohens_g']} | "
          f"significant (p<0.05 & CI excl. 0): {result['significant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
