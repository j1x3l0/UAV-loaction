#!/usr/bin/env python3
"""Plot degradation curves from a V3 aggregate CSV (paper-ready).

Reads an evaluation summary CSV (one row per axis/level, e.g.
``aggregate_summary.csv`` from the V3 eval pipeline) and draws one
success-rate curve per degradation axis with a Wilson CI band.

Input column contract (superset accepted):
  axis, level, success_rate, ci95_low, ci95_high  (required)
  axis_name, unit, episodes, collision_rate        (optional, used when present)

Without ci95 columns the script still plots the point curve and warns.

Example:
  python scripts/plot_v3_aggregate.py \\
      --csv ../reports/v3_scale_curriculum_eval_.../aggregate_summary.csv \\
      --output figures/v3_depth_scale.png --title "V3: depth-scale degradation"

Notes for paper use: this is a *plotting* tool only — significance
claims come from utils/stats.py (paired_bootstrap / mcnemar) on the
per-episode outcomes. Figures generated from legacy-unaligned data must
be labelled diagnostic, not formal V3 results (see reports/README.md).
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paper-style defaults, consistent with plot_degradation_curves.py.
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})

PALETTE = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12",
           "#8E44AD", "#16A085", "#C0392B", "#2C3E50"]


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def required_columns(rows):
    return set(rows[0].keys()) & {"axis", "level", "success_rate"}


# Columns that are not degradation-axis names (single-axis aggregate CSVs
# use the axis name as the header, e.g. "depth_scale,successes,episodes,...").
_NON_AXIS_COLUMNS = {
    "axis", "axis_name", "level", "unit", "successes", "episodes",
    "success_rate", "ci95_low", "ci95_high", "collision_rate",
    "timeout_rate", "avg_reward", "reward_std", "avg_steps",
    "success_ci_low", "success_ci_high",
}


def infer_axis_names(rows, axis_column=None):
    """Return {axis_name: rows} from either layout.

    Layout 1: an ``axis`` column (multi-axis evaluation summary).
    Layout 2: no ``axis`` column — a single degradation axis whose name
    is the header (``depth_scale,successes,...``) and whose values are
    the levels. ``axis_column`` names that header explicitly.
    """
    if "axis" in rows[0]:
        axes = {}
        for row in rows:
            axes.setdefault(row["axis"], []).append(row)
        return axes
    if axis_column is not None:
        if axis_column not in rows[0]:
            raise ValueError(f"--axis-column '{axis_column}' not in CSV")
        # Single-axis aggregate: the column header is the axis name and
        # its values are the degradation levels.
        if "level" not in rows[0]:
            for row in rows:
                row["level"] = row[axis_column]
        return {axis_column: rows}
    axis_names = [c for c in rows[0] if c not in _NON_AXIS_COLUMNS]
    if len(axis_names) != 1:
        raise ValueError(
            "cannot infer axis: no 'axis' column and headers are "
            f"{sorted(set(rows[0]) - _NON_AXIS_COLUMNS)}; "
            "pass --axis-column to disambiguate")
    return {axis_names[0]: rows}


def to_float(rows, key, default=None):
    values = []
    for row in rows:
        raw = row.get(key)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(default)
    return np.array(values)


def plot_ablation_bar(ax, rows, color):
    """Horizontal bar chart for ablation CSVs (label/ablation column).

    Rows are sorted by success rate (baseline first if present) with
    Wilson CI error bars when ci columns exist.
    """
    label_key = "label" if "label" in rows[0] else "ablation"
    names = [row[label_key] for row in rows]
    success = to_float(rows, "success_rate")
    order = np.argsort(success)[::-1]
    base_first = [i for i in range(len(names)) if names[i] == "baseline"]
    if base_first:
        base = base_first[0]
        order = np.array([base] + [i for i in order if i != base])

    names_o = [names[i] for i in order]
    success_o = success[order]
    y = np.arange(len(names_o))
    ax.barh(y, success_o, height=0.6, color=color, alpha=0.85)
    has_ci = "success_ci_low" in rows[0] and "success_ci_high" in rows[0]
    ci_low_key = "success_ci_low" if has_ci else "ci95_low"
    ci_high_key = "success_ci_high" if has_ci else "ci95_high"
    if ci_low_key in rows[0] and ci_high_key in rows[0]:
        lo = to_float(rows, ci_low_key)[order]
        hi = to_float(rows, ci_high_key)[order]
        xerr = np.maximum(0.0, np.stack([success_o - lo, hi - success_o]))
        ax.errorbar(success_o, y, xerr=xerr,
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_yticks(y, names_o)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Success rate (%)")
    ax.grid(True, axis="x", alpha=0.3)


def plot_axis(ax, rows, color, label):
    """Draw one axis: success rate vs level with Wilson CI band."""
    levels = to_float(rows, "level")
    success = to_float(rows, "success_rate")
    order = np.argsort(levels)
    levels, success = levels[order], success[order]

    has_ci = "ci95_low" in rows[0] and "ci95_high" in rows[0]
    if has_ci:
        lo = to_float(rows, "ci95_low")[order]
        hi = to_float(rows, "ci95_high")[order]
        ax.fill_between(levels, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(levels, success, marker="o", linewidth=1.8,
            color=color, label=label)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="aggregate CSV path")
    parser.add_argument("--output", required=True, help="output figure path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--axis-column", default=None,
                        help="explicit axis column when the CSV mixes "
                             "several comparison columns (default: infer)")
    parser.add_argument("--min-episodes", type=int, default=0,
                        help="skip rows with fewer episodes than this")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        sys.exit(f"error: no rows in {args.csv}")
    missing = required_columns(rows) - {"axis", "level", "success_rate"}
    if missing:
        sys.exit(f"error: {args.csv} missing columns: {sorted(missing)}")

    if args.min_episodes:
        n_ep = to_float(rows, "episodes", default=float("inf"))
        rows = [r for r, n in zip(rows, n_ep) if n >= args.min_episodes]

    if "ci95_low" not in rows[0] or "ci95_high" not in rows[0]:
        print("warning: no ci95_low/ci95_high columns; CI band omitted")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Ablation CSVs (label/ablation column) get a horizontal bar chart.
    if "label" in rows[0] or "ablation" in rows[0]:
        fig, ax = plt.subplots(figsize=(6.5, max(3, 0.45 * len(rows) + 1.5)))
        plot_ablation_bar(ax, rows, PALETTE[0])
        if args.title:
            ax.set_title(args.title)
        fig.tight_layout()
        fig.savefig(args.output)
        print(f"Saved: {args.output}  (bar chart, {len(rows)} rows)")
        return

    # Group rows by axis (or infer a single axis from the header).
    axes = infer_axis_names(rows, args.axis_column)
    if len(axes) == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        (name, axis_rows), = axes.items()
        label = axis_rows[0].get("axis_name") or name
        plot_axis(ax, axis_rows, PALETTE[0], label)
        ax.set_xlabel("Level (%s)" % axis_rows[0].get("unit", "").strip()
                      if axis_rows[0].get("unit") else "Level")
        if args.title:
            ax.set_title(args.title)
        ax.legend(loc="best")
    else:
        n_axes = len(axes)
        fig, axs = plt.subplots(1, n_axes, figsize=(4.8 * n_axes, 4),
                                squeeze=False)
        for i, (name, axis_rows) in enumerate(axes.items()):
            ax = axs[0][i]
            label = axis_rows[0].get("axis_name") or name
            plot_axis(ax, axis_rows, PALETTE[i % len(PALETTE)], label)
            ax.set_xlabel("Level (%s)" % axis_rows[0].get("unit", "").strip()
                          if axis_rows[0].get("unit") else "Level")
            ax.set_title(label, fontsize=12)
        if args.title:
            fig.suptitle(args.title, y=1.02)

    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved: {args.output}  (axes: {list(axes)})")


if __name__ == "__main__":
    main()
