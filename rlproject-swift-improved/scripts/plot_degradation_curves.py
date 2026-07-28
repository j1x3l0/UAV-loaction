"""
plot_degradation_curves.py — 衰减曲线可视化 v2

用法:
  python scripts/plot_degradation_curves.py eval_results/degradation_20260722_XXXXXX.json
  python scripts/plot_degradation_curves.py eval_results/degradation_20260722_XXXXXX.json --output figures/
"""

import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 论文级样式 ──
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

AXIS_COLORS = {
    'gaussian':           '#E74C3C',  # 红
    'resolution':         '#2980B9',  # 蓝
    'depth_noise':        '#27AE60',  # 绿
    'lighting':           '#F39C12',  # 橙
    'viewpoint_uncertainty': '#8E44AD',  # 紫
    'depth_failure':       '#C0392B',
    'occlusion':           '#7F8C8D',
    'depth_scale':         '#16A085',
    'combined':            '#2C3E50',
}

AXIS_LABELS = {
    'gaussian':           'Gaussian Count (%)',
    'resolution':         'Render Resolution (px)',
    'depth_noise':        'Depth Noise (sigma)',
    'lighting':           'Lighting Offset (EV)',
    'viewpoint_uncertainty': 'Viewpoint Uncertainty (deg)',
    'depth_failure':       'Depth Failure (%)',
    'occlusion':           'Occlusion (%)',
    'depth_scale':         'Depth Scale',
    'combined':            'Combined Severity',
}


def present_axes(results):
    """Return axes in first-occurrence order for the selected evaluation suite."""
    return list(dict.fromkeys(row['axis'] for row in results))


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def plot_single_axis(ax, axis_name, results, add_critical_point=True):
    """在单个子图上绘制一条衰减曲线"""
    axis_data = [r for r in results if r['axis'] == axis_name]
    if not axis_data:
        return

    axis_data.sort(key=lambda r: r['level'])
    levels = [r['level'] for r in axis_data]
    sr = [r['success_rate'] for r in axis_data]
    cr = [r['collision_rate'] for r in axis_data]

    color = AXIS_COLORS.get(axis_name, '#333')
    label = AXIS_LABELS.get(axis_name, axis_name)

    # SR 主曲线
    ax.plot(levels, sr, 'o-', color=color, linewidth=2, markersize=7,
            label='Success Rate')
    # CR 虚线
    ax.plot(levels, cr, 's--', color=color, linewidth=1, markersize=5,
            alpha=0.5, label='Collision Rate')

    # 标注临界点 (SR < 50%)
    if add_critical_point:
        for l, s in zip(levels, sr):
            if s < 50:
                ax.axvline(x=l, color=color, linestyle=':', alpha=0.6,
                          linewidth=1.5)
                ax.annotate(f'critical: {l}', xy=(l, s),
                           xytext=(l, s - 15),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=0.6),
                           fontsize=9, color=color)
                break

    ax.set_xlabel(AXIS_LABELS.get(axis_name, axis_name))
    ax.set_ylabel('Rate (%)')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)


def plot_all_axes(results, output_path, title=None):
    """绘制所有退化轴的全景图"""
    axes_names = present_axes(results)
    n = len(axes_names)

    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 3.5), sharey=True)
    axes = np.atleast_1d(axes)

    for i, axis_name in enumerate(axes_names):
        plot_single_axis(axes[i], axis_name, results)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_critical_points(results, output_path):
    """绘制临界点汇总图"""
    axes_names = present_axes(results)
    critical = {}
    for axis_name in axes_names:
        axis_data = sorted(
            [r for r in results if r['axis'] == axis_name],
            key=lambda r: r['level'])
        for r in axis_data:
            if r['success_rate'] < 50:
                critical[axis_name] = r['level']
                break
        if axis_name not in critical:
            critical[axis_name] = None

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [AXIS_LABELS.get(a, a) for a in axes_names]
    values = [critical[a] if critical[a] is not None else 0 for a in axes_names]
    colors = [AXIS_COLORS.get(a, '#333') for a in axes_names]

    bars = ax.barh(names, values, color=colors, edgecolor='white', height=0.6)
    ax.set_xlabel('Critical Degradation Level\n(Success Rate drops below 50%)')
    ax.set_title('Degradation Axis Lethality Ranking')

    # 标注
    for bar, val, name in zip(bars, values, axes_names):
        if critical[name] is not None:
            unit = next((r['unit'] for r in results if r['axis'] == name), '')
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val}{unit}', va='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_table(results, output_path):
    """生成退化轴汇总表格图"""
    axes_names = present_axes(results)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    rows = []
    for axis_name in axes_names:
        axis_data = sorted(
            [r for r in results if r['axis'] == axis_name],
            key=lambda r: r['level'])
        for r in axis_data:
            rows.append([
                AXIS_LABELS.get(axis_name, axis_name),
                f"{r['level']}{r['unit']}",
                f"{r['success_rate']:.1f}%",
                f"{r['collision_rate']:.1f}%",
                f"{r['avg_reward']:.0f}",
            ])

    col_labels = ['Axis', 'Level', 'Success %', 'Collision %', 'Avg Reward']
    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制衰减曲线')
    parser.add_argument('json_path', help='eval_degradation.py输出的JSON')
    parser.add_argument('--output', default='eval_results',
                       help='输出目录')
    parser.add_argument('--title', default=None,
                       help='图表标题')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_results(args.json_path)
    results = data['results']
    timestamp = data.get('timestamp', 'unknown')

    prefix = os.path.join(args.output, f'degradation_{timestamp}')

    plot_all_axes(results, f'{prefix}_all_axes.png', args.title)
    plot_critical_points(results, f'{prefix}_critical.png')
    plot_summary_table(results, f'{prefix}_table.png')

    print(f"\nDone. Output in: {args.output}/")


if __name__ == "__main__":
    main()
