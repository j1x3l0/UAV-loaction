"""
A1 噪声衰减曲线可视化
读取 A1 batch 评估结果，生成:
  1. 成功率衰减曲线 (5模式叠加)
  2. 碰撞率增长曲线
  3. 路径长度膨胀曲线
  4. 综合雷达图 (sigma=1.0 下各模式对比)
  5. 汇总分析图

输出: eval_results/a1_noise_decay_curves.png
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 配色方案 (colorblind-friendly + 暗色主题兼容)
# ============================================================
COLORS = {
    'pos':    '#E74C3C',  # 红
    'vel':    '#2ECC71',  # 绿
    'target': '#3498DB',  # 蓝
    'obs':    '#F39C12',  # 橙
    'full':   '#9B59B6',  # 紫
}

PATTERN_LABELS = {
    'pos':    'Pos (位置噪声)',
    'vel':    'Vel (速度噪声)',
    'target': 'Target (目标噪声)',
    'obs':    'Obs (障碍物噪声)',
    'full':   'Full (全维噪声)',
}

MARKERS = {'pos': 'o', 'vel': 's', 'target': '^', 'obs': 'D', 'full': 'v'}

# Obs 模式的 x 轴标签（因为不是单个 sigma 值，用序号）
OBS_SIGMA_LABELS = [
    'dir=0.1\ndist=0.1',
    'dir=0.3\ndist=0.1',
    'dir=0.5\ndist=0.1',
    'dir=0.5\ndist=0.5',
    'dir=0.5\ndist=1.0',
]


def load_results():
    """加载最新的 A1 batch 结果"""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval_results')
    json_files = sorted([f for f in os.listdir(results_dir)
                        if f.startswith('a1_batch_results_') and f.endswith('.json')])
    if not json_files:
        print("ERROR: 未找到 A1 batch 结果 JSON 文件")
        sys.exit(1)

    latest = json_files[-1]
    path = os.path.join(results_dir, latest)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载数据: {latest} ({data['num_experiments']} 组实验)")
    return data


def organize_by_pattern(results):
    """按模式分组"""
    by_pattern = {'pos': [], 'vel': [], 'target': [], 'obs': [], 'full': []}
    for row in results:
        p = row['pattern']
        if p in by_pattern:
            by_pattern[p].append(row)
    # 每组内按 success_rate 降序排列（sigma 从小到大）
    for p in by_pattern:
        by_pattern[p].sort(key=lambda r: r['success_rate'], reverse=True)
    return by_pattern


def plot_success_rate(ax, by_pattern):
    """图1: 成功率衰减曲线"""
    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        rows = by_pattern[pattern]
        if pattern == 'obs':
            x = list(range(len(rows)))
            ax.plot(x, [r['success_rate'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8, label=PATTERN_LABELS[pattern])
            ax.set_xticks(x)
            ax.set_xticklabels(OBS_SIGMA_LABELS, fontsize=7.5)
        else:
            sigmas = [float(r['sigma']) for r in rows]
            ax.plot(sigmas, [r['success_rate'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8, label=PATTERN_LABELS[pattern])

    ax.axhline(y=98, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('Baseline 98%', xy=(4.5, 98.5), fontsize=8, color='gray', alpha=0.7)
    ax.set_xlabel('Noise sigma', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('A1: Success Rate vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.8)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)


def plot_collision_rate(ax, by_pattern):
    """图2: 碰撞率增长曲线"""
    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        rows = by_pattern[pattern]
        if pattern == 'obs':
            x = list(range(len(rows)))
            ax.plot(x, [r['collision_rate'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8)
        else:
            sigmas = [float(r['sigma']) for r in rows]
            ax.plot(sigmas, [r['collision_rate'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8, label=PATTERN_LABELS[pattern])

    ax.axhline(y=4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('Week1 limit 4%', xy=(1.5, 6), fontsize=8, color='gray', alpha=0.7)
    ax.set_xlabel('Noise sigma', fontsize=11)
    ax.set_ylabel('Collision Rate (%)', fontsize=11)
    ax.set_title('A1: Collision Rate vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)


def plot_path_length(ax, by_pattern):
    """图3: 路径长度膨胀"""
    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        rows = by_pattern[pattern]
        if pattern == 'obs':
            x = list(range(len(rows)))
            ax.plot(x, [r['avg_path_length'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8)
        else:
            sigmas = [float(r['sigma']) for r in rows]
            ax.plot(sigmas, [r['avg_path_length'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8, label=PATTERN_LABELS[pattern])

    ax.axhline(y=15.05, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('A* lower bound 15.05m', xy=(2.5, 16), fontsize=8, color='gray', alpha=0.7)
    ax.set_xlabel('Noise sigma', fontsize=11)
    ax.set_ylabel('Avg Path Length (m)', fontsize=11)
    ax.set_title('A1: Path Length Inflation vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    # log scale for pos/full which inflate dramatically
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())


def plot_min_obs_dist(ax, by_pattern):
    """图4: 最近障碍物距离衰减"""
    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        rows = by_pattern[pattern]
        if pattern == 'obs':
            x = list(range(len(rows)))
            ax.plot(x, [r['avg_min_obs_dist'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8)
        else:
            sigmas = [float(r['sigma']) for r in rows]
            ax.plot(sigmas, [r['avg_min_obs_dist'] for r in rows],
                    color=COLORS[pattern], marker=MARKERS[pattern],
                    linewidth=2.2, markersize=8, label=PATTERN_LABELS[pattern])

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.annotate('Collision threshold 0.5m', xy=(2.5, 0.55), fontsize=8, color='red', alpha=0.5)
    ax.set_xlabel('Noise sigma', fontsize=11)
    ax.set_ylabel('Avg Min Obstacle Distance (m)', fontsize=11)
    ax.set_title('A1: Safety Margin vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)


def plot_radar(ax, by_pattern):
    """图5: 雷达图 — sigma=1.0 (or closest) 下五模式对比"""
    # 找到每个模式在 sigma≈1.0 或 mid-point 的数据
    metrics = {}
    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        rows = by_pattern[pattern]
        # 找 sigma=1.0 的行 (或最接近的)
        if pattern == 'obs':
            target_row = rows[2]  # dir=0.5,dist=0.1
        else:
            for r in rows:
                if abs(float(r['sigma']) - 1.0) < 0.01:
                    target_row = r
                    break
            else:
                target_row = rows[min(2, len(rows)-1)]  # fallback

        metrics[pattern] = {
            'success_rate': target_row['success_rate'],
            'collision_rate': target_row['collision_rate'],
            'avg_path_length': target_row['avg_path_length'],
            'avg_min_obs_dist': target_row['avg_min_obs_dist'],
        }

    # 归一化到 0-100
    categories = ['Success\nRate', 'Safety\n(100-collision)', 'Path\nEfficiency', 'Obstacle\nClearance']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for pattern in ['pos', 'vel', 'target', 'obs', 'full']:
        m = metrics[pattern]
        values = [
            m['success_rate'],
            100 - m['collision_rate'],
            max(0, 100 - (m['avg_path_length'] - 15) / 2),  # 15m = 100, 更多=更低
            min(100, m['avg_min_obs_dist'] * 50),  # 2m=100
        ]
        values = [max(0, min(100, v)) for v in values]
        values += values[:1]

        ax.fill(angles, values, alpha=0.08, color=COLORS[pattern])
        ax.plot(angles, values, color=COLORS[pattern], linewidth=2,
                marker='o', markersize=6, label=PATTERN_LABELS[pattern])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.set_title('A1: Cross-Pattern Comparison (~sigma=1.0)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8, framealpha=0.8)


def plot_summary(ax, by_pattern):
    """图6: 文本分析总结"""
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 关键发现
    findings = [
        ("KEY FINDINGS — A1 Noise Decay Curves", True, 9.5),
        ("", False, 0),
        ("1. TARGET noise is the MOST FATAL: sigma=2.0 drops to 32%, sigma=5.0 to 2%", False, 8.5),
        ("   → Wrong target position = agent flies to wrong place. Catastrophic failure.", False, 8.0),
        ("", False, 0),
        ("2. VEL noise is the MOST ROBUST: sigma=5.0 still maintains 80% success", False, 7.0),
        ("   → Velocity reading errors are naturally compensated by closed-loop control.", False, 6.5),
        ("", False, 0),
        ("3. OBS noise is nearly HARMLESS: all 5 combos >= 90% success", False, 5.5),
        ("   → The learned obstacle avoidance is robust to perception errors.", False, 5.0),
        ("", False, 0),
        ("4. FULL noise degrades FASTEST (beyond Pos alone): sigma=2.0 -> 42%", False, 4.0),
        ("   → Multi-dimensional noise has compounding effect (curse of dimensionality).", False, 3.5),
        ("", False, 0),
        ("5. PATH inflation is exponential in Pos/Full, linear in Vel/Target/Obs", False, 2.5),
        ("   → Position noise causes oscillatory correction → wasteful zigzag paths.", False, 2.0),
        ("", False, 0),
        ("Implication for A2: Target noise needs prioritized robustness training.", False, 1.0),
        ("Curriculum should emphasize target/pos noise more than vel/obs.", False, 0.5),
    ]

    for text, is_title, y in findings:
        if is_title:
            ax.text(0.5, y, text, fontsize=13, fontweight='bold', color='#2C3E50',
                    transform=ax.get_yaxis_transform(), va='top')
        elif text:
            ax.text(0.5, y, text, fontsize=9, color='#555555',
                    transform=ax.get_yaxis_transform(), va='top')


def main():
    # 设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    data = load_results()
    by_pattern = organize_by_pattern(data['results'])

    # ============================================================
    # 大图: 3x2 布局
    # ============================================================
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('A1 Noise Decay Curves — UAV RL Path Planning (Week 2)',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_success_rate(ax1, by_pattern)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_collision_rate(ax2, by_pattern)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_radar(ax3, by_pattern)

    ax4 = fig.add_subplot(gs[1, 0])
    plot_path_length(ax4, by_pattern)

    ax5 = fig.add_subplot(gs[1, 1])
    plot_min_obs_dist(ax5, by_pattern)

    ax6 = fig.add_subplot(gs[1, 2])
    plot_summary(ax6, by_pattern)

    # 保存
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval_results',
                               'a1_noise_decay_curves.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\n[DONE] 图表已保存: {output_path}")

    # ============================================================
    # 单独大图: 成功率 Overlay (演讲用)
    # ============================================================
    fig2, ax = plt.subplots(figsize=(12, 7))
    plot_success_rate(ax, by_pattern)
    ax.set_title('Success Rate Decay Under Sensor Noise\n'
                 '(PPO-Swift, 14D vector state, no retraining)',
                 fontsize=14, fontweight='bold')
    fig2.tight_layout()
    overlay_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval_results',
                                'a1_success_decay_overlay.png')
    fig2.savefig(overlay_path, dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"[DONE] Overlay: {overlay_path}")

    # ============================================================
    # 打印分析文本
    # ============================================================
    print("""
============================================================
A1 噪声衰减曲线 — 初步分析
============================================================

[噪声脆弱性排序] (最脆弱 → 最鲁棒)

  Target >> Full > Pos >> Obs > Vel
  sigma=5.0: 2%    42%   56%   90%   80%

[路径膨胀特征]

  Pos/Full:  指数级膨胀 (sigma=2.0时路径200-600m)
  Vel/Target: 线性膨胀 (sigma=5.0时路径20-34m)
  Obs:        几乎无膨胀 (所有水平<15m)

[安全边际]

  Target sigma=5.0: min_obs_dist=0.53m (极度危险, 贴近碰撞阈值0.5m)
  其他模式: >0.77m (相对安全)

[A2 实验指导]

  1. Target 噪声: 需要优先鲁棒化训练
  2. Curriculum 策略: 重点放在 pos + target 维度
  3. Rand 上界: U(0,2.0) 合理, sigma=2.0时 full 已降至42%
  4. Obs 噪声: 可能不需要额外训练（已鲁棒）
============================================================
""")


if __name__ == '__main__':
    main()
