"""
degradation_utils.py — 5退化轴的参数化控制 v2

架构位置: envs/ (Extension层)
WHY 集中定义:
  - 5条退化轴统一声明 → eval和绘图脚本共用同一份定义
  - 每个退化函数独立可测 → 可组合、可替换
  - 与VisualDroneEnv解耦 → 真实3DGS接入后只需替换渲染器，退化逻辑不变
数据流: config dict → apply_degradation(depth, rgb, renderer, config) → degraded_depth
边界: 只负责观测退化，不修改物理参数、不修改奖励函数
风险: 真实3DGS的退化表现可能与mock不同 → 真3DGS后需重新标定水平值

退化轴总览:
  1. gaussian     — 高斯球稀疏化 (修改renderer内部)
  2. resolution   — 渲染分辨率降低
  3. depth_noise  — 深度图空间相关噪声
  4. lighting     — RGB光照偏移 (EV档)
  5. viewpoint_uncertainty — 视角不确定性
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


# ─── 退化轴定义 ──────────────────────────────────────────────────
# WHY 集中声明: eval_degradation.py 和 plot_degradation_curves.py 共用

DEGRADATION_AXES = {
    'gaussian': {
        'name': '高斯球稀疏化',
        'levels': [100, 50, 20, 5, 2],
        'unit': '%',
        'description': '按opacity×体积重要性保留的高斯球比例 (v2: 加强2%临界点，评估opacity-only/随机/投影视觉贡献)',
    },
    'resolution': {
        'name': '渲染分辨率',
        'levels': [64, 32, 16, 8, 2],
        'unit': 'px',
        'description': '深度图降采样分辨率（上采样回64×64），分别支持nearest/bilinear插值；深度保留float避免uint8量化',
    },
    'depth_noise': {
        'name': '深度噪声',
        'levels': [0.0, 0.1, 0.25, 0.5, 1.0],
        'unit': 'σ',
        'description': '空间相关噪声标准差 (v2: 扩大范围，加入距离相关、结构性空洞、局部偏置/尺度漂移、flying pixels)',
    },
    'lighting': {
        'name': '光照偏移',
        'levels': [0, 1, 2, 3, 4],
        'unit': 'EV',
        'description': 'RGB曝光偏移（EV档），depth-only策略下保留为负对照',
    },
    'viewpoint_uncertainty': {
        'name': '视角不确定性',
        'levels': [360, 270, 180, 90, 45],
        'unit': '°',
        'description': '训练视角覆盖不足导致的重建不确定性 (联合评估: SR + 超时率 + 平均奖励，因主表现为超时增加)',
    },
    'depth_failure': {
        'name': '深度大面积失效',
        'levels': [0, 25, 50, 75, 90],
        'unit': '%',
        'description': '固定空间块中的深度像素失效并返回最大量程',
    },
    'occlusion': {
        'name': '相机遮挡',
        'levels': [0, 25, 50, 75, 90],
        'unit': '%',
        'description': '从图像边缘向中心扩张的近距离前景遮挡',
    },
    'depth_scale': {
        'name': '深度尺度偏差',
        'levels': [1.0, 0.75, 0.5, 0.25, 0.1],
        'unit': '×',
        'description': '系统性深度尺度低估',
    },
    'combined': {
        'name': '组合退化',
        'levels': [0.0, 0.25, 0.5, 0.75, 1.0],
        'unit': 'severity',
        'description': '联合分辨率、深度噪声、深度失效、尺度偏差和视角不确定性',
    },
}


# ─── 退化函数 ───────────────────────────────────────────────────
# 每个函数: 输入观测 → 应用退化 → 输出退化后观测
# WHY 独立函数: 可单独测试、可任意组合、可替换实现

def apply_gaussian_sparsification(obstacles: np.ndarray,
                                   keep_ratio: float,
                                   seed: int = 42) -> np.ndarray:
    """
    高斯球稀疏化: 随机删除障碍物来模拟 (mock实现)

    Args:
        obstacles: (N, 4) 障碍物数组 [x, y, z, radius]
        keep_ratio: 保留比例 (0-1), 100=全保留, 5=仅保留5%
        seed: 随机种子
    Returns:
        稀疏化后的障碍物数组
    """
    if keep_ratio >= 1.0 or len(obstacles) == 0:
        return obstacles.copy()

    n_keep = max(1, int(len(obstacles) * keep_ratio))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(obstacles), n_keep, replace=False)
    return obstacles[idx].copy()


def apply_resolution_downscale(depth: np.ndarray,
                                target_res: int,
                                interpolation: str = 'bilinear') -> np.ndarray:
    """
    分辨率降低: 降采样后上采样回原尺寸 (v2: 支持插值方式选择)

    v2改进: 支持 nearest/bilinear 两种插值；深度保留 float 避免 uint8 量化

    Args:
        depth: (H, W, 1) 深度图
        target_res: 目标分辨率 (如 64→32→16→8→2)
        interpolation: 'nearest' 或 'bilinear'
    Returns:
        (H, W, 1) 模糊后的深度图 (尺寸不变)
    """
    from PIL import Image

    h, w = depth.shape[:2]
    # Pillow does not support bilinear resize for uint16 ``I;16`` images.
    # Mode ``F`` preserves float depth and supports both interpolation modes.
    img = Image.fromarray(depth[..., 0].astype(np.float32), mode='F')
    
    # 选择插值方式
    pil_interp = Image.NEAREST if interpolation.lower() == 'nearest' else Image.BILINEAR
    
    # 降采样 → 上采样
    img_small = img.resize((target_res, target_res), pil_interp)
    img_back = img_small.resize((w, h), pil_interp)
    
    result = np.array(img_back, dtype=np.float32)
    return result[..., np.newaxis]


def apply_lighting_offset(rgb: np.ndarray, ev_offset: float) -> np.ndarray:
    """
    光照偏移: 对RGB做EV调整

    WHY 影响深度: 在真实3DGS中，光照变化会影响深度估计的置信度；
                 当前mock中RGB为零，此函数仅预留接口

    Args:
        rgb: (H, W, 3) RGB图像
        ev_offset: EV偏移量 (0=无变化, +1=两倍亮, -1=一半亮)
    Returns:
        调整后的RGB图像
    """
    return rgb * (2.0 ** ev_offset)


def apply_perlin_depth_noise(depth: np.ndarray,
                              sigma: float,
                              scale: int = 4,
                              seed: int = 42) -> np.ndarray:
    """
    深度噪声: 空间相关噪声 (v2增强版)

    WHY 空间相关: 独立像素噪声过强且不现实；空间相关噪声更接近
                  真实3DGS渲染的几何误差分布

    v2增强: 多频叠加 + 距离相关衰减 + 结构性空洞 + 局部偏置/尺度漂移

    Args:
        depth: (H, W, 1) 深度图
        sigma: 噪声标准差 (0~1.0m)
        scale: 噪声空间频率 (越小越粗糙)
        seed: 随机种子 (用于结构性空洞生成)
    Returns:
        (H, W, 1) 加噪深度图
    """
    if sigma <= 0:
        return depth

    h, w = depth.shape[:2]
    rng = np.random.RandomState(seed)
    
    # 1. 多频Perlin-like噪声基础
    xs, ys = np.mgrid[0:h, 0:w] / scale
    base_noise = (np.sin(xs * 1.7) * np.cos(ys * 2.3) +
                  np.sin(xs * 0.7 + ys * 1.1) * 0.5 +
                  np.cos(xs * 2.9 - ys * 0.8) * 0.3)
    base_noise = base_noise / base_noise.std()
    
    # 2. 距离相关衰减 (远处噪声更大 — 模拟深度不确定性随距离增加)
    cx, cy = w / 2, h / 2
    distance_field = np.sqrt((xs * scale - cx)**2 + (ys * scale - cy)**2) / np.sqrt(cx**2 + cy**2)
    distance_modulation = 0.5 + 0.5 * distance_field  # [0.5, 1.0]
    
    # 3. 结构性空洞 (局部深度缺失，模拟GS重建失败的区域)
    n_holes = max(1, int(sigma * 5))  # sigma越大空洞越多
    hole_mask = np.ones((h, w), dtype=np.float32)
    for _ in range(n_holes):
        hole_cx = rng.randint(h // 4, 3 * h // 4)
        hole_cy = rng.randint(w // 4, 3 * w // 4)
        hole_rad = int(5 + sigma * 10)  # 半径 5~15 像素
        yy, xx = np.ogrid[:h, :w]
        hole_dist = np.sqrt((yy - hole_cx)**2 + (xx - hole_cy)**2)
        hole_mask[hole_dist < hole_rad] *= (1 - sigma * 0.3)  # 强度与sigma相关
    
    # 4. 局部尺度漂移 (某些区域整体偏置，不仅是噪声)
    local_bias = np.zeros((h, w), dtype=np.float32)
    n_bias = max(1, int(sigma * 3))
    for _ in range(n_bias):
        bias_cx = rng.randint(h // 6, 5 * h // 6)
        bias_cy = rng.randint(w // 6, 5 * w // 6)
        bias_rad = int(10 + sigma * 15)
        yy, xx = np.ogrid[:h, :w]
        bias_dist = np.sqrt((yy - bias_cx)**2 + (xx - bias_cy)**2)
        bias_strength = sigma * 0.2 * np.exp(-(bias_dist ** 2) / (bias_rad ** 2))
        local_bias += bias_strength
    
    # 5. 边缘flying pixels (深度突跳，高频成分)
    edge_noise = rng.randn(h // 4, w // 4) * sigma * 0.15
    edge_noise_large = np.kron(edge_noise, np.ones((4, 4)))[:h, :w]
    
    # 组合所有噪声源
    total_noise = (base_noise * distance_modulation * hole_mask + 
                   local_bias + 
                   edge_noise_large)
    total_noise = total_noise / (total_noise.std() + 1e-7) * sigma
    
    return np.clip(depth + total_noise.reshape(h, w, 1), 0.1, 20.0)


def apply_viewpoint_restriction(depth: np.ndarray,
                                 coverage_deg: float,
                                 seed: int = 42) -> np.ndarray:
    """
    视角覆盖限制: 模拟有限视角下的深度图退化

    WHY: 3DGS重建质量与训练视角覆盖正相关。视角不足时，
         未观测区域的深度推测不可靠 → 增加深度不确定性

    Mock实现: 在深度图上叠加与缺失视角成正比的空间相关不确定性
    - 360° = 全向 → 无退化
    - 45°  = 极度受限 → 强噪声 + 外围模糊

    Args:
        depth: (H, W, 1) 深度图
        coverage_deg: 视角覆盖角度 (360=全向, 越小越受限)
        seed: 随机种子
    Returns:
        (H, W, 1) 退化深度图
    """
    if coverage_deg >= 360:
        return depth

    h, w = depth.shape[:2]
    rng = np.random.RandomState(seed)

    # 缺失比例 → 退化强度
    missing_ratio = 1.0 - (coverage_deg / 360.0)

    # 外围模糊: 图像边缘的深度不确定性更高
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    edge_dist = np.sqrt((xs - cx)**2 + (ys - cy)**2) / (np.sqrt(cx**2 + cy**2))
    blur_mask = edge_dist * missing_ratio  # 中心清晰，边缘模糊

    # 空间相关噪声 (低频 → 模拟未观测区域的大片不确定)
    coarse_noise = (np.sin(xs * 0.5 + ys * 0.3) * np.cos(ys * 0.6 - xs * 0.4))
    coarse_noise = coarse_noise / coarse_noise.std()

    # 组合退化
    noise_magnitude = blur_mask * missing_ratio * 3.0  # max ~3m uncertainty
    degraded = depth + (coarse_noise * noise_magnitude).reshape(h, w, 1)
    return np.clip(degraded, 0.1, 20.0)


def apply_depth_failure(depth: np.ndarray, failure_percent: float,
                        seed: int = 42) -> np.ndarray:
    """Replace deterministic rectangular regions with maximum-range depth."""
    ratio = float(np.clip(failure_percent / 100.0, 0.0, 1.0))
    if ratio <= 0:
        return depth
    h, w = depth.shape[:2]
    rng = np.random.RandomState(seed)
    mask = np.zeros((h, w), dtype=bool)
    target = int(round(ratio * h * w))
    while int(mask.sum()) < target:
        block_h = rng.randint(max(2, h // 8), max(3, h // 2))
        block_w = rng.randint(max(2, w // 8), max(3, w // 2))
        y = rng.randint(0, max(1, h - block_h + 1))
        x = rng.randint(0, max(1, w - block_w + 1))
        mask[y:y + block_h, x:x + block_w] = True
    # Trim excess deterministically so the requested percentage is exact.
    indices = np.flatnonzero(mask)
    if len(indices) > target:
        mask.flat[indices[target:]] = False
    result = depth.copy()
    result[..., 0][mask] = 20.0
    return result


def apply_occlusion(depth: np.ndarray, occlusion_percent: float) -> np.ndarray:
    """Apply a near-field foreground occluder covering the requested area."""
    ratio = float(np.clip(occlusion_percent / 100.0, 0.0, 1.0))
    if ratio <= 0:
        return depth
    h, w = depth.shape[:2]
    occluded_rows = min(h, int(round(h * ratio)))
    result = depth.copy()
    # Bottom-up obstruction models a lens/body obstruction while preserving
    # a shrinking upper field of view.
    result[h - occluded_rows:, :, 0] = 0.1
    return result


def apply_depth_scale_bias(depth: np.ndarray, scale: float) -> np.ndarray:
    """Apply a global multiplicative depth calibration error."""
    return np.clip(depth * float(scale), 0.1, 20.0)


def apply_combined_degradation(depth: np.ndarray, severity: float,
                               seed: int = 42) -> np.ndarray:
    """Apply a calibrated mixture of all depth-affecting degradations."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity <= 0:
        return depth
    target_res = max(1, int(round(64 * (1.0 - severity) + severity)))
    result = apply_resolution_downscale(depth, target_res)
    result = apply_perlin_depth_noise(result, sigma=4.0 * severity, seed=seed)
    result = apply_depth_failure(result, failure_percent=80.0 * severity,
                                 seed=seed)
    result = apply_depth_scale_bias(result, scale=1.0 - 0.75 * severity)
    result = apply_viewpoint_restriction(
        result, coverage_deg=360.0 - 345.0 * severity, seed=seed)
    return result


# ─── 退化应用器 ──────────────────────────────────────────────────
# WHY 统一入口: 环境只需调一个函数，退化组合由配置控制
# 边界: 不修改renderer内部状态（gaussian除外，它需要修改obstacles）

def apply_degradation_pipeline(depth: np.ndarray,
                                rgb: np.ndarray,
                                obstacles: np.ndarray,
                                deg_config: Dict[str, Any],
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按配置依次应用所有退化

    Args:
        depth: (H, W, 1) 深度图
        rgb: (H, W, 3) RGB图像
        obstacles: (N, 4) 障碍物数组 (用于gaussian退化)
        deg_config: {'gaussian': 50, 'resolution': 32, ...}
        seed: 随机种子
    Returns:
        (degraded_depth, degraded_rgb, degraded_obstacles)
    """
    if not deg_config:
        return depth, rgb, obstacles

    degraded_depth = depth
    degraded_rgb = rgb
    degraded_obstacles = obstacles

    # 1. 高斯稀疏化 (修改障碍物 → 影响后续render)
    if 'gaussian' in deg_config:
        ratio = deg_config['gaussian'] / 100.0
        degraded_obstacles = apply_gaussian_sparsification(
            degraded_obstacles, ratio, seed)

    # 2. 分辨率降低
    if 'resolution' in deg_config:
        degraded_depth = apply_resolution_downscale(
            degraded_depth, deg_config['resolution'])

    # 3. 深度噪声
    if 'depth_noise' in deg_config:
        degraded_depth = apply_perlin_depth_noise(
            degraded_depth, deg_config['depth_noise'])

    # 4. 光照偏移 (影响RGB → 间接影响深度置信度)
    if 'lighting' in deg_config:
        degraded_rgb = apply_lighting_offset(degraded_rgb, deg_config['lighting'])

    # 5. 视角覆盖
    if 'viewpoint_uncertainty' in deg_config:
        degraded_depth = apply_viewpoint_restriction(
            degraded_depth, deg_config['viewpoint_uncertainty'], seed)

    # 6–9. Structural depth failures used after the original Phase V2 suite.
    if 'depth_failure' in deg_config:
        degraded_depth = apply_depth_failure(
            degraded_depth, deg_config['depth_failure'], seed)
    if 'occlusion' in deg_config:
        degraded_depth = apply_occlusion(
            degraded_depth, deg_config['occlusion'])
    if 'depth_scale' in deg_config:
        degraded_depth = apply_depth_scale_bias(
            degraded_depth, deg_config['depth_scale'])
    if 'combined' in deg_config:
        degraded_depth = apply_combined_degradation(
            degraded_depth, deg_config['combined'], seed)

    return degraded_depth, degraded_rgb, degraded_obstacles


# ── 自检 ──
if __name__ == "__main__":
    print("=" * 50)
    print("Degradation Utils Self-Check")
    print("=" * 50)

    depth = np.random.rand(64, 64, 1).astype(np.float32) * 10
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    obstacles = np.array([
        [5.0, 0.0, 3.0, 1.5],
        [8.0, 2.0, 4.0, 1.2],
    ])

    # 测试全部5轴
    full_config = {
        'gaussian': 50,
        'resolution': 16,
        'depth_noise': 0.05,
        'lighting': 2,
        'viewpoint_uncertainty': 180,
    }
    dd, dr, do = apply_degradation_pipeline(depth, rgb, obstacles, full_config)
    print(f"gaussian: {len(obstacles)}→{len(do)} obstacles")
    print(f"resolution: depth shape {dd.shape} (unchanged)")
    print(f"depth_noise: std {dd.std():.3f} (was {depth.std():.3f})")
    print(f"lighting: rgb max {dr.max():.3f}")
    print(f"viewpoint: depth range [{dd.min():.1f}, {dd.max():.1f}]")
    print(f"depth stats: min={dd.min():.2f} max={dd.max():.2f} mean={dd.mean():.2f}")

    # 测试空配置
    dd2, dr2, do2 = apply_degradation_pipeline(depth, rgb, obstacles, {})
    assert np.allclose(depth, dd2), "empty config should be no-op"
    print("empty config: no-op ✓")

    # 测试5轴各有默认level
    for axis_name, axis_info in DEGRADATION_AXES.items():
        levels = axis_info['levels']
        assert len(levels) == 5, f"{axis_name}: expected 5 levels, got {len(levels)}"
        print(f"  {axis_name:12s}: {levels} {axis_info['unit']}")

    print(f"\n{len(DEGRADATION_AXES)} axes × 5 levels — all OK")
    print("Degradation Utils self-check passed")
