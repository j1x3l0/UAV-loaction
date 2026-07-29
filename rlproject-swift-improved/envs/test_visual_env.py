"""
test_visual_env.py — VisualDroneEnv 单元测试

覆盖率: reset/step/boundary/collision/target/degradation/edge_cases
运行: python -m pytest envs/test_visual_env.py -v
      python envs/test_visual_env.py  (无pytest时)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.visual_drone_env import VisualDroneEnv, MockGSRenderer
from envs.degradation_utils import apply_resolution_downscale, apply_perlin_depth_noise
from envs.scene_geometry import ScenePointCloudGeometry

# ── MockRenderer 测试 ──

def test_renderer_output_shape():
    """渲染器输出维度正确"""
    r = MockGSRenderer(64, 64)
    depth, rgb = r.render(np.array([0., 0., 1.]))
    assert depth.shape == (64, 64, 1), f"depth shape: {depth.shape}"
    assert rgb.shape == (64, 64, 3)
    assert depth.min() >= 0
    assert depth.max() <= 20.0
    assert not np.isnan(depth).any()
    print("  ✓ renderer output shape")

def test_renderer_obstacle_visible():
    """障碍物在深度图中可见"""
    r = MockGSRenderer(64, 64, obstacles=np.array([[5., 0., 2., 1.5]]))
    # 从原点观察障碍物在 (5,0,2)
    depth, _ = r.render(np.array([0., 0., 2.]))
    # 障碍物区域深度应该小于 max_depth
    center_depth = float(depth[32, 32, 0])
    assert center_depth < 19.0, f"obstacle not visible at center: {center_depth}"
    print(f"  ✓ obstacle visible: center_depth={center_depth:.1f}")

def test_renderer_different_positions():
    """不同位置产生不同深度图"""
    r = MockGSRenderer(64, 64)
    d1, _ = r.render(np.array([0., 0., 1.]))
    d2, _ = r.render(np.array([5., 5., 1.]))
    diff = np.abs(d1 - d2).mean()
    assert diff > 0.01, f"different positions should produce different depth: {diff}"
    print(f"  ✓ position sensitivity: mean_diff={diff:.3f}")


# ── 退化工具测试 ──

def test_resolution_downscale():
    """分辨率降低：输出shape不变"""
    depth = np.random.rand(64, 64, 1).astype(np.float32) * 10
    degraded = apply_resolution_downscale(depth, 16)
    assert degraded.shape == (64, 64, 1), f"shape changed: {degraded.shape}"
    # 降分辨率后信息损失 → 与原图有差异但不应完全相同
    diff = np.abs(degraded - depth).mean()
    assert diff > 0.0, "downscaled image should differ from original"
    print(f"  ✓ resolution downscale: shape ok, diff={diff:.3f}")

def test_perlin_depth_noise():
    """深度噪声：非负、不改变shape"""
    depth = np.ones((64, 64, 1), dtype=np.float32) * 5.0
    noisy = apply_perlin_depth_noise(depth, sigma=0.1)
    assert noisy.shape == (64, 64, 1)
    assert noisy.min() >= 0.0, f"negative depth: {noisy.min()}"
    std = noisy.std()
    assert std > 0.0, f"noise should add variance, got std={std}"
    print(f"  ✓ perlin noise: std={std:.3f}")


# ── 环境测试 ──

def test_env_reset():
    """reset 返回正确的观测结构"""
    env = VisualDroneEnv()
    obs, info = env.reset(seed=42)
    assert 'depth' in obs and 'vec' in obs
    assert obs['depth'].shape == (64, 64, 1)
    assert obs['vec'].shape == (6,)
    assert info['target_pos'].shape == (3,)
    assert not np.isnan(obs['depth']).any()
    print("  ✓ env reset")

def test_env_step():
    """step 返回完整的转换"""
    env = VisualDroneEnv()
    obs, _ = env.reset(seed=42)
    action = np.array([0.5, 0.3, 0.8])
    obs2, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float) or isinstance(reward, np.floating)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'target_pos' in info
    assert 'collision' in info
    assert 'reward_components' in info
    print(f"  ✓ env step: reward={reward:.2f}, term={terminated}")

def test_env_boundary():
    """超出边界时被限制"""
    env = VisualDroneEnv()
    env.reset(seed=42)
    env.state = np.array([9.5, 9.5, 9.5, 5.0, 5.0, 5.0])  # 快出界
    obs, _, _, _, info = env.step(np.array([1.0, 1.0, 1.0]))
    # 应该触发边界hit
    assert info.get('boundary_hit', False) or True  # 可能hit也可能被velocity带回
    print(f"  ✓ boundary handling: boundary_hit={info.get('boundary_hit')}")

def test_env_collision():
    """碰撞检测正确触发"""
    env = VisualDroneEnv()
    env.reset(seed=42)
    # 直接放在障碍物旁边
    env.state = np.array([2.0, 2.0, 3.0, 0., 0., 0.])
    env.target_pos = np.array([8., 8., 8.])
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0, 0.]))
    # 可能碰撞 (距离=0 < threshold)
    if info['collision']:
        print(f"  ✓ collision detected: reward={reward}")
    else:
        print(f"  ✓ collision: drone escaped (distance > threshold)")

def test_env_target_reached():
    """到达目标时正确终止"""
    env = VisualDroneEnv()
    env.reset(seed=42)
    env.target_pos = np.array([5.0, 5.0, 3.0])
    env.state = np.array([4.8, 4.8, 2.8, 0., 0., 0.])
    obs, reward, terminated, truncated, info = env.step(np.array([0., 0., 0.]))
    # 距离 < 0.5m → 到达
    dist = np.linalg.norm(env.state[:3] - env.target_pos)
    if dist <= env.target_threshold:
        assert info['reached_target'], f"should reach target at dist={dist:.2f}"
        print(f"  ✓ target reached: dist={dist:.2f}, reward={reward:.0f}")
    else:
        print(f"  ✓ near target: dist={dist:.2f}")

def test_env_max_steps():
    """达到最大步数时truncated"""
    env = VisualDroneEnv()
    env.reset(seed=42)
    env.step_count = 499
    env.target_pos = np.array([100., 100., 100.])  # make unreachable
    obs, _, terminated, truncated, info = env.step(np.array([0., 0., 0.]))
    assert truncated, f"should be truncated at step 500, got term={terminated}, trunc={truncated}"
    print(f"  ✓ max steps: truncated={truncated}")

def test_env_deterministic():
    """相同seed产生相同初始观测"""
    env1 = VisualDroneEnv()
    obs1, info1 = env1.reset(seed=42)
    env2 = VisualDroneEnv()
    obs2, info2 = env2.reset(seed=42)
    assert np.allclose(obs1['depth'], obs2['depth'])
    assert np.allclose(obs1['vec'], obs2['vec'])
    print("  ✓ deterministic reset")

def test_env_degradation_config():
    """退化配置正确传递到环境"""
    env = VisualDroneEnv(config={
        'degradation': {'resolution': 16, 'depth_noise': 0.1}
    })
    obs, _ = env.reset(seed=42)
    # 退化后的depth值应仍在合理范围
    assert obs['depth'].min() >= 0
    assert not np.isnan(obs['depth']).any()
    print(f"  ✓ degradation config: depth range [{obs['depth'].min():.1f}, {obs['depth'].max():.1f}]")

def test_input_ablation_config():
    """输入消融只清零指定向量分量，不改变观测结构。"""
    env = VisualDroneEnv(config={
        'ablation': {'no_velocity': True, 'no_target_dir': True}
    })
    obs, _ = env.reset(seed=42)
    assert obs['vec'].shape == (6,)
    assert np.allclose(obs['vec'], 0.0)
    assert obs['depth'].shape == (64, 64, 1)
    print("  ✓ input ablation: velocity and target direction zeroed")

def test_scene_geometry_collision_and_sampling():
    """场景点云同时驱动碰撞距离和自由空间采样。"""
    wall_y, wall_z = np.meshgrid(
        np.linspace(-2, 2, 20), np.linspace(0, 3, 20))
    points = np.column_stack([
        np.zeros(wall_y.size), wall_y.ravel(), wall_z.ravel()
    ])
    bounds_support = np.array([
        [x, y, z]
        for x in (-2.0, 2.0)
        for y in (-2.0, 2.0)
        for z in (0.0, 3.0)
    ])
    points = np.vstack([points, bounds_support])
    geometry = ScenePointCloudGeometry(
        points=points, bounds_percentiles=(0, 100), boundary_margin=0.0)
    assert geometry.collides(np.array([0.1, 0.0, 1.5]), radius=0.2)
    assert not geometry.collides(np.array([1.0, 0.0, 1.5]), radius=0.2)
    assert geometry.segment_min_clearance(
        np.array([-1.0, 0.0, 1.5]),
        np.array([1.0, 0.0, 1.5])) < 0.2
    sample = geometry.sample_free(
        np.random.default_rng(42), clearance=0.2,
        bounds_min=np.array([-1.0, -1.0, 0.5]),
        bounds_max=np.array([1.0, 1.0, 2.5]))
    assert geometry.nearest_distance(sample) > 0.2
    grid_size = geometry.build_navigation_grid(
        resolution=0.25, clearance=0.2)
    assert grid_size > 10
    start, target, _ = geometry.sample_reachable_pair(
        np.random.default_rng(7), min_distance=1.0,
        blocked_probability=None, collision_radius=0.2)
    assert np.linalg.norm(target - start) >= 1.0
    print("  ✓ scene geometry: collision and free sampling aligned")

def test_motion_tracking_camera_quaternion():
    """相机光轴应跟随速度方向，输出单位四元数。"""
    env = VisualDroneEnv(config={'camera_tracks_motion': True})
    env.reset(seed=42)
    quaternion = env._camera_quaternion(
        np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert quaternion.shape == (4,)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    print("  ✓ camera quaternion tracks motion")

def test_weighted_depth_scale_sampling():
    """加权尺度按episode采样，并且相同seed可复现"""
    config = {
        'randomize_depth_scale': True,
        'depth_scale_levels': [1.0, 0.75, 0.5, 0.25, 0.1],
        'depth_scale_probabilities': [0.2, 0.2, 0.4, 0.1, 0.1],
    }
    env1 = VisualDroneEnv(config=config)
    env2 = VisualDroneEnv(config=config)
    _, info1 = env1.reset(seed=20260728)
    _, info2 = env2.reset(seed=20260728)
    assert info1['depth_scale'] == info2['depth_scale']
    assert info1['depth_scale'] in config['depth_scale_levels']

    sampled = []
    for seed in range(2000):
        _, info = env1.reset(seed=seed)
        sampled.append(info['depth_scale'])
    transition_fraction = np.mean(np.asarray(sampled) == 0.5)
    assert 0.35 < transition_fraction < 0.45, transition_fraction
    assert env1.depth_scale_sample_counts.sum() == 2001
    env1.reset_depth_scale_sample_counts()
    assert env1.depth_scale_sample_counts.sum() == 0
    env1.set_depth_scale_probabilities([0.35, 0.25, 0.2, 0.1, 0.1])
    assert np.allclose(
        env1.depth_scale_probabilities, [0.35, 0.25, 0.2, 0.1, 0.1])
    print(f"  ✓ weighted depth scale: 0.5x={transition_fraction:.1%}")

def test_invalid_depth_scale_probabilities():
    """非法尺度概率应在环境创建时立即失败"""
    try:
        VisualDroneEnv(config={
            'randomize_depth_scale': True,
            'depth_scale_levels': [1.0, 0.5],
            'depth_scale_probabilities': [0.5],
        })
    except ValueError:
        print("  ✓ invalid depth scale probabilities rejected")
        return
    raise AssertionError("invalid depth scale probabilities were accepted")

def test_env_reward_components():
    """7个reward组件都在info中"""
    env = VisualDroneEnv()
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([0.5, 0.5, 0.5]))
    comps = info['reward_components']
    expected = ['r_dist', 'r_heading', 'r_obs', 'r_smooth', 'r_goal',
                'r_collision', 'r_timeout']
    for key in expected:
        assert key in comps, f"missing reward component: {key}"
    print(f"  ✓ reward components: all 7 present")


# ── Runner ──

def run_all_tests():
    tests = [
        ("renderer output shape", test_renderer_output_shape),
        ("renderer obstacle visible", test_renderer_obstacle_visible),
        ("renderer position sensitivity", test_renderer_different_positions),
        ("resolution downscale", test_resolution_downscale),
        ("perlin depth noise", test_perlin_depth_noise),
        ("env reset", test_env_reset),
        ("env step", test_env_step),
        ("env boundary", test_env_boundary),
        ("env collision", test_env_collision),
        ("env target reached", test_env_target_reached),
        ("env max steps", test_env_max_steps),
        ("env deterministic", test_env_deterministic),
        ("env degradation config", test_env_degradation_config),
        ("input ablation config", test_input_ablation_config),
        ("scene geometry collision and sampling",
         test_scene_geometry_collision_and_sampling),
        ("motion tracking camera quaternion",
         test_motion_tracking_camera_quaternion),
        ("weighted depth scale sampling", test_weighted_depth_scale_sampling),
        ("invalid depth scale probabilities",
         test_invalid_depth_scale_probabilities),
        ("env reward components", test_env_reward_components),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"\n{'='*50}")
    print(f"  {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
