"""
带传感器噪声的无人机环境
噪声仅加在观测输出上，不影响内部物理状态（即 step() 中的动力学计算始终使用真实状态）

5种噪声模式：
  - pos    : 仅位置观测噪声 (dim 0-2)
  - vel    : 仅速度观测噪声 (dim 3-5)
  - target : 仅目标相对位置+距离噪声 (dim 6-9)
  - obs    : 仅障碍物方向+距离噪声 (dim 10-13)
  - full   : 全部14维观测噪声
"""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.drone_env import DroneEnv
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 5种噪声模式的维度掩码定义
# ============================================================
NOISE_PATTERNS: Dict[str, Dict[str, float]] = {
    'pos': {
        'sigma_pos': 1.0,          # dim 0-2: 位置
    },
    'vel': {
        'sigma_vel': 1.0,          # dim 3-5: 速度
    },
    'target': {
        'sigma_target': 1.0,       # dim 6-8: 目标相对位置
        'sigma_target_dist': 1.0,  # dim 9:   目标距离
    },
    'obs': {
        'sigma_obs_dir': 1.0,      # dim 10-12: 障碍物方向
        'sigma_obs_dist': 1.0,     # dim 13:    障碍物距离
    },
    'full': {
        'sigma_pos': 1.0,
        'sigma_vel': 1.0,
        'sigma_target': 1.0,
        'sigma_target_dist': 1.0,
        'sigma_obs_dir': 1.0,
        'sigma_obs_dist': 1.0,
    },
}

# 每个模式影响的维度范围（用于测试验证）
NOISE_PATTERN_DIMS: Dict[str, list] = {
    'pos':    [0, 1, 2],
    'vel':    [3, 4, 5],
    'target': [6, 7, 8, 9],
    'obs':    [10, 11, 12, 13],
    'full':   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
}


class NoisyDroneEnv(DroneEnv):
    """
    带传感器噪声的无人机路径规划环境

    噪声注入策略：
    - 内部状态 self.state 保持纯净（step() 使用真实物理）
    - 噪声仅在 _get_observation() 中叠加在 14 维观测向量上
    - 这模拟了真实场景：物理世界是确定的，但传感器读数有误差

    支持两种构造方式：
    1. 精细控制：NoisyDroneEnv(noise_config={'sigma_pos': 0.5, 'sigma_vel': 0.2})
    2. 模式化：  NoisyDroneEnv.from_pattern('pos', sigma=0.5)

    noise_config 完整键名：
        sigma_pos        : 位置观测噪声 std (dim 0-2),  默认 0.0
        sigma_vel        : 速度观测噪声 std (dim 3-5),  默认 0.0
        sigma_target     : 目标相对位置噪声 std (dim 6-8), 默认 0.0
        sigma_target_dist: 目标距离噪声 std (dim 9),    默认 0.0
        sigma_obs_dir    : 障碍物方向噪声 std (dim 10-12), 默认 0.0
        sigma_obs_dist   : 障碍物距离噪声 std (dim 13), 默认 0.0
    """

    # 14 维观测向量的语义分组
    DIM_POS         = slice(0, 3)     # 位置 [x, y, z]
    DIM_VEL         = slice(3, 6)     # 速度 [vx, vy, vz]
    DIM_TARGET      = slice(6, 9)     # 目标相对位置 [dx, dy, dz]
    DIM_TARGET_DIST = 9               # 目标距离
    DIM_OBS_DIR     = slice(10, 13)   # 最近障碍物方向 [ox, oy, oz]
    DIM_OBS_DIST    = 13              # 最近障碍物距离

    # ---- 公开属性：维度分组名称（供外部脚本引用） ----
    DIM_NAMES = [
        'pos_x', 'pos_y', 'pos_z',           # 0-2
        'vel_x', 'vel_y', 'vel_z',           # 3-5
        'tgt_dx', 'tgt_dy', 'tgt_dz',        # 6-8
        'tgt_dist',                           # 9
        'obs_dx', 'obs_dy', 'obs_dz',        # 10-12
        'obs_dist',                           # 13
    ]

    # ---- 噪声模式注册表（公开，供外部查询） ----
    PATTERNS = list(NOISE_PATTERNS.keys())        # ['pos', 'vel', 'target', 'obs', 'full']
    PATTERN_DIMS = NOISE_PATTERN_DIMS              # 每个模式控制的维度索引

    @classmethod
    def from_pattern(cls, pattern: str, sigma: float = 1.0, **env_kwargs):
        """
        工厂方法：通过噪声模式名创建环境

        Args:
            pattern: 噪声模式名，必须是 'pos', 'vel', 'target', 'obs', 'full' 之一
            sigma:   噪声标准差（对所有相关维度统一使用此值）
            **env_kwargs: 传递给 DroneEnv.__init__ 的参数（如 config）

        Returns:
            NoisyDroneEnv 实例

        Raises:
            ValueError: pattern 不在 NOISE_PATTERNS 中

        Example:
            >>> env = NoisyDroneEnv.from_pattern('pos', sigma=0.5)
            >>> env = NoisyDroneEnv.from_pattern('full', sigma=0.3)
        """
        if pattern not in NOISE_PATTERNS:
            raise ValueError(
                f"未知噪声模式 '{pattern}'，可选: {list(NOISE_PATTERNS.keys())}"
            )

        # 复制模板并缩放 sigma
        template = NOISE_PATTERNS[pattern]
        noise_config = {k: v * sigma for k, v in template.items()}

        logger.info(f"从模式创建: pattern='{pattern}', sigma={sigma} → {noise_config}")
        return cls(noise_config=noise_config, **env_kwargs)

    def __init__(self, noise_config: Dict[str, float] = None, **kwargs):
        super().__init__(**kwargs)

        self.noise_config = noise_config or {}
        self._current_pattern: Optional[str] = None  # 当前活跃模式名（通过 set_pattern 设置）

        # 构建噪声标准差向量 (14维)
        self.noise_std = np.zeros(14, dtype=np.float32)
        self._rebuild_noise_std()

        active = [(k, v) for k, v in self.noise_config.items() if v > 0]
        if active:
            logger.info(f"NoisyDroneEnv 初始化 | 活跃噪声维度: {active}")
        else:
            logger.info("NoisyDroneEnv 初始化 | 无噪声 (等效于 DroneEnv)")

    def _rebuild_noise_std(self):
        """根据 noise_config 重建 14 维噪声标准差向量"""
        self.noise_std[self.DIM_POS] = self.noise_config.get('sigma_pos', 0.0)
        self.noise_std[self.DIM_VEL] = self.noise_config.get('sigma_vel', 0.0)
        self.noise_std[self.DIM_TARGET] = self.noise_config.get('sigma_target', 0.0)
        self.noise_std[self.DIM_TARGET_DIST] = self.noise_config.get('sigma_target_dist', 0.0)
        self.noise_std[self.DIM_OBS_DIR] = self.noise_config.get('sigma_obs_dir', 0.0)
        self.noise_std[self.DIM_OBS_DIST] = self.noise_config.get('sigma_obs_dist', 0.0)

    def _get_observation(self):
        """
        在父类纯净观测基础上加高斯噪声
        关键设计：父类的 _build_state_vector() 使用 self.state（真实物理状态）
        噪声只加在返回给 agent 的观测上，不影响动力学
        """
        clean_obs = super()._get_observation()

        # 生成 14 维独立高斯噪声
        noise = np.random.normal(0.0, self.noise_std).astype(np.float32)
        noisy_obs = clean_obs + noise

        return noisy_obs

    def get_clean_observation(self):
        """调试用：获取纯净观测（绕过噪声）"""
        return super()._get_observation()

    def get_active_dims(self) -> list:
        """返回当前有噪声注入的维度索引列表"""
        return [i for i, s in enumerate(self.noise_std) if s > 0]

    def set_pattern(self, pattern: str, sigma: float = 1.0):
        """
        运行时动态切换噪声模式

        Args:
            pattern: 噪声模式名
            sigma:   噪声标准差
        """
        if pattern not in NOISE_PATTERNS:
            raise ValueError(
                f"未知噪声模式 '{pattern}'，可选: {list(NOISE_PATTERNS.keys())}"
            )

        template = NOISE_PATTERNS[pattern]
        self.noise_config = {k: v * sigma for k, v in template.items()}
        self._current_pattern = pattern
        self._rebuild_noise_std()

        logger.info(f"切换噪声模式: pattern='{pattern}', sigma={sigma}")

    def set_noise(self, **kwargs):
        """
        运行时动态调整单个噪声水平（精细控制）
        用法: env.set_noise(sigma_pos=0.5, sigma_vel=0.2)
        """
        for key, value in kwargs.items():
            self.noise_config[key] = value
        self._current_pattern = None  # 精细调整后不再是预定义模式
        self._rebuild_noise_std()

    # ---- 便捷属性：查询当前各维度噪声 std ----
    @property
    def sigma_pos(self) -> float:
        return self.noise_config.get('sigma_pos', 0.0)

    @property
    def sigma_vel(self) -> float:
        return self.noise_config.get('sigma_vel', 0.0)

    @property
    def sigma_target(self) -> float:
        return self.noise_config.get('sigma_target', 0.0)

    @property
    def sigma_target_dist(self) -> float:
        return self.noise_config.get('sigma_target_dist', 0.0)

    @property
    def sigma_obs_dir(self) -> float:
        return self.noise_config.get('sigma_obs_dir', 0.0)

    @property
    def sigma_obs_dist(self) -> float:
        return self.noise_config.get('sigma_obs_dist', 0.0)


# ============================================================
# 测试代码：每种噪声模式单独测试
# ============================================================
def _test_pattern_distribution(pattern: str, sigma: float = 0.5, n_samples: int = 10000):
    """
    单模式噪声分布验证：
    1. 用指定模式创建环境
    2. 多次采样 clean/noisy 观测
    3. 验证：噪声维度标准差 ≈ sigma，非噪声维度标准差 ≈ 0
    """
    print(f"\n{'='*70}")
    print(f"测试 [{pattern}] 模式 | sigma={sigma} | {n_samples} 次采样")
    print(f"{'='*70}")

    env = NoisyDroneEnv.from_pattern(pattern, sigma=sigma)
    active_dims = NOISE_PATTERN_DIMS[pattern]
    all_dims = list(range(14))
    inactive_dims = [d for d in all_dims if d not in active_dims]

    print(f"  活跃维度 (应有噪声): {active_dims}")
    print(f"  非活跃维度 (应无噪声): {inactive_dims}")

    # 收集多次采样的噪声
    noise_samples = np.zeros((n_samples, 14), dtype=np.float32)
    for i in range(n_samples):
        env.reset(seed=i)
        noisy = env._get_observation()
        clean = env.get_clean_observation()
        noise_samples[i] = noisy - clean

    # 计算每个维度的噪声标准差
    noise_std_empirical = np.std(noise_samples, axis=0)
    noise_mean_empirical = np.mean(noise_samples, axis=0)

    # 打印结果
    print(f"\n  {'维度':<12} {'名称':<12} {'目标σ':>8} {'实测σ':>8} {'实测μ':>8} {'状态':>10}")
    print(f"  {'-'*58}")
    all_pass = True
    for dim in range(14):
        name = NoisyDroneEnv.DIM_NAMES[dim]
        target = env.noise_std[dim]
        measured_std = noise_std_empirical[dim]
        measured_mean = noise_mean_empirical[dim]

        if dim in active_dims:
            # 活跃维度：标准差应接近 sigma（允许 5% 容差）
            std_ok = abs(measured_std - target) / target < 0.05 if target > 0 else True
            mean_ok = abs(measured_mean) < 0.05  # 均值应接近 0
            status = "PASS" if (std_ok and mean_ok) else "FAIL"
            if not (std_ok and mean_ok):
                all_pass = False
        else:
            # 非活跃维度：标准差应接近 0
            std_ok = measured_std < 1e-6
            mean_ok = abs(measured_mean) < 1e-6
            status = "PASS" if (std_ok and mean_ok) else "FAIL"
            if not (std_ok and mean_ok):
                all_pass = False

        print(f"  {dim:<12} {name:<12} {target:8.4f} {measured_std:8.4f} {measured_mean:+8.4f} {status:>10}")

    env.close()

    if all_pass:
        print(f"\n  [PASS] [{pattern}] 模式测试通过：噪声分布正确")
    else:
        print(f"\n  [FAIL] [{pattern}] 模式测试失败")

    return all_pass


def _test_pattern_isolation():
    """
    验证不同模式之间互不干扰：
    对每种模式，确认噪声仅注入目标维度，其他维度完全无噪声
    """
    print(f"\n{'='*70}")
    print("交叉验证：各模式噪声隔离性")
    print(f"{'='*70}")

    n_samples = 5000
    sigma = 0.5
    results = {}

    for pattern in NOISE_PATTERNS.keys():
        env = NoisyDroneEnv.from_pattern(pattern, sigma=sigma)
        active = set(NOISE_PATTERN_DIMS[pattern])
        inactive = set(range(14)) - active

        noise_samples = np.zeros((n_samples, 14), dtype=np.float32)
        for i in range(n_samples):
            env.reset(seed=i)
            noisy = env._get_observation()
            clean = env.get_clean_observation()
            noise_samples[i] = noisy - clean

        noise_std = np.std(noise_samples, axis=0)

        # 活跃维度：std 应 > 0.9*sigma
        active_ok = all(noise_std[d] > 0.9 * sigma for d in active)
        # 非活跃维度：std 应 < 1e-6
        inactive_ok = all(noise_std[d] < 1e-6 for d in inactive) if inactive else True

        results[pattern] = active_ok and inactive_ok

        active_min = min(noise_std[list(active)]) if active else 0.0
        inactive_max = max(noise_std[list(inactive)]) if inactive else 0.0
        print(f"  {pattern:8s} | 活跃维度 {sorted(active)}: "
              f"min_σ={active_min:.4f} (目标 {sigma}) | "
              f"非活跃 max_σ={inactive_max:.2e} | "
              f"{'PASS' if results[pattern] else 'FAIL'}")

        env.close()

    all_ok = all(results.values())
    print(f"\n  {'[PASS] 隔离性测试通过' if all_ok else '[FAIL] 隔离性测试失败'}")
    return all_ok


def _test_episode_with_pattern(pattern: str, sigma: float = 0.5):
    """用指定噪声模式运行一个完整 episode（功能测试）"""
    print(f"\n  [{pattern}] 运行 episode...", end=" ")

    env = NoisyDroneEnv.from_pattern(pattern, sigma=sigma)
    obs, info = env.reset(seed=42)
    ep_reward = 0.0
    step = 0

    while True:
        # 随机动作（不做推理，只验证环境运行正常）
        action = np.random.uniform(-1, 1, 3).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        step += 1

        # 验证：有噪声的维度在 noisy obs 中确实偏离 clean obs
        if step == 1:
            clean = env.get_clean_observation()
            active_dims = NOISE_PATTERN_DIMS[pattern]
            for d in active_dims:
                assert obs[d] != clean[d], \
                    f"[{pattern}] 活跃维度 {d} 无噪声注入: noisy={obs[d]:.4f} == clean={clean[d]:.4f}"

            inactive_dims = [d for d in range(14) if d not in active_dims]
            for d in inactive_dims:
                assert np.isclose(obs[d], clean[d], atol=1e-6), \
                    f"[{pattern}] 非活跃维度 {d} 意外注入噪声: noisy={obs[d]:.4f} != clean={clean[d]:.4f}"

        if terminated or truncated:
            break

    env.close()
    result = "success" if info.get('reached_target') else \
             "collision" if info.get('collision') else "timeout"
    print(f"完成 | steps={step} | reward={ep_reward:.2f} | {result}")
    return True


def main():
    """完整测试套件：5种噪声模式分布 + 隔离性 + episode 运行"""
    import sys

    print("=" * 70)
    print("NoisyDroneEnv — 5种噪声模式完整测试")
    print("=" * 70)

    all_passed = True

    # ===== Part 1: 分布测试（大规模采样验证） =====
    print("\n" + "#" * 70)
    print("█  Part 1: 噪声分布统计测试（10K 采样/模式）")
    print("#" * 70)
    for pattern in NOISE_PATTERNS.keys():
        if not _test_pattern_distribution(pattern, sigma=0.5, n_samples=10000):
            all_passed = False

    # ===== Part 2: 隔离性交叉验证 =====
    print("\n" + "#" * 70)
    print("█  Part 2: 模式隔离性交叉验证")
    print("#" * 70)
    if not _test_pattern_isolation():
        all_passed = False

    # ===== Part 3: Episode 功能测试 =====
    print("\n" + "#" * 70)
    print("█  Part 3: 各模式 Episode 功能测试")
    print("#" * 70)
    for pattern in NOISE_PATTERNS.keys():
        try:
            _test_episode_with_pattern(pattern, sigma=0.5)
        except Exception as e:
            print(f"\n  [FAIL] [{pattern}] Episode 测试异常: {e}")
            all_passed = False

    # ===== Part 4: set_pattern() 运行时切换测试 =====
    print("\n" + "#" * 70)
    print("█  Part 4: 运行时模式切换 set_pattern() 测试")
    print("#" * 70)

    env = NoisyDroneEnv.from_pattern('pos', sigma=0.3)
    assert env._current_pattern is None  # from_pattern 不设置 _current_pattern
    assert env.sigma_pos == 0.3
    assert env.sigma_vel == 0.0

    # 切换到 vel 模式
    env.set_pattern('vel', sigma=0.7)
    assert env.sigma_vel == 0.7
    assert env.sigma_pos == 0.0
    assert env._current_pattern == 'vel'

    # 切换到 full 模式
    env.set_pattern('full', sigma=0.2)
    assert env.sigma_pos == 0.2
    assert env.sigma_vel == 0.2
    assert env.sigma_target == 0.2

    env.close()
    print("  [PASS] set_pattern() 运行时切换测试通过")

    # ===== Part 5: 工厂方法错误处理 =====
    print("\n" + "#" * 70)
    print("█  Part 5: 错误处理测试")
    print("#" * 70)
    try:
        NoisyDroneEnv.from_pattern('invalid_pattern', sigma=0.5)
        print("  [FAIL] 应抛出 ValueError")
        all_passed = False
    except ValueError as e:
        print(f"  [PASS] 正确拒绝未知模式: {e}")

    # ===== 汇总 =====
    print("\n" + "=" * 70)
    if all_passed:
        print("[PASS] 全部测试通过：5种噪声模式实现正确")
    else:
        print("[FAIL] 部分测试失败，请检查输出")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
