# 基于 Swift 的无人机 RL 路径规划 — 研究方案

> 整理时间：2026-07-17
> 基于 Nature 2023 "Champion-level drone racing using deep reinforcement learning" 的后续研究方向

---

## 一、研究方案总览：双轨并行推进

```
Week 1-3  │  子课题A: 感知退化鲁棒性量化                    │
          │  ├─ 噪声注入框架搭建                            │
          │  ├─ 系统性衰减曲线测量                           │
          │  └─ 消融实验: 各维度噪声敏感性                   │
          │                                                │
Week 1-6  │  子课题B: 环境策略用于障碍物导航                  │
          │  ├─ Week 1-2: 环境策略框架搭建(SAC)              │
          │  ├─ Week 3-4: 双策略联合训练                     │
          │  └─ Week 5-6: 泛化评估 + 消融实验                │
          │                                                │
Week 7-8  │  整合 + 论文写作                                 │
```

---

## 二、子课题 A：感知退化下的策略鲁棒性量化

### 2.1 科学问题

PPO 策略对状态估计噪声的脆弱性是否存在**相变现象**——是否存在某一临界噪声水平，超过后策略性能从接近完美突然崩溃到完全失效？

### 2.2 为什么这个方向是空白的

1. **竞速社区的假设是"先做好感知"**：无人机竞速的核心瓶颈在速度（80km/h+）和加速度（4g+）。在这个量级下，感知差 10ms 延迟就撞门了。社区的逻辑是：感知是不可妥协的前提条件，研究"感知不好的情况下怎么办"在竞速场景中缺乏实际意义。

2. **RL 路径规划社区和感知鲁棒性社区几乎没有交集**：物流无人机、巡检无人机飞行速度慢得多（5-15m/s），感知退化是常态（GPS 丢失、光照变化、传感器噪声）。但这个场景的研究者通常不用 RL——他们用经典规划（A*、RRT）或 MPC。

3. **研究范式问题**：做鲁棒性的研究者倾向于形式化方法（H∞ 控制、Lyapunov 稳定性），做 RL 的研究者倾向于最大化奖励。两边用不同的语言、发不同的会议。感知退化下的 RL 策略鲁棒性恰好处于两者之间。

### 2.3 发展前景

这个子课题的价值正在上升。2025 年无人机法规趋严（欧盟 U-space、FAA Remote ID），安全论证成为商用部署的前提。如果你的工作能提出"RL 策略在什么感知条件下仍然安全运行"的系统性分析方法，它有潜力成为一个**被引用较多的基准性工作**。

### 2.4 实验设计

**Phase A1：噪声类型分解（6 组实验）**

| 实验组 | 噪声注入目标 | 噪声模型 | 预期现象 |
|--------|------------|---------|---------|
| A1-Baseline | 无噪声 | — | 98% 成功率 |
| A1-Pos | 位置估计 (x,y,z) | N(0, σ_p²)，σ ∈ [0.1, 0.5, 1.0, 2.0] | 路径振荡，到达时间增加 |
| A1-Vel | 速度估计 (vx,vy,vz) | N(0, σ_v²)，σ ∈ [0.1, 1.0, 3.0] | 控制不平稳，超调 |
| A1-Target | 目标相对位置 | N(0, σ_t²) | 方向错误，绕路 |
| A1-Obs | 障碍物距离+方向 | N(0, σ_o²) | 碰撞率上升 |
| A1-Full | 全部 14 维 | 各维度独立同分布噪声 | 复合效应 |

**Phase A2：噪声感知训练（4 组实验）**

| 实验组 | 训练条件 | 测试条件 | 目的 |
|--------|---------|---------|------|
| A2-Clean | 无噪声 | 噪声 | 上界（当前状态） |
| A2-Fixed | 固定噪声 σ=0.5 | 噪声 | 单点鲁棒性 |
| A2-Rand | 每轮采样 σ ~ U(0, σ_max) | 噪声 | 域随机化 |
| A2-Curric | 课程学习 σ: 0 → σ_max | 噪声 | 渐进式适应 |

**评估指标**：

- 成功率衰减曲线 S(σ)（核心）
- 平均碰撞率、平均路径长度、动作平滑度随 σ 的变化
- 相变临界点 σ_c（定义为 S(σ_c) < 50%）

### 2.5 实现代码（核心框架）

```python
# drone_env_noisy.py — 在 step() 中对观测加噪声
class NoisyDroneEnv(DroneEnv):
    def __init__(self, noise_config=None, **kwargs):
        super().__init__(**kwargs)
        self.noise_config = noise_config or {}

    def _get_observation(self):
        clean_obs = super()._get_observation()
        noise = np.zeros(14, dtype=np.float32)

        # 位置噪声 (dim 0-2)
        sigma_pos = self.noise_config.get('sigma_pos', 0.0)
        noise[0:3] = np.random.normal(0, sigma_pos, 3)

        # 速度噪声 (dim 3-5)
        sigma_vel = self.noise_config.get('sigma_vel', 0.0)
        noise[3:6] = np.random.normal(0, sigma_vel, 3)

        # 目标相对位置噪声 (dim 6-8)
        sigma_target = self.noise_config.get('sigma_target', 0.0)
        noise[6:9] = np.random.normal(0, sigma_target, 3)

        # 障碍物方向噪声 (dim 10-12)
        sigma_obs_dir = self.noise_config.get('sigma_obs_dir', 0.0)
        noise[10:13] = np.random.normal(0, sigma_obs_dir, 3)

        # 障碍物距离噪声 (dim 13)
        sigma_obs_dist = self.noise_config.get('sigma_obs_dist', 0.0)
        noise[13] = np.random.normal(0, sigma_obs_dist)

        return clean_obs + noise
```

---

## 三、子课题 B：环境策略用于障碍物导航

### 3.1 科学问题

能否用第二个 RL 智能体（SAC）自动生成难度适中的障碍物布局，使飞行策略（PPO）在不需人工课程设计的情况下，学会在任意障碍物配置中导航？

### 3.2 为什么这个方向是空白的

Wang et al. (ICRA 2025) 的 Environment Policy 论文 2024 年 10 月才挂在 arXiv 上，到现在不到一年。社区还没有足够时间消化和扩展到新领域。

这是一个**真正的空白**——不是因为它难做或没价值，而是因为它太新了。这是最值得关注的方向类型：一个刚发表的有影响力的方法，还没有被应用到其他领域。

**前景**：最有潜力的方向。Wang et al. 被 ICRA 2025 接收（顶会背书），方法本身泛用性强（不限于竞速），障碍物导航是自然的下一个应用场景。时间窗口好——现在开始做，6-12 个月后投稿时还没有竞争。

### 3.3 系统架构

```
┌─────────────────────────────────────────────────┐
│  环境策略 π_env (SAC)                             │
│  ┌─────────────────────────────────────────────┐ │
│  │ 状态: [PPO成功率(last N episodes),            │ │
│  │        障碍物数量, 平均障碍物密度,             │ │
│  │        当前难度等级, 训练轮数进度]             │ │
│  │ 动作: [障碍物数量增减 Δn,                     │ │
│  │        障碍物速度范围, 障碍物分布半径,          │ │
│  │        动态障碍物比例]                        │ │
│  │ 奖励: 最高当 PPO成功率 ∈ [50%, 80%]            │ │
│  └─────────────────────────────────────────────┘ │
│                     │ 生成场景                     │
│                     ▼                             │
│  ┌─────────────────────────────────────────────┐ │
│  │ 飞行策略 π_fly (PPO)                         │ │
│  │ 状态: 14维向量 (含目标+障碍物信息)             │ │
│  │ 动作: 3维推力指令                            │ │
│  │ 奖励: 7组件分层奖励                          │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 3.4 实验设计

**Phase B1：环境策略验证（4 组实验）**

| 实验组 | 训练方式 | 测试方式 |
|--------|---------|---------|
| B1-Fixed3 | 固定 3 个静态障碍物（当前） | 新随机障碍物布局 |
| B1-Rand10 | 每轮随机 10 个静态障碍物 | 新随机障碍物布局 |
| B1-ManualCurric | 手工课程：3→5→10→15 个 | 新随机障碍物布局 |
| **B1-EnvPolicy** | 环境策略自动调节 | 新随机障碍物布局 |

**Phase B2：泛化极限测试**

- 障碍物数量泛化：训练时最多 15 个 → 测试时 20、30、50 个
- 障碍物运动泛化：训练时静态 + 低速 → 测试时高速运动
- 空间规模泛化：训练时 20×20×10m → 测试时 50×50×20m

### 3.5 实现代码（环境策略核心）

```python
# env_policy.py — 环境策略 (SAC) 实现
class EnvironmentPolicy:
    """
    用 SAC 学习自动调节障碍物布局难度
    基于 Wang et al. (ICRA 2025) 的 Environment Policy 框架
    """
    def __init__(self, state_dim=5, action_dim=4):
        self.sac = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4
        )

    def get_env_state(self, fly_history, current_config):
        """
        环境策略的状态
        fly_history: 最近 N 轮的飞行策略表现记录
        current_config: 当前障碍物配置
        """
        recent_success_rate = np.mean([
            h['success'] for h in fly_history[-50:]
        ]) if len(fly_history) >= 50 else 0.0

        progress = current_config['episode'] / current_config['max_episodes']

        return np.array([
            recent_success_rate,                    # PPO 近期成功率
            current_config['num_obstacles'] / 30,   # 障碍物数量 (归一化)
            current_config['obs_density'],           # 障碍物密度
            current_config['dynamic_ratio'],         # 动态障碍物比例
            progress                                 # 训练进度
        ], dtype=np.float32)

    def get_reward(self, fly_success_rate):
        """
        环境策略的奖励函数
        核心思想：飞行策略成功率在 50-80% 区间时奖励最高
        （太容易=没学到东西，太难=学不会）
        """
        if 0.50 <= fly_success_rate <= 0.80:
            return 1.0  # 最佳难度区间
        elif fly_success_rate < 0.30:
            return -1.0  # 太难了，需要简化
        elif fly_success_rate > 0.95:
            return -0.5  # 太简单了，需要增加难度
        else:
            # 在过渡区间给予平滑奖励
            return 1.0 - 2.0 * abs(fly_success_rate - 0.65) / 0.35

    def apply_action(self, action, current_config):
        """
        将环境策略的动作映射为障碍物配置变更
        action: [Δn_norm, speed_range_norm, spread_norm, dynamic_ratio_norm]
        """
        # Δn: 障碍物数量变化 (-5 ~ +5)
        delta_n = int(action[0] * 5)

        new_config = current_config.copy()
        new_config['num_obstacles'] = max(2, min(
            30, current_config['num_obstacles'] + delta_n
        ))
        new_config['max_obs_speed'] = action[1] * 3.0     # 0 ~ 3 m/s
        new_config['obs_spread'] = action[2] * 20.0       # 0 ~ 20m 分布半径
        new_config['dynamic_ratio'] = action[3] * 0.8     # 0 ~ 80% 动态

        return new_config
```

**联合训练流程**：

```python
# train_with_env_policy.py
def train_joint(fly_ppo, env_policy, num_episodes):
    """
    飞行策略 (PPO) 和环境策略 (SAC) 交替训练
    """
    fly_history = []
    env_config = {'num_obstacles': 3, 'dynamic_ratio': 0.0,
                  'max_obs_speed': 0.0, 'obs_spread': 10.0,
                  'episode': 0, 'max_episodes': num_episodes}

    for ep in range(num_episodes):
        env_config['episode'] = ep

        # === Step 1: 环境策略选择障碍物布局 ===
        env_state = env_policy.get_env_state(fly_history, env_config)
        env_action = env_policy.sac.select_action(env_state)
        env_config = env_policy.apply_action(env_action, env_config)

        # === Step 2: 用新布局创建环境并训练飞行策略 ===
        env = DroneEnv(config=env_config)
        fly_results = train_ppo_one_rollout(fly_ppo, env)

        # === Step 3: 记录飞行策略表现 ===
        fly_history.append({
            'success': fly_results['success_rate'],
            'collision': fly_results['collision_rate'],
            'avg_reward': fly_results['avg_reward'],
            'config': env_config.copy()
        })

        # === Step 4: 给环境策略发奖励并更新 ===
        env_reward = env_policy.get_reward(fly_results['success_rate'])
        env_policy.sac.store_transition(
            env_state, env_action, env_reward
        )

        if ep % 10 == 0:
            env_policy.sac.update()

    return fly_ppo, env_policy, fly_history
```

---

## 四、推荐阅读：完整论文列表

### 4.1 必读 — Swift 及其直接后继（7 篇）

| # | 论文 | 期刊/会议 | 为什么必读 |
|---|------|----------|-----------|
| 1 | Kaufmann et al. **"Champion-Level Drone Racing Using Deep Reinforcement Learning."** *Nature*, 620(7976):982–987, 2023. | Nature | 一切的基础。读 Method 部分（PPO 架构、奖励设计、残差模型） |
| 2 | Wang\*, Xing\*, Messikommer, Scaramuzza. **"Environment as Policy: Learning to Race in Unseen Tracks."** *ICRA 2025*, pp.11333–11339. arXiv:2410.22308. | ICRA 2025 | **子课题 B 的直接基础**。读 Environment Policy 的 SAC 设计和奖励函数 |
| 3 | Geles, Bauersfeld, Romero, Xing, Scaramuzza. **"Demonstrating Agile Flight from Pixels without State Estimation."** *RSS 2024*. arXiv:2406.12505. | RSS 2024 | 非对称 Actor-Critic 的详细实现。读 Method §3 |
| 4 | Ferede, Blaha, Lucassen, De Wagter, De Croon. **"One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms."** *ICRA 2025*, pp.6357–6363. arXiv:2504.21586. | ICRA 2025 | 域随机化水平的系统性实验方法论。读 Experiment §4 |
| 5 | Xing, Romero, Bauersfeld, Scaramuzza. **"Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight."** *CoRL 2024*, pp.2542–2556. arXiv:2403.12203. | CoRL 2024 | Teacher-Student 三阶段框架。如果你后续想做特权信息蒸馏，这是基础 |
| 6 | Romero, Shenai, Geles, Aljalbout, Scaramuzza. **"Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight."** arXiv:2501.14377, 2025. | 预印本 | DreamerV3 在无人机上的应用。如果你后续要加入视觉，读这篇 |
| 7 | Hanover et al. **"Autonomous Drone Racing: A Survey."** *IEEE Transactions on Robotics*, 40:3044–3067, 2024. | IEEE T-RO | 领域全景图。论文 Introduction/Related Work 写作的参考模板 |

### 4.2 方法基础（5 篇）

| # | 论文 | 主题 | 为什么需要 |
|---|------|------|-----------|
| 8 | Schulman et al. **"Proximal Policy Optimization Algorithms."** arXiv:1707.06347, 2017. | PPO 原始论文 | 方法部分的算法描述出处 |
| 9 | Schulman et al. **"High-Dimensional Continuous Control Using Generalized Advantage Estimation."** ICLR 2016. arXiv:1506.02438. | GAE 原始论文 | GAE(λ=0.95) 的理论基础 |
| 10 | Andrychowicz et al. (OpenAI). **"Solving Rubik's Cube with a Robot Hand."** arXiv:1910.07113, 2019. | 自动域随机化 (ADR) | ADR 算法细节。如果你在子课题 A 中想做自适应噪声调节 |
| 11 | Peng, Andrychowicz, Zaremba, Abbeel. **"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization."** *ICRA 2018*. arXiv:1710.06537. | 动力学随机化 | 域随机化的奠基性工作。子课题 A 的 Related Work 引用 |
| 12 | Achiam, Held, Tamar, Abbeel. **"Constrained Policy Optimization."** *ICML 2017*. arXiv:1705.10528. | 安全 RL 理论 | CMDP 形式化定义 + 拉格朗日方法 vs CPO 的理论对比 |

### 4.3 参考与对比（3 篇）

| # | 论文 | 主题 | 为什么需要 |
|---|------|------|-----------|
| 13 | Loquercio et al. (UZH RPG). **"Learning High-Speed Flight in the Wild."** *Science Robotics*, 6(59), 2021. | Agile Autonomy | 纯视觉高速避障穿越。作为你纯向量状态方案的对比 |
| 14 | Song et al. (UZH RPG). **"Reaching the Limit in Autonomous Racing: Optimal Control Versus Reinforcement Learning."** *Science Robotics*, 8(82), 2023. | RL vs 最优控制 | 论证 RL 在路径规划中优于经典方法的重要引用 |
| 15 | Stooke, Achiam, Abbeel. **"Responsive Safety in Reinforcement Learning by PID Lagrangian Methods."** *ICML 2020*. arXiv:2007.03964. | PID Lagrangian | 如果你后续想做 Safe RL（C1 方向的延伸），PID Lagrangian 是 PPO-Lagrangian 的改进版 |

### 4.4 推荐阅读顺序

```
第一周  → 读 #1 (Swift) + #7 (Survey)，建立全局认知
第二周  → 读 #2 (Environment Policy) + #4 (One Net)，理解你要实现的方法
第三周  → 读 #8 (PPO) + #9 (GAE)，写代码时作为算法参考
第五周  → 读 #11 (Dynamics Rand) + #12 (CPO)，为论文 Related Work 写作做准备
```

**阅读策略**：不要试图从头到尾读完每一篇。对 Swift (#1) 精读 Method 和 Extended Data；对其他论文，先读 Abstract + Figures（看图理解方法），再读 Method（看公式和伪代码），最后读 Experiment（看实验设计可以借鉴什么）。

---

## 五、补充子课题方向

以下是从 Swift 后续工作中挖掘的其他可扩展方向，作为备选：

### 5.1 特权信息蒸馏敏感度分析（基于 Geles et al. RSS 2024）

- **源工作**：非对称 Actor-Critic（训练时 Critic 有完整状态，Actor 只用视觉）
- **扩展方向**：在向量状态环境中逐维度测试蒸馏效果——Critic 看到完整 14 维，Actor 逐步去掉障碍物精确距离、目标距离、速度等维度
- **优势**：不需要视觉输入，实验设计清晰——控制变量法，逐个维度对比

### 5.2 自适应域随机化水平调节（基于 Ferede et al. ICRA 2025）

- **源工作**：固定随机化水平的手动选择（0%/10%/20%/30%）
- **扩展方向**：让随机化水平在训练过程中根据策略表现自动调整，类似 Wang et al. 的思想，但随机化的不是环境布局而是物理参数
- **优势**：这个概念在文献中还没有被提出过

### 5.3 Teacher-Student 知识迁移效率分析（基于 Xing et al. CoRL 2024）

- **源工作**：Teacher（特权信息）→ IL 蒸馏 → RL 微调
- **扩展方向**：用你已有的 98% 成功率 PPO 策略作为 Teacher，训练 Student（输入更少维度或加入噪声），研究最优组合参数
- **优势**：Teacher 数据已就绪，不需要重新训练

### 5.4 世界模型 vs 无模型 RL 样本效率对比（基于 Dream to Fly 2025）

- **源工作**：DreamerV3 在视觉任务上成功的，PPO 失败
- **扩展方向**：在纯向量状态下对比 DreamerV3 和 PPO 在高密度障碍物场景下的样本效率
- **优势**：对照实验设计，能直接回答"什么时候需要世界模型"这个问题

### 5.5 推荐优先级矩阵

| 排名 | 子课题 | 来源 | 工作量 | 发表潜力 | 理由 |
|------|--------|------|--------|---------|------|
| 🥇 | 环境策略用于障碍物导航 | Wang et al. 2024 | 4-6 周 | ⭐⭐⭐⭐⭐ | 新方法+新场景，时间窗口好 |
| 🥈 | 感知退化下的鲁棒性量化 | 自研 | 2-3 周 | ⭐⭐⭐⭐ | 快出结果，基准性工作 |
| 🥉 | 自适应域随机化水平 | Ferede et al. 2025 | 3-5 周 | ⭐⭐⭐⭐ | 概念新，文献未出现 |
| 4 | 特权信息蒸馏敏感度分析 | Geles et al. 2024 | 3-4 周 | ⭐⭐⭐ | 系统性强，但创新度中等 |
| 5 | Teacher-Student 知识迁移 | Xing et al. 2024 | 3-4 周 | ⭐⭐⭐ | Teacher 数据已就绪 |
| 6 | 世界模型 vs PPO 样本效率 | Dream to Fly 2025 | 5-8 周 | ⭐⭐⭐ | 需要实现 DreamerV3，工程量大 |

---

## 六、学术继承关系说明

### 6.1 Swift 源码现状

Swift (Nature 2023) 论文**没有发布完整的可运行源码**。官方提供了 Zenodo 记录（[7955278](https://zenodo.org/records/7955278)）包含伪代码，社区整理为 [RoboDD/drone_policy](https://github.com/RoboDD/drone_policy)。

但 UZH RPG 组有大量完整开源的相关项目：

| 技术要素 | 开源项目 |
|---------|---------|
| PPO 训练管线 | [Flightmare](https://github.com/uzh-rpg/flightmare) — MIT 协议, C++/Python, 内置 PPO + 向量化环境 |
| 域随机化 sim-to-real | [sim2real_drone_racing](https://github.com/uzh-rpg/sim2real_drone_racing) — 完整 CNN + Domain Rand + 零样本迁移 |
| 高速避障穿越 | [agile_autonomy](https://github.com/uzh-rpg/agile_autonomy) — 特权学习 + 纯机载感知 |
| 特技飞行动作 | [deep_drone_acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) — PPO + 参考轨迹引导 |
| Actor-Critic MPC | [acmpc_public](https://github.com/uzh-rpg/acmpc_public) — 可微分 MPC 嵌入 RL 框架 |

### 6.2 你的项目与 Swift 的对齐关系

| 要素 | Swift / UZH 标准 | 你的实现 | 对齐程度 |
|------|-----------------|---------|---------|
| 算法 | PPO-Clip | PPO-Clip | ✅ 完全一致 |
| 网络架构 | 两层 MLP (128×128) | 两层 MLP (128×128), Actor/Critic 共享 | ✅ 完全一致 |
| 优势估计 | GAE (λ=0.95) | GAE (λ=0.95), per-environment | ✅ 完全一致 |
| 初始化 | Orthogonal (gain=√2, 0.01) | Orthogonal (gain=√2, 0.01) | ✅ 完全一致 |
| 梯度裁剪 | max_grad_norm=0.5 | max_grad_norm=0.5 | ✅ 完全一致 |
| 并行环境 | 多环境并行 | 8×SyncVectorEnv | ✅ 一致 |
| 仿真器 | Flightmare（C++/Unity） | 自研 Python Gymnasium | ⚠️ 引擎不同 |
| 奖励函数 | progress + gate perception | 7组件分层奖励 | ⚠️ 场景不同，理念一致 |
| 状态空间 | 含 gate 相对位姿 | 含目标 + 障碍物相对信息 | ⚠️ 场景不同，结构类比 |
| 观测 | VIO + 相机门检测 | 完美 14 维向量 | ⚠️ 上帝视角 |
| 残差建模 | GP + k-NN | 未实现 | ❌ 缺失 |
| 动作空间 | 集体推力 + 体角速度 (4D) | 推力 xyz (3D) | ⚠️ 模型不同 |

### 6.3 论文写作中的表述建议

> "We adopt the control policy architecture of Swift (Kaufmann et al., Nature 2023): a two-layer MLP (128 units per layer) with shared feature extractor, trained via PPO-Clip with GAE (λ=0.95, γ=0.99). Orthogonal initialization (gain=√2 for hidden layers, 0.01 for output layers) and gradient clipping (max_norm=0.5) follow the Swift training protocol.
>
> Our work extends this architecture from the racing domain to obstacle-rich path planning, and investigates [你的创新点]."

---

*文档整理时间：2026-07-17*
*基于 Swift (Nature 2023) 及其后续工作的研究方案*
