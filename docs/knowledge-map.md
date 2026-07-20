# 🗺️ UAV RL 项目 — 团队认知地图

> **用途**: 团队共用的系统理解地图。每次合并 PR 后同步更新。
> **维护规则**: 谁改了代码，谁更新地图。AI 生成的 PR 也必须附带地图更新。
> **最后更新**: 2026-07-18

---

## 一、系统全景图

```mermaid
graph TB
    subgraph "🎯 目标层: 研究问题"
        Q["无人机3D路径规划<br/>DRL方法<br/>目标: 发表级成果"]
    end

    subgraph "🧠 算法层: RL核心"
        PPO["PPO Agent<br/>ppo_agent.py"]
        SAC["SAC Agent (Legacy)<br/>sac_agent.py"]
        GAE["GAE 优势估计<br/>λ=0.95, γ=0.99"]
        ENT["自适应熵系数<br/>AdaptiveEntropyCoeff"]
    end

    subgraph "🌍 环境层: Simulation"
        ENV_SWIFT["DroneEnv Swift版<br/>14D向量状态<br/>7组件奖励"]
        ENV_NOISY["DroneEnv Noisy<br/>观测噪声包装器"]
        ENV_LEGACY["DroneEnv 旧版<br/>10D + 深度图像"]
    end

    subgraph "🏗️ 基础设施层"
        TRAIN["训练循环<br/>train.py<br/>8环境并行 + TensorBoard"]
        CONFIG["TrainingConfig<br/>config.py<br/>超参数管理"]
        EVAL["评估系统<br/>eval_baseline.py<br/>6维指标"]
    end

    subgraph "📦 产出层"
        MODEL["训练模型<br/>saved_models/"]
        LOGS["TensorBoard日志<br/>logs/"]
        DOCS["文档<br/>research_plan.md<br/>improvement_report.md"]
    end

    Q --> PPO
    Q --> SAC
    PPO --> GAE
    PPO --> ENT
    PPO --> TRAIN
    SAC --> TRAIN
    TRAIN --> ENV_SWIFT
    TRAIN --> ENV_NOISY
    TRAIN --> ENV_LEGACY
    CONFIG --> TRAIN
    TRAIN --> MODEL
    TRAIN --> LOGS
    TRAIN --> EVAL
    EVAL --> DOCS
```

---

## 二、数据流：一次 PPO 训练步骤

```mermaid
sequenceDiagram
    participant E as 8×DroneEnv
    participant M as PPO.model (ActorCritic)
    participant B as PPO.memory (Buffer)
    participant G as PPO.compute_gae()
    participant U as PPO.update()

    Note over E,U: === 采样阶段 (rollout_steps=256/环境) ===

    loop 256 steps × 8 envs
        E->>E: state = [x,y,z, vx,vy,vz, dx,dy,dz, dist_tgt, odir_xyz, dist_obs]
        E->>M: forward(state) → (mean, std, value)
        M-->>E: action ~ Normal(mean, std)
        E->>E: step(action) → next_state, reward, done
        E->>B: store(state, action, reward, log_prob, value, entropy)
    end

    Note over E,U: === GAE 计算阶段 ===

    B->>G: rewards, values, dones (per environment)
    G->>G: δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    G->>G: A_t = δ_t + γ·λ·(1-done)·A_{t+1}
    G->>G: A = (A - mean) / (std + 1e-8)
    G-->>B: advantages, returns

    Note over E,U: === PPO 更新阶段 (10 epochs) ===

    loop 10 epochs
        U->>U: Shuffle & minibatch (batch_size=64)
        U->>U: ratio = exp(log_prob_new - log_prob_old)
        U->>U: L_clip = -min(ratio·A, clip(ratio, 0.8, 1.2)·A)
        U->>U: L_critic = MSE(value, returns)
        U->>U: L_total = L_clip + 0.5·L_critic - α·entropy
        U->>U: backward() + clip_grad(0.5) + step()
    end

    U->>U: AdaptiveEntropyCoeff.update(entropy)
```

---

## 三、模块职责矩阵

| 模块 | 文件 | 职责 | 依赖 | 被依赖 |
|------|------|------|------|--------|
| **环境 (Swift)** | `drone_env.py` | 14D状态构建、7组件奖励、物理仿真 | gymnasium, numpy | train.py, eval_baseline.py |
| **环境 (Noisy)** | `drone_env_noisy.py` | 观测噪声包装（继承DroneEnv） | drone_env.py | train.py（可选） |
| **PPO Agent** | `ppo_agent.py` | ActorCritic网络、GAE、PPO更新、自适应熵 | torch, numpy | train.py |
| **SAC Agent (旧)** | `sac_agent.py` | CNN+MLP SAC实现（已弃用） | torch, numpy | 无 |
| **训练循环** | `train.py` | 8环境并行、评估、日志、模型保存 | ppo_agent, drone_env, config | 无 |
| **配置** | `config.py` | 超参数管理、CLI参数解析 | argparse | train.py |
| **评估** | `eval_baseline.py` | 加载模型评估基线性能 | ppo_agent, drone_env | 无 |
| **冒烟测试** | `smoke_test.py` | GAE修复验证 | ppo_agent | 无 |

---

## 四、关键概念速查表

| 概念 | 位置 | 一句话解释 |
|------|------|-----------|
| **14D 状态向量** | `drone_env.py:137-146` | [位置(3), 速度(3), 目标相对位置(3), 目标距离(1), 障碍物方向(3), 障碍物距离(1)] |
| **7组件奖励** | `drone_env.py:251-319` | 距离引导 + 速度方向 + 障碍物惩罚 + 动作平滑 + 到达奖励 + 碰撞惩罚 + 超时惩罚 |
| **GAE (λ=0.95)** | `ppo_agent.py:160-194` | 广义优势估计，平衡偏差-方差。**关键**: per-environment 计算，不可跨环境边界 |
| **PPO-Clip** | `ppo_agent.py:346-349` | `L = -min(ratio·A, clip(ratio, 0.8, 1.2)·A)` |
| **自适应熵系数** | `ppo_agent.py:16-39` | `target_entropy = -action_dim = -3`，动态调整探索强度 |
| **Orthogonal 初始化** | `ppo_agent.py:76-85` | hidden gain=√2, actor output gain=0.01, critic output gain=1.0 |
| **8环境并行** | `train.py:116` | SyncVectorEnv，数据形状: (rollout_steps, num_envs) → 转置为 (num_envs, rollout_steps) 计算 GAE |
| **学习率衰减** | `train.py:204-206` | `lr = 3e-4 × (1 - progress)`，线性衰减至 0 |

---

## 五、🌫️ 理解雾图 (Fog Map)

> 绿色 = 团队理解清晰 | 黄色 = 部分理解 | 红色 = 理解不足 | 灰色 = 未探索

```mermaid
quadrantChart
    title 模块理解度 vs 重要性
    x-axis "理解度低" --> "理解度高"
    y-axis "重要性低" --> "重要性高"
    quadrant-1 "🔥 关键风险区"
    quadrant-2 "✅ 核心掌握区"
    quadrant-3 "📦 可推迟区"
    quadrant-4 "⚠️ 需巩固区"
    "7组件奖励函数": [0.7, 0.95]
    "GAE per-env计算": [0.85, 0.9]
    "PPO-Clip损失": [0.8, 0.85]
    "ActorCritic网络": [0.9, 0.8]
    "14D状态空间": [0.85, 0.75]
    "自适应熵系数": [0.6, 0.7]
    "8环境并行采样": [0.75, 0.65]
    "Orthogonal初始化": [0.5, 0.55]
    "CNN+MLP旧架构": [0.2, 0.3]
    "SAC Agent": [0.15, 0.2]
    "drone_env_noisy": [0.85, 0.75]
    "Sim-to-Real迁移": [0.05, 0.5]
    "动态障碍物": [0.05, 0.6]
    "多目标导航": [0.05, 0.45]
    "奖励权重调优": [0.4, 0.6]
    "超参数敏感性": [0.35, 0.55]
```

### 当前雾区清单

| 雾区 | 严重度 | 影响 | 解锁方式 |
|------|--------|------|----------|
| Sim-to-Real 迁移 | 🔴 完全未知 | 论文可发表性的关键门槛 | 阅读 Swift 论文章节 4-5，域随机化实验 |
| 动态障碍物 | 🔴 完全未知 | 限制研究深度 | 扩展 drone_env，添加移动障碍物 |
| 超参数敏感性 | 🟡 部分理解 | 调参效率低 | 系统性消融实验 |
| 奖励权重调优 | 🟡 部分理解 | 可能进一步优化 | 基于 reward_components 分析调整权重 |

---

## 六、文件依赖图

```mermaid
graph LR
    subgraph "运行时依赖"
        TRAIN[train.py] -->|import| PPO[ppo_agent.py]
        TRAIN -->|import| ENV[drone_env.py]
        TRAIN -->|import| CFG[config.py]
        ENV -->|gymnasium| GYM[Gymnasium]
        PPO -->|torch| TORCH[PyTorch]
        EVAL[eval_baseline.py] -->|import| PPO
        EVAL -->|import| ENV
        NOISY[drone_env_noisy.py] -->|inherits| ENV
        SAC[sac_agent.py] -->|torch| TORCH
    end

    subgraph "文档依赖"
        REPORT[improvement_report.md] -.->|引用| PPO
        REPORT -.->|引用| ENV
        REPORT -.->|引用| TRAIN
        PLAN[research_plan.md] -.->|引用| REPORT
        WEEKLY[weekly_plan.md] -.->|引用| PLAN
    end
```

---

## 七、团队认知快照

> 每次有人深入理解了一个模块，在这里更新日期和姓名。

| 模块 | 最后深入理解者 | 日期 | 备注 |
|------|---------------|------|------|
| drone_env.py (Swift) | — | 2026-07-12 | 改进报告中详细记录 |
| ppo_agent.py (PPO+GAE) | — | 2026-07-12 | 改进报告中详细记录 |
| train.py (并行训练) | — | 2026-07-12 | 改进报告中详细记录 |
| config.py (超参数) | — | 2026-07-12 | 需系统性消融实验 |
| drone_env_noisy.py | — | 2026-07-19 | 5种噪声模式实现+10K采样验证，分布正确性确认 |
| sac_agent.py (旧版) | — | — | 已弃用 |
| Sim-to-Real | — | — | 待研究 |

---

## 八、地图维护规则

1. **PR 合并前**: 如果代码改动影响了模块职责、数据流或依赖关系，必须更新此地图
2. **新增模块**: 添加到职责矩阵和依赖图
3. **理解度变化**: 有人深入研究某模块后，更新雾图和认知快照
4. **每周 Review**: 团队周会时花 5 分钟确认地图是否反映当前理解状态
5. **AI 生成代码**: PR 描述中必须包含此地图的更新 diff
