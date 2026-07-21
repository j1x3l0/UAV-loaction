# rlproject-swift-improved — 无人机 RL 路径规划 (v1 向量PPO基线)

> **状态**: v1 基线，v2 在此基础上扩展视觉管线 (3DGS + CNN + PPO)
> **最佳模型**: `saved_models/best/ppo_swift_3000ep_20260712_115059` — 98% 成功率

## 模块结构

```
rlproject-swift-improved/
├── core/                       # 核心RL库 (无内部依赖)
│   ├── config.py               #   训练超参数配置
│   └── ppo_agent.py            #   PPO算法 (ActorCritic + GAE + Update)
│
├── envs/                       # 环境 (依赖 core)
│   ├── drone_env.py            #   基线环境 (14D向量 + 7组件奖励)
│   └── drone_env_noisy.py      #   噪声注入变体 (继承 DroneEnv)
│
├── scripts/                    # 可运行脚本 (依赖 core + envs)
│   ├── train.py                #   PPO训练入口
│   ├── eval_baseline.py        #   模型评估
│   ├── plot_a1_curves.py       #   噪声衰减曲线可视化
│   └── smoke_test.py           #   管线验证 (6步冒烟测试)
│
├── saved_models/best/          # 最佳模型
├── tests/                      # 测试 (待建)
└── README.md
```

## 依赖关系

```
core/config.py    ← 独立
core/ppo_agent.py ← 独立
envs/drone_env.py ← 独立
envs/drone_env_noisy.py → envs/drone_env.py (继承)
scripts/train.py → core.* + envs.*
scripts/eval_baseline.py → core.ppo_agent + envs.drone_env
scripts/smoke_test.py → core.ppo_agent + envs.drone_env
```

## 快速开始

```bash
# 从项目根目录运行 (rlproject-swift-improved/)
cd rlproject-swift-improved

# 快速冒烟测试
python scripts/smoke_test.py

# 评估最佳模型
python scripts/eval_baseline.py

# 训练 (默认3000ep)
python scripts/train.py

# 快速测试训练
python scripts/train.py --episodes 100 --lr 1e-4
```

## 关键设计

- **14D 向量状态**: [pos(3), vel(3), target_dir(3), dist(1), obs_dir(3), obs_dist(1)]
- **3D 连续动作**: 推力 xyz, 归一化到 [-1, 1], tanh 约束均值
- **7组件奖励**: 距离引导 + 速度方向 + 障碍物惩罚 + 动作平滑 + 到达 + 碰撞 + 超时
- **GAE per-environment**: 并行环境下按环境分别计算 GAE (曾导致 0% 成功率的关键 bug)
- **共享特征提取器**: Actor/Critic 共享第一层 MLP (Swift 实践)
