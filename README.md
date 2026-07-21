# UAV Visual RL — 3DGS 视觉导航策略鲁棒性研究

> 基于深度强化学习的无人机 3D 视觉导航 | 目标会议: ICRA 2027 (~2026年9月截止)
> 硬件: 3× RTX 3080 (10GB VRAM)

## 核心思路

```
手机视频 → 3DGS重建 → 深度图渲染 → CNN编码器 → 共享MLP → 动作(3D推力)
                                             ↑
                             PPO Update (GAE + Clip)
```

**核心问题**: 3DGS 渲染质量下降时，视觉导航策略的性能如何退化？哪些退化轴最致命？如何训练鲁棒策略？

## 项目结构

```
rlproject-基本框架/
├── README.md                       # 本文件
├── CLAUDE.md                       # AI 协作原则 + 项目技术细节
├── MASTER_PLAN.md                  # v2 总计划 (21天 × 3 GPU)
├── PROGRESS.md                     # 进度日志
├── docs/                           # 研究文档
│   ├── grad-nav-comparison.md      #   GRaD-Nav vs 本项目的详细对比
│   ├── plan-v1-to-v2-migration.md  #   v1→v2 迁移记录
│   └── reading-list-v2.md          #   论文阅读清单 (15篇)
├── rlproject-swift-improved/       # v1 向量PPO基线 (98%成功率)
│   ├── core/                       #   核心库 (PPO, Config)
│   ├── envs/                       #   环境 (DroneEnv, NoisyEnv)
│   └── scripts/                    #   训练/评估/可视化脚本
├── grad_nav-main/                  # GRaD-Nav 参考实现 (Stanford IROS 2025)
├── past/                           # v1 历史归档 (旧实验、日志、论文)
└── jxl_better_vibe_coding/         # 开发笔记 + 代码审查
```

## 快速开始

```bash
# 冒烟测试 (6步管线验证)
cd rlproject-swift-improved
python scripts/smoke_test.py

# 评估最佳模型
python scripts/eval_baseline.py

# 训练 (3000 episodes)
python scripts/train.py

# 快速测试
python scripts/train.py --episodes 100 --lr 1e-4
```

## 环境

- **空间**: 20×20×10m
- **无人机**: 质点模型, 质量1.0, 最大推力10.0, dt=0.05s (20Hz)
- **观测 (v1)**: 14D 向量 — pos(3) + vel(3) + target_dir(3) + dist(1) + obs_dir(3) + obs_dist(1)
- **观测 (v2)**: 深度图 (64×64×1) + 向量 (6D)
- **动作**: 3D 连续推力, 归一化到 [-1,1]
- **奖励**: 7组件密集奖励 (距离引导 + 速度方向 + 障碍物惩罚 + 动作平滑 + 到达 + 碰撞 + 超时)

## 关键设计决策

1. **PPO vs DDRL (SHAC)**: 选 PPO — 鲁棒性分析不需要极致样本效率；DDRL 梯度穿过退化 GS 可能不稳定
2. **深度图 vs RGB**: 选深度图 — 光照不变性 + 几何重建质量的直接反映
3. **简化动力学 vs 全四元数**: 用质点模型 — 隔离视觉退化效应，与 v1 保持动作空间一致
4. **共享特征提取器**: Actor/Critic 共享，减少参数，加速训练

## 5 条退化轴 (v2 核心实验)

| 退化轴 | 操作 | 假设敏感度 |
|--------|------|-----------|
| 高斯球稀疏化 | 按 opacity×scale 保留 {100%-5%} | ⭐⭐⭐ 最高 |
| 分辨率降低 | 渲染分辨率 {256-16} → 上采样 | ⭐⭐ |
| 视角覆盖衰减 | 限制相机朝向 {360°-45°} | ⭐⭐⭐ |
| 光照偏移 | RGB × 2^(EV), EV∈{0,±4} | ⭐ (深度图不变) |
| 深度噪声 | PerlinNoise × σ, σ∈{0-0.2} | ⭐⭐ |

## 论文进度

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 代码构建 (visual env + CNN PPO) | ✅ 已完成 |
| Phase V1 | Baseline 训练 (2场景) | ⬜ 待 GPU |
| Phase V2 | 5轴衰减曲线 | ⬜ 未开始 |
| Phase V3 | 鲁棒训练 + 对比 + 消融 (12组实验) | ⬜ 未开始 |
| Phase V4 | 论文写作 (21天目标) | ⬜ 未开始 |

## License

MIT
