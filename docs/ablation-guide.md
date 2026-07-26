# 常数深度消融实验 (P0 诊断)

**目的**：检验模型是否真的学到了视觉特征，或者只是依赖向量信息。

> 如果消融后 SR 仍然很高，说明当前实验不能证明视觉鲁棒性 → 需要从根本上改变后续实验设计

---

## 实验设计

### 消融 1: 常数深度 (const_depth)

**操作**：将深度图替换为常数 5.0m

```python
ablation_config = {'const_depth': True}
env = VisualDroneEnv(config={'ablation': ablation_config})
```

**模型可用信息**：
- ✅ Velocity (vx, vy, vz) — 真实
- ✅ Target direction (Δx, Δy, Δz) — 真实
- ❌ Depth image — 固定 5.0m，无信息

**预期结果**：
- 如果 `SR(const_depth) ≈ SR(baseline)` → **⚠️ 严重问题**：模型未学到深度信息
- 如果 `SR(const_depth) < SR(baseline) - 20%` → ✅ 正常：模型依赖深度

---

### 消融 2: 无目标方向 (no_target_dir)

**操作**：将目标方向向量置为 (0, 0, 0)

```python
ablation_config = {'no_target_dir': True}
env = VisualDroneEnv(config={'ablation': ablation_config})
```

**模型可用信息**：
- ✅ Velocity (vx, vy, vz) — 真实
- ❌ Target direction — 全 0
- ✅ Depth image — 真实

**预期结果**：
- 如果 `SR(no_target_dir)` 仍然很高（>70%）→ **⚠️ 问题**：模型过度依赖完美目标信息（真机无法提供）
- 如果 `SR(no_target_dir)` 显著下降（<50%）→ ✅ 正常：模型需要目标信息

---

### 消融 3: 极端情况 (const_depth + no_target_dir)

**操作**：同时应用上两个消融

```python
ablation_config = {'const_depth': True, 'no_target_dir': True}
```

**预期**：SR 应该接近 0（模型完全丧失高级信息，只能依赖速度）

---

## 使用方式

### 快速运行（50 episodes）

```bash
cd rlproject-swift-improved
python scripts/eval_ablation.py --model saved_models/visual_ppo_best.pth
```

### 详细运行（200 episodes，更稳定）

```bash
python scripts/eval_ablation.py \
  --model saved_models/visual_ppo_best.pth \
  --episodes 200 \
  --output eval_results/ablation_detailed
```

---

## 结果解释

### 情况 1：const_depth SR ≈ baseline SR （⚠️ 问题）

```
Baseline SR: 75.5%
const_depth SR: 74.3%  ← 几乎没降

结论: 模型未学到深度信息！
可能原因:
  - 深度噪声太弱（v2 已改）
  - CNN 编码器未真正利用深度特征
  - 模型过度依赖目标方向（消融 2 验证）
  
改进方向:
  - 检查 CNN 的特征可视化（depth 是否被编码？）
  - 尝试 RGB-D 融合，确保深度编码被使用
  - 重新设计观测：concat depth + direction（而非分开）
```

### 情况 2：no_target_dir SR 仅微小下降（⚠️ 问题）

```
Baseline SR: 75.5%
no_target_dir SR: 74.1%  ← 只降了 1.4%

结论: 模型严重过度依赖目标方向！
原因分析:
  - target_direction = (Δx, Δy, Δz) 本质上是"作弊"：
    在真机上，我们无法获得真实的相对位置
  - 模型应该从深度图推断目标位置 (里程计)
  
改进方向:
  - 移除 target_direction，完全依赖深度视觉和速度估计
  - 引入噪声的速度估计（模拟真机 IMU 噪声）
  - 要求模型学习 SLAM/VIO：从深度序列中自己定位
```

### 情况 3：Both ablation SR ≈ 0 （✅ 正常）

```
Baseline SR: 75.5%
both SR: 2.1%  ← 几乎崩溃

结论: ✅ 正常行为
模型确实依赖深度和目标信息的某种组合
```

---

## 诊断流程

```
1. 跑 3 个消融实验（const_depth, no_target_dir, both）

2. 看 const_depth 结果
   ├─ SR 降幅 < 5%？ → 问题 1：模型未用深度
   ├─ SR 降幅 20-50%？ → ✅ 正常：部分依赖深度
   └─ SR 降幅 > 50%？ → ✅ 正常：完全依赖深度

3. 看 no_target_dir 结果
   ├─ SR 降幅 < 10%？ → 问题 2：模型过度依赖目标方向
   ├─ SR 降幅 20-50%？ → ✅ 正常：需要目标信息但不完全依赖
   └─ SR 降幅 > 50%？ → ✅ 正常：完全依赖目标方向

4. 结合分析
   ├─ 问题 1 + 问题 2 → 深度编码器可能完全失效，CNN 网络结构有问题
   ├─ 只有问题 1 → 深度信息被后期融合，但 CNN 未充分提取
   ├─ 只有问题 2 → 深度编码有效，但网络架构过度信任目标方向
   └─ 都正常 → baseline 可信，进入 Phase V2 衰减测试
```

---

## 输出格式

```json
{
  "baseline": {
    "success_rate": 75.5,
    "collision_rate": 24.5,
    "timeout_rate": 0.0,
    "avg_reward": 65.3
  },
  "ablation_const_depth": {
    "success_rate": 74.3,
    "collision_rate": 25.7,
    "timeout_rate": 0.0,
    "avg_reward": 61.2
  },
  "ablation_no_target_dir": {
    "success_rate": 45.2,
    "collision_rate": 54.8,
    "timeout_rate": 0.0,
    "avg_reward": -10.5
  },
  "ablation_both": {
    "success_rate": 2.1,
    "collision_rate": 97.9,
    "timeout_rate": 0.0,
    "avg_reward": -150.3
  },
  "analysis": {
    "const_depth_sr_delta": -1.2,
    "no_target_dir_sr_delta": -30.3,
    "interpretation": {...}
  }
}
```

---

## 关键发现用途

这个消融实验的结果会直接影响 **后续 Phase V3 的实验设计**：

| 发现 | V3 改进 |
|------|--------|
| 模型未用深度（const_depth SR ≈ baseline） | 重新设计 CNN 或观测融合方式，不进入鲁棒训练 |
| 模型过度依赖目标方向（no_target_dir SR ≈ baseline） | 移除 target_direction，改用里程计或 SLAM |
| 两者都健康（正常降幅） | ✅ 可进入 V3 鲁棒训练 |
