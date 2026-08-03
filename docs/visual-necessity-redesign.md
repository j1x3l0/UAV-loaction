# 视觉必要性重设计（草案）

> 日期：2026-08-03 | 状态：**草案，隔离实验，失败可放弃**
> 背景：对齐 V3 消融与必要性验证表明深度视觉只是中等贡献（纯避障 episode 下去掉深度仅掉 6-8pp）。要支撑强「视觉导航」主张，需重设计任务使**深度成为必要输入**。

## 一、当前任务为何不强制视觉

| 因素 | 现状 | 后果 |
|------|------|------|
| 目标方向 | 免费提供在 `vec[3:6]`（target_dir = target_pos − pos） | 策略知道去哪，不需要看 |
| 障碍密度 | gate 场景障碍稀疏/边缘 | 盲飞（bump）也能 16% 绕过 |
| 深度唯一信息 | 只有障碍位置 | 而障碍本就不强制避让 |

**必要条件**：任务中必须存在**只在深度图里、且完成必需**的信息。

## 二、三个候选方案（按成本升序）

### D1：碰撞半径硬化（Obstacle Foresight）——纯配置，零代码
- 提高 `drone_collision_radius`（如 0.5→1.0→1.5m），使障碍等效变粗、盲飞必撞。
- 策略必须提前在深度里看到障碍并绕行；`const_depth` 应显著崩塌。
- 代价：任务整体变难，需重新找可学半径。
- **可隔离性**：只改评估配置，不动代码。**失败成本**：几乎为零。

### D2：去目标向量 + 目标锚定场景特征（Gate-as-Goal）——需 wrapper
- `vec` 中去掉 `target_dir`（改为零）；目标固定放在场景特征处（如 gate 开口）。
- 策略只能靠深度图几何（gate 开口结构）推断「往哪飞」。
- 风险：目标必须深度可辨（gate 开口），且初始探索难。
- **可隔离性**：Gym wrapper 修改观测，不动核心 env。
- **失败成本**：新 wrapper 文件，可删。

### D3：稠密合成障碍场（Synthetic Corridor）——需合成数据
- 新建稠密点云场景（多柱/迷宫），直线必撞、盲飞必败。
- 在合成场景上重训 + 必要性消融。
- 成本最高，但最能干净地证明「深度必要」。
- **可隔离性**：独立数据 + 独立实验目录。

## 三、推荐路线

1. **先 D1**（最便宜）：扫 `drone_collision_radius ∈ {0.75, 1.0, 1.25}`，在 curriculum robust 模型上测 baseline vs `const_depth`。若某个半径下 `const_depth` 崩塌而 baseline 保持，深度必要性即被诱导。
2. **D1 不够再 D2**：wrapper 去 `target_dir` + 目标锚定 gate。
3. **D3 兜底**：若场景本身撑不起，用合成场景。

## 四、隔离原则（不碰原代码）

- 不改 `envs/visual_drone_env.py`、`scripts/train_visual.py`、`core/`、`configs/px4_gate_mid_alignment.json`。
- 新文件全放 `rlproject-swift-improved/experiments/visual_necessity/`，失败直接删目录。
- 评估复用现有 `eval_v3_ablation.py --avoidance-probability` 与 `paired_robustness_stats.py`。

## 五、验收标准

- 目标：存在某任务变体使 `const_depth` 相对 baseline 崩塌（如 SR 差 ≥30pp）且 baseline 本身可学（SR ≥ 30%）。
- 达成 → 视觉必要性在改进任务下成立，可写入论文；未达成 → 维持弱化方向，放弃本重设计。
