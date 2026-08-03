# D3-lite 视觉必要性门控通过

日期：2026-08-03。合成稠密走廊任务，验证深度是否为必要输入。

## 方法

- **场景**：合成密集障碍走廊——5 面墙沿 +X 交替开口（slalom），直线路径必被阻挡。`data/d3_lite/dense_corridor.ply`（gsplat 场景）+ `dense_corridor_collision.ply`（35160 碰撞点）。
- **任务**：全避障（avoidance=1.0），目标方向保留在 vec（知道去哪），深度是唯一避障信号。
- **训练**：1200ep，seed 0，默认相机（fov=90 固定 +X），0.5m 碰撞半径，隔离脚本 `experiments/visual_necessity/train_d3lite.py`。
- **评估**：50ep，baseline（有深度）vs const_depth（深度替换为固定 5m）。

## 结果（SR %）

| 条件 | SR |
|------|:---:|
| baseline（有深度） | **54.0** |
| const_depth（无深度） | **0.0** |

**门控：baseline ≥ 30% 且 baseline − const_depth ≥ 30pp → PASS**（54% 可学，差 54pp）。

## 结论

**深度必要性成立**：在密集障碍走廊中，策略有深度可学会穿行（54%），无深度完全失败（0%）。
深度图是避障的必要输入。

- 对比 gate 场景（深度仅中等贡献）、D1（碰撞半径硬化失败）、D2（去向量不可解）：
  **障碍密度（使盲飞必撞）是诱导深度必要性的正确杠杆**。
- 论文主张升级：视觉必要性在 D3-lite 任务下实证成立；可作 gate 场景鲁棒性分析的
  **互补实验**，支撑「视觉导航」框架。

## 文件

- `result.json`：baseline/const_depth SR + gate 判定。
- 生成器：`experiments/visual_necessity/gen_dense_corridor.py`；训练：`train_d3lite.py`；评估：`eval_d3lite_necessity.py`。
