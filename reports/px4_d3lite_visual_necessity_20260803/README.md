# D3-lite 视觉必要性门控通过

日期：2026-08-03。合成稠密走廊任务，验证深度是否为必要输入。

## 方法

- **场景**：合成密集障碍走廊——5 面墙沿 +X 交替开口（slalom），直线路径必被阻挡。`data/d3_lite/dense_corridor.ply`（gsplat 场景）+ `dense_corridor_collision.ply`（35160 碰撞点）。
- **任务**：全避障（avoidance=1.0），目标方向保留在 vec（知道去哪），深度是唯一避障信号。
- **训练**：1200ep，seed 0，默认相机（fov=90 固定 +X），0.5m 碰撞半径，隔离脚本 `experiments/visual_necessity/train_d3lite.py`。
- **评估**：50ep，baseline（有深度）vs const_depth（深度替换为固定 5m）。

## 结果（SR %，单 seed 门控）

| 条件 | SR |
|------|:---:|
| baseline（有深度） | **54.0** |
| const_depth（无深度） | **0.0** |

**门控：baseline ≥ 30% 且 baseline − const_depth ≥ 30pp → PASS**（54% 可学，差 54pp）。

## 5-seed 强化结果（2026-08-03）

| seed | baseline | const_depth | diff | paired CI | McNemar p |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 54 | 0 | +54pp | [40,68] | <0.001 |
| 1 | 88 | 42 | +46pp | [32,60] | <0.001 |
| 2 | 52 | 0 | +52pp | [38,66] | <0.001 |
| 3 | 100 | 100 | 0pp | [0,0] | 1.0 |
| 4 | 46 | 0 | +46pp | [32,60] | <0.001 |

**深度必要性显著支持（4/5 seeds，p<0.001，配对 bootstrap CI 不含 0）**。诚实标注：
- seed0/2/4：const_depth 严格崩塌到 0%（深度严格必要）。
- seed1：const_depth 42%（深度显著帮助，但非严格必要）。
- seed3：无深度也 100%——学出了**利用 target_dir 泄露走廊布局**的策略（向量编码了足够
  导航信息），深度对它不必要。这是必要性测试的已知失败模式（向量泄露/记忆），论文须如实报告。

## 结论

- 深度必要性在 D3-lite 任务下**显著成立（4/5 seeds）**，但非绝对——部分 seed 能从向量
  泄漏中绕开深度。门控通过（单 seed 门控 + 4/5 显著）。
- 对比 gate 场景（深度仅中等贡献）：**障碍密度是诱导深度必要性的正确杠杆**。
- 论文可主张「视觉必要性在稠密障碍任务下显著成立」，同时诚实报告 seed 方差与向量泄露边界。

## 文件

- `result.json`：单 seed 门控；`seed{0-4}.json`：5-seed 逐 seed 结果。
- 生成器：`experiments/visual_necessity/gen_dense_corridor.py`；训练：`train_d3lite.py`；评估：`eval_d3lite_necessity.py`。
