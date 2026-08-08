# V3c 跨场景泛化（sv_1007 / sv_917 left / sv_917 right）

日期：2026-08-07。主场景 sv_1007 训练后的跨场景泛化测试，用 sv_917 left/right 两个
新场景做 zero-shot 泛化评估。

## 方法

- **场景**：sv_1007（主场景，22×25×9m）、sv_917_3_left（278,016 高斯）、
  sv_917_3_right（257,626 高斯）。left/right 的 gsplat 由 nerfstudio checkpoint
  导出（`utils/extract_ply.py`），对齐用合成 identity（fx≈97.14 相机，与主场景一致）。
- **模型**：各场景 seed0 final（sv_1007 / left / right）。
- **评估**：`eval_v3c_cross_scene.py`，3×3 矩阵（源模型 × 目标场景），50ep/单元，
  统一对齐相机。GPU0 only。
- 原始数据：`matrix.json`。

## 结果（SR %，5-seed 均值；此表为 seed0）

| 源模型 \ 目标场景 | sv_1007 | left | right |
|:---:|:---:|:---:|:---:|
| **sv_1007** | **42.0** | **0.0** | **0.0** |
| **left** | 0.0 | **50.0** | 46.0 |
| **right** | 0.0 | 36.0 | **42.0** |

（left/right 训练 best_SR：left 53/52/55%，right 43/45/45%；sv_1007 seed0 42%）

## 关键发现：跨场景泛化失败 ⚠️

1. **对角线（in-domain）正常**：各场景自身模型在自己场景上 42-50%，任务可学。
2. **非对角线（cross-scene）全部崩塌**：
   - **sv_1007 → left/right：0%**——主场景模型在 left/right 上完全失败。
   - **left/right → sv_1007：0%**——新场景模型无法在 sv_1007 上工作。
   - **left ↔ right：46%/36%**——两个新场景之间部分泛化（场景相似度高）。
3. **主场景 sv_1007 与 left/right 之间零泛化**，但 left/right 之间有部分泛化。

## 解读（诚实边界）

- sv_1007 是完整 gate 大场景（22×25×9m），left/right 是较小的局部场景——**场景域差异
  大**，视觉特征（深度图分布、障碍布局）不匹配，导致零泛化。
- left/right 场景结构相似（同源拍摄，只是视角/位置不同），故有部分泛化（36-46%）。
- **这符合预期但必须如实报告**：当前模型是**场景特定**的，不具备跨场景泛化能力。
  对 ICRA 论文而言，这是重要边界——视觉 RL 在单一重建场景训练无法泛化到新场景。

## 待办

- [ ] 若需跨场景泛化证据：考虑 domain randomization / 多场景联合训练 / 更通用特征
      编码器（当前 CNN 是场景特定）
- [ ] 与 D3-lite（合成任务）的泛化对比：合成走廊可泛化（统一分布），真实重建场景不行
- [ ] 论文如实写"跨场景泛化失败，为当前方法的明确边界"，或用它论证 domain
      randomization 的必要性

## 文件

- `matrix.json`：3×3 矩阵完整数据（含 per-episode detail）
- 工具：`experiments/visual_necessity/eval_v3c_cross_scene.py`
- 服务器日志：`/root/px4-deploy/v3c_cross_scene.log`
