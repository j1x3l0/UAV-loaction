# sv_1007 输入依赖消融（clean 5 seeds）

日期：2026-08-06。新主场景 sv_1007 的输入消融——验证深度、速度、目标方向各输入
对任务的必要性。这是视觉必要性叙事的直接证据（衔接 D1/D2 失败的旧叙事）。

## 方法

- **模型**：`saved_models/v3_sv1007/seed*_3000_final.pth`（clean 5 seeds）
- **评估**：`scripts/eval_v3_ablation.py --ablation all`，50ep/配置，对齐 fx≈97.14 相机，
  0.5m 碰撞半径，混合 clear/avoidance episodes（env 默认 0.5）
- **GPU0 only**（GPU1 保留）
- 工具：`run_sv1007_ablation.sh`；原始数据 `reports/px4_sv1007_ablation_20260806/seed{0-4}.json`

## 结果（SR 均值 5 seeds）

| 消融 | seed0 | seed1 | seed2 | seed3 | seed4 | 均值 | vs baseline |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **baseline** | 42 | 60 | 58 | 56 | 54 | **54.0%** | — |
| const_depth（去深度） | 34 | 36 | 34 | 40 | 42 | **37.2%** | −16.8pp |
| no_velocity（去速度） | 4 | 8 | 10 | 8 | 6 | **7.2%** | −46.8pp |
| no_target_dir（去目标） | 2 | 0 | 2 | 0 | 0 | **0.8%** | −53.2pp |
| all_inputs_ablated | 0 | 0 | 0 | 0 | 0 | **0.0%** | −54.0pp |

## 结论

1. **目标方向是最关键的输入**：去掉后 SR 崩塌到 0.8%（−53.2pp）——无目标信号任务
   不可解（与旧 D2 失败一致：去目标向量 + 静态深度图，feedforward 无法定位目标）。
2. **速度向量高度必要**：去掉后 7.2%（−46.8pp）。
3. **深度中等必要**：const_depth 仍 37.2%（−16.8pp）。深度在混合任务下有实质贡献，
   但不是唯一决定性的——与旧 D1 结论（纯避障下深度必要性未诱导）部分一致，但
   sv_1007 混合任务下深度的贡献（16.8pp）比旧场景更明显。
4. **视觉必要性叙事**：sv_1007 上深度提供 16.8pp 的增益，但目标/速度向量贡献更大。
   这与 D3-lite（稠密走廊，深度必要）形成对照——深度必要性依赖任务中的障碍密度。

## 待办

- [ ] 与 D3-lite 必要性结果（depth 4/5 seed 显著）合并，形成"深度必要性随障碍密度
      增强"的论文叙事
- [ ] 可选：`--avoidance-probability 1.0` 纯避障消融，隔离深度在避障中的必要性
      （对照 D1 的失败结论在 sv_1007 上是否复现）

## 文件

- `seed{0-4}.json`：各 seed 逐配置 SR
- `sv1007_input_ablation.png`：混合任务输入消融柱状图
- 服务器日志：`/root/px4-deploy/sv1007_ablation.log`
