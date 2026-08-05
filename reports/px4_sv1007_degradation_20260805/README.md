# sv_1007 主场景退化鲁棒性评估（clean vs curriculum，5 seeds）

日期：2026-08-05~06。新主场景 sv_1007（完整 gate，22×25×9m）上的退化鲁棒性
对比。这是主场景切换后的**公平对比**（clean 3000ep vs curriculum 3000ep，各 5 seeds）。

## 方法

- **clean**：`saved_models/v3_sv1007/seed*_3000_final.pth`（5 seeds，best_SR 43/60/55/64/44）
- **curriculum**：`saved_models/v3_sv1007_curriculum/seed*_robust_best.pth`（5 seeds，
  训练期间鲁棒性最优 checkpoint；curriculum best_SR 41/40/51/34/36）
- **评估**：`scripts/eval_degradation.py --axis all`，7 轴 × 5 档 × 50ep，对齐 fx≈97.14
  相机，0.5m 碰撞半径。GPU0 only（GPU1 保留）。
- 工具：`run_sv1007_degradation.sh`；原始数据 `reports/px4_sv1007_degradation_20260805/`。

## 结果（clean − curriculum，SR 均值 5 seeds）

### 无退化档（光照 0EV）：**+20.8pp**（clean 52.4% vs curriculum 31.6%）

各 seed 无退化 SR：clean `40 60 56 54 52`；curriculum `34 30 36 32 26`。

### 各轴关键档位

| 轴 | 档位 | clean | curriculum | 差(clean−cur) |
|----|------|:---:|:---:|:---:|
| 高斯稀疏化 | 100% | 52.4 | 31.6 | **+20.8** |
| 高斯稀疏化 | 5% | 33.2 | 26.4 | +6.8 |
| 高斯稀疏化 | 2% | 20.4 | 16.8 | +3.6 |
| 分辨率 | 64px | 52.4 | 31.6 | **+20.8** |
| 分辨率 | 16px | 46.0 | 30.0 | +16.0 |
| 分辨率 | 2px | 35.2 | 26.4 | +8.8 |
| 深度噪声 | 0.0σ | 52.4 | 31.6 | **+20.8** |
| 深度噪声 | 1.0σ | 46.8 | 33.2 | +13.6 |
| 光照 | 0/4EV | 52.4 | 31.6 | **+20.8**（不变，深度模型天然不变） |
| 视角不确定 | 360° | 52.4 | 31.6 | **+20.8** |
| 视角不确定 | 45° | 19.2 | 17.6 | +1.6 |
| 深度大面积失效 | 50% | 5.6 | 5.6 | +0.0 |
| 相机遮挡 | 50% | 0.0 | 3.2 | −3.2 |

## 关键发现：与旧场景结论相反 ⚠️

**在 sv_1007 上，clean 全面优于 curriculum**——几乎每个轴每档 clean 都领先
10~21pp。这与旧主场景 gate_mid_new 的结论**相反**（那里 curriculum robust-best 在
视角轴显著优于 clean：clean 45° 掉到 18%，curriculum 不掉）。

可能原因（待验证）：
1. **curriculum 的 clean SR 天然低**（31.6% vs clean 52.4%）：curriculum 训练混合低
   尺度退化样本，牺牲了 clean SR。旧场景里这个损失被退化鲁棒性收益弥补，sv_1007 上
   没有出现收益。
2. **sv_1007 更复杂**（完整 gate，大场景），scale_curriculum 的退化采样干扰更大，
   robust_best 未达到旧场景的鲁棒水平。
3. 需要核对 curriculum 训练日志的 robust eval（min/mean SR）确认 robust_best 的鲁棒
   能力是否真实低于 clean。

## checkpoint 诊断（2026-08-06）：结论稳健

核对 curriculum final vs robust_best 在无退化档的独立评估（50ep）：

| 模型 | seed0 | seed1 | seed2 | seed3 | seed4 | 均值 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| curriculum **robust_best** | 34 | 30 | 36 | 32 | 26 | **31.6%** |
| curriculum **final** | 36 | 32 | 50 | 34 | 28 | **36.0%** |
| clean final | 40 | 60 | 56 | 54 | 52 | **52.4%** |

- final 比 robust_best 略好（36.0 vs 31.6），但**仍远低于 clean 的 52.4%**。
- 训练日志中 robust eval（20ep）末段 1x≈50% 是**内评估的乐观值**（20ep 方差大、
  且 robust eval 环境与独立评估对齐方式可能不同），不代表 robust_best 的真实独立性能。
- **结论**：curriculum 在 sv_1007 上 clean SR 就是低（不是 checkpoint 选择问题）。
  sv_1007 的 scale_curriculum 训练牺牲了干净性能，且未带来可测的退化鲁棒性收益。

## 待办

- [x] 核对 curriculum 训练 robust eval 曲线（结论：final≈robust_best≈36%，仍低于 clean 52.4%）
- [ ] 决定：sv_1007 主场景的退化结论如何入论文（"curriculum 优势不泛化到复杂场景"，
      诚实报告 clean 全面占优）
- [ ] 可考虑：curriculum 在 sv_1007 为何失效——场景复杂度 / 退化混合比例 / 训练时长

## 文件

- `clean/seed{0-4}.csv`、`curriculum/seed{0-4}.csv`：各 seed 逐轴逐档 SR
- `summarize.py`、`key_diffs.py`：汇总脚本
- 服务器日志：`/root/px4-deploy/sv1007_degradation.log`
