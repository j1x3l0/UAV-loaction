# 视觉必要性重设计实验（隔离，可放弃）

> 本目录是隔离实验，**不修改任何核心代码**（`envs/`、`scripts/train_visual.py`、`core/`、`configs/`）。
> 失败直接删除本目录即可回退。设计草案见 `docs/visual-necessity-redesign.md`。

## sv_1007 低保真数据清洗

`clean_sv1007.py` 清洗低保真 sv_1007 重建中的噪声簇。低保真 3DGS 会产生
大量微小的孤立高斯斑点（1-3 voxel），它们不是真实障碍物，却会：
1. 在渲染深度图上产生虚假深度尖刺；
2. 把原始点云的"障碍簇计数"虚高（sv_1007 原始 0.5m 体素有 16 个连通分量，
   其中 13 个是噪声）。

清洗逻辑：去掉 ground 板层（z < `--ground-z`）后对剩余点做连通分量，
保留 `>= --min-cluster-voxels` 的障碍簇，丢弃噪声簇。输出渲染器 gsplat ply、
碰撞 ply（env 的 `--collision-ply` 需要的 xyz 格式）与碰撞点 npy，并可选
写结构化统计报告（`--report-json`）。

```bash
python clean_sv1007.py \
  --ply ../../data/gs_data/sv_1007_gate_mid/splatfacto/2024-10-07_145741/exports/splat/splat.ply \
  --out-ply ../../data/gs_data/sv_1007_gate_mid/cleaned/sv1007_clean.ply \
  --out-collision-ply ../../data/gs_data/sv_1007_gate_mid/cleaned/sv1007_clean_collision.ply \
  --out-npy ../../data/gs_data/sv_1007_gate_mid/cleaned/sv1007_clean.npy \
  --report-json ../../data/gs_data/sv_1007_gate_mid/cleaned/sv1007_clean_report.json
```

结果（voxel=0.5, ground_z=0.5, min_cluster_voxels=8）：**ground + 5 障碍簇**，
移除 13 个噪声簇（30 个噪声点）。`sv1007_alignment.json` 已从"16 障碍簇"
更正为清洗后计数。逐簇明细见 `reports/px4_sv1007_clean_20260804/`。

## D1：碰撞半径硬化扫描（首选，纯配置）

在纯避障 episode（avoid=1.0）下扫 `drone_collision_radius`，测 curriculum robust 模型的
baseline vs `const_depth`。若某半径下 `const_depth` 相对 baseline 崩塌（差 ≥30pp）而
baseline 仍可学，深度必要性即被诱导。

```bash
# 服务器，conda 环境
cd /root/rlproject-swift-improved
for R in 0.75 1.0 1.25; do
  /root/miniconda3/envs/myconda/bin/python scripts/eval_v3_ablation.py \
    --model cur_robust_seed1=saved_models/v3_aligned_gentle/seed1_robust_best.pth \
    --model cur_robust_seed2=saved_models/v3_aligned_gentle/seed2_robust_best.pth \
    --ablation baseline,const_depth --avoidance-probability 1.0 \
    --collision-radius $R \
    --alignment configs/px4_gate_mid_alignment.json \
    --ply /root/data/gs_data/ply_exports/gate_mid_new_gs.ply \
    --collision-ply /root/data/point_cloud/gate_mid_new.ply \
    --episodes 50 \
    --output /root/px4-deploy/camera-registration/necessity_scan_r${R}.json
done
```

验收：存在某 R 使 baseline SR ≥ 30% 且 `baseline − const_depth ≥ 30pp`。

**D1 结果（2026-08-03）：失败。** 纯避障下 radius {0.75, 1.0} 时 baseline 与 const_depth 一起崩塌（
8-40%，噪声主导，const_depth 偶而反高）；radius 1.25 时 `build_navigation_grid` 报
「largest free-space component is too small」（自由空间不足以建网格）。**碰撞半径硬化未
诱导深度必要性**——现有模型在 0.5m 训练，高半径对 baseline/const_depth 同等地过难。
要推进 D1 需在高半径下重训（昂贵且不确定），或转 D2/D3。

## D2：去目标向量 + gate 锚定（需 wrapper，后续）

`vec` 去 `target_dir` 后，目标必须深度可辨（gate 开口）。当前 `no_target_dir` 消融显示
0-2%（任务不可解）——单纯去向量不够，需把目标固定在 gate 并重训。暂缓。

## D2 结果（2026-08-03）：失败——去向量使任务不可解而非深度必要

D2 训练（无目标向量 + 固定 gate 目标 [0,0,1.73]，1000ep）后评估（50ep/单元）：

| 场景 | baseline（有深度） | const_depth |
|------|:---:|:---:|
| 纯空旷 (avoid=0) | 2% | 2% |
| 纯避障 (avoid=1) | 0% | 0% |

**baseline 都不可学**（2%/0%），不满足「baseline 可学 且 const_depth 崩塌」的门控。
**诊断**：深度图是静态场景，不标记抽象目标位置；去掉 `target_dir` 后，feedforward 策略
无法仅从「深度 + 距离奖励」定位目标 → 任务不可解，而非深度必要。D2 的假设（去向量→
深度必要）被实证否定。

**出路**：D2 需重构（目标必须深度可见且初始朝向一致 + 大幅加长训练，仍不确定）；
或转 D3-lite（合成稠密障碍走廊）；或维持弱化路线。详见 `docs/visual-necessity-redesign.md`。

## 回滚

删本目录 + 删除服务器端输出 JSON 即可，不影响主线。
