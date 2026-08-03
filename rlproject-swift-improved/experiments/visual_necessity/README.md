# 视觉必要性重设计实验（隔离，可放弃）

> 本目录是隔离实验，**不修改任何核心代码**（`envs/`、`scripts/train_visual.py`、`core/`、`configs/`）。
> 失败直接删除本目录即可回退。设计草案见 `docs/visual-necessity-redesign.md`。

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

## D2：去目标向量 + gate 锚定（需 wrapper，后续）

`vec` 去 `target_dir` 后，目标必须深度可辨（gate 开口）。当前 `no_target_dir` 消融显示
0-2%（任务不可解）——单纯去向量不够，需把目标固定在 gate 并重训。暂缓。

## 回滚

删本目录 + 删除服务器端输出 JSON 即可，不影响主线。
