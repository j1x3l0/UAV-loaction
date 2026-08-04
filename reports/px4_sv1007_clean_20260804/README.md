# sv_1007 低保真点云清洗（16 → 5+ground 障碍簇）

日期：2026-08-05。对主场景 sv_1007 的 3DGS 重建点云做低保真清洗：
去掉**没有实际意义的噪声簇**（孤立的 1–3 体素小团），保留真实障碍与地面。

## 动机

`docs/two-week-icra-plan.md` 与 `experiments/visual_necessity/sv1007_alignment.json`
此前标注主场景为 **"16 障碍簇"**。该数字来自对原始点云做 0.5m 体素连通分量的直接计数，
但它把重建噪声也算进了障碍：实际的真实障碍簇明显更少。

## 方法

清洗脚本 `experiments/visual_necessity/clean_sv1007.py`（v2，2026-08-05）：

1. 丢弃 NaN/inf 点（低保真重建偶发）。
2. 用 `--ground-z 0.5` 把地面 slab（z<0.5m，68261 点）与障碍点分开——否则地面会把所有
   障碍粘成一个连通分量，无法数出真实簇数。
3. 对非地面点做 0.5m 体素连通分量标记。
4. 保留地面 + 体素数 ≥ `--min-cluster-voxels 8` 的障碍簇；丢弃其余噪声簇。

运行参数（复现）：

```bash
python experiments/visual_necessity/clean_sv1007.py \
  --ply data/point_cloud/sv_1007_gate_mid.ply \
  --voxel 0.5 --ground-z 0.5 --min-cluster-voxels 8 \
  --out-ply data/point_cloud/cleaned/sv_1007_gate_mid_clean.ply \
  --out-collision-ply data/point_cloud/cleaned/sv_1007_gate_mid_clean_collision.ply \
  --out-npy data/point_cloud/cleaned/sv_1007_gate_mid_clean.npy \
  --report-json data/point_cloud/cleaned/sv_1007_clean_report.json
```

## 结果

| 指标 | 原始 | 清洗后 |
|------|:---:|:---:|
| 总点数 | 265631 | 265601 |
| 地面 slab 点 | 68261 | 68261 |
| 非地面连通分量 | **18** | 5 障碍簇 + 13 噪声被移除 |
| 障碍簇（≥8 体素） | 5 | 5 |
| 移除噪声点数 | — | 30 |
| 导航自由网格节点 | 9673 | **9688**（+0.15%，噪声不占自由空间） |

### 保留的 5 个真实障碍簇

| # | 体素 | 点数 | 位置 (m) | 说明 |
|:---:|:---:|:---:|---|------|
| 3 | 4640 | 175798 | x[-10.6,10.0] y[-13.0,10.3] z[0.5,7.1] | 主结构（gate + 墙） |
| 6 | 36 | 156 | x[-10.3,-7.1] y[7.9,11.4] z[2.9,4.3] | 角落障碍 |
| 2 | 33 | 144 | x[-11.2,-9.5] y[2.2,5.3] z[6.1,7.5] | 角落障碍（高处） |
| 11 | 29 | 11997 | x[-1.1,-0.4] y[-1.1,1.0] z[0.5,1.9] | 中央低矮结构 |
| 17 | 19 | 9245 | x[3.5,4.9] y[-2.2,-0.5] z[0.5,1.9] | 中央低矮结构 |

### 移除的 13 个噪声簇

均为 1–3 体素、1–14 点的孤立小团，总计 30 点（原始点云的 0.011%）。
它们是低保真重建的浮动伪影，渲染成虚假深度尖刺、抬高表观障碍数，但对导航任务
没有实际意义。

## 结论

- **主场景真实障碍簇数：5 个 + 地面**，而非文档此前声称的 16 个。
- 清洗只移除 0.011% 的点，导航自由网格反而微增（噪声体素让出的空间），
  说明被移除的确实是纯噪声，不损失任何真实障碍。
- 文档口径已修正：`sv1007_alignment.json` 与 `docs/two-week-icra-plan.md`。

## 文件

- 清洗脚本：`experiments/visual_necessity/clean_sv1007.py`
- 清洗产物（`data/` 下，gitignore，服务器同步）：
  - `data/point_cloud/cleaned/sv_1007_gate_mid_clean.ply`（渲染器 gsplat ply）
  - `data/point_cloud/cleaned/sv_1007_gate_mid_clean_collision.ply`（碰撞 ply）
  - `data/point_cloud/cleaned/sv_1007_gate_mid_clean.npy`（碰撞点数组）
  - `data/point_cloud/cleaned/sv_1007_clean_report.json`（结构化统计）
- 对比图：`sv1007_clean_before_after.png`（俯视密度 + 簇大小分布）
