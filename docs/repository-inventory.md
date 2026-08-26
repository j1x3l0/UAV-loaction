# Repository inventory

盘点日期：2026-08-04（首次：2026-08-01）。

## 容量概览

| 路径 | 约占用 | 分类 |
| --- | ---: | --- |
| `data/` | 5.8 GB | 本地数据，已忽略 |
| `PX4-Autopilot-1.17.0/` | 431 MB | 本地第三方源树，已忽略 |
| `PX4-Autopilot-1.17.0.tar` | 378 MB | 本地离线包，已忽略 |
| `reports/` | 4.9 MB | 实验产物，部分跟踪 |
| `past/` | 8.4 MB | 历史归档（含 2026-08-04 移入的 better_vibe_coding/） |
| `docs/` | 3.1 MB | 当前文档与论文 |
| `grad_nav-main/` | 1.4 MB | 第三方参考实现 |
| `rlproject-swift-improved/` | 1.3 MB | 核心项目 |

Git 当前跟踪约 21 MB 的工作树内容，其中 `reports/` 约 7 MB、169 个文件。
仓库对象数据库约 583 MB，说明历史中存在较大的已提交对象；后续应避免继续提交
数据、模型和第三方源码包。

## 整理保护项

- 当前核心代码目录暂不改名，避免破坏 Python import 和脚本工作目录。
- `data/`、PX4 源树及压缩包不移动、不删除。
- `past/` 在完成重复内容比对前不删除。
- 已暂存的 MAVLink 实现和测试属于开发中改动，不纳入目录整理提交。
- `reports/` 中被阶段报告引用的结果在建立替代索引前不移动。

## 已知待办

- ✅ 已核实 GitHub 功能与 PR 状态（2026-08-04）：git push/gh CLI 均正常（认证已修复）；PR #13/#14 有用部分已本地合并，#10 冗余待关，#11 requirements 位置待统一。
- ✅ 已建立 Phase V3/PX4 报告索引（`reports/README.md` 更新至 2026-08-04）。
- ✅ 已筛选 `reports/` 二进制图（2026-08-04）：95 张 → 26 张。删除 69 张 legacy-unaligned/废弃路线图（v3_scale_*、v3_ckptfix_*、phase_v2_*、v3_clean_recovery、v3_scale_curriculum_gate），保留 plot_gallery_20260802（20 张）+ px4 对齐主线证据（注册 montage 4 张、对齐退化 2 张）；各报告 JSON 聚合结果全部保留。跟踪二进制 12MB → ~5MB。
- ✅ 已完成 `past/` 与 `docs/` 重复材料比对（2026-08-04）：无内容级重复；同名文件（knowledge-map、understanding-gate）为 v1/v2 版本关系，以 docs/ 为现行版；归档说明见 `past/README.md`。
- ✅ `better_vibe_coding/` 已归档（2026-08-04）：外部项目、全仓库零引用，由根目录移入 `past/better_vibe_coding/`（归档而非冻结，避免根目录误导新人）。

