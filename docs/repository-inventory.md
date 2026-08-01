# Repository inventory

盘点日期：2026-08-01。

## 容量概览

| 路径 | 约占用 | 分类 |
| --- | ---: | --- |
| `data/` | 5.8 GB | 本地数据，已忽略 |
| `PX4-Autopilot-1.17.0/` | 431 MB | 本地第三方源树，已忽略 |
| `PX4-Autopilot-1.17.0.tar` | 378 MB | 本地离线包，已忽略 |
| `reports/` | 94 MB | 实验产物，部分跟踪 |
| `past/` | 8.1 MB | 历史归档 |
| `docs/` | 3.1 MB | 当前文档与论文 |
| `grad_nav-main/` | 1.4 MB | 第三方参考实现 |
| `rlproject-swift-improved/` | 1.3 MB | 核心项目 |
| `better_vibe_coding/` | 132 KB | 开发参考资料 |

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

- 筛选 `reports/` 中 70 个已跟踪二进制图，只保留关键图。
- 为实验结果建立 Phase V2/V3 索引。
- 检查 `past/` 与 `docs/` 的同名或重复材料。
- 决定 `better_vibe_coding/` 是继续冻结还是迁入历史归档。
- 在线核实 GitHub PR 状态；当前本机 GitHub CLI 认证失效。

