# 协作交接文档 — UAV 视觉导航 (v2/v3 阶段)

> 生成：2026-08-02 | 目标读者：新加入的协作开发者
> 核心目标：让你无需长时间摸索，就能在**不破坏门控纪律**的前提下接手有效工作。

---

## 1. 项目一句话定位

基于深度强化学习（PPO）的无人机 3D 视觉导航研究：无人机以 **3DGS 渲染的深度图**
为观测学习穿越真实场景（gate_mid）到达目标，重点研究**渲染质量退化下的鲁棒性**。
目标会议：ICRA 2027。

**当前阶段（2026-08-03）**：环境对齐门控、相机统一、对齐 V3（gentle curriculum 3 seeds）
与对照评估**已完成**；配置状态为 `aligned_v3_experiment_complete=true`、
`publication_ready=false`。当前主线是**补足论文统计证据**（逐 episode 配对检验、扩 seeds）
与**重设计视觉必要性验证**（当前消融显示策略主要靠向量输入，深度视觉只是中等贡献）。

---

## 2. 上手（5 分钟内）

```bash
# 克隆 + 同步
git clone https://github.com/j1x3l0/UAV-loaction.git
cd UAV-loaction
git checkout main && git pull

# 网络代理（国内网络访问 GitHub 需要，仓库级配置已写 .git/config）
# git config http.proxy http://127.0.0.1:7897   # Clash 端口，按需修改
```

**当前仓库状态（2026-08-03）**：

- 所有 V2/V3 核心 + PX4 平台代码已合并到 `main`（PR #6/#9），`main` 与 `origin/main` 同步
- 环境对齐、相机统一、对齐 V3、三方对照、输入消融均已提交
- 遗留本地分支 `codex/pr5-effective`（旧实验，已废弃，可删）

---

## 3. 核心架构（读这三个文件即可入门）

```
手机视频 → 3DGS重建(.ply) → gsplat深度渲染(64×64) → CNN编码器 → 共享MLP → 动作(3D推力)
                              ↑                                 ↑
                    degradation_utils(5退化轴)          PPO Update (GAE + Clip)
```

| 文件 | 职责 |
|------|------|
| `rlproject-swift-improved/envs/visual_drone_env.py` | 训练环境：观测/动作/质点动力学/7组件奖励/退化 |
| `rlproject-swift-improved/envs/scene_geometry.py` | 碰撞几何：Gaussian中心点云 + KD-tree、连通域、测地距离 |
| `rlproject-swift-improved/envs/gs_renderer.py` | gsplat 渲染器（GPU 1.8ms/帧，CPU 回退 23ms/帧） |
| `rlproject-swift-improved/core/visual_ppo_agent.py` | CNN 编码器 + Actor-Critic + PPO |
| `rlproject-swift-improved/scripts/train_visual.py` | 训练入口（3 seeds × 500，三类 checkpoint） |
| `rlproject-swift-improved/integrations/` | PX4 接口：MAVLink offboard、NED↔3DGS 对齐、策略桥接 |
| `rlproject-swift-improved/configs/px4_gate_mid_alignment.json` | PX4↔场景坐标锚点与相机内参 |

---

## 4. 当前门控进度（按顺序）

| # | 门禁 | 状态 | 备注 |
|--:|------|:----:|------|
| 1 | 接口单元测试 | ✅ | 坐标/限幅/NaN/心跳/超时 |
| 2 | PX4 SIH 部署 | ✅ | v1.17.0，250Hz，MAVLink 14550 |
| 3 | 旧 checkpoint 接口门禁 | ✅ | 只证链路，`aligned_v3_result=false` |
| 4 | 5帧轴向注册 | ✅ | 轴向判别通过 |
| 5 | 64×64 中心裁剪门禁 | ✅ | 正确转换 MAE 0.187 < 错误 0.202 |
| 6 | 30 位姿 GPU 注册统计 | ✅ | 覆盖率 30/30，22/30 逐位姿正确轴，`provisional_pass=true` |
| 7 | 飞行区最小净空检查 | ✅ | 0.45m 净空连通自由空间 153 m³ |
| 8 | 只读观测桥 | ✅ | PX4位姿→3DGS深度+策略向量，不发控制 |
| 9 | 遥测回放 + 定点悬停渲染 | ✅ | 160 悬停样本全有效深度 |
| 10 | 训练相机统一（fx≈97.14 + 目标偏航） | ✅ | 与观测桥 `camera_c2w` 一致 |
| 11 | clean 长验证 + 全量 gentle V3 3 seeds | ✅ | robust min 36-38% vs clean 14% |
| 12 | V3 三方对照 + 输入消融 | ✅ | 见下方「当前风险」 |
| — | **论文统计证据** | ⚠️ | 逐 episode 配对检验、5 seeds 待补 |
| — | **视觉必要性** | ⚠️ | 消融显示向量主导，需重设计任务再验证 |

> 门禁细节见 `PROGRESS.md` 末尾两节与 `reports/` 下对应报告目录。

---

## 5. 分工轨道

```
关键路径（服务器，GPU0 空闲 3090）
  逐 episode 统计证据（配对检验/5 seeds）→ 视觉必要性重验证 → 论文统计 → V3c 跨场景

并行轨道（本地 MacBook，全部无 GPU 要求）
  Track D 工程收尾（requirements/CI/测试）→ Track C 论文准备 → 场景资产预处理 → PX4 数据层
```

**GPU 结论**：视觉 PPO 仅 0.63M 参数（<2GB/种子），gsplat 渲染 265K 高斯 <1GB；
1 张 3090 可同时跑 3 种子 V3（6GB）+ Replica 3DGS 训练（8GB）。唯一显存大头是
Replica 3DGS 训练（4–12GB），与主任务错峰即可。详见会话中的 GPU 核算。

---

## 6. 可领取的任务

### A. 净空检查（关键路径，纯 CPU，优先）⭐

**目标**：验证 PX4 可飞区间映射到场景后，无碰撞净空不足。

- 入口：`rlproject-swift-improved/scripts/validate_scene_alignment.py`
- 输入：`configs/px4_gate_mid_alignment.json` + `scene_geometry`（Gaussian 点云 KD-tree）
- 运行：`python scripts/validate_scene_alignment.py --seed 20260728`
- 输出：每个映射体积网格点的最小碰撞距离，报告 <安全半径 的格子占比
- **验收**：全部网格点净空 ≥ 无人机碰撞半径（0.25m）+ 边距，否则列出违规坐标

### B. CI workflow 修复（并行，几分钟）⭐

**问题**：`.github/workflows/build-px4-sih-focal.yml` 触发条件绑定已删除的
`codex/phase-v2-v3-progress` 分支 → workflow 永远不会自动运行。

**要求**：改为 `workflow_dispatch` 手动触发 + `main` push 触发；避免长期绑定
临时分支名。验收：`gh workflow run` 能手动触发成功。

### C. 依赖声明（并行，几分钟）

仓库没有 `requirements.txt` / `pyproject.toml`。至少声明 `pymavlink`（
`integrations/mavlink_offboard.py` 依赖）。验收：`pip install -e .` 或
`pip install -r requirements.txt` 在干净环境可安装。

### D. 集成测试运行与补齐（并行）

- 现有测试：`integrations/test_mavlink_offboard.py`、`test_px4_scene_alignment.py`、
  `test_rl_policy_offboard.py`、`test_px4_offboard.py`
- 本机尚无 pytest：先 `pip install pytest` 跑通全部，修复失败项
- 验收：`pytest rlproject-swift-improved/ -q` 全绿

### E. 论文准备（Track C，长期）

- Related Work：按 `docs/reading-list-v2.md`（15篇，三级优先级）整理笔记
- 统计工具：Wilson CI、配对 bootstrap、McNemar 脚本（CPU 可跑，供 V3 正式评估用）
- 从已有 reports/ CSV/JSON 出图（matplotlib）

### F. 场景资产预处理（V3c 前置）

- `gate_left`/`gate_right` PLY 提取：`utils/extract_ply.py`（纯 CPU）
- mesh/SDF 碰撞导出、坐标校准测试、episode manifest 文档

---

## 7. 纪律与红线（违反=工作作废）

1. **论文主张纪律**：`publication_ready=false` 期间不宣称「论文结论已成立」。统计证据（逐 episode 配对检验、扩 seeds）与视觉必要性未过前，不进入 V3c 跨场景并宣称视觉泛化。不把旧 `legacy-unaligned` checkpoint 当新 V3 初始化
2. **旧 V1/V2/V3 全部是 `legacy-unaligned`**，只能做方法学诊断，不得写入论文正式结论
3. **门控纪律**：每阶段预注册验收门槛；同类操作连续失败两次即停止并记录，不反复重试
4. **禁止提交到 Git**：模型权重、训练日志、PID、`data/`（5.8GB）、PX4 源码包（431MB）、
   Replica 原始资产。`reports/` 只提交聚合结果 + 关键图表，二进制大图需人工筛选
5. **结论边界**：渲染质量 ≠ 物理真实性；PX4 接口通过 ≠ 视觉导航成功；三层仿真
   （gsplat 轻量 / PX4+Gazebo / Isaac）证据不得混称同一保真度
6. 认证/网络状态不明时不猜测、不重复启动服务器操作

---

## 8. 环境与资源

| 资源 | 规格 | 用途 |
|------|------|------|
| 服务器（Ubuntu 20.04） | 2×RTX 3090 (24GB), 240GB RAM, 3TB | 训练 + 评估 + PX4 SIH（一张 3090 被其他任务占用） |
| MacBook（开发机） | Apple Silicon | 代码、绘图、QGC 客户端、pymavlink 测试、CPU 轨道全部工作 |
| GitHub | j1x3l0/UAV-loaction | 所有代码与文档；大文件走 Release/制品存储 |

服务器训练环境：`/root/miniconda3/envs/myconda`（含 CUDA torch + gsplat）。
PX4 部署脚本：`scripts/px4-sih-server.sh`；部署说明 `docs/px4-sih-server-deployment.md`。

---

*详细背景：`MASTER_PLAN.md`（总计划）、`PROGRESS.md`（逐日进度与验收门槛）、
`CURRENT_PHASE_COMPLETION_REPORT.md`（legacy 阶段报告）、`docs/SIMULATION_PLATFORM_REQUIREMENTS.md`
（三级验证体系）、`CLAUDE.md`（AI 协作原则）。*
