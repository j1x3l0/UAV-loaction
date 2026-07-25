# v2 进度日志

> 更新：随时 | 基于：MASTER_PLAN.md | 当前日期：2026-07-25

---

## 状态总览

| Phase | 步骤 | 状态 | 完成日期 |
|-------|------|------|---------|
| **Phase 0** | 代码构建 | ✅ 已完成 | 2026-07-25 |
| **Phase V1** | Baseline 训练 | ⬜ 就绪(待GPU服务器) | — |
| **Phase V2** | 衰减曲线 | ⬜ 未开始 | — |
| **Phase V3** | 鲁棒训练+对比+消融 | ⬜ 未开始 | — |
| **Phase V3b** | 消融实验 | ⬜ 未开始 | — |
| **Phase V3c** | 跨场景泛化 | ⬜ 未开始 | — |
| **Phase V4** | 论文 | ⬜ 未开始 | — |

> 状态标记：⬜ 未开始 | 🔵 进行中 | ✅ 已完成 | ❌ 已放弃 | ⏸️ 暂停 | ⚠️ 受阻

---

## Phase 0：代码构建

### Step 0.1：3DGS 场景数据就绪

| 任务 | 状态 | 备注 |
|------|------|------|
| 下载 GRaD-Nav 场景 (L0) | ➖ 不需要 | 改用已有 nerfstudio ckpts (见决策记录 D1) |
| 下载 Mip-NeRF360 (L1) | ➖ 不需要 | 同上 |
| 下载 Replica + 转3DGS (L1) | 🔵 | 18 场景 mesh+texture 已下载 (data/replica/)，3DGS 训练待定 |
| 已有场景 ckpt → .ply | ✅ | utils/extract_ply.py；4 场景 (gate_mid ×2, gate_left, gate_right) |
| 配置 Nerfstudio + gsplat | ✅ | nerfstudio 安装失败(PyAV) → 绕过：extract_ply + gsplat 直接渲染 |

**Gate G1**：`GS.render(pose)` 无报错 ✅ (2026-07-25, CPU 回退验证；GPU 路径待服务器 benchmark)

实测 (gate_mid_new_gs.ply, 368,965 Gaussians, 64×64)：
- 加载 1.2~1.4s | CPU step 23~25ms | depth [0.1, 20.0] 无 NaN

### Step 0.2：visual_drone_env.py

| 任务 | 状态 | 备注 |
|------|------|------|
| VisualDroneEnv 实现 | ✅ | mock + gsplat 双渲染器接口 |
| degradation_utils.py 实现 | ✅ | |

**验收**：`reset()+step()` 100步无crash ✅ (2026-07-25，训练循环 512 步 + eval 20ep 无 crash)

### Step 0.3：visual_ppo_agent.py

| 任务 | 状态 | 备注 |
|------|------|------|
| VisualEncoder (CNN) 实现 | ✅ | |
| VisualActorCritic 实现 | ✅ | |
| PPO.update() 适配 Dict 观测 | ✅ | |

**验收**：forward `(batch,1,64,64)` → `(batch,3)` ✅ (训练循环中 actor/critic loss 正常)

### Step 0.4：train_visual.py

| 任务 | 状态 | 备注 |
|------|------|------|
| 训练管线改造 | ✅ | 2026-07-25 补齐 --renderer/--ply CLI + ply 路径解析 |
| 100ep 冒烟测试 | ✅ | mock + 真实GS(CPU) 全链路跑通；saved_models/visual_ppo_best.pth |

**Gate G2**：100ep loss下降，SR>0% 🔵 部分 (管线联通已验证；SR>0% 待 Phase V1 完整训练确认)

### Step 0.5：eval_degradation.py

| 任务 | 状态 | 备注 |
|------|------|------|
| 批量评估脚本 | ✅ | mock 版已产出 eval_results/degradation_20260722_*；真实GS版随 Phase V2 重跑 |

---

## Phase V1：Baseline 训练

### Step 1.1：V1-Baseline-S1

| 项目 | 值 |
|------|-----|
| **状态** | ⬜ 就绪待跑 |
| **场景** | gate_mid (data/gs_data/ply_exports/gate_mid_new_gs.ply, 368,965 Gaussians) |
| **训练量** | 3000ep × 8 envs |
| **GPU** | GPU0 |
| **开始时间** | — |
| **结束时间** | — |
| **最终成功率** | — |
| **模型路径** | — |

**启动前检查清单**：① 服务器上先跑 `python envs/gs_renderer.py` 验证 GPU 路径 + benchmark fps → ② 100ep 冒烟确认 SR>0% → ③ 再启动 3000ep 正式训练。
启动命令：`python scripts/train_visual.py --episodes 3000 --envs 8 --renderer gsplat --ply data/gs_data/ply_exports/gate_mid_new_gs.ply`

### Step 1.2：V1-Baseline-multi

| 项目 | 值 |
|------|-----|
| **状态** | ⬜ |
| **场景** | garden + room0 |
| **训练量** | 3000ep |
| **GPU** | GPU1 |
| **开始时间** | — |
| **结束时间** | — |
| **最终成功率** | — |
| **模型路径** | — |

### Step 1.3：GRaD-Nav PPO 基线

| 项目 | 值 |
|------|-----|
| **状态** | ⬜ |
| **场景** | gate_mid |
| **GPU** | GPU2 |
| **结果** | — |

**Gate G3**：V1-Baseline SR > 80% ⬜

---

## Phase V2：衰减曲线

### 衰减曲线数据

| 退化轴 | 状态 | 数据路径 | 临界点(σ_c) | 曲线形状 |
|--------|------|---------|------------|---------|
| 高斯球稀疏化 | ⬜ | — | — | — |
| 分辨率降低 | ⬜ | — | — | — |
| 视角覆盖 | ⬜ | — | — | — |
| 光照偏移 | ⬜ | — | — | — |
| 深度噪声 | ⬜ | — | — | — |
| 组合退化 | ⬜ | — | — | — |

**Gate G4**：至少1轴出现 >80%→<20% 突变 ⬜

### 分析产出

| 产出 | 状态 | 路径 |
|------|------|------|
| 主图1：5条衰减曲线 | ⬜ | — |
| 附表1：临界点汇总 | ⬜ | — |
| 主图2：相变热力图 | ⬜ | — |
| 致命轴排序 | ⬜ | — |

---

## Phase V3：鲁棒训练 + 对比

### 训练模型

| 实验 | 状态 | 场景 | 开始 | 结束 | 模型路径 |
|------|------|------|------|------|---------|
| V3-Rand | ⬜ | garden+room0 | — | — | — |
| V3-Fixed | ⬜ | garden+room0 | — | — | — |
| V3-Curric | ⬜ | garden+room0 | — | — | — |
| V3-DDRL | ⬜ | gate_mid | — | — | — |
| V3-BC | ⬜ | garden+room0 | — | — | — |

### 对比分析

| 产出 | 状态 | 路径 |
|------|------|------|
| 主图3：鲁棒训练对比 | ⬜ | — |
| 附图：PPO vs DDRL vs BC | ⬜ | — |

---

## Phase V3b：消融实验

| 消融 | 状态 | 训练开始 | 训练结束 | 模型路径 | 结论 |
|------|------|---------|---------|---------|------|
| 无深度图 (RGB) | ⬜ | — | — | — | — |
| 无速度向量 | ⬜ | — | — | — | — |
| 浅CNN (1层) | ⬜ | — | — | — | — |
| 无特权Critic | ⬜ | — | — | — | — |

| 产出 | 状态 | 路径 |
|------|------|------|
| 附表2：消融对比矩阵 | ⬜ | — |

---

## Phase V3c：跨场景泛化

| 实验 | 状态 | 训练场景 | 测试场景 | 结果 |
|------|------|---------|---------|------|
| L1 跨场景 | ⬜ | garden | office0, room0, apartment0 | — |
| L3 大规模 (ScanNet+HM3D) | ⬜ | garden+room0 | ScanNet×5 + HM3D×10 | — |
| 极端边界测试 | ⬜ | — | 全场景×最差退化 | — |

| 产出 | 状态 | 路径 |
|------|------|------|
| 主图4：跨场景泛化矩阵 | ⬜ | — |

---

## Phase V4：论文

| 章节 | 状态 | 备注 |
|------|------|------|
| Related Work 大纲 | ⬜ | |
| Method §1-2 | ⬜ | |
| Method §3-4 | ⬜ | |
| Experiments §4.1 | ⬜ | |
| Experiments §4.2-4.5 | ⬜ | |
| Introduction | ⬜ | |
| Discussion + Conclusion | ⬜ | |
| 图表 finalize | ⬜ | |
| 初稿整合 | ⬜ | |

---

## GPU 使用记录

| 日期 | GPU0 | GPU1 | GPU2 | 备注 |
|------|------|------|------|------|
| | | | | |

---

## 问题日志

> 记录所有遇到的问题、原因、解决方案。按日期倒序。

| 日期 | Phase | 问题 | 严重度 | 状态 | 解决方案 |
|------|-------|------|--------|------|---------|
| 2026-07-25 | 0 | train_visual.py 无 --renderer/--ply 参数，文档中的 Phase V1 命令会报 unrecognized arguments；且 make_env 不传渲染器配置 → 静默回退 mock | 🔴 阻断 | ✅ 已修 | 补 CLI + make_env 透传 + resolve_ply_path (cwd/repo/ply_exports 三级解析)，全链路实测通过 |
| 2026-07-25 | 0 | gs_renderer GPU 路径未验证：gsplat 1.5.3 rasterization 返回 (colors, alphas, info) 而非 dict，outputs['rgb'] 会 TypeError；且每帧重复上传 ~20MB 高斯参数 | 🔴 高 | ✅ 已修 (待GPU验证) | 按 1.5.3 API 重写：RGB+ED + alpha 掩码空洞置 max_depth + __init__ 缓存 GPU 张量 + quats 归一化 |
| 2026-07-25 | 0 | venv312 未装 torch；实际训练用系统 Python 3.14 (torch 2.13.0+cpu, gsplat 1.5.3) | 🟡 低 | ✅ 已记录 | 服务器需装 CUDA 版 torch + gsplat |
| 2026-07-24 | 0 | nerfstudio 安装失败 (PyAV 编译问题) | 🟡 中 | ✅ 已绕过 | 不装 nerfstudio，utils/extract_ply.py 直接从 ckpt 提取 Gaussian 参数 → .ply，gsplat 直接渲染 |

---

## 决策记录

> 记录所有偏离原计划的决策。

| 日期 | 原计划 | 实际决策 | 原因 |
|------|--------|---------|------|
| 2026-07-25 | 下载 GRaD-Nav / Mip-NeRF360 场景训练 V1 (D1) | 用已有 4 个 nerfstudio ckpt (gate_mid ×2, gate_left, gate_right) 提取 .ply | 数据已在本地且与目标应用场景(穿越门)直接相关；省去下载+训练 3DGS 的 1-2 天 |
| 2026-07-25 | V1-Baseline-S1 场景 = Mip-NeRF360 garden | 改为 gate_mid_new (368,965 Gaussians) | 与 v1 子课题A 及 GRaD-Nav 对比实验的场景一致性 |
| 2026-07-24 | nerfstudio pipeline 渲染 | extract_ply.py + gsplat 直接渲染 .ply | nerfstudio 安装失败(PyAV)；gsplat 渲染不依赖 nerfstudio 运行时 |

---

## 风险触发记录

> 记录 MASTER_PLAN.md §十 中风险的实际触发情况。

| 日期 | 风险编号 | 触发情况 | 降级措施 |
|------|---------|---------|---------|
| | | | |

---

## 实验数据索引

> 所有产出的 CSV / JSON / 模型 / 图表 的路径汇总。

| 类别 | 路径 | 说明 |
|------|------|------|
| 3DGS 场景 | data/gs_data/ply_exports/ (4 × .ply, 16.7~23.9 MB) | gate_mid ×2, gate_left, gate_right；由 nerfstudio ckpt 提取 |
| 碰撞点云 | data/point_cloud/ (6 × .ply) | 碰撞检测用 |
| Replica | data/replica/ (18 场景 mesh+texture) | 待 3DGS 训练 (Phase V3c 候选) |
| 模型 | rlproject-swift-improved/saved_models/visual_ppo_best.pth | mock 冒烟训练产出 (非正式 baseline) |
| 退化评估 (mock) | rlproject-swift-improved/eval_results/degradation_20260722_* | mock 渲染器版衰减曲线，Phase V2 需用真实 GS 重跑 |

---

*最后更新：2026-07-25*
