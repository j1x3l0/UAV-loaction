# v2 进度日志

> 更新：随时 | 基于：MASTER_PLAN.md | 当前日期：2026-07-29

---

## 状态总览

| Phase | 步骤 | 状态 | 完成日期 |
|-------|------|------|---------|
| **Phase 0** | 代码构建 | ✅ 已完成 | 2026-07-25 |
| **Phase V1** | Baseline 训练 | ✅ 已完成（3 seeds） | 2026-07-27 |
| **Phase V2** | 衰减曲线 | ✅ 扩展结构退化后找到明确相变 | 2026-07-27 |
| **Phase V3** | 鲁棒训练+对比+消融 | 🔵 均匀随机化评估完成，发现0.5×跨种子不稳定 | — |
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

**Gate G2**：100ep loss下降，SR>0% ✅（Phase V1 三种子训练及独立评估已确认）

### Step 0.5：eval_degradation.py

| 任务 | 状态 | 备注 |
|------|------|------|
| 批量评估脚本 | ✅ | mock 版已产出 eval_results/degradation_20260722_*；真实GS版随 Phase V2 重跑 |

---

## Phase V1：Baseline 训练

### Step 1.1：V1-Baseline-S1

| 项目 | 值 |
|------|-----|
| **状态** | ✅ 已完成（以 3×500 多种子替代单次 3000ep） |
| **场景** | gate_mid (`gate_mid_new_gs.ply`，服务器实载 265,631 Gaussians) |
| **训练量** | 500ep × 8 envs × 3 seeds |
| **GPU** | GPU0 |
| **开始时间** | 2026-07-27 05:25（服务器时间） |
| **结束时间** | 2026-07-27 06:14（最晚模型写入时间） |
| **训练末次平均成功率** | 80.0%（seed 0/1/2：80% / 83% / 77%） |
| **独立 clean 成功率** | 82.7%（496/600，Wilson 95% CI 79.4–85.5%） |
| **模型路径** | 服务器：`saved_models/b0_logstd_fixed_alpha001_3x500_20260727_052519/` |

训练后的策略熵为 2.40–2.65（目标 2.5），确认 `log_std` 不再锁死在
`std=1 / entropy=4.26`。

### Step 1.1a：高熵基线与修复版独立 clean 对比

评估设置：真实 gsplat、相同的 200 个独立 episode seeds（base seed
`20260728`）、确定性策略；每组 3 个训练种子，共 1,200 episodes。

| 组别 | seed0 SR | seed1 SR | seed2 SR | 汇总 SR | 汇总 CR | 平均奖励 |
|------|---------:|---------:|---------:|--------:|--------:|---------:|
| 高熵基线（entropy=4.26） | 89.5% | 79.5% | 78.5% | 82.5%（495/600，95% CI 79.3–85.3%） | 17.5% | 405.02 |
| 修复版（目标 entropy=2.5） | 83.5% | 82.5% | 82.0% | 82.7%（496/600，95% CI 79.4–85.5%） | 17.3% | 408.32 |

配对结果：426 个 episode 两组均成功，35 个均失败，高熵独有成功 69
个，修复版独有成功 70 个。两组 clean 成功率实质持平（差
`+0.17 pp`），但修复版跨 seed 波动更小，且消除了不可训练的方差上限锁死。

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

**Gate G3**：V1-Baseline SR > 80% ✅（独立 clean：82.7%，600 episodes）

---

## Phase V2：衰减曲线

### 衰减曲线数据

| 退化轴 | 状态 | 数据路径 | 临界点(σ_c) | 曲线形状 |
|--------|------|---------|------------|---------|
| 高斯球稀疏化 | ✅ | `reports/phase_v2_formal_5x5x50_20260727_082212/` | >2%（未跌破50%） | 近似平坦 |
| 分辨率降低 | ✅ | 同上 | <2px（未跌破50%） | 极端档下降 6 pp |
| 视角覆盖 | ✅ | 同上 | <45°（未跌破50%） | 极端档下降 2 pp |
| 光照偏移 | ✅ 负对照 | 同上 | N/A | 完全平坦，符合 depth-only 预期 |
| 深度噪声 | ✅ | 同上 | >1.0σ（未跌破50%） | 近似平坦 |
| 组合退化 | ⬜ | — | — | — |

**Gate G4**：至少1轴出现 >80%→<20% 突变 ✅。扩展结构退化后，
深度尺度从 1.0× 的 83.5%（200ep）降至 0.25× 的 13.5%
（Wilson 95% CI 9.4–18.9%）；50% 与20% 的离散临界档均为 0.25×，
相变区间位于 0.5×–0.25×。

正式设置：修复版 seed0 最佳模型、真实 gsplat、5轴×5档×50 episodes，
各档使用相同 base seed `20260728`，总计 1,250 episodes。

真实GS输入消融（seed2 高熵最佳模型，200 episodes/config）：

| 配置 | SR | CR | Timeout | 结论 |
|------|---:|---:|--------:|------|
| baseline | 82.0% | 18.0% | 0.0% | clean 参考 |
| const depth | 71.0% | 14.5% | 14.5% | 深度视觉贡献约 11 pp |
| no target direction | 0.0% | 15.5% | 84.5% | 目标方向是任务必要输入 |
| both | 0.0% | 1.0% | 99.0% | 两类输入同时移除后任务不可解 |

### 分析产出

| 产出 | 状态 | 路径 |
|------|------|------|
| 主图1：5条衰减曲线 | ✅ | `reports/phase_v2_formal_5x5x50_20260727_082212/degradation_20260727_082215_all_axes.png` |
| 附表1：临界点汇总 | ✅ | `reports/phase_v2_formal_5x5x50_20260727_082212/critical_analysis.json` |
| 主图2：相变热力图 | ⬜ | — |
| 致命轴排序 | ✅（当前范围） | 分辨率（−6 pp）> 视角覆盖（−2 pp）> Gaussian/深度噪声/光照（0 pp） |

扩展结构退化确认：

| 退化轴 | 正常档 SR | 极端/关键档 SR | 结论 |
|--------|----------:|---------------:|------|
| 深度大面积失效 | 68% | 90%失效：68% | 非致命；出现4%超时 |
| 底部相机遮挡 | 68% | 75%遮挡：84% | 非有效负向退化，固定遮挡反而提供提示 |
| 深度尺度偏差 | 83.5%（200ep） | 0.25×：13.5%（200ep） | **最致命轴，明确相变** |
| 组合退化 | 68% | severity 0.5/0.75：54% | 有下降但未过50%，且1.0档非单调 |

---

## Phase V3：鲁棒训练 + 对比

### 训练模型

| 实验 | 状态 | 场景 | 开始 | 结束 | 模型路径 |
|------|------|------|------|------|---------|
| V3-Rand-Scale | ✅ 训练完成 | gate_mid | 2026-07-27 10:17 | 2026-07-27 | 服务器 `saved_models/v3_scale_rand_3x500_20260727_101721/` |
| V3-Weighted-Scale | ❌ 完成但未通过验收 | gate_mid | 2026-07-28 07:14 | 2026-07-28 08:41 | 服务器 `saved_models/v3_scale_weighted_3x500_20260728_071450/` |
| V3-Fixed | ⬜ | garden+room0 | — | — | — |
| V3-Curric | ⚠️ robust-best三项通过，clean差2.17 pp | gate_mid | 2026-07-28 10:11 | 2026-07-29 02:40 | 服务器 `saved_models/v3_curriculum_ckptfix_3x500_20260728_133555/` |
| V3-DDRL | ⬜ | gate_mid | — | — | — |
| V3-BC | ⬜ | garden+room0 | — | — | — |

V3-Rand-Scale 训练摘要（clean 评估）：

| Seed | 最终 SR | 训练中最佳 SR | 最终 entropy | 备注 |
|-----:|--------:|--------------:|--------------:|------|
| 0 | 70% | 87% | 2.68 | 中后期出现明显波动 |
| 1 | 75% | 80% | 2.67 | 中后期出现明显波动 |
| 2 | 82% | 84% | 2.43 | 相对稳定 |
| 平均 | 75.7% | 83.7% | 2.59 | 最佳 checkpoint 已保留 |

三个最佳 checkpoint 的五档独立评估已完成：真实 gsplat、每个训练 seed
每档 200 episodes、各档共享 base seed `20260728`，共 3,000 episodes。

| 深度尺度 | 汇总 SR（600ep） | Wilson 95% CI | CR | Timeout | 原模型 SR | 变化 |
|---------:|-----------------:|--------------:|---:|--------:|----------:|-----:|
| 1.0× | 81.83% | 78.55–84.71% | 18.17% | 0.00% | 83.5% | −1.67 pp |
| 0.75× | 81.17% | 77.84–84.09% | 18.83% | 0.00% | 80.5% | +0.67 pp |
| 0.5× | 57.17% | 53.17–61.07% | 20.33% | 22.50% | 73.0% | −15.83 pp |
| 0.25× | 76.17% | 72.60–79.40% | 19.33% | 4.50% | 13.5% | +62.67 pp |
| 0.1× | 77.33% | 73.82–80.50% | 20.00% | 2.67% | 13.5% | +63.83 pp |

**验收结论：部分通过。** clean 几乎保持（−1.67 pp），0.25× 和 0.1×
分别提升 62.67/63.83 pp；但 seed0 在 0.5× 仅 14.5% SR、66.5%
超时，seed1/2 同档为 75.0%/82.0%，说明均匀随机化产生跨 seed、
非单调的中间尺度鲁棒性空洞。

完整数据：`reports/v3_scale_eval_3x5x200_20260728_065821/`。
下一步采用课程式或重加权尺度采样：增加 0.5× 相变区附近权重，降低
0.25×/0.1× 比例；当前保留 seed2 作为最稳定 checkpoint。

V3-Weighted-Scale 按 `1.0/0.75/0.5/0.25/0.1× =
20%/20%/40%/10%/10%` 完成三个训练 seed；服务器测试 16/16
通过，2,000 次采样中 0.5× 占 40.0%。随后按相同协议完成 3,000
episodes 独立验收：

| 深度尺度 | 汇总 SR（600ep） | Wilson 95% CI | Timeout | 相对均匀V3 |
|---------:|-----------------:|--------------:|--------:|-----------:|
| 1.0× | 77.00% | 73.47–80.19% | 1.83% | −4.83 pp |
| 0.75× | 76.83% | 73.29–80.03% | 1.00% | −4.33 pp |
| 0.5× | 73.67% | 70.00–77.03% | 1.83% | +16.50 pp |
| 0.25× | 51.17% | 47.17–55.15% | 22.83% | −25.00 pp |
| 0.1× | 60.17% | 56.20–64.01% | 13.83% | −17.17 pp |

0.5× 的单 seed SR 为 76.5%/64.0%/80.5%，已修复原先的 seed0
空洞；但 clean 未达 80%，两个极端档未达 70%，且极端档 timeout
超过 10%，因此总体验收失败。下一步不再盲跑固定概率 3×500，而是先实现
带概率日志的阶段式 curriculum，并以单 seed 小规模试跑作为启动门槛。

完整数据：`reports/v3_scale_weighted_eval_3x5x200_20260728_084229/`。

### V3-Curriculum 小规模门控

阶段概率（尺度顺序均为 `1.0/0.75/0.5/0.25/0.1×`）：

- foundation（0–30%）：35%/25%/20%/10%/10%
- transition（30–70%）：25%/20%/30%/15%/10%
- robustness（70–100%）：25%/15%/25%/20%/15%

seed2×200 训练最终 clean SR 85%、entropy 2.80。最佳 checkpoint 的真实
gsplat 五档×100 episodes 门控结果：

| 深度尺度 | SR | Wilson 95% CI | CR | Timeout |
|---------:|---:|--------------:|---:|--------:|
| 1.0× | 83% | 74.45–89.11% | 17% | 0% |
| 0.75× | 80% | 71.12–86.66% | 20% | 0% |
| 0.5× | 82% | 73.33–88.30% | 18% | 0% |
| 0.25× | 87% | 79.02–92.24% | 13% | 0% |
| 0.1× | 78% | 68.93–85.00% | 22% | 0% |

**门控通过：** clean≥80%、0.5×≥70%、0.25×/0.1×≥65%、
所有 timeout≤10% 四项均满足。下一步进入 V3-Curriculum
3 seeds×500 正式训练，之后重复 3×5×200 独立评估。

完整数据：`reports/v3_scale_curriculum_gate_seed2_5x100_20260728_103900/`。

### V3-Curriculum 正式 3×500 验收

三个训练 seed 的 clean 最佳 SR 为 85%/86%/90%。使用各自 clean 最佳
checkpoint 完成真实 gsplat 3×5×200（3,000 episodes）评估：

| 深度尺度 | 汇总 SR（600ep） | Wilson 95% CI | Timeout | 相对均匀V3 |
|---------:|-----------------:|--------------:|--------:|-----------:|
| 1.0× | 78.33% | 74.86–81.44% | 0.00% | −3.50 pp |
| 0.75× | 78.00% | 74.51–81.13% | 0.17% | −3.17 pp |
| 0.5× | 68.00% | 64.16–71.61% | 9.83% | +10.83 pp |
| 0.25× | 78.17% | 74.69–81.29% | 0.00% | +2.00 pp |
| 0.1× | 63.33% | 59.40–67.09% | 12.50% | −14.00 pp |

正式验收四项均未通过；0.5× 的单 seed SR 为
75.5%/75.0%/53.5%。但本轮暴露出 checkpoint 选择混杂：
`train_visual.py` 只按 clean SR 保存最佳模型，seed1/2 的 best 文件在
robustness 阶段结束前已经写入。因此当前结果不能等同于“最终 curriculum
策略”的公平评估。

下一步先修复为同时保存 final checkpoint，并使用小规模多尺度验证分数选择
robust-best checkpoint；在此之前不再启动完整 3×500。

完整数据：`reports/v3_scale_curriculum_eval_3x5x200_20260728_124659/`。

### Checkpoint 选择修复与三方对照

已修复为同时保存 clean-best、五尺度 min/mean 选择的 robust-best 和 final。
真实 gsplat 短跑确认三类文件均生成且可加载；正式重跑后完成同一批模型三方
对照：

| 深度尺度 | Clean-best SR | Final SR | Robust-best SR | Robust 95% CI |
|---------:|--------------:|---------:|---------------:|--------------:|
| 1.0× | 73.67% | 70.00% | 77.83% | 74.34–80.97% |
| 0.75× | 72.00% | 72.67% | 78.83% | 75.39–81.91% |
| 0.5× | 69.00% | 71.00% | 77.00% | 73.47–80.19% |
| 0.25× | 65.00% | 75.33% | 75.83% | 72.25–79.09% |
| 0.1× | 72.33% | 77.00% | 76.00% | 72.42–79.25% |

robust-best 在所有尺度均优于 clean-best（+3.67 至 +10.83 pp），且每档
timeout≤1.5%；0.5× 单 seed SR 为 77.5%/67.0%/86.5%。

正式标准四项中通过三项：所有汇总档≥70%、每 seed 0.5×≥60%、
所有 timeout≤10%；仅 clean 汇总 77.83% 未达到80%，差2.17 pp。
这证明 checkpoint 修复有效，剩余问题已收敛为小幅 clean 性能缺口。

下一步先测试带 clean 下限约束的 robust checkpoint 选择，或从 robust-best
做短程 clean fine-tuning；暂不直接再跑完整 3×500。

完整对照：`reports/v3_checkpoint_selection_comparison_20260729/`。

### Clean-recovery 单 seed 收尾

从 seed0 robust-best 以 `60/15/10/10/5%` 尺度概率、`3e-5` 学习率继续
训练100 updates。训练中 clean 最佳仅77%，最终73%。随后执行唯一一次真实
gsplat 五尺度×100门控：

| 深度尺度 | SR | CR | Timeout |
|---------:|---:|---:|--------:|
| 1.0× | 75% | 25% | 0% |
| 0.75× | 74% | 25% | 1% |
| 0.5× | 78% | 21% | 1% |
| 0.25× | 78% | 22% | 0% |
| 0.1× | 73% | 27% | 0% |

退化档与 timeout 门槛通过，但 clean 仅75%，未达到80%，且未改善原 seed0
robust-best 的正式 clean 结果。按照预先约定的停止规则，clean-recovery
路线已终止，不扩展到3 seeds、不再重跑。

完整结果：`reports/v3_clean_recovery_gate_seed0_5x100_20260729_053619/`。

### 对比分析

| 产出 | 状态 | 路径 |
|------|------|------|
| 主图3：鲁棒训练对比 | 🔵 | `reports/v3_scale_eval_3x5x200_20260728_065821/`（分 seed 图已完成，待汇总图） |
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
| 2026-07-26 | 3×500 高熵基线 | 其他任务占用 | — | run `b0_entropy_sign_fixed_3x500_20260726_191516` |
| 2026-07-27 | 3×500 熵修复版 + 6×200 clean 评估 | 其他任务占用 | — | 训练与评估均完成，GPU0 已释放 |

---

## 问题日志

> 记录所有遇到的问题、原因、解决方案。按日期倒序。

| 日期 | Phase | 问题 | 严重度 | 状态 | 解决方案 |
|------|-------|------|--------|------|---------|
| 2026-07-27 | V1 | `std=clamp(exp(log_std), max=1)` 在 `log_std>0` 后产生零梯度，策略熵永久锁死 4.26 | 🔴 高 | ✅ 已修 | forward 使用 `exp(log_std)`，每次 optimizer step 后直接约束参数；初始 alpha 由 0.1 降至 0.01；3 项回归测试通过 |
| 2026-07-27 | V2 | Pillow 对 uint16 (`I;16`) 深度图执行 bilinear resize 报 `ValueError: image has wrong mode` | 🔴 阻断 | ✅ 已修 | 分辨率退化改用 float32 Pillow `F` mode，试跑和正式评估通过 |
| 2026-07-27 | V2 | 五轴预设极端档未使 SR 跌破 50%/20%，无法定位相变临界点 | 🟡 中 | ⚠️ 待扩展 | 扩展极端档并加入组合/遮挡/深度失效退化后复测，再决定 V3 |
| 2026-07-27 | V2 | 组合退化 severity=1.0 的 SR 高于0.5/0.75档，曲线非单调 | 🟡 中 | ✅ 已识别 | 不用组合轴确定临界点；采用单变量深度尺度200ep确认实验 |
| 2026-07-27 | V3 | 均匀极端尺度随机化导致 clean SR 大幅波动（单次评估最低2%） | 🟡 中 | 🔵 待评估 | 保留最佳 checkpoint；完成五档鲁棒评估后决定课程式/加权采样 |
| 2026-07-27 | V1 | canonical `train_visual.py` 依赖缺失的 `utils.metrics`，且调用缺失的 `get_actions_batch` | 🔴 阻断 | ✅ 已修 | 补齐 Wilson CI 工具和批量 CNN 推理接口，服务器真实GS训练验证 |
| 2026-07-26 | V1 | 自适应熵更新 loss 符号反向，高熵时 alpha 反而增大 | 🔴 高 | ✅ 已修 | 使用 `loss = log_alpha * (entropy - target_entropy)` 并添加方向回归测试 |
| 2026-07-25 | 0 | train_visual.py 无 --renderer/--ply 参数，文档中的 Phase V1 命令会报 unrecognized arguments；且 make_env 不传渲染器配置 → 静默回退 mock | 🔴 阻断 | ✅ 已修 | 补 CLI + make_env 透传 + resolve_ply_path (cwd/repo/ply_exports 三级解析)，全链路实测通过 |
| 2026-07-25 | 0 | gs_renderer GPU 路径未验证：gsplat 1.5.3 rasterization 返回 (colors, alphas, info) 而非 dict，outputs['rgb'] 会 TypeError；且每帧重复上传 ~20MB 高斯参数 | 🔴 高 | ✅ 已修 (待GPU验证) | 按 1.5.3 API 重写：RGB+ED + alpha 掩码空洞置 max_depth + __init__ 缓存 GPU 张量 + quats 归一化 |
| 2026-07-25 | 0 | venv312 未装 torch；实际训练用系统 Python 3.14 (torch 2.13.0+cpu, gsplat 1.5.3) | 🟡 低 | ✅ 已记录 | 服务器需装 CUDA 版 torch + gsplat |
| 2026-07-24 | 0 | nerfstudio 安装失败 (PyAV 编译问题) | 🟡 中 | ✅ 已绕过 | 不装 nerfstudio，utils/extract_ply.py 直接从 ckpt 提取 Gaussian 参数 → .ply，gsplat 直接渲染 |

---

## 决策记录

> 记录所有偏离原计划的决策。

| 日期 | 原计划 | 实际决策 | 原因 |
|------|--------|---------|------|
| 2026-07-27 | 单种子 3000ep 后进入 V2 | 先用 3×500 多种子建立可重复 baseline，并以 6×200 独立 clean 评估过 Gate G3 | 多种子和独立评估比训练中单次 20/100 episodes 更可靠；可先发现并修复熵锁死 |
| 2026-07-27 | 将高熵训练视为熵修复实验 | 归档为高熵基线，另跑 `initial_alpha=0.01` + 可训练 `log_std` 修复版 | 符号虽正确，但硬 clamp 仍把熵锁死在 4.26 |
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
| 高熵基线 | `reports/high_entropy_baseline_3x500_20260726_191516/` | 3×500 完整日志、最佳模型、README |
| 熵对比 clean 评估 | `reports/entropy_clean_6x200_20260727_081142/` | 6 模型×200 episodes 的 JSON、CSV、日志 |
| Phase V2 试跑 | `reports/phase_v2_smoke_5x5x10_20260727_082037/` | 5轴×5档×10 episodes + 真实渲染变化验证 |
| Phase V2 正式评估 | `reports/phase_v2_formal_5x5x50_20260727_082212/` | 5轴×5档×50 episodes、CSV/JSON、曲线、临界点分析 |
| Phase V2 结构退化 | `reports/phase_v2_structural_formal_4x5x50_20260727_083353/` | 深度失效、遮挡、尺度偏差、组合退化 |
| 深度尺度确认 | `reports/phase_v2_depth_scale_confirm_5x200_20260727_100845/` | 每档200 episodes；确认0.25×同时跌破50%/20% |
| 真实GS输入消融 | `reports/ablation_seed2_500ep/ablation_20260726_191216.json` | baseline / const-depth / no-target / both |
| 原始 3×500 基线 | `reports/b0_fixed_3x500_20260726_122223/` | 早期三种子日志与最佳模型 |
| 3DGS 场景 | data/gs_data/ply_exports/ (4 × .ply, 16.7~23.9 MB) | gate_mid ×2, gate_left, gate_right；由 nerfstudio ckpt 提取 |
| 碰撞点云 | data/point_cloud/ (6 × .ply) | 碰撞检测用 |
| Replica | data/replica/ (18 场景 mesh+texture) | 待 3DGS 训练 (Phase V3c 候选) |
| 模型 | rlproject-swift-improved/saved_models/visual_ppo_best.pth | mock 冒烟训练产出 (非正式 baseline) |
| 退化评估 (mock) | rlproject-swift-improved/eval_results/degradation_20260722_* | mock 渲染器版衰减曲线，Phase V2 需用真实 GS 重跑 |

---

*最后更新：2026-07-25*
