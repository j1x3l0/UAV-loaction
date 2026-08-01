# UAV 视觉导航当前阶段任务完成报告

> 报告日期：2026-07-29  
> 项目：UAV-loaction  
> 当前分支：`codex/phase-v2-v3-progress`  
> 当前阶段：真实 3DGS 退化评估、鲁棒训练与 checkpoint 选择验证

---

## 1. 阶段结论

当前阶段的主要实验链路已经完成：

1. 修复视觉 PPO 熵系数问题并建立可靠 clean 基线。
2. 在真实 gsplat 渲染环境完成五轴退化评估。
3. 确认深度尺度偏差是当前最致命退化轴。
4. 完成均匀随机化、固定重加权和阶段式 curriculum 三类鲁棒训练。
5. 发现并修复“只按 clean SR 保存模型”的 checkpoint 选择混杂。
6. 完成 clean-best、final、robust-best 三方独立对照。

多尺度 `robust-best` checkpoint 已证明有效：五个尺度均优于
`clean-best`，提升 3.67–10.83 个百分点；五档汇总成功率均超过 75%，
每档 timeout 不超过 1.5%。

预设四项正式标准中通过三项。唯一未通过项是 clean 成功率：
`77.83% < 80%`，差 2.17 个百分点。因此当前阶段结论为：

**核心鲁棒性问题已经解决，checkpoint 选择机制验证有效；剩余问题收敛为
小幅 clean 性能缺口。**

---

## 2. 已完成任务

### 2.1 基础训练与评估修复

| 任务 | 状态 | 结果 |
|------|:----:|------|
| 熵系数符号修复 | ✅ | `loss = log_alpha * (entropy - target)` 生效 |
| 初始熵权重修复 | ✅ | 避免 `log_std=1 / entropy=4.26` 锁死 |
| 可靠 clean 评估 | ✅ | 使用 3 seeds、每 seed 200 episodes |
| 真实 gsplat 评估入口 | ✅ | 明确接收 renderer 与 PLY 路径 |
| 真实 Gaussian 稀疏化 | ✅ | 不再仅修改 mock 障碍物 |
| 退化输入有效性验证 | ✅ | 分辨率、噪声、视角、稀疏化均验证；光照为负对照 |

修复版基线 clean 成功率为 `82.7%`（496/600，Wilson 95% CI
79.4–85.5%），并消除了高熵版本的跨 seed 不稳定。

### 2.2 Phase V2 真实退化评估

已完成真实 gsplat 的 5 轴 × 5 档 × 50 episodes 正式评估，并扩展结构退化。

主要发现：

- Gaussian 稀疏化、深度噪声和光照对当前 depth-only 策略影响有限。
- 极端分辨率与视角覆盖仅产生小幅下降。
- 深度尺度从 1.0× 降至 0.25× 时，成功率由 83.5% 降至 13.5%。
- 深度尺度偏差是当前实验范围内最致命退化轴。
- 相变区域位于 0.5×–0.25×。

### 2.3 Phase V3 鲁棒训练

| 方案 | 正式结果 | 主要结论 |
|------|----------|----------|
| 均匀五尺度随机化 | 部分通过 | 极端尺度显著改善，但 seed0 在 0.5× 出现空洞 |
| 固定重加权 | 未通过 | 修复 0.5×，但性能转移并损失于 0.25×/0.1× |
| 阶段式 curriculum 小试 | 门控通过 | seed2×200 在五档均达到 78%–87% |
| 阶段式 curriculum 3×500 | 初次未通过 | 暴露 clean-only checkpoint 选择混杂 |
| checkpoint 修复后重跑 | 三项通过 | robust-best 明显优于 clean-best 与 final |

### 2.4 Checkpoint 机制修复

训练现在同时保存：

- `seedN_best.pth`：clean-best；
- `seedN_robust_best.pth`：优先最大化五尺度最低 SR，再最大化平均 SR；
- `seedN_final.pth`：训练结束模型。

已完成：

- curriculum 阶段边界测试；
- checkpoint 路径测试；
- robust score 排序测试；
- 16/16 环境回归测试；
- 真实 gsplat 短跑；
- 三类 checkpoint 生成与加载验证；
- 0% SR 时 clean-best 不保存的边界修复。

---

## 3. 最终三方对照

训练运行：

`v3_curriculum_ckptfix_3x500_20260728_133555`

评估设置：

- 场景：`gate_mid`
- 渲染器：真实 gsplat
- 基础 episode seed：`20260728`
- robust-best：3 seeds × 5 档 × 200 episodes，共 3,000 episodes
- clean-best / final：各 3 seeds × 5 档 × 100 episodes

| 深度尺度 | Clean-best SR | Final SR | Robust-best SR | Robust 95% CI |
|---------:|--------------:|---------:|---------------:|--------------:|
| 1.0× | 73.67% | 70.00% | **77.83%** | 74.34–80.97% |
| 0.75× | 72.00% | 72.67% | **78.83%** | 75.39–81.91% |
| 0.5× | 69.00% | 71.00% | **77.00%** | 73.47–80.19% |
| 0.25× | 65.00% | 75.33% | **75.83%** | 72.25–79.09% |
| 0.1× | 72.33% | **77.00%** | 76.00% | 72.42–79.25% |

robust-best 的 0.5× 单 seed 成功率：

- seed0：77.5%
- seed1：67.0%
- seed2：86.5%

robust-best 汇总 timeout：

| 尺度 | Timeout |
|-----:|--------:|
| 1.0× | 0.17% |
| 0.75× | 0.33% |
| 0.5× | 0.17% |
| 0.25× | 0.50% |
| 0.1× | 1.50% |

### 正式验收

| 验收项 | 标准 | 结果 | 状态 |
|--------|------|------|:----:|
| Clean 汇总 SR | ≥80% | 77.83% | ❌ |
| 每个尺度汇总 SR | ≥70% | 最低 75.83% | ✅ |
| 每 seed 的 0.5× SR | ≥60% | 最低 67.0% | ✅ |
| 每个尺度汇总 timeout | ≤10% | 最高 1.50% | ✅ |

**总体验收：部分通过（3/4）。**

---

## 4. 技术判断

### 已确认

1. 深度尺度偏差是策略的主要结构性脆弱点。
2. 单纯增加某一尺度采样概率会发生性能转移，不能稳定覆盖全部尺度。
3. 阶段式 curriculum 比固定重加权更均衡。
4. 只按 clean SR 选择 checkpoint 会系统性低估 curriculum 的最终鲁棒性。
5. 多尺度 worst-case 优先选择能显著改善跨尺度表现。
6. 当前 robust-best 的主要问题不再是退化鲁棒性，而是 clean SR 略低。

### 当前风险

1. robust checkpoint 选择每档仅使用 20 episodes，仍存在统计噪声。
2. seed1 的整体表现弱于 seed0/2，跨 seed 方差尚未完全消除。
3. 当前只在 gate_mid 场景完成正式验证，跨场景泛化尚未开始。
4. clean 目标差 2.17 pp，不能宣称四项正式标准全部通过。

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| 总进度 | `PROGRESS.md` |
| 三方 checkpoint 对照报告 | `reports/v3_checkpoint_selection_comparison_20260729/README.md` |
| 三方汇总 CSV | `reports/v3_checkpoint_selection_comparison_20260729/aggregate_comparison.csv` |
| 三方汇总 JSON | `reports/v3_checkpoint_selection_comparison_20260729/summary.json` |
| robust-best 原始评估 | `reports/v3_ckptfix_robust_eval_3x5x200_20260728_160950/` |
| final 原始评估 | `reports/v3_ckptfix_final_eval_3x5x100_20260728_160950/` |
| clean-best 原始评估 | `reports/v3_ckptfix_clean_eval_3x5x100_20260729_022411/` |
| checkpoint 修复代码 | `rlproject-swift-improved/scripts/train_visual.py` |
| checkpoint helper 测试 | `rlproject-swift-improved/scripts/test_train_visual_helpers.py` |

训练模型和日志已下载到本地 reports 目录，但按仓库策略未提交模型权重、
训练日志或 PID 临时文件。

---

## 6. GitHub 状态

- Pull Request：[#6](https://github.com/j1x3l0/UAV-loaction/pull/6)
- PR 状态：OPEN
- 合并状态：BLOCKED / REVIEW_REQUIRED
- 远端当前包含 checkpoint 修复提交：`f36ca52`
- 最终三方对照提交：`78e6222`
- 当前状态：`78e6222` 已在本地提交，但因 GitHub 网络超时尚未推送，
  本地分支领先远端 1 个提交。

---

## 7. 下一阶段建议

短程 clean-recovery 已完成单 seed 试验：100 updates、`3e-5` 学习率、
`60/15/10/10/5%` 尺度分布。唯一一次五尺度×100门控得到
`75/74/78/78/73%`，clean 未达到80%。该路线已按停止规则终止，不扩展到
3 seeds，也不再重跑。

下一阶段不再继续单场景 clean 调参，直接进入 Phase V3c 跨场景泛化，并并行
准备 Phase V3b 消融和论文图表。当前 robust-best 作为 V3 主模型，clean
77.83%的缺口作为限制项如实报告。

---

## 8. 阶段完成度

| 阶段 | 完成度 |
|------|-------:|
| Phase 0 代码与真实 GS 环境 | 100% |
| Phase V1 基线训练与 clean 评估 | 100% |
| Phase V2 退化曲线与致命轴定位 | 100% |
| Phase V3 鲁棒训练与 checkpoint 选择 | 约 90% |
| Phase V3b 消融 | 0% |
| Phase V3c 跨场景泛化 | 0% |
| Phase V4 论文整理 | 约 10% |

当前研究主线已完成从“定位退化轴”到“获得稳定多尺度策略”的闭环。完成
clean 2.17 pp 缺口修复后，即可将 V3 鲁棒训练阶段标记为正式完成。
