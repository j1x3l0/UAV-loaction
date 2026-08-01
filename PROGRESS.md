# v2 进度日志

> 更新：随时 | 基于：MASTER_PLAN.md | 当前日期：2026-08-01

> **执行顺序变更（2026-08-01）：** 已确认此前仿真环境未完成最终对齐，
> 因此旧 V3 训练、退化曲线和 checkpoint 对照全部降级为
> `legacy-unaligned` 历史诊断，不再计入正式 V3 完成度或论文主结果。
> 当前唯一主线是先完成 PX4 配置、SIH/SITL 接口门控和最终环境对齐验收；
> 验收通过后从新基线重新执行 V3、V3b 和 V3c。

---

## 状态总览

| Phase | 步骤 | 状态 | 完成日期 |
|-------|------|------|---------|
| **Phase 0** | 代码构建 | ✅ 已完成 | 2026-07-25 |
| **Phase V1** | Baseline 训练 | ✅ 已完成（3 seeds） | 2026-07-27 |
| **Phase V2** | 衰减曲线 | ✅ 扩展结构退化后找到明确相变 | 2026-07-27 |
| **Phase V3** | 鲁棒训练+对比 | ⏸️ 旧结果无效，PX4配置及最终对齐后重做 | — |
| **Phase V3b** | 消融实验 | ⏸️ 等待新V3基线 | — |
| **Phase V3c** | 跨场景泛化 | ⏸️ 等待新V3基线及场景资产 | — |
| **Phase V3d** | 场景几何闭环 | ⏸️ 旧轻量仿真仅保留诊断价值，等待PX4环境验收 | — |
| **高保真验证** | PX4 SIH/SITL→Gazebo→Isaac/Pegasus | 🔵 当前最高优先级：配置和接口门控 | — |
| **Phase V4** | 论文 | ⬜ 未开始 | — |

> 状态标记：⬜ 未开始 | 🔵 进行中 | ✅ 已完成 | ❌ 已放弃 | ⏸️ 暂停 | ⚠️ 受阻

### 历史结果有效性标记（2026-08-01修订）

| 阶段 | 标记 | 可保留内容 | 不允许直接使用的结论 |
|------|------|------------|----------------------|
| P0 / Phase 0 | ✅ 工程有效，部分组件经 V3d 修订 | 真实 gsplat 渲染、退化实现、熵与 checkpoint 修复、统计评估管线 | 旧版“渲染跑通”等同于真实场景导航闭环 |
| V1 | 🟡 Legacy 内部有效 | 旧任务下的多种子收敛、82.7%独立SR、熵修复与复现性证据 | 82.7%代表在真实 gate_mid 几何中导航 |
| V2 | 🟡 Legacy 策略敏感性有效 | 配对seed、Wilson CI、真实渲染退化、深度尺度输入敏感性 | 0.25×是普适安全临界点，或平坦曲线证明相应退化不重要 |
| 旧 V3 / V3d | 🟡 Legacy 诊断有效 | 训练管线、失败模式、几何错位诊断、checkpoint机制和实验设计经验 | 旧SR、鲁棒性提升或视觉门控作为正式验收与论文主结果 |
| PX4配置后新V3 | ⬜ 待重做 | 通过PX4接口、动力学、坐标系、碰撞与观测对齐门控后的新实验 | 在环境验收完成前启动正式多种子V3或沿用旧checkpoint |

旧 V1/V2/V3 数据不删除，统一作为 `legacy-unaligned` 历史对照。最终正文中的
真实性、退化临界点、视觉贡献和鲁棒训练结论必须在 PX4 配置及最终环境对齐
验收后重新验证；旧结果只能用于代码演进、内部有效性和“错位环境会产生
误导性高SR”的方法学分析。

### 当前强制执行顺序

1. 完成 PX4 依赖、SIH/SITL 构建和 MAVLink Offboard 接口测试；
2. 验证 ENU/NED、单位尺度、相机位姿、控制频率和 failsafe；
3. 在 PX4 环境中完成视觉、碰撞几何、边界和起终点协议对齐；
4. 运行小规模 baseline/const-depth 门控，确认任务可学且视觉确有贡献；
5. 固化新场景版本、episode manifest、评估种子和验收标准；
6. 从头重训 V3，不加载旧 `legacy-unaligned` checkpoint；
7. 新 V3 通过后再执行 V3b 消融、V3c 泛化和论文正式统计。

---

## 项目缺口与收尾路线图（2026-07-30）

### 1. 目前的不足与下一步计划

当前系统只能称为“使用真实3DGS观测、且视觉与碰撞几何初步对齐的研究
仿真”，不能称为已经足够真实，也不能据此声称具备真机可迁移性。

| 不足 | 当前证据 | 对结论的影响 | 下一步及验收 |
|------|----------|--------------|--------------|
| 策略是否真正使用深度尚未成立 | 旧对齐门控 baseline 21%、const-depth 24% | 不能把成功归因于3DGS视觉 | 完成同源几何小试，再做 baseline/const-depth 各200 episodes，同seed并按 clear/avoidance 分层；baseline必须可学习且稳定优于const-depth，否则停止扩展 |
| 碰撞几何仍是Gaussian中心近似 | 同源诊断中位误差0.37m、相关系数0.949，但P90仍为1.45m | 薄结构、半透明区域和Gaussian尺度可能导致碰撞偏差 | 导出mesh或构建SDF/occupancy，与Gaussian KD-tree交叉验证；报告碰撞一致率、假阳性和假阴性 |
| 动力学过于理想化 | 当前为三维质点、直接thrust控制 | 无法覆盖姿态、旋翼、电机、飞控和高速耦合 | 在研究门控通过后接入Isaac Sim/Pegasus或等价六自由度模型，加入PX4/ROS 2 SIL；先做10–20 episodes接口原型 |
| 传感器和状态过于理想化 | 精确速度、目标方向、相机位姿，无延迟或漂移 | sim-to-real差距被低估 | 加入IMU/里程计噪声、目标方向误差、相机/控制延迟、丢帧、运动模糊、曝光和深度缺失，并逐项消融 |
| 单场景、单小试种子 | 最终对齐环境仅 gate_mid，当前小试 seed2 | 不能证明重复性或泛化 | 先过单seed门控，再做3×500；随后准备至少3个完整场景，每个都有3DGS、mesh/SDF、边界和固定起终点集 |
| 任务和安全指标不充分 | 目前主要报告SR、collision、timeout | 不足以描述高速、安全和效率 | 增加路径长度比、最小障碍间距、完成时间、控制平滑度、碰撞速度和推理时延 |
| 与相关工作的对照不足 | 尚无GRaD-Nav/DDRL/BC或传统规划器同协议结果 | 论文贡献难以定位，与Swift等工作差距无法量化 | 固定同一场景、起终点、传感器和预算，对比PPO、无视觉策略、传统规划器以及至少一个学习基线 |
| 统计证据仍不完整 | 部分正式实验仅单模型或每档50 episodes | 容易受训练seed和episode抽样影响 | 正式结论至少3训练seeds；clean和关键档每seed≥200 episodes，Wilson CI并补paired bootstrap/McNemar检验 |
| 3DGS真实性边界未系统报告 | 使用真实重建，但没有与原始图像/深度真值比较 | “真实渲染”不能等同于“真实传感器” | 报告重建来源、Gaussian数量、位姿范围、深度覆盖、空洞率，并用实测或几何真值校准深度 |

执行优先级：

1. **P0：视觉必要性门控。** 同源几何100-update训练及
   baseline/const-depth各200 episodes分层评估已完成。正常深度总SR为31%，
   const-depth为9.5%，视觉必要性成立；但avoidance仅6.8%，因此不扩展
   3×500，转入测地奖励和避障课程修复。
2. **P1：建立最终对齐baseline。** 仅在P0通过后运行3 seeds×500，并用相同
   base seed做每seed 200 episodes独立clean评估；同时保存
   clean-best、robust-best和final。
3. **P2：重做关键退化与消融。** 先重做深度尺度和视觉输入消融；结果稳定后
   再重做其他退化轴，避免一次性复制全部legacy实验。
4. **P3：跨场景。** 完成gate-left/right及至少一个Replica场景的3DGS和
   mesh/SDF对齐，再做留一场景测试。
5. **P4：高保真复验。** Isaac Sim/Pegasus负责六自由度动力学、PX4/ROS 2、
   传感器和SIL/HIL；当前3DGS渲染器可继续提供观测。DGX Spark只是候选算力
   平台，不是仿真真实性本身。
6. **P5：通信感知扩展。** Sionna RT只在视觉导航主线成立后接入，用于
   RSS/SINR/outage，不替代物理、碰撞或视觉验证。

### 2. 之前实验需要补做的部分

| 历史实验 | 已有价值 | 必须补做 | 完成条件 |
|----------|----------|----------|----------|
| V1 legacy baseline | 证明训练管线、熵修复和多seed复现有效 | 在同源Gaussian/mesh碰撞几何上重训3×500；按clear/avoidance独立评估 | clean汇总及各seed结果、Wilson CI、collision、timeout齐全 |
| 高熵与修复版对照 | 证明高熵并未提升汇总SR，修复降低跨seed波动 | 对齐环境只需做小规模复核，不必完整复制旧3×500，除非熵再次异常 | entropy不锁死、alpha方向正确、策略性能不劣化 |
| V2五轴退化 | 评估脚本、真实退化实现和配对seed方法有效 | 用最终对齐baseline重新验证每一轴确实改变真实渲染；深度only的光照仍作为负对照 | 先每档10 episodes试跑，再5轴×5档×50；关键相变档补到200 |
| 深度尺度相变 | legacy任务中确认0.5×–0.25×崩溃区间 | 在对齐baseline上重做1.0/0.75/0.5/0.25/0.1× | 三seed曲线、CI和50%/20%临界点重新计算 |
| V3随机/加权/curriculum | 证明采样分布和checkpoint选择会改变鲁棒性 | 不直接沿用旧SR；先选最有希望的curriculum在新baseline上单seed门控，再决定3seed | clean、各尺度、单seed0.5×和timeout门槛预注册 |
| robust-best checkpoint选择 | legacy环境中优于clean-best/final | 在对齐训练中继续同时保存三类checkpoint并做同seed对照 | robust-best收益在三seed和关键尺度上重复出现 |
| clean-recovery | 已按停止规则失败 | 不再原样重跑 | 仅当新环境出现同类clean缺口且有新机制假设时才恢复 |
| V3b无速度小试 | 已完成100 updates，最佳clean仅20%，说明速度输入重要 | 下载并归档结果；不扩展三seed。以后若研究无速度，应改为历史堆叠/状态估计后重新设计 | 新设计先过单seed门控 |
| V3b无深度/浅CNN | 尚无从头重训证据 | const-depth从头重训、浅CNN容量对照，各3seeds；测试时清零仅保留为依赖性诊断 | 同训练预算、同checkpoint协议、同评估seeds |
| GRaD-Nav/DDRL/BC | 尚未实现公平对照 | 至少补一个学习基线和一个传统/无视觉基线；无法复现的工作明确写作定性比较 | 同观测、同动力学、同场景与计算预算 |
| V3c跨场景 | 仅完成资产审计 | 导出/训练标准3DGS，生成mesh/SDF碰撞几何，建立坐标校准和固定episode manifest | 至少3场景可重复运行，训练场景与测试场景严格隔离 |

### 3. 还需要做的部分

- **场景协议：** 为每个场景保存3DGS PLY、mesh/SDF、坐标系、尺度、边界、
  相机内参和固定episode seed清单；加入自动对齐检查。
- **任务协议：** 分开报告clear与avoidance；增加不同距离、窄门、遮挡、极端
  初始朝向和动态扰动。
- **算法基线：** 目标方向+速度的无视觉MLP、深度PPO、传统局部规划器、至少
  一个相关学习方法；统一参数量、训练步数和评估集。
- **鲁棒性：** 深度尺度、噪声、分辨率、视角、Gaussian稀疏化、延迟、位姿
  漂移、运动模糊、丢帧和组合退化；所有退化必须先验证真实渲染差异。
- **高保真仿真：** Isaac Sim/Pegasus六自由度多旋翼、风、碰撞接触、
  PX4/ROS 2 SIL；随后评估控制频率和端到端延迟。
- **跨仿真器验证：** 当前轻量环境训练，同一checkpoint在高保真仿真中测试，
  报告sim-to-sim性能下降；条件允许再进行HIL或低速真机安全测试。
- **工程复现：** 固化conda/container、依赖版本、服务器命令、随机种子、
  checkpoint哈希、数据下载/转换脚本和一键评估。
- **论文工作：** 重新定义主张边界，补Related Work、方法图、实验协议、
  局限性、失败案例、统计检验和计算成本；Swift只做能力差距对照，不能宣称
  已达到其高速自主飞行真实性。

### 4. 需要持续补充到本文件的内容

每个新实验完成后必须追加以下信息，缺一项则状态只能标为“部分完成”：

1. run名称、Git commit、数据/场景版本、训练seed、base evaluation seed。
2. 模型类型和checkpoint选择规则，训练updates、环境数、rollout steps、
   总交互步数、GPU、开始/结束时间和实际耗时。
3. renderer、3DGS PLY、碰撞mesh/SDF/点云、相机跟踪和退化配置。
4. aggregate及per-seed的SR、Wilson 95% CI、collision、timeout、路径效率、
   最小间距和推理时延。
5. clear/avoidance分层结果以及baseline/const-depth的配对差异。
6. CSV、JSON、曲线、日志和模型的本地路径；明确哪些文件不提交GitHub。
7. 预注册验收门槛、是否通过、停止原因及是否允许扩展下一阶段。
8. 与legacy结果比较时注明任务定义差异，禁止直接混合汇总。
9. 失败实验也必须记录根因、已尝试次数和终止决定，避免重复消耗GPU。
10. 高保真或真机相关结果需单独注明SIL/HIL/实机层级，不得用渲染质量替代
    物理真实性结论。

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

## Phase V1：Baseline 训练（Legacy-Unaligned）

> 有效性：旧任务定义内有效；不得作为最终真实场景导航 baseline。V3d 对齐
> 环境训练完成后重新建立新 V1。

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

## Phase V2：衰减曲线（Legacy-Unaligned）

> 有效性：真实渲染输入敏感性与评估方法有效；场景安全边界和通用退化结论
> 待 V3d 对齐环境复现。`0.25×` 仅称为旧策略的离散经验崩溃档。

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

2026-07-29 清点与实现：

- 已有真实 GS 测试时输入诊断（旧 seed2，200 episodes）：baseline 82%，
  const-depth 71%，no-target-direction 0%，两者同时移除 0%。
- `VisualDroneEnv` 已增加 `no_velocity`，评估脚本扩展为 baseline、
  单输入移除与组合移除，并输出 Wilson 95% CI、CSV 和 JSON。
- 这类“在已训练模型上清零输入”只能回答策略依赖性，不能替代从头重训的
  正式 V3b 消融。
- 当前 Actor/Critic 共享同一 CNN、向量输入和共享层，没有位置或障碍距离等
  特权 Critic 输入。因此“无特权 Critic”在现代码中没有可移除对象，原计划
  该项无效；如需研究非对称 Critic，应先实现一个有特权信息的对照架构。
- 本机缺少 gymnasium/pytest；服务器常用 Python 路径也未恢复到此前 CUDA
  gsplat 环境。遵循连续失败两次即停止规则，本轮没有伪造或重复启动消融评估。

随后已定位实际训练环境 `/root/miniconda3/envs/myconda`，并完成 robust-best
seed2 的真实 gsplat 7×100 输入门控（base seed 20260728）：

| 配置 | SR | Wilson 95% CI | Collision | Timeout |
|------|---:|--------------:|----------:|--------:|
| baseline | 81% | 72.2–87.5% | 19% | 0% |
| const-depth | 77% | 67.8–84.2% | 23% | 0% |
| no-velocity | 11% | 6.3–18.6% | 53% | 36% |
| no-target-direction | 0% | 0–3.7% | 7% | 93% |
| no-depth + no-velocity | 12% | 7.0–19.8% | 60% | 28% |
| no-velocity + no-target-direction | 0% | 0–3.7% | 7% | 93% |
| all-inputs-ablated | 0% | 0–3.7% | 0% | 100% |

baseline 与旧200-episode结果82%一致，门控可信。速度输入移除导致 -70pp，
因此执行了无速度 curriculum seed2×100 重训小试
`v3b_no_velocity_curriculum_seed2_100_pilot_20260729`。首次配置因内部评估
规模过大已主动终止；修正版使用20 clean与每档5次内部评估，100 updates、
204,800步，用时15分52秒。最佳clean仅20%，final clean 15%；末次五尺度
内部SR为20%/0%/0%/0%/40%，min=0%、mean=12%。三个checkpoint均已生成。
该小试已明确失败，不再运行每档100次独立门控，也不扩展到3 seeds；如以后
研究无速度输入，必须先加入历史帧或状态估计器形成新的可检验假设。

| 消融 | 状态 | 训练开始 | 训练结束 | 模型路径 | 结论 |
|------|------|---------|---------|---------|------|
| 无深度图 (RGB) | ⬜ 正式重训待执行 | — | — | — | const-depth 测试时诊断为 -11pp |
| 无速度向量 | ❌ 单seed小试失败，终止扩展 | 2026-07-29 | 2026-07-29 | `saved_models/v3b_no_velocity_curriculum_seed2_100_pilot_20260729/` | 最佳clean 20%，五尺度min 0% |
| 浅CNN (1层) | ⬜ | — | — | — | — |
| 无特权Critic | ➖ 当前架构不适用 | — | — | — | 当前 Critic 无特权输入 |

| 产出 | 状态 | 路径 |
|------|------|------|
| 输入依赖门控 | ✅ | `reports/v3b_input_gate_seed2_7x100_20260729/` |
| 附表2：消融对比矩阵 | 🔵 | 待正式重训 |

---

## Phase V3d：场景几何闭环修复

2026-07-29 完成第一优先级修复：

- 审计确认原环境目标高度可达8m，而 gate_mid 真实重建有效 z 范围约
  -0.2–3.8m；渲染场景与三个固定球形碰撞体不一致。
- 新增 `ScenePointCloudGeometry`：使用与3DGS同坐标系的稠密点云和 KD-tree
  计算碰撞/安全距离，不再用硬编码球体代表真实场景。
- 根据点云1–99分位与安全边距自动生成场景边界。
- 构建最大连通自由空间网格，起终点只从同一连通分量采样。
- 50% episode 的起终点直线路径被场景几何阻挡，用于显式测试绕障能力；
  其余为直线可达对照。
- 相机光轴随速度方向变化，低速时朝向目标；不再固定使用单位姿态。
- gate_mid_new 配套稠密点云已同步到服务器：
  `/root/data/point_cloud/gate_mid_new.ply`。
- 回归测试19/19通过；20次真实GS reset中9次为绕障任务，最大连通自由空间
  含5,494个网格点。修复后的深度范围约0.30–20m，不再出现原先全图
  0.11–0.29m的异常近深度。
- 1轮端到端训练 smoke 已通过；对齐环境小试
  `v3_aligned_geometry_seed2_100_20260729` 已完成100 updates、204,800步，
  用时27分28秒，最佳内部 clean SR 20%。
- 独立视觉必要性门控
  `v3_aligned_visual_gate_seed2_7x100_20260729` 已完成：

| 配置 | SR | Wilson 95% CI | Collision | Timeout |
|------|---:|--------------:|----------:|--------:|
| baseline | 21% | 14.2–30.0% | 79% | 0% |
| const-depth | 24% | 16.7–33.2% | 76% | 0% |
| no-velocity | 5% | 2.2–11.2% | 95% | 0% |
| no-target-direction | 1% | 0.2–5.4% | 99% | 0% |
| no-depth + no-velocity | 5% | 2.2–11.2% | 95% | 0% |
| no-velocity + no-target-direction | 0% | 0–3.7% | 100% | 0% |
| all-inputs-ablated | 0% | 0–3.7% | 100% | 0% |

判定：门控失败，**不扩展3×500**。正常depth未优于const-depth（-3pp，CI
高度重叠），且baseline只有21%。失败全部为碰撞。下一步先按直线可达/需绕障
分层统计，并诊断点云碰撞半径、运动相机、任务难度和奖励。

2026-07-30 已完成深度—碰撞几何对齐诊断。使用原稠密碰撞点云时，30/30
位姿可比较，但渲染深度与点云前向深度的中位绝对误差为1.29m、P90为
3.15m、相关系数为0.676，未通过0.75m门槛。分层后，需绕障样本的中位误差
仅0.38m（相关系数0.869），直线可达样本却为2.24m（相关系数0.497），说明
碰撞点云把部分3DGS可见表面错误地标成了自由空间。

改用同一份 `gate_mid_new_gs.ply` 的 Gaussian 中心构建碰撞几何后，30/30
位姿的中位误差降至0.37m、相关系数升至0.949，对齐诊断通过。单种子小试
`v3_gscloud_aligned_seed2_100_20260730` 已完成100 updates、204,800步，
耗时25分15秒，训练内部最佳clean为40%。随后完成
baseline/const-depth各200 episodes分层门控：

| 配置 | 总SR | Wilson 95% CI | Clear SR | Avoidance SR | Collision | Timeout |
|------|-----:|--------------:|---------:|-------------:|----------:|--------:|
| baseline | 31.0% | 25.0–37.7% | 56.7% | 6.8% | 69.0% | 0.0% |
| const-depth | 9.5% | 6.2–14.4% | 15.5% | 3.9% | 86.5% | 4.0% |

视觉必要性通过：baseline高21.5pp且CI不重叠；但导航门控失败，绕障任务
93.2%碰撞，不能扩展3×500。奖励审计确认旧奖励每步惩罚绝对欧氏距离并奖励
速度直接指向目标，与必要绕行冲突。

已实现可选的自由空间测地进度奖励，绕障时关闭直接目标heading奖励；训练任务
按10%→30%→50%避障比例分三阶段，内部评估始终保持50%避障。服务器20/20
回归和真实gsplat单update冒烟通过。
`v3_geodesic_avoidance_curriculum_seed2_200_20260730` 已完成200 updates、
409,600步，内部最佳SR仅10%。正式2×200门控中baseline总SR 7.0%、clear
13.4%、avoidance 1.0%，const-depth总SR 2.0%；该奖励路线失败并停止。

为区分“任务不可解”和“PPO未学会”，新增最短路径＋路径点PD oracle。在完全
相同的真实gsplat、Gaussian碰撞、base seed `20260728` 和200 episodes上：

| 控制器 | 总SR | Wilson 95% CI | Clear SR | Avoidance SR | Collision | Timeout |
|--------|-----:|--------------:|---------:|-------------:|----------:|--------:|
| 路径点oracle | 98.5% | 95.7–99.5% | 97.9% | 99.0% | 1.5% | 0.0% |

因此场景、起终点采样和低层动力学可解，当前瓶颈确定为PPO奖励/观测学习。已将
测地势能改为8邻点连续近似，并加入仅用于训练塑形的局部安全路径heading；
路径点不进入策略观测。服务器20/20回归和真实gsplat冒烟通过。
`v3_hybrid_geodesic_seed2_100_20260730` 已完成100 updates、204,800步，
用时25分25秒，内部最佳SR为50%。随后完成baseline/const-depth各200
episodes分层门控：

| 配置 | 总SR | Wilson 95% CI | Clear SR | Avoidance SR | Collision | Timeout |
|------|-----:|--------------:|---------:|-------------:|----------:|--------:|
| baseline | 48.0% | 41.2–54.9% | 86.6% | 11.7% | 52.0% | 0.0% |
| const-depth | 11.0% | 7.4–16.1% | 19.6% | 2.9% | 71.5% | 17.5% |

视觉必要性再次通过（-37pp），clear任务已可学习，但avoidance仍有88.3%碰撞，
因此门控失败，**不扩展3×500并停止继续调节测地奖励权重**。

下一步改为诊断“最终目标方向观测”与“局部安全路径奖励”不一致的问题。已实现
可选 `--waypoint-observation`，用局部安全路径方向替代最终目标方向；该实验
只用于判断是否需要分层规划器，不作为端到端视觉导航结果。服务器21/21回归
通过。真实gsplat单update冒烟连续两次未完整收尾：第一次因启动PATH未包含
Conda内的Ninja，第二次训练和评估已执行但checkpoint父目录未预创建。已在
`train_visual()` 中补齐checkpoint目录创建；按失败两次即终止的约定，本轮
未做第三次重试。用户于后续回合明确要求继续后，修复版冒烟完整生成
best/final checkpoint。`v3_waypoint_obs_seed2_100_20260730` 已完成100
updates、204,800步，用时23分57秒，内部最佳SR为90%；一次性2×200门控也
已完成：

| 配置 | 总SR | Wilson 95% CI | Clear SR | Avoidance SR | Collision | Timeout |
|------|-----:|--------------:|---------:|-------------:|----------:|--------:|
| baseline | 92.5% | 88.0–95.4% | 96.9% | 88.3% | 7.5% | 0.0% |
| const-depth | 84.0% | 78.3–88.4% | 96.9% | 71.8% | 16.0% | 0.0% |

局部路径点使avoidance从上一轮11.7%升至88.3%，证明分层规划+局部控制可行，
但baseline仅比const-depth高8.5pp，未达到预设20pp视觉贡献门槛。总体门控
判为失败，**不扩展多种子**。该结果只作为特权路径点/规划器上界，不作为
端到端3DGS视觉导航成果。测地奖励与路径点观测调参路线至此终止，下一步转入
PX4+Gazebo接口验证准备。

PX4接口准备已开始。服务器审计确认当前Ubuntu 20.04 overlay环境未安装Docker、
Gazebo、PX4、ROS 2或colcon，因此暂不污染系统依赖。已新增传输无关的
`integrations/px4_offboard.py`，完成策略ENU加速度到PX4 NED
`TrajectorySetpoint`的坐标转换、水平/垂直限幅、未控字段NaN、20Hz心跳和
0.25秒陈旧检测；本地和服务器单元测试均通过。下一步先部署零外部仿真依赖的
PX4 SIH，验证连接、解锁、Offboard、悬停和失联保护，再进入Gazebo
`x500_depth`，详细协议见 `docs/PX4_GAZEBO_INTERFACE_PLAN.md`。

该修复会改变任务定义，旧模型只能作为“未对齐环境”历史对照，不能直接与新
环境成功率混合汇总。完整结果：
`reports/v3_aligned_visual_gate_seed2_7x100_20260729/`。

---

## 可选扩展：Sionna RT 通信感知导航

2026-07-29 完成可行性评估，记录于
`docs/sionna-integration-assessment.md`。Sionna 只用于离线生成 RSS/SINR/outage
无线电地图，不替代3DGS渲染、碰撞几何或动力学，并使用独立 Python 环境。

启用门槛：场景对齐 baseline 可学习、正常深度显著优于 const-depth、坐标
一致性通过、至少一个 V3b 正式三种子消融完成。门槛通过前不安装 Sionna，
避免偏离当前核心问题。

---

## Phase V3c：跨场景泛化

2026-07-29 资产审计：

- 服务器仅发现一个标准真实 3DGS：`gate_mid_new_gs.ply`。
- 本地有 gate_mid/left/right 的 nerfstudio checkpoint 与若干普通/导出点云，
  但 Replica 18 场景目前只有 mesh/texture，尚未训练为 3DGS；garden 也未就绪。
- 当前 `VisualDroneEnv` 的边界、目标分布和三个球形碰撞体固定，不随 PLY
  改变。仅替换 PLY 最多是视觉域迁移代理，不是完整跨场景导航泛化。
- 因此 garden→office0/room0/apartment0 正式 V3c 暂停。恢复条件是每个场景
  同时具备标准 3DGS PLY、坐标变换/边界、碰撞几何和可复现起终点协议。

| 实验 | 状态 | 训练场景 | 测试场景 | 结果 |
|------|------|---------|---------|------|
| L0 gate-family 视觉域代理 | ⚠️ 待导出/校准 | gate_mid_new | gate_mid/left/right | 不得写作完整跨场景导航 |
| L1 跨场景 | ⚠️ 资产受阻 | garden | office0, room0, apartment0 | 缺标准3DGS与场景几何协议 |
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
| 2026-07-30 | 测地奖励+避障课程 seed2×200 小试 | — | — | run `v3_geodesic_avoidance_curriculum_seed2_200_20260730`；前序同源几何100-update和2×200门控已完成 |
| 2026-07-26 | 3×500 高熵基线 | 其他任务占用 | — | run `b0_entropy_sign_fixed_3x500_20260726_191516` |
| 2026-07-27 | 3×500 熵修复版 + 6×200 clean 评估 | 其他任务占用 | — | 训练与评估均完成，GPU0 已释放 |

---

## 问题日志

> 记录所有遇到的问题、原因、解决方案。按日期倒序。

| 日期 | Phase | 问题 | 严重度 | 状态 | 解决方案 |
|------|-------|------|--------|------|---------|
| 2026-07-30 | V3d | 原碰撞点云漏掉部分3DGS可见表面，clear样本中位深度误差2.24m | 🔴 高 | ✅ 根因已定位，修复验证中 | 改用同一3DGS PLY的Gaussian中心构建碰撞几何；30位姿中位误差降至0.37m、相关系数升至0.949；待策略门控确认 |
| 2026-07-29 | V3b | 测试时移除速度使SR从81%降至11%，从头无速度小试最佳clean仅20% | 🟡 中 | ❌ 当前路线终止 | 不扩展三seed；后续只有加入历史帧/状态估计器后才作为新实验恢复 |
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
| 2026-07-30 | 继续使用独立稠密点云作为碰撞几何并扩展3×500 | 先以3DGS Gaussian中心作为同源碰撞几何跑单seed视觉必要性门控 | 原点云在clear样本与渲染深度严重不一致；同源几何诊断通过，但仍需证明策略确实利用深度 |
| 2026-07-30 | 直接用当前仿真支撑真实性结论 | 将其限定为研究仿真，并把Isaac Sim/Pegasus+PX4/ROS 2列为后续独立高保真验证层 | 真实3DGS只提升观测真实性，不能替代六自由度动力学、传感器、飞控和sim-to-real验证 |
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
| 2026-07-30 | 场景/碰撞错位 | 原稠密碰撞点云与3DGS可见表面在clear样本中明显不一致 | 停止3×500扩展；改用同源Gaussian几何，先做30位姿诊断和单seed视觉门控 |
| 2026-07-29 | 单场景过拟合 | 当前只有gate_mid具备可运行真实3DGS闭环 | V3c暂停；先建立场景资产协议，不把视觉域替换写作跨场景导航 |
| 2026-07-29 | 低保真动力学 | 当前质点模型无法支撑真机或高速飞行主张 | 将Isaac Sim/Pegasus SIL列为主线门控通过后的必做复验 |

---

## 实验数据索引

> 所有产出的 CSV / JSON / 模型 / 图表 的路径汇总。

| 类别 | 路径 | 说明 |
|------|------|------|
| 路径点观测诊断 | `reports/v3_waypoint_obs_gate_seed2_2x200_20260730/` | baseline 92.5%、avoidance 88.3%，但仅比const-depth高8.5pp；只作为规划器上界 |
| hybrid-geodesic门控 | `reports/v3_hybrid_geodesic_gate_seed2_2x200_20260730/` | baseline 48%、avoidance 11.7%，视觉有效但绕障失败 |
| 路径点oracle可解性门控 | `reports/v3_waypoint_oracle_200_20260730/` | 200 episodes，总SR 98.5%、avoidance 99.0% |
| 纯测地奖励失败实验 | `reports/v3_geodesic_avoidance_curriculum_seed2_200_20260730/` | 200 updates，内部最佳SR 10%；模型日志仅本地 |
| 纯测地奖励2×200门控 | `reports/v3_geodesic_avoidance_gate_seed2_2x200_20260730/` | baseline 7%、avoidance 1%，终止 |
| 同源几何100-update训练 | `reports/v3_gscloud_aligned_seed2_100_20260730/` | 模型与日志仅本地保存，不提交GitHub |
| 同源几何2×200视觉门控 | `reports/v3_gscloud_aligned_gate_seed2_2x200_20260730/` | baseline 31%、const-depth 9.5%；视觉有效但绕障失败 |
| 原点云对齐诊断 | `reports/v3_aligned_diagnostic_20260730/` | 30位姿；中位误差1.29m，未通过 |
| 同源Gaussian对齐诊断 | `reports/v3_aligned_diagnostic_gscloud_20260730/` | 30位姿；中位误差0.37m、相关系数0.949，通过 |
| 对齐视觉门控 | `reports/v3_aligned_visual_gate_seed2_7x100_20260729/` | 原碰撞点云环境；baseline 21%、const-depth 24%，失败 |
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

## PX4 SIH 部署进度（2026-08-01）

- ✅ Ubuntu 20.04 服务器完成 PX4 `v1.17.0`（`d6f12ad`）源码与精确子模块部署。
- ✅ `px4_sitl_default/bin/px4` 构建成功，ELF 与动态库依赖检查通过。
- ✅ `sihsim_quadx` 以 250 Hz、1.0 倍实时速度启动，MAVLink 14550 心跳验证通过。
- ✅ 固化服务器启动、停止、状态和心跳检查脚本；部署说明见 `docs/px4-sih-server-deployment.md`。
- ✅ MAVLink Offboard 冒烟测试通过：确认 Armed/Offboard，完成起飞、悬停、原生落地与 Disarm；ULog 无 failsafe。
- ✅ 旧 checkpoint 接口门禁通过：严格加载 `b0_logstd_fixed.../seed0_best.pth`，在 1 m 定点保持下发送 55 个策略加速度前馈 setpoint；最大水平偏移 0.207 m，策略段高度 0.628–1.013 m，原生落地并 Disarm，ULog `failsafe=0`。
- ⚠️ 上述门禁使用固定 5 m 深度与固定目标向量，只证明 checkpoint→推理→MAVLink→PX4 链路；报告明确标记 `aligned_v3_result=false`，不得计入 V3 或论文指标。
- ✅ 新增显式 `gate_mid_new` 坐标配置：以训练帧154作为 PX4 在1 m悬停、零姿态的锚点，固定 scene↔NED 旋转/平移、FRD↔OpenCV 光学轴和64×64中心裁剪内参；本地与服务器锚点/可逆性测试通过。
- ✅ 训练相机5帧 GPU 注册通过轴向判别：完整竖屏缩放下正确转换覆盖率1.000、MAE 0.162、亮度相关0.227，错误轴相关-0.143；策略64×64中心裁剪下正确转换MAE 0.187、相关0.031，错误轴相关-0.170。
- ✅ gsplat GPU 基础性能确认：合成100 Gaussian首次调用0.55秒，CUDA预热后完整265,631 Gaussian的64×64渲染约1.8 ms；先前60秒等待为首次初始化/会话轮询误判，不是持续故障。
- ⚠️ 5帧正确转换虽在指标和结构上优于错误候选，但绝对亮度相关仍低；当前只确认坐标轴、锚点和策略内参，不构成照片级渲染质量证据。
- ⏳ 下一门槛：完成30位姿策略64×64 GPU统计与映射飞行体积净空门禁，再实现只读在线观测桥；全部通过前 `formal_v3_ready=false`，不启动正式V3。

---

## PX4–3DGS 相机注册进度（2026-08-02）

- ✅ PX4 SIH 部署与 Offboard 链路通过。
- ✅ 旧 checkpoint→推理→MAVLink→PX4 接口门禁通过（仅限接口验证，`aligned_v3_result=false`）。
- ✅ PX4 NED↔3DGS↔相机坐标变换测试通过。
- ✅ 5 帧 GPU 相机轴向注册通过。
- ✅ 策略实际 64×64 中心裁剪门禁初步通过：正确转换 MAE `0.1869`、亮度相关 `0.0306`；错误转换 MAE `0.2016`、相关 `-0.1700`。
- ✅ 265,631 Gaussian 预热后渲染约 `1.8 ms/帧`。
- ✅ PX4 当前已停止，无 SCP、注册或渲染残留进程。
- ⚠️ `formal_v3_ready=false`，尚不能开始正式 V3。
- ⚠️ 30 位姿门禁所需图片上传接近 60 秒未完成，已及时中止；服务器当前图片目录有 16 张、约 14 MB，现有完整文件保留。当前主要阻塞是文件传输稳定性，不是 PX4、CUDA 或 gsplat 性能。

### 后续执行顺序（2026-08-02）

1. 剩余图片改为每批 3–5 张上传，每批核对数量和大小。
2. 完成 30 位姿、64×64 中心裁剪 GPU 注册统计。
3. 检查映射后 PX4 飞行区域的最小碰撞净空。
4. 实现只读观测桥：PX4 位姿→3DGS 深度和策略向量，不发送控制。
5. 做遥测回放和定点悬停渲染测试。
6. 门禁全部通过后，重新训练三种子正式 V3。

---

*最后更新：2026-08-02*
