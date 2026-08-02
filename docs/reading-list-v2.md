# 论文阅读清单 v2：3DGS 视觉导航策略鲁棒性

> 整理时间：2026-07-21
> 对应计划：[research_plan_v2.md](../research_plan_v2.md) | [weekly_plan_v2.md](../weekly_plan_v2.md)

---

## 阅读分级

```
🔴 必读精读 — 方法/实验设计直接依赖，不读没法写代码
🟡 必读泛读 — Related Work 引用 + 实验方法论参考
🟢 选读     — Discussion 扩展 + 后续方向储备
```

---

## 一、🔴 必读精读（5篇）— Week 1 完成

### 1. GRaD-Nav — 最直接相关

| 项目 | 内容 |
|------|------|
| **标题** | GRaD-Nav: Efficiently Learning Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics |
| **作者** | Qianzhong Chen, Jiankai Sun, Naixiang Gao, JunEn Low, Timothy Chen, Mac Schwager |
| **单位** | Stanford Multi-Robot Systems Lab |
| **发表** | IROS 2025 |
| **arXiv** | [2503.03984](https://arxiv.org/abs/2503.03984) |
| **代码** | [github.com/Qianzhong-Chen/grad_nav](https://github.com/Qianzhong-Chen/grad_nav) |
| **为什么必读** | 唯一一个3DGS+RL+无人机的完整开源管线。我们要直接复现/改造它。 |

**精读重点**：
- §3 Method：3DGS渲染管线怎么集成的？可微分动力学的数学形式？
- §4 Experiments：训练了多少步？用什么超参数？CENet的结构？
- 开源代码：`renderer` 模块和 `ddrl` 模块的具体实现

**对我们的用途**：代码基础设施 + DDRL baseline + 论文结构参照

**已确认的技术差异（2026-07-21 代码审查）**：

| 维度 | GRaD-Nav (实际代码) | 我们的 v2 |
|------|----------|----------|
| 算法 | SHAC (可微分RL) + PPO基线(同环境) | PPO (无模型RL) |
| 梯度流 | 策略→动作→动力学→下一状态→loss，全链可微 (no_grad=False) | 策略→动作，靠reward信号 |
| 观测 | **57D向量**: RGB→冻结SqueezeNet→16D特征 + 状态 + VAE latent(24D) | 深度图 64×64 + 速度 + 目标方向 |
| 视觉编码器 | **冻结的SqueezeNet** (pretrained, 不参与RL训练) → FC(512→16) | 3层Conv 端到端训练 |
| 动作 | 体角速度(3D) + 归一化推力 → 4D | 3D推力 xyz → 3D |
| Actor网络 | MLP [512, 256, 128] (ELU + LayerNorm) | MLP [128, 128] |
| VAE | Encoder [256,256,256] → latent(24D) → Decoder [32,64,128,256] | 无 |
| 训练时间 | ~3.5h (宣称, SHAC) | 估计 20-35h (5000ep) |
| 并行环境 | 4096 (默认) / 128 (论文) | 8 |
| 动力学 | 200Hz内部仿真, 四元数 + PD控制器 + 电机延迟(0.7-0.8) + 气动阻力 + 转子噪声 | 简化质点模型 (20Hz) |
| 奖励 | 10组件全部正 (survive_reward=8.0, waypoint=4.0×4) | 7组件有正有负 |
| 碰撞检测 | 点云最近距离 (从ply文件) | 球体代理 (硬编码) |

**GRaD-Nav 的关键局限 → 我们的切入点**：
1. ⚡ 仅静态场景 — 障碍物不能动（超出我们范围）
2. ⚡ 需要预采集场景 — 不能探索未知空间（超出我们范围）
3. 🎯 **渲染质量依赖 — "GS quality impacts performance" 一句话带过，零实验** ← 我们的核心贡献
4. 🎯 **无安全保证 — 没有鲁棒性验证或对抗条件测试** ← 我们的临界点分析直接回应
5. ⚡ 手工奖励塑形 — 依赖参考轨迹和waypoint（我们的reward不依赖参考轨迹）
6. 🎯 **光照/纹理/sensor noise — sim-to-real视觉分布偏移** ← 我们的光照偏移+深度噪声退化轴

> 详细对比见 [docs/grad-nav-comparison.md](grad-nav-comparison.md)

---

### 2. SOUS VIDE — 行为克隆对照

| 项目 | 内容 |
|------|------|
| **标题** | SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum |
| **作者** | JunEn Low, Maximilian Adang, Javier Yu, Keiko Nagami, Mac Schwager |
| **单位** | Stanford MSL |
| **发表** | IEEE RA-L 2025 |
| **arXiv** | [2412.16346](https://arxiv.org/abs/2412.16346) |
| **代码** | 开源（随论文） |
| **为什么必读** | 12小时训完、105次真机飞行。BC pipeline比DDRL更简单，是我们最可靠的fallback方案。 |

**精读重点**：
- §3.1 FiGS simulator：1-2分钟视频→3DGS→130fps渲染，具体怎么做？
- §3.2 SV-Net：SqueezeNet+光流+IMU→动作，为什么这么设计？
- §4.3 Robustness tests：30%质量变化、40m/s阵风、60%亮度——他们怎么测的？

**对我们的用途**：BC baseline + 场景重建pipeline参考 + 鲁棒性测试方法论

---

### 3. 3D Gaussian Splatting 原论文

| 项目 | 内容 |
|------|------|
| **标题** | 3D Gaussian Splatting for Real-Time Radiance Field Rendering |
| **作者** | Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis |
| **单位** | Inria |
| **发表** | SIGGRAPH 2023 (Best Paper) |
| **项目页** | [repo-sam.inria.fr/fungraph/3d-gaussian-splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |
| **为什么必读** | 不理解GS的数学就无法设计退化实验。高斯球剪枝、密度控制、各向异性协方差——这些是退化轴的理论基础。 |

**精读重点**：
- §3 Differentiable 3D Gaussian Splatting：高斯球的6个参数（位置、协方差、颜色、透明度）
- §4 Optimization：稠密化/剪枝策略（这是我们"高斯球稀疏化"退化轴的技术来源）
- §5 Rendering：tile-based rasterizer 的性能特性

**对我们的用途**：退化轴设计 + 3DGS质量评估指标理解

---

### 4. Swift — 无人机RL基础设施参照

| 项目 | 内容 |
|------|------|
| **标题** | Champion-Level Drone Racing Using Deep Reinforcement Learning |
| **作者** | Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Müller, Vladlen Koltun, Davide Scaramuzza |
| **单位** | UZH Robotics and Perception Group |
| **发表** | Nature, 620(7976):982–987, 2023 |
| **为什么必读** | 你当前的PPO管线基于Swift。视觉版需要确认：Swift的PPO配置在视觉输入下是否需要调整？ |

**精读重点**（重读）：
- Method §Control policy：网络架构在视觉输入下的设计
- Extended Data：训练超参数的完整清单
- 与你的 `ppo_agent.py` 逐项对照

**对我们的用途**：确保视觉PPO的超参选择有论文依据

---

### 5. EmbodiedSplat — 场景重建→导航的训练范式

| 项目 | 内容 |
|------|------|
| **标题** | EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device |
| **作者** | Gunjan Chhablani, Xiaomeng Ye, Muhammad Zubair Irshad, Zsolt Kira |
| **单位** | Georgia Tech & Toyota Research Institute |
| **发表** | ICCV 2025 |
| **arXiv** | [2509.17430](https://arxiv.org/abs/2509.17430) |
| **代码** | [github.com/gchhablani/embodied-splat](https://github.com/gchhablani/embodied-splat) |
| **为什么必读** | iPhone拍场景→3DGS→Habitat-Sim→微调导航策略——pipeline和我们最接近。Sim-vs-real相关系数0.87-0.97。 |

**精读重点**：
- §3 Pipeline：Polycam→Nerfstudio→DN-Splatter→Habitat-Sim的完整链条
- §4.2 Sim-vs-real correlation：他们怎么量化仿真和真机的性能对应关系？
- 附录：场景采集协议（光照、视角、时长）

**对我们的用途**：场景重建pipeline工程参考 + sim-vs-real验证方法论

---

## 二、🟡 必读泛读（6篇）— Week 2 完成

### 6. GRaD-Nav++

| 项目 | 内容 |
|------|------|
| **标题** | GRAD-NAV++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics |
| **作者** | 同一Stanford团队 |
| **发表** | IEEE RA-L 2026 (published Dec 2025) |
| **arXiv** | [2506.14009](https://arxiv.org/abs/2506.14009) |
| **为什么读** | GRaD-Nav的升级版。读它对GRaD-Nav做了哪些改进，不读方法细节。 |

**泛读重点**：Abstract + Figures + Discussion中的limitations（这些limitations是我们的机会）

---

### 7. Ferede et al. — 域随机化方法论

| 项目 | 内容 |
|------|------|
| **标题** | One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms |
| **作者** | Robin Ferede 等 |
| **单位** | TU Delft MAVLab |
| **发表** | ICRA 2025 |
| **arXiv** | [2504.21586](https://arxiv.org/abs/2504.21586) |
| **为什么读** | 域随机化水平选择的实验方法论——0%/10%/20%/30%怎么选的？为什么这样设计？直接指导我们的退化水平间隔设计。 |

**泛读重点**：Experiment §4 的实验设计 + Figure 4 的曲线形状

---

### 8. Peng et al. — 动力学随机化

| 项目 | 内容 |
|------|------|
| **标题** | Sim-to-Real Transfer of Robotic Control with Dynamics Randomization |
| **作者** | Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel |
| **单位** | UC Berkeley / OpenAI |
| **发表** | ICRA 2018 |
| **arXiv** | [1710.06537](https://arxiv.org/abs/1710.06537) |
| **为什么读** | 域随机化的奠基性工作。参数范围选择的理论依据。Related Work必须引用。 |

**泛读重点**：§3-4 随机化参数选择逻辑 + 为什么随机化有效

---

### 9. Andrychowicz et al. (OpenAI) — ADR

| 项目 | 内容 |
|------|------|
| **标题** | Solving Rubik's Cube with a Robot Hand |
| **作者** | OpenAI (Marcin Andrychowicz 等) |
| **发表** | arXiv 2019 |
| **arXiv** | [1910.07113](https://arxiv.org/abs/1910.07113) |
| **为什么读** | ADR (Automatic Domain Randomization) — 自动难度调节。思考：ADR是否能替代我们手工定义的5条退化轴？ |

**泛读重点**：§4 ADR 的伪代码 (Algorithm 1) + 自动调节 vs 手工调节的对比

---

### 10. Geles et al. — 非对称 Actor-Critic

| 项目 | 内容 |
|------|------|
| **标题** | Demonstrating Agile Flight from Pixels without State Estimation |
| **作者** | Ismail Geles, Leonard Bauersfeld, Angel Romero, Jiaxu Xing, Davide Scaramuzza |
| **单位** | UZH RPG |
| **发表** | RSS 2024 |
| **arXiv** | [2406.12505](https://arxiv.org/abs/2406.12505) |
| **为什么读** | 非对称Actor-Critic的具体实现——训练时Critic有特权信息，Actor只用视觉。我们的消融4（无特权Critic）直接依赖这篇。 |

**泛读重点**：§3 Method — 非对称架构的具体设计 + 特权信息的选择

---

### 11. Hanover et al. — 领域综述

| 项目 | 内容 |
|------|------|
| **标题** | Autonomous Drone Racing: A Survey |
| **作者** | Drew Hanover 等 |
| **发表** | IEEE Transactions on Robotics, 40:3044–3067, 2024 |
| **为什么读** | Introduction 和 Related Work 写作的参考模板。确认我们不遗漏关键的对比工作。 |

**泛读重点**：§1 Introduction（学习顶刊综述的问题定义方式）+ §5 Open Challenges

---

## 三、🟢 选读（4篇）— Week 3 及以后

### 12. GaussFly

| 项目 | 内容 |
|------|------|
| **标题** | GaussFly: Contrastive Reinforcement Learning for Visuomotor Policies in 3D Gaussian Fields |
| **作者** | Yuhang Zhang 等 (NTU) |
| **发表** | 2025 |
| **arXiv** | [2604.05062](https://arxiv.org/abs/2604.05062) |
| **为什么选读** | 对比学习+3DGS+RL——和我们的PPO方案不同。Discussion中作为替代方案引用。 |

---

### 13. Liquid Networks + 3DGS

| 项目 | 内容 |
|------|------|
| **标题** | Gaussian Splatting to Real World Flight Navigation Transfer with Liquid Networks |
| **作者** | Alex Quach, Makram Chahine, Alexander Amini, Ramin Hasani, Daniela Rus |
| **单位** | MIT CSAIL |
| **发表** | CoRL 2024 |
| **arXiv** | [2406.15149](https://arxiv.org/abs/2406.15149) |
| **为什么选读** | 3DGS+Liquid NN+真机部署。架构不同但pipeline类比。Discussion中作为替代网络架构引用。 |

---

### 14. Dream to Fly

| 项目 | 内容 |
|------|------|
| **标题** | Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight |
| **作者** | Angel Romero, Ishaan Shenai, Ismail Geles, Sammy Aljalbout, Davide Scaramuzza |
| **单位** | UZH RPG |
| **发表** | arXiv 2025 |
| **arXiv** | [2501.14377](https://arxiv.org/abs/2501.14377) |
| **为什么选读** | DreamerV3在无人机上的应用。如果后续想对比"世界模型 vs 无模型在3DGS中谁更鲁棒"，这篇是基础。 |

---

### 15. Xing et al. — Teacher-Student

| 项目 | 内容 |
|------|------|
| **标题** | Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight |
| **作者** | Jiaxu Xing, Angel Romero, Leonard Bauersfeld, Davide Scaramuzza |
| **单位** | UZH RPG |
| **发表** | CoRL 2024 |
| **arXiv** | [2403.12203](https://arxiv.org/abs/2403.12203) |
| **为什么选读** | Teacher-Student三阶段框架（特权训练→IL蒸馏→RL微调）。如果我们的视觉PPO收敛困难，这篇提供了BC预热+RL微调的方法。 |

---

## 四、阅读时间线

```
Week 1
  D1 am  GRaD-Nav           ████████████████ 精读 §3-4 + 代码
  D1 pm  SOUS VIDE          ████████████████ 精读 §3 + FiGS
  D2 am  3DGS 原论文         ████████████     精读 §3-5
  D3 am  EmbodiedSplat      ████████████     精读 §3-4
  D4-D6  Swift              ██████           重读 Method + 对照代码

Week 2
  D8 am  Ferede (域随机化)    ████████         泛读 §4
  D9 am  Peng (动力学随机化)  ████████         泛读 §3-4
  D10 am ADR (OpenAI)       ████████         泛读 §4
  D11    碎片时间补           Geles + GRaD-Nav++

Week 3
  D15 am Loquercio (Agile)   ██████           泛读 §2-3
  D15 pm Song (RL vs MPC)    ██████           泛读 全文
  D16 am 重读 GRaD-Nav + SOUS VIDE 的 Related Work

Week 3+ (论文写作时)
  GaussFly / Liquid+3DGS / Dream to Fly / Xing  — Discussion引用
```

---

## 五、论文引用速查

### Related Work 段落规划

| 段落 | 主题 | 引用 |
|------|------|------|
| §2.1 | 3DGS for Robot Simulation | GRaD-Nav, SOUS VIDE, EmbodiedSplat, VR-Robo, RL-GSBridge |
| §2.2 | Visual RL for Drone Navigation | Swift, Agile Autonomy, Dream to Fly |
| §2.3 | Domain Randomization & Robustness | Peng (ICRA 2018), Ferede (ICRA 2025), ADR (OpenAI 2019) |
| §2.4 | Perception Degradation in RL | 你的v1子课题A工作 |

### 实验对比引用

| 对比对象 | 引用 | 对比维度 |
|----------|------|---------|
| 可微分RL baseline | GRaD-Nav | 样本效率、退化鲁棒性 |
| BC baseline | SOUS VIDE | 训练时间、退化鲁棒性 |
| 域随机化方法论 | Ferede, Peng | 退化水平选择、实验设计 |
| 非对称架构 | Geles (RSS 2024) | 消融实验 |
| 向量噪声参照 | 你的v1子课题A | 噪声结构差异→鲁棒性差异 |

---

## 六、已有论文（本地PDF）

| 论文 | 本地路径 | 状态 |
|------|---------|------|
| GRaD-Nav (v3) | `docs/2503.03984v3.pdf` | ✅ 已有（8页） |
| Peng et al. (Dynamics Rand) | `docs/1710.06537v3.pdf` | ❌ 未见（原清单声称已有，实际缺失） |
| Ferede et al. (One Net) | `docs/2504.21586v1.pdf` | ❌ 未见（原清单声称已有，实际缺失） |
| Ferede 精读笔记 | `docs/_ferede_full.txt` | ❌ 未见（原清单声称已有，实际缺失） |
| Peng 精读笔记 | `docs/_peng_full.txt` | ❌ 未见（原清单声称已有，实际缺失） |

**精读笔记**（`docs/reading-notes/`）：
- ✅ `grad-nav.md` — GRaD-Nav 全文精读（2026-08-02，含实验数据核对）

**需要下载的**：Peng, Ferede, SOUS VIDE, 3DGS原论文, EmbodiedSplat, Geles (RSS 2024), ADR

---

*整理时间：2026-07-21*
*共 15 篇：🔴5篇必读精读 + 🟡6篇必读泛读 + 🟢4篇选读*
