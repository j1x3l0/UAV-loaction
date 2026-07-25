# 教训知识库 — UAV Visual RL 开发实战经验积累

> **活文档 (Living Document)**：每轮开发后由 `@lesson-capturer` 自动更新。
> 组织方式：按分类 → 按日期倒序。所有教训来自真实开发过程。
>
> 每个教训回答：**什么场景、什么问题、什么根因、什么教训、下次怎么做。**

---

## 有效模式 ✅

> 被验证有效的做法，值得推广和复用。

### 教训: 论文速览后必须做代码审查
- **日期:** 2026-07-22
- **分类:** 有效模式 ✅
- **场景:** 阅读 GRaD-Nav 论文后直接做对比分析
- **问题:** 论文描述与代码实现存在多处关键差异（SqueezeNet 冻结、观测是 57D 向量而非图像、CENet vs VAE）
- **根因:** 论文写作有简化/美化倾向，代码是实现真相
- **教训:** 论文速览 → 代码审查 → 修正文档——三轮递进，不可跳过代码直接信论文
- **行动:** 任何论文分析都必须包含代码审查环节，文档留修正空间
- **关联:** @knowledge-pack-generator, docs/grad-nav-comparison.md

### 教训: 多文件修改后做交叉一致性验证
- **日期:** 2026-07-25
- **分类:** 有效模式 ✅
- **场景:** 重构degradation_utils时修改4个文件的import路径
- **问题:** 一处import遗漏导致smoke test失败
- **根因:** 人脑不擅长追踪跨文件的一致性
- **教训:** 任何涉及 3+ 文件的修改，完成后跑一次 grep 交叉验证 + import检查
- **行动:** 修改后运行 `python -c "from envs.X import Y"` 逐模块验证
- **关联:** @output-reviewer, degradation_utils.py

### 教训: 退化轴集中管理优于分散定义
- **日期:** 2026-07-25
- **分类:** 有效模式 ✅
- **场景:** v2退化轴原本分散在visual_drone_env.py和eval_degradation.py两处
- **问题:** 两份 DEGRADATION_AXES 定义可能不同步，新退化轴接入不一致
- **根因:** 缺乏"单一数据源"原则 — 同一概念不宜多处定义
- **教训:** 创建 degradation_utils.py 作为退化轴的唯一权威来源，eval和plot都从这里引用
- **行动:** 任何跨模块共享的常量/配置都应提取到专用模块
- **关联:** degradation_utils.py, @knowledge-map-maintainer

---

## 反模式 ❌

> 被验证有害的做法，必须避免。

### 教训: 不要让 Agent 一次做太多
- **日期:** 2026-07-22
- **分类:** 反模式 ❌
- **场景:** 要求 agent "审查整个代码库"
- **问题:** 输出超时或被截断，结果不完整
- **根因:** 单次 prompt 的任务量超过了模型的上下文处理能力
- **教训:** 一次只给一个具体任务。大任务必须拆分后才交给 agent。
- **行动:** 遵循 @modular-architect 的拆分原则——每个 Prompt = 一个模块
- **关联:** @modular-architect, CLAUDE.md

### 教训: 不要在没确认的情况下删除文件
- **日期:** 2026-07-22
- **分类:** 反模式 ❌
- **场景:** 发现"显然无用"的文件想删除
- **问题:** 所谓"显然"往往建立在不够完整的理解之上
- **根因:** 文件依赖关系可能隐藏在非显而易见的地方
- **教训:** 先列出待删文件清单并逐项确认，不要直接删除
- **行动:** 删除操作分两步——列表确认 → 确认后删除
- **关联:** CLAUDE.md

---

## 陷阱 ⚠️

> 容易踩的坑，表面看起来没问题但实际有风险。

### 教训: MockRenderer深度分布≠真实3DGS深度分布
- **日期:** 2026-07-25
- **分类:** 陷阱 ⚠️
- **场景:** 在MockGSRenderer上训练收敛后直接迁移到真实3DGS
- **问题:** Mock使用简单射线-球体求交，真实3DGS深度图有复杂几何纹理和噪声模式
- **根因:** Mock渲染器只是一个开发占位，不是3DGS的仿真器
- **教训:** Mock上训练的模型迁移到真实3DGS后必须重新tune超参和验证退化曲线
- **行动:** Phase 1接入真实3DGS后的第一步：在clean条件下重新训练baseline，对比Mock训练的SR
- **关联:** visual_drone_env.py:MockGSRenderer, research_plan_v2.md

### 教训: Windows GBK编码导致Unicode字符crash
- **日期:** 2026-07-25
- **分类:** 陷阱 ⚠️
- **场景:** 单元测试中使用了 ✓ ✗ Unicode字符，Windows终端默认GBK编码
- **问题:** `UnicodeEncodeError: 'gbk' codec can't encode character`
- **根因:** Windows Python默认编码不是UTF-8
- **教训:** 测试文件中的特殊字符需配 `PYTHONIOENCODING=utf-8`，或在测试中使用ASCII替代
- **行动:** 项目README中添加Windows环境配置说明；CI/CD强制UTF-8
- **关联:** test_visual_env.py, test_visual_agent.py

---

## 洞察 💡

> 深层认知突破，改变了对某个问题的理解方式。

### 教训: PPO vs DDRL的选择不是性能问题而是实验设计问题
- **日期:** 2026-07-24
- **分类:** 洞察 💡
- **场景:** 对比本项目的PPO与GRaD-Nav的SHAC/DDRL时
- **问题:** 最初纠结于"PPO样本效率低"，但这不是关键
- **根因:** 鲁棒性分析的核心问题是"视觉退化如何影响策略"，而非"如何最快训练策略"
- **教训:** PPO的reward信号不依赖GS精度 → 比DDRL更适合做"退化轴效应"的公平比较。选型标准应从"训练效率"转为"实验有效性"
- **行动:** 论文中Method部分的选型论证应强调这个实验设计维度
- **关联:** research_plan_v2.md, docs/grad-nav-comparison.md

---

*最后更新: 2026-07-25 | 由 @lesson-capturer 维护 | 共 8 条教训*
