# Vibe Coding 操作速查卡

> 每次写代码时打开，2 分钟找到答案。需要深入理解时 → [pro-handbook.md](pro-handbook.md)

---

## 🔧 必要项：每次开发必看

### Agent 速查表

| Agent | 一句话 | 触发词 | 何时不用 |
|-------|--------|--------|---------|
| `@spec-writer` | 模糊想法 → 结构化需求 | "我想要一个…但不确定怎么做" | 需求已明确 |
| `@modular-architect` | 大任务拆成 3-5 个模块 | 预计 >200 行 | 单文件小改 |
| `@knowledge-map-maintainer` | 更新项目认知地图 | 一轮开发结束 | 单个文件小改 |
| `@fast-debugger` | 贴错误 → 定位 → 最小修复 | 任何报错 | 重构级改动 |
| `@knowledge-pack-generator` | 代码改动 → WHY+数据流+边界+风险 | 每次代码改动后 | 纯配置/文档修改 |
| `@microworld-builder` | 生成交互式教学脚本 | 复杂模块需深度理解 | 简单工具函数 |
| `@understanding-reviewer` | PR 审查可读性 | PR Review | Bug 验证 |
| `@understanding-gate` | 合并前理解门槛检查 | 合并前最后一步 | 日常开发 |
| `@output-reviewer` | AI 输出最终审查 | 交付重要输出前 | 探索性对话 |
| `@lesson-capturer` | 总结教训 → 写入教训库 | 一轮开发结束 | 纯探索/无改动 |

### 场景决策树

```
接到新需求          代码写完了             遇到错误
  │                    │                     │
  ├─ 模糊?             ├─ @knowledge-pack-   └─ @fast-debugger
  │  → @spec-writer    │   generator           不要自己逐行排查
  │                    ├─ @understanding-      贴完整错误日志
  ├─ >200行?           │   reviewer
  │  → @modular-       ├─ @understanding-
  │     architect      │   gate
  │                    ├─ @output-reviewer
  └─ 直接写            ├─ @knowledge-map-
     (≤200行/模块)     │   maintainer
                       └─ @lesson-capturer
```

### 升级决策速查

| 你用的模型 | 遇到以下情况 → 升级 | 判断依据 |
|-----------|-------------------|---------|
| **haiku** | → **sonnet** | 任务需"判断"而不仅是"执行" |
| **haiku** | → **opus** | 需深度推理/交叉验证 |
| **sonnet** | → **opus** | 需"故意找茬"式审查/跨系统分析 |
| **sonnet** | → **人类** | 技术选型/不可逆操作/安全敏感 |
| **opus** | → **人类** | 涉及金钱/合规/安全的最终决策 |

### 核心反模式

| # | 不要 | 原因 | 正确做法 |
|---|------|------|---------|
| 1 | 一次让 Agent 做太多 | 超时/截断/质量下降 | 一个 Prompt = 一个模块（≤200行） |
| 2 | 跳过代码直接信论文 | 论文描述≠代码实现 | 论文速览 → 代码审查 → 修正文档 |
| 3 | 同时改多个不相关文件 | 上下文污染/不一致 | 一次只做一类修改 |
| 4 | 未确认就删除文件 | 依赖关系可能隐藏 | 先列表确认 → 再删 |

### 日常开发标准流程

```
规则锚定 → 需求拆分(@spec-writer/@modular-architect)
   → 逐模块写代码（≤200行/模块）
   → @knowledge-pack-generator（每个模块）
   → @understanding-reviewer → @understanding-gate
   → @knowledge-map-maintainer → @lesson-capturer
   → 合并
```

---

## 📋 可加项：按需查阅

### 内置 Agent

| Agent | 用途 |
|-------|------|
| `Explore` | 只读搜索，扫多个文件/目录 |
| `general-purpose` | 复杂多步搜索+执行 |
| `Plan` | 设计实现方案 |

### 30 秒安装

```powershell
Copy-Item -Recurse .claude\agents <目标项目>\.claude\agents
Copy-Item -Recurse templates\* <目标项目>
```

### 能力边界三层速查

| 层级 | 不擅长 | 遇到这些 → 手动 | 遇到这些 → 升级 |
|------|--------|----------------|----------------|
| **haiku** | 深度推理、创造性设计、安全审计 | 架构决策、不可逆操作 | 需判断的任务 → sonnet；需审查的任务 → opus |
| **sonnet** | 对抗性验证、极端边界分析、跨领域综合 | 技术选型、破坏性变更确认、部署审批 | 需"找茬"的审查 → opus |
| **opus** | 替代领域专家、100% 事实保证、实时操作 | 金钱/合规/安全决策、领域专业知识 | → 人类 |

### 手动接管信号

以下信号出现时应人工介入：Agent 输出含 `⚠️ 不确定` / 同一建议反复出现无进展 / 涉及不可逆操作 / 涉及合规/财务 / 你感觉"不对劲"

---

*由 [pro-handbook.md](pro-handbook.md) 提供完整方法论支撑 | 最后更新：2026-07-23*
