# 认知债务防御 — Vibe Coding 工程化知识库

> **Vibe Coding 的终点不是"代码能跑"，而是"团队能懂"。**
>
> 这是一个**"方法论 + 工具配置 + 实战记录"三位一体**的知识工程项目。
> 核心问题：**认知债务 = 代码库存量 − 团队理解力**。
>
> `.claude/agents/` 下的 10 个 Agent 配置可直接被 Claude Code 加载使用。

---

## 📚 两份文档，覆盖全部

| 文档 | 定位 | 阅读时长 | 何时读 |
|------|------|---------|--------|
| **[flash-card.md](flash-card.md)** | 日常操作速查卡 | 2 分钟 | 每次写代码时打开 |
| **[pro-handbook.md](pro-handbook.md)** | 完整方法论手册 | 20 分钟 | 想深度学习时 |

---

## 🤖 10 个 Claude Code Agent 速览

| 层 | Agent | 模型 | 何时用 |
|----|-------|------|--------|
| 规划 | `@spec-writer` | Sonnet | 需求模糊 → 结构化 Spec |
| 规划 | `@modular-architect` | Sonnet | 需求 >200 行 → 拆 3-5 个模块 |
| 规划 | `@knowledge-map-maintainer` | Haiku | 任何改动后 → 更新认知地图 |
| 执行 | 主对话 | 继承 | 按模块逐个写代码 |
| 执行 | `@fast-debugger` | Sonnet | 粘贴错误 → 最小修复 |
| 执行 | `@microworld-builder` | Sonnet | 复杂模块 → 交互式训练场 |
| 质量 | `@knowledge-pack-generator` | Sonnet | 每次代码改动 → 知识包 |
| 质量 | `@understanding-reviewer` | Opus | PR Review → 4 层理解审查 |
| 质量 | `@understanding-gate` | Opus | 合并前 → 通过/不通过 |
| 质量 | `@output-reviewer` | Opus | 交付重要 AI 输出 → 5 层终审 |
| 反思 | `@lesson-capturer` | Haiku | 一轮开发结束 → 总结教训写入教训库 |

**核心抽象**：理解门槛 = **定位 / 数据流 / 边界 / 故障** 4 层模型。

**升级路径**：haiku → sonnet（需判断）→ opus（需审查）→ 人类（需决策）。

---

## 🧰 5 个配套工件模板（`templates/`）

| 模板 | 防御层 | 作用 |
|------|--------|------|
| [`templates/CLAUDE.md`](templates/CLAUDE.md) | 项目心智模型 | AI 读取的项目入口 |
| [`templates/RULES.md`](templates/RULES.md) | 工程规则 | 技术栈/目录/规范/红线/质量门槛 |
| [`templates/docs/knowledge-map.md`](templates/docs/knowledge-map.md) | 知识地图 | 系统长什么样 |
| [`templates/docs/understanding-gate.md`](templates/docs/understanding-gate.md) | 理解门槛 | 合并前 4 层检查清单 |
| [`templates/.github/pull_request_template.md`](templates/.github/pull_request_template.md) | PR 知识包 | GitHub PR 默认填充模板 |

## 🚀 30 秒安装

```powershell
Copy-Item -Recurse .claude\agents <目标项目>\.claude\agents
Copy-Item -Recurse templates\* <目标项目>
```

---

## 🗂️ 仓库结构

```
better_vibe_coding/
├── flash-card.md                          # 日常操作速查（2 分钟）
├── pro-handbook.md                        # 完整方法论手册（20 分钟）
├── README.md                              # 本文件（导航入口）
├── .gitignore
│
├── .claude/agents/                       # 10 个可加载 Agent 配置
│   ├── spec-writer.md
│   ├── modular-architect.md
│   ├── knowledge-pack-generator.md
│   ├── knowledge-map-maintainer.md
│   ├── fast-debugger.md
│   ├── microworld-builder.md
│   ├── understanding-reviewer.md
│   ├── understanding-gate.md
│   ├── output-reviewer.md
│   └── lesson-capturer.md
│
├── docs/
│   └── lessons-learned.md                # 教训库（活文档）
│
├── change/                               # 方法论实战应用
│   ├── grad-nav-code-review.md           # GRaD-Nav 代码审查案例
│   └── grad-nav-diff-review.md           # 差异自审案例
│
└── templates/                            # 5 个配套工件模板
    ├── CLAUDE.md
    ├── RULES.md
    ├── docs/
    │   ├── knowledge-map.md
    │   └── understanding-gate.md
    └── .github/
        └── pull_request_template.md
```

---

## ⚖️ 核心格言

- 约束在前，代码在后。规则后置的成本是前置的 3×。
- 一个 Prompt = 一个模块（≤200 行）。
- 每增加一行代码，就增加一行理解。
- Code Review 的终点不是 LGTM，而是"我懂了"。
- AI 生成的终点不是"写完了"，而是"审过了"。
- 知道局限的 Agent 比永远说"能做到"的 Agent 更可信。

---

## 📖 版本说明

- 方法论版本: **V2.1** — 新增能力边界声明机制 + 文档合并为 flash/pro 双文档体系
- Agent 数量: **10**（规划层 3 + 执行层 1主对话+2 + 质量层 4 + 反思层 1）
- 文档体系: flash-card.md（日常速查）+ pro-handbook.md（完整方法论）+ README.md（导航）
- 本仓库由个人研究者构建，欢迎借鉴但请保留来源引用。
