# RULES.md — UAV Visual RL 3D Path Planning 工程规则

> 作用："约束在前，代码在后。规则后置的成本是前置的 3×。"
> 本文件锁定项目的技术栈、目录结构、命名规范、安全红线、质量门槛。
> 在为本项目生成任何代码之前，请先理解并锁定以下规则。

---

## 1. 技术栈

- **框架**: Gymnasium (RL环境), PyTorch (深度学习)
- **语言**: Python 3.10+ (类型注解推荐但不强制)
- **渲染**: 3DGS (gsplat/Nerfstudio, Phase 1), MockGSRenderer (Phase 0开发用)
- **可视化**: Matplotlib (论文级图表), 可选 TensorBoard
- **包管理**: pip (当前), 考虑迁移至 poetry/uv
- **禁止依赖**: TensorFlow (团队只维护PyTorch), Open3D (除非Phase 1确认需要)

## 2. 目录结构

```
jxl-UAV-loaction-01-main/
├── rlproject-swift-improved/     → v1/v2 主代码库
│   ├── envs/                     → RL环境 (Extension层)
│   │   ├── drone_env.py          → v1 向量环境
│   │   ├── visual_drone_env.py   → v2 视觉环境
│   │   └── degradation_utils.py  → v2 退化工具
│   ├── core/                     → 智能体核心 (Foundation层)
│   │   ├── ppo_agent.py          → v1 向量PPO
│   │   └── visual_ppo_agent.py   → v2 视觉PPO
│   └── scripts/                  → 训练/评估/绘图 (Application层)
│       ├── train.py / train_visual.py
│       ├── eval_baseline.py / eval_degradation.py
│       └── plot_*.py
├── rlproject-jxl-rlib/           → 旧版 (已弃用, 保留作基线对比)
├── docs/                         → 文档 (知识地图/理解门槛/教训库)
├── past/                         → v1 历史归档
├── better_vibe_coding/           → AI协作框架参考
├── CLAUDE.md                     → 项目心智模型入口
├── RULES.md                      → 本文件 (工程规则锚定)
└── weekly_plan_v2.md             → 当前周计划
```

## 3. 代码规范

- **命名**: snake_case 变量/函数, PascalCase 类, UPPER_SNAKE 常量
- **注释**: 每个模块头部必须有知识包注释 (位置/WHY/数据流/边界/风险)
- **类型注解**: 公开接口必须标注 (返回值 + 关键参数)
- **错误处理**: 不使用裸 except; catch 具体异常并 log
- **日志**: 使用 `logging.getLogger(__name__)`, 禁止 `print()` 在生产代码中
- **行数限制**: 每个模块 ≤ 200 行 (一个 Prompt = 一个模块)

## 4. 安全红线（绝对禁止）

- 禁止硬编码密钥、Token、API Key
- 禁止使用 `eval()`、`exec()` 或等效动态执行
- 禁止忽略输入校验（环境 step 的 action 必须 clip）
- 禁止在 `__init__` 中执行网络请求或大文件下载 (延迟加载)
- 禁止修改 past/ 归档目录中的文件 (历史记录不可篡改)

## 5. 质量门槛 —— 代码交付前必须通过

- [ ] 模块头部知识包完整 (位置/WHY/数据流/边界/风险)
- [ ] 3 种输入自检: 正常 / 空 / 异常
- [ ] 无硬编码值 (提取为参数或常量)
- [ ] 关键函数的错误处理完整
- [ ] 不引入 RULES.md 禁止的依赖
- [ ] ≤ 200 行 (超过需经 @modular-architect 拆分)

## 6. 边界与解释权

- 本规则一经锁定，AI 在本次会话及后续会话中不得擅自变更。
- 如需新增/修改规则，必须由人类明确确认；AI 只能建议，不能自决。
- 与本规则冲突的代码视为"未通过质量门槛"，禁止交付。
- `past/` 目录为只读归档，任何修改需显式人工批准。
