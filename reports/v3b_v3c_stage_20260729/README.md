# V3b 消融与 V3c 跨场景阶段报告

日期：2026-07-29

## 结论

- V3b 已完成代码与实验定义审计，扩展了无需重训的输入依赖诊断；本轮未完成
  新评估，因为可用运行环境未成功恢复。
- V3c 正式实验当前不能启动。服务器只有一个标准真实 3DGS 场景，且环境的
  碰撞几何不随场景变化；直接更换 PLY 会混淆视觉域迁移和导航场景泛化。

## V3b：当前证据

已有真实 gsplat 结果来自
`reports/ablation_seed2_500ep/ablation_20260726_191216.json`，每项 200 episodes：

| 配置 | SR | Collision | Timeout | 相对 baseline |
|------|---:|----------:|--------:|--------------:|
| baseline | 82.0% | 18.0% | 0.0% | 0 pp |
| const-depth | 71.0% | 14.5% | 14.5% | -11 pp |
| no-target-direction | 0.0% | 15.5% | 84.5% | -82 pp |
| const-depth + no-target-direction | 0.0% | 1.0% | 99.0% | -82 pp |

这证明该 checkpoint 强依赖目标方向，并会利用深度，但它是测试时清零输入，
不是从头重训的结构消融。

本轮代码新增：

- `no_velocity` 输入清零；
- baseline、三种单输入/组合输入诊断；
- 成功率 Wilson 95% CI；
- CSV 与 JSON 双格式输出。

### 2026-07-29 新门控

robust-best seed2、真实 gsplat、base seed 20260728、每配置100 episodes：

| 配置 | SR | Collision | Timeout |
|------|---:|----------:|--------:|
| baseline | 81% | 19% | 0% |
| const-depth | 77% | 23% | 0% |
| no-velocity | 11% | 53% | 36% |
| no-target-direction | 0% | 7% | 93% |
| no-depth + no-velocity | 12% | 60% | 28% |
| no-velocity + no-target-direction | 0% | 7% | 93% |
| all-inputs-ablated | 0% | 0% | 100% |

完整数据位于 `reports/v3b_input_gate_seed2_7x100_20260729/`。baseline复现通过，
速度输入贡献明确，已进入无速度策略从头重训小试。

当前网络不是非对称 Actor-Critic。Actor 和 Critic 使用相同的 depth、velocity、
target-direction 和共享特征，因此原计划中的“无特权 Critic”没有有效对照。

## V3c：阻塞证据

服务器资产检查只发现：

`/root/data/gs_data/ply_exports/gate_mid_new_gs.ply`

本地 Replica 场景仍是 mesh/texture，尚未训练为 3DGS。现有环境同时固定使用
`[-10,10]×[-10,10]×[0,10]` 边界、固定目标分布和三个球形碰撞体。不同 PLY
若没有对应坐标标定与碰撞几何，只能构成“换渲染背景”的视觉域代理实验。

## 下一次允许启动的顺序

1. 固定并记录一个可复现的 CUDA + torch + gsplat Python 环境。
2. 运行 V3b 输入诊断门控：robust-best seed2，7 配置 × 100 episodes。
3. 若输出合理，扩为 3 seeds × 7 配置 × 200 episodes。
4. 实现浅 CNN 和无速度输入的从头训练版本；每个版本先 1 seed × 100 episodes。
5. 为 garden、office0、room0、apartment0 各准备标准 3DGS PLY、坐标变换、
   场景边界、碰撞几何与共同起终点采样协议。
6. 先做 3 场景 × clean × 20 episodes V3c 冒烟，再执行
   3 场景 × 5 档 × 50 episodes 正式评估。

任何步骤连续失败两次即终止该步骤，不重复启动。
