# UAV Visual RL 修改建议（基于 V1/V2 实测）

> 依据：200-episode clean 复核与真实 gate_mid Phase V2（1250 episodes）

## 结论

当前最优模型的独立 clean 成功率为 75.5%（95% CI 69.1%–80.9%），尚未稳健
通过 G3。Phase V2 中没有任何退化轴跌破 50% 或 20%，G4 也未通过。建议暂停
V3 大规模鲁棒训练，先修复训练稳定性、评估统计和退化定义。

## P0：训练正确性

### 1. 修复自适应熵系数发散

`AdaptiveEntropyCoeff` 接收的是正值 Normal entropy，却把目标设为
`-action_dim`。因此 `entropy - target_entropy` 始终为正，`log_alpha` 会持续
增大。训练日志中 total loss 从正常值降到 -3498，而 entropy 固定约 4.26，
与该问题一致。

建议：

- 将目标改为同一符号体系下的正 entropy，例如 2.0–3.0。
- 对 `log_alpha` 增加上下界，例如 `[-9.2, -1.6]`（alpha 约 1e-4–0.2）。
- 记录 `entropy_coeff` 曲线，并增加“有限且未触顶”的测试。
- 在修复前不要用 total loss 判断收敛。

验收：3000ep 内 alpha 不单调爆炸，total loss 有界，entropy 随训练逐步下降。

### 2. 修复动作截断与 log-prob 不一致

当前从 Normal 采样后直接 clamp 到 `[-1,1]`，却按截断后的动作计算普通 Normal
log-prob。边界动作的概率密度不正确，会给 PPO ratio 引入偏差。

建议使用 tanh-squashed Gaussian，并加入 Jacobian 修正；保存 pre-tanh action
或一致的 corrected log-prob。

验收：边界动作比例、PPO approximate KL、clip fraction 可监控且无异常尖峰。

### 3. 修复 GAE bootstrap 状态

rollout 末尾 bootstrap 当前使用最后一个 `state`，不是最后 transition 的
`next_state`。应按每个环境保存最后 next observation，并在非终止时估计
`V(s_{t+1})`。

验收：构造短轨迹单元测试，与手算 GAE/returns 一致。

## P0：可靠评估

### 4. 固定评估协议

- 模型选择：至少 100 episodes，不再用 20 episodes 选择 best。
- 最终报告：200–500 episodes，报告 Wilson 95% CI。
- 所有退化档共享 episode seeds；同时用 3 组独立 seed sets 检查泛化。
- 保存 per-episode JSON：seed、结果类型、奖励、步数、最小障碍距离。
- 同时报告 SR、CR、timeout、平均奖励和成功轨迹长度。

验收：重复执行同一 seed set 得到逐 episode 完全一致结果。

### 5. 多随机种子训练

至少训练 3 个随机种子。论文结果报告均值、标准差和置信区间，避免单次训练的
90% 小样本结果被误认为稳定基线。

G3 建议改为：3 seeds × 200 eval 中，平均 SR > 80%，且 95% CI 下界尽量接近
或超过 80%。

## P1：提升 clean baseline

建议按最小实验矩阵排查：

| 实验 | 变化 | 目的 |
|---|---|---|
| B0 | 修复上述 P0，其他不变 | 建立正确基线 |
| B1 | fixed entropy alpha=0.01 | 隔离自适应熵问题 |
| B2 | rollout 256→512 | 降低优势估计方差 |
| B3 | hidden 128→256 | 检查容量不足 |
| B4 | 加 observation normalization | 稳定 depth/vec 尺度 |
| B5 | clean curriculum | 降低早期碰撞 |

先用 500ep × 3 seeds 筛选，再对前两名运行 3000ep。

## P1：重新定义退化强度

### Gaussian

当前 opacity×体积 top-k 在 5% 时仍保留主要表面，SR 基本不降。建议同时评估：

- opacity-only、投影视觉贡献、随机删除三种排序；
- 进一步加入 2%、1%、0.5%；
- 统计图像 coverage、空洞率、PSNR/Depth-MAE，确认退化强度单调。

### 分辨率

当前 8px 与 4px 的 bilinear 结果趋同。建议使用 `{64,32,16,8,2}`，并分别比较
nearest/bilinear；深度图优先保留 float，避免 uint8 量化混入额外误差。

### 深度噪声

σ≤0.2m 对当前策略过弱。建议扩展到 `{0,0.1,0.25,0.5,1.0}`，并增加：

- 距离相关噪声；
- 结构性空洞；
- 局部偏置/尺度漂移；
- 边缘 flying pixels。

### 视角覆盖

目前是后处理噪声模拟，不是真实训练视角缺失。短期应明确命名为
`viewpoint_uncertainty`；长期应从受限相机集合重建/裁剪 Gaussians。
评价时联合使用 SR、timeout 和平均奖励，因为本次主要表现为超时增加。

### 光照

depth-only 策略下应保留为负对照，不计入“致命轴”排序。若研究光照鲁棒性，
必须引入 RGB/RGB-D policy 或光照影响深度估计的上游模型。

## P2：场景与任务一致性

当前碰撞几何来自手工球体，而视觉来自 gate_mid 3DGS，两者可能不完全对齐。
应把碰撞体、起终点和 3DGS 坐标统一可视化，检查策略是否在利用与视觉无关的
向量信息或几何错位。

建议做两项消融：

- depth 置常数，仅保留 vec；
- target direction 去除，仅保留 depth+velocity。

若 depth 置常数仍有高 SR，说明当前实验不能证明视觉鲁棒性。

## 推荐执行顺序

1. 修复 entropy、动作分布和 GAE，并补单元测试。
2. 跑 B0/B1 的 500ep × 3 seeds。
3. 做常数 depth 与无 target-direction 消融。
4. 统一碰撞几何与 3DGS 坐标。
5. 重标定 Gaussian、noise、viewpoint 退化范围。
6. baseline 达到新 G3 后重跑 Phase V2。
7. 只有出现稳定、可解释的临界点后，再进入 V3 Rand/Fixed/Curric。

