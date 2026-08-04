# V3 hybrid-geodesic 单种子门控

日期：2026-07-30

## 协议

- 训练：seed2，100 updates，8 environments，204,800 steps；
- 渲染器：真实 gsplat；
- 视觉与碰撞：同一 `gate_mid_new_gs.ply`；
- 训练课程：避障任务比例 10% -> 30% -> 50%；
- 奖励：连续测地进度 + 局部安全路径 heading；
- 独立评估：base seed `20260728`；
- 配置：baseline、const-depth，各200 episodes；
- 分层：直线可达 clear、必须绕行 avoidance。

## 结果

| 配置 | 总SR | Wilson 95% CI | Clear SR | Avoidance SR | Collision | Timeout |
|------|-----:|--------------:|---------:|-------------:|----------:|--------:|
| baseline | 48.0% | 41.2–54.9% | 86.6% | 11.7% | 52.0% | 0.0% |
| const-depth | 11.0% | 7.4–16.1% | 19.6% | 2.9% | 71.5% | 17.5% |

## 判定

- 正常深度相对const-depth提高37个百分点，视觉必要性成立；
- clear SR达到86.6%，说明策略可以学习基本到达；
- avoidance SR仅11.7%，88.3%的avoidance任务发生碰撞；
- 不扩展3 seeds x 500 updates；
- 停止继续调整测地奖励权重。

下一步用局部安全路径方向作为诊断性观测，判断失败是否源于最终目标方向观测
与隐藏局部路径奖励不一致。该诊断若成功，只能支持分层规划路线，不能作为
端到端视觉导航结果。

原始数据：

- `ablation_20260730_025521.csv`
- `ablation_20260730_025521.json`
- `watcher.log`（仅本地保存，不提交GitHub）
