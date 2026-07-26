# V1-Baseline-S1 训练结果报告

## 结论

V1-Baseline-S1 已完成 3000 episodes 训练，最终成功率 90%，达到 Phase V1
Gate G3（成功率 > 80%）。训练期间未出现 Traceback、NaN 或 CUDA OOM。

## 实验配置

| 项目 | 值 |
|---|---|
| 场景 | gate_mid 3DGS |
| 算法 | Visual PPO |
| GPU | NVIDIA RTX 3090，GPU 0 |
| Episodes | 3000 |
| 并行环境 | 8 |
| Rollout steps | 256 |
| 总环境步数 | 6,144,000 |
| 学习率 | 3e-4，线性衰减 |
| 渲染器 | gsplat 1.5.3，64×64 depth |

## 结果

| 指标 | 结果 |
|---|---|
| 总耗时 | 7 小时 32 分 |
| 平均吞吐 | 227 steps/s |
| 最终成功率 | 90% |
| 最终碰撞率 | 10% |
| 最终平均奖励 | 438.14 |
| 历史最佳成功率 | 90% |
| 最佳检查点平均奖励 | 455.93 |
| Gate G3 | 通过 |

评估每次使用 20 episodes，因此中途成功率存在 65%–90% 波动。最佳模型保存于
`visual_ppo_best.pth`，对应首次达到 90% 成功率的检查点；它的平均奖励 455.93，
高于最终检查点的 438.14。

## 启动前修复

真实 GPU 自检最初得到全 0.10m 深度。根因是标准 3DGS PLY 中 `scale_*` 保存为
对数尺度、`opacity` 保存为 logit、`f_dc_*` 保存为零阶球谐系数，而加载器将其
作为直接渲染参数使用。修复解码后，测试位姿深度范围恢复为约 1.35–20.00m。

## 文件与校验

| 文件 | SHA-256 |
|---|---|
| `train.log` | `6b2dc5250afb96c75910e360fc0cdf121de41178c02c132d189d651ac5998ecd` |
| `visual_ppo_best.pth` | `d793d0a7759f07e03820254b9772565488c9ec114c3184a9876ebc9b60b8348c` |

## 下一阶段前的阻塞项

当前 `eval_degradation.py` 默认创建 mock 环境，没有向真实评估环境传入
`renderer=gsplat` 和 PLY 路径；真实 GS 的 Gaussian 稀疏化也尚未操作
GSplatRenderer 内部 Gaussian 张量。因此不能直接把现有 `--axis all` 输出当作
Phase V2 真实 3DGS 结果。应先补齐真实渲染参数、真实 Gaussian 稀疏化和小规模
一致性测试，再启动完整衰减曲线。

