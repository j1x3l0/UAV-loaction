# 无人机强化学习项目 (Drone RL Project)

本项目是一个基于深度强化学习的无人机三维路径规划系统，使用PPO和SAC算法，支持深度传感器输入和向量状态混合观测空间。

## 📁 项目结构概览

### 核心代码文件

#### ✅ **可用文件（当前推荐使用）**

| 文件 | 功能描述 | 状态 | 使用说明 |
|------|----------|------|----------|
| **`drone_env.py`** | 无人机三维路径规划环境，支持深度传感器和向量状态混合观测 | ✅ 正常工作 | 核心环境文件，支持深度图像输入 |
| **`ppo_agent.py`** | PPO智能体实现，支持深度图像编码器和向量状态编码器 | ✅ 正常工作 | 核心PPO算法实现 |
| **`train.py`** | 主训练脚本，支持PPO、SAC算法和对比实验 | ✅ 正常工作 | 训练模型：`python train.py --algorithm ppo --episodes 5000` |
| **`evaluate_ppo_model.py`** | PPO模型评估脚本，计算成功率、碰撞率等指标 | ✅ 正常工作 | 评估模型：`python evaluate_ppo_model.py` |
| **`config.py`** | 训练参数配置类，支持命令行参数解析 | ✅ 正常工作 | 统一配置管理 |
| **`requirements.txt`** | Python依赖包列表 | ✅ 正常工作 | 安装依赖：`pip install -r requirements.txt` |

#### 📊 **评估和结果文件**

| 文件 | 功能描述 | 状态 |
|------|----------|------|
| `evaluation_results_ppo_20260129_151159.json` | 最新PPO模型评估结果（50次） | ✅ 可用 |
| `baseline_evaluation.py` | 基线算法对比评估 | ✅ 可用 |
| `baseline_comparison.py` | 算法对比实验脚本 | ✅ 可用 |

#### 📝 **文档和报告**

| 文件 | 内容描述 | 状态 |
|------|----------|------|
| `docs/future_improvements.md` | 项目未来改进方向 | ✅ 可用 |
| `docs/cnn_optimization_report.md` | CNN优化报告 | ✅ 可用 |
| `docs/depth_image_size_comparison_report.md` | 深度图像尺寸比较报告 | ✅ 可用 |
| `docs/rl_lib_vs_isaac_gym.md` | RLlib与Isaac Gym对比报告 | ✅ 可用 |
| `docs/TRAINING_GUIDE.md` | 训练使用指南 | ✅ 可用 |

### 目录结构

```
├── logs/                          # 训练日志（TensorBoard格式）
│   ├── ppo_5000ep_20260127_163348/  # 5000轮PPO训练日志
│   ├── ppo_3000ep_20260127_162233/  # 3000轮PPO训练日志
│   ├── ppo_500ep_20260127_160939/   # 500轮PPO训练日志
│   └── sac_5000ep_20260130_145454/  # 5000轮SAC训练日志
├── saved_models/                  # 保存的模型文件
│   ├── ppo_5000ep_20260127_163348   # 5000轮训练PPO模型
│   ├── ppo_3000ep_20260127_162233   # 3000轮训练PPO模型
│   └── sac_5000ep_20260130_145454   # 5000轮训练SAC模型
├── comparison_results/            # 算法对比结果与评估数据
└── docs/                          # 文档与报告
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖：torch, numpy, gymnasium, tensorboard, matplotlib
```

### 2. 训练模型

```bash
# 训练PPO算法（5000轮，深度图像尺寸16）
python train.py --algorithm ppo --episodes 5000 --depth-size 16 --batch-size 10

# 训练SAC算法（实验性）
python train.py --algorithm sac --episodes 3000 --depth-size 16 --batch-size 10

# 运行对比实验（PPO vs SAC）
python train.py --algorithm compare
```

### 3. 评估模型

```bash
# 评估最新训练的PPO模型
python evaluate_ppo_model.py

# 评估结果包含：
# - 成功率、碰撞率、超时率
# - 奖励统计
# - Episode长度统计
# - 目标距离统计
```

### 4. 查看训练日志

```bash
# 启动TensorBoard查看训练曲线
tensorboard --logdir logs/

# 然后在浏览器中打开：http://localhost:6006
```

## 📊 评估指标说明

### 成功率统计
- **成功率**：成功到达目标的比例
- **碰撞率**：与障碍物碰撞的比例
- **超时率**：超出最大步数的比例

### 奖励统计
- **平均累计奖励**：每个episode的平均总奖励
- **奖励标准差**：奖励的波动程度
- **最大/最小奖励**：最佳和最差表现

### Episode统计
- **平均长度**：完成一个episode的平均步数
- **长度标准差**：步数的波动程度

## 🔧 配置文件

`config.py` 提供统一的训练参数配置：

```python
from config import TrainingConfig

# 创建默认配置
config = TrainingConfig()
config.max_episodes = 5000
config.depth_image_size = 16
config.use_depth_sensor = True

# 从命令行参数创建配置
args = parse_arguments()
config.from_args(args)
```

## 💡 使用建议

### 推荐工作流程
1. **快速训练**：使用 `--episodes 500` 进行快速测试训练
2. **完整训练**：使用 `--episodes 5000` 进行完整训练
3. **模型评估**：使用 `evaluate_ppo_model.py` 评估模型性能
4. **结果分析**：使用TensorBoard查看训练曲线

### 性能调优
- **增加训练轮数**：当前模型在5000轮后仍需进一步训练
- **调整学习率**：尝试不同的学习率（默认3e-4）
- **优化奖励函数**：修改 `drone_env.py` 中的奖励函数
- **调整深度图像尺寸**：16x16已优化，可尝试32x32

## ❓ 常见问题

### Q: 如何保存和加载模型？
A: 训练时模型会自动保存到 `saved_models/` 目录。评估脚本会自动加载最新模型。

### Q: 评估结果保存在哪里？
A: 评估结果自动保存为JSON文件，存放在 `comparison_results/` 目录下，文件名包含时间戳，如 `evaluation_results_ppo_20260129_151159.json`。

### Q: 如何可视化训练过程？
A: 使用TensorBoard：`tensorboard --logdir logs/`，然后访问 `http://localhost:6006`。

## 📈 最新模型性能

基于 `evaluation_results_ppo_20260129_151159.json`（50次评估）：
- **成功率**: 0.00% (需进一步训练优化)
- **碰撞率**: 36.00%
- **超时率**: 64.00%
- **平均最终距离目标**: 8.73米

## 🔄 更新日志

### 2026-07-12
- 清理旧版代码：删除12个旧版训练脚本、8个旧版评估/测试脚本
- 清理旧版数据：删除2025年12月的训练日志、实验结果目录和旧模型
- 删除孤立文件 `drone_space_config.py` 和根目录旧模型文件
- 归类整理：5份 `.md` 文档移入 `docs/`，评估结果JSON移入 `comparison_results/`
- 更新README文档，移除已删除文件的引用

### 2026-01-29
- 修复 `evaluate_ppo_model.py` 中的numpy兼容性问题
- 更新README文档，明确可用文件状态
- 完成5000轮PPO训练和评估

### 2025-12-31
- 添加深度传感器支持
- 优化CNN编码器架构
- 实现混合状态空间（深度图像+向量状态）

## 📞 支持与贡献

如需技术支持或希望贡献代码，请：
1. 检查现有评估结果和日志
2. 参考 `docs/future_improvements.md` 中的改进建议
3. 使用最新可用的脚本（标记为✅的文件）

---
*项目状态：实验性开发阶段，PPO算法可用，SAC算法待优化*