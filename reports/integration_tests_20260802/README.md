# 集成测试套件运行报告

日期：2026-08-02。范围：`rlproject-swift-improved/` 全部测试（12 个文件、83 个用例）。

## 环境

| 项 | 值 |
|----|----|
| 平台 | Windows 10 Pro（x86_64） |
| Python | 3.13.0 |
| 依赖 | `requirements.txt` 全量安装（numpy 2.5.1 / scipy 1.18.0 / gymnasium 1.3.0 / torch 2.13.0+cpu / pymavlink） |
| pytest | 最新版（venv 内安装） |
| 硬件 | 纯 CPU（无 GPU、无 gsplat） |

> 说明：测试全部无需 3DGS 数据文件或 GPU。`test_gs_renderer_camera.py`
> 通过 `GSplatRenderer.__new__` 绕过 PLY 加载，仅验证相机内参与坐标系校验逻辑。

## 命令与结果

```bash
python -m pytest rlproject-swift-improved/ -q
```

**83 passed in 6.64s — 0 failed / 0 error / 0 skipped**（两次独立运行复现一致：6.21s / 5.75s / 6.64s）。

## 分文件明细

| 测试文件 | 用例数 | 覆盖内容 | 结果 |
|----------|-------:|----------|:----:|
| `core/test_adaptive_entropy.py` | 3 | 自适应熵系数方向（state/visual agent） | ✅ |
| `core/test_visual_agent.py` | 12 | CNN 编码器形状、Actor-Critic 输出、GAE、PPO store/update、save/load、LR 调度、空记忆/单环境批量 | ✅ |
| `envs/test_gs_renderer_camera.py` | 5 | device 校验、内参保留、c2w 右旋/正交归一校验 | ✅ |
| `envs/test_mock_collision_alignment.py` | 1 | 渲染球体与碰撞球体几何对齐 | ✅ |
| `envs/test_visual_env.py` | 21 | 渲染形状/遮挡/位置区分、降采样、噪声、reset/step/边界/碰撞/到达/最大步数、确定性、退化与消融配置、路点观测、场景几何采样/测地距离、运动相机四元数、深度尺度采样、奖励组件 | ✅ |
| `integrations/test_mavlink_offboard.py` | 9 | MAVLink type_mask、offboard 模式、心跳解码、local_position 解码、arm/land 命令 | ✅ |
| `integrations/test_px4_offboard.py` | 4 | ENU↔NED、动作限幅/NaN 字段、非法动作拒绝、心跳保护 | ✅ |
| `integrations/test_px4_scene_alignment.py` | 5 | 锚点重建相机位姿、右旋保持、向量往返、yaw→NED 映射 | ✅ |
| `integrations/test_read_only_observation_bridge.py` | 10 | 观测构造、姿态→c2w、非有限位姿拒绝、target 解析、遥测加载校验 | ✅ |
| `integrations/test_rl_policy_offboard.py` | 6 | 时长上界、悬停包络/高度/水平逃逸拒绝、NED→ENU、观测形状 | ✅ |
| `scripts/test_train_visual_helpers.py` | 5 | curriculum 边界、checkpoint 路径、鲁棒分数、恢复采样分布 | ✅ |
| `scripts/test_validate_camera_registration.py` | 2 | 全帧/策略中心裁剪内参换算 | ✅ |

## 结论

- **交接文档所列 4 个集成测试**（`test_mavlink_offboard` / `test_px4_scene_alignment` / `test_rl_policy_offboard` / `test_px4_offboard`）全部通过。
- 全套 83 用例在干净环境（全新 venv + `requirements.txt`）直接通过，**无需修复任何失败项**——同时交叉验证了依赖声明完整且版本兼容。
- 测试覆盖的 PX4 链路（offboard 协议编码、坐标映射、心跳/限幅防护、观测桥）均为纯逻辑单元，不触碰真实 PX4 或 3DGS 数据；真实 SIH 联调另有 `px4_policy_interface_smoke_20260801` / `px4_hover_observation_test_20260802` 等报告。
