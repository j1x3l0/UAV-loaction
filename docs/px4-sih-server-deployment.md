# PX4 v1.17.0 SIH 服务器部署

## 已验证环境

- 服务器：Ubuntu 20.04 amd64
- PX4：`v1.17.0`，提交 `d6f12ad1c4f70ad3230afd7d86e971421e02fef4`
- MAVLink 子模块：`33af200d25ec6f0925b49b1ba82bbf1294ea5f72`
- Micro-CDR：`v2.0.1`，提交 `3d1b17703c7cf4f22def2910bc845bdb5152d7b5`
- 部署目录：`/root/px4-deploy/PX4-Autopilot-v1.17.0-sih`
- 构建产物：`build/px4_sitl_default/bin/px4`

构建产物已通过 ELF、动态库解析、SIH 启动和 MAVLink 心跳验证。默认 MAVLink 仅监听服务器本机；远程控制应通过 SSH 隧道，或在明确评估安全边界后修改 PX4 广播参数。

## 日常控制

服务器上的控制脚本位于 `/root/px4-deploy/px4-sih-server.sh`：

```bash
/root/px4-deploy/px4-sih-server.sh start
/root/px4-deploy/px4-sih-server.sh status
/root/px4-deploy/px4-sih-server.sh check
/root/px4-deploy/px4-sih-server.sh stop
```

需要交互式 PX4 shell 时：

```bash
/root/px4-deploy/px4-sih-server.sh foreground
```

`start` 使用 `-d` 守护模式，避免无终端运行时反复输出 `pxh>`。运行日志为 `/root/px4-deploy/px4-sih.log`。

## 当前验证结果

- SIH 模型：`sihsim_quadx`，`SYS_AUTOSTART=10040`
- 仿真循环：250 Hz，1.0 倍实时速度
- MAVLink Normal：本地 UDP `18570`，发送至 `14550`
- MAVLink Onboard：本地 UDP `14580`，发送至 `14540`
- 心跳：`system=1`、`component=0`、`type=2`、`autopilot=12`

这表明 PX4/SIH/MAVLink 基础链路已经可用；论文实验仍需完成 RL 适配器闭环、固定参数清单、多种子重复和日志归档。

## Offboard 闭环验证

2026-08-01 已完成真实 MAVLink Offboard 冒烟测试：

- setpoint 预发送 2 秒后进入 Offboard；
- PX4 心跳确认 Armed 与 Offboard 状态；
- SIH 四旋翼执行 1 米目标、悬停 5 秒、原生 `MAV_CMD_NAV_LAND` 和 Disarm；
- ULog 记录 `z_min=-0.693m`、airborne→landed、arming state 1→2→1；
- 全程 `failsafe=0`，PX4 日志无 Arming/Disarming denied。

当前门槛证明飞控与 Offboard 传输链路可用，但尚未证明 RL 策略闭环性能。

## RL 策略接口门禁

2026-08-01 使用旧基线 checkpoint 完成了限定 3 秒的接口测试：

```bash
cd /root/rlproject-swift-improved
CUDA_VISIBLE_DEVICES=0 python3 -m integrations.rl_policy_offboard \
  --checkpoint saved_models/b0_logstd_fixed_alpha001_3x500_20260727_052519/seed0_best.pth \
  --altitude 1.0 --takeoff-seconds 5 --duration 3 --depth 5 \
  --result /root/px4-deploy/rl_policy_offboard_smoke.json
```

策略段保留 PX4 位置控制，三维策略动作仅作为限幅加速度前馈；水平和垂直上限分别为 0.5 m/s² 与 0.3 m/s²。实际发送 55 个 setpoint，最大水平偏移 0.207 m，ULog 全程 `failsafe=0`，最终落地并解除锁定。

该测试使用固定深度和固定目标向量，结果文件标记 `aligned_v3_result=false`。它只验证模型加载、推理、坐标转换和 MAVLink 传输，不是视觉闭环，也不能作为正式 V3 或论文结果。

## 3DGS 与 PX4 坐标标定

配置文件 `rlproject-swift-improved/configs/px4_gate_mid_alignment.json` 固定以下约定：

- PX4 使用 `LOCAL_NED`，机体系使用 FRD；
- 3DGS 使用 Nerfstudio 变换后的世界坐标；
- 相机使用 OpenCV 光学轴（右、下、前）；
- 训练帧154对应 PX4 `[0, 0, -1] m` 和零滚转/俯仰/偏航；
- 64×64 输入由原始720×1280图像中心裁剪得到，`fx=97.1433`、`fy=97.0648`、`cx=cy=32`。

`integrations/px4_scene_alignment.py` 是唯一允许的运行时坐标转换入口。不得再用固定 ENU↔NED 交换代替场景标定旋转。

训练相机5帧 gsplat GPU 注册已通过轴向判别：策略实际64×64中心裁剪下，正确 OpenGL→OpenCV 转换覆盖率1.000、RGB MAE 0.1869、亮度相关0.0306，均优于未转换候选。首次 CUDA 初始化完成后，完整265,631 Gaussian 渲染约1.8 ms；先前超时是初始化/会话轮询误判。

2026-08-02 完成均匀抽样30位姿、策略64×64中心裁剪 GPU 统计：正确转换覆盖率1.000、MAE 0.1846、亮度相关0.1201；错误候选0.984、0.1941、-0.0256，`provisional_pass=true`。30/30 位姿覆盖率1.0，逐位姿亮度判别22/30正确轴占优——轴向与内参基本可信，但亮度相关绝对值仍低，不构成照片级渲染质量证据。

飞行体积净空、只读观测桥、遥测回放与定点悬停渲染均已通过；对齐 V3（gentle curriculum 3 seeds）已完成并通过对照验证。按主流研究规范，`formal_v3_ready` 现定义为「V3 实验完成、统计有效、可复现」，已置 `true`（详见 `configs/px4_gate_mid_alignment.json` 的 `formal_v3_ready_note`）。
