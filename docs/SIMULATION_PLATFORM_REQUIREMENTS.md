# UAV-loaction 仿真平台需求与实施方案

日期：2026-07-30（2026-07-31 修订：确认联想笔记本 8 GB 显存，评估为不需要，计划简化为 MacBook 开发/观察机 + 服务器）

## 1. 目的

本文明确 AirSim、Isaac Sim、Gazebo、Pegasus、PX4、3DGS、Sionna 和计算硬件
在本项目中的职责，给出当前服务器与笔记本的部署边界，并定义从轻量算法验证
过渡到高保真飞控验证的实施顺序和验收标准。

核心目标不是用一个大型仿真器替换全部现有代码，而是建立分层验证体系：

1. 保留当前真实 gsplat 轻量环境，用于快速训练、奖励调试和大规模消融；
2. 使用 PX4 SITL + Gazebo 验证飞控接口、动力学、传感器延迟和碰撞；
3. 使用 Isaac Sim + Pegasus + PX4 完成高保真视觉、动力学和跨场景验证；
4. 只在研究通信感知导航时引入 Sionna。

## 2. 组件职责

| 组件 | 本质 | 在项目中的作用 | 不负责的内容 |
|---|---|---|---|
| 当前 gsplat 环境 | 轻量导航环境 | 真实 3DGS 深度、快速 PPO 训练与退化评估 | 完整飞控、电机和气动 |
| Gazebo | 机器人三维仿真器 | 刚体动力学、碰撞、深度相机、IMU、风和 Headless 批量实验 | 飞控逻辑 |
| AirSim | Unreal/Unity 无人机仿真器 | 无人机、相机、天气及 PX4 接口 | 长期维护稳定性 |
| Isaac Sim | NVIDIA 通用机器人仿真器 | PhysX、RTX 传感器、USD 场景、ROS 2、合成数据 | 开箱即用的完整无人机栈 |
| Pegasus | Isaac Sim 无人机扩展 | 多旋翼动力学、无人机 API 和 PX4 集成 | 场景资产与策略训练算法 |
| PX4 | 飞控软件 | 姿态、速度、位置控制，状态估计和失效保护 | 世界、相机和碰撞仿真 |
| 3DGS/NuRec | 场景表示与重建工具 | 真实感外观、深度观测和真实场景资产 | 默认可靠的物理碰撞网格 |
| Sionna | 无线传播仿真 | RSS、SINR、射线传播和通信遮挡 | 无人机动力学与导航物理 |
| DGX Spark/服务器/笔记本 | 计算硬件 | 承载上述软件 | 不提供仿真功能本身 |

推荐的数据与控制链路为：

```text
PPO 高层策略
  -> 速度/航点命令
PX4 SITL
  -> 姿态与执行器控制
Gazebo 或 Isaac Sim + Pegasus
  -> 动力学、碰撞、相机、深度、IMU
Mesh/SDF + 3DGS/NuRec
  -> 物理几何 + 真实感外观
```

## 3. 现有硬件条件

### 3.1 训练服务器（已实测）

- 架构：x86_64；
- 系统：Ubuntu 20.04.3 LTS；
- GPU：2 × NVIDIA GeForce RTX 3090，每张 24 GB；
- 驱动：535.98，CUDA capability 显示 12.2；
- 内存：240 GiB；
- 磁盘：约 3 TB 可用；
- 当前未安装 Docker；
- 适合 Headless 训练、渲染和批量评估。

硬件算力足够，主要限制是 Ubuntu 20.04 和旧驱动。最新 Isaac Sim 的官方
x86_64 要求为 Ubuntu 22.04/24.04，并给出更新的测试驱动和 RTX 4080 级最低
GPU 配置。因此，不能把“3090 可以尝试运行旧版/适配版”表述为“当前系统满足
最新 Isaac Sim 官方要求”。

### 3.2 联想拯救者 Y7000P（评估结论：不需要）

- GPU：RTX 4060 Laptop，8 GB 显存（`nvidia-smi` 确认）。

评估结论：本项目不需要第二台电脑。所有角色均可由现有开发机（MacBook）与
服务器覆盖：

- 开发、绘图、可视化：MacBook；
- QGroundControl 监控：MacBook 作为客户端经网络连接服务器 SITL 的 MAVLink；
- Gazebo 渲染观察：服务器 headless 渲染 + VNC/X 转发，MacBook 查看；
- Isaac Sim：8 GB 显存低于最低 RTX 4080（16 GB），只能在服务器运行。

联想笔记本不承担任何不可替代职责，也不需维护独立的 PX4+Gazebo 环境，故不
列入计划。若未来出现 Windows 目标部署或需要本地 GUI 调试，再重新评估。

## 4. 平台选择

### 4.1 AirSim

优点：

- 无人机、RGB/Depth、天气和 PX4 接口较完整；
- 适合复现已有 AirSim 工作或快速演示。

不足：

- 微软原版技术栈已停止积极演进，正式版本和 Unreal 依赖偏旧；
- 与本项目 3DGS/NuRec、USD 和长期 ROS 2 路线结合需要较多自定义工作；
- 新建主线会增加版本和依赖维护负担。

结论：可以运行，但不作为本项目主平台。

### 4.2 PX4 SITL + Gazebo

优点：

- 当前服务器最容易落地；
- PX4 官方支持 `x500` 和带前向深度相机的 `x500_depth`；
- 支持 Headless、碰撞、IMU、深度、风和 Offboard/MAVSDK/ROS 接口；
- 适合验证从 PPO 高层动作到真实飞控的完整控制链路。

限制：

- Ubuntu 20.04 原生路径应使用兼容版本或 Gazebo Classic；
- 新版 Gazebo Harmonic 的预编译包面向 Ubuntu 22.04/24.04；
- 视觉真实感与 3DGS 原生结合不如 Isaac/NuRec 路线。

结论：作为现有服务器的第一落地平台和中间验证层。

### 4.3 Isaac Sim + Pegasus + PX4

优点：

- 提供 PhysX、RTX 相机/深度、IMU、接触、USD、ROS 2、合成数据和域随机化；
- Pegasus 补充多旋翼动力学和 PX4 集成；
- 更适合后续接入 NuRec、USD、3DGS 外观与 Mesh/SDF 碰撞；
- 适合作为论文中的高保真主验证平台。

限制：

- 当前服务器系统与驱动不满足最新官方组合；
- 安装体积、显存占用和集成成本明显高于 Gazebo；
- 联想笔记本（8 GB）已评估为不需要，不承担任何仿真职责；Isaac Sim 只在服务器
  运行；
- 3DGS 外观不能自动等价为可靠碰撞几何，仍需 Mesh、SDF、占据图或显式
  Gaussian 碰撞校准。

结论：作为最终高保真平台，但在独立环境验证成功前不阻塞当前实验。

## 5. 最佳总体方案

项目采用三级验证，而不是在 AirSim、Gazebo 和 Isaac Sim 中三选一：

| 层级 | 平台 | 主要用途 | 结果定位 |
|---|---|---|---|
| L1 | 当前 gsplat 轻量环境 | 奖励、网络、退化、消融和候选模型筛选 | 算法证据 |
| L2 | PX4 SITL + Gazebo | 飞控、动力学、传感器、延迟和碰撞验证 | 控制链路证据 |
| L3 | Isaac Sim + Pegasus + PX4 | 高保真场景、RTX 传感器、跨场景和 sim-to-sim | 高保真泛化证据 |

AirSim不进入主线；Sionna列为可选扩展。只有当论文问题明确包含无线信道、
通信中断、基站覆盖或通信感知路径规划时，才增加 Sionna。

## 6. 设备分工

### 开发/观察机（MacBook）

- VS Code/Codex 开发；
- QGroundControl 客户端（经网络连接服务器 SITL 的 MAVLink）；
- 绘图、场景/深度可视化；
- MAVSDK/ROS 2 客户端调试；
- VNC/X 转发查看服务器 Gazebo 渲染；
- 20–100 episodes 结果查看与人工抽查；
- 不部署自己的 PX4/Gazebo/Isaac。

### 双 3090 服务器

- 当前 3DGS 环境的正式训练；
- 3 seeds、多退化档和跨场景批量评估；
- Gazebo Headless 正式实验；
- 完成系统兼容改造后的 Isaac Sim Headless；
- 模型、日志和大型资产本地保存，不上传 GitHub。

## 7. 实施顺序

### 阶段 A：不改服务器系统

1. 完成当前 hybrid-geodesic 门控，不重复启动已有训练；
2. 固化当前模型、数据、seed 和轻量环境结果；
3. 安装 PX4 SITL 与兼容的 Gazebo；
4. 启动单架 `x500_depth`，完成起飞、悬停、速度控制和降落；
5. 接入 PPO 动作，建立动作限幅、控制频率、坐标系和时间戳转换；
6. 运行 20 episodes Headless 冒烟，再运行 100 episodes 单 seed 门控。

### 阶段 B：服务器 Isaac 原型

1. 在隔离的 Ubuntu 22.04 环境或容器中部署，避免污染当前训练环境；
2. 选择与 Pegasus 明确兼容的 Isaac Sim 版本；
3. 只启用一架无人机、一个低分辨率深度相机和 IMU；
4. 验证 Pegasus 多旋翼、PX4 SITL 和 ROS 2/MAVSDK 通路；
5. 运行固定场景 20–50 episodes，不进行正式训练；
6. 连续两次出现相同安装或运行失败则停止该路径，记录原因，不反复重试。

### 阶段 C：服务器 Isaac 正式环境

1. 等现有训练完成后安排维护窗口；
2. 优先通过隔离的 Ubuntu 22.04 环境或容器部署，避免污染当前训练环境；
3. 升级驱动前保存当前环境清单并确认所有训练已停止；
4. 先运行 Isaac Compatibility Checker 和官方 Headless 示例；
5. 再安装 Pegasus、PX4 和项目桥接层；
6. 通过小门控后才导入复杂 3DGS/NuRec 资产；
7. 正式运行多 seed 和跨场景评估。

## 8. 3DGS 场景资产要求

每个正式场景必须同时具备：

- 标准 3DGS PLY 或 NuRec/3DGRUT 视觉资产；
- Mesh、SDF、占据图或经过验证的 Gaussian 碰撞几何；
- 视觉坐标到物理世界坐标的变换；
- 米制尺度、重力方向和相机坐标约定；
- 场景边界、可飞空间和无人机安全半径；
- 可复现的起点/终点采样协议；
- 深度图与物理射线/网格深度的对齐测试；
- 资产来源、许可证和版本记录。

只替换 PLY 背景而不替换碰撞几何，不能称为真正的跨场景导航泛化。

## 9. 首轮验收标准

### PX4 + Gazebo 接口门控

- 起飞、悬停、速度控制、降落均可复现；
- 坐标系、时间同步和动作限幅测试通过；
- 深度、IMU 和碰撞事件能够被策略环境读取；
- 连续 100 episodes 无进程崩溃；
- timeout 不高于 10%；
- 输出 SR、collision、timeout、路径长度、控制延迟和仿真实时率。

### Isaac + Pegasus 冒烟门控

- Headless 启动成功且无持续显存溢出；
- PX4 解锁、起飞和 Offboard 控制成功；
- 深度图、IMU、位姿与碰撞时间戳一致；
- 单场景至少 20 episodes 可重复运行；
- 与 Gazebo 使用相同模型、起终点 seed 和动作接口；
- 记录 sim-to-sim 的成功率与碰撞率变化。

### 正式高保真评估

- 至少 3 个训练 seed；
- 至少 3 个几何与外观均不同的场景；
- clean 与传感器噪声、延迟、风场、深度尺度等退化；
- 每个主要条件至少 100 episodes，核心结论建议 200 episodes；
- 报告 Wilson 95% CI、collision、timeout 和失败类型；
- 轻量环境、Gazebo、Isaac 三层结果不得混称为同一保真度证据。

## 10. 风险与停止条件

- 不在训练运行期间升级服务器驱动或系统；
- 不在主环境直接覆盖 CUDA、PyTorch 或 gsplat 依赖；
- Isaac/Gazebo/PX4 使用独立目录、容器或版本锁定环境；
- 同类安装或运行操作连续失败两次即停止并记录；
- GPU、VPN 或 SSH 状态不明时不猜测、不重复启动；
- 模型权重、训练日志、PID、大型 3DGS/Replica/NuRec 资产不提交 GitHub；
- 未完成视觉—物理几何对齐前，不宣称真实场景避障或跨场景泛化。

## 11. 决策结论

1. **立即实施：** 当前服务器部署 PX4 SITL + Gazebo Headless；
2. **并行开发：** 开发/观察在现有 MacBook（代码、绘图、QGC 客户端连接服务器
   SITL）；仿真与训练全部在服务器；联想笔记本评估为不需要；
3. **最终主验证：** 服务器兼容升级后运行 Isaac Sim + Pegasus + PX4；
4. **不选主线：** 原版 AirSim；
5. **条件性扩展：** Sionna，仅用于通信感知研究；
6. **保留现有投入：** 当前真实 gsplat 环境继续作为高吞吐算法筛选层。

## 12. 参考资料

- [PX4 仿真总览](https://docs.px4.io/main/en/simulation/)
- [PX4 Gazebo 仿真](https://docs.px4.io/v1.17/en/sim_gazebo_gz/)
- [PX4 预构建 SITL 与容器](https://docs.px4.io/main/en/simulation/px4_sitl_prebuilt_packages)
- [Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Sim 容器安装](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [Pegasus Simulator 安装](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html)
- [Microsoft AirSim](https://github.com/microsoft/AirSim)
- [Sionna RT](https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html)
- [NVIDIA NuRec for Robotics](https://docs.nvidia.com/nurec/robotics/index.html)
