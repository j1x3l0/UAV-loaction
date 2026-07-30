# PX4 + Gazebo 接口验证计划

日期：2026-07-30

## 当前结论

服务器为Ubuntu 20.04 overlay环境，目前没有Docker、Gazebo、PX4、ROS 2或
colcon。直接安装最新版Gazebo/ROS 2会污染现有gsplat训练环境，因此第一步只
提交传输无关的PX4适配层，不立即修改系统依赖。

## 控制接口

当前策略输出三维归一化加速度，环境坐标按ENU解释；PX4
`TrajectorySetpoint`使用NED。适配层完成：

- ENU `[east, north, up]` 到NED `[north, east, down]` 转换；
- 水平与垂直加速度分别限幅；
- 未控制的position、velocity、yaw字段设为NaN；
- 20 Hz Offboard心跳节拍和0.25秒本地陈旧检测；
- 非有限动作拒绝。

正式接入优先使用ROS 2 `TrajectorySetpoint`。PX4要求Offboard存活信号持续
高于2 Hz，并应在切换Offboard和解锁前先发送稳定setpoint流。

## 分阶段门控

1. **接口单元测试：** 坐标、限幅、NaN字段、心跳和超时全部通过；
2. **PX4 SIH：** 无Gazebo，验证连接、解锁、起飞、Offboard、悬停、失联降级；
3. **Gazebo x500_depth：** Headless运行，读取深度、IMU、位姿和碰撞；
4. **策略桥接：** 先固定动作，再接PPO；频率20 Hz，超时立即输出零/保持并退出
   Offboard；
5. **20 episodes门控：** 无崩溃、timeout不高于10%，记录SR、碰撞、控制延迟
   和实时率；
6. 通过后才扩大到100 episodes并导入自定义场景。

## 部署决策

- 首选：在独立Ubuntu 22.04宿主机或具备Docker能力的环境运行新版PX4+
  Gazebo；
- 当前服务器：先尝试PX4 SIH零外部仿真依赖；如果需要Gazebo，使用独立容器
  或维护窗口，避免覆盖CUDA/gsplat环境；
- 笔记本：适合Gazebo GUI、QGroundControl和人工观察；
- Isaac Sim/Pegasus在PX4+Gazebo接口门控通过后再启动。

## 安全边界

- SITL验证通过前不得接真机；
- 不直接控制电机；
- 必须配置Offboard失联保护；
- 所有坐标系和单位写入日志；
- 不把PX4接口通过等同于视觉导航成功；
- 同类部署失败两次即停止，不反复修改服务器。

## 参考

- [PX4 Offboard Mode](https://docs.px4.io/main/en/flight_modes/offboard)
- [PX4 TrajectorySetpoint](https://docs.px4.io/main/en/msg_docs/TrajectorySetpoint)
- [PX4 Simulation](https://docs.px4.io/main/en/simulation/)
- [PX4 Gazebo](https://docs.px4.io/v1.17/en/sim_gazebo_gz/)
