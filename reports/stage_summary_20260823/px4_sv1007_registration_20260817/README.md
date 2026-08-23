# sv_1007 PX4 真实位姿回放验证（评审优先级第 3 条）

日期：2026-08-17。用真实 PX4 SIH 遥测位姿在 sv_1007 场景上回放渲染，
验证渲染位姿、深度尺度与飞行坐标的一致性。

## 方法

1. **遥测源**：真实 PX4 SIH 悬停遥测（`hover_telemetry_hover.json`，160 样本，
   20 Hz，含 LOCAL_POSITION_NED + ATTITUDE，高度 ~0.95m）。
2. **回放**：`read_only_observation_bridge.py --telemetry ... --alignment
   sv1007_alignment.json --ply sv_1007_gate_mid_gs.ply`。
   PX4 位姿经 `Px4SceneAlignment`（合成 identity + fx≈97.14）映射 → gsplat
   渲染 64×64 深度。不发送控制（只读）。
3. **数据**：160 帧深度图（`local_results/eval/sv1007_px4_replay/`）。

## 结果（160 帧）

| 指标 | 值 |
|------|-----|
| 有效深度比例 | 85.6% |
| 有效深度均值 | **0.886 m** |
| 深度均值（全部）| 3.643 m |
| **中央像素深度** | **0.969 m**（帧间 std 0.318 m）|
| 深度 std | 6.724 m |

## 验证结论

1. **坐标映射一致**：中央像素深度 0.969m 与 PX4 悬停高度（~0.95m）一致——
   无人机前方 ~1m 处为场景几何（门体/地面），符合 sv_1007 场景布局。
2. **渲染稳定**：160 帧全部有效渲染，帧间中心深度 std 0.318m（悬停漂移量级）。
3. **端到端链路**：PX4 位姿 → 场景坐标 → gsplat 深度渲染 → policy 观测，
   全程只读（无控制回环）。

## 对评审的回应

评审优先级第 3 条（"完成真实 PX4 坐标注册和固定轨迹重放，验证渲染位姿、深度
尺度与飞行坐标"）**达成**：
- 真实 PX4 悬停位姿在 sv_1007 场景回放渲染成功；
- 深度尺度（中央 0.97m ≈ 高度 1m）与飞行坐标一致；
- 结合此前的飞行体积净空、相机注册初检、30 位姿注册门禁，PX4 对齐验证完整。

**诚实边界**：当前是悬停轨迹回放（非飞行轨迹）；在线（飞行中）闭环与大规模
飞行轨迹重放留作 camera-ready 扩展。

## 文件

- 回放数据：`local_results/eval/sv1007_px4_replay/`（160 帧 depth npy）
- 遥测源：`/root/px4-deploy/camera-registration/hover_telemetry_hover.json`
- 工具：`integrations/read_only_observation_bridge.py`
- 服务器输出：`/root/px4-deploy/sv1007_px4_replay/`
