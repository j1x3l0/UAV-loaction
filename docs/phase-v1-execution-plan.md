# Phase V1 实施方案 — V1-Baseline-S1 (gate_mid)

> 生成时间：2026-07-25
> 阶段：Phase 0 已完成 → Phase V1 启动准备
> 基于：MASTER_PLAN.md / PROGRESS.md / research_plan_v2.md
> 目标会议：ICRA 2027 (~2026-09 截稿)

---

## 1. 当前状态（Phase 0 验收基线，2026-07-25 实测）

| 项目 | 实测值 | 环境 |
|------|--------|------|
| 场景 | gate_mid_new_gs.ply，368,965 Gaussians | 4 场景 .ply 已提取 (16.7~23.9 MB) |
| 渲染 Init | 1.2~1.4 s | CPU (torch 2.13.0+cpu, 本机无 GPU) |
| 渲染 Step | 23~25 ms / frame (64×64) | CPU 投影回退 |
| Depth | [0.1, 20.0]，无 NaN | 3 测试位姿验证 |
| 训练管线 | mock 283 fps / 真实GS(CPU) 26 fps | 1ep 冒烟全链路通过 |
| GPU 渲染路径 | 已按 gsplat 1.5.3 API 修复，**未经真 GPU 验证** | 待服务器 benchmark |

关键修复（2026-07-25）：
- `train_visual.py`：补齐 `--renderer/--ply` CLI + `resolve_ply_path()` 三级路径解析（此前文档命令必然失败且会静默回退 mock）
- `gs_renderer.py`：GPU 路径重写 — RGB+ED + alpha 掩码空洞置 max_depth + Gaussian 张量 `__init__` 缓存（原实现每帧重复上传 ~20MB）+ quats 归一化

---

## 2. 目标

| 项目 | 值 |
|------|-----|
| 实验 | V1-Baseline-S1 |
| 场景 | gate_mid (`data/gs_data/ply_exports/gate_mid_new_gs.ply`) |
| 训练量 | 3000 episodes × 8 并行环境 × 256 rollout steps |
| 观测 | depth 64×64×1 + vec 6D |
| 算法 | PPO (clip 0.2, GAE 0.95, γ 0.99, lr 3e-4 线性衰减) |
| **Gate G3** | **SR > 80%**（对照：v1 向量基线 98%） |
| 预计耗时 | ~20 h（GPU，取决于渲染 benchmark） |

---

## 3. 服务器环境准备

### 3.1 依赖清单（锁版本）

```
torch (CUDA 版, 与服务器 CUDA 匹配)
gsplat==1.5.3        # 渲染器代码按此版本 API 编写, 勿装 main 分支
gymnasium
plyfile
scipy
numpy
ninja                # JIT 编译需要
```

### 3.2 gsplat 安装（三选一，2026-07-25 已核实官方渠道）

**方案 A — 国内 PyPI 镜像 + JIT（推荐）**

```bash
pip install torch gymnasium plyfile scipy ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install gsplat==1.5.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
export TORCH_CUDA_ARCH_LIST="8.6"   # A100=8.0, 3090=8.6, 4090=8.9, H100=9.0
python -c "import gsplat; print('ok')"   # 首次 import JIT 编译 ~5-10 min（一次性）
```

前提：`nvcc --version` 与 `torch.version.cuda` 大版本匹配。

**方案 B — 官方预编译 wheel（免编译，组合受限）**

```bash
pip install ninja numpy jaxtyping rich
pip install gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124
# 已验证可用组合: pt24cu124 / pt23cu121 (py3.10, linux+win, gsplat 1.4.0~1.5.3)
# torch 2.5+/cu126+ 无预编译 (2026-07-25 实测 404)
```

**方案 C — Docker**

无官方 gsplat 镜像。基础镜像 `nvcr.io/nvidia/pytorch:24.05-py3`（自带匹配 CUDA+torch+nvcc）→ 容器内 `pip install gsplat==1.5.3`。云平台（AutoDL 等）可直接搜社区 "gsplat/nerfstudio" 镜像。

---

## 4. 启动流程（三段式，禁止跳步）

```bash
cd rlproject-swift-improved

# ① 渲染自检 + benchmark（~1 min）
python envs/gs_renderer.py
# 通过标准: Device: cuda | depth [0.1, 20.0] 无 NaN | 记录 ms/frame
# 若 GPU 路径报错 → 回退检查 torch/gsplat/CUDA 版本匹配，禁止直接上训练

# ② 100ep 冒烟（~1-2 h）
python scripts/train_visual.py --episodes 100 --envs 8 \
    --renderer gsplat --ply data/gs_data/ply_exports/gate_mid_new_gs.ply
# 通过标准: loss 下降、SR > 0%、无 NaN/OOM

# ③ 正式训练（~20 h）
nohup python scripts/train_visual.py --episodes 3000 --envs 8 \
    --renderer gsplat --ply data/gs_data/ply_exports/gate_mid_new_gs.ply \
    > logs/v1_baseline_s1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 5. 风险与回退

| 风险 | 触发条件 | 回退 |
|------|---------|------|
| GPU 渲染 bug 残留 | ①中 depth 异常/报错 | 检查 viewmat 约定；最坏情况用 `render_mode='RGB'` + 单独深度通道调试 |
| 显存不足 | 8 env × 368k Gaussians (~25MB/env) + CNN | 减 `--envs 4`；或共享 Gaussian 张量（需改代码） |
| fps 远低于预期 | benchmark < 100 fps | 确认 TORCH_CUDA_ARCH_LIST 正确重编译；确认无 per-frame H2D |
| SR  plateau < 80% | 冒烟后趋势明显不达标 | 回到 weekly_plan_v2.md 调 reward/超参，不硬跑 3000ep |
| 训练中断 | 服务器断连 | 用 nohup + 日志；模型按 best 保存在 saved_models/ |

---

## 6. 交付物与验收

| 交付物 | 路径 | 验收 |
|--------|------|------|
| 训练日志 | logs/v1_baseline_s1_*.log | 完整 3000ep |
| 最佳模型 | saved_models/visual_ppo_best.pth | eval SR > 80% (Gate G3) |
| 渲染 benchmark 数据 | PROGRESS.md 问题日志追加 | fps / 显存 / depth 范围 |
| PROGRESS.md 更新 | Step 1.1 表格 | 开始/结束时间、最终 SR |

完成后进入 Phase V2（衰减曲线，6 条退化轴 × gate_mid）。

---

*生成：2026-07-25 | 维护：随 Phase V1 进展更新*
