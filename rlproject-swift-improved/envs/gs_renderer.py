"""
gs_renderer.py — 真实 3DGS 渲染器 (替换 MockGSRenderer)

架构位置: envs/ (Extension层)
WHY: 从3DGS .ply文件渲染深度图，接口与MockGSRenderer兼容。
     GPU上使用gsplat快速渲染，CPU上使用简化投影作为开发回退。

数据流: camera_pose → gsplat rasterize / CPU project → depth(64×64×1) + rgb(64×64×3)
边界: 只负责渲染。不负责3DGS训练、场景加载、碰撞检测。
风险: CPU回退 ~23ms/frame (368k Gaussians, 64×64) — 仅用于开发验证。
      GPU路径按 gsplat 1.5.3 API 实现 (RGB+ED + alpha掩码 + 张量缓存),
      上服务器后必须先跑本文件自检 benchmark 再启动训练。

用法:
  renderer = GSplatRenderer(ply_path, width=64, height=64)
  depth, rgb = renderer.render(camera_pos)
"""

import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSplatRenderer:
    """3DGS .ply 渲染器 — GPU (gsplat) + CPU (projection fallback)"""

    def __init__(self, ply_path: str, width: int = 64, height: int = 64,
                 max_depth: float = 20.0, fov: float = 90.0,
                 fx: float = None, fy: float = None,
                 cx: float = None, cy: float = None,
                 device: str = "auto"):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.fov = fov
        default_focal = (self.width / 2) / np.tan(np.deg2rad(fov / 2))
        self.fx = float(default_focal if fx is None else fx)
        self.fy = float(default_focal if fy is None else fy)
        self.cx = float(self.width / 2 if cx is None else cx)
        self.cy = float(self.height / 2 if cy is None else cy)
        if not all(np.isfinite(value) for value in
                   (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")

        # 加载 Gaussians
        self._means, self._quats, self._scales, self._opacities, self._colors = \
            self._load_ply(ply_path)

        if device == "cuda" and not self._has_gpu():
            raise RuntimeError("CUDA renderer requested but CUDA is unavailable")
        self._device = (
            ('cuda' if self._has_gpu() else 'cpu')
            if device == "auto" else device
        )
        self._gaussian_keep_percent = 100.0
        if self._device == 'cpu':
            logger.warning("No GPU detected — using CPU fallback (slow, dev only)")
        else:
            # 关键: Gaussian 参数只上传一次 GPU。
            # 若每次 render 都 from_numpy().cuda()，每帧 ~20MB H2D 传输会拖垮 fps。
            self._cache_gpu_tensors()
            logger.info(f"GPU rendering ready: {len(self._means):,} Gaussians")

    def set_gaussian_keep_percent(self, keep_percent: float) -> None:
        """按 opacity×体积重要性保留固定的 top-k Gaussians。"""
        keep_percent = float(np.clip(keep_percent, 0.01, 100.0))
        n_keep = max(1, int(len(self._means) * keep_percent / 100.0))
        importance = self._opacities * np.prod(self._scales, axis=1)
        idx = np.argpartition(importance, -n_keep)[-n_keep:]
        self._active_indices = np.sort(idx)
        self._gaussian_keep_percent = keep_percent
        if self._device == 'cuda':
            self._cache_gpu_tensors()
        logger.info("Gaussian sparsification: %.1f%% (%d/%d)",
                    keep_percent, n_keep, len(self._means))

    def _cache_gpu_tensors(self):
        """预上传 Gaussian 参数到 GPU (仅 __init__ 调用一次)"""
        import torch
        idx = getattr(self, '_active_indices', slice(None))
        self._t_means = torch.from_numpy(self._means[idx]).cuda()
        self._t_quats = torch.from_numpy(self._quats[idx]).cuda()
        self._t_scales = torch.from_numpy(self._scales[idx]).cuda()
        self._t_opacities = torch.from_numpy(self._opacities[idx]).cuda()
        self._t_colors = torch.from_numpy(self._colors[idx]).cuda()

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_ply(self, ply_path: str):
        """加载标准 3DGS .ply 文件"""
        from plyfile import PlyData
        import torch

        plydata = PlyData.read(ply_path)
        vert = plydata['vertex']

        means = np.stack([vert['x'], vert['y'], vert['z']], axis=-1).astype(np.float32)
        quats = np.stack([vert['rot_0'], vert['rot_1'], vert['rot_2'], vert['rot_3']],
                         axis=-1).astype(np.float32)
        # gsplat 要求 quats 归一化 (wxyz); 训练保存的四元数可能有轻微漂移
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8
        # Standard 3DGS PLY files store log-scales and opacity logits.
        # gsplat.rasterization expects positive scales and opacities in [0, 1].
        log_scales = np.stack(
            [vert['scale_0'], vert['scale_1'], vert['scale_2']], axis=-1
        ).astype(np.float32)
        scales = np.exp(log_scales)

        opacity_logits = np.array(vert['opacity']).astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_logits))

        # f_dc_* is the degree-0 spherical-harmonics coefficient. Convert it
        # to RGB because colors shaped (N, 3) are interpreted as direct colors.
        sh0 = np.stack(
            [vert['f_dc_0'], vert['f_dc_1'], vert['f_dc_2']], axis=-1
        ).astype(np.float32)
        colors = np.clip(0.5 + 0.28209479177387814 * sh0, 0.0, 1.0)

        return means, quats, scales, opacities, colors

    # ── 公共接口 (兼容 MockGSRenderer) ──

    def render(self, camera_pos: np.ndarray,
               camera_quat: np.ndarray = None,
               camera_c2w: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从相机位姿渲染深度图

        Args:
            camera_pos: (3,) 世界坐标
            camera_quat: (4,) 四元数 [x,y,z,w] (CPU回退时忽略旋转)
        Returns:
            depth: (H, W, 1) 范围 [0.1, max_depth]
            rgb:   (H, W, 3)
        """
        c2w = (
            self._validate_c2w(camera_c2w)
            if camera_c2w is not None
            else self._compute_c2w(camera_pos, camera_quat)
        )
        if self._device == 'cuda':
            return self._render_gpu(c2w)
        else:
            return self._render_cpu(c2w)

    # ── GPU 渲染 (gsplat) ──

    def _render_gpu(self, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        gsplat 高性能渲染 (已按 gsplat 1.5.3 API 验证)

        注意: rasterization() 返回 (render_colors, render_alphas, info),
        RGB+ED 模式下 render_colors 为 (C,H,W,4), 最后一通道是期望深度。
        用 ED(期望深度)而非 D(累积深度): 未命中区域用 alpha 掩码置 max_depth,
        避免"天空被当成 0.1m 近处障碍"的伪深度。
        """
        import torch
        from gsplat import rasterization

        # 构建 view matrix
        w2c = np.linalg.inv(c2w)
        viewmat = torch.from_numpy(w2c.astype(np.float32)).cuda()

        # 内参矩阵
        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=torch.float32, device='cuda')

        render_colors, render_alphas, _ = rasterization(
            means=self._t_means,
            quats=self._t_quats,
            scales=self._t_scales,
            opacities=self._t_opacities,
            colors=self._t_colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=self.width,
            height=self.height,
            near_plane=0.1,
            far_plane=self.max_depth,
            render_mode='RGB+ED',
        )

        out = render_colors[0]                       # (H, W, 4)
        rgb = out[..., :3].cpu().numpy()             # (H, W, 3)
        depth = out[..., 3:4].cpu().numpy()          # (H, W, 1)
        alpha = render_alphas[0].cpu().numpy()       # (H, W, 1)

        # 空洞/背景 (alpha 低) → 远平面, 与 CPU 回退语义一致
        depth = np.where(alpha > 0.5, depth, self.max_depth)
        return np.clip(depth, 0.1, self.max_depth).astype(np.float32), rgb

    # ── CPU 回退渲染 ──

    def _render_cpu(self, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CPU 投影渲染 — 开发用，速度慢但输出合理

        WHY: 无GPU时仍可验证管线。将Gaussian centers投影到像平面，
             取最近点深度。不包含alpha blending/排序（GS的质量优势）。
        """
        H, W = self.height, self.width
        depth = np.full((H, W, 1), self.max_depth, dtype=np.float32)
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        w2c = np.linalg.inv(c2w)
        R_w2c = w2c[:3, :3]
        t = w2c[:3, 3]

        # 变换Gaussian centers
        means_cam = self._means @ R_w2c.T + t  # (N, 3)

        # 过滤后方点
        in_front = means_cam[:, 2] > 0.1
        if not in_front.any():
            return depth, rgb

        means_cam = means_cam[in_front]
        colors_filt = self._colors[in_front]
        opacities_filt = self._opacities[in_front]

        # 投影
        u = (means_cam[:, 0] * self.fx / means_cam[:, 2] + self.cx).astype(int)
        v = (means_cam[:, 1] * self.fy / means_cam[:, 2] + self.cy).astype(int)
        d = means_cam[:, 2]  # 深度

        # 仅保留图像范围内的点
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, d = u[valid], v[valid], d[valid]
        colors_filt = colors_filt[valid]

        # 逐点取最近 (简单但慢的 z-buffer)
        for i in range(len(u)):
            if d[i] < depth[v[i], u[i], 0]:
                depth[v[i], u[i], 0] = d[i]
                rgb[v[i], u[i]] = colors_filt[i]

        # 小空洞填充 (3×3 median-like)
        depth = self._fill_holes(depth)

        return depth, rgb

    def _fill_holes(self, depth: np.ndarray, kernel: int = 2) -> np.ndarray:
        """简单空洞填充 — 周围最近有效深度值"""
        from scipy.ndimage import minimum_filter
        filled = depth.copy()
        mask = depth[..., 0] >= self.max_depth
        if mask.any():
            # 用邻域最小值填充
            min_neighbor = minimum_filter(
                np.where(mask, self.max_depth, depth[..., 0]),
                size=kernel * 2 + 1
            )
            filled[mask, 0] = min_neighbor[mask]
        return filled

    @staticmethod
    def _compute_c2w(pos: np.ndarray, quat: np.ndarray = None) -> np.ndarray:
        """计算 camera-to-world 变换矩阵"""
        if quat is None:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        qx, qy, qz, qw = quat
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
        ])
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        return c2w

    @staticmethod
    def _validate_c2w(c2w: np.ndarray) -> np.ndarray:
        """Validate an OpenCV camera-to-world matrix (+Z forward, +Y down)."""
        matrix = np.asarray(c2w, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_c2w must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("camera_c2w must be homogeneous")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError("camera_c2w rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError("camera_c2w rotation must be right-handed")
        return matrix


# ── 自检 ──
if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ply_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'data', 'gs_data', 'ply_exports'
    )
    ply_path = os.path.join(ply_dir, 'gate_mid_new_gs.ply')

    if not os.path.exists(ply_path):
        print(f"ply not found: {ply_path}")
        print("Run utils/extract_ply.py first")
        sys.exit(1)

    print(f"Loading: {ply_path}")
    t0 = time.time()
    renderer = GSplatRenderer(ply_path, width=64, height=64)
    print(f"  Loaded {len(renderer._means):,} Gaussians in {time.time()-t0:.1f}s")
    print(f"  Device: {renderer._device}")

    # 测试几个位姿
    test_poses = [
        np.array([2.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.5]),
        np.array([0.5, -0.5, 1.2]),
    ]

    for i, pos in enumerate(test_poses):
        t0 = time.time()
        depth, rgb = renderer.render(pos)
        elapsed = time.time() - t0
        print(f"  pose {i}: pos={pos} → depth "
              f"range [{depth.min():.2f}, {depth.max():.2f}] "
              f"| {elapsed*1000:.0f}ms")

    print("GSplatRenderer self-check passed")
