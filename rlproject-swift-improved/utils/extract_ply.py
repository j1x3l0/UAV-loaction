"""
extract_ply.py — 从 Nerfstudio checkpoint 提取 3DGS .ply 文件

WHY: nerfstudio 安装失败(PyAV编译问题), 但 checkpoint 中有完整的
     Gaussian 参数。提取为 .ply 后用 gsplat 直接渲染。

用法:
  python utils/extract_ply.py
  → 为 gs_data 下所有场景生成 .ply
"""

import torch
import numpy as np
import os, sys, struct
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
GS_DATA = ROOT.parent / "data" / "gs_data"


def extract_gaussian_params(ckpt_path: Path) -> dict:
    """从 Nerfstudio ckpt 提取 Gaussian 参数 (flat keys with dots)"""
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)

    # keys are flat: '_model.gauss_params.means', etc.
    pipeline = ckpt['pipeline']
    prefix = '_model.gauss_params.'
    params = {}
    for key, val in pipeline.items():
        if key.startswith(prefix):
            param_name = key[len(prefix):]
            params[param_name] = val.numpy()

    return {
        'means': params['means'],
        'quats': params['quats'],
        'scales': params['scales'],
        'opacities': params['opacities'],
        'features_dc': params['features_dc'],
    }


def save_gs_ply(params: dict, output_path: Path):
    """
    保存为标准 3D Gaussian Splatting .ply 格式
    兼容 gsplat / most GS renderers
    """
    N = len(params['means'])
    features_dc = params['features_dc']         # (N, 3)
    opacities = params['opacities']             # (N, 1)
    scales = params['scales']                   # (N, 3)
    quats = params['quats']                     # (N, 4)
    means = params['means']                     # (N, 3)

    # 定义属性
    properties = [
        ('x', 'float'),
        ('y', 'float'),
        ('z', 'float'),
        ('nx', 'float'),  # normal (unused, set 0)
        ('ny', 'float'),
        ('nz', 'float'),
        ('f_dc_0', 'float'),
        ('f_dc_1', 'float'),
        ('f_dc_2', 'float'),
        ('opacity', 'float'),
        ('scale_0', 'float'),
        ('scale_1', 'float'),
        ('scale_2', 'float'),
        ('rot_0', 'float'),
        ('rot_1', 'float'),
        ('rot_2', 'float'),
        ('rot_3', 'float'),
    ]

    vertices = []
    for i in range(N):
        x, y, z = means[i]
        r, g, b = features_dc[i]
        # SH DC → 0-1 RGB via sigmoid
        r = 1.0 / (1.0 + np.exp(-r))
        g = 1.0 / (1.0 + np.exp(-g))
        b = 1.0 / (1.0 + np.exp(-b))
        opacity = 1.0 / (1.0 + np.exp(-opacities[i][0]))
        sx, sy, sz = np.exp(scales[i])
        qw, qx, qy, qz = quats[i]
        vertices.append((
            x, y, z,         # position
            0.0, 0.0, 0.0,   # normal (unused)
            r, g, b,          # color
            opacity,          # opacity
            sx, sy, sz,       # scale
            qw, qx, qy, qz,   # rotation (quaternion)
        ))

    # 写 PLY (binary)
    with open(output_path, 'wb') as f:
        # Header
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {N}\n"
        for name, dtype in properties:
            header += f"property {dtype} {name}\n"
        header += "end_header\n"
        f.write(header.encode())

        # Binary data
        fmt = '<' + 'f' * (len(properties) - 1)  # x,y,z already in loop
        for v in vertices:
            f.write(struct.pack('<3f', v[0], v[1], v[2]))   # x,y,z
            f.write(struct.pack('<3f', v[3], v[4], v[5]))   # nx,ny,nz
            f.write(struct.pack('<3f', v[6], v[7], v[8]))   # f_dc
            f.write(struct.pack('<f', v[9]))                 # opacity
            f.write(struct.pack('<3f', v[10], v[11], v[12])) # scale
            f.write(struct.pack('<4f', v[13], v[14], v[15], v[16]))  # rot

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  → saved: {output_path.name} ({N:,} Gaussians, {size_mb:.1f} MB)")


def find_checkpoint(scene_dir: Path) -> Path | None:
    """在场景目录中定位 config.yml 和 checkpoint"""
    config_path = None
    ckpt_path = None
    for root, _, files in os.walk(scene_dir):
        if 'config.yml' in files:
            config_path = Path(root) / 'config.yml'
        if 'nerfstudio_models' in root:
            for f in files:
                if f.endswith('.ckpt'):
                    ckpt_path = Path(root) / f
                    break
    return ckpt_path


def main():
    os.makedirs(GS_DATA / "ply_exports", exist_ok=True)

    scenes = [d for d in GS_DATA.iterdir() if d.is_dir()]

    for scene_dir in sorted(scenes):
        scene_name = scene_dir.name
        print(f"\n=== {scene_name} ===")

        ckpt_path = find_checkpoint(scene_dir)
        if ckpt_path is None:
            print(f"  ⚠️ no checkpoint found, skipping")
            continue

        print(f"  ckpt: {ckpt_path.name}")
        params = extract_gaussian_params(ckpt_path)
        print(f"  Gaussians: {len(params['means']):,}")

        output_path = GS_DATA / "ply_exports" / f"{scene_name}_gs.ply"
        save_gs_ply(params, output_path)

    print(f"\nDone. ply files in: {GS_DATA / 'ply_exports'}")


if __name__ == "__main__":
    main()
