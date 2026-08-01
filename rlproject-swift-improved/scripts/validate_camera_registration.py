#!/usr/bin/env python3
"""Validate exported 3DGS coordinates against recorded training cameras.

The Nerfstudio transform files store OpenGL camera poses (+Y up, -Z view).
gsplat consumes OpenCV camera poses (+Y down, +Z view), so the expected
conversion flips the camera Y and Z axes without changing world coordinates.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gs_renderer import GSplatRenderer


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def scaled_camera(transforms, max_dimension, center_crop_square=False):
    source_width = int(transforms["w"])
    source_height = int(transforms["h"])
    if center_crop_square:
        crop_size = min(source_width, source_height)
        crop_left = (source_width - crop_size) / 2.0
        crop_top = (source_height - crop_size) / 2.0
        scale = min(1.0, float(max_dimension) / crop_size)
        width = height = max(1, int(round(crop_size * scale)))
        scale_x = scale_y = width / crop_size
    else:
        crop_left = crop_top = 0.0
        crop_size = None
        scale = min(1.0, float(max_dimension) / max(source_width, source_height))
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        scale_x = width / source_width
        scale_y = height / source_height
    return {
        "width": width,
        "height": height,
        "fx": float(transforms["fl_x"]) * scale_x,
        "fy": float(transforms["fl_y"]) * scale_y,
        "cx": (float(transforms["cx"]) - crop_left) * scale_x,
        "cy": (float(transforms["cy"]) - crop_top) * scale_y,
        "center_crop_square": bool(center_crop_square),
        "crop_box": (
            [crop_left, crop_top, crop_left + crop_size, crop_top + crop_size]
            if crop_size is not None else None
        ),
    }


def image_metrics(reference, rendered, valid_mask):
    reference = np.asarray(reference, dtype=np.float32) / 255.0
    rendered = np.asarray(rendered, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    coverage = float(mask.mean())
    if not np.any(mask):
        return {"coverage": coverage, "mae": None, "luma_correlation": None}
    ref_pixels = reference[mask]
    render_pixels = rendered[mask]
    mae = float(np.mean(np.abs(ref_pixels - render_pixels)))
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    ref_luma = ref_pixels @ weights
    render_luma = render_pixels @ weights
    correlation = (
        float(np.corrcoef(ref_luma, render_luma)[0, 1])
        if np.std(ref_luma) > 1e-6 and np.std(render_luma) > 1e-6
        else None
    )
    return {"coverage": coverage, "mae": mae, "luma_correlation": correlation}


def to_uint8(image):
    return np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)


def labelled_panel(image, label):
    from PIL import Image, ImageDraw

    panel = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, min(panel.width, 210), 18), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return panel


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--indices", default="0,77,154,231,308")
    parser.add_argument("--max-dimension", type=int, default=320)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--center-crop-square", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.transforms, encoding="utf-8") as handle:
        transforms = json.load(handle)
    camera = scaled_camera(
        transforms, args.max_dimension, center_crop_square=args.center_crop_square
    )
    renderer = GSplatRenderer(
        args.ply,
        width=camera["width"],
        height=camera["height"],
        fx=camera["fx"],
        fy=camera["fy"],
        cx=camera["cx"],
        cy=camera["cy"],
        device=args.device,
    )
    frames = transforms["frames"]
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    candidates = {
        "opengl_to_opencv": OPENGL_TO_OPENCV,
        "as_recorded_opencv": np.eye(4),
    }
    records = []
    rows = []
    data_root = os.path.dirname(os.path.abspath(args.transforms))

    for index in indices:
        frame = frames[index]
        image_path = os.path.join(data_root, frame["file_path"])
        reference = Image.open(image_path).convert("RGB")
        if camera["crop_box"] is not None:
            reference = reference.crop(tuple(camera["crop_box"]))
        reference = reference.resize(
            (camera["width"], camera["height"]), Image.Resampling.BILINEAR
        )
        panels = [labelled_panel(np.asarray(reference), f"reference {index}")]
        recorded_c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        for name, camera_axis_transform in candidates.items():
            c2w = recorded_c2w @ camera_axis_transform
            depth, rgb = renderer.render(
                recorded_c2w[:3, 3], camera_c2w=c2w
            )
            valid = depth[..., 0] < renderer.max_depth - 1e-4
            metrics = image_metrics(reference, rgb, valid)
            records.append({
                "frame_index": index,
                "file_path": frame["file_path"],
                "candidate": name,
                **metrics,
            })
            panels.append(labelled_panel(to_uint8(rgb), name))
        row = Image.new("RGB", (camera["width"] * len(panels), camera["height"]))
        for panel_index, panel in enumerate(panels):
            row.paste(panel, (panel_index * camera["width"], 0))
        rows.append(row)

    summary = {}
    for name in candidates:
        selected = [record for record in records if record["candidate"] == name]
        summary[name] = {
            "mean_coverage": float(np.mean([row["coverage"] for row in selected])),
            "mean_mae": float(np.mean([
                row["mae"] for row in selected if row["mae"] is not None
            ])),
            "mean_luma_correlation": float(np.mean([
                row["luma_correlation"] for row in selected
                if row["luma_correlation"] is not None
            ])),
        }
    expected = summary["opengl_to_opencv"]
    alternative = summary["as_recorded_opencv"]
    provisional_pass = (
        expected["mean_coverage"] >= 0.5
        and expected["mean_luma_correlation"] > alternative["mean_luma_correlation"]
    )
    report = {
        "camera": camera,
        "indices": indices,
        "summary": summary,
        "provisional_pass": provisional_pass,
        "records": records,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    montage = Image.new(
        "RGB", (rows[0].width, sum(row.height for row in rows)), (0, 0, 0)
    )
    y = 0
    for row in rows:
        montage.paste(row, (0, y))
        y += row.height
    montage_path = os.path.splitext(output_path)[0] + "_montage.png"
    montage.save(montage_path)
    print(json.dumps({"summary": summary, "provisional_pass": provisional_pass}, indent=2))
    print(f"Saved: {output_path}")
    print(f"Saved: {montage_path}")


if __name__ == "__main__":
    main()
