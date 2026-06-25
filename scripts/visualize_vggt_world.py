#!/usr/bin/env python3
"""Lightweight visual inspection for saved VGGT world outputs.

Reads existing outputs from:
  outputs/<run_id>/world_reconstruction/vggt_inference_smoke/

Writes safe, downsampled visualization artifacts to:
  outputs/<run_id>/world_reconstruction/vggt_visualization/

This script does not import VGGT, instantiate models, download weights, or run
inference. It only consumes saved .npy/.npz/.json artifacts.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prototype4_matplotlib")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved VGGT world reconstruction artifacts.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id to visualize.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Minimum confidence for points.")
    parser.add_argument("--max-points", type=int, default=50000, help="Maximum points to write/render.")
    parser.add_argument("--input-dir", default=None, help="Override VGGT inference artifact directory.")
    parser.add_argument("--output-dir", default=None, help="Override visualization output directory.")
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite_mask(points, confidence, threshold):
    return np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence > threshold)


def downsample_indices(count, max_points):
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points).round().astype(np.int64)


def colorize(frame_index, confidence):
    frame_index = np.asarray(frame_index)
    confidence = np.asarray(confidence)
    if frame_index.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    max_frame = max(int(frame_index.max()), 1)
    hue = frame_index.astype(np.float32) / max_frame
    conf_norm = confidence.astype(np.float32)
    if conf_norm.size and np.nanmax(conf_norm) > np.nanmin(conf_norm):
        conf_norm = (conf_norm - np.nanmin(conf_norm)) / (np.nanmax(conf_norm) - np.nanmin(conf_norm))
    else:
        conf_norm = np.ones_like(conf_norm)
    # Compact hand-rolled color ramp: red -> green -> blue, brightened by confidence.
    r = np.clip(1.5 - np.abs(3.0 * hue - 0.0), 0, 1)
    g = np.clip(1.5 - np.abs(3.0 * hue - 1.5), 0, 1)
    b = np.clip(1.5 - np.abs(3.0 * hue - 3.0), 0, 1)
    colors = np.stack([r, g, b], axis=1)
    colors = 40 + colors * (120 + 95 * conf_norm[:, None])
    return np.clip(colors, 0, 255).astype(np.uint8)


def load_points(input_dir, threshold, max_points):
    preview_path = input_dir / "point_cloud_preview.npz"
    source = "point_cloud_preview.npz"
    if preview_path.exists():
        npz = np.load(preview_path)
        points = npz["points"].astype(np.float32)
        confidence = npz["confidence"].astype(np.float32)
        frame_index = npz["frame_index"].astype(np.int16)
        valid = finite_mask(points, confidence, threshold)
        valid_indices = np.flatnonzero(valid)
        selected = valid_indices[downsample_indices(len(valid_indices), max_points)] if len(valid_indices) else valid_indices
        return {
            "source": source,
            "points": points[selected],
            "confidence": confidence[selected],
            "frame_index": frame_index[selected],
            "valid_count_in_source": int(valid.sum()),
            "source_count": int(points.shape[0]),
        }

    world_points_path = input_dir / "world_points.npy"
    conf_path = input_dir / "world_points_conf.npy"
    if not world_points_path.exists() or not conf_path.exists():
        raise FileNotFoundError("Missing point_cloud_preview.npz and world_points/world_points_conf arrays in {}".format(input_dir))

    source = "world_points.npy"
    world_points = np.load(world_points_path, mmap_mode="r")
    world_conf = np.load(conf_path, mmap_mode="r")
    # Expected [1, S, H, W, 3] and [1, S, H, W]. Use batch 0.
    points = np.asarray(world_points[0]).reshape(-1, 3).astype(np.float32)
    confidence = np.asarray(world_conf[0]).reshape(-1).astype(np.float32)
    frame_count, height, width = world_points.shape[1], world_points.shape[2], world_points.shape[3]
    frame_index = np.repeat(np.arange(frame_count, dtype=np.int16), height * width)
    valid = finite_mask(points, confidence, threshold)
    valid_indices = np.flatnonzero(valid)
    selected = valid_indices[downsample_indices(len(valid_indices), max_points)] if len(valid_indices) else valid_indices
    return {
        "source": source,
        "points": points[selected],
        "confidence": confidence[selected],
        "frame_index": frame_index[selected],
        "valid_count_in_source": int(valid.sum()),
        "source_count": int(points.shape[0]),
    }


def write_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("element vertex {}\n".format(len(points)))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2])
            ))


def save_scatter_png(path, x, y, color_values, xlabel, ylabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    if len(x):
        scatter = ax.scatter(x, y, c=color_values, s=0.35, cmap="viridis", alpha=0.75, linewidths=0)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="frame index")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_hist_png(path, values, title, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    if len(values):
        ax.hist(values, bins=80, color="#376795", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", linewidth=0.25, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ranges(points):
    if not len(points):
        return {"min_xyz": None, "max_xyz": None, "span_xyz": None}
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    return {
        "min_xyz": min_xyz.astype(float).tolist(),
        "max_xyz": max_xyz.astype(float).tolist(),
        "span_xyz": (max_xyz - min_xyz).astype(float).tolist(),
    }


def per_frame_counts(frame_index):
    if not len(frame_index):
        return {}
    unique, counts = np.unique(frame_index, return_counts=True)
    return {str(int(frame)): int(count) for frame, count in zip(unique, counts)}


def write_html(path, summary, image_paths):
    rel_images = [Path(p).name for p in image_paths]
    cards = "\n".join('<section><h2>{}</h2><img src="{}" alt="{}"></section>'.format(name, img, name) for name, img in zip(["Top-down X/Z", "Side X/Y", "Confidence Histogram"], rel_images))
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>VGGT World Preview - {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1 {{ font-size: 24px; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
    .box {{ background: white; border: 1px solid #d8dde6; border-radius: 6px; padding: 12px; }}
    img {{ max-width: 100%; background: white; border: 1px solid #d8dde6; border-radius: 6px; }}
    section {{ margin: 22px 0; }}
    code {{ background: #eef1f5; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>VGGT World Preview: {run_id}</h1>
  <div class=\"meta\">
    <div class=\"box\"><strong>Status</strong><br>{status}</div>
    <div class=\"box\"><strong>Points rendered</strong><br>{rendered}</div>
    <div class=\"box\"><strong>Confidence threshold</strong><br>{threshold}</div>
    <div class=\"box\"><strong>PLY</strong><br><code>{ply}</code></div>
  </div>
  {cards}
</body>
</html>
""".format(
        run_id=summary["run_id"],
        status=summary["status"],
        rendered=summary["counts"]["rendered_points"],
        threshold=summary["parameters"]["confidence_threshold"],
        ply=Path(summary["artifacts"]["ply_point_cloud"]).name,
        cards=cards,
    )
    path.write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else PROJECT_ROOT / "outputs" / args.run_id / "world_reconstruction" / "vggt_inference_smoke"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "world_reconstruction" / "vggt_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(input_dir / "metadata.json")
    camera_metadata = load_json(input_dir / "camera_metadata.json")
    reconstruction_manifest = load_json(input_dir / "world_reconstruction_manifest.json")

    loaded = load_points(input_dir, args.confidence_threshold, args.max_points)
    points = loaded["points"]
    confidence = loaded["confidence"]
    frame_index = loaded["frame_index"]
    colors = colorize(frame_index, confidence)

    ply_path = output_dir / "vggt_world_preview.ply"
    topdown_path = output_dir / "preview_topdown_xz.png"
    side_path = output_dir / "preview_side_xy.png"
    hist_path = output_dir / "confidence_histogram.png"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "index.html"

    write_ply(ply_path, points, colors)
    save_scatter_png(topdown_path, points[:, 0] if len(points) else [], points[:, 2] if len(points) else [], frame_index, "x", "z", "VGGT point cloud top-down X/Z")
    save_scatter_png(side_path, points[:, 0] if len(points) else [], points[:, 1] if len(points) else [], frame_index, "x", "y", "VGGT point cloud side X/Y")
    save_hist_png(hist_path, confidence, "VGGT point confidence", "confidence")

    summary = {
        "status": "complete",
        "stage": "vggt_world_visualization",
        "run_id": args.run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source": loaded["source"],
        "parameters": {
            "confidence_threshold": args.confidence_threshold,
            "max_points": args.max_points,
        },
        "counts": {
            "source_points": loaded["source_count"],
            "valid_points_in_source": loaded["valid_count_in_source"],
            "rendered_points": int(len(points)),
            "frames_present_in_rendered_points": per_frame_counts(frame_index),
        },
        "coordinate_ranges": ranges(points),
        "confidence": {
            "min": float(confidence.min()) if len(confidence) else None,
            "max": float(confidence.max()) if len(confidence) else None,
            "mean": float(confidence.mean()) if len(confidence) else None,
            "median": float(np.median(confidence)) if len(confidence) else None,
        },
        "camera": {
            "frame_count": camera_metadata.get("frame_count") if isinstance(camera_metadata, dict) else None,
            "pose_enc_shape": camera_metadata.get("pose_enc_shape") if isinstance(camera_metadata, dict) else None,
            "extrinsics_shape": camera_metadata.get("extrinsics_shape") if isinstance(camera_metadata, dict) else None,
            "intrinsics_shape": camera_metadata.get("intrinsics_shape") if isinstance(camera_metadata, dict) else None,
        },
        "inference_metadata_status": metadata.get("status") if isinstance(metadata, dict) else None,
        "reconstruction_manifest_status": reconstruction_manifest.get("status") if isinstance(reconstruction_manifest, dict) else None,
        "artifacts": {
            "summary": str(summary_path),
            "ply_point_cloud": str(ply_path),
            "topdown_png": str(topdown_path),
            "side_png": str(side_path),
            "confidence_histogram_png": str(hist_path),
            "html": str(html_path),
        },
        "safety": {
            "did_run_vggt_inference": False,
            "did_import_vggt": False,
            "full_resolution_rendering": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(html_path, summary, [topdown_path, side_path, hist_path])
    print("Wrote VGGT visualization outputs to {}".format(output_dir))
    print("Summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
