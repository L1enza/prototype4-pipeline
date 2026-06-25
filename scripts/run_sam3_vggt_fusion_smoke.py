#!/usr/bin/env python3
"""Fuse filtered SAM 3 masks with saved VGGT world points.

This is a lightweight smoke test. It does not run SAM 3 or VGGT inference.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prototype4_matplotlib")

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse filtered SAM 3 masks with saved VGGT world points.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--confidence-threshold", type=float, default=1.05, help="Minimum VGGT world_points_conf value.")
    parser.add_argument("--max-ply-points", type=int, default=50000, help="Maximum points to write to the PLY.")
    parser.add_argument("--filtered-dir", default=None, help="Override filtered SAM 3 directory.")
    parser.add_argument("--vggt-dir", default=None, help="Override VGGT smoke output directory.")
    parser.add_argument("--output-dir", default=None, help="Override fusion output directory.")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_confident(points, conf, threshold):
    return np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf >= threshold)


def resize_mask(mask_path, width, height):
    mask = Image.open(mask_path).convert("L")
    resized = mask.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def point_stats(points, conf):
    if len(points) == 0:
        return {
            "point_count": 0,
            "centroid_xyz": None,
            "min_xyz": None,
            "max_xyz": None,
            "range_xyz": None,
            "confidence_mean": None,
            "confidence_min": None,
            "confidence_max": None,
        }
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    return {
        "point_count": int(len(points)),
        "centroid_xyz": points.mean(axis=0).astype(float).tolist(),
        "min_xyz": min_xyz.astype(float).tolist(),
        "max_xyz": max_xyz.astype(float).tolist(),
        "range_xyz": (max_xyz - min_xyz).astype(float).tolist(),
        "confidence_mean": float(conf.mean()),
        "confidence_min": float(conf.min()),
        "confidence_max": float(conf.max()),
    }


def color_for(index):
    return COLORS[index % len(COLORS)]


def rgba_mask(mask, color, alpha=92):
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = mask.astype(np.uint8) * alpha
    return Image.fromarray(overlay, mode="RGBA")


def save_frame_overlay(frame_record, output_path):
    image = Image.open(frame_record["original_path"]).convert("RGBA")
    draw_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for idx, mask_record in enumerate(frame_record["masks"]):
        mask_path = mask_record["source_mask_path"]
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        color = color_for(idx)
        draw_layer = Image.alpha_composite(draw_layer, rgba_mask(mask, color))
    composed = Image.alpha_composite(image, draw_layer).convert("RGB")
    draw = ImageDraw.Draw(composed)
    for idx, mask_record in enumerate(frame_record["masks"]):
        color = color_for(idx)
        bbox = mask_record.get("bbox_2d") or {}
        if bbox:
            draw.rectangle((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]), outline=color, width=3)
        foot = mask_record.get("bottom_center_point_2d")
        label_xy = (bbox.get("x0", 8), max(0, bbox.get("y0", 18) - 18)) if bbox else (8, 8 + idx * 16)
        label = "f{} m{} p{}".format(mask_record["frame_index"], mask_record["mask_id"], mask_record["point_count"])
        draw.text(label_xy, label, fill=color)
        if foot:
            x = int(round(foot["x"]))
            y = int(round(foot["y"]))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)
    return str(output_path)


def save_scatter(path, point_sets, axes, title, xlabel, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    any_points = False
    for idx, item in enumerate(point_sets):
        pts = item["points"]
        if len(pts) == 0:
            continue
        any_points = True
        color = np.asarray(color_for(idx), dtype=np.float32) / 255.0
        ax.scatter(pts[:, axes[0]], pts[:, axes[1]], s=3.5, color=color, alpha=0.72, linewidths=0, label="f{} m{}".format(item["frame_index"], item["mask_id"]))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any_points:
        ax.legend(markerscale=3, fontsize=7, loc="best", frameon=True)
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def write_ply(path, point_sets, max_points):
    all_rows = []
    for idx, item in enumerate(point_sets):
        pts = item["points"]
        if len(pts) == 0:
            continue
        color = np.asarray(color_for(idx), dtype=np.uint8)
        colors = np.repeat(color[None, :], len(pts), axis=0)
        all_rows.append((pts, colors))
    if all_rows:
        points = np.concatenate([row[0] for row in all_rows], axis=0)
        colors = np.concatenate([row[1] for row in all_rows], axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    if len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
        points = points[selected]
        colors = colors[selected]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("element vertex {}\n".format(len(points)))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2])))
    return str(path), int(len(points))


def main():
    args = parse_args()
    filtered_dir = Path(args.filtered_dir) if args.filtered_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_filtered"
    vggt_dir = Path(args.vggt_dir) if args.vggt_dir else PROJECT_ROOT / "outputs" / args.run_id / "world_reconstruction" / "vggt_inference_smoke"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "fusion" / "sam3_vggt_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    world_points_path = vggt_dir / "world_points.npy"
    world_conf_path = vggt_dir / "world_points_conf.npy"
    if not world_points_path.exists() or not world_conf_path.exists():
        raise FileNotFoundError("Missing VGGT world_points.npy or world_points_conf.npy in {}".format(vggt_dir))
    if not filtered_dir.exists():
        raise FileNotFoundError("Missing filtered SAM 3 directory {}".format(filtered_dir))

    world_points = np.load(world_points_path, mmap_mode="r")
    world_conf = np.load(world_conf_path, mmap_mode="r")
    if world_points.ndim != 5 or world_conf.ndim != 4:
        raise ValueError("Expected world_points [B,F,H,W,3] and confidence [B,F,H,W], got {} and {}".format(world_points.shape, world_conf.shape))
    frame_count = int(world_points.shape[1])
    point_h = int(world_points.shape[2])
    point_w = int(world_points.shape[3])

    frame_records = []
    mask_records = []
    point_sets = []
    skipped = []

    for frame_dir in sorted(filtered_dir.glob("frame_*")):
        metadata_path = frame_dir / "frame_metadata.json"
        if not metadata_path.exists():
            continue
        filtered_meta = load_json(metadata_path)
        frame_index = int(filtered_meta["frame_index"])
        if frame_index >= frame_count:
            skipped.append({"frame_index": frame_index, "reason": "no_matching_vggt_frame"})
            continue
        frame_points = np.asarray(world_points[0, frame_index], dtype=np.float32)
        frame_conf = np.asarray(world_conf[0, frame_index], dtype=np.float32)
        frame_record = {
            "frame_index": frame_index,
            "filtered_metadata_path": str(metadata_path),
            "original_path": filtered_meta.get("original_path"),
            "vggt_frame_index": frame_index,
            "vggt_point_resolution": {"height": point_h, "width": point_w},
            "masks": [],
        }
        for kept in filtered_meta.get("kept_masks", []):
            mask_path = Path(kept["mask_path"])
            resized_mask = resize_mask(mask_path, point_w, point_h)
            valid = resized_mask & finite_confident(frame_points, frame_conf, args.confidence_threshold)
            points = frame_points[valid].reshape(-1, 3).astype(np.float32)
            conf = frame_conf[valid].reshape(-1).astype(np.float32)
            stats = point_stats(points, conf)
            record = {
                "frame_index": frame_index,
                "vggt_frame_index": frame_index,
                "mask_id": int(kept["mask_index"]),
                "bbox_2d": kept.get("bbox"),
                "centroid_2d": kept.get("centroid"),
                "bottom_center_point_2d": kept.get("bottom_center_point"),
                "source_mask_path": str(mask_path),
                "source_single_mask_overlay_path": kept.get("single_mask_overlay_path"),
                "resized_mask_resolution": {"height": point_h, "width": point_w},
                "confidence_threshold": args.confidence_threshold,
                "sam_confidence_score": kept.get("confidence_score"),
                "known_limitations": ["Referee may survive geometry filtering; handle later with appearance/referee classifier."],
                **stats,
            }
            frame_record["masks"].append(record)
            mask_records.append(record)
            point_sets.append({"frame_index": frame_index, "mask_id": int(kept["mask_index"]), "points": points, "confidence": conf})
        overlay_path = output_dir / "frame_{:03d}_kept_mask_ids.png".format(frame_index)
        frame_record["kept_mask_id_overlay_path"] = save_frame_overlay(frame_record, overlay_path)
        frame_records.append(frame_record)

    topdown_path = output_dir / "players_topdown_xz.png"
    side_path = output_dir / "players_side_xy.png"
    ply_path = output_dir / "kept_player_points.ply"
    topdown = save_scatter(topdown_path, point_sets, (0, 2), "SAM 3 kept masks in VGGT world points: top-down X/Z", "x", "z")
    side = save_scatter(side_path, point_sets, (0, 1), "SAM 3 kept masks in VGGT world points: side X/Y", "x", "y")
    ply, ply_points = write_ply(ply_path, point_sets, args.max_ply_points)

    total_points = int(sum(record["point_count"] for record in mask_records))
    nonempty_masks = int(sum(1 for record in mask_records if record["point_count"] > 0))
    metadata = {
        "status": "complete",
        "stage": "sam3_vggt_fusion_smoke",
        "run_id": args.run_id,
        "inputs": {
            "filtered_sam3_dir": str(filtered_dir),
            "vggt_dir": str(vggt_dir),
            "world_points": str(world_points_path),
            "world_points_conf": str(world_conf_path),
        },
        "parameters": {
            "confidence_threshold": args.confidence_threshold,
            "max_ply_points": args.max_ply_points,
            "frame_mapping": "filtered frame_index maps directly to VGGT frame axis",
        },
        "vggt_shapes": {
            "world_points": list(world_points.shape),
            "world_points_conf": list(world_conf.shape),
        },
        "known_limitations": [
            "Referee may survive SAM 3 geometry filtering and should be handled later by appearance/referee classifier.",
            "This smoke test uses direct frame-index alignment between filtered masks and VGGT frame slices.",
            "Field-space alignment is not applied here; coordinates are raw VGGT world coordinates.",
        ],
        "frames": frame_records,
        "masks": mask_records,
        "skipped": skipped,
        "artifacts": {
            "topdown_xz": topdown,
            "side_xy": side,
            "ply_point_cloud": ply,
        },
    }
    summary = {
        "status": "complete",
        "stage": "sam3_vggt_fusion_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "fusion_metadata.json"),
        "parameters": metadata["parameters"],
        "counts": {
            "frames_processed": len(frame_records),
            "kept_masks_processed": len(mask_records),
            "nonempty_3d_masks": nonempty_masks,
            "total_confident_player_points": total_points,
            "ply_points_written": ply_points,
            "skipped_frames": len(skipped),
        },
        "artifacts": metadata["artifacts"],
        "frame_overlays": [frame["kept_mask_id_overlay_path"] for frame in frame_records],
        "known_limitations": metadata["known_limitations"],
    }
    write_json(output_dir / "fusion_metadata.json", metadata)
    write_json(output_dir / "fusion_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
