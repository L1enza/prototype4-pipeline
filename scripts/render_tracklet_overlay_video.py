#!/usr/bin/env python3
"""Render a short MP4 visualizing smoke-stage player tracklets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS_RGB = [
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
    (0, 114, 178),
    (213, 94, 0),
]
KNOWN_LIMITATIONS = [
    "This renders the existing 4-frame smoke tracklets only.",
    "Tracklets may contain identity swaps, especially in clustered players.",
    "Referee may still appear as a tracklet.",
    "This is not full-video tracking or final visualization.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render MP4 overlay for player tracklet smoke outputs.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--tracklet-metadata", default=None, help="Override tracklet metadata path.")
    parser.add_argument("--sampled-frames-dir", default=None, help="Override sampled frames directory.")
    parser.add_argument("--filtered-dir", default=None, help="Override filtered SAM 3 directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--fps", type=float, default=2.0, help="Output MP4 frames per second.")
    parser.add_argument("--mask-alpha", type=int, default=70, help="Mask fill alpha, 0-255.")
    parser.add_argument("--line-width", type=int, default=4, help="BBox and outline line width.")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def color_for(track_id):
    return COLORS_RGB[(int(track_id) - 1) % len(COLORS_RGB)]


def detections_by_frame(detections):
    by_frame = {}
    for det in detections:
        by_frame.setdefault(int(det["frame_index"]), []).append(det)
    for frame_dets in by_frame.values():
        frame_dets.sort(key=lambda item: int(item["track_id"]))
    return by_frame


def sampled_frame_path(sampled_frames_dir, frame_index):
    candidates = [
        sampled_frames_dir / "frame_{:06d}.jpg".format(frame_index),
        sampled_frames_dir / "frame_{:03d}.jpg".format(frame_index),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    all_frames = sorted(sampled_frames_dir.glob("frame_*.jpg"))
    if frame_index < len(all_frames):
        return all_frames[frame_index]
    return None


def draw_label(draw, xy, text, fill, bg=(0, 0, 0)):
    x, y = int(xy[0]), int(xy[1])
    try:
        bbox = draw.textbbox((x, y), text)
    except Exception:
        bbox = (x, y, x + 8 * len(text), y + 14)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, fill=fill)


def mask_outline(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def render_frame(frame_path, frame_index, detections, output_path, mask_alpha, line_width):
    image = Image.open(frame_path).convert("RGBA")
    width, height = image.size
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    outline_jobs = []

    for det in detections:
        track_id = int(det["track_id"])
        color = color_for(track_id)
        mask_path = det.get("mask_path")
        if mask_path and Path(mask_path).exists():
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            if mask.shape[:2] == (height, width):
                overlay[..., 0][mask] = color[0]
                overlay[..., 1][mask] = color[1]
                overlay[..., 2][mask] = color[2]
                overlay[..., 3][mask] = mask_alpha
                outline_jobs.append((mask, color, track_id))

    composed = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA")).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)
    for mask, color, _track_id in outline_jobs:
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.drawContours(cv_img, mask_outline(mask), -1, bgr, max(1, line_width - 1))
    composed = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(composed)

    for det in detections:
        track_id = int(det["track_id"])
        color = color_for(track_id)
        bbox = det.get("bbox_2d") or {}
        has_3d = det.get("centroid_3d") is not None
        if bbox:
            rect = (int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"]))
            draw.rectangle(rect, outline=color, width=line_width + (2 if has_3d else 0))
            if has_3d:
                draw.rectangle((rect[0] - 3, rect[1] - 3, rect[2] + 3, rect[3] + 3), outline=(255, 255, 255), width=1)
            label_xy = (rect[0], max(0, rect[1] - 22))
        else:
            centroid = det.get("centroid_2d") or {"x": 12, "y": 42}
            label_xy = (centroid["x"], centroid["y"])
        label = "T{}".format(track_id)
        if has_3d:
            label += " 3D"
        draw_label(draw, label_xy, label, fill=color)
        foot = det.get("foot_point_2d")
        if foot:
            x = int(round(foot["x"]))
            y = int(round(foot["y"]))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=1)

    draw_label(draw, (12, 12), "frame {:03d} | detections {}".format(frame_index, len(detections)), fill=(255, 255, 255), bg=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)
    return str(output_path)


def write_mp4(frame_paths, output_path, fps):
    if not frame_paths:
        raise ValueError("No rendered frames were provided for MP4 writing.")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError("Could not read rendered frame {}".format(frame_paths[0]))
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open MP4 writer for {}".format(output_path))
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("Could not read rendered frame {}".format(path))
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return str(output_path)


def main():
    args = parse_args()
    tracklet_metadata_path = Path(args.tracklet_metadata) if args.tracklet_metadata else PROJECT_ROOT / "outputs" / args.run_id / "player_tracks" / "tracklet_smoke" / "tracklet_metadata.json"
    sampled_frames_dir = Path(args.sampled_frames_dir) if args.sampled_frames_dir else PROJECT_ROOT / "outputs" / args.run_id / "sampled_frames"
    filtered_dir = Path(args.filtered_dir) if args.filtered_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_filtered"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "visualizations" / "tracklet_overlay_smoke"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    tracklet_metadata = load_json(tracklet_metadata_path)
    by_frame = detections_by_frame(tracklet_metadata.get("detections", []))
    rendered_frames = []
    frame_records = []
    for frame_index in sorted(by_frame):
        frame_path = sampled_frame_path(sampled_frames_dir, frame_index)
        if frame_path is None:
            frame_meta_path = Path(by_frame[frame_index][0].get("source_frame_metadata", ""))
            if frame_meta_path.exists():
                frame_path = Path(load_json(frame_meta_path).get("original_path"))
        if frame_path is None or not Path(frame_path).exists():
            frame_records.append({"frame_index": frame_index, "status": "skipped", "reason": "missing_sampled_frame"})
            continue
        output_path = frames_dir / "frame_{:03d}_tracklet_overlay.png".format(frame_index)
        rendered = render_frame(Path(frame_path), frame_index, by_frame[frame_index], output_path, args.mask_alpha, args.line_width)
        rendered_frames.append(Path(rendered))
        frame_records.append({
            "frame_index": frame_index,
            "status": "rendered",
            "source_frame_path": str(frame_path),
            "rendered_frame_path": rendered,
            "detections": len(by_frame[frame_index]),
            "detections_with_3d_support": sum(1 for det in by_frame[frame_index] if det.get("centroid_3d") is not None),
        })

    mp4_path = output_dir / "tracklet_overlay_smoke.mp4"
    mp4 = write_mp4(rendered_frames, mp4_path, args.fps)
    metadata = {
        "status": "complete",
        "stage": "tracklet_overlay_video_smoke",
        "run_id": args.run_id,
        "inputs": {
            "tracklet_metadata": str(tracklet_metadata_path),
            "filtered_sam3_dir": str(filtered_dir),
            "sampled_frames_dir": str(sampled_frames_dir),
        },
        "parameters": {
            "fps": args.fps,
            "mask_alpha": args.mask_alpha,
            "line_width": args.line_width,
            "mp4_writer": "opencv mp4v",
        },
        "frames": frame_records,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "mp4": mp4,
            "rendered_frames_dir": str(frames_dir),
            "rendered_frames": [str(path) for path in rendered_frames],
        },
    }
    summary = {
        "status": "complete",
        "stage": "tracklet_overlay_video_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "render_metadata.json"),
        "counts": {
            "frames_rendered": len(rendered_frames),
            "detections_rendered": sum(row.get("detections", 0) for row in frame_records),
            "frames_skipped": sum(1 for row in frame_records if row.get("status") == "skipped"),
        },
        "artifacts": metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(output_dir / "render_metadata.json", metadata)
    write_json(output_dir / "render_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
