#!/usr/bin/env python3
"""Render synchronized broadcast tracking and top-down field projection panels."""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

KNOWN_LIMITATIONS = [
    "Manual homography is approximate.",
    "Tracking IDs may still switch.",
    "Referee may still be included.",
    "Player foot points are noisy.",
    "This is a short-clip smoke visualization, not final full-game analytics.",
]
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
    parser = argparse.ArgumentParser(description="Render synchronized tracking overlay and top-down field projection smoke visualization.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--tracking-frames-dir", default=None, help="Override stabilized tracking overlay frames directory.")
    parser.add_argument("--tracking-metadata", default=None, help="Override stabilized tracking metadata path.")
    parser.add_argument("--projected-points", default=None, help="Override projected_player_points.json path.")
    parser.add_argument("--field-template", default="assets/field_templates/nll_field_topdown.png", help="Top-down field template image.")
    parser.add_argument("--heatmap", default=None, help="Optional heatmap overlay image.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--fps", type=float, default=6.0, help="Output video/GIF FPS.")
    parser.add_argument("--trail-length", type=int, default=7, help="Number of recent frames to draw as fading trails.")
    parser.add_argument("--field-panel-width", type=int, default=720, help="Width of the right top-down field panel.")
    parser.add_argument("--gap", type=int, default=12, help="Gap between left and right panels.")
    parser.add_argument("--dot-radius", type=int, default=6, help="Current projected player dot radius.")
    parser.add_argument("--use-static-heatmap", action=argparse.BooleanOptionalAction, default=True, help="Blend the optional all-player heatmap under projected dots.")
    parser.add_argument("--static-heatmap-alpha", type=float, default=0.18, help="Alpha for optional static heatmap overlay.")
    parser.add_argument("--glow-alpha", type=float, default=0.22, help="Alpha for cumulative in-run glow generated from projected points.")
    parser.add_argument("--glow-sigma", type=float, default=14.0, help="Gaussian sigma for cumulative in-run glow.")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def color_for(track_id):
    return COLORS[(int(track_id) - 1) % len(COLORS)]


def frame_index_from_path(path):
    match = re.search(r"frame_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_tracking_frames(frames_dir):
    frames = []
    for path in sorted(frames_dir.glob("frame_*_tracking_overlay.png")):
        frame_index = frame_index_from_path(path)
        if frame_index is not None:
            frames.append((frame_index, path))
    return frames


def group_points(points):
    by_frame = {}
    by_track = {}
    for point in points:
        frame_index = int(point["frame_index"])
        track_id = int(point["track_id"])
        by_frame.setdefault(frame_index, []).append(point)
        by_track.setdefault(track_id, []).append(point)
    for rows in by_frame.values():
        rows.sort(key=lambda item: int(item["track_id"]))
    for rows in by_track.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return by_frame, by_track


def detections_by_frame(tracking_metadata):
    by_frame = {}
    for det in tracking_metadata.get("detections", []):
        by_frame.setdefault(int(det["frame_index"]), []).append(det)
    return by_frame


def resize_with_aspect(image, target_height=None, target_width=None):
    width, height = image.size
    if target_height is not None:
        scale = float(target_height) / float(height)
    elif target_width is not None:
        scale = float(target_width) / float(width)
    else:
        return image.copy(), 1.0
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def blend_static_heatmap(field_image, heatmap_path, alpha):
    if not heatmap_path or not Path(heatmap_path).exists():
        return field_image
    heat = Image.open(heatmap_path).convert("RGB").resize(field_image.size, Image.Resampling.LANCZOS)
    return Image.blend(field_image.convert("RGB"), heat, max(0.0, min(1.0, float(alpha))))


def cumulative_glow(points, template_size, max_frame, sigma):
    width, height = template_size
    density = np.zeros((height, width), dtype=np.float32)
    for point in points:
        if int(point["frame_index"]) > max_frame or not point.get("inside_field_template_bounds"):
            continue
        xy = point["projected_field_point"]
        x = int(round(float(xy["x"])))
        y = int(round(float(xy["y"])))
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1.0
    if np.max(density) <= 0:
        return None
    if sigma > 0:
        density = cv2.GaussianBlur(density, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
    if np.max(density) <= 0:
        return None
    norm = np.clip(density / np.max(density) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))


def draw_field_panel(field_base, all_points, by_track, frame_index, panel_width, trail_length, dot_radius, glow_alpha, glow_sigma):
    field = field_base.convert("RGB")
    glow = cumulative_glow(all_points, field.size, frame_index, glow_sigma)
    if glow is not None and glow_alpha > 0:
        field = Image.blend(field, glow, max(0.0, min(1.0, float(glow_alpha))))
    draw = ImageDraw.Draw(field)
    current_points = [point for point in all_points if int(point["frame_index"]) == frame_index and point.get("inside_field_template_bounds")]
    for track_id, rows in sorted(by_track.items()):
        history = [row for row in rows if row.get("inside_field_template_bounds") and frame_index - trail_length <= int(row["frame_index"]) <= frame_index]
        if len(history) >= 2:
            for idx in range(1, len(history)):
                a = history[idx - 1]["projected_field_point"]
                b = history[idx]["projected_field_point"]
                fade = idx / max(1, len(history) - 1)
                base = color_for(track_id)
                color = tuple(int(c * (0.35 + 0.65 * fade)) for c in base)
                draw.line((a["x"], a["y"], b["x"], b["y"]), fill=color, width=3)
    for point in current_points:
        track_id = int(point["track_id"])
        xy = point["projected_field_point"]
        x = int(round(float(xy["x"])))
        y = int(round(float(xy["y"])))
        color = color_for(track_id)
        draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=color, outline=(255, 255, 255), width=2)
        draw.text((x + dot_radius + 3, y - dot_radius), "T{}".format(track_id), fill=color)
    draw.rectangle((8, 8, 210, 34), fill=(255, 255, 255), outline=(20, 20, 20))
    draw.text((14, 14), "field frame {:03d} | players {}".format(frame_index, len(current_points)), fill=(0, 0, 0))
    resized, scale = resize_with_aspect(field, target_width=panel_width)
    return resized, scale, len(current_points)


def compose_frame(left_path, field_panel, output_path, gap):
    left = Image.open(left_path).convert("RGB")
    field_resized, _ = resize_with_aspect(field_panel, target_height=left.height)
    width = left.width + int(gap) + field_resized.width
    height = max(left.height, field_resized.height)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas.paste(left, (0, 0))
    canvas.paste(field_resized, (left.width + int(gap), 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, height - 24), "broadcast tracking", fill=(255, 255, 255))
    draw.text((left.width + int(gap) + 12, height - 24), "top-down field projection", fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def write_mp4(frame_paths, output_path, fps):
    if not frame_paths:
        raise ValueError("No frames available for MP4 writing.")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError("Could not read {}".format(frame_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open MP4 writer for {}".format(output_path))
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("Could not read {}".format(path))
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return str(output_path)


def write_gif(frame_paths, output_path, fps):
    if not frame_paths:
        raise ValueError("No frames available for GIF writing.")
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    try:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    finally:
        for image in images:
            image.close()
    return str(output_path)


def main():
    args = parse_args()
    tracking_frames_dir = Path(args.tracking_frames_dir) if args.tracking_frames_dir else PROJECT_ROOT / "outputs" / args.run_id / "short_clip_tracking_stabilized" / "frames"
    tracking_metadata_path = Path(args.tracking_metadata) if args.tracking_metadata else PROJECT_ROOT / "outputs" / args.run_id / "short_clip_tracking_stabilized" / "tracking_metadata.json"
    projected_points_path = Path(args.projected_points) if args.projected_points else PROJECT_ROOT / "outputs" / args.run_id / "field_calibration_smoke" / "projected_player_points.json"
    field_template_path = project_path(args.field_template)
    heatmap_path = Path(args.heatmap) if args.heatmap else PROJECT_ROOT / "outputs" / args.run_id / "heatmap_smoke" / "heatmap_all_players_overlay.png"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "side_by_side_field_tracking_smoke"
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    tracking_metadata = load_json(tracking_metadata_path)
    projected_payload = load_json(projected_points_path)
    all_projected = projected_payload.get("points", [])
    in_bounds = [point for point in all_projected if point.get("inside_field_template_bounds")]
    outside = [point for point in all_projected if not point.get("inside_field_template_bounds")]
    by_frame, by_track = group_points(in_bounds)
    det_by_frame = detections_by_frame(tracking_metadata)
    tracking_frames = load_tracking_frames(tracking_frames_dir)

    field_base = Image.open(field_template_path).convert("RGB")
    if args.use_static_heatmap:
        field_base = blend_static_heatmap(field_base, heatmap_path, args.static_heatmap_alpha)

    rendered = []
    frame_records = []
    projected_rendered = 0
    missing_projected = 0
    outside_skipped = 0
    for frame_index, left_path in tracking_frames:
        current_points = by_frame.get(frame_index, [])
        expected = len(det_by_frame.get(frame_index, []))
        current_outside = [point for point in outside if int(point["frame_index"]) == frame_index]
        field_panel, _scale, current_count = draw_field_panel(field_base, in_bounds, by_track, frame_index, args.field_panel_width, args.trail_length, args.dot_radius, args.glow_alpha, args.glow_sigma)
        output_path = frames_dir / "frame_{:03d}_side_by_side.png".format(frame_index)
        rendered_path = compose_frame(left_path, field_panel, output_path, args.gap)
        rendered.append(Path(rendered_path))
        projected_rendered += current_count
        outside_skipped += len(current_outside)
        missing = max(0, expected - current_count - len(current_outside))
        missing_projected += missing
        frame_records.append({
            "frame_index": int(frame_index),
            "tracking_frame": str(left_path),
            "rendered_frame": rendered_path,
            "tracking_detections": expected,
            "projected_points_rendered": current_count,
            "projected_points_outside_bounds_skipped": len(current_outside),
            "missing_projected_points": missing,
            "track_ids_rendered": sorted(int(point["track_id"]) for point in current_points),
        })

    mp4_path = output_dir / "side_by_side_tracking_field.mp4"
    gif_path = output_dir / "side_by_side_tracking_field.gif"
    mp4 = None
    mp4_error = None
    try:
        mp4 = write_mp4(rendered, mp4_path, args.fps)
    except Exception as exc:
        mp4_error = {"type": exc.__class__.__name__, "message": str(exc)}
    gif = write_gif(rendered, gif_path, args.fps)

    tracks_rendered = sorted({int(point["track_id"]) for point in in_bounds})
    metadata = {
        "status": "complete",
        "stage": "side_by_side_field_tracking_smoke",
        "run_id": args.run_id,
        "inputs": {
            "tracking_frames_dir": str(tracking_frames_dir),
            "tracking_metadata": str(tracking_metadata_path),
            "projected_points": str(projected_points_path),
            "field_template": str(field_template_path),
            "heatmap": str(heatmap_path) if heatmap_path.exists() else None,
        },
        "parameters": {
            "fps": args.fps,
            "trail_length": args.trail_length,
            "field_panel_width": args.field_panel_width,
            "gap": args.gap,
            "dot_radius": args.dot_radius,
            "use_static_heatmap": args.use_static_heatmap,
            "static_heatmap_alpha": args.static_heatmap_alpha,
            "glow_alpha": args.glow_alpha,
            "glow_sigma": args.glow_sigma,
        },
        "frames": frame_records,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "mp4": mp4,
            "mp4_error": mp4_error,
            "gif": gif,
            "frames_dir": str(frames_dir),
            "rendered_frames": [str(path) for path in rendered],
        },
    }
    summary = {
        "status": "complete",
        "stage": "side_by_side_field_tracking_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "side_by_side_metadata.json"),
        "counts": {
            "frames_rendered": len(rendered),
            "projected_points_rendered": projected_rendered,
            "missing_projected_points": missing_projected,
            "tracks_rendered": len(tracks_rendered),
            "points_outside_bounds_skipped": outside_skipped,
        },
        "tracks_rendered": tracks_rendered,
        "artifacts": metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(output_dir / "side_by_side_metadata.json", metadata)
    write_json(output_dir / "side_by_side_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
