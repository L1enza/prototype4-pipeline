#!/usr/bin/env python3
"""Create first smoke heatmaps from projected field-space player foot points."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

KNOWN_LIMITATIONS = [
    "Manual homography is approximate.",
    "Player foot points are noisy.",
    "Tracking IDs may still switch.",
    "Referee may still be included.",
    "This is an all-player smoke heatmap, not final player-specific analytics.",
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
    parser = argparse.ArgumentParser(description="Create top-down heatmap smoke outputs from field-projected player points.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--projected-points", default=None, help="Override projected_player_points.json path.")
    parser.add_argument("--field-template", default="assets/field_templates/nll_field_topdown.png", help="Top-down field template image.")
    parser.add_argument("--calibration-metadata", default=None, help="Override calibration metadata path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--min-track-points", type=int, default=1, help="Minimum in-bounds points for a track to be included.")
    parser.add_argument("--per-track-min-points", type=int, default=2, help="Minimum in-bounds points needed to write a per-track heatmap.")
    parser.add_argument("--sigma", type=float, default=18.0, help="Gaussian blur sigma in template pixels.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay alpha from 0 to 1.")
    parser.add_argument("--point-weight", type=float, default=1.0, help="Density increment per projected point.")
    parser.add_argument("--make-by-frame-gif", action=argparse.BooleanOptionalAction, default=True, help="Write cumulative heatmap_by_frame.gif.")
    parser.add_argument("--make-by-frame-mp4", action=argparse.BooleanOptionalAction, default=True, help="Write cumulative heatmap_by_frame.mp4 directly from PNG animation frames.")
    parser.add_argument("--gif-fps", type=float, default=6.0, help="By-frame GIF FPS.")
    parser.add_argument("--mp4-fps", type=float, default=None, help="By-frame MP4 FPS. Defaults to --gif-fps.")
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


def inside_points(points):
    return [point for point in points if point.get("inside_field_template_bounds")]


def group_by_track(points):
    tracks = {}
    for point in points:
        tracks.setdefault(int(point["track_id"]), []).append(point)
    for rows in tracks.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return tracks


def filter_tracks(points, min_track_points):
    tracks = group_by_track(points)
    included = {track_id for track_id, rows in tracks.items() if len(rows) >= min_track_points}
    return [point for point in points if int(point["track_id"]) in included], tracks, included


def density_from_points(points, size, point_weight):
    width, height = size
    density = np.zeros((height, width), dtype=np.float32)
    for point in points:
        xy = point.get("projected_field_point") or {}
        x = int(round(float(xy.get("x", -1))))
        y = int(round(float(xy.get("y", -1))))
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += float(point_weight)
    return density


def smooth_density(density, sigma):
    if density.size == 0:
        return density
    if float(sigma) <= 0:
        return density.copy()
    return cv2.GaussianBlur(density, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)


def normalize_density(density):
    max_value = float(np.max(density)) if density.size else 0.0
    if max_value <= 1e-9:
        return np.zeros_like(density, dtype=np.uint8), max_value
    norm = np.clip(density / max_value * 255.0, 0, 255).astype(np.uint8)
    return norm, max_value


def save_heatmap_images(points, template_image, output_heatmap, output_overlay, sigma, alpha, point_weight):
    template_rgb = np.asarray(template_image.convert("RGB"), dtype=np.uint8)
    density = density_from_points(points, template_image.size, point_weight)
    smoothed = smooth_density(density, sigma)
    norm, max_density = normalize_density(smoothed)
    heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat_rgb).save(output_heatmap)
    overlay = cv2.addWeighted(template_rgb, 1.0 - float(alpha), heat_rgb, float(alpha), 0.0)
    Image.fromarray(overlay).save(output_overlay)
    return {
        "raw_density_sum": float(np.sum(density)),
        "smoothed_density_max": max_density,
        "nonzero_density_pixels": int(np.count_nonzero(density)),
        "heatmap": str(output_heatmap),
        "overlay": str(output_overlay),
    }


def save_track_point_overlay(points, template_image, output_path):
    image = template_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    tracks = group_by_track(points)
    for track_id, rows in sorted(tracks.items()):
        color = color_for(track_id)
        coords = []
        for row in rows:
            xy = row["projected_field_point"]
            x = int(round(float(xy["x"])))
            y = int(round(float(xy["y"])))
            coords.append((x, y))
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=2)
        for x, y in coords:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(255, 255, 255), width=1)
        if coords:
            draw.text((coords[-1][0] + 6, coords[-1][1] - 6), "T{}".format(track_id), fill=color)
    image.save(output_path)
    return str(output_path)


def write_mp4(frame_paths, output_path, fps):
    if not frame_paths:
        return None, {"type": "ValueError", "message": "No frames available for MP4 writing."}
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None, {"type": "ValueError", "message": "Could not read {}".format(frame_paths[0])}
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return None, {"type": "RuntimeError", "message": "OpenCV could not open MP4 writer for {}".format(output_path)}
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                return None, {"type": "ValueError", "message": "Could not read {}".format(path)}
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return str(output_path), None


def write_gif(frame_paths, output_path, fps):
    if not frame_paths:
        return None
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    try:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    finally:
        for image in images:
            image.close()
    return str(output_path)


def write_by_frame_animation(points, template_image, frames_dir, gif_path, mp4_path, sigma, alpha, point_weight, gif_fps, mp4_fps, make_gif=True, make_mp4=True):
    frames = sorted({int(point["frame_index"]) for point in points})
    if not frames:
        return {"frames": [], "gif": None, "mp4": None, "mp4_error": None, "status": "skipped_no_frames"}
    frames_dir.mkdir(parents=True, exist_ok=True)
    rendered_frames = []
    for frame_index in frames:
        frame_points = [point for point in points if int(point["frame_index"]) <= frame_index]
        scratch_heat = frames_dir / "_heat_frame_{:03d}.png".format(frame_index)
        output_frame = frames_dir / "frame_{:03d}_heatmap.png".format(frame_index)
        save_heatmap_images(frame_points, template_image, scratch_heat, output_frame, sigma, alpha, point_weight)
        frame = Image.open(output_frame).convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.text((12, 12), "frame <= {:03d}".format(frame_index), fill=(0, 0, 0))
        frame.save(output_frame)
        rendered_frames.append(output_frame)
        scratch_heat.unlink(missing_ok=True)
    gif = write_gif(rendered_frames, gif_path, gif_fps) if make_gif else None
    mp4 = None
    mp4_error = None
    if make_mp4:
        mp4, mp4_error = write_mp4(rendered_frames, mp4_path, mp4_fps)
    return {
        "status": "complete",
        "frames": [str(path) for path in rendered_frames],
        "frames_dir": str(frames_dir),
        "gif": gif,
        "mp4": mp4,
        "mp4_error": mp4_error,
    }


def main():
    args = parse_args()
    projected_path = Path(args.projected_points) if args.projected_points else PROJECT_ROOT / "outputs" / args.run_id / "field_calibration_smoke" / "projected_player_points.json"
    calibration_metadata_path = Path(args.calibration_metadata) if args.calibration_metadata else PROJECT_ROOT / "outputs" / args.run_id / "field_calibration_smoke" / "calibration_metadata.json"
    field_template_path = project_path(args.field_template)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "heatmap_smoke"
    by_track_dir = output_dir / "heatmap_by_track"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_track_dir.mkdir(parents=True, exist_ok=True)

    projected_payload = load_json(projected_path)
    all_points = projected_payload.get("points", [])
    calibration_metadata = load_json(calibration_metadata_path) if calibration_metadata_path.exists() else None
    template = Image.open(field_template_path).convert("RGB")
    in_bounds = inside_points(all_points)
    included_points, tracks, included_tracks = filter_tracks(in_bounds, args.min_track_points)
    tracks_filtered_out = sorted(int(track_id) for track_id, rows in tracks.items() if len(rows) < args.min_track_points)

    all_heat = save_heatmap_images(
        included_points,
        template,
        output_dir / "heatmap_all_players.png",
        output_dir / "heatmap_all_players_overlay.png",
        args.sigma,
        args.alpha,
        args.point_weight,
    )
    trails_path = save_track_point_overlay(included_points, template, output_dir / "projected_points_used_for_heatmap.png")

    per_track = []
    for track_id, rows in sorted(group_by_track(included_points).items()):
        if len(rows) < args.per_track_min_points:
            per_track.append({"track_id": int(track_id), "status": "skipped", "reason": "too_few_points", "point_count": len(rows)})
            continue
        heat_path = by_track_dir / "track_{:03d}_heatmap.png".format(track_id)
        overlay_path = by_track_dir / "track_{:03d}_heatmap_overlay.png".format(track_id)
        stats = save_heatmap_images(rows, template, heat_path, overlay_path, args.sigma, args.alpha, args.point_weight)
        per_track.append({"track_id": int(track_id), "status": "written", "point_count": len(rows), **stats})

    by_frame_animation = write_by_frame_animation(
        included_points,
        template,
        output_dir / "frames",
        output_dir / "heatmap_by_frame.gif",
        output_dir / "heatmap_by_frame.mp4",
        args.sigma,
        args.alpha,
        args.point_weight,
        args.gif_fps,
        args.mp4_fps if args.mp4_fps else args.gif_fps,
        make_gif=args.make_by_frame_gif,
        make_mp4=args.make_by_frame_mp4,
    )

    metadata = {
        "status": "complete",
        "stage": "heatmap_smoke",
        "run_id": args.run_id,
        "inputs": {
            "projected_points": str(projected_path),
            "field_template": str(field_template_path),
            "calibration_metadata": str(calibration_metadata_path),
        },
        "parameters": {
            "min_track_points": args.min_track_points,
            "per_track_min_points": args.per_track_min_points,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "point_weight": args.point_weight,
            "make_by_frame_gif": args.make_by_frame_gif,
            "make_by_frame_mp4": args.make_by_frame_mp4,
            "gif_fps": args.gif_fps,
            "mp4_fps": args.mp4_fps if args.mp4_fps else args.gif_fps,
        },
        "calibration_status": calibration_metadata.get("status") if calibration_metadata else None,
        "tracks": {
            "all_track_ids_in_bounds": sorted(int(track_id) for track_id in tracks),
            "included_track_ids": sorted(int(track_id) for track_id in included_tracks),
            "filtered_out_track_ids": tracks_filtered_out,
        },
        "per_track_heatmaps": per_track,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "heatmap_all_players": all_heat["heatmap"],
            "heatmap_all_players_overlay": all_heat["overlay"],
            "projected_points_used_for_heatmap": trails_path,
            "heatmap_by_track_dir": str(by_track_dir),
            "heatmap_animation_frames_dir": by_frame_animation.get("frames_dir"),
            "heatmap_animation_frames": by_frame_animation.get("frames", []),
            "heatmap_by_frame_gif": by_frame_animation.get("gif"),
            "heatmap_by_frame_mp4": by_frame_animation.get("mp4"),
            "heatmap_by_frame_mp4_error": by_frame_animation.get("mp4_error"),
        },
    }
    summary = {
        "status": "complete",
        "stage": "heatmap_smoke_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "heatmap_metadata.json"),
        "counts": {
            "total_projected_points": len(all_points),
            "points_inside_bounds": len(in_bounds),
            "points_outside_bounds": len(all_points) - len(in_bounds),
            "points_included_after_track_filter": len(included_points),
            "number_of_tracks": len(tracks),
            "tracks_included": len(included_tracks),
            "tracks_filtered_out": len(tracks_filtered_out),
            "per_track_heatmaps_written": sum(1 for row in per_track if row["status"] == "written"),
            "per_track_heatmaps_skipped": sum(1 for row in per_track if row["status"] == "skipped"),
            "heatmap_animation_frames": len(by_frame_animation.get("frames", [])),
        },
        "mp4_status": "complete" if by_frame_animation.get("mp4") else "failed" if by_frame_animation.get("mp4_error") else "skipped",
        "mp4_error": by_frame_animation.get("mp4_error"),
        "tracks_included": sorted(int(track_id) for track_id in included_tracks),
        "tracks_filtered_out": tracks_filtered_out,
        "artifacts": metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(output_dir / "heatmap_metadata.json", metadata)
    write_json(output_dir / "heatmap_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
