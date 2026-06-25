#!/usr/bin/env python3
"""Project stabilized player foot points onto a top-down field template via homography."""

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

KNOWN_LIMITATIONS = [
    "Manual calibration points are approximate.",
    "Broadcast camera perspective may change.",
    "Foot points are noisy.",
    "Tracking IDs may still switch.",
    "Referee may still be included.",
    "This is only a field-calibration smoke test, not final heatmap tracking.",
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
    (0, 114, 178),
    (213, 94, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a manual homography field-calibration smoke test.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--config", default="configs/nll_field_homography_points.json", help="Manual homography point config JSON.")
    parser.add_argument("--tracking-metadata", default=None, help="Override stabilized tracking metadata path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum real point correspondences required.")
    parser.add_argument("--point-radius", type=int, default=5, help="Rendered point radius.")
    parser.add_argument("--trail-width", type=int, default=3, help="Rendered track trail width.")
    parser.add_argument("--gif-fps", type=float, default=6.0, help="Top-down by-frame GIF FPS.")
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


def is_zero_pair(values):
    return len(values) == 2 and float(values[0]) == 0.0 and float(values[1]) == 0.0


def validate_points(points, min_points):
    errors = []
    if len(points) < min_points:
        errors.append("At least {} point correspondences are required; found {}.".format(min_points, len(points)))
    names = set()
    for index, point in enumerate(points):
        name = point.get("name", "point_{}".format(index))
        if name in names:
            errors.append("Duplicate calibration point name: {}".format(name))
        names.add(name)
        video_xy = point.get("video_xy")
        field_xy = point.get("field_xy")
        if not isinstance(video_xy, list) or len(video_xy) != 2:
            errors.append("{} has invalid video_xy; expected [x, y].".format(name))
            continue
        if not isinstance(field_xy, list) or len(field_xy) != 2:
            errors.append("{} has invalid field_xy; expected [x, y].".format(name))
            continue
        if is_zero_pair(video_xy):
            errors.append("{} video_xy is still placeholder [0, 0].".format(name))
        if is_zero_pair(field_xy):
            errors.append("{} field_xy is still placeholder [0, 0].".format(name))
    if errors:
        raise ValueError("Invalid homography config: " + " ".join(errors))


def draw_calibration_points(image_path, points, key, output_path, point_radius):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, point in enumerate(points):
        x, y = point[key]
        color = color_for(index + 1)
        x = int(round(float(x)))
        y = int(round(float(y)))
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color, outline=(255, 255, 255), width=2)
        draw.text((x + point_radius + 3, y - point_radius), point.get("name", str(index)), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def compute_homography(points):
    video = np.asarray([point["video_xy"] for point in points], dtype=np.float32)
    field = np.asarray([point["field_xy"] for point in points], dtype=np.float32)
    matrix, inliers = cv2.findHomography(video, field, method=0)
    if matrix is None:
        raise RuntimeError("cv2.findHomography returned None; check point correspondences.")
    return matrix.astype(np.float64), inliers


def transform_point(matrix, xy):
    src = np.asarray([[[float(xy[0]), float(xy[1])]]], dtype=np.float64)
    dst = cv2.perspectiveTransform(src, matrix)[0, 0]
    return [float(dst[0]), float(dst[1])]


def extract_projected_points(tracking_metadata, matrix, template_size):
    width, height = template_size
    projected = []
    for det in tracking_metadata.get("detections", []):
        foot = det.get("foot_point_2d")
        if not foot:
            continue
        projected_xy = transform_point(matrix, [foot["x"], foot["y"]])
        inside = bool(0.0 <= projected_xy[0] < width and 0.0 <= projected_xy[1] < height)
        projected.append({
            "frame_index": int(det["frame_index"]),
            "track_id": int(det["track_id"]),
            "mask_id": int(det.get("mask_id", -1)),
            "video_foot_point": {"x": float(foot["x"]), "y": float(foot["y"])},
            "projected_field_point": {"x": projected_xy[0], "y": projected_xy[1]},
            "inside_field_template_bounds": inside,
            "source_frame_metadata": det.get("source_frame_metadata"),
            "match_score": det.get("match_score"),
            "match_source": det.get("match_source"),
        })
    projected.sort(key=lambda item: (item["frame_index"], item["track_id"], item["mask_id"]))
    return projected


def grouped_by_track(points):
    tracks = {}
    for point in points:
        tracks.setdefault(int(point["track_id"]), []).append(point)
    for rows in tracks.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return tracks


def draw_projected_tracks(template_path, projected_points, output_path, point_radius, trail_width, max_frame=None):
    image = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if max_frame is not None:
        points = [point for point in projected_points if point["frame_index"] <= max_frame]
    else:
        points = list(projected_points)
    tracks = grouped_by_track(points)
    for track_id, rows in sorted(tracks.items()):
        color = color_for(track_id)
        coords = [(float(row["projected_field_point"]["x"]), float(row["projected_field_point"]["y"])) for row in rows if row["inside_field_template_bounds"]]
        if len(coords) >= 2:
            draw.line([(int(round(x)), int(round(y))) for x, y in coords], fill=color, width=trail_width)
        for row in rows:
            x = int(round(float(row["projected_field_point"]["x"])))
            y = int(round(float(row["projected_field_point"]["y"])))
            if not row["inside_field_template_bounds"]:
                continue
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color, outline=(255, 255, 255), width=1)
        if coords:
            x, y = coords[-1]
            draw.text((int(round(x)) + point_radius + 2, int(round(y)) - point_radius), "T{}".format(track_id), fill=color)
    if max_frame is not None:
        draw.text((12, 12), "frame <= {:03d}".format(max_frame), fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def write_topdown_gif(template_path, projected_points, output_path, point_radius, trail_width, fps):
    frames = sorted({int(point["frame_index"]) for point in projected_points})
    if not frames:
        return None
    images = []
    for frame_index in frames:
        scratch = output_path.parent / "_gif_frame_{:03d}.png".format(frame_index)
        draw_projected_tracks(template_path, projected_points, scratch, point_radius, trail_width, max_frame=frame_index)
        images.append(Image.open(scratch).convert("P", palette=Image.ADAPTIVE))
        scratch.unlink(missing_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    try:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    finally:
        for image in images:
            image.close()
    return str(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "field_calibration_smoke"
    tracking_metadata_path = Path(args.tracking_metadata) if args.tracking_metadata else PROJECT_ROOT / "outputs" / args.run_id / "short_clip_tracking_stabilized" / "tracking_metadata.json"
    config_path = project_path(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "stage": "field_calibration_smoke",
        "run_id": args.run_id,
        "inputs": {"tracking_metadata": str(tracking_metadata_path), "config": str(config_path)},
        "known_limitations": KNOWN_LIMITATIONS,
    }
    try:
        config = load_json(config_path)
        points = config.get("points", [])
        validate_points(points, args.min_points)
        video_frame = project_path(config["video_frame"])
        field_template = project_path(config["field_template"])
        if not video_frame.exists():
            raise FileNotFoundError("Configured video_frame does not exist: {}".format(video_frame))
        if not field_template.exists():
            raise FileNotFoundError("Configured field_template does not exist: {}".format(field_template))
        if not tracking_metadata_path.exists():
            raise FileNotFoundError("Tracking metadata does not exist: {}".format(tracking_metadata_path))

        matrix, inliers = compute_homography(points)
        np.save(output_dir / "homography_matrix.npy", matrix)
        video_diag = draw_calibration_points(video_frame, points, "video_xy", output_dir / "calibration_points_video.png", args.point_radius)
        template_diag = draw_calibration_points(field_template, points, "field_xy", output_dir / "calibration_points_template.png", args.point_radius)
        tracking_metadata = load_json(tracking_metadata_path)
        template_image = Image.open(field_template).convert("RGB")
        projected = extract_projected_points(tracking_metadata, matrix, template_image.size)
        write_json(output_dir / "projected_player_points.json", {"points": projected})
        topdown = draw_projected_tracks(field_template, projected, output_dir / "projected_tracks_topdown.png", args.point_radius, args.trail_width)
        topdown_gif = write_topdown_gif(field_template, projected, output_dir / "projected_tracks_topdown_by_frame.gif", args.point_radius, args.trail_width, args.gif_fps)
        metadata.update({
            "status": "complete",
            "config": config,
            "homography_matrix": matrix.tolist(),
            "homography_inliers": inliers.flatten().astype(int).tolist() if inliers is not None else None,
            "projected_points": projected,
            "artifacts": {
                "homography_matrix": str(output_dir / "homography_matrix.npy"),
                "calibration_points_video": video_diag,
                "calibration_points_template": template_diag,
                "projected_player_points": str(output_dir / "projected_player_points.json"),
                "projected_tracks_topdown": topdown,
                "projected_tracks_topdown_by_frame_gif": topdown_gif,
            },
        })
        exit_code = 0
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "error": {"type": exc.__class__.__name__, "message": str(exc), "traceback": traceback.format_exc()},
        })
        exit_code = 1

    write_json(output_dir / "calibration_metadata.json", metadata)
    projected_points = metadata.get("projected_points", [])
    summary = {
        "status": metadata["status"],
        "stage": "field_calibration_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "calibration_metadata.json"),
        "counts": {
            "calibration_points": len(metadata.get("config", {}).get("points", [])) if metadata.get("config") else 0,
            "projected_points": len(projected_points),
            "projected_points_inside_template": sum(1 for point in projected_points if point.get("inside_field_template_bounds")),
            "projected_tracks": len({point["track_id"] for point in projected_points}),
        },
        "artifacts": metadata.get("artifacts", {}),
        "known_limitations": KNOWN_LIMITATIONS,
    }
    if metadata.get("error"):
        summary["error"] = metadata["error"]
    write_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
