#!/usr/bin/env python3
"""Project tracked foot points with multiple calibration keyframe homographies."""

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
    "Nearest-keyframe homography selection is a first smoke implementation; interpolation is not enabled yet.",
    "Hard camera cuts should be represented as separate keyframe/view segments, not interpolated.",
    "Broadcast camera pan/zoom between keyframes may still create projection drift.",
    "Foot points are noisy.",
    "Tracking IDs may still switch.",
    "Referee may still be included.",
    "This is a multi-homography smoke projection, not final field calibration.",
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
    parser = argparse.ArgumentParser(description="Project tracking points using multiple homography calibration keyframes.")
    parser.add_argument("--run-id", default="nll_test4", help="Prototype 4 run id.")
    parser.add_argument("--config", default="configs/nll_test4_multi_homography_points.json", help="Multi-homography config JSON.")
    parser.add_argument("--tracking-metadata", default="outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/tracking_metadata.json", help="Tracking metadata JSON to project.")
    parser.add_argument("--output-dir", default="outputs/nll_test4/multi_homography_demos/segment_20s_10s_multi_homography", help="Output directory for multi-homography projection artifacts.")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum point pairs required per keyframe.")
    parser.add_argument("--max-keyframe-distance", type=int, default=18, help="Warn when a tracking frame is this many sampled frames from its nearest calibration keyframe.")
    parser.add_argument("--assignment", choices=["nearest"], default="nearest", help="Frame-to-homography assignment strategy. Only nearest is implemented for this smoke stage.")
    parser.add_argument("--point-radius", type=int, default=5, help="Rendered point radius.")
    parser.add_argument("--trail-width", type=int, default=3, help="Rendered track trail width.")
    parser.add_argument("--gif-fps", type=float, default=6.0, help="Top-down by-frame GIF FPS.")
    return parser.parse_args()


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def color_for(track_id):
    return COLORS[(int(track_id) - 1) % len(COLORS)]


def is_zero_pair(values):
    return len(values) == 2 and float(values[0]) == 0.0 and float(values[1]) == 0.0


def normalize_keyframe_points(keyframe):
    if keyframe.get("points"):
        return keyframe["points"]
    video_points = keyframe.get("video_points", [])
    template_points = keyframe.get("template_points") or keyframe.get("field_points", [])
    names = keyframe.get("point_names", [])
    points = []
    for index, (video_xy, field_xy) in enumerate(zip(video_points, template_points)):
        name = names[index] if index < len(names) else "point_{:02d}".format(index + 1)
        points.append({"name": name, "video_xy": video_xy, "field_xy": field_xy})
    return points


def validate_points(points, min_points, keyframe_name):
    errors = []
    if len(points) < min_points:
        errors.append("{} needs at least {} point correspondences; found {}.".format(keyframe_name, min_points, len(points)))
    names = set()
    for index, point in enumerate(points):
        name = point.get("name", "point_{}".format(index))
        if name in names:
            errors.append("{} has duplicate calibration point name: {}".format(keyframe_name, name))
        names.add(name)
        video_xy = point.get("video_xy")
        field_xy = point.get("field_xy")
        if not isinstance(video_xy, list) or len(video_xy) != 2:
            errors.append("{}:{} has invalid video_xy; expected [x, y].".format(keyframe_name, name))
            continue
        if not isinstance(field_xy, list) or len(field_xy) != 2:
            errors.append("{}:{} has invalid field_xy; expected [x, y].".format(keyframe_name, name))
            continue
        if is_zero_pair(video_xy):
            errors.append("{}:{} video_xy is still placeholder [0, 0].".format(keyframe_name, name))
        if is_zero_pair(field_xy):
            errors.append("{}:{} field_xy is still placeholder [0, 0].".format(keyframe_name, name))
    if errors:
        raise ValueError("Invalid multi-homography config: " + " ".join(errors))


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


def keyframe_frame_index(keyframe):
    if keyframe.get("frame_index") is None:
        raise ValueError("Keyframe {} is missing frame_index.".format(keyframe.get("name", "unnamed")))
    return int(keyframe["frame_index"])


def prepare_keyframes(config, min_points, output_dir, point_radius):
    prepared = []
    raw_keyframes = config.get("keyframes", [])
    if not raw_keyframes:
        raise ValueError("Multi-homography config must contain at least one keyframe.")
    field_template = project_path(config["field_template"])
    for index, keyframe in enumerate(raw_keyframes):
        name = keyframe.get("name") or "keyframe_{:02d}".format(index + 1)
        points = normalize_keyframe_points(keyframe)
        validate_points(points, min_points, name)
        matrix, inliers = compute_homography(points)
        matrix_path = output_dir / "homography_{}.npy".format(name)
        np.save(matrix_path, matrix)
        video_frame = project_path(keyframe["frame_path"]) if keyframe.get("frame_path") else None
        video_diag = None
        if video_frame and video_frame.exists():
            video_diag = draw_calibration_points(video_frame, points, "video_xy", output_dir / "calibration_points_{}_video.png".format(name), point_radius)
        template_diag = draw_calibration_points(field_template, points, "field_xy", output_dir / "calibration_points_{}_template.png".format(name), point_radius)
        prepared.append({
            "name": name,
            "frame_index": keyframe_frame_index(keyframe),
            "frame_path": rel_path(video_frame) if video_frame else keyframe.get("frame_path"),
            "points": points,
            "homography_matrix": matrix,
            "homography_matrix_path": str(matrix_path),
            "homography_inliers": inliers.flatten().astype(int).tolist() if inliers is not None else None,
            "video_diagnostic": video_diag,
            "template_diagnostic": template_diag,
            "view_segment": keyframe.get("view_segment"),
            "interpolate": bool(keyframe.get("interpolate", False)),
        })
    prepared.sort(key=lambda row: row["frame_index"])
    return prepared


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


def assign_keyframe(frame_index, keyframes):
    return min(keyframes, key=lambda row: (abs(int(frame_index) - int(row["frame_index"])), int(row["frame_index"])))


def extract_projected_points(tracking_metadata, keyframes, template_size, max_keyframe_distance):
    width, height = template_size
    projected = []
    frame_summary = {}
    homography_summary = {row["name"]: {"frames": set(), "points": 0, "inside": 0, "outside": 0} for row in keyframes}
    far_frames = set()
    for det in tracking_metadata.get("detections", []):
        foot = det.get("foot_point_2d")
        if not foot:
            continue
        frame_index = int(det["frame_index"])
        keyframe = assign_keyframe(frame_index, keyframes)
        distance = abs(frame_index - int(keyframe["frame_index"]))
        if distance > max_keyframe_distance:
            far_frames.add(frame_index)
        projected_xy = transform_point(keyframe["homography_matrix"], [foot["x"], foot["y"]])
        inside = bool(0.0 <= projected_xy[0] < width and 0.0 <= projected_xy[1] < height)
        row = {
            "frame_index": frame_index,
            "track_id": int(det["track_id"]),
            "mask_id": int(det.get("mask_id", -1)),
            "video_foot_point": {"x": float(foot["x"]), "y": float(foot["y"])},
            "projected_field_point": {"x": projected_xy[0], "y": projected_xy[1]},
            "inside_field_template_bounds": inside,
            "homography_keyframe": keyframe["name"],
            "homography_keyframe_frame_index": int(keyframe["frame_index"]),
            "homography_frame_distance": int(distance),
            "source_frame_metadata": det.get("source_frame_metadata"),
            "match_score": det.get("match_score"),
            "match_source": det.get("match_source"),
        }
        projected.append(row)
        frame_row = frame_summary.setdefault(frame_index, {"frame_index": frame_index, "homography_keyframe": keyframe["name"], "homography_frame_distance": int(distance), "points": 0, "inside": 0, "outside": 0})
        frame_row["points"] += 1
        if inside:
            frame_row["inside"] += 1
        else:
            frame_row["outside"] += 1
        hrow = homography_summary[keyframe["name"]]
        hrow["frames"].add(frame_index)
        hrow["points"] += 1
        if inside:
            hrow["inside"] += 1
        else:
            hrow["outside"] += 1
    projected.sort(key=lambda item: (item["frame_index"], item["track_id"], item["mask_id"]))
    for row in homography_summary.values():
        row["frames"] = sorted(int(item) for item in row["frames"])
        row["frame_count"] = len(row["frames"])
    return projected, dict(sorted(frame_summary.items())), homography_summary, sorted(int(item) for item in far_frames)


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
            if not row["inside_field_template_bounds"]:
                continue
            x = int(round(float(row["projected_field_point"]["x"])))
            y = int(round(float(row["projected_field_point"]["y"])))
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
        scratch = output_path.parent / "_multi_homography_frame_{:03d}.png".format(frame_index)
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


def serializable_keyframes(keyframes):
    rows = []
    for row in keyframes:
        rows.append({
            "name": row["name"],
            "frame_index": row["frame_index"],
            "frame_path": row["frame_path"],
            "points": row["points"],
            "homography_matrix": row["homography_matrix"].tolist(),
            "homography_matrix_path": row["homography_matrix_path"],
            "homography_inliers": row["homography_inliers"],
            "video_diagnostic": row["video_diagnostic"],
            "template_diagnostic": row["template_diagnostic"],
            "view_segment": row.get("view_segment"),
            "interpolate": row.get("interpolate", False),
        })
    return rows


def main():
    args = parse_args()
    config_path = project_path(args.config)
    tracking_metadata_path = project_path(args.tracking_metadata)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "stage": "multi_homography_projection",
        "run_id": args.run_id,
        "inputs": {"config": str(config_path), "tracking_metadata": str(tracking_metadata_path)},
        "parameters": {"assignment": args.assignment, "max_keyframe_distance": args.max_keyframe_distance, "min_points": args.min_points},
        "known_limitations": KNOWN_LIMITATIONS,
    }
    try:
        if not config_path.exists():
            raise FileNotFoundError("Multi-homography config does not exist: {}".format(config_path))
        if not tracking_metadata_path.exists():
            raise FileNotFoundError("Tracking metadata does not exist: {}".format(tracking_metadata_path))
        config = load_json(config_path)
        field_template = project_path(config["field_template"])
        if not field_template.exists():
            raise FileNotFoundError("Configured field_template does not exist: {}".format(field_template))
        tracking_metadata = load_json(tracking_metadata_path)
        template_image = Image.open(field_template).convert("RGB")
        keyframes = prepare_keyframes(config, args.min_points, output_dir, args.point_radius)
        projected, by_frame, by_homography, far_frames = extract_projected_points(tracking_metadata, keyframes, template_image.size, args.max_keyframe_distance)
        write_json(output_dir / "projected_player_points.json", {"points": projected})
        write_json(output_dir / "projection_summary_by_frame.json", {"frames": [row for _, row in sorted(by_frame.items())]})
        write_json(output_dir / "homography_assignment_summary.json", {"homographies": by_homography, "far_frames": far_frames})
        topdown = draw_projected_tracks(field_template, projected, output_dir / "projected_tracks_topdown.png", args.point_radius, args.trail_width)
        topdown_gif = write_topdown_gif(field_template, projected, output_dir / "projected_tracks_topdown_by_frame.gif", args.point_radius, args.trail_width, args.gif_fps)
        metadata.update({
            "status": "complete",
            "config": config,
            "field_template": str(field_template),
            "keyframes": serializable_keyframes(keyframes),
            "projected_points": projected,
            "projection_summary_by_frame": [row for _, row in sorted(by_frame.items())],
            "homography_assignment_summary": by_homography,
            "far_frames": far_frames,
            "artifacts": {
                "projected_player_points": str(output_dir / "projected_player_points.json"),
                "projection_summary_by_frame": str(output_dir / "projection_summary_by_frame.json"),
                "homography_assignment_summary": str(output_dir / "homography_assignment_summary.json"),
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

    write_json(output_dir / "multi_homography_projection_metadata.json", metadata)
    projected_points = metadata.get("projected_points", [])
    by_homography = metadata.get("homography_assignment_summary", {})
    summary = {
        "status": metadata["status"],
        "stage": "multi_homography_projection_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "multi_homography_projection_metadata.json"),
        "counts": {
            "keyframes": len(metadata.get("keyframes", [])),
            "projected_points": len(projected_points),
            "projected_points_inside_template": sum(1 for point in projected_points if point.get("inside_field_template_bounds")),
            "projected_points_outside_template": sum(1 for point in projected_points if not point.get("inside_field_template_bounds")),
            "projected_tracks": len({point["track_id"] for point in projected_points}),
            "frames_far_from_keyframe": len(metadata.get("far_frames", [])),
        },
        "homography_assignment_summary": by_homography,
        "far_frames": metadata.get("far_frames", []),
        "artifacts": metadata.get("artifacts", {}),
        "known_limitations": KNOWN_LIMITATIONS,
    }
    if metadata.get("error"):
        summary["error"] = metadata["error"]
    write_json(output_dir / "multi_homography_projection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
