#!/usr/bin/env python3
"""Convert Andrew lacrosse_sam floor_xy_ft outputs into Prototype 4 heatmap points."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

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

SUPPORTED_SCHEMAS = {
    "vggt_birds_eye_v1",
    "cotracker_vggt_player_trajectories_v1",
    "sam_body4d_vggt_birds_eye_v1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapt Andrew lacrosse_sam floor_xy_ft player locations into Prototype 4 projected_player_points.json."
    )
    parser.add_argument("--andrew-json", default=None, help="Andrew output JSON with frames[].players[].floor_xy_ft.")
    parser.add_argument("--field-template", default="assets/field_templates/nll_field_topdown.png", help="Top-down NLL field template image.")
    parser.add_argument("--output-dir", default="outputs/adapter_smoke_test", help="Output directory for adapter artifacts.")
    parser.add_argument("--run-id", default="andrew_adapter_smoke", help="Run id recorded in metadata.")
    parser.add_argument("--origin", choices=["center", "top_left", "bottom_left"], default="top_left", help="Origin convention used by Andrew floor_xy_ft.")
    parser.add_argument("--center-y-axis", choices=["up", "down"], default="up", help="For --origin center, direction of positive y in floor_xy_ft.")
    parser.add_argument("--field-length-ft", type=float, default=200.0, help="NLL field/rink length in feet.")
    parser.add_argument("--field-width-ft", type=float, default=85.0, help="NLL field/rink width in feet.")
    parser.add_argument(
        "--template-field-bounds-px",
        default=None,
        help="Playable field bounds inside template as x0,y0,x1,y1. Defaults to the full template image.",
    )
    parser.add_argument("--source-video-metadata", default=None, help="Optional metadata path to record for traceability.")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Force creation/use of a tiny synthetic Andrew-style input JSON.")
    parser.add_argument("--run-heatmap", action="store_true", help="Run scripts/run_heatmap_smoke.py on the converted points after conversion.")
    parser.add_argument("--heatmap-output-dir", default=None, help="Output dir for optional heatmap run. Defaults to <output-dir>/heatmap.")
    parser.add_argument("--heatmap-min-track-points", type=int, default=1, help="Minimum track points passed to heatmap smoke.")
    return parser.parse_args()


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bounds(value, template_size):
    width, height = template_size
    if not value:
        return (0.0, 0.0, float(width), float(height))
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--template-field-bounds-px must be x0,y0,x1,y1.")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid template field bounds: x1/y1 must be greater than x0/y0.")
    return (x0, y0, x1, y1)


def color_for(track_id):
    return COLORS[(int(track_id) - 1) % len(COLORS)]


def create_synthetic_andrew_json(path):
    payload = {
        "schema": "prototype4_synthetic_andrew_floor_xy_v1",
        "synthetic": True,
        "description": "Synthetic adapter smoke input. These are not real player detections or model outputs.",
        "frames": [
            {
                "frame": 0,
                "timestamp_sec": 0.0,
                "players": [
                    {"object_id": 1, "floor_xy_ft": [95.0, 38.0], "source": "synthetic"},
                    {"object_id": 2, "floor_xy_ft": [103.0, 48.0], "source": "synthetic"},
                ],
            },
            {
                "frame": 1,
                "timestamp_sec": 0.2,
                "players": [
                    {"object_id": 1, "floor_xy_ft": [98.0, 39.0], "source": "synthetic"},
                    {"object_id": 2, "floor_xy_ft": [106.0, 49.0], "source": "synthetic"},
                ],
            },
            {
                "frame": 2,
                "timestamp_sec": 0.4,
                "players": [
                    {"object_id": 1, "floor_xy_ft": [101.0, 40.0], "source": "synthetic"},
                    {"object_id": 2, "floor_xy_ft": [109.0, 50.0], "source": "synthetic"},
                ],
            },
        ],
    }
    write_json(path, payload)
    return payload


def schema_name(payload):
    schema = payload.get("schema")
    if schema:
        return str(schema)
    if payload.get("frames") and all("players" in frame for frame in payload.get("frames", [])):
        return "unknown_frames_players_floor_xy"
    return "unknown"


def iter_players(payload):
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Andrew JSON must contain a top-level frames list.")
    for frame_row in frames:
        frame_index = frame_row.get("frame", frame_row.get("frame_index"))
        if frame_index is None:
            continue
        players = frame_row.get("players", [])
        if not isinstance(players, list):
            continue
        for player_index, player in enumerate(players):
            floor_xy = player.get("floor_xy_ft")
            if not floor_xy or len(floor_xy) != 2:
                continue
            object_id = player.get("object_id", player.get("track_id", player.get("id", player_index + 1)))
            yield int(frame_index), frame_row, int(object_id), player_index, player, [float(floor_xy[0]), float(floor_xy[1])]


def canonical_field_xy(floor_xy, origin, field_length, field_width, center_y_axis):
    x, y = float(floor_xy[0]), float(floor_xy[1])
    if origin == "top_left":
        return x, y
    if origin == "bottom_left":
        return x, field_width - y
    if origin == "center":
        field_x = x + field_length / 2.0
        if center_y_axis == "up":
            field_y = field_width / 2.0 - y
        else:
            field_y = y + field_width / 2.0
        return field_x, field_y
    raise ValueError("Unsupported origin: {}".format(origin))


def field_to_template_px(canonical_xy, bounds, field_length, field_width):
    field_x, field_y = canonical_xy
    x0, y0, x1, y1 = bounds
    px = x0 + (field_x / field_length) * (x1 - x0)
    py = y0 + (field_y / field_width) * (y1 - y0)
    return px, py


def convert_payload(payload, args, template_size, bounds):
    points = []
    skipped = 0
    for frame_index, frame_row, object_id, player_index, player, floor_xy in iter_players(payload):
        field_x, field_y = canonical_field_xy(floor_xy, args.origin, args.field_length_ft, args.field_width_ft, args.center_y_axis)
        px, py = field_to_template_px([field_x, field_y], bounds, args.field_length_ft, args.field_width_ft)
        inside_field_ft = bool(0.0 <= field_x <= args.field_length_ft and 0.0 <= field_y <= args.field_width_ft)
        inside_template = bool(bounds[0] <= px <= bounds[2] and bounds[1] <= py <= bounds[3])
        timestamp = frame_row.get("timestamp_sec", frame_row.get("time_sec", frame_row.get("timestamp")))
        point = {
            "frame_index": int(frame_index),
            "track_id": int(object_id),
            "mask_id": int(player.get("mask_id", player_index)),
            "projected_field_point": {"x": float(px), "y": float(py)},
            "inside_field_template_bounds": bool(inside_field_ft and inside_template),
            "field_xy_ft": {"x": float(field_x), "y": float(field_y)},
            "source_floor_xy_ft": {"x": float(floor_xy[0]), "y": float(floor_xy[1])},
            "source": player.get("source", schema_name(payload)),
            "source_object_id": int(object_id),
        }
        if timestamp is not None:
            point["timestamp_sec"] = float(timestamp)
        for key in [
            "team",
            "team_color",
            "color",
            "camera_xy",
            "visible_track_points",
            "mask_gated_track_points",
            "vggt_points",
            "sampled_points",
            "median_depth_conf",
            "mesh_foot_pixels",
            "mesh_fit_points",
            "mesh_render_points",
        ]:
            if key in player:
                point[key] = player[key]
        points.append(point)
    points.sort(key=lambda row: (int(row["frame_index"]), int(row["track_id"]), int(row.get("mask_id", 0))))
    return {"points": points}, skipped


def summarize(payload, points, args, bounds, andrew_json_path, synthetic):
    field_x = [point["field_xy_ft"]["x"] for point in points]
    field_y = [point["field_xy_ft"]["y"] for point in points]
    frames = sorted({int(point["frame_index"]) for point in points})
    tracks = sorted({int(point["track_id"]) for point in points})
    inside = [point for point in points if point.get("inside_field_template_bounds")]
    outside = [point for point in points if not point.get("inside_field_template_bounds")]
    return {
        "status": "complete",
        "stage": "andrew_floor_xy_adapter",
        "run_id": args.run_id,
        "synthetic": bool(synthetic),
        "supported_schemas": sorted(SUPPORTED_SCHEMAS),
        "detected_schema": schema_name(payload),
        "inputs": {
            "andrew_json": str(andrew_json_path),
            "field_template": str(project_path(args.field_template)),
            "source_video_metadata": str(project_path(args.source_video_metadata)) if args.source_video_metadata else None,
        },
        "parameters": {
            "origin": args.origin,
            "center_y_axis": args.center_y_axis,
            "field_length_ft": args.field_length_ft,
            "field_width_ft": args.field_width_ft,
            "template_field_bounds_px": list(bounds),
        },
        "counts": {
            "frames": len(frames),
            "player_points": len(points),
            "points_inside_field_bounds": len(inside),
            "points_outside_field_bounds": len(outside),
            "tracks_found": len(tracks),
        },
        "ranges_ft": {
            "x_min": min(field_x) if field_x else None,
            "x_max": max(field_x) if field_x else None,
            "y_min": min(field_y) if field_y else None,
            "y_max": max(field_y) if field_y else None,
        },
        "frames": frames,
        "tracks": tracks,
        "notes": [
            "Manual homography is not required for this adapter.",
            "Coordinates are converted from Andrew floor_xy_ft into the current Prototype 4 heatmap pixel format.",
            "Synthetic outputs are structural smoke tests only and are not real model results." if synthetic else "Input is treated as external Andrew pipeline output.",
        ],
    }


def draw_preview(template_path, points, output_path, with_tracks):
    image = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    by_track = {}
    for point in points:
        if not point.get("inside_field_template_bounds"):
            continue
        by_track.setdefault(int(point["track_id"]), []).append(point)
    for rows in by_track.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    for track_id, rows in sorted(by_track.items()):
        color = color_for(track_id)
        coords = []
        for row in rows:
            xy = row["projected_field_point"]
            coords.append((float(xy["x"]), float(xy["y"])))
        if with_tracks and len(coords) >= 2:
            draw.line([(int(round(x)), int(round(y))) for x, y in coords], fill=color, width=3)
        for x, y in coords:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=1)
        if coords:
            x, y = coords[-1]
            draw.text((int(round(x)) + 7, int(round(y)) - 7), "T{}".format(track_id), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def run_heatmap(args, output_dir, projected_path, summary_path):
    heatmap_dir = project_path(args.heatmap_output_dir) if args.heatmap_output_dir else output_dir / "heatmap"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_heatmap_smoke.py"),
        "--run-id",
        args.run_id,
        "--projected-points",
        str(projected_path),
        "--field-template",
        str(project_path(args.field_template)),
        "--calibration-metadata",
        str(summary_path),
        "--output-dir",
        str(heatmap_dir),
        "--min-track-points",
        str(args.heatmap_min_track_points),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output_dir": str(heatmap_dir),
    }


def main():
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = project_path(args.field_template)
    if not template_path.exists():
        raise SystemExit("Missing field template: {}".format(template_path))
    template = Image.open(template_path).convert("RGB")
    bounds = parse_bounds(args.template_field_bounds_px, template.size)

    synthetic = bool(args.synthetic_smoke or not args.andrew_json)
    if synthetic:
        andrew_json_path = output_dir / "synthetic_andrew_floor_xy_input.json"
        payload = create_synthetic_andrew_json(andrew_json_path)
    else:
        andrew_json_path = project_path(args.andrew_json)
        if not andrew_json_path.exists():
            raise SystemExit("Missing Andrew JSON: {}".format(andrew_json_path))
        payload = load_json(andrew_json_path)

    detected = schema_name(payload)
    if detected not in SUPPORTED_SCHEMAS and not synthetic and detected != "unknown_frames_players_floor_xy":
        print(
            json.dumps(
                {
                    "warning": "Unrecognized schema; attempting generic frames[].players[].floor_xy_ft extraction.",
                    "detected_schema": detected,
                    "supported_schemas": sorted(SUPPORTED_SCHEMAS),
                },
                indent=2,
            )
        )

    converted, _skipped = convert_payload(payload, args, template.size, bounds)
    points = converted["points"]
    projected_path = output_dir / "projected_player_points.json"
    summary_path = output_dir / "andrew_floor_xy_adapter_summary.json"
    write_json(projected_path, converted)
    topdown = draw_preview(template_path, points, output_dir / "andrew_floor_xy_topdown_preview.png", with_tracks=False)
    tracks = draw_preview(template_path, points, output_dir / "andrew_floor_xy_tracks_preview.png", with_tracks=True)
    summary = summarize(payload, points, args, bounds, andrew_json_path, synthetic)
    summary["artifacts"] = {
        "projected_player_points": str(projected_path),
        "topdown_preview": topdown,
        "tracks_preview": tracks,
    }
    if args.run_heatmap:
        heatmap_result = run_heatmap(args, output_dir, projected_path, summary_path)
        summary["heatmap_run"] = heatmap_result
        summary["artifacts"]["heatmap_output_dir"] = heatmap_result["output_dir"]
        if heatmap_result["returncode"] != 0:
            summary["status"] = "heatmap_failed"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
