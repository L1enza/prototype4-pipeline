#!/usr/bin/env python3
"""Render moving TOR/OSH team polygons from existing nll_test4 color-only outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import render_side_by_side_field_tracking_smoke as side
import run_team_assignment_and_polygon_demo as v1

TEAM_MAP = {
    "team_a": {
        "abbr": "TOR",
        "name": "Toronto Rock",
        "fill": (246, 246, 246),
        "line": (28, 92, 210),
        "accent": (210, 42, 48),
    },
    "team_b": {
        "abbr": "OSH",
        "name": "Oshawa FireWolves",
        "fill": (105, 28, 38),
        "line": (226, 145, 54),
        "accent": (245, 205, 122),
    },
}
OTHER_COLORS = {"official": (238, 206, 34), "unknown": (145, 145, 145)}


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Render real-team moving polygons and team heatmaps.")
    parser.add_argument("--tracking-metadata", default="outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/tracking_metadata.json")
    parser.add_argument("--projected-points", default="outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/projected_player_points.json")
    parser.add_argument("--assignments", default="outputs/nll_test4/team_assignment_demo_v2/track_team_assignments_v2.json")
    parser.add_argument("--profile", default="outputs/nll_test4/game_team_color_profile/team_color_profile.json")
    parser.add_argument("--field-template", default="assets/field_templates/nll_field_topdown.png")
    parser.add_argument("--video", default="/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/nll_test4.mp4")
    parser.add_argument("--output-dir", default="outputs/nll_test4/team_polygon_demo")
    parser.add_argument("--min-team-confidence", type=float, default=0.65)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--trail-length", type=int, default=12)
    parser.add_argument("--smooth-alpha", type=float, default=0.55)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--field-panel-width", type=int, default=720)
    return parser.parse_args()


def decoded_frame_map(tracking):
    return {
        int(row["clip_frame_index"]): {
            "source_frame_index": int(row["source_frame_index"]),
            "timestamp_seconds": float(row.get("timestamp_seconds", 0.0)),
            "frame_path": row.get("frame_path"),
        }
        for row in tracking.get("decode", {}).get("decoded_frames", [])
    }


def group_detections(tracking):
    by_frame = defaultdict(list)
    by_key = {}
    for det in tracking.get("detections", []):
        frame = int(det["frame_index"])
        track = int(det["track_id"])
        by_frame[frame].append(det)
        by_key[(frame, track)] = det
    return by_frame, by_key


def group_points(points):
    by_frame = defaultdict(list)
    for point in points:
        by_frame[int(point["frame_index"])].append(point)
    return by_frame


def accepted_assignment(row, min_conf):
    cls = row.get("assigned_class")
    if cls not in ("team_a", "team_b"):
        return False, "official_or_unknown"
    if row.get("evidence_mode") != "color_only":
        return False, "not_color_only"
    if float(row.get("confidence") or 0.0) < min_conf:
        return False, "low_team_assignment_confidence"
    return True, None


def smooth_points(points_by_frame, assignments_by_id, min_conf, alpha, max_gap):
    states = {}
    smoothed = {}
    excluded = defaultdict(list)
    for frame in sorted(points_by_frame):
        frame_out = {}
        for point in points_by_frame[frame]:
            track = int(point["track_id"])
            assignment = assignments_by_id.get(track, {})
            ok, reason = accepted_assignment(assignment, min_conf)
            if not ok:
                excluded[frame].append({"track_id": track, "reason": reason})
                continue
            if not point.get("inside_field_template_bounds"):
                excluded[frame].append({"track_id": track, "reason": "projected_point_out_of_bounds"})
                continue
            xy = point["projected_field_point"]
            raw = np.asarray([float(xy["x"]), float(xy["y"])], dtype=np.float32)
            state = states.get(track)
            if state is None or frame - state["frame"] > max_gap:
                value = raw
            else:
                value = (1.0 - alpha) * state["xy"] + alpha * raw
            states[track] = {"frame": frame, "xy": value}
            frame_out[track] = {"x": float(value[0]), "y": float(value[1]), "raw_x": float(raw[0]), "raw_y": float(raw[1])}
        smoothed[frame] = frame_out
    return smoothed, excluded


def hull_geometry(coords):
    if len(coords) >= 3:
        hull = cv2.convexHull(np.asarray(coords, dtype=np.float32)).reshape(-1, 2)
        vertices = [{"x": float(x), "y": float(y)} for x, y in hull]
        return "polygon", vertices
    if len(coords) == 2:
        return "line", [{"x": float(x), "y": float(y)} for x, y in coords]
    if len(coords) == 1:
        return "point", [{"x": float(coords[0][0]), "y": float(coords[0][1])}]
    return "none", []


def centroid(coords):
    if not coords:
        return None
    return {"x": float(statistics.mean([x for x, _y in coords])), "y": float(statistics.mean([y for _x, y in coords]))}


def draw_field_frame(template, frame, smoothed, assignments_by_id, history, output_path):
    image = template.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    frame_record = {
        "frame_index": frame,
        "timestamp": None,
        "accepted_track_ids_by_team": {"TOR": [], "OSH": []},
        "projected_player_coordinates": {"TOR": [], "OSH": []},
        "geometry": {},
        "excluded_tracks": [],
    }
    for label, team in TEAM_MAP.items():
        team_tracks = [
            (track, xy)
            for track, xy in smoothed.get(frame, {}).items()
            if assignments_by_id.get(track, {}).get("assigned_class") == label
        ]
        coords = [(xy["x"], xy["y"]) for _track, xy in team_tracks]
        geom_type, vertices = hull_geometry(coords)
        frame_record["geometry"][team["abbr"]] = {
            "geometry_type": geom_type,
            "polygon_vertices": vertices,
            "centroid": centroid(coords),
            "team_assignment_confidence": {
                str(track): assignments_by_id[track].get("confidence") for track, _xy in team_tracks
            },
        }
        frame_record["accepted_track_ids_by_team"][team["abbr"]] = [track for track, _xy in team_tracks]
        frame_record["projected_player_coordinates"][team["abbr"]] = [
            {"track_id": track, **xy} for track, xy in team_tracks
        ]
        rgba = team["line"] + (70,)
        line = team["line"] + (240,)
        if geom_type == "polygon":
            xy_list = [(v["x"], v["y"]) for v in vertices]
            draw.polygon(xy_list, fill=rgba, outline=line)
        elif geom_type == "line":
            draw.line([(v["x"], v["y"]) for v in vertices], fill=line, width=6)
        for track, trail in history[label].items():
            recent = [(f, xy) for f, xy in trail if frame - 12 <= f <= frame]
            if len(recent) >= 2:
                draw.line([(xy["x"], xy["y"]) for _f, xy in recent], fill=team["accent"] + (150,), width=3)
        for track, xy in team_tracks:
            draw.ellipse((xy["x"] - 7, xy["y"] - 7, xy["x"] + 7, xy["y"] + 7), fill=team["line"] + (255,), outline=(255, 255, 255, 255), width=2)
            draw.text((xy["x"] + 10, xy["y"] - 8), f"{team['abbr']} T{track}", fill=team["line"] + (255,))
    draw.rectangle((8, 8, 470, 44), fill=(255, 255, 255, 235), outline=(0, 0, 0, 255))
    draw.text((16, 17), "Toronto Rock vs Oshawa FireWolves | frame {:03d}".format(frame), fill=(0, 0, 0, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return frame_record, image


def draw_broadcast_frame(frame_rgb, frame, dets, smoothed_frame, assignments_by_id, output_path):
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    team_foot = defaultdict(list)
    for det in dets:
        track = int(det["track_id"])
        assignment = assignments_by_id.get(track, {})
        cls = assignment.get("assigned_class", "unknown")
        if cls in TEAM_MAP and track in smoothed_frame:
            team = TEAM_MAP[cls]
            color = team["line"]
            label = team["abbr"]
        else:
            color = OTHER_COLORS.get(cls, OTHER_COLORS["unknown"])
            label = "OFF" if cls == "official" else "UNK"
        box = det.get("bbox_2d", {})
        draw.rectangle((box.get("x0", 0), box.get("y0", 0), box.get("x1", 0), box.get("y1", 0)), outline=color + (255,), width=3)
        foot = det.get("foot_point_2d")
        if foot:
            draw.ellipse((foot["x"] - 4, foot["y"] - 4, foot["x"] + 4, foot["y"] + 4), fill=color + (255,))
            if cls in TEAM_MAP and track in smoothed_frame:
                team_foot[cls].append((float(foot["x"]), float(foot["y"])))
        draw.text((box.get("x0", 0), max(0, box.get("y0", 0) - 15)), f"{label} T{track}", fill=color + (255,))
    for cls, coords in team_foot.items():
        if len(coords) >= 3:
            hull = cv2.convexHull(np.asarray(coords, dtype=np.float32)).reshape(-1, 2)
            draw.polygon([(float(x), float(y)) for x, y in hull], outline=TEAM_MAP[cls]["accent"] + (220,), fill=TEAM_MAP[cls]["accent"] + (35,))
        elif len(coords) == 2:
            draw.line(coords, fill=TEAM_MAP[cls]["accent"] + (220,), width=4)
    draw.rectangle((8, 8, 420, 42), fill=(0, 0, 0, 165))
    draw.text((16, 17), "Broadcast team polygons | frame {:03d}".format(frame), fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image


def heatmap(points, template, output_path, color):
    width, height = template.size
    density = np.zeros((height, width), dtype=np.float32)
    for xy in points:
        x = int(round(xy["x"]))
        y = int(round(xy["y"]))
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1.0
    if np.max(density) > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=18, sigmaY=18)
        density = density / max(float(np.max(density)), 1e-6)
    rgba = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pix = np.asarray(rgba).copy()
    alpha = np.clip(density * 190, 0, 190).astype(np.uint8)
    pix[:, :, 0] = color[0]
    pix[:, :, 1] = color[1]
    pix[:, :, 2] = color[2]
    pix[:, :, 3] = alpha
    overlay = Image.alpha_composite(template.convert("RGBA"), Image.fromarray(pix, "RGBA")).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return str(output_path)


def main():
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = load_json(project_path(args.tracking_metadata))
    projected = load_json(project_path(args.projected_points)).get("points", [])
    assignments = load_json(project_path(args.assignments))
    profile = load_json(project_path(args.profile)) if project_path(args.profile).exists() else {}
    template = Image.open(project_path(args.field_template)).convert("RGB")
    if any(row.get("evidence_mode") != "color_only" for row in assignments):
        raise ValueError("Assignments must be color_only")
    assignments_by_id = {int(row["track_id"]): row for row in assignments}
    det_by_frame, _det_by_key = group_detections(tracking)
    points_by_frame = group_points(projected)
    frame_lookup = decoded_frame_map(tracking)
    needed_sources = [row["source_frame_index"] for row in frame_lookup.values()]
    frames_by_source = v1.read_needed_frames(project_path(args.video), needed_sources)
    smoothed, excluded = smooth_points(points_by_frame, assignments_by_id, args.min_team_confidence, args.smooth_alpha, args.max_gap)
    history = {label: defaultdict(list) for label in TEAM_MAP}
    frame_records = []
    field_paths, broadcast_paths, side_paths = [], [], []
    all_team_points = {"TOR": [], "OSH": []}
    frame_indices = sorted(frame_lookup)
    for frame in frame_indices:
        for track, xy in smoothed.get(frame, {}).items():
            cls = assignments_by_id.get(track, {}).get("assigned_class")
            if cls in TEAM_MAP:
                history[cls][track].append((frame, xy))
        field_record, field_img = draw_field_frame(template, frame, smoothed, assignments_by_id, history, output_dir / "field_frames" / f"frame_{frame:03d}_field_team_polygon.png")
        field_record["timestamp"] = frame_lookup.get(frame, {}).get("timestamp_seconds")
        field_record["excluded_tracks"] = excluded.get(frame, [])
        field_paths.append(output_dir / "field_frames" / f"frame_{frame:03d}_field_team_polygon.png")
        source = frame_lookup.get(frame, {}).get("source_frame_index")
        if source in frames_by_source:
            broadcast = draw_broadcast_frame(frames_by_source[source], frame, det_by_frame.get(frame, []), smoothed.get(frame, {}), assignments_by_id, output_dir / "broadcast_frames" / f"frame_{frame:03d}_broadcast_team_polygon.png")
            broadcast_paths.append(output_dir / "broadcast_frames" / f"frame_{frame:03d}_broadcast_team_polygon.png")
            field_panel, _scale = side.resize_with_aspect(field_img, target_height=broadcast.height)
            canvas = Image.new("RGB", (broadcast.width + 12 + field_panel.width, broadcast.height), (18, 18, 18))
            canvas.paste(broadcast, (0, 0))
            canvas.paste(field_panel, (broadcast.width + 12, 0))
            side_path = output_dir / "side_by_side_frames" / f"frame_{frame:03d}_side_by_side_team_polygon.png"
            side_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(side_path)
            side_paths.append(side_path)
        for abbr in ("TOR", "OSH"):
            all_team_points[abbr].extend(field_record["projected_player_coordinates"][abbr])
        frame_records.append(field_record)

    broadcast_mp4, broadcast_err = v1.write_mp4(broadcast_paths, output_dir / "broadcast_team_polygons.mp4", args.fps)
    field_mp4, field_err = v1.write_mp4(field_paths, output_dir / "field_team_polygons.mp4", args.fps)
    side_mp4, side_err = v1.write_mp4(side_paths, output_dir / "side_by_side_team_polygons.mp4", args.fps)
    tor_heat = heatmap(all_team_points["TOR"], template, output_dir / "toronto_heatmap.png", TEAM_MAP["team_a"]["line"])
    osh_heat = heatmap(all_team_points["OSH"], template, output_dir / "oshawa_heatmap.png", TEAM_MAP["team_b"]["line"])
    combined_base = Image.open(tor_heat).convert("RGB")
    combined_overlay = Image.open(osh_heat).convert("RGB")
    combined = Image.blend(combined_base, combined_overlay, 0.5)
    combined_path = output_dir / "combined_team_heatmap.png"
    combined.save(combined_path)

    geometry_counts = {"TOR": Counter(), "OSH": Counter()}
    excluded_counts = Counter()
    for row in frame_records:
        for team in ("TOR", "OSH"):
            geometry_counts[team][row["geometry"][team]["geometry_type"]] += 1
        for item in row["excluded_tracks"]:
            excluded_counts[item["reason"]] += 1
    summary = {
        "stage": "team_polygon_demo",
        "status": "complete",
        "evidence_mode": "color_only",
        "embedding_backend": "none",
        "scorebug_priors_used": False,
        "team_mapping": {
            "team_a": {"abbr": "TOR", "name": "Toronto Rock"},
            "team_b": {"abbr": "OSH", "name": "Oshawa FireWolves"},
        },
        "profile_source": str(project_path(args.profile)),
        "frames_processed": len(frame_records),
        "counts": {
            "TOR_projected_observations": len(all_team_points["TOR"]),
            "OSH_projected_observations": len(all_team_points["OSH"]),
            "geometry_counts": {team: dict(counts) for team, counts in geometry_counts.items()},
            "excluded_tracks_by_reason": dict(excluded_counts),
        },
        "smoothing": {"alpha": args.smooth_alpha, "max_gap": args.max_gap, "flicker_note": "simple per-track EMA; polygons may still flicker when visible player count crosses 2/3 thresholds"},
        "artifacts": {
            "broadcast_team_polygons_mp4": broadcast_mp4,
            "broadcast_team_polygons_mp4_error": broadcast_err,
            "field_team_polygons_mp4": field_mp4,
            "field_team_polygons_mp4_error": field_err,
            "side_by_side_team_polygons_mp4": side_mp4,
            "side_by_side_team_polygons_mp4_error": side_err,
            "toronto_heatmap": tor_heat,
            "oshawa_heatmap": osh_heat,
            "combined_team_heatmap": str(combined_path),
            "team_polygon_frames": str(output_dir / "team_polygon_frames.json"),
            "team_polygon_summary": str(output_dir / "team_polygon_summary.json"),
        },
    }
    write_json(output_dir / "team_polygon_frames.json", {"frames": frame_records})
    write_json(output_dir / "team_polygon_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

