#!/usr/bin/env python3
"""Assign anonymous tracks to team-level classes and render team polygons."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEAM_COLORS = {
    "team_a": (220, 40, 40),
    "team_b": (30, 105, 230),
    "official": (245, 210, 35),
    "unknown": (150, 150, 150),
}

KNOWN_LIMITATIONS = [
    "Team assignment is uniform-color clustering only; it does not know roster names or jersey numbers.",
    "White uniforms, goalies, officials, shadows, and broadcast compression can confuse color evidence.",
    "Referee detection is a conservative heuristic and may leave some officials as unknown or team-like tracks.",
    "Similar uniform colors can collapse into one cluster or produce low-confidence assignments.",
    "Moving polygons depend on tracked projected foot points; ID switches and homography drift affect team shape.",
    "Scorebug colors are not parsed in this smoke stage; clusters are named team_a and team_b.",
    "This stage can later consume Andrew floor_xy_ft coordinates instead of manual homography projection points.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create team assignment, heatmaps, trails, and moving polygons from existing projected tracking outputs.")
    parser.add_argument("--run-id", default="nll_test4")
    parser.add_argument("--tracking-metadata", default="outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/tracking_metadata.json")
    parser.add_argument("--projected-points", default="outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/projected_player_points.json")
    parser.add_argument("--video", default="/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/nll_test4.mp4")
    parser.add_argument("--field-template", default="assets/field_templates/nll_field_topdown.png")
    parser.add_argument("--team-output-dir", default="outputs/nll_test4/team_assignment_demo")
    parser.add_argument("--polygon-output-dir", default="outputs/nll_test4/team_polygon_demo")
    parser.add_argument("--min-track-observations", type=int, default=4)
    parser.add_argument("--max-observations-per-track", type=int, default=24)
    parser.add_argument("--unknown-confidence-threshold", type=float, default=0.42)
    parser.add_argument("--official-score-threshold", type=float, default=0.82)
    parser.add_argument("--trail-length", type=int, default=10)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--heatmap-sigma", type=float, default=20.0)
    parser.add_argument("--heatmap-alpha", type=float, default=0.55)
    parser.add_argument("--field-panel-width", type=int, default=720)
    parser.add_argument("--scorebug-priors", default=None, help="Optional JSON with scorebug team color priors. Not required for this smoke stage.")
    return parser.parse_args()


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clamp_box(box, width, height):
    x0 = int(max(0, min(width - 1, round(float(box["x0"])))))
    y0 = int(max(0, min(height - 1, round(float(box["y0"])))))
    x1 = int(max(0, min(width - 1, round(float(box["x1"])))))
    y1 = int(max(0, min(height - 1, round(float(box["y1"])))))
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)
    return x0, y0, x1, y1


def torso_box_from_bbox(box, width, height):
    x0, y0, x1, y1 = clamp_box(box, width, height)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    tx0 = x0 + int(round(0.14 * bw))
    tx1 = x0 + int(round(0.86 * bw))
    ty0 = y0 + int(round(0.18 * bh))
    ty1 = y0 + int(round(0.70 * bh))
    return clamp_box({"x0": tx0, "y0": ty0, "x1": tx1, "y1": ty1}, width, height)


def sharpness_score(crop_rgb):
    if crop_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def color_features(crop_rgb):
    if crop_rgb.size == 0:
        return None
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    flat_rgb = crop_rgb.reshape(-1, 3)
    flat_hsv = hsv.reshape(-1, 3)
    flat_lab = lab.reshape(-1, 3)
    sat = flat_hsv[:, 1].astype(np.float32)
    val = flat_hsv[:, 2].astype(np.float32)
    mask = val > 30
    if np.count_nonzero(mask) < 12:
        mask = np.ones(len(flat_rgb), dtype=bool)
    hist_mask = mask.reshape(hsv.shape[:2]).astype(np.uint8)
    med_rgb = np.median(flat_rgb[mask], axis=0)
    med_hsv = np.median(flat_hsv[mask], axis=0)
    med_lab = np.median(flat_lab[mask], axis=0)
    hist_h = cv2.calcHist([hsv], [0], hist_mask, [18], [0, 180]).flatten().astype(np.float32)
    hist_s = cv2.calcHist([hsv], [1], hist_mask, [8], [0, 256]).flatten().astype(np.float32)
    hist_v = cv2.calcHist([hsv], [2], hist_mask, [8], [0, 256]).flatten().astype(np.float32)
    hist = np.concatenate([hist_h, hist_s, hist_v])
    hist = hist / max(float(np.linalg.norm(hist)), 1e-6)
    low_sat_fraction = float(np.mean(sat < 45))
    white_fraction = float(np.mean((sat < 45) & (val > 170)))
    dark_fraction = float(np.mean(val < 65))
    gray_fraction = float(np.mean((sat < 38) & (val >= 65) & (val <= 190)))
    edges = cv2.Canny(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY), 60, 150)
    edge_density = float(np.mean(edges > 0))
    return {
        "median_rgb": med_rgb.astype(np.float32),
        "median_hsv": med_hsv.astype(np.float32),
        "median_lab": med_lab.astype(np.float32),
        "hist": hist.astype(np.float32),
        "low_sat_fraction": low_sat_fraction,
        "white_fraction": white_fraction,
        "dark_fraction": dark_fraction,
        "gray_fraction": gray_fraction,
        "edge_density": edge_density,
        "sharpness": sharpness_score(crop_rgb),
    }


def crop_quality(box, frame_width, frame_height, sharpness):
    x0, y0, x1, y1 = clamp_box(box, frame_width, frame_height)
    area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    frame_area = float(frame_width * frame_height)
    area_score = min(1.0, area / max(1.0, frame_area * 0.018))
    border_penalty = 0.0
    margin = 6
    if x0 <= margin or y0 <= margin or x1 >= frame_width - margin or y1 >= frame_height - margin:
        border_penalty = 0.25
    sharp_score = min(1.0, math.log1p(max(0.0, sharpness)) / math.log1p(900.0))
    return max(0.0, min(1.0, 0.62 * area_score + 0.38 * sharp_score - border_penalty))


def read_needed_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video {}".format(video_path))
    frames = {}
    try:
        for index in sorted(set(int(i) for i in frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frames[index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return frames


def decoded_frame_map(tracking_metadata):
    mapping = {}
    for row in tracking_metadata.get("decode", {}).get("decoded_frames", []):
        mapping[int(row["clip_frame_index"])] = {
            "source_frame_index": int(row["source_frame_index"]),
            "timestamp_seconds": float(row.get("timestamp_seconds", 0.0)),
            "frame_path": row.get("frame_path"),
        }
    return mapping


def detections_by_track(detections):
    tracks = {}
    for det in detections:
        tracks.setdefault(int(det["track_id"]), []).append(det)
    for rows in tracks.values():
        rows.sort(key=lambda item: (int(item["frame_index"]), int(item.get("mask_id", 0))))
    return tracks


def select_observations(rows, max_count):
    def score(det):
        area = float(det.get("bbox_area") or 0.0)
        sam = float(det.get("sam_confidence_score") or 0.0)
        return area * (0.5 + 0.5 * sam)
    ranked = sorted(rows, key=score, reverse=True)
    return sorted(ranked[:max_count], key=lambda item: int(item["frame_index"]))


def aggregate_track_features(track_id, rows, frame_lookup, source_frames, max_observations):
    observations = []
    selected = select_observations(rows, max_observations)
    for det in selected:
        clip_idx = int(det["frame_index"])
        source_info = frame_lookup.get(clip_idx)
        if not source_info:
            continue
        frame = source_frames.get(source_info["source_frame_index"])
        if frame is None:
            continue
        height, width = frame.shape[:2]
        box = det.get("bbox_2d") or {}
        x0, y0, x1, y1 = torso_box_from_bbox(box, width, height)
        crop = frame[y0 : y1 + 1, x0 : x1 + 1]
        feats = color_features(crop)
        if not feats:
            continue
        quality = crop_quality({"x0": x0, "y0": y0, "x1": x1, "y1": y1}, width, height, feats["sharpness"])
        observations.append({
            "frame_index": clip_idx,
            "source_frame_index": source_info["source_frame_index"],
            "timestamp_seconds": source_info.get("timestamp_seconds"),
            "bbox": box,
            "torso_bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "quality": float(quality),
            "median_rgb": feats["median_rgb"],
            "median_hsv": feats["median_hsv"],
            "median_lab": feats["median_lab"],
            "hist": feats["hist"],
            "low_sat_fraction": feats["low_sat_fraction"],
            "white_fraction": feats["white_fraction"],
            "dark_fraction": feats["dark_fraction"],
            "gray_fraction": feats["gray_fraction"],
            "edge_density": feats["edge_density"],
            "sharpness": feats["sharpness"],
            "crop_rgb": crop,
        })
    if not observations:
        return None
    weights = np.array([max(0.05, obs["quality"]) for obs in observations], dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), 1e-6)
    med_rgb = np.sum(np.stack([obs["median_rgb"] for obs in observations]) * weights[:, None], axis=0)
    med_hsv = np.sum(np.stack([obs["median_hsv"] for obs in observations]) * weights[:, None], axis=0)
    med_lab = np.sum(np.stack([obs["median_lab"] for obs in observations]) * weights[:, None], axis=0)
    hist = np.sum(np.stack([obs["hist"] for obs in observations]) * weights[:, None], axis=0)
    hist = hist / max(float(np.linalg.norm(hist)), 1e-6)
    quality_values = [obs["quality"] for obs in observations]
    official_score = official_likelihood(observations)
    best_obs = max(observations, key=lambda obs: obs["quality"])
    feature_vector = np.concatenate([
        med_lab / np.array([255.0, 255.0, 255.0], dtype=np.float32),
        med_hsv / np.array([180.0, 255.0, 255.0], dtype=np.float32),
        hist * 0.65,
    ]).astype(np.float32)
    return {
        "track_id": int(track_id),
        "observations": observations,
        "feature_vector": feature_vector,
        "representative_jersey_color_rgb": [int(round(x)) for x in med_rgb.tolist()],
        "representative_jersey_color_hsv": [float(x) for x in med_hsv.tolist()],
        "representative_jersey_color_lab": [float(x) for x in med_lab.tolist()],
        "observation_count": len(observations),
        "evidence_quality": float(min(1.0, 0.55 * (len(observations) / 10.0) + 0.45 * statistics.mean(quality_values))),
        "official_score": float(official_score),
        "best_observation": best_obs,
        "track_frame_count": len(rows),
    }


def official_likelihood(observations):
    if not observations:
        return 0.0
    qualities = np.array([max(0.05, obs["quality"]) for obs in observations], dtype=np.float32)
    qualities = qualities / max(float(np.sum(qualities)), 1e-6)
    low_sat = float(np.sum([obs["low_sat_fraction"] * w for obs, w in zip(observations, qualities)]))
    white = float(np.sum([obs["white_fraction"] * w for obs, w in zip(observations, qualities)]))
    dark = float(np.sum([obs["dark_fraction"] * w for obs, w in zip(observations, qualities)]))
    gray = float(np.sum([obs["gray_fraction"] * w for obs, w in zip(observations, qualities)]))
    edge = float(np.sum([obs["edge_density"] * w for obs, w in zip(observations, qualities)]))
    stripe_like = min(1.0, edge / 0.18)
    black_white_mix = min(1.0, (dark + white) / 0.52)
    neutral_body = min(1.0, (low_sat + gray) / 1.25)
    return float(max(0.0, min(1.0, 0.42 * black_white_mix + 0.38 * neutral_body + 0.20 * stripe_like)))


def kmeans_two(features, max_iter=50):
    features = np.asarray(features, dtype=np.float32)
    if len(features) < 2:
        return np.zeros(len(features), dtype=np.int32), features.copy()
    dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(dists)), dists.shape)
    centers = np.stack([features[i], features[j]], axis=0)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(max_iter):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for cluster in (0, 1):
            members = features[new_labels == cluster]
            if len(members):
                new_centers[cluster] = np.mean(members, axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels = new_labels
        centers = new_centers
    return labels, centers


def assign_teams(track_features, args):
    assignments = {}
    feature_rows = []
    feature_track_ids = []
    for track_id, feat in sorted(track_features.items()):
        if feat["observation_count"] < args.min_track_observations:
            assignments[track_id] = base_assignment(feat, "unknown", 0.2, "too_few_color_observations")
        elif feat["official_score"] >= args.official_score_threshold:
            assignments[track_id] = base_assignment(feat, "official", feat["official_score"], "strong_black_white_gray_official_color_heuristic")
        else:
            feature_track_ids.append(track_id)
            feature_rows.append(feat["feature_vector"])
    if len(feature_rows) < 2:
        for track_id in feature_track_ids:
            assignments[track_id] = base_assignment(track_features[track_id], "unknown", 0.3, "not_enough_non_official_tracks_for_two_team_clustering")
        return assignments, None
    labels, centers = kmeans_two(feature_rows)
    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    if counts[1] > counts[0]:
        team_for_label = {1: "team_a", 0: "team_b"}
    else:
        team_for_label = {0: "team_a", 1: "team_b"}
    distances = np.linalg.norm(np.asarray(feature_rows)[:, None, :] - centers[None, :, :], axis=2)
    all_dist = distances.flatten()
    typical = float(np.median(all_dist)) if len(all_dist) else 1.0
    typical = max(typical, 0.08)
    for idx, track_id in enumerate(feature_track_ids):
        feat = track_features[track_id]
        label = int(labels[idx])
        sorted_dist = sorted(float(x) for x in distances[idx].tolist())
        nearest = sorted_dist[0]
        second = sorted_dist[1] if len(sorted_dist) > 1 else nearest + typical
        margin = max(0.0, min(1.0, (second - nearest) / max(second, typical, 1e-6)))
        compactness = max(0.0, min(1.0, 1.0 - nearest / (typical * 2.5)))
        confidence = max(0.0, min(1.0, 0.30 * feat["evidence_quality"] + 0.45 * margin + 0.25 * compactness))
        if confidence < args.unknown_confidence_threshold:
            assignments[track_id] = base_assignment(feat, "unknown", confidence, "low_cluster_confidence")
        else:
            assignments[track_id] = base_assignment(feat, team_for_label[label], confidence, "clustered_by_track_level_uniform_color")
            assignments[track_id]["cluster_label"] = int(label)
            assignments[track_id]["cluster_distance"] = float(nearest)
            assignments[track_id]["cluster_margin"] = float(margin)
    diagnostics = {
        "cluster_counts": {"cluster_0": counts[0], "cluster_1": counts[1]},
        "team_for_cluster_label": {str(k): v for k, v in team_for_label.items()},
        "centers": centers.tolist(),
    }
    return assignments, diagnostics


def base_assignment(feat, assigned_class, confidence, reason):
    return {
        "track_id": int(feat["track_id"]),
        "assigned_class": assigned_class,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "representative_jersey_color_rgb": feat["representative_jersey_color_rgb"],
        "representative_jersey_color_hsv": feat["representative_jersey_color_hsv"],
        "representative_jersey_color_lab": feat["representative_jersey_color_lab"],
        "number_of_observations_used": int(feat["observation_count"]),
        "track_frame_count": int(feat["track_frame_count"]),
        "evidence_quality": float(feat["evidence_quality"]),
        "official_score": float(feat["official_score"]),
        "reason": reason,
        "reason_for_unknown_assignment": reason if assigned_class == "unknown" else None,
    }


def save_contact_sheet(track_features, assignments, output_path):
    tiles = []
    for track_id, feat in sorted(track_features.items()):
        obs = feat["best_observation"]
        crop = Image.fromarray(obs["crop_rgb"]).convert("RGB")
        crop.thumbnail((120, 150), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (160, 205), (245, 245, 245))
        x = (160 - crop.width) // 2
        tile.paste(crop, (x, 8))
        draw = ImageDraw.Draw(tile)
        assign = assignments[int(track_id)]
        color = TEAM_COLORS.get(assign["assigned_class"], TEAM_COLORS["unknown"])
        draw.rectangle((0, 0, 159, 204), outline=color, width=4)
        draw.text((8, 162), "T{} {}".format(track_id, assign["assigned_class"]), fill=color)
        draw.text((8, 181), "conf {:.2f}".format(assign["confidence"]), fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return None
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 160, rows * 205), (235, 235, 235))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 160, (idx // cols) * 205))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def save_cluster_preview(track_features, assignments, output_path):
    width, height = 720, 360
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), "Track uniform color features: hue x saturation, tile color = representative RGB", fill=(0, 0, 0))
    for track_id, feat in sorted(track_features.items()):
        hsv = feat["representative_jersey_color_hsv"]
        x = int(40 + (float(hsv[0]) / 180.0) * (width - 90))
        y = int(height - 40 - (float(hsv[1]) / 255.0) * (height - 90))
        rgb = tuple(int(v) for v in feat["representative_jersey_color_rgb"])
        assigned = assignments[int(track_id)]["assigned_class"]
        outline = TEAM_COLORS.get(assigned, TEAM_COLORS["unknown"])
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=rgb, outline=outline, width=4)
        draw.text((x + 14, y - 8), "T{}".format(track_id), fill=outline)
    draw.line((40, height - 40, width - 40, height - 40), fill=(0, 0, 0), width=2)
    draw.line((40, height - 40, 40, 40), fill=(0, 0, 0), width=2)
    draw.text((width // 2 - 30, height - 24), "hue", fill=(0, 0, 0))
    draw.text((8, 42), "sat", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def draw_tracking_overlay(frame_rgb, detections, assignments):
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in detections:
        track_id = int(det["track_id"])
        assign = assignments.get(track_id, {"assigned_class": "unknown", "confidence": 0.0})
        cls = assign["assigned_class"]
        color = TEAM_COLORS.get(cls, TEAM_COLORS["unknown"])
        box = det.get("bbox_2d") or {}
        x0, y0, x1, y1 = clamp_box(box, image.width, image.height)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        foot = det.get("foot_point_2d") or {}
        if "x" in foot and "y" in foot:
            fx, fy = int(round(foot["x"])), int(round(foot["y"]))
            draw.ellipse((fx - 4, fy - 4, fx + 4, fy + 4), fill=color, outline=(255, 255, 255))
        label = "T{} {}".format(track_id, cls.replace("team_", ""))
        tw = max(72, len(label) * 7)
        draw.rectangle((x0, max(0, y0 - 18), x0 + tw, y0), fill=color)
        draw.text((x0 + 3, max(0, y0 - 16)), label, fill=(255, 255, 255) if cls != "official" else (0, 0, 0))
    return image


def write_gif(frame_paths, output_path, fps):
    if not frame_paths:
        return None
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)
    finally:
        for image in images:
            image.close()
    return str(output_path)


def write_mp4(frame_paths, output_path, fps):
    if not frame_paths:
        return None, {"type": "ValueError", "message": "No frames available."}
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None, {"type": "ValueError", "message": "Could not read {}".format(frame_paths[0])}
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return None, {"type": "RuntimeError", "message": "OpenCV could not open MP4 writer."}
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


def group_projected_points(points):
    by_frame = {}
    by_track = {}
    for point in points:
        if not point.get("inside_field_template_bounds"):
            continue
        frame_index = int(point["frame_index"])
        track_id = int(point["track_id"])
        by_frame.setdefault(frame_index, []).append(point)
        by_track.setdefault(track_id, []).append(point)
    for rows in by_frame.values():
        rows.sort(key=lambda row: int(row["track_id"]))
    for rows in by_track.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return by_frame, by_track


def point_xy(point):
    xy = point["projected_field_point"]
    return float(xy["x"]), float(xy["y"])


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.contourArea(contour))


def draw_team_field_frame(template, points_by_frame, points_by_track, assignments, frame_index, trail_length):
    field = template.convert("RGB")
    overlay = Image.new("RGBA", field.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(field)
    frame_points = points_by_frame.get(frame_index, [])
    metrics = {
        "frame_index": int(frame_index),
        "team_a_visible_player_count": 0,
        "team_b_visible_player_count": 0,
        "team_a_polygon_area": 0.0,
        "team_b_polygon_area": 0.0,
        "team_a_centroid": None,
        "team_b_centroid": None,
        "centroid_distance": None,
        "unknown_count": 0,
        "official_count": 0,
    }
    for track_id, rows in sorted(points_by_track.items()):
        cls = assignments.get(track_id, {}).get("assigned_class", "unknown")
        if cls not in ("team_a", "team_b"):
            continue
        history = [row for row in rows if frame_index - trail_length <= int(row["frame_index"]) <= frame_index]
        if len(history) >= 2:
            coords = [point_xy(row) for row in history]
            color = TEAM_COLORS[cls]
            draw.line(coords, fill=color, width=3)
    team_current = {"team_a": [], "team_b": []}
    for point in frame_points:
        track_id = int(point["track_id"])
        cls = assignments.get(track_id, {}).get("assigned_class", "unknown")
        x, y = point_xy(point)
        if cls in team_current:
            team_current[cls].append((x, y, track_id))
        elif cls == "official":
            metrics["official_count"] += 1
        else:
            metrics["unknown_count"] += 1
    for cls in ("team_a", "team_b"):
        coords = [(x, y) for x, y, _tid in team_current[cls]]
        metrics["{}_visible_player_count".format(cls)] = len(coords)
        color = TEAM_COLORS[cls]
        if len(coords) >= 3:
            hull = cv2.convexHull(np.asarray(coords, dtype=np.float32)).reshape(-1, 2)
            poly = [(float(x), float(y)) for x, y in hull]
            draw_overlay.polygon(poly, fill=color + (55,), outline=color + (210,))
            metrics["{}_polygon_area".format(cls)] = polygon_area(poly)
        elif len(coords) == 2:
            draw.line(coords, fill=color, width=5)
        if coords:
            cx = float(sum(x for x, _ in coords) / len(coords))
            cy = float(sum(y for _, y in coords) / len(coords))
            metrics["{}_centroid".format(cls)] = {"x": cx, "y": cy}
    field = Image.alpha_composite(field.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(field)
    for cls in ("team_a", "team_b"):
        color = TEAM_COLORS[cls]
        for x, y, track_id in team_current[cls]:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255), width=2)
            draw.text((x + 9, y - 8), "T{}".format(track_id), fill=color)
        centroid = metrics["{}_centroid".format(cls)]
        if centroid:
            cx, cy = centroid["x"], centroid["y"]
            draw.rectangle((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 255, 255), outline=color, width=3)
    ca = metrics["team_a_centroid"]
    cb = metrics["team_b_centroid"]
    if ca and cb:
        metrics["centroid_distance"] = float(math.hypot(ca["x"] - cb["x"], ca["y"] - cb["y"]))
        draw.line((ca["x"], ca["y"], cb["x"], cb["y"]), fill=(40, 40, 40), width=2)
    draw.rectangle((8, 8, 330, 40), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((14, 16), "frame {:03d} | A {} B {} U {} O {}".format(
        frame_index,
        metrics["team_a_visible_player_count"],
        metrics["team_b_visible_player_count"],
        metrics["unknown_count"],
        metrics["official_count"],
    ), fill=(0, 0, 0))
    return field, metrics


def resize_with_aspect(image, target_height=None, target_width=None):
    width, height = image.size
    if target_height is not None:
        scale = float(target_height) / float(height)
    elif target_width is not None:
        scale = float(target_width) / float(width)
    else:
        return image.copy()
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def compose_side_by_side(left, right, output_path):
    right = resize_with_aspect(right, target_height=left.height)
    gap = 12
    canvas = Image.new("RGB", (left.width + gap + right.width, max(left.height, right.height)), (18, 18, 18))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, canvas.height - 24), "broadcast team tracks", fill=(255, 255, 255))
    draw.text((left.width + gap + 12, canvas.height - 24), "team field polygons", fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def density_heatmap(points, template, output_heat, output_overlay, sigma, alpha):
    width, height = template.size
    density = np.zeros((height, width), dtype=np.float32)
    for point in points:
        x, y = point_xy(point)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            density[yi, xi] += 1.0
    if sigma > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
    if float(np.max(density)) > 0:
        norm = np.clip(density / float(np.max(density)) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros((height, width), dtype=np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    output_heat.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heat).save(output_heat)
    overlay = cv2.addWeighted(np.asarray(template.convert("RGB")), 1.0 - alpha, heat, alpha, 0)
    Image.fromarray(overlay).save(output_overlay)
    return str(output_heat), str(output_overlay), int(np.count_nonzero(density))


def save_team_heatmaps(points, assignments, template, output_dir, sigma, alpha):
    team_points = {"team_a": [], "team_b": []}
    for point in points:
        cls = assignments.get(int(point["track_id"]), {}).get("assigned_class", "unknown")
        if cls in team_points:
            team_points[cls].append(point)
    artifacts = {}
    for cls in ("team_a", "team_b"):
        heat, overlay, nonzero = density_heatmap(
            team_points[cls], template,
            output_dir / "{}_heatmap.png".format(cls),
            output_dir / "{}_heatmap_overlay.png".format(cls),
            sigma, alpha,
        )
        artifacts["{}_heatmap".format(cls)] = heat
        artifacts["{}_heatmap_overlay".format(cls)] = overlay
        artifacts["{}_nonzero_density_pixels".format(cls)] = nonzero
    combined = template.convert("RGB")
    overlay = Image.new("RGBA", combined.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for cls, rows in team_points.items():
        color = TEAM_COLORS[cls]
        for point in rows:
            x, y = point_xy(point)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color + (70,))
    combined = Image.alpha_composite(combined.convert("RGBA"), overlay).convert("RGB")
    combined_path = output_dir / "team_specific_heatmap.png"
    combined.save(combined_path)
    artifacts["team_specific_heatmap"] = str(combined_path)
    return artifacts


def render_assignment_overlay(frames_by_source, frame_lookup, detections_by_frame_map, assignments, output_dir, fps):
    frames_dir = output_dir / "overlay_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for frame_index in sorted(detections_by_frame_map):
        source = frame_lookup.get(frame_index, {}).get("source_frame_index")
        if source not in frames_by_source:
            continue
        image = draw_tracking_overlay(frames_by_source[source], detections_by_frame_map[frame_index], assignments)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 210, 34), fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((14, 15), "frame {:03d}".format(frame_index), fill=(0, 0, 0))
        path = frames_dir / "frame_{:03d}_team_assignment.png".format(frame_index)
        image.save(path)
        rendered.append(path)
    mp4, mp4_error = write_mp4(rendered, output_dir / "team_assignment_overlay.mp4", fps)
    gif = write_gif(rendered, output_dir / "team_assignment_overlay.gif", fps)
    return {"frames_dir": str(frames_dir), "frames": [str(p) for p in rendered], "mp4": mp4, "mp4_error": mp4_error, "gif": gif}


def render_polygon_outputs(frames_by_source, frame_lookup, detections_by_frame_map, projected_points, assignments, template, output_dir, fps, trail_length, field_panel_width):
    frames_root = output_dir / "frames"
    field_dir = frames_root / "field"
    side_dir = frames_root / "side_by_side"
    field_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    by_frame, by_track = group_projected_points(projected_points)
    frame_indices = sorted(set(detections_by_frame_map) | set(by_frame))
    field_frames = []
    side_frames = []
    frame_metrics = []
    for frame_index in frame_indices:
        field, metrics = draw_team_field_frame(template, by_frame, by_track, assignments, frame_index, trail_length)
        field_resized = resize_with_aspect(field, target_width=field_panel_width)
        field_path = field_dir / "frame_{:03d}_team_polygon_field.png".format(frame_index)
        field_resized.save(field_path)
        field_frames.append(field_path)
        source = frame_lookup.get(frame_index, {}).get("source_frame_index")
        if source in frames_by_source:
            left = draw_tracking_overlay(frames_by_source[source], detections_by_frame_map.get(frame_index, []), assignments)
            side_path = side_dir / "frame_{:03d}_team_polygon_side_by_side.png".format(frame_index)
            compose_side_by_side(left, field_resized, side_path)
            side_frames.append(side_path)
        frame_metrics.append(metrics)
    field_mp4, field_mp4_error = write_mp4(field_frames, output_dir / "team_polygon_field.mp4", fps)
    field_gif = write_gif(field_frames, output_dir / "team_polygon_field.gif", fps)
    side_mp4, side_mp4_error = write_mp4(side_frames, output_dir / "team_polygon_side_by_side.mp4", fps)
    side_gif = write_gif(side_frames, output_dir / "team_polygon_side_by_side.gif", fps)
    artifacts = {
        "field_frames_dir": str(field_dir),
        "side_by_side_frames_dir": str(side_dir),
        "team_polygon_field_mp4": field_mp4,
        "team_polygon_field_mp4_error": field_mp4_error,
        "team_polygon_field_gif": field_gif,
        "team_polygon_side_by_side_mp4": side_mp4,
        "team_polygon_side_by_side_mp4_error": side_mp4_error,
        "team_polygon_side_by_side_gif": side_gif,
    }
    return frame_metrics, artifacts


def main():
    args = parse_args()
    tracking_path = project_path(args.tracking_metadata)
    projected_path = project_path(args.projected_points)
    video_path = Path(args.video)
    field_template_path = project_path(args.field_template)
    team_output = project_path(args.team_output_dir)
    polygon_output = project_path(args.polygon_output_dir)
    team_output.mkdir(parents=True, exist_ok=True)
    polygon_output.mkdir(parents=True, exist_ok=True)

    tracking = load_json(tracking_path)
    projected_payload = load_json(projected_path)
    projected_points = projected_payload.get("points", [])
    detections = tracking.get("detections", [])
    frame_lookup = decoded_frame_map(tracking)
    detections_by_frame_map = {}
    for det in detections:
        detections_by_frame_map.setdefault(int(det["frame_index"]), []).append(det)
    tracks = detections_by_track(detections)
    needed_source_frames = [frame_lookup[int(det["frame_index"])]["source_frame_index"] for det in detections if int(det["frame_index"]) in frame_lookup]
    frames_by_source = read_needed_frames(video_path, needed_source_frames)

    track_features = {}
    skipped_tracks = {}
    for track_id, rows in sorted(tracks.items()):
        feat = aggregate_track_features(track_id, rows, frame_lookup, frames_by_source, args.max_observations_per_track)
        if feat is None:
            skipped_tracks[track_id] = "no_clean_video_color_observations"
        else:
            track_features[track_id] = feat
    assignments, cluster_diagnostics = assign_teams(track_features, args)
    for track_id, reason in skipped_tracks.items():
        assignments[track_id] = {
            "track_id": int(track_id),
            "assigned_class": "unknown",
            "confidence": 0.0,
            "representative_jersey_color_rgb": None,
            "representative_jersey_color_hsv": None,
            "representative_jersey_color_lab": None,
            "number_of_observations_used": 0,
            "track_frame_count": len(tracks[track_id]),
            "evidence_quality": 0.0,
            "official_score": 0.0,
            "reason": reason,
            "reason_for_unknown_assignment": reason,
        }
    assignment_list = [assignments[track_id] for track_id in sorted(assignments)]
    scorebug_used = False
    scorebug_note = "scorebug color priors were not used; team names remain unresolved as team_a/team_b"
    if args.scorebug_priors and Path(args.scorebug_priors).exists():
        scorebug_note = "scorebug priors file was provided but automatic reliable mapping is not implemented in this smoke stage"

    contact = save_contact_sheet(track_features, assignments, team_output / "team_assignment_contact_sheet.png")
    cluster_preview = save_cluster_preview(track_features, assignments, team_output / "team_color_cluster_preview.png")
    assignment_overlay = render_assignment_overlay(frames_by_source, frame_lookup, detections_by_frame_map, assignments, team_output, args.fps)
    class_counts = {}
    for assignment in assignment_list:
        class_counts[assignment["assigned_class"]] = class_counts.get(assignment["assigned_class"], 0) + 1
    confidences = [a["confidence"] for a in assignment_list]
    confidence_summary = {
        "min": min(confidences) if confidences else None,
        "max": max(confidences) if confidences else None,
        "mean": float(statistics.mean(confidences)) if confidences else None,
        "median": float(statistics.median(confidences)) if confidences else None,
    }
    assignment_metadata = {
        "status": "complete",
        "stage": "team_assignment_demo",
        "run_id": args.run_id,
        "inputs": {
            "tracking_metadata": str(tracking_path),
            "projected_points": str(projected_path),
            "video": str(video_path),
            "field_template": str(field_template_path),
        },
        "parameters": {
            "min_track_observations": args.min_track_observations,
            "max_observations_per_track": args.max_observations_per_track,
            "unknown_confidence_threshold": args.unknown_confidence_threshold,
            "official_score_threshold": args.official_score_threshold,
        },
        "scorebug": {"used": scorebug_used, "note": scorebug_note, "priors_path": args.scorebug_priors},
        "cluster_diagnostics": cluster_diagnostics,
        "track_assignments": assignment_list,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "track_team_assignments": str(team_output / "track_team_assignments.json"),
            "team_assignment_summary": str(team_output / "team_assignment_summary.json"),
            "team_color_cluster_preview": cluster_preview,
            "team_assignment_contact_sheet": contact,
            "team_assignment_overlay_mp4": assignment_overlay["mp4"],
            "team_assignment_overlay_mp4_error": assignment_overlay["mp4_error"],
            "team_assignment_overlay_gif": assignment_overlay["gif"],
        },
    }
    assignment_summary = {
        "status": "complete",
        "stage": "team_assignment_summary",
        "run_id": args.run_id,
        "output_dir": str(team_output),
        "counts": {
            "tracks_total": len(assignments),
            "team_a": class_counts.get("team_a", 0),
            "team_b": class_counts.get("team_b", 0),
            "official": class_counts.get("official", 0),
            "unknown": class_counts.get("unknown", 0),
            "tracks_with_color_observations": len(track_features),
            "detections_used": len(detections),
        },
        "confidence_distribution": confidence_summary,
        "scorebug_used": scorebug_used,
        "scorebug_note": scorebug_note,
        "artifacts": assignment_metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(team_output / "track_team_assignments.json", assignment_list)
    write_json(team_output / "team_assignment_metadata.json", assignment_metadata)
    write_json(team_output / "team_assignment_summary.json", assignment_summary)

    template = Image.open(field_template_path).convert("RGB")
    in_bounds_points = [point for point in projected_points if point.get("inside_field_template_bounds")]
    frame_metrics, polygon_artifacts = render_polygon_outputs(
        frames_by_source,
        frame_lookup,
        detections_by_frame_map,
        in_bounds_points,
        assignments,
        template,
        polygon_output,
        args.fps,
        args.trail_length,
        args.field_panel_width,
    )
    heatmap_artifacts = save_team_heatmaps(in_bounds_points, assignments, template, polygon_output, args.heatmap_sigma, args.heatmap_alpha)
    polygon_artifacts.update(heatmap_artifacts)
    metrics_payload = {
        "status": "complete",
        "stage": "team_polygon_metrics",
        "run_id": args.run_id,
        "frames": frame_metrics,
        "known_limitations": KNOWN_LIMITATIONS,
    }
    visible_team_a = sum(m["team_a_visible_player_count"] for m in frame_metrics)
    visible_team_b = sum(m["team_b_visible_player_count"] for m in frame_metrics)
    polygon_summary = {
        "status": "complete",
        "stage": "team_polygon_summary",
        "run_id": args.run_id,
        "output_dir": str(polygon_output),
        "counts": {
            "frames_rendered": len(frame_metrics),
            "projected_points_in_bounds": len(in_bounds_points),
            "team_a_projected_points_rendered": visible_team_a,
            "team_b_projected_points_rendered": visible_team_b,
            "team_a_tracks": class_counts.get("team_a", 0),
            "team_b_tracks": class_counts.get("team_b", 0),
            "official_tracks": class_counts.get("official", 0),
            "unknown_tracks": class_counts.get("unknown", 0),
        },
        "artifacts": polygon_artifacts,
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(polygon_output / "team_polygon_metrics.json", metrics_payload)
    write_json(polygon_output / "team_polygon_summary.json", polygon_summary)

    final = {
        "team_assignment": assignment_summary,
        "team_polygon": polygon_summary,
    }
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
