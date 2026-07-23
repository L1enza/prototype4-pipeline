#!/usr/bin/env python3
"""Polished NJ Devils hockey demo with stable tracks and team labels.

V2 is intentionally isolated from the working Phase A/B outputs.  It reruns the
existing detector on all source frames, improves association/persistence and
track-level classification, then reuses the validated broadcast-to-rink
homographies without modifying registration or manual annotations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_hockey_polygon_demo as phase_a
import run_hockey_rink_registration_and_polygons as phase_b


DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/njdevils.mp4"
DEFAULT_V1 = "outputs/njdevils/hockey_polygon_demo"
DEFAULT_OUTPUT = "outputs/njdevils/hockey_polygon_demo_v2"
DEFAULT_RINK = "assets/hockey/icerink.jpg"
DEFAULT_SAM3_REPO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3"
DEFAULT_SAM3_CACHE = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/.hf_cache/hub/models--facebook--sam3"
REPRESENTATIVE_FRAMES = [0, 70, 138, 210, 276]

TEAM_COLORS = {
    "devils": (42, 42, 235),
    "flyers": (35, 150, 255),
    "official": (35, 220, 220),
    "unknown": (155, 155, 155),
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--v1-dir", default=DEFAULT_V1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rink", default=DEFAULT_RINK)
    parser.add_argument("--max-track-age", type=int, default=12)
    parser.add_argument("--max-stitch-gap", type=int, default=15)
    parser.add_argument("--min-track-detections", type=int, default=3)
    return parser.parse_args()


def bbox_state(box: list[int] | list[float]) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in box]
    return np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0 + 1.0, y1 - y0 + 1.0], dtype=np.float64)


def state_box(state: np.ndarray, width: int, height: int) -> list[int]:
    cx, cy, box_width, box_height = [float(value) for value in state[:4]]
    box_width = max(8.0, box_width)
    box_height = max(16.0, box_height)
    return phase_a.clamp_box(
        [cx - box_width * 0.5, cy - box_height * 0.5, cx + box_width * 0.5, cy + box_height * 0.5],
        width,
        height,
    )


def histogram_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denominator) if denominator > 1e-9 else 0.0


def torso_crop_box(box: list[int], width: int, height: int) -> list[int]:
    """Upper torso only: deliberately excludes most pants, skates, and ice."""
    x0, y0, x1, y1 = box
    box_width = x1 - x0 + 1
    box_height = y1 - y0 + 1
    return phase_a.clamp_box(
        [x0 + 0.12 * box_width, y0 + 0.07 * box_height, x1 - 0.12 * box_width, y0 + 0.51 * box_height],
        width,
        height,
    )


def torso_v2_features(frame: np.ndarray, box: list[int], overlap: float = 0.0) -> dict:
    height, width = frame.shape[:2]
    tx0, ty0, tx1, ty1 = torso_crop_box(box, width, height)
    crop = frame[ty0 : ty1 + 1, tx0 : tx1 + 1]
    empty = {
        "white": 0.0, "red": 0.0, "orange": 0.0, "black": 0.0,
        "stripe": 0.0, "lab_red": 0.0, "blur_variance": 0.0,
        "quality": 0.0, "overlap": float(overlap), "histogram": None,
    }
    if crop.size == 0:
        return empty
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    white = (saturation < 67) & (value > 142)
    red = ((hue < 8) | (hue > 172)) & (saturation > 92) & (value > 55)
    orange = (hue >= 8) & (hue <= 29) & (saturation > 88) & (value > 66)
    black = value < 82
    column_white = np.mean(white, axis=0)
    column_black = np.mean(black, axis=0)
    stripe_signal = column_white - column_black
    transitions = float(np.mean(np.abs(np.diff(stripe_signal)) > 0.22)) if stripe_signal.size > 1 else 0.0
    balanced = min(float(np.mean(white)), float(np.mean(black))) * 2.0
    stripe = float(min(1.0, 2.7 * transitions + 0.58 * balanced))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    pixel_quality = min(1.0, crop.shape[0] * crop.shape[1] / 1700.0)
    blur_quality = min(1.0, blur_variance / 115.0)
    occlusion_quality = max(0.12, 1.0 - 1.35 * float(overlap))
    quality = float(pixel_quality * (0.25 + 0.75 * blur_quality) * occlusion_quality)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm > 1e-8:
        hist /= norm
    return {
        "white": float(np.mean(white)),
        "red": float(np.mean(red)),
        "orange": float(np.mean(orange)),
        "black": float(np.mean(black)),
        "stripe": stripe,
        "lab_red": float(np.clip((float(np.mean(lab[:, :, 1])) - 128.0) / 45.0, -1.0, 1.0)),
        "blur_variance": blur_variance,
        "quality": quality,
        "overlap": float(overlap),
        "histogram": hist.tolist(),
        "crop_box": [tx0, ty0, tx1, ty1],
    }


def feature_vector(features: dict) -> np.ndarray:
    return np.asarray([
        float(features["white"]),
        float(features["red"]),
        float(features["orange"]),
        float(features["black"]),
        float(features["lab_red"]),
    ], dtype=np.float64)


def softmax(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    exponents = {key: math.exp(min(30.0, value - maximum)) for key, value in scores.items()}
    total = sum(exponents.values())
    return {key: float(value / total) for key, value in exponents.items()}


def base_crop_probabilities(features: dict, legacy_team: str | None = None) -> dict[str, float]:
    white = float(features["white"])
    red = float(features["red"])
    orange = float(features["orange"])
    black = float(features["black"])
    stripe = float(features["stripe"])
    # Team evidence is torso-led.  Black is nearly neutral because both teams
    # wear black pants and residual pants pixels can enter small crops.
    scores = {
        "flyers": 3.25 * white + 0.55 * orange - 1.65 * red - 0.08 * black,
        "devils": 4.00 * red + 0.55 * max(0.0, features["lab_red"]) - 1.20 * white + 0.05 * black,
        "official": 2.70 * stripe + 0.75 * min(white, black) - 1.65 * (red + orange),
        "unknown": 0.36,
    }
    if legacy_team == "official" and red + orange < 0.08:
        scores["official"] += 0.72
    return softmax(scores)


def attach_legacy_matches(
    detections_by_frame: list[list[dict]],
    legacy_tracking: dict,
    legacy_assignments: dict[int, dict],
) -> int:
    legacy_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in legacy_tracking.get("detections", []):
        legacy_by_frame[int(row["source_frame_index"])].append(row)
    matches = 0
    for frame_index, detections in enumerate(detections_by_frame):
        candidates = []
        legacy_rows = legacy_by_frame.get(frame_index, [])
        for detection_index, detection in enumerate(detections):
            for legacy_index, legacy in enumerate(legacy_rows):
                overlap = phase_a.bbox_iou(detection["bbox"], legacy["bbox"])
                if overlap >= 0.45:
                    candidates.append((-overlap, detection_index, legacy_index))
        used_detections: set[int] = set()
        used_legacy: set[int] = set()
        for negative_overlap, detection_index, legacy_index in sorted(candidates):
            if detection_index in used_detections or legacy_index in used_legacy:
                continue
            legacy = legacy_rows[legacy_index]
            legacy_track = int(legacy["track_id"])
            assignment = legacy_assignments.get(legacy_track, {})
            detections[detection_index]["legacy_track_id"] = legacy_track
            detections[detection_index]["legacy_team"] = assignment.get("team", legacy.get("team", "unknown"))
            detections[detection_index]["legacy_match_iou"] = float(-negative_overlap)
            used_detections.add(detection_index)
            used_legacy.add(legacy_index)
            matches += 1
    return matches


def enrich_detections(frames: list[np.ndarray], detections_by_frame: list[list[dict]]) -> None:
    for frame, detections in zip(frames, detections_by_frame):
        for index, detection in enumerate(detections):
            overlap = max(
                [phase_a.bbox_iou(detection["bbox"], other["bbox"]) for other_index, other in enumerate(detections) if other_index != index]
                or [0.0]
            )
            features = torso_v2_features(frame, detection["bbox"], overlap)
            detection["torso_v2"] = features
            detection["torso_histogram"] = features["histogram"]
            detection["base_team_probabilities"] = base_crop_probabilities(features, detection.get("legacy_team"))


def predict_track_state(track: dict, frame_index: int) -> np.ndarray:
    delta = max(0, int(frame_index) - int(track["state_frame"]))
    predicted = np.asarray(track["state"], dtype=np.float64).copy()
    predicted[:4] += predicted[4:] * float(delta)
    predicted[2] = max(8.0, predicted[2])
    predicted[3] = max(16.0, predicted[3])
    return predicted


def update_track_state(track: dict, detection: dict, frame_index: int) -> None:
    measurement = bbox_state(detection["bbox"])
    predicted = predict_track_state(track, frame_index)
    previous_position = np.asarray(track["state"], dtype=np.float64)[:4]
    delta = max(1, int(frame_index) - int(track["state_frame"]))
    alpha = np.asarray([0.62, 0.62, 0.46, 0.46], dtype=np.float64)
    filtered = predicted[:4] + alpha * (measurement - predicted[:4])
    measured_velocity = (filtered - previous_position) / float(delta)
    velocity = 0.72 * np.asarray(track["state"], dtype=np.float64)[4:] + 0.28 * measured_velocity
    track["state"] = np.concatenate([filtered, velocity])
    track["state_frame"] = int(frame_index)
    track["last_detection_frame"] = int(frame_index)
    track["detections"].append(detection)
    old_hist = np.asarray(track["appearance"], dtype=np.float64)
    new_hist = np.asarray(detection["torso_histogram"], dtype=np.float64)
    blended = 0.86 * old_hist + 0.14 * new_hist
    norm = float(np.linalg.norm(blended))
    track["appearance"] = (blended / norm if norm > 1e-8 else blended).tolist()


def association_cost(track: dict, detection: dict, frame_index: int, frame_size: tuple[int, int]) -> tuple[float, dict] | None:
    width, height = frame_size
    predicted = predict_track_state(track, frame_index)
    predicted_box = state_box(predicted, width, height)
    measured = bbox_state(detection["bbox"])
    gap = max(1, frame_index - int(track["last_detection_frame"]))
    distance = float(np.linalg.norm(predicted[:2] - measured[:2]))
    gate = 34.0 + 11.0 * gap + 0.55 * max(predicted[3], measured[3])
    overlap = phase_a.bbox_iou(predicted_box, detection["bbox"])
    appearance = histogram_similarity(track.get("appearance"), detection.get("torso_histogram"))
    size_ratio = float(min(predicted[2] * predicted[3], measured[2] * measured[3]) / max(predicted[2] * predicted[3], measured[2] * measured[3], 1.0))
    if distance > gate and overlap < 0.015:
        return None
    if appearance < 0.18 and overlap < 0.05:
        return None
    legacy_votes = Counter(row.get("legacy_team") for row in track["detections"] if row.get("legacy_team"))
    legacy_team = legacy_votes.most_common(1)[0][0] if legacy_votes else None
    detection_legacy = detection.get("legacy_team")
    compatibility_penalty = 0.0
    if legacy_team in ("devils", "flyers", "official") and detection_legacy in ("devils", "flyers", "official") and legacy_team != detection_legacy:
        legacy_fraction = legacy_votes[legacy_team] / max(1, sum(legacy_votes.values()))
        if legacy_fraction >= 0.70:
            return None
        compatibility_penalty = 0.20
    cost = (
        0.43 * min(2.0, distance / max(gate, 1.0))
        + 0.23 * (1.0 - overlap)
        + 0.19 * (1.0 - max(0.0, min(1.0, appearance)))
        + 0.15 * (1.0 - size_ratio)
        + compatibility_penalty
    )
    if cost > 0.94:
        return None
    return float(cost), {
        "predicted_box": predicted_box,
        "distance": distance,
        "gate": gate,
        "iou": overlap,
        "appearance_similarity": appearance,
        "size_similarity": size_ratio,
        "gap": gap,
    }


def associate_with_motion_memory(
    detections_by_frame: list[list[dict]],
    frame_size: tuple[int, int],
    max_age: int = 12,
) -> tuple[dict[int, dict], list[dict]]:
    tracks: dict[int, dict] = {}
    next_track_id = 1
    events = []
    for frame_index, detections in enumerate(detections_by_frame):
        candidates = []
        for track_id, track in tracks.items():
            if frame_index - int(track["last_detection_frame"]) > max_age:
                continue
            for detection_index, detection in enumerate(detections):
                association = association_cost(track, detection, frame_index, frame_size)
                if association is not None:
                    cost, details = association
                    candidates.append((cost, track_id, detection_index, details))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for cost, track_id, detection_index, details in sorted(candidates, key=lambda row: row[0]):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            detection = detections[detection_index]
            detection["raw_v2_track_id"] = track_id
            detection["association_cost"] = float(cost)
            update_track_state(tracks[track_id], detection, frame_index)
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            events.append({"frame_index": frame_index, "track_id": track_id, "event": "matched", **details})
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track_id = next_track_id
            next_track_id += 1
            state = bbox_state(detection["bbox"])
            detection["raw_v2_track_id"] = track_id
            detection["association_cost"] = None
            tracks[track_id] = {
                "raw_track_id": track_id,
                "state": np.concatenate([state, np.zeros(4, dtype=np.float64)]),
                "state_frame": frame_index,
                "last_detection_frame": frame_index,
                "appearance": list(detection["torso_histogram"]),
                "detections": [detection],
            }
            events.append({"frame_index": frame_index, "track_id": track_id, "event": "created"})
    return tracks, events


def build_video_prototypes(detections_by_frame: list[list[dict]]) -> tuple[dict[str, list[float]], dict]:
    seeds: dict[str, list[np.ndarray]] = {"devils": [], "flyers": []}
    seed_sources = Counter()
    for detections in detections_by_frame:
        for detection in detections:
            features = detection["torso_v2"]
            if features["quality"] < 0.35:
                continue
            vector = feature_vector(features)
            legacy = detection.get("legacy_team")
            if features["white"] >= 0.34 and features["red"] <= 0.10:
                seeds["flyers"].append(vector)
                seed_sources["strict_flyers_white"] += 1
            elif features["red"] >= 0.13 and features["white"] <= 0.42:
                seeds["devils"].append(vector)
                seed_sources["strict_devils_red"] += 1
            elif legacy == "flyers" and features["white"] >= 0.22 and features["red"] < 0.12:
                seeds["flyers"].append(vector)
                seed_sources["legacy_supported_flyers"] += 1
            elif legacy == "devils" and features["red"] >= 0.075:
                seeds["devils"].append(vector)
                seed_sources["legacy_supported_devils"] += 1
    prototypes = {}
    for team in ("devils", "flyers"):
        if not seeds[team]:
            raise RuntimeError(f"No high-confidence {team} crops available for video-specific prototype")
        matrix = np.stack(seeds[team], axis=0)
        # Coordinate-wise median prevents a few bad legacy labels from pulling
        # the video-specific prototype toward the opposing jersey.
        prototypes[team] = np.median(matrix, axis=0).astype(float).tolist()
    return prototypes, {"seed_counts": {team: len(rows) for team, rows in seeds.items()}, "seed_sources": dict(seed_sources)}


def apply_video_prototypes(detections_by_frame: list[list[dict]], prototypes: dict[str, list[float]]) -> None:
    weights = np.asarray([2.0, 2.5, 1.0, 0.20, 0.8], dtype=np.float64)
    for detections in detections_by_frame:
        for detection in detections:
            features = detection["torso_v2"]
            probabilities = dict(detection["base_team_probabilities"])
            vector = feature_vector(features)
            similarities = {}
            for team in ("devils", "flyers"):
                delta = vector - np.asarray(prototypes[team], dtype=np.float64)
                similarities[team] = float(math.exp(-7.0 * float(np.sum(weights * delta * delta))))
            scores = {
                "devils": math.log(max(probabilities["devils"], 1e-8)) + 0.90 * similarities["devils"],
                "flyers": math.log(max(probabilities["flyers"], 1e-8)) + 0.90 * similarities["flyers"],
                "official": math.log(max(probabilities["official"], 1e-8)),
                "unknown": math.log(max(probabilities["unknown"], 1e-8)),
            }
            final = softmax(scores)
            detection["prototype_similarity"] = similarities
            detection["team_probabilities"] = final
            detection["raw_crop_label"] = max(final, key=final.get)


def track_summary_for_stitch(track: dict) -> dict:
    detections = sorted(track["detections"], key=lambda row: int(row["frame_index"]))
    histograms = [np.asarray(row["torso_histogram"], dtype=np.float64) for row in detections if row.get("torso_histogram")]
    appearance = np.mean(np.stack(histograms), axis=0) if histograms else np.zeros(72, dtype=np.float64)
    norm = float(np.linalg.norm(appearance))
    if norm > 1e-8:
        appearance /= norm
    probabilities = {
        team: float(np.mean([row["team_probabilities"][team] for row in detections]))
        for team in ("devils", "flyers", "official", "unknown")
    }
    legacy_track_votes = Counter(
        int(row["legacy_track_id"]) for row in detections if row.get("legacy_track_id") is not None
    )
    dominant_legacy_track = legacy_track_votes.most_common(1)[0] if legacy_track_votes else (None, 0)
    states = [bbox_state(row["bbox"]) for row in detections]
    velocity = np.zeros(4, dtype=np.float64)
    if len(states) >= 2:
        delta = max(1, int(detections[-1]["frame_index"]) - int(detections[-2]["frame_index"]))
        velocity = (states[-1] - states[-2]) / float(delta)
    return {
        "start": int(detections[0]["frame_index"]),
        "end": int(detections[-1]["frame_index"]),
        "start_state": states[0],
        "end_state": states[-1],
        "end_velocity": velocity,
        "appearance": appearance.tolist(),
        "probabilities": probabilities,
        "dominant_legacy_track_id": dominant_legacy_track[0],
        "dominant_legacy_track_fraction": float(dominant_legacy_track[1] / max(1, sum(legacy_track_votes.values()))),
    }


def stitch_track_fragments(
    tracks: dict[int, dict],
    max_gap: int = 15,
) -> tuple[dict[int, dict], list[dict], int]:
    summaries = {track_id: track_summary_for_stitch(track) for track_id, track in tracks.items()}
    candidates = []
    for left_id, left in summaries.items():
        for right_id, right in summaries.items():
            if left_id == right_id:
                continue
            gap = right["start"] - left["end"]
            if gap < 1 or gap > max_gap:
                continue
            predicted = np.asarray(left["end_state"]) + np.asarray(left["end_velocity"]) * float(gap)
            start = np.asarray(right["start_state"])
            distance = float(np.linalg.norm(predicted[:2] - start[:2]))
            gate = 24.0 + 10.0 * gap + 0.70 * max(predicted[3], start[3])
            appearance = histogram_similarity(left["appearance"], right["appearance"])
            size_similarity = float(min(predicted[2] * predicted[3], start[2] * start[3]) / max(predicted[2] * predicted[3], start[2] * start[3], 1.0))
            left_team = max(("devils", "flyers", "official"), key=lambda team: left["probabilities"][team])
            right_team = max(("devils", "flyers", "official"), key=lambda team: right["probabilities"][team])
            team_compatibility = 1.0 if left_team == right_team else 0.0
            shared_legacy_identity = bool(
                left["dominant_legacy_track_id"] is not None
                and left["dominant_legacy_track_id"] == right["dominant_legacy_track_id"]
                and left["dominant_legacy_track_fraction"] >= 0.60
                and right["dominant_legacy_track_fraction"] >= 0.60
            )
            minimum_appearance = 0.58 if shared_legacy_identity else 0.78
            if distance > gate or appearance < minimum_appearance or size_similarity < 0.42 or not team_compatibility:
                continue
            cost = 0.48 * distance / max(gate, 1.0) + 0.34 * (1.0 - appearance) + 0.18 * (1.0 - size_similarity)
            if cost <= (0.68 if shared_legacy_identity else 0.58):
                candidates.append((cost, left_id, right_id, {
                    "gap": gap, "predicted_center_distance": distance, "distance_gate": gate,
                    "appearance_similarity": appearance, "size_similarity": size_similarity,
                    "team_compatibility": bool(team_compatibility),
                    "shared_legacy_identity": shared_legacy_identity,
                }))
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    stitch_events = []
    for cost, left_id, right_id, details in sorted(candidates, key=lambda row: row[0]):
        if left_id in outgoing or right_id in incoming:
            continue
        outgoing[left_id] = right_id
        incoming[right_id] = left_id
        stitch_events.append({"from_raw_track_id": left_id, "to_raw_track_id": right_id, "cost": float(cost), **details})

    chains = []
    for track_id in sorted(tracks, key=lambda value: summaries[value]["start"]):
        if track_id in incoming:
            continue
        chain = [track_id]
        while chain[-1] in outgoing:
            chain.append(outgoing[chain[-1]])
        chains.append(chain)
    stitched: dict[int, dict] = {}
    for stable_id, chain in enumerate(chains, start=1):
        detections = []
        for raw_id in chain:
            detections.extend(tracks[raw_id]["detections"])
        detections.sort(key=lambda row: int(row["frame_index"]))
        for detection in detections:
            detection["track_id"] = stable_id
        stitched[stable_id] = {
            "track_id": stable_id,
            "raw_track_ids": chain,
            "detections": detections,
        }

    # Remaining switch count is an explicit heuristic, not ground truth: count
    # strong non-overlapping transition candidates that were not stitched.
    selected = {(event["from_raw_track_id"], event["to_raw_track_id"]) for event in stitch_events}
    remaining = sum(
        1 for cost, left_id, right_id, _details in candidates
        if (left_id, right_id) not in selected and left_id not in outgoing and right_id not in incoming
    )
    return stitched, stitch_events, remaining


def track_renderable(track: dict, minimum_detections: int = 3) -> bool:
    detections = track["detections"]
    heights = [row["bbox"][3] - row["bbox"][1] + 1 for row in detections]
    areas = [(row["bbox"][2] - row["bbox"][0] + 1) * height for row, height in zip(detections, heights)]
    track["median_bbox_height"] = float(np.median(heights))
    track["median_bbox_area"] = float(np.median(areas))
    return bool(len(detections) >= minimum_detections and track["median_bbox_height"] >= 38.0 and track["median_bbox_area"] >= 1300.0)


def assign_stable_teams(tracks: dict[int, dict], minimum_detections: int = 3) -> tuple[dict[int, dict], dict]:
    assignments = {}
    low_quality_ignored = 0
    raw_transitions = 0
    tracks_with_raw_flicker = 0
    for track_id, track in tracks.items():
        detections = sorted(track["detections"], key=lambda row: int(row["frame_index"]))
        labels = [row["raw_crop_label"] for row in detections]
        transitions = sum(a != b for a, b in zip(labels, labels[1:]))
        raw_transitions += transitions
        tracks_with_raw_flicker += int(transitions > 0)
        totals = {team: 0.0 for team in ("devils", "flyers", "official", "unknown")}
        total_weight = 0.0
        used = 0
        legacy_votes = Counter()
        for detection in detections:
            features = detection["torso_v2"]
            if detection.get("legacy_team"):
                legacy_votes[detection["legacy_team"]] += 1
            if features["quality"] < 0.25:
                low_quality_ignored += 1
                continue
            weight = float(features["quality"])
            total_weight += weight
            used += 1
            for team in totals:
                totals[team] += weight * float(detection["team_probabilities"][team])
        averages = {team: totals[team] / max(total_weight, 1e-8) for team in totals}
        ordered = sorted(averages.items(), key=lambda item: item[1], reverse=True)
        best_team, best_score = ordered[0]
        margin = float(best_score - ordered[1][1])
        legacy_official_fraction = legacy_votes["official"] / max(1, sum(legacy_votes.values()))
        legacy_devils_fraction = legacy_votes["devils"] / max(1, sum(legacy_votes.values()))
        legacy_total = max(1, sum(legacy_votes.values()))
        legacy_dominant_team, legacy_dominant_count = legacy_votes.most_common(1)[0] if legacy_votes else (None, 0)
        legacy_dominant_fraction = legacy_dominant_count / legacy_total
        mean_features = {
            key: float(np.mean([row["torso_v2"][key] for row in detections]))
            for key in ("white", "red", "orange", "black", "stripe", "quality", "blur_variance")
        }
        # Preserve successful official identification unless sustained red/orange
        # torso evidence clearly contradicts it.
        if legacy_official_fraction >= 0.34 and mean_features["red"] + mean_features["orange"] < 0.095:
            best_team = "official"
            best_score = max(best_score, 0.68)
            margin = max(margin, 0.18)
        elif averages["official"] >= 0.43 and mean_features["stripe"] >= 0.30 and mean_features["red"] < 0.06:
            best_team = "official"
        elif legacy_devils_fraction >= 0.25 and mean_features["red"] >= 0.16:
            # Red is primary Devils evidence even if white ice or an opponent
            # leaks into a torso crop during overlap.
            best_team = "devils"
            best_score = max(best_score, 0.68 + 0.16 * min(1.0, mean_features["red"] / 0.30))
            margin = max(margin, 0.15)
        elif legacy_dominant_team == "devils" and legacy_dominant_fraction >= 0.55 and mean_features["red"] >= 0.055:
            # White ice leaking into a small torso box must not override a
            # sustained red-jersey track identity.
            best_team = "devils"
            best_score = max(best_score, 0.62 + 0.20 * min(1.0, mean_features["red"] / 0.25))
            margin = max(margin, 0.13)
        elif (
            legacy_dominant_team == "flyers"
            and legacy_dominant_fraction >= 0.55
            and mean_features["white"] >= 0.18
            and mean_features["red"] < 0.20
        ):
            best_team = "flyers"
            best_score = max(best_score, 0.62 + 0.16 * min(1.0, mean_features["white"] / 0.55))
            margin = max(margin, 0.12)
        elif legacy_dominant_team == "unknown" and legacy_dominant_fraction >= 0.68:
            # Most persistent board/ice artifacts were already rejected by the
            # Phase A track review.  V2 requires strong red evidence to recover
            # one as a player; white background alone is not sufficient.
            if mean_features["red"] >= 0.16 and mean_features["quality"] >= 0.45:
                best_team = "devils"
                margin = max(margin, 0.11)
            else:
                best_team = "unknown"
        elif legacy_dominant_team == "devils" and legacy_dominant_fraction >= 0.55 and mean_features["red"] < 0.055:
            # White-only conflicting evidence is ambiguous, not sufficient to
            # silently convert a formerly red track into a Flyer.
            best_team = "unknown"
        if used < 2 or best_score < 0.37 or margin < 0.065:
            best_team = "unknown"
        if best_team == "devils" and mean_features["red"] < 0.040:
            best_team = "unknown"
        if best_team == "flyers" and mean_features["white"] < 0.115:
            best_team = "unknown"
        renderable = track_renderable(track, minimum_detections)
        if not renderable:
            best_team = "unknown"
        elif best_team == "official" and (
            track["median_bbox_height"] < 80.0 or track["median_bbox_area"] < 5000.0
        ):
            # Thin rink markings and glass supports can have a stripe-like
            # signature. Preserve only person-scale official tracks.
            best_team = "unknown"
        elif best_team == "flyers" and mean_features["black"] < 0.03 and mean_features["red"] < 0.02:
            # Bright boards and kickplate graphics can be white/orange but do
            # not contain the Flyers' black or red uniform detail.
            best_team = "unknown"
        confidence = float(min(0.99, max(0.30, 0.48 + 0.48 * margin + 0.10 * best_score)))
        assignments[track_id] = {
            "track_id": track_id,
            "raw_track_ids": list(track["raw_track_ids"]),
            "team": best_team,
            "confidence": confidence,
            "locked_label": True,
            "label_change_policy": "static track-level lock; no frame-level changes",
            "detection_count": len(detections),
            "strong_observations_used": used,
            "frame_start": int(detections[0]["frame_index"]),
            "frame_end": int(detections[-1]["frame_index"]),
            "duration_frames": int(detections[-1]["frame_index"] - detections[0]["frame_index"] + 1),
            "renderable_on_ice_track": renderable,
            "temporal_probabilities": averages,
            "probability_margin": margin,
            "mean_torso_features": mean_features,
            "legacy_team_votes": dict(legacy_votes),
            "raw_per_crop_label_transitions": transitions,
            "stabilized_label_transitions": 0,
            "median_bbox_height": track["median_bbox_height"],
            "median_bbox_area": track["median_bbox_area"],
            "method": "video_prototype_torso_color_temporal_lock_v2",
        }
    return assignments, {
        "raw_per_crop_label_transitions": raw_transitions,
        "tracks_with_raw_crop_label_flicker": tracks_with_raw_flicker,
        "stabilized_label_transitions": 0,
        "low_quality_crops_ignored": low_quality_ignored,
    }


def smooth_and_fill_tracks(
    tracks: dict[int, dict],
    assignments: dict[int, dict],
    frame_size: tuple[int, int],
    max_gap: int = 15,
) -> tuple[dict[int, list[dict]], dict]:
    width, height = frame_size
    by_frame: dict[int, list[dict]] = defaultdict(list)
    gap_events = 0
    predicted_boxes = 0
    raw_jitter = []
    smooth_jitter = []
    for track_id, track in tracks.items():
        if not assignments[track_id]["renderable_on_ice_track"]:
            continue
        detections = sorted(track["detections"], key=lambda row: int(row["frame_index"]))
        raw_states = np.stack([bbox_state(row["bbox"]) for row in detections], axis=0)
        median_states = raw_states.copy()
        for index in range(len(raw_states)):
            lo, hi = max(0, index - 2), min(len(raw_states), index + 3)
            median_states[index] = np.median(raw_states[lo:hi], axis=0)
        smooth_states = median_states.copy()
        for index in range(1, len(smooth_states)):
            gap = max(1, int(detections[index]["frame_index"]) - int(detections[index - 1]["frame_index"]))
            alpha_center = min(0.76, 0.48 + 0.04 * (gap - 1))
            alpha_size = min(0.68, 0.38 + 0.04 * (gap - 1))
            smooth_states[index, :2] = (1.0 - alpha_center) * smooth_states[index - 1, :2] + alpha_center * median_states[index, :2]
            target_size = median_states[index, 2:]
            previous_size = smooth_states[index - 1, 2:]
            target_size = np.clip(target_size, previous_size * 0.78, previous_size * 1.28)
            smooth_states[index, 2:] = (1.0 - alpha_size) * previous_size + alpha_size * target_size
        if len(raw_states) >= 3:
            raw_acceleration = np.diff(raw_states[:, :2], n=2, axis=0)
            smooth_acceleration = np.diff(smooth_states[:, :2], n=2, axis=0)
            scale = np.maximum(raw_states[1:-1, 3], 1.0)
            raw_jitter.extend((np.linalg.norm(raw_acceleration, axis=1) / scale).tolist())
            smooth_jitter.extend((np.linalg.norm(smooth_acceleration, axis=1) / scale).tolist())
        for detection, state in zip(detections, smooth_states):
            frame_index = int(detection["frame_index"])
            display = {
                "frame_index": frame_index,
                "track_id": track_id,
                "bbox": state_box(state, width, height),
                "raw_bbox": list(detection["bbox"]),
                "detected": True,
                "predicted": False,
                "detector_score": float(detection.get("score") or 0.0),
                "source": detection.get("source"),
                "team": assignments[track_id]["team"],
                "team_confidence": assignments[track_id]["confidence"],
                "torso_v2": detection["torso_v2"],
                "legacy_track_id": detection.get("legacy_track_id"),
                "legacy_team": detection.get("legacy_team"),
                "raw_crop_label": detection["raw_crop_label"],
                "team_probabilities": detection["team_probabilities"],
            }
            display["footpoint"] = [float(value) for value in phase_a.footpoint(display["bbox"])]
            by_frame[frame_index].append(display)
        for index in range(len(detections) - 1):
            left_frame = int(detections[index]["frame_index"])
            right_frame = int(detections[index + 1]["frame_index"])
            gap = right_frame - left_frame
            if gap <= 1 or gap - 1 > max_gap:
                continue
            gap_events += 1
            for frame_index in range(left_frame + 1, right_frame):
                fraction = (frame_index - left_frame) / float(gap)
                state = (1.0 - fraction) * smooth_states[index] + fraction * smooth_states[index + 1]
                display = {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "bbox": state_box(state, width, height),
                    "raw_bbox": None,
                    "detected": False,
                    "predicted": True,
                    "detector_score": None,
                    "source": "motion_interpolation_between_fresh_detections",
                    "team": assignments[track_id]["team"],
                    "team_confidence": float(assignments[track_id]["confidence"] * math.exp(-min(frame_index - left_frame, right_frame - frame_index) / 18.0)),
                    "prediction_gap_length": gap - 1,
                }
                display["footpoint"] = [float(value) for value in phase_a.footpoint(display["bbox"])]
                by_frame[frame_index].append(display)
                predicted_boxes += 1
    return dict(by_frame), {
        "short_detection_gap_events_bridged": gap_events,
        "predicted_boxes_rendered": predicted_boxes,
        "mean_normalized_center_jitter_before": float(np.mean(raw_jitter)) if raw_jitter else 0.0,
        "mean_normalized_center_jitter_after": float(np.mean(smooth_jitter)) if smooth_jitter else 0.0,
        "jitter_reduction_percent": float(100.0 * (1.0 - np.mean(smooth_jitter) / max(np.mean(raw_jitter), 1e-8))) if raw_jitter else 0.0,
    }


def draw_dashed_rectangle(image: np.ndarray, box: list[int], color: tuple[int, int, int], thickness: int = 2, dash: int = 10) -> None:
    x0, y0, x1, y1 = box
    for start in range(x0, x1 + 1, 2 * dash):
        cv2.line(image, (start, y0), (min(x1, start + dash), y0), color, thickness, cv2.LINE_AA)
        cv2.line(image, (start, y1), (min(x1, start + dash), y1), color, thickness, cv2.LINE_AA)
    for start in range(y0, y1 + 1, 2 * dash):
        cv2.line(image, (x0, start), (x0, min(y1, start + dash)), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, start), (x1, min(y1, start + dash)), color, thickness, cv2.LINE_AA)


def draw_track(image: np.ndarray, row: dict) -> None:
    team = row["team"]
    color = TEAM_COLORS[team]
    box = row["bbox"]
    if row["predicted"]:
        dim = tuple(int(0.58 * value) for value in color)
        draw_dashed_rectangle(image, box, dim, thickness=2)
        suffix = " PRED"
        label_color = dim
    else:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 3, cv2.LINE_AA)
        suffix = ""
        label_color = color
    phase_b.draw_text_box(
        image,
        f"T{row['track_id']} {team.upper()} {float(row['team_confidence']):.2f}{suffix}",
        (box[0], max(20, box[1])),
        label_color,
        0.43,
    )


def v2_header(image: np.ndarray, frame_index: int, counts: Counter, title: str, extra: str = "") -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 78), (7, 7, 7), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
    cv2.putText(image, title, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"frame {frame_index:03d} | NJD {counts['devils']} | PHI {counts['flyers']} | officials {counts['official']} | unknown {counts['unknown']} | predicted {counts['predicted']}",
        (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 2, cv2.LINE_AA,
    )
    if extra:
        cv2.putText(image, extra, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 220, 255), 1, cv2.LINE_AA)


def segmentation_audit(sam3_repo: Path, sam3_cache: Path = Path(DEFAULT_SAM3_CACHE)) -> dict:
    checkpoint_files = []
    if sam3_repo.exists():
        for suffix in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
            checkpoint_files.extend(str(path) for path in sam3_repo.rglob(suffix))
    if sam3_cache.exists():
        checkpoint_files.extend(str(path) for path in sam3_cache.glob("snapshots/*/sam3.pt") if path.exists())
    checkpoint_files = sorted(set(checkpoint_files))
    checkpoint_details = []
    for checkpoint_string in checkpoint_files:
        checkpoint = Path(checkpoint_string)
        try:
            resolved = checkpoint.resolve()
            checkpoint_details.append({
                "path": str(checkpoint),
                "resolved_path": str(resolved),
                "size_bytes": int(resolved.stat().st_size),
            })
        except OSError:
            checkpoint_details.append({"path": str(checkpoint), "resolved_path": None, "size_bytes": None})
    try:
        import torch
        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        cuda_devices = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - environment diagnostic
        torch_version = None
        cuda_available = False
        cuda_devices = 0
        torch_error = repr(exc)
    else:
        torch_error = None
    try:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
        nvidia_smi = {"returncode": probe.returncode, "stdout": probe.stdout.strip(), "stderr": probe.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        nvidia_smi = {"returncode": None, "error": repr(exc)}
    packages = {
        name: bool(importlib.util.find_spec(name))
        for name in ("sam3", "sam2", "segment_anything", "ultralytics", "supervision")
    }
    source_package_present = bool((sam3_repo / "sam3" / "__init__.py").exists())
    usable = bool(
        cuda_available
        and sam3_repo.exists()
        and checkpoint_files
        and (packages["sam3"] or source_package_present)
    )
    blockers = []
    if not cuda_available:
        blockers.append("torch_cuda_unavailable")
    if not checkpoint_files:
        blockers.append("no_local_sam_checkpoint")
    if not packages["sam3"] and not source_package_present:
        blockers.append("sam3_python_source_unavailable")
    if not packages["sam2"] and not packages["segment_anything"]:
        blockers.append("no_installed_sam2_or_segment_anything_fallback")
    return {
        "sam3_ran": False,
        "segmentation_available": usable,
        "sam3_source_repository": str(sam3_repo),
        "sam3_source_repository_exists": sam3_repo.exists(),
        "sam3_source_package_present": source_package_present,
        "sam3_checkpoint_cache": str(sam3_cache),
        "checkpoint_files": checkpoint_files,
        "checkpoint_details": checkpoint_details,
        "torch_version": torch_version,
        "torch_import_error": torch_error,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_devices,
        "nvidia_smi": nvidia_smi,
        "installed_packages": packages,
        "blockers": blockers,
        "fallback_behavior": "tracking and labels rendered without fabricated silhouette masks",
    }


def crop_tile(frame: np.ndarray, row: dict, width: int = 190, height: int = 132) -> np.ndarray:
    box = row["torso_v2"].get("crop_box") or torso_crop_box(row["bbox"], frame.shape[1], frame.shape[0])
    x0, y0, x1, y1 = [int(value) for value in box]
    crop = frame[y0 : y1 + 1, x0 : x1 + 1]
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    if crop.size:
        scale = min(width / crop.shape[1], (height - 34) / crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
        x = (width - resized.shape[1]) // 2
        y = 32 + (height - 32 - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    label = f"T{row['track_id']} {row['team'].upper()} {row['team_confidence']:.2f}"
    cv2.rectangle(canvas, (0, 0), (width, 31), (12, 12, 12), -1)
    cv2.putText(canvas, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEAM_COLORS[row["team"]], 1, cv2.LINE_AA)
    return canvas


def team_contact_sheet(frames: list[np.ndarray], display_by_frame: dict[int, list[dict]]) -> np.ndarray:
    categories = ["devils", "flyers", "official", "unknown", "disagreement"]
    candidates: dict[str, list[tuple[float, int, dict]]] = {category: [] for category in categories}
    per_track_best: dict[str, dict[int, tuple[float, int, dict]]] = {category: {} for category in categories}
    for frame_index, rows in display_by_frame.items():
        for row in rows:
            if row["predicted"] or "torso_v2" not in row:
                continue
            raw_box = row.get("raw_bbox") or row["bbox"]
            raw_width = max(1, raw_box[2] - raw_box[0] + 1)
            raw_height = max(1, raw_box[3] - raw_box[1] + 1)
            aspect = raw_height / raw_width
            upright = 1.12 <= aspect <= 3.50 and raw_width >= 16 and raw_width * raw_height >= 1200
            features = row["torso_v2"]
            team_compatible = (
                (row["team"] == "devils" and features["red"] >= 0.055)
                or (
                    row["team"] == "flyers"
                    and features["white"] >= 0.30
                    and features["red"] <= 0.16
                    and (features["black"] >= 0.03 or features["red"] >= 0.03)
                )
                or (row["team"] == "official" and features["stripe"] >= 0.20 and features["white"] >= 0.10 and features["black"] >= 0.08)
                or row["team"] == "unknown"
            )
            if not upright or not team_compatible:
                continue
            score = float(row["team_confidence"] * row["torso_v2"]["quality"])
            previous = per_track_best[row["team"]].get(int(row["track_id"]))
            if previous is None or score > previous[0]:
                per_track_best[row["team"]][int(row["track_id"])] = (score, frame_index, row)
            if row.get("legacy_team") and row["legacy_team"] != row["team"]:
                previous = per_track_best["disagreement"].get(int(row["track_id"]))
                if previous is None or score > previous[0]:
                    per_track_best["disagreement"][int(row["track_id"])] = (score, frame_index, row)
    for category in categories:
        candidates[category] = list(per_track_best[category].values())
    columns, tile_width, tile_height = 6, 190, 160
    sheet = np.full((len(categories) * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for category_index, category in enumerate(categories):
        chosen = sorted(candidates[category], key=lambda item: item[0], reverse=True)[:columns]
        for column, (_score, frame_index, row) in enumerate(chosen):
            tile = crop_tile(frames[frame_index], row, tile_width, tile_height)
            sheet[category_index * tile_height : (category_index + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile
        cv2.putText(
            sheet, category.upper(), (7, category_index * tile_height + tile_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return sheet


def decode_capture_frame(cap: cv2.VideoCapture, fallback: np.ndarray) -> np.ndarray:
    ok, frame = cap.read()
    return frame if ok and frame is not None else fallback.copy()


def main() -> int:
    args = parse_args()
    video_path = project_path(args.video)
    v1_dir = project_path(args.v1_dir)
    output_dir = project_path(args.output_dir)
    rink_path = project_path(args.rink)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "full_resolution_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = phase_b.decode_video(video_path)
    if len(frames) != 277:
        raise ValueError(f"Expected 277 source frames, decoded {len(frames)}")
    frame_size = (metadata["width"], metadata["height"])
    fps = float(metadata["fps"])
    rink = cv2.imread(str(rink_path), cv2.IMREAD_COLOR)
    if rink is None:
        raise FileNotFoundError(rink_path)
    canonical_size = (rink.shape[1], rink.shape[0])
    playable_contour = phase_b.rink_playable_contour(rink)

    legacy_tracking = load_json(v1_dir / "tracking_results.json")
    legacy_assignment_payload = load_json(v1_dir / "team_assignments.json")
    legacy_assignments = {int(row["track_id"]): row for row in legacy_assignment_payload["assignments"]}
    registration_payload = load_json(v1_dir / "per_frame_rink_homographies.json")
    registrations = {int(row["frame_index"]): row for row in registration_payload["frames"]}
    if sorted(registrations) != list(range(277)):
        raise ValueError("V1 registration does not contain all 277 source-frame indices")

    detector = phase_a.HockeyDetector()
    detections_by_frame: list[list[dict]] = []
    outside_excluded = 0
    for frame_index, frame in enumerate(frames):
        detections, excluded = detector.detect(frame)
        for detection in detections:
            detection["frame_index"] = frame_index
            detection["source_frame_index"] = frame_index
        detections_by_frame.append(detections)
        outside_excluded += excluded
        if frame_index % 25 == 0 or frame_index == len(frames) - 1:
            print(f"V2 detection frame {frame_index + 1}/{len(frames)}", flush=True)

    legacy_matches = attach_legacy_matches(detections_by_frame, legacy_tracking, legacy_assignments)
    enrich_detections(frames, detections_by_frame)
    raw_tracks, association_events = associate_with_motion_memory(detections_by_frame, frame_size, args.max_track_age)
    prototypes, prototype_diagnostics = build_video_prototypes(detections_by_frame)
    apply_video_prototypes(detections_by_frame, prototypes)
    stitched_tracks, stitch_events, remaining_switch_candidates = stitch_track_fragments(raw_tracks, args.max_stitch_gap)
    assignments, classification_stats = assign_stable_teams(stitched_tracks, args.min_track_detections)
    display_by_frame, smoothing_stats = smooth_and_fill_tracks(
        stitched_tracks, assignments, frame_size, args.max_stitch_gap
    )

    segmentation = segmentation_audit(Path(DEFAULT_SAM3_REPO))
    write_json(output_dir / "segmentation_status.json", segmentation)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    paths = {
        "boxes": output_dir / "broadcast_player_boxes_stabilized.mp4",
        "masks": output_dir / "broadcast_player_masks_stabilized.mp4",
        "broadcast_polygons": output_dir / "broadcast_team_polygons_stabilized.mp4",
        "rink_polygons": output_dir / "rink_team_polygons_stabilized.mp4",
        "side_by_side": output_dir / "side_by_side_team_polygons_stabilized.mp4",
        "comparison": output_dir / "tracking_before_after_comparison.mp4",
        "showcase": output_dir / "final_hockey_showcase_v2.mp4",
    }
    canonical_panel_width = int(round(canonical_size[0] * frame_size[1] / canonical_size[1]))
    writers = {
        "boxes": cv2.VideoWriter(str(paths["boxes"]), fourcc, fps, frame_size),
        "masks": cv2.VideoWriter(str(paths["masks"]), fourcc, fps, frame_size),
        "broadcast_polygons": cv2.VideoWriter(str(paths["broadcast_polygons"]), fourcc, fps, frame_size),
        "rink_polygons": cv2.VideoWriter(str(paths["rink_polygons"]), fourcc, fps, canonical_size),
        "side_by_side": cv2.VideoWriter(str(paths["side_by_side"]), fourcc, fps, (frame_size[0] + canonical_panel_width, frame_size[1])),
        "comparison": cv2.VideoWriter(str(paths["comparison"]), fourcc, fps, (frame_size[0] * 2, frame_size[1])),
        "showcase": cv2.VideoWriter(str(paths["showcase"]), fourcc, fps, frame_size),
    }
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("Could not open all V2 video writers")
    old_cap = cv2.VideoCapture(str(v1_dir / "broadcast_player_boxes.mp4"))
    if not old_cap.isOpened():
        raise FileNotFoundError(v1_dir / "broadcast_player_boxes.mp4")

    projected_observations = []
    rink_heat_points = {"devils": [], "flyers": []}
    render_stats = Counter()
    representative_boxes = []
    representative_masks = []
    representative_polygons = []
    comparison_images = []
    visual_case_scores = []
    for frame_index, raw in enumerate(frames):
        record = registrations[frame_index]
        rows = sorted(display_by_frame.get(frame_index, []), key=lambda row: row["bbox"][3])
        boxes_frame = raw.copy()
        masks_frame = raw.copy()
        broadcast_frame = raw.copy()
        showcase_frame = raw.copy()
        rink_frame = rink.copy()
        counts = Counter()
        team_broadcast = {"devils": [], "flyers": []}
        team_rink = {"devils": [], "flyers": []}
        if record.get("status") == "ok" and record.get("homography_matrix") is not None:
            phase_b.draw_visible_region(rink_frame, record["homography_matrix"], frame_size)
            render_stats["registration_ok_frames"] += 1
        else:
            render_stats["registration_rejected_frames"] += 1
        for row in rows:
            draw_track(boxes_frame, row)
            draw_track(masks_frame, row)
            draw_track(broadcast_frame, row)
            draw_track(showcase_frame, row)
            counts[row["team"]] += 1
            counts["predicted"] += int(row["predicted"])
            if row["predicted"]:
                render_stats["predicted_boxes_drawn"] += 1
                continue
            if row["team"] == "official":
                render_stats["official_observations_excluded"] += 1
                continue
            if row["team"] not in ("devils", "flyers"):
                render_stats["unknown_observations_excluded"] += 1
                continue
            if record.get("status") != "ok" or record.get("homography_matrix") is None:
                render_stats["fresh_team_observations_rejected_by_registration"] += 1
                continue
            foot = tuple(float(value) for value in row["footpoint"])
            projected = phase_b.project_broadcast_point(record["homography_matrix"], foot)
            if projected is None:
                render_stats["nonfinite_projections"] += 1
                continue
            if not phase_b.point_inside_rink(playable_contour, projected):
                render_stats["out_of_rink_projections"] += 1
                continue
            team = row["team"]
            team_broadcast[team].append(foot)
            team_rink[team].append(projected)
            rink_heat_points[team].append(projected)
            render_stats[f"{team}_projected_observations"] += 1
            color = TEAM_COLORS[team]
            cv2.circle(broadcast_frame, (int(round(foot[0])), int(round(foot[1]))), 5, color, -1, cv2.LINE_AA)
            cv2.circle(showcase_frame, (int(round(foot[0])), int(round(foot[1]))), 5, color, -1, cv2.LINE_AA)
            cv2.circle(rink_frame, (int(round(projected[0])), int(round(projected[1]))), 8, color, -1, cv2.LINE_AA)
            projected_observations.append({
                "source_frame_index": frame_index,
                "track_id": int(row["track_id"]),
                "team": team,
                "team_confidence": float(row["team_confidence"]),
                "broadcast_footpoint": list(foot),
                "canonical_rink_xy": list(projected),
                "registration_status": record["status"],
                "registration_confidence": float(record.get("confidence") or 0.0),
                "reference_frame_index": record.get("reference_frame_index"),
                "projection_direction": "fresh_smoothed_broadcast_footpoint_to_rink_uses_stored_H_directly",
                "predicted_track_box": False,
            })
        if record.get("status") == "ok":
            for team in ("devils", "flyers"):
                stable = phase_b.team_polygon_is_stable(team_broadcast[team], team_rink[team])
                if stable:
                    phase_b.draw_team_polygon(broadcast_frame, team_broadcast[team], TEAM_COLORS[team])
                    phase_b.draw_team_polygon(showcase_frame, team_broadcast[team], TEAM_COLORS[team])
                    phase_b.draw_team_polygon(rink_frame, team_rink[team], TEAM_COLORS[team])
                    render_stats[f"frames_with_{team}_polygon"] += 1
                else:
                    render_stats[f"frames_with_insufficient_{team}_points"] += 1
        else:
            phase_b.rejection_banner(broadcast_frame, record)
            phase_b.rejection_banner(showcase_frame, record)
            phase_b.rejection_banner(rink_frame, record)

        v2_header(boxes_frame, frame_index, counts, "HOCKEY TRACKING V2 - STABILIZED BOXES", "solid=fresh detection | dashed/dim=motion prediction")
        v2_header(masks_frame, frame_index, counts, "HOCKEY TRACKING V2 - SEGMENTATION LAYER", "SAM3 unavailable: no fabricated masks; stabilized tracking retained")
        cv2.rectangle(masks_frame, (20, masks_frame.shape[0] - 55), (masks_frame.shape[1] - 20, masks_frame.shape[0] - 15), (25, 25, 25), -1)
        cv2.putText(masks_frame, "SEGMENTATION UNAVAILABLE ON THIS HOST - TRACKING-ONLY VIEW", (40, masks_frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (180, 210, 255), 2, cv2.LINE_AA)
        v2_header(broadcast_frame, frame_index, counts, "RINK-REGISTERED TEAM POLYGONS V2", "predicted boxes are display-only and never projected")
        v2_header(rink_frame, frame_index, counts, "CANONICAL RINK POLYGONS V2", "blue polygon = current broadcast footprint")
        v2_header(showcase_frame, frame_index, counts, "NJ DEVILS HOCKEY SHOWCASE V2", "stable labels | fresh detections only in polygons | dynamic rink registration")

        old_frame = decode_capture_frame(old_cap, raw)
        cv2.rectangle(old_frame, (0, 0), (old_frame.shape[1], 38), (5, 5, 5), -1)
        cv2.putText(old_frame, "V1: FRAGMENTED / UNSMOOTHED", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2, cv2.LINE_AA)
        comparison = np.hstack([old_frame, boxes_frame])

        writers["boxes"].write(boxes_frame)
        writers["masks"].write(masks_frame)
        writers["broadcast_polygons"].write(broadcast_frame)
        writers["rink_polygons"].write(rink_frame)
        rink_resized = cv2.resize(rink_frame, (canonical_panel_width, frame_size[1]), interpolation=cv2.INTER_AREA)
        writers["side_by_side"].write(np.hstack([broadcast_frame, rink_resized]))
        writers["comparison"].write(comparison)
        writers["showcase"].write(showcase_frame)

        max_overlap = max(
            [phase_a.bbox_iou(a["bbox"], b["bbox"]) for i, a in enumerate(rows) for b in rows[i + 1 :]] or [0.0]
        )
        predicted_count = sum(row["predicted"] for row in rows)
        official_count = sum(row["team"] == "official" for row in rows)
        visual_case_scores.append((max_overlap + 0.18 * predicted_count + 0.12 * official_count, frame_index))
        if frame_index in REPRESENTATIVE_FRAMES:
            representative_boxes.append(boxes_frame.copy())
            representative_masks.append(masks_frame.copy())
            representative_polygons.append(broadcast_frame.copy())
            comparison_images.append(comparison.copy())
            cv2.imwrite(str(diagnostic_dir / f"frame_{frame_index:06d}_stabilized_boxes.png"), boxes_frame)
            cv2.imwrite(str(diagnostic_dir / f"frame_{frame_index:06d}_team_polygons.png"), broadcast_frame)
            cv2.imwrite(str(diagnostic_dir / f"frame_{frame_index:06d}_canonical_rink.png"), rink_frame)

    old_cap.release()
    for writer in writers.values():
        writer.release()

    # Additional full-resolution cases prioritize overlaps, prediction gaps,
    # and official interactions for the required visual audit.
    special_cases = []
    for _score, frame_index in sorted(visual_case_scores, reverse=True):
        if all(abs(frame_index - chosen) >= 8 for chosen in special_cases):
            special_cases.append(frame_index)
        if len(special_cases) >= 6:
            break
    boxes_cap = cv2.VideoCapture(str(paths["boxes"]))
    index = 0
    while boxes_cap.isOpened():
        ok, frame = boxes_cap.read()
        if not ok:
            break
        if index in special_cases:
            cv2.imwrite(str(diagnostic_dir / f"case_frame_{index:06d}_stability.png"), frame)
        index += 1
    boxes_cap.release()

    labels = [f"frame {index}" for index in REPRESENTATIVE_FRAMES]
    cv2.imwrite(str(output_dir / "representative_stabilized_tracking_frames.png"), phase_b.montage(representative_boxes, labels))
    cv2.imwrite(str(output_dir / "representative_segmentation_frames.png"), phase_b.montage(representative_masks, labels))
    cv2.imwrite(str(output_dir / "representative_team_polygon_frames_v2.png"), phase_b.montage(representative_polygons, labels))
    cv2.imwrite(str(output_dir / "tracking_stability_comparison.png"), phase_b.montage(comparison_images, labels, columns=1, tile_size=(1280, 360)))
    contact_sheet = team_contact_sheet(frames, display_by_frame)
    cv2.imwrite(str(output_dir / "team_assignment_contact_sheet_v2.png"), contact_sheet)
    cv2.imwrite(
        str(output_dir / "devils_rink_heatmap_v2.png"),
        phase_b.rink_heatmap(rink, rink_heat_points["devils"], "NEW JERSEY DEVILS - V2 CANONICAL RINK HEATMAP", cv2.COLORMAP_HOT, playable_contour),
    )
    cv2.imwrite(
        str(output_dir / "flyers_rink_heatmap_v2.png"),
        phase_b.rink_heatmap(rink, rink_heat_points["flyers"], "PHILADELPHIA FLYERS - V2 CANONICAL RINK HEATMAP", cv2.COLORMAP_TURBO, playable_contour),
    )

    assignment_rows = [assignments[key] for key in sorted(assignments)]
    write_json(output_dir / "team_assignments_v2.json", {
        "method": "video_prototype_torso_color_temporal_lock_v2",
        "video_specific_prototypes": prototypes,
        "prototype_diagnostics": prototype_diagnostics,
        "assignments": assignment_rows,
    })
    serialized_tracks = []
    for frame_index in range(len(frames)):
        for row in display_by_frame.get(frame_index, []):
            clean = {key: value for key, value in row.items() if key not in ("torso_v2", "team_probabilities")}
            serialized_tracks.append(clean)
    write_json(output_dir / "stabilized_tracking_results.json", {
        "source_frame_count": len(frames),
        "source_frame_indices_preserved": True,
        "detections_run_on_every_frame": True,
        "tracks": serialized_tracks,
    })
    with (output_dir / "stabilized_tracking_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source_frame_index", "track_id", "team", "team_confidence", "detected", "predicted", "x0", "y0", "x1", "y1", "foot_x", "foot_y"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in serialized_tracks:
            writer.writerow({
                "source_frame_index": row["frame_index"], "track_id": row["track_id"], "team": row["team"],
                "team_confidence": row["team_confidence"], "detected": row["detected"], "predicted": row["predicted"],
                "x0": row["bbox"][0], "y0": row["bbox"][1], "x1": row["bbox"][2], "y1": row["bbox"][3],
                "foot_x": row["footpoint"][0], "foot_y": row["footpoint"][1],
            })
    write_json(output_dir / "projected_rink_observations_v2.json", {"observations": projected_observations})
    with (output_dir / "projected_rink_observations_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source_frame_index", "track_id", "team", "team_confidence", "broadcast_foot_x", "broadcast_foot_y", "canonical_rink_x", "canonical_rink_y", "registration_confidence", "reference_frame_index"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in projected_observations:
            writer.writerow({
                "source_frame_index": row["source_frame_index"], "track_id": row["track_id"], "team": row["team"],
                "team_confidence": row["team_confidence"], "broadcast_foot_x": row["broadcast_footpoint"][0],
                "broadcast_foot_y": row["broadcast_footpoint"][1], "canonical_rink_x": row["canonical_rink_xy"][0],
                "canonical_rink_y": row["canonical_rink_xy"][1], "registration_confidence": row["registration_confidence"],
                "reference_frame_index": row["reference_frame_index"],
            })

    accepted_assignments = [row for row in assignment_rows if row["renderable_on_ice_track"]]
    team_tracks = Counter(row["team"] for row in accepted_assignments)
    team_fresh_observations = Counter()
    for track_id, track in stitched_tracks.items():
        if assignments[track_id]["renderable_on_ice_track"]:
            team_fresh_observations[assignments[track_id]["team"]] += len(track["detections"])
    average_duration = statistics.mean(row["duration_frames"] for row in accepted_assignments) if accepted_assignments else 0.0
    legacy_accepted = sum(bool(row.get("renderable_on_ice_track")) for row in legacy_assignments.values())
    video_checks = [phase_b.verify_video(path, 277) for path in paths.values()]
    summary = {
        "stage": "hockey_polygon_demo_v2",
        "status": "complete" if all(row["fully_decodable"] for row in video_checks) else "video_verification_failed",
        "video": metadata,
        "inputs": {
            "video": str(video_path),
            "canonical_rink": str(rink_path),
            "registration": str(v1_dir / "per_frame_rink_homographies.json"),
            "manual_annotations": str(v1_dir / "rink_registration_annotations" / "rink_keypoints.json"),
            "legacy_tracking": str(v1_dir / "tracking_results.json"),
            "legacy_team_assignments": str(v1_dir / "team_assignments.json"),
        },
        "tracking": {
            "detector": "existing OpenCV HOG plus hockey-color component detector on every source frame",
            "tracker": "constant-velocity state prediction, torso appearance matching, 12-frame memory, smoothing, internal-gap interpolation, post-pass stitching",
            "legacy_raw_track_fragments_before": len(legacy_assignments),
            "enhanced_raw_track_fragments_before_stitching": len(raw_tracks),
            "track_fragments_after_stitching": len(stitched_tracks),
            "legacy_accepted_tracks_before": legacy_accepted,
            "accepted_tracks_after": len(accepted_assignments),
            "average_accepted_track_duration_frames": float(average_duration),
            "short_detection_gaps_bridged": smoothing_stats["short_detection_gap_events_bridged"],
            "predicted_boxes_rendered": smoothing_stats["predicted_boxes_rendered"],
            "stitched_track_fragments": len(stitch_events),
            "remaining_id_switch_candidates_heuristic": remaining_switch_candidates,
            "remaining_id_switch_note": "No identity ground truth is available; this is the count of unstitched high-affinity non-overlapping fragment pairs, not a measured MOT ID-switch count.",
            "legacy_detection_matches": legacy_matches,
            "outside_ice_candidates_excluded": outside_excluded,
            **smoothing_stats,
        },
        "classification": {
            "team_definitions": {
                "devils": "red torso primary; black pants are not primary evidence",
                "flyers": "white torso primary; orange/black accents secondary",
                "official": "black-white torso striping plus preserved reliable legacy evidence",
            },
            "video_specific_prototypes": prototypes,
            "prototype_diagnostics": prototype_diagnostics,
            "devils_tracks": team_tracks["devils"],
            "flyers_tracks": team_tracks["flyers"],
            "official_tracks": team_tracks["official"],
            "unknown_tracks": team_tracks["unknown"],
            "devils_fresh_observations": team_fresh_observations["devils"],
            "flyers_fresh_observations": team_fresh_observations["flyers"],
            "official_fresh_observations": team_fresh_observations["official"],
            "unknown_fresh_observations": team_fresh_observations["unknown"],
            **classification_stats,
        },
        "segmentation": segmentation,
        "registration": {
            "source_reused_unchanged": True,
            "ok_frames": render_stats["registration_ok_frames"],
            "rejected_frames": render_stats["registration_rejected_frames"],
            "blue_visible_rink_polygon_preserved": True,
            "player_projection": "fresh smoothed box footpoints only; stored broadcast-to-canonical H used directly",
            "predicted_boxes_projected": False,
            "stale_coordinates_reused": False,
        },
        "rendering": {**dict(render_stats), "special_visual_case_frames": special_cases},
        "video_verification": video_checks,
        "outputs": {key: str(value) for key, value in paths.items()} | {
            "team_assignment_contact_sheet": str(output_dir / "team_assignment_contact_sheet_v2.png"),
            "tracking_stability_comparison": str(output_dir / "tracking_stability_comparison.png"),
            "representative_stabilized_tracking_frames": str(output_dir / "representative_stabilized_tracking_frames.png"),
            "representative_segmentation_frames": str(output_dir / "representative_segmentation_frames.png"),
            "representative_team_polygon_frames": str(output_dir / "representative_team_polygon_frames_v2.png"),
            "devils_rink_heatmap": str(output_dir / "devils_rink_heatmap_v2.png"),
            "flyers_rink_heatmap": str(output_dir / "flyers_rink_heatmap_v2.png"),
            "stabilized_tracking_json": str(output_dir / "stabilized_tracking_results.json"),
            "stabilized_tracking_csv": str(output_dir / "stabilized_tracking_results.csv"),
            "team_assignments": str(output_dir / "team_assignments_v2.json"),
            "projected_observations_json": str(output_dir / "projected_rink_observations_v2.json"),
            "projected_observations_csv": str(output_dir / "projected_rink_observations_v2.csv"),
            "segmentation_status": str(output_dir / "segmentation_status.json"),
            "summary": str(output_dir / "hockey_demo_v2_summary.json"),
            "full_resolution_diagnostics": str(diagnostic_dir),
        },
    }
    write_json(output_dir / "tracking_stitch_events.json", {"events": stitch_events, "association_event_count": len(association_events)})
    write_json(output_dir / "hockey_demo_v2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
