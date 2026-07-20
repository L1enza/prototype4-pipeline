#!/usr/bin/env python3
"""Validate color-only team assignments against the persistent game profile."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_team_assignment_and_polygon_demo as v1
import run_team_assignment_and_polygon_demo_v2 as v2
from prototype4_pipeline.integrations.team_appearance_features import load_appearance_backend

REAL_TEAMS = {
    "team_a": {"abbr": "TOR", "name": "Toronto Rock", "color": (245, 245, 245), "outline": (20, 80, 210)},
    "team_b": {"abbr": "OSH", "name": "Oshawa FireWolves", "color": (100, 28, 34), "outline": (225, 145, 55)},
    "official": {"abbr": "OFF", "name": "Official", "color": (235, 205, 35), "outline": (0, 0, 0)},
    "unknown": {"abbr": "UNK", "name": "Unknown", "color": (145, 145, 145), "outline": (80, 80, 80)},
}


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(1.0 - float(np.dot(a, b)) / denom)


def vector_distance(a, b):
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def validation_config(profile_config):
    return {
        "run_id": "nll_test4",
        "inputs": {
            "video": "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/nll_test4.mp4",
            "clean_crop_metadata": "outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json",
            "v1_assignments": "outputs/nll_test4/team_assignment_demo/track_team_assignments.json",
        },
        "torso_crop_fractions": {"x_start": 0.15, "x_end": 0.85, "y_start": 0.12, "y_end": 0.62},
        "observation_selection": {"max_observations_per_track": 30, "min_observations_per_track": 4, "min_frame_spacing": 2},
        "feature_weights": {"appearance": 0.0, "color": 1.0},
        "appearance": {"embedding_backend": "none", "allow_download_weights": False},
        "clustering": {"gmm_random_seed": 17, "unknown_posterior_threshold": 0.65, "assignment_margin_threshold": 0.12, "min_evidence_quality": 0.32},
        "official": {"score_threshold": 0.93, "v1_official_bonus": 0.3},
        "outlier_rejection": {"color_cosine_distance_threshold": 0.35, "appearance_cosine_distance_threshold": 0.35, "max_reject_fraction": 0.30},
        "rendering": {"fps": 6.0},
        "scorebug_priors": {"enabled": False},
    }


def build_segment_signatures(tracking_path, cfg):
    tracking = load_json(tracking_path)
    tracks = defaultdict(list)
    for det in tracking.get("detections", []):
        tracks[int(det["track_id"])].append(det)
    frame_lookup = v1.decoded_frame_map(tracking)
    needed_sources = [row["source_frame_index"] for row in frame_lookup.values()]
    frames_by_source = v1.read_needed_frames(Path(cfg["inputs"]["video"]), needed_sources)
    clean_index = v2.build_clean_crop_index(project_path(cfg["inputs"]["clean_crop_metadata"]))
    backend = load_appearance_backend("none", False)
    signatures = {}
    for track_id, rows in sorted(tracks.items()):
        raw_obs = v2.build_track_observations(track_id, rows, frame_lookup, frames_by_source, clean_index, backend, cfg)
        kept, rejected = v2.reject_outliers(raw_obs, cfg, False)
        sig = v2.aggregate_signature(track_id, kept, rejected, cfg, "color_only")
        if sig is not None:
            signatures[track_id] = sig
    v1_assignments = {}
    assignments, diagnostics = v2.assign_tracks(signatures, cfg, v1_assignments)
    rows = [assignments[track_id] for track_id in sorted(assignments)]
    return tracking, rows, diagnostics, signatures


def profile_match(assignments, profile):
    profiles = profile["team_profiles"]
    profile_labs = {label: data.get("mean_representative_median_lab") for label, data in profiles.items()}
    cluster_to_tracks = defaultdict(list)
    for row in assignments:
        if row.get("automatic_cluster") is not None and row.get("assigned_class") in ("team_a", "team_b"):
            cluster_to_tracks[str(row["automatic_cluster"])].append(row)
    cluster_map = {}
    confidence = {}
    for cluster_id, rows in cluster_to_tracks.items():
        lab = [float(np.mean([row["representative_median_lab"][idx] for row in rows])) for idx in range(3)]
        distances = {label: vector_distance(lab, prof_lab) for label, prof_lab in profile_labs.items() if prof_lab is not None}
        ranked = sorted(distances.items(), key=lambda item: item[1])
        if ranked:
            cluster_map[cluster_id] = ranked[0][0]
            margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 99.0
            confidence[cluster_id] = float(max(0.0, min(1.0, margin / 45.0)))
    remapped = []
    for row in assignments:
        out = dict(row)
        if row.get("assigned_class") in ("team_a", "team_b") and row.get("automatic_cluster") is not None:
            mapped_label = cluster_map.get(str(row["automatic_cluster"]), row["assigned_class"])
            out["profile_matched_class"] = mapped_label
            out["profile_team_abbreviation"] = profile["team_profiles"][mapped_label]["team_abbreviation"]
            out["profile_team_name"] = profile["team_profiles"][mapped_label]["team_name"]
            out["profile_match_confidence"] = confidence.get(str(row["automatic_cluster"]), 0.0)
        else:
            out["profile_matched_class"] = row.get("assigned_class", "unknown")
            out["profile_team_abbreviation"] = None
            out["profile_team_name"] = REAL_TEAMS.get(out["profile_matched_class"], REAL_TEAMS["unknown"])["name"]
            out["profile_match_confidence"] = None
        remapped.append(out)
    return remapped, cluster_map, confidence


def save_contact_sheet(signatures, assignments, output_path):
    tiles = []
    for row in sorted(assignments, key=lambda item: int(item["track_id"])):
        sig = signatures.get(int(row["track_id"]))
        if not sig:
            continue
        crop = Image.fromarray(sig["best_crop_rgb"].astype(np.uint8)).convert("RGB")
        crop.thumbnail((128, 144), Image.Resampling.LANCZOS)
        cls = row.get("profile_matched_class", row.get("assigned_class", "unknown"))
        real = REAL_TEAMS.get(cls, REAL_TEAMS["unknown"])
        tile = Image.new("RGB", (205, 220), (245, 245, 245))
        tile.paste(crop, ((205 - crop.width) // 2, 8))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 204, 219), outline=real["outline"], width=4)
        draw.text((8, 155), f"T{row['track_id']:03d} {real['abbr']}", fill=real["outline"])
        draw.text((8, 174), f"raw {row.get('automatic_cluster')} conf {row.get('confidence',0):.2f}", fill=(0, 0, 0))
        draw.text((8, 193), f"profile {row.get('profile_match_confidence')}", fill=(70, 70, 70))
        tiles.append(tile)
    cols = min(5, max(1, len(tiles)))
    rows = max(1, math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 205, rows * 220), (235, 235, 235))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 205, (idx // cols) * 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def render_overlay(tracking, assignments, output_dir, fps):
    by_id = {int(row["track_id"]): row for row in assignments}
    frame_lookup = v1.decoded_frame_map(tracking)
    det_by_frame = defaultdict(list)
    for det in tracking.get("detections", []):
        det_by_frame[int(det["frame_index"])].append(det)
    needed_sources = [row["source_frame_index"] for row in frame_lookup.values()]
    frames_by_source = v1.read_needed_frames(Path("/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/nll_test4.mp4"), needed_sources)
    frames_dir = output_dir / "overlay_frames"
    rendered = []
    for frame_index in sorted(det_by_frame):
        source = frame_lookup.get(frame_index, {}).get("source_frame_index")
        if source not in frames_by_source:
            continue
        img = Image.fromarray(frames_by_source[source]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for det in det_by_frame[frame_index]:
            row = by_id.get(int(det["track_id"]), {})
            cls = row.get("profile_matched_class", row.get("assigned_class", "unknown"))
            real = REAL_TEAMS.get(cls, REAL_TEAMS["unknown"])
            box = det.get("bbox_2d", {})
            color = real["outline"]
            draw.rectangle((box.get("x0", 0), box.get("y0", 0), box.get("x1", 0), box.get("y1", 0)), outline=color, width=4)
            draw.text((box.get("x0", 0), max(0, box.get("y0", 0) - 16)), f"{real['abbr']} T{det['track_id']}", fill=color)
        path = frames_dir / f"frame_{frame_index:03d}_team_validation.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        rendered.append(path)
    mp4, mp4_error = v1.write_mp4(rendered, output_dir / "team_validation_overlay.mp4", fps)
    gif = v1.write_gif(rendered, output_dir / "team_validation_overlay.gif", fps)
    return {"frames": len(rendered), "mp4": mp4, "mp4_error": mp4_error, "gif": gif}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate color-only team profile on existing nll_test4 segment tracking outputs.")
    parser.add_argument("--config", default="configs/nll_test4_team_profile.json")
    parser.add_argument("--profile", default="outputs/nll_test4/game_team_color_profile/team_color_profile.json")
    parser.add_argument("--output-dir", default="outputs/nll_test4/multi_segment_team_assignment_validation")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(project_path(args.config))
    profile = load_json(project_path(args.profile))
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = validation_config(config)

    segment_summaries = []
    for segment in config["validation"]["preferred_segments"]:
        tracking_path = project_path(segment["tracking_metadata"])
        seg_dir = output_dir / segment["name"]
        if not tracking_path.exists():
            segment_summaries.append({"segment": segment, "status": "missing_tracking_metadata"})
            continue
        tracking, assignments, diagnostics, signatures = build_segment_signatures(tracking_path, cfg)
        remapped, cluster_map, match_conf = profile_match(assignments, profile)
        counts = Counter(row.get("profile_matched_class", row.get("assigned_class")) for row in remapped)
        near_boundary = [
            int(row["track_id"])
            for row in remapped
            if row.get("assigned_class") in ("team_a", "team_b")
            and (float(row.get("assignment_margin") or 0.0) < 0.18 or float(row.get("profile_match_confidence") or 0.0) < 0.35)
        ]
        contact = save_contact_sheet(signatures, remapped, seg_dir / "team_validation_contact_sheet.png")
        overlay = render_overlay(tracking, remapped, seg_dir, 6.0)
        write_json(seg_dir / "team_validation_assignments.json", remapped)
        seg_summary = {
            "segment": segment,
            "status": "complete",
            "evidence_mode": "color_only",
            "embedding_backend": "none",
            "scorebug_priors_used": False,
            "counts": {
                "TOR": counts.get("team_a", 0),
                "OSH": counts.get("team_b", 0),
                "official": counts.get("official", 0),
                "unknown": counts.get("unknown", 0),
            },
            "raw_cluster_to_profile_label": cluster_map,
            "raw_cluster_match_confidence": match_conf,
            "raw_cluster_label_to_team_before_profile": diagnostics.get("label_to_team"),
            "cluster_ids_flipped_relative_to_training_profile": cluster_map != profile.get("raw_cluster_label_to_team", {}),
            "profile_matching_corrected_flip": bool(cluster_map),
            "near_decision_boundary_tracks": near_boundary,
            "visually_wrong_tracks_reported": [],
            "artifacts": {"contact_sheet": contact, **overlay},
        }
        write_json(seg_dir / "team_validation_summary.json", seg_summary)
        segment_summaries.append(seg_summary)

    overall = {
        "stage": "multi_segment_team_assignment_validation",
        "status": "complete",
        "evidence_mode": "color_only",
        "embedding_backend": "none",
        "scorebug_priors_used": False,
        "profile": str(project_path(args.profile)),
        "segments": segment_summaries,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "multi_segment_team_assignment_validation_summary.json", overall)
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

