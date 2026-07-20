#!/usr/bin/env python3
"""Build a persistent TOR/OSH color profile from color-only V2 evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean_vector(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), list)]
    if not values:
        return None
    return [float(statistics.mean(items)) for items in zip(*values)]


def mean_scalar(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.mean(values)) if values else None


def parse_args():
    parser = argparse.ArgumentParser(description="Build nll_test4 team color profile from V2 color-only outputs.")
    parser.add_argument("--config", default="configs/nll_test4_team_profile.json")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(project_path(args.config))
    inputs = config["inputs"]
    output_dir = project_path(args.output_dir or config["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = load_json(project_path(inputs["color_only_assignments"]))
    signatures = load_json(project_path(inputs["color_only_signatures"]))
    summary = load_json(project_path(inputs["color_only_summary"]))
    clusters = load_json(project_path(inputs["cluster_centers"]))

    if summary.get("embedding_backend_used") != "none":
        raise ValueError("Expected color-only V2 summary with embedding_backend_used=none")
    if not summary.get("color_only_fallback_used"):
        raise ValueError("Expected color_only_fallback_used=true")
    if summary.get("scorebug_priors_used"):
        raise ValueError("Scorebug priors must be disabled for this profile")
    if any(row.get("evidence_mode") != "color_only" for row in assignments):
        raise ValueError("All assignments must have evidence_mode=color_only")

    by_track_signature = {int(row["track_id"]): row for row in signatures}
    by_class = defaultdict(list)
    assignment_counts = Counter()
    raw_cluster_map = defaultdict(list)
    for row in assignments:
        cls = row.get("assigned_class")
        assignment_counts[cls] += 1
        if cls in ("team_a", "team_b"):
            sig = by_track_signature.get(int(row["track_id"]), {})
            merged = dict(sig)
            merged.update(row)
            by_class[cls].append(merged)
            raw_cluster_map[str(row.get("automatic_cluster"))].append(int(row["track_id"]))

    profiles = {}
    for label, real in config["team_label_to_real_team"].items():
        rows = by_class.get(label, [])
        profiles[label] = {
            **real,
            "source_label": label,
            "track_ids": sorted(int(row["track_id"]) for row in rows),
            "track_count": len(rows),
            "mean_representative_median_lab": mean_vector(rows, "representative_median_lab") or mean_vector(rows, "median_lab"),
            "mean_representative_median_hsv": mean_vector(rows, "representative_median_hsv") or mean_vector(rows, "median_hsv"),
            "mean_white_fraction": mean_scalar(rows, "white_fraction"),
            "mean_dark_fraction": mean_scalar(rows, "dark_fraction"),
            "mean_neutral_fraction": mean_scalar(rows, "neutral_fraction"),
            "mean_color_consistency": mean_scalar(rows, "color_consistency"),
            "mean_assignment_confidence": mean_scalar(rows, "confidence"),
            "dominant_lab_colors": [item for row in rows for item in row.get("dominant_lab_colors", [])[:1]][:8],
        }

    profile = {
        "stage": "game_team_color_profile",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run_id", "nll_test4"),
        "evidence_mode": "color_only",
        "embedding_backend": "none",
        "scorebug_priors_used": False,
        "team_profiles": profiles,
        "team_label_to_real_team": config["team_label_to_real_team"],
        "assignment_counts": dict(assignment_counts),
        "raw_cluster_track_ids": dict(raw_cluster_map),
        "raw_cluster_label_to_team": clusters.get("label_to_team", {}),
        "inputs": {key: str(project_path(value)) for key, value in inputs.items()},
        "artifacts": {
            "team_color_profile": str(output_dir / "team_color_profile.json"),
            "profile_summary": str(output_dir / "team_color_profile_summary.json"),
        },
        "known_limitations": [
            "Built from the current color-only V2 20-second segment evidence.",
            "Profile matching validates stable labels but does not replace manual visual review.",
            "Officials and unknown tracks are preserved outside team profiles.",
        ],
    }
    summary_out = {
        "status": "complete",
        "stage": "game_team_color_profile_summary",
        "counts": {
            "team_a_tracks": profiles.get("team_a", {}).get("track_count", 0),
            "team_b_tracks": profiles.get("team_b", {}).get("track_count", 0),
            "official_tracks": assignment_counts.get("official", 0),
            "unknown_tracks": assignment_counts.get("unknown", 0),
        },
        "team_label_to_real_team": config["team_label_to_real_team"],
        "artifacts": profile["artifacts"],
    }
    write_json(output_dir / "team_color_profile.json", profile)
    write_json(output_dir / "team_color_profile_summary.json", summary_out)
    print(json.dumps(summary_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

