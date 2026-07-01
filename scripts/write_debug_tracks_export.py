#!/usr/bin/env python3
"""Consolidate nll_test4 track evidence into a friend-reviewable debug export."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = {
    "tracking_metadata": "outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/tracking_metadata.json",
    "projected_player_points": "outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/projected_player_points.json",
    "clean_crop_metadata": "outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json",
    "crop_visibility_predictions": "outputs/nll_test4/jersey_crop_visibility_audit/crop_visibility_predictions.json",
    "track_visibility_summary": "outputs/nll_test4/jersey_crop_visibility_audit/track_visibility_summary.json",
    "crop_ocr_predictions": "outputs/nll_test4/jersey_number_ocr_baseline/crop_ocr_predictions.json",
    "track_jersey_number_predictions": "outputs/nll_test4/jersey_number_ocr_baseline/track_jersey_number_predictions.json",
    "team_roster_lookup": "outputs/roster_metadata_validation/team_roster_lookup.json",
    "ambiguous_numbers": "outputs/roster_metadata_validation/ambiguous_numbers.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write consolidated per-track debug evidence for nll_test4.")
    parser.add_argument("--run-id", default="nll_test4")
    for name, default in DEFAULT_SOURCES.items():
        parser.add_argument("--" + name.replace("_", "-"), default=default)
    parser.add_argument("--output", default="outputs/nll_test4/debug_tracks.json")
    parser.add_argument("--summary-output", default="outputs/nll_test4/debug_tracks_summary.json")
    parser.add_argument(
        "--curated-output",
        default="progress_outputs/nll_test4_checkpoint/json/debug_tracks.json",
    )
    parser.add_argument(
        "--curated-summary-output",
        default="progress_outputs/nll_test4_checkpoint/json/debug_tracks_summary.json",
    )
    parser.add_argument("--best-crops-per-type", type=int, default=3)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object in {}".format(path))
    return payload


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def finite_values(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)):
            result.append(float(value))
    return result


def average_or_none(values: list[Any]) -> float | None:
    numeric = finite_values(values)
    return float(mean(numeric)) if numeric else None


def maximum_or_none(values: list[Any]) -> float | None:
    numeric = finite_values(values)
    return max(numeric) if numeric else None


def bbox_as_list(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    required = ("x0", "y0", "x1", "y1")
    if not all(key in value for key in required):
        return None
    return [float(value[key]) for key in required]


def records(payload: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows = payload.get(key, [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def index_by_track(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result = defaultdict(list)
    for row in rows:
        track_id = as_int(row.get("track_id"))
        if track_id is not None:
            result[track_id].append(row)
    return result


def best_paths(rows: list[dict[str, Any]], score_key: str, limit: int) -> list[str]:
    ranked = sorted(
        [row for row in rows if row.get("crop_path")],
        key=lambda row: float(row.get(score_key) or 0.0),
        reverse=True,
    )
    paths = []
    for row in ranked:
        path = portable_path(row.get("crop_path"))
        if path and path not in paths:
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def best_number_region_paths(rows: list[dict[str, Any]], limit: int) -> list[str]:
    ranked = sorted(rows, key=lambda row: float(row.get("ocr_readiness_score") or 0.0), reverse=True)
    paths = []
    for row in ranked:
        path = portable_path(row.get("number_region_path"))
        if path and path not in paths:
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def normalize_candidates(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    raw = row.get("all_candidate_numbers", [])
    result = []
    if not isinstance(raw, list):
        return result
    for value in raw:
        if isinstance(value, dict):
            number = value.get("number", value.get("jersey_number", value.get("candidate_number")))
            confidence = value.get("confidence", value.get("score", value.get("weighted_vote")))
        else:
            number = value
            confidence = None
        if number is None:
            continue
        result.append({"number": str(number), "confidence": confidence})
    return result


def observation_jersey_probs(rows: list[dict[str, Any]]) -> dict[str, float]:
    probs = {}
    for row in rows:
        number = row.get("candidate_number")
        if number is None:
            continue
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = row.get("variant_consensus")
        score = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        key = str(number)
        probs[key] = max(probs.get(key, 0.0), score)
    return probs


def main() -> int:
    args = parse_args()
    if args.best_crops_per_type < 1:
        raise ValueError("--best-crops-per-type must be at least 1")

    source_paths = {
        name: resolve_path(getattr(args, name))
        for name in DEFAULT_SOURCES
    }
    source_payloads = {name: load_optional(path) for name, path in source_paths.items()}
    found_sources = [portable_path(path) for name, path in source_paths.items() if source_payloads[name] is not None]
    missing_sources = [portable_path(path) for name, path in source_paths.items() if source_payloads[name] is None]

    tracking = source_payloads["tracking_metadata"]
    if tracking is None:
        raise FileNotFoundError("Tracking metadata is required: {}".format(source_paths["tracking_metadata"]))
    projections = source_payloads["projected_player_points"]
    clean = source_payloads["clean_crop_metadata"]
    visibility_crops = source_payloads["crop_visibility_predictions"]
    visibility_tracks = source_payloads["track_visibility_summary"]
    ocr_crops = source_payloads["crop_ocr_predictions"]
    ocr_tracks = source_payloads["track_jersey_number_predictions"]

    tracking_rows = records(tracking, "detections")
    projection_rows = records(projections, "points")
    clean_rows = records(clean, "crops")
    visibility_crop_rows = records(visibility_crops, "crops")
    visibility_track_rows = records(visibility_tracks, "tracks")
    ocr_crop_rows = records(ocr_crops, "predictions")
    ocr_track_rows = records(ocr_tracks, "tracks")

    detections_by_track = index_by_track(tracking_rows)
    clean_by_track = index_by_track(clean_rows)
    visibility_crops_by_track = index_by_track(visibility_crop_rows)
    visibility_tracks_by_id = {
        as_int(row.get("track_id")): row for row in visibility_track_rows if as_int(row.get("track_id")) is not None
    }
    ocr_crops_by_track = index_by_track(ocr_crop_rows)
    ocr_tracks_by_id = {
        as_int(row.get("track_id")): row for row in ocr_track_rows if as_int(row.get("track_id")) is not None
    }

    decode_by_frame = {
        int(row["clip_frame_index"]): row
        for row in tracking.get("decode", {}).get("decoded_frames", [])
        if isinstance(row, dict) and row.get("clip_frame_index") is not None
    }
    projection_index = {}
    projection_fallback = {}
    for row in projection_rows:
        track_id = as_int(row.get("track_id"))
        frame_index = as_int(row.get("frame_index"))
        mask_id = as_int(row.get("mask_id"))
        if track_id is None or frame_index is None:
            continue
        projection_fallback[(track_id, frame_index)] = row
        projection_index[(track_id, frame_index, mask_id)] = row

    clean_by_track_frame = defaultdict(list)
    for row in clean_rows:
        key = (as_int(row.get("track_id")), as_int(row.get("frame_index")))
        if None not in key:
            clean_by_track_frame[key].append(row)
    visibility_by_track_frame = defaultdict(list)
    for row in visibility_crop_rows:
        key = (as_int(row.get("track_id")), as_int(row.get("frame_index")))
        if None not in key:
            visibility_by_track_frame[key].append(row)
    ocr_by_track_frame = defaultdict(list)
    for row in ocr_crop_rows:
        key = (as_int(row.get("track_id")), as_int(row.get("frame_index")))
        if None not in key:
            ocr_by_track_frame[key].append(row)

    ocr_engine_available = bool((ocr_crops or {}).get("ocr_engine", {}).get("available", False))
    debug_tracks = []
    for track_id in sorted(detections_by_track):
        detections = sorted(detections_by_track[track_id], key=lambda row: int(row.get("frame_index", -1)))
        track_clean = clean_by_track.get(track_id, [])
        track_visibility = visibility_crops_by_track.get(track_id, [])
        visibility_track = visibility_tracks_by_id.get(track_id, {})
        track_ocr_crops = ocr_crops_by_track.get(track_id, [])
        ocr_track = ocr_tracks_by_id.get(track_id, {})
        timestamps = []
        observations = []
        for detection in detections:
            frame_index = int(detection["frame_index"])
            mask_id = as_int(detection.get("mask_id"))
            decode_row = decode_by_frame.get(frame_index, {})
            timestamp = decode_row.get("timestamp_seconds")
            if isinstance(timestamp, (int, float)):
                timestamps.append(float(timestamp))
            projection = projection_index.get((track_id, frame_index, mask_id))
            if projection is None:
                projection = projection_fallback.get((track_id, frame_index))
            frame_clean = clean_by_track_frame.get((track_id, frame_index), [])
            frame_visibility = visibility_by_track_frame.get((track_id, frame_index), [])
            frame_ocr = ocr_by_track_frame.get((track_id, frame_index), [])
            crop_quality = maximum_or_none([row.get("crop_quality_score") for row in frame_clean])
            readiness = maximum_or_none(
                [row.get("ocr_readiness_score") for row in frame_ocr]
                + [row.get("quality_score") for row in frame_visibility]
            )
            occlusion_score = maximum_or_none(
                [row.get("quality", {}).get("occlusion_score") for row in frame_clean]
            )
            point = projection.get("projected_field_point") if projection else None
            observations.append(
                {
                    "frame_index": frame_index,
                    "source_frame_index": decode_row.get("source_frame_index"),
                    "timestamp": timestamp,
                    "mask_id": mask_id,
                    "bbox": bbox_as_list(detection.get("bbox_2d")),
                    "detection_confidence": detection.get("sam_confidence_score"),
                    "association_confidence": detection.get("match_score"),
                    "projected_field_point": {
                        "x": point.get("x") if isinstance(point, dict) else None,
                        "y": point.get("y") if isinstance(point, dict) else None,
                    },
                    "inside_field_template_bounds": (
                        projection.get("inside_field_template_bounds") if projection else None
                    ),
                    "crop_quality": crop_quality,
                    "ocr_readiness_score": readiness,
                    "jersey_probs": observation_jersey_probs(frame_ocr),
                    "occlusion_score": occlusion_score,
                }
            )

        start_time = min(timestamps) if timestamps else None
        end_time = max(timestamps) if timestamps else None
        duration = end_time - start_time if start_time is not None and end_time is not None else None
        accepted_number = ocr_track.get("accepted_number")
        candidates = normalize_candidates(ocr_track)
        jersey_confidence = float(ocr_track.get("consensus_score") or 0.0)
        if accepted_number is not None:
            jersey_status = "accepted"
        elif not ocr_engine_available:
            jersey_status = "ocr_unavailable"
        elif candidates:
            jersey_status = "low_confidence"
        else:
            jersey_status = "no_candidate"

        team_abbreviation = None
        team_confidence = 0.0
        if accepted_number is not None and team_abbreviation:
            roster_status = "candidate_lookup_available"
        else:
            roster_status = "not_attempted"

        full_body = [row for row in track_clean if row.get("crop_type") == "full_body"]
        torso = [row for row in track_clean if row.get("crop_type") == "torso"]
        num_projected = sum(
            1 for row in observations if row["projected_field_point"]["x"] is not None
        )
        num_ocr_ready = sum(1 for row in track_visibility if bool(row.get("ocr_candidate")))
        number_regions = [row for row in track_ocr_crops if row.get("number_region_path")]
        debug_tracks.append(
            {
                "track_id": str(track_id),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "num_observations": len(observations),
                "resolved_player_id": None,
                "resolved_confidence": 0.0,
                "is_player": True,
                "player_likelihood": None,
                "team_assignment": {
                    "team_abbreviation": team_abbreviation,
                    "team_confidence": team_confidence,
                    "team_color_rgb": None,
                },
                "jersey_number": {
                    "accepted_number": str(accepted_number) if accepted_number is not None else None,
                    "candidate_numbers": candidates,
                    "confidence": jersey_confidence,
                    "status": jersey_status,
                },
                "roster_resolution": {
                    "status": roster_status,
                    "candidates": [],
                    "candidate_names_included": False,
                },
                "evidence": {
                    "tracking_confidence": average_or_none([row.get("match_score") for row in detections]),
                    "detection_confidence": average_or_none(
                        [row.get("sam_confidence_score") for row in detections]
                    ),
                    "crop_quality": maximum_or_none(
                        [row.get("crop_quality_score") for row in track_clean]
                    ),
                    "ocr_readiness_score": maximum_or_none(
                        [row.get("ocr_readiness_score") for row in track_ocr_crops]
                        + [row.get("quality_score") for row in track_visibility]
                    ),
                    "jersey_confidence": jersey_confidence,
                    "visibility_readiness": visibility_track.get("readiness"),
                    "num_clean_crops": len(track_clean),
                    "num_ocr_ready_crops": num_ocr_ready,
                    "num_number_regions": len(number_regions),
                    "num_projected_points": num_projected,
                },
                "best_crops": {
                    "clean_full_body": best_paths(
                        full_body, "crop_quality_score", args.best_crops_per_type
                    ),
                    "torso": best_paths(torso, "crop_quality_score", args.best_crops_per_type),
                    "number_regions": best_number_region_paths(
                        number_regions, args.best_crops_per_type
                    ),
                },
                "observations": observations,
            }
        )

    total_observations = sum(row["num_observations"] for row in debug_tracks)
    known_limitations = [
        "Tracks are short-clip heuristic tracklets and may contain identity switches.",
        "The active-player filter may retain a referee; is_player reflects upstream inclusion, not a trained classifier.",
        "Projected field points use a manual static homography that can drift during camera pan or zoom.",
        "Torso selection and OCR readiness are heuristic and do not prove that a jersey number is readable.",
        "No OCR engine was available; no jersey numbers were recognized or accepted.",
        "No team assignment or roster resolution was attempted.",
        "No player names or resolved player identities are included.",
    ]
    summary = {
        "status": "complete",
        "stage": "debug_tracks_export_summary",
        "run_id": args.run_id,
        "counts": {
            "tracks": len(debug_tracks),
            "total_observations": total_observations,
            "tracks_with_projected_field_points": sum(
                1 for row in debug_tracks if row["evidence"]["num_projected_points"] > 0
            ),
            "tracks_with_clean_crops": sum(
                1 for row in debug_tracks if row["evidence"]["num_clean_crops"] > 0
            ),
            "tracks_with_ocr_ready_crops": sum(
                1 for row in debug_tracks if row["evidence"]["num_ocr_ready_crops"] > 0
            ),
            "tracks_with_accepted_jersey_number": sum(
                1 for row in debug_tracks if row["jersey_number"]["accepted_number"] is not None
            ),
            "tracks_with_resolved_player_identity": sum(
                1 for row in debug_tracks if row["resolved_player_id"] is not None
            ),
        },
        "ocr_engine_available": ocr_engine_available,
        "source_files_found": found_sources,
        "source_files_missing": missing_sources,
        "known_limitations": known_limitations,
        "artifacts": {
            "debug_tracks": portable_path(args.output),
            "debug_tracks_summary": portable_path(args.summary_output),
            "curated_debug_tracks": portable_path(args.curated_output),
            "curated_debug_tracks_summary": portable_path(args.curated_summary_output),
        },
    }

    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    curated_path = resolve_path(args.curated_output)
    curated_summary_path = resolve_path(args.curated_summary_output)
    debug_serialized = json.dumps(debug_tracks, indent=2, sort_keys=True) + "\n"
    summary_serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    for path in (output_path, summary_path, curated_path, curated_summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(debug_serialized, encoding="utf-8")
    curated_path.write_text(debug_serialized, encoding="utf-8")
    summary_path.write_text(summary_serialized, encoding="utf-8")
    curated_summary_path.write_text(summary_serialized, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
