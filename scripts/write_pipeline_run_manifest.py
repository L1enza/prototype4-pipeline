#!/usr/bin/env python3
"""Write a conservative master manifest for a Prototype 4 pipeline checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the nll_test4 pipeline checkpoint manifest.")
    parser.add_argument("--run-id", default="nll_test4")
    parser.add_argument(
        "--clean-crop-summary",
        default="outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_summary.json",
    )
    parser.add_argument(
        "--visibility-summary",
        default="outputs/nll_test4/jersey_crop_visibility_audit/jersey_crop_visibility_summary.json",
    )
    parser.add_argument(
        "--ocr-summary",
        default="outputs/nll_test4/jersey_number_ocr_baseline/jersey_ocr_summary.json",
    )
    parser.add_argument(
        "--roster-summary",
        default="outputs/roster_metadata_validation/roster_validation_summary.json",
    )
    parser.add_argument(
        "--demo-summary",
        default=(
            "outputs/nll_test4/calibrated_segment_demos/"
            "segment_20s_10s_calibrated/demo_summary.json"
        ),
    )
    parser.add_argument("--output", default="outputs/nll_test4/pipeline_run_manifest.json")
    parser.add_argument(
        "--curated-output",
        default="progress_outputs/nll_test4_checkpoint/json/pipeline_run_manifest.json",
    )
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


def read_optional_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, "{}: {}".format(type(exc).__name__, exc)
    if not isinstance(payload, dict):
        return None, "top_level_json_is_not_an_object"
    return payload, None


def artifact_path(summary: dict[str, Any] | None, key: str) -> str | None:
    if not summary:
        return None
    value = summary.get("artifacts", {}).get(key)
    return portable_path(value) if value else None


def present_paths(values: list[str | None]) -> list[str]:
    result = []
    for value in values:
        if not value or value in result:
            continue
        path = resolve_path(value) if not Path(value).is_absolute() else Path(value)
        if path.exists():
            result.append(portable_path(path))
    return result


def stage_record(
    name: str,
    summary_path: Path,
    summary: dict[str, Any] | None,
    error: str | None,
    completion_note: str,
) -> dict[str, Any]:
    status = summary.get("status") if summary else "missing"
    if summary is None:
        state = "missing"
    elif status == "complete":
        state = "complete"
    elif status in {"valid_with_warnings", "complete_with_warnings"}:
        state = "complete_with_warnings"
    elif status in {"ocr_unavailable", "blocked"}:
        state = "blocked"
    else:
        state = "available_with_nonstandard_status"
    return {
        "name": name,
        "summary_found": summary is not None,
        "summary_file": portable_path(summary_path),
        "summary_read_error": error,
        "reported_stage": summary.get("stage") if summary else None,
        "reported_status": status,
        "state": state,
        "counts": summary.get("counts", {}) if summary else {},
        "completion_note": completion_note,
    }


def unique_strings(values: list[Any]) -> list[str]:
    result = []
    for value in values:
        if not isinstance(value, str) or not value or value in result:
            continue
        result.append(value)
    return result


def main() -> int:
    args = parse_args()
    input_specs = {
        "clean_crop_extraction": resolve_path(args.clean_crop_summary),
        "jersey_crop_visibility_audit": resolve_path(args.visibility_summary),
        "jersey_number_ocr_baseline": resolve_path(args.ocr_summary),
        "roster_metadata_validation": resolve_path(args.roster_summary),
        "calibrated_segment_demo": resolve_path(args.demo_summary),
    }
    loaded = {}
    errors = {}
    for name, path in input_specs.items():
        loaded[name], errors[name] = read_optional_json(path)

    clean = loaded["clean_crop_extraction"]
    visibility = loaded["jersey_crop_visibility_audit"]
    ocr = loaded["jersey_number_ocr_baseline"]
    roster = loaded["roster_metadata_validation"]
    demo = loaded["calibrated_segment_demo"]

    ocr_engine = ocr.get("ocr_engine", {}) if ocr else {}
    ocr_engine_available = bool(ocr_engine.get("available", False))
    accepted_jersey_numbers = int((ocr or {}).get("counts", {}).get("accepted_track_numbers", 0) or 0)
    identities_assigned = any(
        bool(value)
        for value in (
            (visibility or {}).get("identity_assignment_performed", False),
            (ocr or {}).get("identity_assignment_performed", False),
            (roster or {}).get("player_identity_assignment_performed", False),
        )
    )

    found_inputs = [portable_path(path) for name, path in input_specs.items() if loaded[name] is not None]
    missing_inputs = [portable_path(path) for name, path in input_specs.items() if loaded[name] is None]
    all_summaries_found = not missing_inputs

    stage_summaries = {
        "calibrated_segment_demo": stage_record(
            "calibrated_segment_demo",
            input_specs["calibrated_segment_demo"],
            demo,
            errors["calibrated_segment_demo"],
            "Short calibrated tracking, projection, heatmap, and side-by-side demo completed.",
        ),
        "clean_crop_extraction": stage_record(
            "clean_crop_extraction",
            input_specs["clean_crop_extraction"],
            clean,
            errors["clean_crop_extraction"],
            "Clean full-body and torso crops were extracted directly from original video frames.",
        ),
        "jersey_crop_visibility_audit": stage_record(
            "jersey_crop_visibility_audit",
            input_specs["jersey_crop_visibility_audit"],
            visibility,
            errors["jersey_crop_visibility_audit"],
            "Heuristic OCR-readiness audit completed; it did not perform OCR or identity assignment.",
        ),
        "jersey_number_ocr_baseline": stage_record(
            "jersey_number_ocr_baseline",
            input_specs["jersey_number_ocr_baseline"],
            ocr,
            errors["jersey_number_ocr_baseline"],
            (
                "OCR engine is available and the baseline stage ran."
                if ocr_engine_available
                else "Baseline inspection completed, but OCR recognition is blocked because no engine is available."
            ),
        ),
        "roster_metadata_validation": stage_record(
            "roster_metadata_validation",
            input_specs["roster_metadata_validation"],
            roster,
            errors["roster_metadata_validation"],
            "Roster files were validated for future constrained lookup; no player identity was assigned.",
        ),
    }

    output_directories = present_paths(
        [
            (clean or {}).get("output_dir"),
            str(input_specs["jersey_crop_visibility_audit"].parent),
            str(input_specs["jersey_number_ocr_baseline"].parent),
            str(input_specs["roster_metadata_validation"].parent),
            (demo or {}).get("output_dir"),
        ]
    )
    debug_tracks_path = PROJECT_ROOT / "outputs" / args.run_id / "debug_tracks.json"
    debug_tracks_summary_path = PROJECT_ROOT / "outputs" / args.run_id / "debug_tracks_summary.json"
    curated_debug_tracks_path = (
        PROJECT_ROOT / "progress_outputs" / "nll_test4_checkpoint" / "json" / "debug_tracks.json"
    )
    curated_debug_tracks_summary_path = (
        PROJECT_ROOT
        / "progress_outputs"
        / "nll_test4_checkpoint"
        / "json"
        / "debug_tracks_summary.json"
    )
    important_json_files = present_paths(
        [
            *[str(path) for path in input_specs.values()],
            artifact_path(clean, "metadata"),
            artifact_path(visibility, "crop_visibility_predictions"),
            artifact_path(visibility, "track_visibility_summary"),
            artifact_path(ocr, "crop_ocr_predictions"),
            artifact_path(ocr, "track_jersey_number_predictions"),
            artifact_path(roster, "ambiguous_numbers"),
            artifact_path(demo, "tracking_summary"),
            artifact_path(demo, "calibration_projection_summary"),
            artifact_path(demo, "heatmap_summary"),
            artifact_path(demo, "side_by_side_summary"),
            str(debug_tracks_path),
            str(debug_tracks_summary_path),
        ]
    )
    contact_sheet_paths = present_paths(
        [
            artifact_path(clean, "all_tracks_best_clean_crops"),
            artifact_path(clean, "all_tracks_best_torso_crops"),
            artifact_path(visibility, "best_back_candidates_contact_sheet"),
            artifact_path(visibility, "best_front_candidates_contact_sheet"),
            artifact_path(visibility, "best_ocr_ready_crops_contact_sheet"),
            artifact_path(visibility, "top3_ocr_candidates_per_track_contact_sheet"),
            artifact_path(visibility, "rejected_or_borderline_crops_contact_sheet"),
            artifact_path(ocr, "jersey_ocr_contact_sheet"),
            artifact_path(ocr, "number_region_contact_sheet"),
            artifact_path(ocr, "per_track_evidence_contact_sheet"),
        ]
    )
    visualization_paths = present_paths(
        [
            artifact_path(demo, "heatmap_all_players_overlay"),
            artifact_path(demo, "projected_tracks_topdown"),
            artifact_path(demo, "tracking_overlay_gif"),
            artifact_path(demo, "tracking_overlay_mp4"),
            artifact_path(demo, "heatmap_by_frame_gif"),
            artifact_path(demo, "heatmap_by_frame_mp4"),
            artifact_path(demo, "side_by_side_tracking_field_gif"),
            artifact_path(demo, "side_by_side_tracking_field_mp4"),
        ]
    )
    roster_files_used = present_paths(list((roster or {}).get("inputs", [])))

    limitation_values = []
    for summary in (demo, clean, visibility, ocr):
        if summary:
            limitation_values.extend(summary.get("known_limitations", []))
            limitation_values.extend(summary.get("warnings", []))
    limitation_values.extend(
        [
            "No OCR engine was available, so no jersey-number text was recognized."
            if not ocr_engine_available
            else "OCR output still requires temporal validation before acceptance.",
            "No jersey numbers have passed track-level acceptance thresholds."
            if accepted_jersey_numbers == 0
            else "Accepted jersey numbers still require team evidence before roster matching.",
            "No player identities have been assigned.",
            "Roster numbers are not globally unique across teams, and at least one within-team duplicate exists.",
        ]
    )

    if not all_summaries_found:
        checkpoint_status = "incomplete_missing_stage_summaries"
    elif not ocr_engine_available:
        checkpoint_status = "checkpoint_complete_ocr_unavailable"
    elif accepted_jersey_numbers == 0:
        checkpoint_status = "checkpoint_complete_no_accepted_jersey_numbers"
    elif not identities_assigned:
        checkpoint_status = "checkpoint_complete_identity_pending"
    else:
        checkpoint_status = "complete"

    primary_manifest = PROJECT_ROOT / "outputs" / args.run_id / "pipeline_run_manifest.json"
    curated_manifest = PROJECT_ROOT / "progress_outputs" / "nll_test4_checkpoint" / "json" / "pipeline_run_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "checkpoint_status": checkpoint_status,
        "expected_input_summaries": {
            "expected_count": len(input_specs),
            "found_count": len(found_inputs),
            "all_found": all_summaries_found,
            "found": found_inputs,
            "missing": missing_inputs,
            "read_errors": {name: error for name, error in errors.items() if error},
        },
        "completion": {
            "calibrated_tracking_projection_demo_complete": bool(demo and demo.get("status") == "complete"),
            "clean_rgb_crop_extraction_complete": bool(clean and clean.get("status") == "complete"),
            "jersey_crop_visibility_audit_complete": bool(visibility and visibility.get("status") == "complete"),
            "ocr_baseline_stage_inspected": ocr is not None,
            "ocr_engine_available": ocr_engine_available,
            "ocr_engine_name": ocr_engine.get("name") if ocr else None,
            "ocr_downloads_performed": bool((ocr or {}).get("ocr_engine_inspection", {}).get("downloads_performed", False)),
            "accepted_jersey_number_count": accepted_jersey_numbers,
            "jersey_numbers_accepted": accepted_jersey_numbers > 0,
            "roster_metadata_validated": bool(roster and roster.get("status") in {"complete", "valid_with_warnings"}),
            "player_identities_assigned": identities_assigned,
            "player_identity_count": 0,
        },
        "stage_summaries": stage_summaries,
        "debug_tracks_export": {
            "available": debug_tracks_path.is_file() and debug_tracks_summary_path.is_file(),
            "debug_tracks": portable_path(debug_tracks_path),
            "debug_tracks_summary": portable_path(debug_tracks_summary_path),
            "curated_debug_tracks": portable_path(curated_debug_tracks_path),
            "curated_debug_tracks_summary": portable_path(curated_debug_tracks_summary_path),
            "player_names_embedded": False,
        },
        "important_outputs": {
            "directories": output_directories,
            "json_files": important_json_files,
            "contact_sheets": contact_sheet_paths,
            "videos_and_visualizations": visualization_paths,
            "roster_files_used": roster_files_used,
            "manifest_files": [portable_path(primary_manifest), portable_path(curated_manifest)],
        },
        "roster_validation": {
            "status": (roster or {}).get("status"),
            "team_count": (roster or {}).get("counts", {}).get("teams", 0),
            "player_record_count": (roster or {}).get("counts", {}).get("players", 0),
            "shared_numbers_across_teams": (roster or {}).get("counts", {}).get("shared_numbers_across_teams", 0),
            "within_team_duplicate_numbers": (roster or {}).get("counts", {}).get("within_team_duplicate_numbers", 0),
            "identity_assignment_performed": identities_assigned,
            "player_names_embedded_in_manifest": False,
        },
        "known_limitations": unique_strings(limitation_values),
        "next_recommended_step": (
            "Provision one approved lightweight OCR engine and run it only on visibility-audited clean RGB crops; "
            "aggregate complete one- or two-digit predictions across each track, add team evidence, and perform "
            "roster matching only after number and team confidence are both strong."
        ),
    }

    output_path = resolve_path(args.output)
    curated_output_path = resolve_path(args.curated_output)
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curated_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    curated_output_path.write_text(serialized, encoding="utf-8")

    result = {
        "status": "complete",
        "manifest": portable_path(output_path),
        "curated_manifest": portable_path(curated_output_path),
        "checkpoint_status": checkpoint_status,
        "all_expected_input_summaries_found": all_summaries_found,
        "missing_stage_summaries": missing_inputs,
        "ocr_engine_available": ocr_engine_available,
        "accepted_jersey_number_count": accepted_jersey_numbers,
        "player_identities_assigned": identities_assigned,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
