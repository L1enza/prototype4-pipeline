#!/usr/bin/env python3
"""Import portable Mac Tesseract results into nll_test4 debug/review artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import local Tesseract evidence without assigning player identities."
    )
    parser.add_argument(
        "--crop-predictions",
        default="outputs/nll_test4/jersey_ocr_local_results/crop_ocr_predictions.json",
    )
    parser.add_argument(
        "--track-predictions",
        default="outputs/nll_test4/jersey_ocr_local_results/track_jersey_number_predictions.json",
    )
    parser.add_argument(
        "--ocr-summary",
        default="outputs/nll_test4/jersey_ocr_local_results/jersey_ocr_summary.json",
    )
    parser.add_argument("--debug-tracks", default="outputs/nll_test4/debug_tracks.json")
    parser.add_argument(
        "--pipeline-manifest", default="outputs/nll_test4/pipeline_run_manifest.json"
    )
    parser.add_argument(
        "--number-regions-dir",
        default="outputs/nll_test4/jersey_number_ocr_baseline/number_regions",
    )
    parser.add_argument(
        "--output-debug", default="outputs/nll_test4/debug_tracks_with_local_ocr.json"
    )
    parser.add_argument(
        "--output-summary", default="outputs/nll_test4/local_ocr_import_summary.json"
    )
    parser.add_argument(
        "--progress-dir", default="progress_outputs/nll_test4_checkpoint/json"
    )
    parser.add_argument(
        "--review-export-dir", default="review_exports/nll_test4_10s_demo"
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(crop_payload: dict, track_payload: dict, ocr_summary: dict, debug_tracks: list) -> None:
    if not isinstance(debug_tracks, list):
        raise ValueError("debug_tracks.json must contain a top-level track list")
    if not ocr_summary.get("ocr_engine", {}).get("available"):
        raise ValueError("Local OCR summary does not report an available OCR engine")
    if ocr_summary.get("identity_assignment_performed") is not False:
        raise ValueError("Local OCR results unexpectedly report identity assignment")
    crop_rows = crop_payload.get("predictions", [])
    track_rows = track_payload.get("tracks", [])
    if len(crop_rows) != int(ocr_summary.get("counts", {}).get("crops_processed", -1)):
        raise ValueError("Crop prediction count does not match local OCR summary")
    if len(track_rows) != int(ocr_summary.get("counts", {}).get("tracks_processed", -1)):
        raise ValueError("Track prediction count does not match local OCR summary")
    debug_ids = [int(row["track_id"]) for row in debug_tracks]
    prediction_ids = [int(row["track_id"]) for row in track_rows]
    if len(debug_ids) != len(set(debug_ids)) or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Duplicate track IDs found in import inputs")
    if set(debug_ids) != set(prediction_ids):
        raise ValueError("Local OCR track IDs do not exactly match debug track IDs")
    if any(row.get("resolved_player_id") is not None for row in debug_tracks):
        raise ValueError("Existing debug tracks unexpectedly contain resolved player identities")


def ece_number_region_path(local_path: str | None, number_regions_dir: Path) -> str | None:
    if not local_path:
        return None
    source = Path(local_path)
    track_dir = source.parent.name
    candidate = number_regions_dir / track_dir / source.name
    return str(candidate) if candidate.is_file() else None


def compact_crop_evidence(row: dict, number_regions_dir: Path) -> dict:
    return {
        "track_id": int(row["track_id"]),
        "frame_index": int(row["frame_index"]),
        "source_frame_index": row.get("source_frame_index"),
        "timestamp_seconds": row.get("timestamp_seconds"),
        "crop_type": row.get("crop_type"),
        "crop_path": row.get("crop_path"),
        "number_region_path": ece_number_region_path(
            row.get("number_region_path"), number_regions_dir
        ),
        "local_number_region_path": row.get("number_region_path"),
        "ocr_status": row.get("status"),
        "candidate_number": row.get("candidate_number"),
        "confidence": row.get("confidence"),
        "raw_text": row.get("raw_text"),
        "preprocessing_variant": row.get("preprocessing_variant"),
        "page_segmentation_mode": row.get("page_segmentation_mode"),
        "ocr_readiness_score": row.get("ocr_readiness_score"),
        "variant_consensus": row.get("variant_consensus"),
        "reason": row.get("reason"),
    }


def imported_jersey_number(track_prediction: dict, engine: dict) -> dict:
    accepted = track_prediction.get("accepted_number")
    leading = track_prediction.get("best_candidate_number")
    if accepted is not None:
        status = "accepted"
    elif leading is not None:
        status = "candidate_low_confidence"
    else:
        status = "no_candidate"
    return {
        "status": status,
        "accepted_number": accepted,
        "best_candidate_number": leading,
        "candidate_numbers": track_prediction.get("all_candidate_numbers", []),
        "confidence": float(track_prediction.get("consensus_score") or 0.0),
        "vote_count": int(track_prediction.get("vote_count") or 0),
        "selected_crop_count": int(track_prediction.get("selected_crop_count") or 0),
        "usable_prediction_count": int(track_prediction.get("usable_prediction_count") or 0),
        "low_confidence": bool(track_prediction.get("low_confidence", True)),
        "low_confidence_reason": track_prediction.get("low_confidence_reason"),
        "reason": track_prediction.get("reason"),
        "source": "local_tesseract_mac_import",
        "ocr_engine": engine,
        "player_identity_assigned": False,
    }


def update_review_readme(readme_path: Path, counts: dict) -> None:
    text = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else "# nll_test4 10-Second Review Export\n"
    start = text.find("## Current identity status")
    end = text.find("## Geometry and scope")
    section = """## Current identity status

- Tesseract 5.5.2 was run locally on a Mac, where the OCR engine was available.
- Local OCR processed {crops} number-region crops and produced {readable} readable crop.
- Accepted track-level jersey numbers: {accepted}.
- Player names and resolved player identities were not assigned.
- Roster metadata remains available only for future constrained matching.

""".format(
        crops=int(counts.get("crops_processed", 0)),
        readable=int(counts.get("readable_crops", 0)),
        accepted=int(counts.get("accepted_track_numbers", 0)),
    )
    if start >= 0 and end > start:
        text = text[:start] + section + text[end:]
    else:
        text = text.rstrip() + "\n\n" + section
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    crop_path = project_path(args.crop_predictions)
    track_path = project_path(args.track_predictions)
    ocr_summary_path = project_path(args.ocr_summary)
    debug_path = project_path(args.debug_tracks)
    pipeline_manifest_path = project_path(args.pipeline_manifest)
    number_regions_dir = project_path(args.number_regions_dir)
    output_debug_path = project_path(args.output_debug)
    output_summary_path = project_path(args.output_summary)
    progress_dir = project_path(args.progress_dir)
    review_dir = project_path(args.review_export_dir)

    required = [
        crop_path,
        track_path,
        ocr_summary_path,
        debug_path,
        pipeline_manifest_path,
        number_regions_dir,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local OCR import inputs:\n" + "\n".join(missing))

    crop_payload = read_json(crop_path)
    track_payload = read_json(track_path)
    ocr_summary = read_json(ocr_summary_path)
    debug_tracks = read_json(debug_path)
    pipeline_manifest = read_json(pipeline_manifest_path)
    validate_inputs(crop_payload, track_payload, ocr_summary, debug_tracks)

    engine = copy.deepcopy(ocr_summary["ocr_engine"])
    engine_inspection = ocr_summary.get("ocr_engine_inspection", {})
    if engine_inspection.get("tesseract_version"):
        engine["version"] = engine_inspection["tesseract_version"].replace("tesseract ", "")
    crop_by_track = {}
    for row in crop_payload.get("predictions", []):
        crop_by_track.setdefault(int(row["track_id"]), []).append(row)
    prediction_by_track = {
        int(row["track_id"]): row for row in track_payload.get("tracks", [])
    }

    enriched = copy.deepcopy(debug_tracks)
    for track in enriched:
        track_id = int(track["track_id"])
        prediction = prediction_by_track[track_id]
        track["jersey_number_before_local_ocr"] = copy.deepcopy(track.get("jersey_number"))
        track["jersey_number"] = imported_jersey_number(prediction, engine)
        track["local_ocr_evidence"] = [
            compact_crop_evidence(row, number_regions_dir)
            for row in sorted(crop_by_track.get(track_id, []), key=lambda item: int(item["frame_index"]))
        ]
        track["local_ocr_import"] = {
            "source": "Mac portable Tesseract run",
            "engine_available": True,
            "engine": engine,
            "candidate_evidence_present": prediction.get("best_candidate_number") is not None,
            "accepted_track_number": prediction.get("accepted_number"),
            "identity_assignment_performed": False,
        }

    if any(track.get("resolved_player_id") is not None for track in enriched):
        raise RuntimeError("Identity invariant failed after local OCR import")
    if any(track["local_ocr_import"]["identity_assignment_performed"] for track in enriched):
        raise RuntimeError("Local OCR import attempted identity assignment")

    counts = ocr_summary.get("counts", {})
    tracks_with_candidates = sum(
        row.get("best_candidate_number") is not None for row in track_payload.get("tracks", [])
    )
    accepted_tracks = sum(
        row.get("accepted_number") is not None for row in track_payload.get("tracks", [])
    )
    import_summary = {
        "status": "complete",
        "stage": "local_tesseract_ocr_import",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": pipeline_manifest.get("run_id", "nll_test4"),
        "source": {
            "crop_ocr_predictions": str(crop_path),
            "track_jersey_number_predictions": str(track_path),
            "jersey_ocr_summary": str(ocr_summary_path),
            "debug_tracks": str(debug_path),
            "pipeline_run_manifest": str(pipeline_manifest_path),
            "source_hashes_sha256": {
                "crop_ocr_predictions": sha256(crop_path),
                "track_jersey_number_predictions": sha256(track_path),
                "jersey_ocr_summary": sha256(ocr_summary_path),
                "debug_tracks": sha256(debug_path),
                "pipeline_run_manifest": sha256(pipeline_manifest_path),
            },
        },
        "ocr_engine": engine,
        "counts": {
            "debug_tracks_updated": len(enriched),
            "tracks_with_local_ocr_candidate_evidence": tracks_with_candidates,
            "accepted_track_numbers": accepted_tracks,
            "crops_processed": int(counts.get("crops_processed", 0)),
            "readable_crops": int(counts.get("readable_crops", 0)),
            "unreadable_crops": int(counts.get("unreadable_crops", 0)),
            "uncertain_crops": int(counts.get("uncertain_crops", 0)),
        },
        "identity_assignment_performed": False,
        "resolved_player_identity_count": 0,
        "original_debug_tracks_overwritten": False,
        "original_ece_ocr_baseline_overwritten": False,
        "outputs": {
            "debug_tracks_with_local_ocr": str(output_debug_path),
            "local_ocr_import_summary": str(output_summary_path),
        },
    }
    if accepted_tracks != int(counts.get("accepted_track_numbers", -1)):
        raise ValueError("Accepted track count does not match local OCR summary")
    if tracks_with_candidates != int(counts.get("tracks_with_any_candidate_number", -1)):
        raise ValueError("Candidate track count does not match local OCR summary")

    write_json(output_debug_path, enriched)
    write_json(output_summary_path, import_summary)

    progress_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_debug_path, progress_dir / "debug_tracks_with_local_ocr.json")
    shutil.copy2(output_summary_path, progress_dir / "local_ocr_import_summary.json")

    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_debug_path, review_dir / "debug_tracks_with_local_ocr.json")
    shutil.copy2(output_summary_path, review_dir / "local_ocr_import_summary.json")
    shutil.copy2(ocr_summary_path, review_dir / "jersey_ocr_summary_local_tesseract.json")
    update_review_readme(review_dir / "README.md", counts)

    print(json.dumps(import_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
