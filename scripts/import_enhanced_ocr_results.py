#!/usr/bin/env python3
"""Import enhanced Mac Tesseract evidence without changing player identity state."""

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
        description="Import enhanced local OCR evidence into new debug/review artifacts."
    )
    parser.add_argument(
        "--crop-predictions",
        default="outputs/nll_test4/jersey_ocr_enhanced_local_results/crop_ocr_predictions.json",
    )
    parser.add_argument(
        "--track-predictions",
        default="outputs/nll_test4/jersey_ocr_enhanced_local_results/track_jersey_number_predictions.json",
    )
    parser.add_argument(
        "--ocr-summary",
        default="outputs/nll_test4/jersey_ocr_enhanced_local_results/jersey_ocr_summary.json",
    )
    parser.add_argument(
        "--debug-tracks", default="outputs/nll_test4/debug_tracks_with_local_ocr.json"
    )
    parser.add_argument(
        "--pipeline-manifest", default="outputs/nll_test4/pipeline_run_manifest.json"
    )
    parser.add_argument(
        "--enhanced-regions-dir",
        default="outputs/nll_test4/enhanced_number_regions/regions",
    )
    parser.add_argument(
        "--output-debug", default="outputs/nll_test4/debug_tracks_with_enhanced_ocr.json"
    )
    parser.add_argument(
        "--output-summary", default="outputs/nll_test4/enhanced_ocr_import_summary.json"
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


def validate_inputs(crops: dict, tracks: dict, summary: dict, debug_tracks: list) -> None:
    if not isinstance(debug_tracks, list):
        raise ValueError("Enhanced OCR debug input must contain a top-level track list")
    if not summary.get("ocr_engine", {}).get("available"):
        raise ValueError("Enhanced OCR summary does not report an available OCR engine")
    if summary.get("identity_assignment_performed") is not False:
        raise ValueError("Enhanced OCR unexpectedly reports identity assignment")
    crop_rows = crops.get("predictions", [])
    track_rows = tracks.get("tracks", [])
    counts = summary.get("counts", {})
    if len(crop_rows) != int(counts.get("crops_processed", -1)):
        raise ValueError("Enhanced crop prediction count does not match summary")
    if len(track_rows) != int(counts.get("tracks_processed", -1)):
        raise ValueError("Enhanced track prediction count does not match summary")
    debug_ids = {int(row["track_id"]) for row in debug_tracks}
    prediction_ids = [int(row["track_id"]) for row in track_rows]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Duplicate enhanced OCR track IDs")
    if not set(prediction_ids).issubset(debug_ids):
        raise ValueError("Enhanced OCR contains track IDs absent from debug tracks")
    if any(row.get("resolved_player_id") is not None for row in debug_tracks):
        raise ValueError("Input debug tracks unexpectedly contain resolved player identities")


def build_region_index(regions_dir: Path) -> dict[str, str]:
    paths = sorted(regions_dir.rglob("*.png"))
    by_name = {}
    duplicates = []
    for path in paths:
        if path.name in by_name:
            duplicates.append(path.name)
        by_name[path.name] = str(path)
    if duplicates:
        raise ValueError("Enhanced region filenames are not unique: {}".format(duplicates[:5]))
    if not by_name:
        raise ValueError("No enhanced number-region PNGs found under {}".format(regions_dir))
    return by_name


def compact_crop_evidence(row: dict, region_index: dict[str, str]) -> dict:
    local_path = row.get("number_region_path")
    filename = Path(local_path).name if local_path else None
    ece_path = region_index.get(filename) if filename else None
    if ece_path is None:
        raise FileNotFoundError("Could not map enhanced local region to ECE: {}".format(local_path))
    return {
        "track_id": int(row["track_id"]),
        "frame_index": int(row["frame_index"]),
        "source_frame_index": row.get("source_frame_index"),
        "timestamp_seconds": row.get("timestamp_seconds"),
        "crop_type": row.get("crop_type"),
        "source_crop_path": row.get("crop_path"),
        "number_region_path": ece_path,
        "local_number_region_path": local_path,
        "number_region_variant": row.get("number_region_variant"),
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


def enhanced_number_record(prediction: dict | None, engine: dict) -> dict:
    if prediction is None:
        return {
            "status": "not_evaluated",
            "accepted_number": None,
            "best_candidate_number": None,
            "candidate_numbers": [],
            "confidence": 0.0,
            "low_confidence": True,
            "reason": "Track had no OCR-ready enhanced source crop.",
            "source": "enhanced_local_tesseract_mac_import",
            "ocr_engine": engine,
            "player_identity_assigned": False,
        }
    accepted = prediction.get("accepted_number")
    leading = prediction.get("best_candidate_number")
    status = "accepted" if accepted is not None else (
        "candidate_low_confidence" if leading is not None else "no_candidate"
    )
    return {
        "status": status,
        "accepted_number": accepted,
        "best_candidate_number": leading,
        "candidate_numbers": prediction.get("all_candidate_numbers", []),
        "confidence": float(prediction.get("consensus_score") or 0.0),
        "vote_count": int(prediction.get("vote_count") or 0),
        "selected_source_frame_count": int(prediction.get("selected_crop_count") or 0),
        "variant_prediction_count": int(prediction.get("variant_prediction_count") or 0),
        "usable_prediction_count": int(prediction.get("usable_prediction_count") or 0),
        "low_confidence": bool(prediction.get("low_confidence", True)),
        "low_confidence_reason": prediction.get("low_confidence_reason"),
        "reason": prediction.get("reason"),
        "source": "enhanced_local_tesseract_mac_import",
        "ocr_engine": engine,
        "player_identity_assigned": False,
    }


def replace_section(text: str, header: str, next_header: str, replacement: str) -> str:
    start = text.find(header)
    end = text.find(next_header)
    if start >= 0 and end > start:
        return text[:start] + replacement.rstrip() + "\n\n" + text[end:]
    return text.rstrip() + "\n\n" + replacement.rstrip() + "\n"


def update_review_readme(readme_path: Path, counts: dict) -> None:
    text = readme_path.read_text(encoding="utf-8")
    debug_section = """## Debug evidence

`debug_tracks.json` is the original consolidated track export.
`debug_tracks_with_local_ocr.json` adds the basic local Tesseract evidence, and
`debug_tracks_with_enhanced_ocr.json` adds the enhanced-region experiment while
preserving the basic evidence. Import summaries retain source hashes and counts.
"""
    identity_section = """## Current identity status

- Enhanced number regions were generated and tested locally with Tesseract 5.5.2.
- Enhanced OCR processed {processed} regions and produced {readable} readable crops.
- Accepted track-level jersey numbers: {accepted}.
- No player names or resolved player identities were assigned.
- The current bottleneck is jersey digit recognition, not roster lookup.
""".format(
        processed=int(counts.get("crops_processed", 0)),
        readable=int(counts.get("readable_crops", 0)),
        accepted=int(counts.get("accepted_track_numbers", 0)),
    )
    text = replace_section(text, "## Debug evidence", "## Current identity status", debug_section)
    text = replace_section(text, "## Current identity status", "## Geometry and scope", identity_section)
    readme_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    crop_path = project_path(args.crop_predictions)
    track_path = project_path(args.track_predictions)
    summary_path = project_path(args.ocr_summary)
    debug_path = project_path(args.debug_tracks)
    pipeline_path = project_path(args.pipeline_manifest)
    regions_dir = project_path(args.enhanced_regions_dir)
    output_debug = project_path(args.output_debug)
    output_summary = project_path(args.output_summary)
    progress_dir = project_path(args.progress_dir)
    review_dir = project_path(args.review_export_dir)
    required = [crop_path, track_path, summary_path, debug_path, pipeline_path, regions_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing enhanced OCR import inputs:\n" + "\n".join(missing))

    crops = read_json(crop_path)
    tracks = read_json(track_path)
    ocr_summary = read_json(summary_path)
    debug_tracks = read_json(debug_path)
    pipeline = read_json(pipeline_path)
    validate_inputs(crops, tracks, ocr_summary, debug_tracks)
    region_index = build_region_index(regions_dir)

    engine = copy.deepcopy(ocr_summary["ocr_engine"])
    version = ocr_summary.get("ocr_engine_inspection", {}).get("tesseract_version")
    if version:
        engine["version"] = version.replace("tesseract ", "")
    predictions_by_track = {
        int(row["track_id"]): row for row in tracks.get("tracks", [])
    }
    crops_by_track = {}
    for row in crops.get("predictions", []):
        crops_by_track.setdefault(int(row["track_id"]), []).append(row)

    enriched = copy.deepcopy(debug_tracks)
    for track in enriched:
        track_id = int(track["track_id"])
        prediction = predictions_by_track.get(track_id)
        track["enhanced_jersey_number"] = enhanced_number_record(prediction, engine)
        track["enhanced_ocr_evidence"] = [
            compact_crop_evidence(row, region_index)
            for row in sorted(
                crops_by_track.get(track_id, []),
                key=lambda item: (
                    int(item["frame_index"]), item.get("number_region_variant", "")
                ),
            )
        ]
        track["enhanced_ocr_import"] = {
            "evaluated": prediction is not None,
            "source": "Mac enhanced-region Tesseract run",
            "engine_available": True,
            "engine": engine,
            "candidate_evidence_present": bool(
                prediction and prediction.get("best_candidate_number") is not None
            ),
            "accepted_track_number": prediction.get("accepted_number") if prediction else None,
            "identity_assignment_performed": False,
        }
        track["jersey_ocr_experiment_comparison"] = {
            "basic": {
                "best_candidate_number": track.get("jersey_number", {}).get("best_candidate_number"),
                "accepted_number": track.get("jersey_number", {}).get("accepted_number"),
                "status": track.get("jersey_number", {}).get("status"),
            },
            "enhanced": {
                "best_candidate_number": track["enhanced_jersey_number"].get("best_candidate_number"),
                "accepted_number": track["enhanced_jersey_number"].get("accepted_number"),
                "status": track["enhanced_jersey_number"].get("status"),
            },
        }

    if any(track.get("resolved_player_id") is not None for track in enriched):
        raise RuntimeError("Identity invariant failed after enhanced OCR import")
    if any(track["enhanced_ocr_import"]["identity_assignment_performed"] for track in enriched):
        raise RuntimeError("Enhanced OCR import attempted identity assignment")

    counts = ocr_summary.get("counts", {})
    candidate_tracks = sum(
        row.get("best_candidate_number") is not None for row in tracks.get("tracks", [])
    )
    accepted_tracks = sum(
        row.get("accepted_number") is not None for row in tracks.get("tracks", [])
    )
    import_summary = {
        "status": "complete",
        "stage": "enhanced_local_tesseract_ocr_import",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": pipeline.get("run_id", "nll_test4"),
        "source": {
            "crop_ocr_predictions": str(crop_path),
            "track_jersey_number_predictions": str(track_path),
            "jersey_ocr_summary": str(summary_path),
            "debug_tracks_with_local_ocr": str(debug_path),
            "pipeline_run_manifest": str(pipeline_path),
            "source_hashes_sha256": {
                "crop_ocr_predictions": sha256(crop_path),
                "track_jersey_number_predictions": sha256(track_path),
                "jersey_ocr_summary": sha256(summary_path),
                "debug_tracks_with_local_ocr": sha256(debug_path),
                "pipeline_run_manifest": sha256(pipeline_path),
            },
        },
        "ocr_engine": engine,
        "comparison": {
            "basic_local_tesseract": {
                "crops_processed": 90,
                "readable_crops": 1,
                "accepted_track_numbers": 0,
            },
            "enhanced_local_tesseract": {
                "crops_processed": int(counts.get("crops_processed", 0)),
                "readable_crops": int(counts.get("readable_crops", 0)),
                "accepted_track_numbers": accepted_tracks,
            },
        },
        "counts": {
            "debug_tracks_total": len(enriched),
            "tracks_evaluated_with_enhanced_ocr": len(predictions_by_track),
            "tracks_not_evaluated_with_enhanced_ocr": len(enriched) - len(predictions_by_track),
            "tracks_with_enhanced_candidate_evidence": candidate_tracks,
            "accepted_track_numbers": accepted_tracks,
            "crops_processed": int(counts.get("crops_processed", 0)),
            "readable_crops": int(counts.get("readable_crops", 0)),
            "unreadable_crops": int(counts.get("unreadable_crops", 0)),
            "uncertain_crops": int(counts.get("uncertain_crops", 0)),
        },
        "identity_assignment_performed": False,
        "resolved_player_identity_count": 0,
        "original_ocr_outputs_overwritten": False,
        "basic_local_ocr_import_overwritten": False,
        "recommendation": "Move away from generic Tesseract as the main jersey-number recognizer.",
        "outputs": {
            "debug_tracks_with_enhanced_ocr": str(output_debug),
            "enhanced_ocr_import_summary": str(output_summary),
        },
    }
    if candidate_tracks != int(counts.get("tracks_with_any_candidate_number", -1)):
        raise ValueError("Enhanced candidate track count does not match summary")
    if accepted_tracks != int(counts.get("accepted_track_numbers", -1)):
        raise ValueError("Enhanced accepted track count does not match summary")

    write_json(output_debug, enriched)
    write_json(output_summary, import_summary)
    progress_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_debug, progress_dir / "debug_tracks_with_enhanced_ocr.json")
    shutil.copy2(output_summary, progress_dir / "enhanced_ocr_import_summary.json")
    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_debug, review_dir / "debug_tracks_with_enhanced_ocr.json")
    shutil.copy2(output_summary, review_dir / "enhanced_ocr_import_summary.json")
    shutil.copy2(summary_path, review_dir / "jersey_ocr_summary_enhanced_tesseract.json")
    update_review_readme(review_dir / "README.md", counts)

    print(json.dumps(import_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
