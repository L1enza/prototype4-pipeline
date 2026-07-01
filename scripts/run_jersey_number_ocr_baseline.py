#!/usr/bin/env python3
"""Run a top-K, track-level jersey-number OCR baseline on audited clean crops."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_METADATA = "outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json"
DEFAULT_AUDIT = "outputs/nll_test4/jersey_crop_visibility_audit/crop_visibility_predictions.json"
DEFAULT_TRACK_AUDIT = "outputs/nll_test4/jersey_crop_visibility_audit/track_visibility_summary.json"
DEFAULT_OUTPUT = "outputs/nll_test4/jersey_number_ocr_baseline"
DIGIT_WHITELIST = "0123456789"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-K jersey-number OCR and temporal voting.")
    parser.add_argument(
        "--clean-crop-metadata",
        "--crop-metadata",
        dest="crop_metadata",
        default=DEFAULT_CLEAN_METADATA,
        help="Clean crop metadata JSON. --crop-metadata is the portable alias.",
    )
    parser.add_argument(
        "--visibility-audit",
        "--visibility-predictions",
        dest="visibility_predictions",
        default=DEFAULT_AUDIT,
        help="Crop visibility predictions JSON. --visibility-predictions is the portable alias.",
    )
    parser.add_argument("--track-visibility-summary", default=DEFAULT_TRACK_AUDIT)
    parser.add_argument(
        "--number-regions-dir",
        default=None,
        help="Use existing packaged number-region images instead of regenerating them.",
    )
    parser.add_argument(
        "--number-region-manifest",
        default=None,
        help="Use enhanced regions from a manifest; clean crop/audit inputs become optional.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=["auto", "tesseract", "none"], default="auto")
    parser.add_argument("--top-k", type=int, default=5, help="Unique source frames selected per track.")
    parser.add_argument("--all-crops", action="store_true", help="Process all audited crops instead of top-K.")
    parser.add_argument("--minimum-two-digit-votes", type=int, default=2)
    parser.add_argument("--minimum-one-digit-votes", type=int, default=3)
    parser.add_argument("--minimum-consensus", type=float, default=0.67)
    parser.add_argument("--minimum-ocr-confidence", type=float, default=0.45)
    parser.add_argument("--contact-thumb-width", type=int, default=180)
    parser.add_argument("--contact-thumb-height", type=int, default=190)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect_ocr_engines() -> dict:
    modules = {
        module: importlib.util.find_spec(module) is not None
        for module in (
            "pytesseract",
            "easyocr",
            "paddleocr",
            "rapidocr_onnxruntime",
            "onnxruntime",
        )
    }
    binary = shutil.which("tesseract")
    version = None
    if binary:
        result = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, check=False, timeout=10
        )
        if result.stdout:
            version = result.stdout.splitlines()[0].strip()
    return {
        "pytesseract_installed": modules["pytesseract"],
        "tesseract_binary_installed": bool(binary),
        "tesseract_binary": binary,
        "tesseract_version": version,
        "optional_modules": modules,
        "downloads_performed": False,
        "packages_installed_by_script": False,
    }


def choose_engine(requested: str, inspection: dict) -> dict:
    available = inspection["tesseract_binary_installed"]
    if requested == "none":
        return {"name": "none", "available": False, "reason": "OCR disabled by --engine none."}
    if requested == "tesseract" and not available:
        return {
            "name": "tesseract_cli",
            "available": False,
            "reason": "The system Tesseract binary is not installed.",
        }
    if available:
        return {
            "name": "tesseract_cli",
            "available": True,
            "reason": "Using the existing system Tesseract binary without downloads.",
            "digit_whitelist": DIGIT_WHITELIST,
        }
    return {
        "name": "none",
        "available": False,
        "reason": "Neither a usable Tesseract CLI nor another approved lightweight OCR engine is available.",
    }


def audit_records(payload: dict) -> list[dict]:
    records = payload.get("crops", payload.get("predictions", []))
    if not isinstance(records, list):
        raise ValueError("Visibility audit must contain a crops or predictions array")
    return records


def combine_clean_and_audit(
    clean_payload: dict,
    audit_payload: dict,
    require_clean_crop_files: bool = True,
) -> list[dict]:
    clean_by_path = {
        str(Path(row["crop_path"]).resolve()): row for row in clean_payload.get("crops", [])
    }
    combined = []
    missing_clean_rows = []
    for audit in audit_records(audit_payload):
        crop_path = Path(audit.get("crop_path", "")).expanduser().resolve()
        clean = clean_by_path.get(str(crop_path))
        if clean is None:
            missing_clean_rows.append(str(crop_path))
            continue
        if require_clean_crop_files and not crop_path.is_file():
            continue
        readiness = audit.get(
            "ocr_readiness_score",
            audit.get("auto_quality_score", audit.get("quality_score", clean.get("crop_quality_score", 0.0))),
        )
        view = audit.get("heuristic_likely_view", audit.get("likely_view", "unknown"))
        row = dict(clean)
        row.update(
            {
                "crop_path": str(crop_path),
                "ocr_readiness_score": float(readiness or 0.0),
                "heuristic_likely_view": view or "unknown",
                "audit_ocr_candidate": bool(audit.get("ocr_candidate", False)),
                "audit_status": audit.get("audit_status"),
                "audit_rejection_reason": audit.get("rejection_reason"),
                "number_region_box": audit.get("number_region_box"),
                "number_region_guess": audit.get("number_region_guess", "unknown"),
            }
        )
        combined.append(row)
    if not combined:
        raise ValueError("No clean crops could be joined to visibility-audit records")
    if missing_clean_rows:
        print(f"WARNING: {len(missing_clean_rows)} audit rows had no matching clean metadata record")
    return combined


def select_top_crops(records: list[dict], top_k: int, all_crops: bool) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["track_id"])].append(row)
    selected = []
    for track_id in sorted(grouped):
        rows = sorted(
            grouped[track_id],
            key=lambda row: (
                -float(row["ocr_readiness_score"]),
                0 if row.get("crop_type") == "torso" else 1,
                int(row.get("frame_index", 0)),
            ),
        )
        if all_crops:
            selected.extend(rows)
            continue
        unique_frames = []
        used_frames = set()
        for row in rows:
            frame_index = int(row["frame_index"])
            if frame_index in used_frames:
                continue
            used_frames.add(frame_index)
            unique_frames.append(row)
            if len(unique_frames) >= top_k:
                break
        selected.extend(unique_frames)
    return selected


def enhanced_manifest_records(payload: dict, manifest_path: Path) -> list[dict]:
    records = payload.get("regions", [])
    if not isinstance(records, list) or not records:
        raise ValueError("Enhanced number-region manifest must contain a non-empty regions array")
    output = []
    for record in records:
        region_path = Path(record.get("region_path", "")).expanduser()
        if not region_path.is_absolute():
            region_path = manifest_path.parent / region_path
        if not region_path.is_file():
            raise FileNotFoundError("Missing enhanced number region: {}".format(region_path))
        row = dict(record)
        row.update(
            {
                "region_path": str(region_path),
                "crop_path": record.get("source_crop_path", record.get("crop_path")),
                "ocr_readiness_score": float(record.get("ocr_readiness_score") or 0.0),
                "heuristic_likely_view": record.get("likely_view", "unknown"),
                "audit_ocr_candidate": bool(record.get("audit_ocr_candidate", True)),
                "number_region_guess": record.get("number_region_guess", "unknown"),
                "manifest_preprocessed_region": bool(
                    record.get("manifest_preprocessed_region", True)
                ),
            }
        )
        output.append(row)
    return output


def select_manifest_regions(records: list[dict], top_k: int, all_crops: bool) -> list[dict]:
    if all_crops:
        return sorted(
            records,
            key=lambda row: (
                int(row["track_id"]),
                int(row.get("ocr_readiness_rank_within_track", 0)),
                int(row["frame_index"]),
                row.get("variant_name", ""),
            ),
        )

    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        key = (
            int(row["frame_index"]),
            str(row.get("source_crop_path", row.get("crop_path", ""))),
        )
        grouped[int(row["track_id"])][key].append(row)

    selected = []
    for track_id in sorted(grouped):
        source_groups = list(grouped[track_id].items())
        source_groups.sort(
            key=lambda item: (
                -max(float(row["ocr_readiness_score"]) for row in item[1]),
                0 if any(row.get("crop_type") == "torso" for row in item[1]) else 1,
                item[0][0],
            )
        )
        used_frames = set()
        selected_sources = []
        for (frame_index, _source_path), rows in source_groups:
            if frame_index in used_frames:
                continue
            used_frames.add(frame_index)
            selected_sources.append(rows)
            if len(selected_sources) >= top_k:
                break
        for rows in selected_sources:
            selected.extend(sorted(rows, key=lambda row: row.get("variant_name", "")))
    return selected


def default_number_region_box(image: np.ndarray, crop_type: str) -> dict:
    height, width = image.shape[:2]
    if crop_type == "torso":
        return {
            "x0": int(width * 0.05),
            "y0": int(height * 0.03),
            "x1": max(0, int(width * 0.95) - 1),
            "y1": max(0, int(height * 0.97) - 1),
        }
    return {
        "x0": int(width * 0.10),
        "y0": int(height * 0.06),
        "x1": max(0, int(width * 0.90) - 1),
        "y1": max(0, int(height * 0.68) - 1),
    }


def extract_number_region(row: dict, output_root: Path) -> dict:
    image = cv2.imread(row["crop_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read clean crop: {row['crop_path']}")
    height, width = image.shape[:2]
    box = row.get("number_region_box") or default_number_region_box(image, row.get("crop_type", ""))
    x0 = max(0, min(width - 1, int(box.get("x0", 0))))
    y0 = max(0, min(height - 1, int(box.get("y0", 0))))
    x1 = max(x0, min(width - 1, int(box.get("x1", width - 1))))
    y1 = max(y0, min(height - 1, int(box.get("y1", height - 1))))
    pad_x = max(1, int(round((x1 - x0 + 1) * 0.05)))
    pad_y = max(1, int(round((y1 - y0 + 1) * 0.05)))
    x0, x1 = max(0, x0 - pad_x), min(width - 1, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(height - 1, y1 + pad_y)
    region = image[y0 : y1 + 1, x0 : x1 + 1].copy()
    if region.size == 0:
        raise RuntimeError(f"Empty number region for {row['crop_path']}")
    track_id = int(row["track_id"])
    region_dir = output_root / "track_{:03d}".format(track_id)
    region_dir.mkdir(parents=True, exist_ok=True)
    path = region_dir / "track_{:03d}_frame_{:03d}_{}_number_region.png".format(
        track_id, int(row["frame_index"]), row.get("crop_type", "crop")
    )
    if not cv2.imwrite(str(path), region):
        raise RuntimeError(f"Could not write number region: {path}")
    return {
        "image": region,
        "path": str(path),
        "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "width": int(region.shape[1]),
        "height": int(region.shape[0]),
    }


def load_number_region(row: dict, number_regions_root: Path) -> dict:
    track_id = int(row["track_id"])
    frame_index = int(row["frame_index"])
    crop_type = row.get("crop_type", "crop")
    path = (
        number_regions_root
        / "track_{:03d}".format(track_id)
        / "track_{:03d}_frame_{:03d}_{}_number_region.png".format(
            track_id, frame_index, crop_type
        )
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Missing packaged number region for track {} frame {} crop type {}: {}".format(
                track_id, frame_index, crop_type, path
            )
        )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read packaged number region: {}".format(path))
    height, width = image.shape[:2]
    return {
        "image": image,
        "path": str(path),
        "box": row.get("number_region_box"),
        "width": int(width),
        "height": int(height),
        "source": "pre_extracted_number_region",
    }


def load_manifest_number_region(row: dict) -> dict:
    path = Path(row["region_path"])
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not read enhanced number region: {}".format(path))
    height, width = image.shape[:2]
    return {
        "image": image,
        "path": str(path),
        "box": row.get("source_region_box"),
        "width": int(width),
        "height": int(height),
        "source": "enhanced_number_region_manifest",
        "variant_name": row.get("variant_name"),
        "manifest_preprocessed_region": bool(row.get("manifest_preprocessed_region", True)),
    }


def preprocessing_variants(region_bgr: np.ndarray) -> list[dict]:
    height, width = region_bgr.shape[:2]
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(6, 6))
    contrast = clahe.apply(gray)
    _threshold, thresholded = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    scale = max(2.0, min(5.0, 200.0 / max(height, 1)))
    upscaled = cv2.resize(
        contrast,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )
    return [
        {"name": "original", "image": region_bgr, "psm_modes": [7]},
        {"name": "grayscale", "image": gray, "psm_modes": [7]},
        {"name": "contrast_enhanced", "image": contrast, "psm_modes": [7, 8]},
        {"name": "thresholded", "image": thresholded, "psm_modes": [8, 13]},
        {"name": "enlarged_upscaled", "image": upscaled, "psm_modes": [7, 8]},
    ]


def clean_digits(text: str | None) -> dict:
    raw = "" if text is None else str(text)
    digit_string = "".join(character for character in raw if character.isdigit())
    candidate = digit_string if len(digit_string) in (1, 2) else None
    return {
        "parsed_digits": list(digit_string),
        "digit_string": digit_string,
        "candidate_number": candidate,
        "valid_jersey_length": candidate is not None,
    }


def run_tesseract_attempt(image: np.ndarray, tesseract_binary: str, variant: str, psm: int) -> dict:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temporary_path = Path(handle.name)
        if not cv2.imwrite(str(temporary_path), image):
            raise RuntimeError("Could not write temporary OCR input")
        command = [
            tesseract_binary,
            str(temporary_path),
            "stdout",
            "-l",
            "eng",
            "--psm",
            str(psm),
            "-c",
            f"tessedit_char_whitelist={DIGIT_WHITELIST}",
            "tsv",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Tesseract exited {result.returncode}")
        tokens = []
        confidences = []
        reader = csv.DictReader(io.StringIO(result.stdout), delimiter="\t")
        for row in reader:
            token = (row.get("text") or "").strip()
            if not token:
                continue
            tokens.append(token)
            try:
                confidence = float(row.get("conf", -1))
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence >= 0:
                confidences.append(confidence / 100.0)
        raw_text = " ".join(tokens)
        return {
            "preprocessing_variant": variant,
            "page_segmentation_mode": int(psm),
            "raw_text": raw_text,
            **clean_digits(raw_text),
            "confidence": float(np.mean(confidences)) if confidences else None,
            "tokens": tokens,
            "error": None,
        }
    except Exception as exc:
        return {
            "preprocessing_variant": variant,
            "page_segmentation_mode": int(psm),
            "raw_text": None,
            **clean_digits(None),
            "confidence": None,
            "tokens": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def select_best_attempt(attempts: list[dict]) -> tuple[dict | None, dict]:
    usable = [attempt for attempt in attempts if attempt.get("candidate_number")]
    if not usable:
        nonempty = [attempt for attempt in attempts if attempt.get("digit_string")]
        best = max(nonempty, key=lambda row: row.get("confidence") or 0.0) if nonempty else None
        return best, {"variant_candidate_votes": [], "variant_consensus": 0.0}
    grouped = defaultdict(list)
    for attempt in usable:
        grouped[attempt["candidate_number"]].append(attempt)
    rankings = []
    for number, rows in grouped.items():
        confidence_sum = sum(row.get("confidence") or 0.25 for row in rows)
        rankings.append(
            {
                "number": number,
                "variant_vote_count": len(rows),
                "confidence_sum": confidence_sum,
                "mean_confidence": float(np.mean([row.get("confidence") or 0.25 for row in rows])),
            }
        )
    supported_two_digits = [row for row in rankings if len(row["number"]) == 2 and row["variant_vote_count"] >= 2]
    pool = supported_two_digits or rankings
    pool.sort(key=lambda row: (-row["variant_vote_count"], -row["confidence_sum"], row["number"]))
    winner = pool[0]
    winning_attempts = grouped[winner["number"]]
    best = max(winning_attempts, key=lambda row: row.get("confidence") or 0.0)
    total_votes = sum(row["variant_vote_count"] for row in rankings)
    rankings.sort(key=lambda row: (-row["variant_vote_count"], -row["confidence_sum"], row["number"]))
    return best, {
        "variant_candidate_votes": rankings,
        "variant_consensus": winner["variant_vote_count"] / max(total_votes, 1),
    }


def predict_crop(
    row: dict,
    region: dict,
    engine: dict,
    inspection: dict,
    minimum_confidence: float,
) -> dict:
    record = {
        "track_id": int(row["track_id"]),
        "frame_index": int(row["frame_index"]),
        "source_frame_index": row.get("source_frame_index"),
        "timestamp_seconds": row.get("timestamp_seconds"),
        "crop_path": row["crop_path"],
        "crop_type": row.get("crop_type"),
        "heuristic_likely_view": row.get("heuristic_likely_view", "unknown"),
        "number_region_guess": row.get("number_region_guess", "unknown"),
        "ocr_readiness_score": row["ocr_readiness_score"],
        "audit_ocr_candidate": row.get("audit_ocr_candidate"),
        "number_region_path": region["path"],
        "number_region_box": region["box"],
        "number_region_width": region["width"],
        "number_region_height": region["height"],
        "number_region_variant": region.get("variant_name", row.get("variant_name")),
        "number_region_source": region.get("source"),
        "ocr_engine": engine["name"],
        "preprocessing_variant": None,
        "page_segmentation_mode": None,
        "raw_text": None,
        "parsed_digits": [],
        "digit_string": "",
        "candidate_number": None,
        "confidence": None,
        "status": "ocr_unavailable",
        "reason": engine["reason"],
        "ocr_attempts": [],
        "variant_candidate_votes": [],
        "variant_consensus": 0.0,
    }
    if not engine["available"]:
        return record

    attempts = []
    variants = (
        [
            {
                "name": region.get("variant_name", "enhanced_manifest_region"),
                "image": region["image"],
                "psm_modes": [7, 8, 13],
            }
        ]
        if region.get("manifest_preprocessed_region")
        else preprocessing_variants(region["image"])
    )
    for variant in variants:
        for psm in variant["psm_modes"]:
            attempts.append(
                run_tesseract_attempt(
                    variant["image"], inspection["tesseract_binary"], variant["name"], psm
                )
            )
    best, variant_summary = select_best_attempt(attempts)
    record["ocr_attempts"] = attempts
    record.update(variant_summary)
    if best is None:
        record.update({"status": "unreadable", "reason": "No preprocessing variant returned digits."})
        return record

    record.update(
        {
            "preprocessing_variant": best["preprocessing_variant"],
            "page_segmentation_mode": best["page_segmentation_mode"],
            "raw_text": best["raw_text"],
            "parsed_digits": best["parsed_digits"],
            "digit_string": best["digit_string"],
            "candidate_number": best["candidate_number"],
            "confidence": best["confidence"],
        }
    )
    candidate = best["candidate_number"]
    confidence = best["confidence"]
    if candidate is None:
        record.update(
            {"status": "uncertain", "reason": "OCR digits did not form a one- or two-digit number."}
        )
    elif len(candidate) == 1:
        record.update(
            {
                "status": "uncertain",
                "reason": "One digit detected; repeated track evidence is required because it may be partial.",
            }
        )
    elif confidence is None or confidence < minimum_confidence:
        record.update(
            {"status": "uncertain", "reason": "Two digits detected below the OCR confidence threshold."}
        )
    else:
        record.update(
            {"status": "readable", "reason": "Two-digit crop evidence; track consensus is still required."}
        )
    return record


def aggregate_tracks(
    predictions: list[dict],
    engine_available: bool,
    track_audit: dict,
    minimum_two_digit_votes: int,
    minimum_one_digit_votes: int,
    minimum_consensus: float,
) -> list[dict]:
    audit_by_track = {
        int(row["track_id"]): row for row in track_audit.get("tracks", [])
    }
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["track_id"])].append(prediction)
    output = []
    for track_id in sorted(grouped):
        all_rows = grouped[track_id]
        rows = collapse_same_frame_predictions(all_rows)
        usable = [row for row in rows if row.get("candidate_number")]
        raw_counts = Counter(row["candidate_number"] for row in usable)
        two_digit_values = {number for number in raw_counts if len(number) == 2}
        weighted_votes = defaultdict(float)
        frames = defaultdict(list)
        confidences = defaultdict(list)
        partial_downweights = []
        for row in usable:
            number = row["candidate_number"]
            confidence = row.get("confidence")
            weight = (float(confidence) if confidence is not None else 0.30) * max(
                0.10, float(row.get("ocr_readiness_score") or 0.0)
            )
            if len(number) == 1 and any(number in complete for complete in two_digit_values):
                matching = sorted(complete for complete in two_digit_values if number in complete)
                weight *= 0.25
                partial_downweights.append(
                    {"frame_index": row["frame_index"], "partial_digit": number, "matching_complete_candidates": matching}
                )
            weighted_votes[number] += weight
            frames[number].append(row["frame_index"])
            if confidence is not None:
                confidences[number].append(float(confidence))

        ranking = sorted(
            raw_counts,
            key=lambda number: (-weighted_votes[number], -raw_counts[number], number),
        )
        best = ranking[0] if ranking else None
        total_weight = sum(weighted_votes.values())
        consensus = weighted_votes[best] / total_weight if best and total_weight else 0.0
        vote_count = raw_counts[best] if best else 0
        matching_complete_conflict = bool(
            best and len(best) == 1 and any(best in complete for complete in two_digit_values)
        )
        required_votes = minimum_one_digit_votes if best and len(best) == 1 else minimum_two_digit_votes
        accepted = bool(
            best
            and vote_count >= required_votes
            and consensus >= minimum_consensus
            and not matching_complete_conflict
        )
        if not engine_available:
            reason = "OCR engine unavailable; number regions were generated for later processing."
        elif not best:
            reason = "No crop produced a valid one- or two-digit candidate."
        elif matching_complete_conflict:
            reason = "Leading one-digit evidence may be a partial reading of a two-digit candidate."
        elif vote_count < required_votes:
            reason = "Insufficient repeated observations for the leading candidate."
        elif consensus < minimum_consensus:
            reason = "Candidate predictions are inconsistent across selected frames."
        elif accepted:
            reason = "Repeated, consistent track-level number evidence."
        else:
            reason = "Evidence did not pass conservative acceptance rules."
        candidate_rows = []
        for number in ranking:
            values = confidences[number]
            candidate_rows.append(
                {
                    "number": number,
                    "vote_count": raw_counts[number],
                    "weighted_vote": weighted_votes[number],
                    "mean_ocr_confidence": float(np.mean(values)) if values else None,
                    "evidence_frames": sorted(set(frames[number])),
                }
            )
        output.append(
            {
                "track_id": track_id,
                "visibility_readiness": audit_by_track.get(track_id, {}).get("readiness"),
                "all_candidate_numbers": candidate_rows,
                "best_candidate_number": best,
                "vote_count": vote_count,
                "consensus_score": consensus,
                "accepted_number": best if accepted else None,
                "low_confidence": not accepted,
                "low_confidence_reason": None if accepted else reason,
                "reason": reason,
                "selected_crop_count": len(rows),
                "variant_prediction_count": len(all_rows),
                "usable_prediction_count": len(usable),
                "partial_digit_downweights": partial_downweights,
                "evidence": rows,
            }
        )
    return output


def collapse_same_frame_predictions(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_index"])].append(row)
    collapsed = []
    for frame_index in sorted(grouped):
        frame_rows = grouped[frame_index]
        usable = [row for row in frame_rows if row.get("candidate_number")]
        if not usable:
            representative = dict(
                max(frame_rows, key=lambda row: float(row.get("ocr_readiness_score") or 0.0))
            )
            representative["variant_prediction_count"] = len(frame_rows)
            representative["frame_variant_candidate_votes"] = []
            representative["frame_variant_consensus"] = 0.0
            collapsed.append(representative)
            continue

        by_number = defaultdict(list)
        for row in usable:
            by_number[row["candidate_number"]].append(row)
        ranking = []
        for number, number_rows in by_number.items():
            confidence_sum = sum(
                float(row["confidence"]) if row.get("confidence") is not None else 0.25
                for row in number_rows
            )
            ranking.append(
                {
                    "number": number,
                    "variant_vote_count": len(number_rows),
                    "confidence_sum": confidence_sum,
                }
            )
        supported_two_digits = [
            row for row in ranking if len(row["number"]) == 2 and row["variant_vote_count"] >= 2
        ]
        pool = supported_two_digits or ranking
        pool.sort(
            key=lambda row: (-row["variant_vote_count"], -row["confidence_sum"], row["number"])
        )
        winner = pool[0]["number"]
        winning_rows = by_number[winner]
        representative = dict(
            max(
                winning_rows,
                key=lambda row: (
                    float(row["confidence"]) if row.get("confidence") is not None else 0.0,
                    float(row.get("ocr_readiness_score") or 0.0),
                ),
            )
        )
        total_votes = sum(row["variant_vote_count"] for row in ranking)
        ranking.sort(
            key=lambda row: (-row["variant_vote_count"], -row["confidence_sum"], row["number"])
        )
        representative["variant_prediction_count"] = len(frame_rows)
        representative["frame_variant_candidate_votes"] = ranking
        representative["frame_variant_consensus"] = len(winning_rows) / max(total_votes, 1)
        collapsed.append(representative)
    return collapsed


def make_contact_sheet(
    records: list[dict],
    image_key: str,
    output_path: Path,
    thumb_width: int,
    thumb_height: int,
    title: str,
) -> str:
    label_height = 58
    columns = min(6, max(1, len(records)))
    rows = max(1, math.ceil(len(records) / columns))
    title_height = 28
    sheet = Image.new(
        "RGB", (columns * thumb_width, title_height + rows * (thumb_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), title, fill=(0, 0, 0))
    for index, record in enumerate(records):
        image = Image.open(record[image_key]).convert("RGB")
        image.thumbnail((thumb_width - 12, thumb_height - 12))
        x = (index % columns) * thumb_width
        y = title_height + (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y + 6))
        candidate = record.get("candidate_number") or "?"
        view = record.get("heuristic_likely_view", "unknown")
        draw.text(
            (x + 5, y + thumb_height + 4),
            "T{:03d} f{:03d} {} #{}".format(record["track_id"], record["frame_index"], view, candidate),
            fill=(0, 0, 0),
        )
        draw.text(
            (x + 5, y + thumb_height + 23),
            "{} ready {:.3f}".format(record["status"], record["ocr_readiness_score"]),
            fill=(70, 70, 70),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def per_track_evidence(predictions: list[dict], count: int = 3) -> list[dict]:
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["track_id"]].append(row)
    selected = []
    for track_id in sorted(grouped):
        rows = sorted(
            grouped[track_id],
            key=lambda row: (
                0 if row.get("candidate_number") else 1,
                -(row.get("confidence") or 0.0),
                -float(row["ocr_readiness_score"]),
            ),
        )
        selected.extend(rows[:count])
    return selected


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    clean_path = project_path(args.crop_metadata)
    audit_path = project_path(args.visibility_predictions)
    track_audit_path = project_path(args.track_visibility_summary)
    manifest_path = (
        project_path(args.number_region_manifest) if args.number_region_manifest else None
    )
    number_regions_input = (
        project_path(args.number_regions_dir) if args.number_regions_dir else None
    )
    if manifest_path is not None and number_regions_input is not None:
        raise ValueError("Use either --number-region-manifest or --number-regions-dir, not both")
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing enhanced number-region manifest: {manifest_path}")
        manifest_payload = read_json(manifest_path)
        combined = enhanced_manifest_records(manifest_payload, manifest_path)
        selected = select_manifest_regions(combined, args.top_k, args.all_crops)
        region_root = manifest_path.parent / "regions"
        track_audit = (
            read_json(track_audit_path) if track_audit_path.is_file() else {"tracks": []}
        )
        selection_method = "top_ocr_readiness_source_frames_with_all_enhanced_variants"
    else:
        if not track_audit_path.is_file():
            packaged_track_audit = audit_path.parent / "track_visibility_summary.json"
            if packaged_track_audit.is_file():
                track_audit_path = packaged_track_audit
        for path, label in (
            (clean_path, "clean crop metadata"),
            (audit_path, "crop visibility audit"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if number_regions_input is not None and not number_regions_input.is_dir():
            raise FileNotFoundError(f"Missing number regions directory: {number_regions_input}")
        clean_payload = read_json(clean_path)
        audit_payload = read_json(audit_path)
        track_audit = (
            read_json(track_audit_path) if track_audit_path.is_file() else {"tracks": []}
        )
        combined = combine_clean_and_audit(
            clean_payload,
            audit_payload,
            require_clean_crop_files=number_regions_input is None,
        )
        selected = select_top_crops(combined, args.top_k, args.all_crops)
        region_root = number_regions_input or (output_dir / "number_regions")
        selection_method = "top_ocr_readiness_unique_frames_per_track"

    inspection = inspect_ocr_engines()
    engine = choose_engine(args.engine, inspection)

    predictions = []
    for row in selected:
        region = (
            load_manifest_number_region(row)
            if manifest_path is not None
            else (
                load_number_region(row, region_root)
                if number_regions_input is not None
                else extract_number_region(row, region_root)
            )
        )
        predictions.append(
            predict_crop(row, region, engine, inspection, args.minimum_ocr_confidence)
        )
    tracks = aggregate_tracks(
        predictions,
        engine["available"],
        track_audit,
        args.minimum_two_digit_votes,
        args.minimum_one_digit_votes,
        args.minimum_consensus,
    )

    clean_crops_available = all(Path(row["crop_path"]).is_file() for row in predictions)
    review_image_key = (
        "number_region_path"
        if manifest_path is not None
        else ("crop_path" if clean_crops_available else "number_region_path")
    )
    contact_predictions = (
        per_track_evidence(predictions, count=8) if len(predictions) > 180 else predictions
    )
    ocr_sheet = make_contact_sheet(
        contact_predictions,
        review_image_key,
        output_dir / "jersey_ocr_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Top-K clean crops with OCR predictions",
    )
    region_sheet = make_contact_sheet(
        contact_predictions,
        "number_region_path",
        output_dir / "number_region_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Generated number regions",
    )
    evidence_sheet = make_contact_sheet(
        per_track_evidence(predictions),
        review_image_key,
        output_dir / "per_track_evidence_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Best OCR evidence per track",
    )

    status = "complete" if engine["available"] else "ocr_unavailable"
    crop_payload = {
        "status": status,
        "stage": "jersey_number_ocr_crop_predictions",
        "inputs": {
            "clean_crop_metadata": str(clean_path) if manifest_path is None else None,
            "visibility_audit": str(audit_path) if manifest_path is None else None,
            "track_visibility_summary": str(track_audit_path) if track_audit_path.is_file() else None,
            "number_regions_dir": str(number_regions_input) if number_regions_input else None,
            "number_region_manifest": str(manifest_path) if manifest_path else None,
        },
        "selection": {
            "method": selection_method,
            "top_k": args.top_k,
            "all_crops": bool(args.all_crops),
            "view_used_as_filter": False,
            "torso_used_as_tiebreak_only": True,
            "enhanced_variants_per_selected_source": (
                len({row.get("variant_name") for row in selected})
                if manifest_path is not None
                else None
            ),
            "contact_sheet_prediction_count": len(contact_predictions),
        },
        "ocr_engine": engine,
        "predictions": predictions,
    }
    track_payload = {
        "status": status,
        "stage": "jersey_number_ocr_track_predictions",
        "aggregation": {
            "method": "readiness_and_ocr_confidence_weighted_temporal_vote",
            "minimum_two_digit_votes": args.minimum_two_digit_votes,
            "minimum_one_digit_votes": args.minimum_one_digit_votes,
            "minimum_consensus": args.minimum_consensus,
            "matching_partial_digit_downweight": 0.25,
            "single_crop_acceptance_allowed": False,
        },
        "tracks": tracks,
    }
    summary = {
        "status": status,
        "stage": "jersey_number_ocr_baseline_summary",
        "identity_assignment_performed": False,
        "future_identity_path": "track -> team assignment -> jersey number -> roster lookup -> player name",
        "ocr_engine": engine,
        "ocr_engine_inspection": inspection,
        "counts": {
            "tracks_processed": len(tracks),
            "crops_processed": len(predictions),
            "readable_crops": sum(row["status"] == "readable" for row in predictions),
            "unreadable_crops": sum(row["status"] == "unreadable" for row in predictions),
            "uncertain_crops": sum(row["status"] == "uncertain" for row in predictions),
            "ocr_unavailable_crops": sum(row["status"] == "ocr_unavailable" for row in predictions),
            "tracks_with_any_candidate_number": sum(row["best_candidate_number"] is not None for row in tracks),
            "accepted_track_numbers": sum(row["accepted_number"] is not None for row in tracks),
            "low_confidence_tracks": sum(bool(row["low_confidence"]) for row in tracks),
            "number_regions_written": len(predictions),
            "number_regions_processed": len(predictions),
        },
        "artifacts": {
            "crop_ocr_predictions": str(output_dir / "crop_ocr_predictions.json"),
            "track_jersey_number_predictions": str(output_dir / "track_jersey_number_predictions.json"),
            "summary": str(output_dir / "jersey_ocr_summary.json"),
            "jersey_ocr_contact_sheet": ocr_sheet,
            "number_region_contact_sheet": region_sheet,
            "number_regions": str(region_root),
            "per_track_evidence_contact_sheet": evidence_sheet,
        },
        "known_limitations": [
            "Heuristic view direction is retained only as soft metadata and does not filter crops.",
            "One-digit readings require stronger repeated evidence because they may be partial two-digit numbers.",
            "Generic Tesseract OCR may fail on curved, occluded, stylized, or low-resolution jersey digits.",
            "No OCR package, model, or language data is installed or downloaded automatically.",
            "No team, roster, player-name, or identity assignment is performed.",
            "Portable mode consumes pre-extracted number regions and does not require original clean crop image paths.",
            "Enhanced-manifest mode treats each manifest image as an OCR input and does not regenerate regions.",
        ],
    }
    write_json(output_dir / "crop_ocr_predictions.json", crop_payload)
    write_json(output_dir / "track_jersey_number_predictions.json", track_payload)
    write_json(output_dir / "jersey_ocr_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
