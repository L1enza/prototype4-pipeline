#!/usr/bin/env python3
"""Audit clean player crops for likely jersey-number OCR usefulness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = "outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json"
DEFAULT_OUTPUT = "outputs/nll_test4/jersey_crop_visibility_audit"
MANUAL_REVIEW_FIELDS = [
    "track_id",
    "frame_index",
    "crop_path",
    "crop_type",
    "likely_view",
    "auto_quality_score",
    "ocr_candidate",
    "manual_number_visible",
    "manual_number_text",
    "manual_view",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank clean player crops for manual jersey OCR review.")
    parser.add_argument("--clean-crop-metadata", default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-threshold", type=float, default=0.78)
    parser.add_argument("--borderline-threshold", type=float, default=0.68)
    parser.add_argument("--minimum-sharpness", type=float, default=0.24)
    parser.add_argument("--minimum-contrast", type=float, default=0.12)
    parser.add_argument("--maximum-bbox-overlap", type=float, default=0.65)
    parser.add_argument("--top-per-track", type=int, default=3)
    parser.add_argument("--contact-thumb-width", type=int, default=180)
    parser.add_argument("--contact-thumb-height", type=int, default=205)
    parser.add_argument("--contact-sheet-limit", type=int, default=96)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clipped(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def load_face_detectors() -> dict:
    haar_root = Path(cv2.data.haarcascades)
    frontal_path = haar_root / "haarcascade_frontalface_default.xml"
    profile_path = haar_root / "haarcascade_profileface.xml"
    frontal = cv2.CascadeClassifier(str(frontal_path))
    profile = cv2.CascadeClassifier(str(profile_path))
    return {
        "frontal": frontal if not frontal.empty() else None,
        "profile": profile if not profile.empty() else None,
        "frontal_path": str(frontal_path),
        "profile_path": str(profile_path),
    }


def detect_faces(head_bgr: np.ndarray, detectors: dict) -> dict:
    gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = max(1.0, min(4.0, 240.0 / max(min(height, width), 1)))
    enlarged_gray = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )
    enlarged_bgr = cv2.resize(
        head_bgr,
        (enlarged_gray.shape[1], enlarged_gray.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    min_side = max(18, int(min(enlarged_gray.shape[:2]) * 0.10))
    frontal_boxes = []
    profile_boxes = []
    if detectors["frontal"] is not None:
        frontal_boxes = list(
            detectors["frontal"].detectMultiScale(
                enlarged_gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_side, min_side)
            )
        )
    if detectors["profile"] is not None:
        profile_boxes.extend(
            detectors["profile"].detectMultiScale(
                enlarged_gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_side, min_side)
            )
        )
        flipped_gray = cv2.flip(enlarged_gray, 1)
        flipped_boxes = detectors["profile"].detectMultiScale(
            flipped_gray, scaleFactor=1.08, minNeighbors=4, minSize=(min_side, min_side)
        )
        for x, y, box_width, box_height in flipped_boxes:
            profile_boxes.append((enlarged_gray.shape[1] - x - box_width, y, box_width, box_height))

    def skin_ratio(box) -> float:
        x, y, box_width, box_height = [int(value) for value in box]
        region = enlarged_bgr[y : y + box_height, x : x + box_width]
        if region.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 180, 135], dtype=np.uint8),
        )
        return float(np.count_nonzero(skin) / max(skin.size, 1))

    frontal_skin = [skin_ratio(box) for box in frontal_boxes]
    profile_skin = [skin_ratio(box) for box in profile_boxes]
    minimum_skin_ratio = 0.06
    return {
        "frontal_face_count": sum(value >= minimum_skin_ratio for value in frontal_skin),
        "profile_face_count": sum(value >= minimum_skin_ratio for value in profile_skin),
        "raw_frontal_detection_count": len(frontal_boxes),
        "raw_profile_detection_count": len(profile_boxes),
        "maximum_frontal_skin_ratio": max(frontal_skin, default=0.0),
        "maximum_profile_skin_ratio": max(profile_skin, default=0.0),
        "minimum_skin_ratio": minimum_skin_ratio,
    }


def number_region(image_bgr: np.ndarray, crop_type: str) -> tuple[np.ndarray, dict]:
    height, width = image_bgr.shape[:2]
    if crop_type == "torso":
        x0, x1 = int(width * 0.08), int(width * 0.92)
        y0, y1 = int(height * 0.05), int(height * 0.95)
    else:
        x0, x1 = int(width * 0.12), int(width * 0.88)
        y0, y1 = int(height * 0.08), int(height * 0.66)
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return image_bgr[y0:y1, x0:x1], {"x0": x0, "y0": y0, "x1": x1 - 1, "y1": y1 - 1}


def number_region_metrics(region_bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    contrast = clipped(float(gray.std()) / 64.0)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    digit_like = []
    for contour in contours:
        x, y, component_width, component_height = cv2.boundingRect(contour)
        width_ratio = component_width / max(width, 1)
        height_ratio = component_height / max(height, 1)
        area_ratio = component_width * component_height / max(width * height, 1)
        aspect = component_width / max(component_height, 1)
        center_x = (x + component_width / 2.0) / max(width, 1)
        if (
            0.03 <= width_ratio <= 0.55
            and 0.15 <= height_ratio <= 0.88
            and 0.004 <= area_ratio <= 0.30
            and 0.10 <= aspect <= 1.45
            and 0.12 <= center_x <= 0.88
        ):
            digit_like.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(component_width),
                    "height": int(component_height),
                }
            )
    component_score = clipped(len(digit_like) / 2.0)
    edge_score = clipped(edge_density / 0.16)
    signal = clipped(0.42 * contrast + 0.36 * edge_score + 0.22 * component_score)
    return {
        "contrast_score": contrast,
        "edge_density": edge_density,
        "edge_score": edge_score,
        "digit_like_component_count": len(digit_like),
        "digit_like_components": digit_like,
        "number_visibility_signal": signal,
    }


def estimate_view(
    face_metrics: dict,
    number_metrics: dict,
    original_bbox: dict,
) -> tuple[str, str, str, list[str]]:
    evidence = []
    bbox_width = max(1.0, float(original_bbox.get("x1", 0) - original_bbox.get("x0", 0) + 1))
    bbox_height = max(1.0, float(original_bbox.get("y1", 0) - original_bbox.get("y0", 0) + 1))
    bbox_aspect = bbox_width / bbox_height
    if face_metrics["frontal_face_count"] > 0:
        evidence.append("opencv_frontal_face_detection")
        return "front", "chest", "medium", evidence
    if face_metrics["profile_face_count"] > 0:
        evidence.append("opencv_profile_face_detection")
        return "side", "unknown", "medium", evidence
    if bbox_aspect < 0.27 and number_metrics["number_visibility_signal"] < 0.62:
        evidence.append("very_narrow_person_bbox")
        return "side", "unknown", "weak", evidence
    component_count = number_metrics["digit_like_component_count"]
    if (
        number_metrics["number_visibility_signal"] >= 0.90
        and 2 <= component_count <= 4
        and number_metrics["contrast_score"] >= 0.60
        and bbox_aspect >= 0.33
    ):
        evidence.append("large_centered_number_like_signal_without_face_detection")
        return "back", "back", "weak", evidence
    evidence.append("insufficient_orientation_evidence")
    return "unknown", "unknown", "none", evidence


def audit_crop(crop: dict, detectors: dict, args: argparse.Namespace) -> dict:
    crop_path = Path(crop["crop_path"])
    image_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {
            "track_id": int(crop["track_id"]),
            "frame_index": int(crop["frame_index"]),
            "crop_path": str(crop_path),
            "crop_type": crop.get("crop_type"),
            "ocr_candidate": False,
            "audit_status": "rejected",
            "rejection_reason": "crop_file_unreadable",
            "rejection_reasons": ["crop_file_unreadable"],
            "likely_view": "unknown",
            "number_region_guess": "unknown",
            "auto_quality_score": 0.0,
        }

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = clipped(laplacian_variance / (laplacian_variance + 140.0))
    contrast_score = clipped(float(gray.std()) / 64.0)
    region, region_box = number_region(image_bgr, crop.get("crop_type", "unknown"))
    number_metrics = number_region_metrics(region)
    head_fraction = 0.38 if crop.get("crop_type") == "torso" else 0.30
    head_height = max(1, int(round(height * head_fraction)))
    face_metrics = detect_faces(image_bgr[:head_height, :], detectors)
    face_metrics["head_region_fraction"] = head_fraction
    likely_view, number_region_guess, view_confidence, view_evidence = estimate_view(
        face_metrics, number_metrics, crop.get("bbox") or {}
    )

    source_quality = clipped(float(crop.get("crop_quality_score") or 0.0))
    overlap = clipped(float((crop.get("quality") or {}).get("bbox_overlap_fraction") or 0.0))
    border_clipped = bool((crop.get("quality") or {}).get("border_clipped"))
    if crop.get("crop_type") == "torso":
        resolution_score = clipped(min(width / 120.0, height / 100.0))
        minimum_width, minimum_height = 36, 32
    else:
        resolution_score = clipped(min(width / 120.0, height / 160.0))
        minimum_width, minimum_height = 45, 70
    non_occlusion_score = 1.0 - overlap
    audit_score = (
        0.22 * source_quality
        + 0.20 * sharpness_score
        + 0.15 * contrast_score
        + 0.14 * resolution_score
        + 0.21 * number_metrics["number_visibility_signal"]
        + 0.08 * non_occlusion_score
    )
    if likely_view == "side":
        audit_score -= 0.07
    audit_score = clipped(audit_score)

    hard_reasons = []
    soft_reasons = []
    if width < minimum_width or height < minimum_height:
        hard_reasons.append("low_resolution")
    if sharpness_score < args.minimum_sharpness:
        hard_reasons.append("blurry")
    if contrast_score < args.minimum_contrast:
        hard_reasons.append("low_contrast")
    if overlap > args.maximum_bbox_overlap:
        hard_reasons.append("high_bbox_overlap_occlusion_proxy")
    if border_clipped:
        soft_reasons.append("border_clipped")
    if number_metrics["number_visibility_signal"] < 0.38:
        soft_reasons.append("weak_number_region_signal")
    if likely_view == "side":
        soft_reasons.append("side_view_lower_priority")

    ocr_candidate = not hard_reasons and audit_score >= args.candidate_threshold
    if ocr_candidate:
        audit_status = "ocr_ready"
        rejection_reasons = []
    elif not hard_reasons and audit_score >= args.borderline_threshold:
        audit_status = "borderline"
        rejection_reasons = soft_reasons or ["below_candidate_threshold"]
    else:
        audit_status = "rejected"
        rejection_reasons = hard_reasons + soft_reasons
        if not rejection_reasons:
            rejection_reasons = ["below_borderline_threshold"]

    bbox = crop.get("bbox") or {}
    bbox_width = max(0, int(bbox.get("x1", 0) - bbox.get("x0", 0) + 1))
    bbox_height = max(0, int(bbox.get("y1", 0) - bbox.get("y0", 0) + 1))
    return {
        "track_id": int(crop["track_id"]),
        "frame_index": int(crop["frame_index"]),
        "source_frame_index": crop.get("source_frame_index"),
        "timestamp_seconds": crop.get("timestamp_seconds"),
        "crop_path": str(crop_path),
        "crop_type": crop.get("crop_type"),
        "crop_width": int(width),
        "crop_height": int(height),
        "sharpness_score": sharpness_score,
        "laplacian_variance": laplacian_variance,
        "contrast_score": contrast_score,
        "bbox": bbox,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_size": bbox_width * bbox_height,
        "source_crop_quality_score": source_quality,
        "quality_score": audit_score,
        "auto_quality_score": audit_score,
        "resolution_score": resolution_score,
        "bbox_overlap_fraction": overlap,
        "likely_view": likely_view,
        "view_confidence": view_confidence,
        "view_evidence": view_evidence,
        "number_region_guess": number_region_guess,
        "number_region_box": region_box,
        "number_region_metrics": number_metrics,
        "face_detection": face_metrics,
        "ocr_candidate": ocr_candidate,
        "audit_status": audit_status,
        "rejection_reason": ";".join(rejection_reasons) if rejection_reasons else None,
        "rejection_reasons": rejection_reasons,
    }


def ranked(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda row: (
            -float(row.get("auto_quality_score") or 0.0),
            0 if row.get("crop_type") == "torso" else 1,
            int(row.get("frame_index") or 0),
        ),
    )


def top_per_track(records: list[dict], count: int) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["track_id"]].append(record)
    selected = []
    for track_id in sorted(grouped):
        selected.extend(ranked(grouped[track_id])[:count])
    return selected


def make_contact_sheet(
    records: list[dict],
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
        "RGB",
        (columns * thumb_width, title_height + rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), title, fill=(0, 0, 0))
    if not records:
        draw.text((8, title_height + 10), "No matching crops", fill=(80, 80, 80))
    for index, record in enumerate(records):
        image = Image.open(record["crop_path"]).convert("RGB")
        image.thumbnail((thumb_width - 12, thumb_height - 12))
        x = (index % columns) * thumb_width
        y = title_height + (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y + 6))
        label = "T{:03d} f{:03d} {} {}".format(
            record["track_id"], record["frame_index"], record["likely_view"], record["crop_type"]
        )
        draw.text((x + 5, y + thumb_height + 4), label, fill=(0, 0, 0))
        draw.text(
            (x + 5, y + thumb_height + 23),
            "{} q{:.3f}".format(record["audit_status"], record["auto_quality_score"]),
            fill=(70, 70, 70),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def build_track_summaries(records: list[dict], top_count: int) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["track_id"]].append(record)
    summaries = []
    for track_id in sorted(grouped):
        rows = grouped[track_id]
        candidates = ranked([row for row in rows if row["ocr_candidate"]])
        front = ranked([row for row in candidates if row["likely_view"] == "front"])
        back = ranked([row for row in candidates if row["likely_view"] == "back"])
        borderline = [row for row in rows if row["audit_status"] == "borderline"]
        if len(candidates) >= 3:
            readiness = "ready"
        elif candidates or len(borderline) >= 2:
            readiness = "borderline"
        else:
            readiness = "not_ready"
        summaries.append(
            {
                "track_id": int(track_id),
                "total_crops": len(rows),
                "candidate_crop_count": len(candidates),
                "front_candidate_count": len(front),
                "back_candidate_count": len(back),
                "borderline_crop_count": len(borderline),
                "best_front_candidate_paths": [row["crop_path"] for row in front[:top_count]],
                "best_back_candidate_paths": [row["crop_path"] for row in back[:top_count]],
                "best_overall_ocr_candidate_paths": [row["crop_path"] for row in candidates[:top_count]],
                "readiness": readiness,
            }
        )
    return summaries


def write_manual_review_csv(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_REVIEW_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row["track_id"], row["frame_index"], row["crop_type"])):
            writer.writerow(
                {
                    "track_id": record["track_id"],
                    "frame_index": record["frame_index"],
                    "crop_path": record["crop_path"],
                    "crop_type": record["crop_type"],
                    "likely_view": record["likely_view"],
                    "auto_quality_score": "{:.6f}".format(record["auto_quality_score"]),
                    "ocr_candidate": str(bool(record["ocr_candidate"])).lower(),
                    "manual_number_visible": "",
                    "manual_number_text": "",
                    "manual_view": "",
                    "notes": "",
                }
            )
    return str(path)


def main() -> int:
    args = parse_args()
    if args.top_per_track < 1:
        raise ValueError("--top-per-track must be at least 1")
    if not 0.0 <= args.borderline_threshold <= args.candidate_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= borderline <= candidate <= 1")
    metadata_path = project_path(args.clean_crop_metadata)
    output_dir = project_path(args.output_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError("Clean crop metadata not found: {}".format(metadata_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_metadata = read_json(metadata_path)
    detectors = load_face_detectors()
    records = [audit_crop(crop, detectors, args) for crop in clean_metadata.get("crops", [])]
    track_summaries = build_track_summaries(records, args.top_per_track)
    candidates = [record for record in records if record["ocr_candidate"]]
    front_candidates = [record for record in candidates if record["likely_view"] == "front"]
    back_candidates = [record for record in candidates if record["likely_view"] == "back"]
    borderline_or_rejected = [record for record in records if not record["ocr_candidate"]]

    top_three = top_per_track(candidates, args.top_per_track)
    best_sheet = make_contact_sheet(
        top_three,
        output_dir / "best_ocr_ready_crops_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Top OCR-ready crops per track",
    )
    compact_sheet_path = output_dir / "top3_ocr_candidates_per_track_contact_sheet.png"
    shutil.copyfile(best_sheet, compact_sheet_path)
    front_sheet = make_contact_sheet(
        top_per_track(front_candidates, args.top_per_track),
        output_dir / "best_front_candidates_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Best front-facing OCR candidates",
    )
    back_sheet = make_contact_sheet(
        top_per_track(back_candidates, args.top_per_track),
        output_dir / "best_back_candidates_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Best back-facing OCR candidates",
    )
    rejected_sheet_records = top_per_track(ranked(borderline_or_rejected), 2)[: args.contact_sheet_limit]
    rejected_sheet = make_contact_sheet(
        rejected_sheet_records,
        output_dir / "rejected_or_borderline_crops_contact_sheet.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Rejected or borderline crops",
    )
    review_csv = write_manual_review_csv(output_dir / "manual_review_sheet.csv", records)

    predictions_payload = {
        "status": "complete",
        "stage": "jersey_crop_visibility_predictions",
        "source_clean_crop_metadata": str(metadata_path),
        "heuristic_version": "jersey_crop_visibility_v1",
        "view_label_policy": {
            "front": "positive OpenCV frontal-face evidence",
            "side": "positive profile-face evidence or weak narrow-bbox heuristic",
            "back": "weak centered number-like signal without face evidence",
            "unknown": "used whenever orientation evidence is insufficient",
            "view_labels_are_not_identity_evidence": True,
        },
        "crops": records,
    }
    track_payload = {
        "status": "complete",
        "stage": "jersey_crop_track_visibility_summary",
        "tracks": track_summaries,
    }
    summary = {
        "status": "complete",
        "stage": "jersey_crop_visibility_audit_summary",
        "identity_assignment_performed": False,
        "ocr_performed": False,
        "inputs": {"clean_crop_metadata": str(metadata_path)},
        "thresholds": {
            "candidate": args.candidate_threshold,
            "borderline": args.borderline_threshold,
            "minimum_sharpness": args.minimum_sharpness,
            "minimum_contrast": args.minimum_contrast,
            "maximum_bbox_overlap": args.maximum_bbox_overlap,
        },
        "counts": {
            "tracks_audited": len(track_summaries),
            "total_crops_audited": len(records),
            "ocr_ready_crops": len(candidates),
            "front_candidates": len(front_candidates),
            "back_candidates": len(back_candidates),
            "side_candidates": sum(record["ocr_candidate"] and record["likely_view"] == "side" for record in records),
            "unknown_view_candidates": sum(record["ocr_candidate"] and record["likely_view"] == "unknown" for record in records),
            "borderline_crops": sum(record["audit_status"] == "borderline" for record in records),
            "rejected_crops": sum(record["audit_status"] == "rejected" for record in records),
            "ready_tracks": sum(track["readiness"] == "ready" for track in track_summaries),
            "borderline_tracks": sum(track["readiness"] == "borderline" for track in track_summaries),
            "not_ready_tracks": sum(track["readiness"] == "not_ready" for track in track_summaries),
        },
        "artifacts": {
            "crop_visibility_predictions": str(output_dir / "crop_visibility_predictions.json"),
            "track_visibility_summary": str(output_dir / "track_visibility_summary.json"),
            "best_ocr_ready_crops_contact_sheet": best_sheet,
            "top3_ocr_candidates_per_track_contact_sheet": str(compact_sheet_path),
            "best_front_candidates_contact_sheet": front_sheet,
            "best_back_candidates_contact_sheet": back_sheet,
            "rejected_or_borderline_crops_contact_sheet": rejected_sheet,
            "manual_review_sheet": review_csv,
            "summary": str(output_dir / "jersey_crop_visibility_summary.json"),
        },
        "known_limitations": [
            "View direction is heuristic; helmets, low resolution, and occlusion make face detection unreliable.",
            "A back label is weak evidence from number-like image structure, not pose estimation.",
            "Number visibility is estimated from contrast, edges, and component geometry without OCR.",
            "Tracking identity switches can place different players in one track folder.",
            "Manual review remains required before OCR evaluation.",
        ],
    }
    write_json(output_dir / "crop_visibility_predictions.json", predictions_payload)
    write_json(output_dir / "track_visibility_summary.json", track_payload)
    write_json(output_dir / "jersey_crop_visibility_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
