#!/usr/bin/env python3
"""Extract clean full-body and torso crops from the original video for jersey OCR."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/nll_test4.mp4"
DEFAULT_TRACKING = (
    "outputs/nll_test4/calibrated_segment_demos/"
    "segment_20s_10s_calibrated/tracking_metadata.json"
)
DEFAULT_OUTPUT = "outputs/nll_test4/jersey_ocr_clean_crops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clean, annotation-free player crops from original video frames."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--tracking-metadata", default=DEFAULT_TRACKING)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--best-n", type=int, default=8, help="Maximum saved crops per track and crop type.")
    parser.add_argument("--full-padding-x", type=float, default=0.14)
    parser.add_argument("--full-padding-y", type=float, default=0.08)
    parser.add_argument("--torso-padding-x", type=float, default=0.18)
    parser.add_argument("--torso-y-start", type=float, default=0.05)
    parser.add_argument("--torso-y-end", type=float, default=0.66)
    parser.add_argument("--contact-thumb-width", type=int, default=190)
    parser.add_argument("--contact-thumb-height", type=int, default=230)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_bbox(bbox: dict, frame_width: int, frame_height: int) -> dict | None:
    if not bbox:
        return None
    x0 = max(0, min(frame_width - 1, int(math.floor(float(bbox["x0"])))))
    y0 = max(0, min(frame_height - 1, int(math.floor(float(bbox["y0"])))))
    x1 = max(0, min(frame_width - 1, int(math.ceil(float(bbox["x1"])))))
    y1 = max(0, min(frame_height - 1, int(math.ceil(float(bbox["y1"])))))
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def padded_box(bbox: dict, frame_width: int, frame_height: int, pad_x: float, pad_y: float) -> tuple[dict, bool]:
    width = bbox["x1"] - bbox["x0"] + 1
    height = bbox["y1"] - bbox["y0"] + 1
    raw = {
        "x0": math.floor(bbox["x0"] - width * pad_x),
        "y0": math.floor(bbox["y0"] - height * pad_y),
        "x1": math.ceil(bbox["x1"] + width * pad_x),
        "y1": math.ceil(bbox["y1"] + height * pad_y),
    }
    clipped = raw["x0"] < 0 or raw["y0"] < 0 or raw["x1"] >= frame_width or raw["y1"] >= frame_height
    box = {
        "x0": max(0, int(raw["x0"])),
        "y0": max(0, int(raw["y0"])),
        "x1": min(frame_width - 1, int(raw["x1"])),
        "y1": min(frame_height - 1, int(raw["y1"])),
    }
    return box, clipped


def torso_box(
    bbox: dict,
    frame_width: int,
    frame_height: int,
    pad_x: float,
    y_start: float,
    y_end: float,
) -> tuple[dict, bool]:
    width = bbox["x1"] - bbox["x0"] + 1
    height = bbox["y1"] - bbox["y0"] + 1
    raw = {
        "x0": math.floor(bbox["x0"] - width * pad_x),
        "y0": math.floor(bbox["y0"] + height * y_start),
        "x1": math.ceil(bbox["x1"] + width * pad_x),
        "y1": math.ceil(bbox["y0"] + height * y_end),
    }
    clipped = raw["x0"] < 0 or raw["y0"] < 0 or raw["x1"] >= frame_width or raw["y1"] >= frame_height
    box = {
        "x0": max(0, int(raw["x0"])),
        "y0": max(0, int(raw["y0"])),
        "x1": min(frame_width - 1, int(raw["x1"])),
        "y1": min(frame_height - 1, int(raw["y1"])),
    }
    return box, clipped


def crop_image(frame_bgr: np.ndarray, box: dict) -> np.ndarray | None:
    crop = frame_bgr[box["y0"] : box["y1"] + 1, box["x0"] : box["x1"] + 1]
    return crop.copy() if crop.size else None


def intersection_fraction(bbox: dict, other: dict) -> float:
    x0 = max(bbox["x0"], other["x0"])
    y0 = max(bbox["y0"], other["y0"])
    x1 = min(bbox["x1"], other["x1"])
    y1 = min(bbox["y1"], other["y1"])
    if x1 < x0 or y1 < y0:
        return 0.0
    intersection = float((x1 - x0 + 1) * (y1 - y0 + 1))
    area = float((bbox["x1"] - bbox["x0"] + 1) * (bbox["y1"] - bbox["y0"] + 1))
    return intersection / max(area, 1.0)


def max_overlap_fraction(detection: dict, frame_detections: list[dict], frame_width: int, frame_height: int) -> float:
    bbox = normalize_bbox(detection.get("bbox_2d"), frame_width, frame_height)
    if bbox is None:
        return 1.0
    overlaps = []
    for other in frame_detections:
        if other is detection:
            continue
        other_bbox = normalize_bbox(other.get("bbox_2d"), frame_width, frame_height)
        if other_bbox is not None:
            overlaps.append(intersection_fraction(bbox, other_bbox))
    return max(overlaps, default=0.0)


def quality_metrics(
    crop_bgr: np.ndarray,
    bbox: dict,
    frame_width: int,
    frame_height: int,
    overlap_fraction: float,
    border_clipped: bool,
    crop_type: str,
) -> dict:
    height, width = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))

    bbox_height = bbox["y1"] - bbox["y0"] + 1
    bbox_width = bbox["x1"] - bbox["x0"] + 1
    bbox_scale_score = min(1.0, bbox_height / max(frame_height * 0.28, 1.0))
    if crop_type == "torso":
        resolution_score = min(1.0, min(width / 120.0, height / 100.0))
    else:
        resolution_score = min(1.0, min(width / 160.0, height / 220.0))
    sharpness_score = laplacian_variance / (laplacian_variance + 140.0)
    low_blur_score = tenengrad / (tenengrad + 1800.0)
    occlusion_score = 1.0 - min(1.0, overlap_fraction)
    border_score = 0.0 if border_clipped else 1.0
    quality_score = (
        0.25 * bbox_scale_score
        + 0.24 * sharpness_score
        + 0.18 * low_blur_score
        + 0.18 * resolution_score
        + 0.10 * occlusion_score
        + 0.05 * border_score
    )
    return {
        "crop_width": int(width),
        "crop_height": int(height),
        "crop_pixels": int(width * height),
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "laplacian_variance": laplacian_variance,
        "tenengrad": tenengrad,
        "bbox_scale_score": bbox_scale_score,
        "sharpness_score": sharpness_score,
        "low_motion_blur_score": low_blur_score,
        "motion_blur_estimate": 1.0 - low_blur_score,
        "resolution_score": resolution_score,
        "bbox_overlap_fraction": overlap_fraction,
        "occlusion_score": occlusion_score,
        "occlusion_method": "maximum_bbox_intersection_fraction_proxy",
        "border_clipped": border_clipped,
        "crop_quality_score": quality_score,
    }


def make_candidate(
    crop_bgr: np.ndarray,
    crop_type: str,
    crop_box_value: dict,
    bbox: dict,
    detection: dict,
    decode_record: dict,
    frame_width: int,
    frame_height: int,
    overlap_fraction: float,
    border_clipped: bool,
) -> dict:
    metrics = quality_metrics(
        crop_bgr,
        bbox,
        frame_width,
        frame_height,
        overlap_fraction,
        border_clipped,
        crop_type,
    )
    return {
        "track_id": int(detection["track_id"]),
        "frame_index": int(detection["frame_index"]),
        "source_frame_index": int(decode_record["source_frame_index"]),
        "timestamp_seconds": decode_record.get("timestamp_seconds"),
        "mask_id": detection.get("mask_id"),
        "bbox": bbox,
        "crop_box": crop_box_value,
        "crop_type": crop_type,
        "sam_confidence_score": detection.get("sam_confidence_score"),
        "crop_quality_score": metrics["crop_quality_score"],
        "quality": metrics,
        "crop_path": None,
        "source": "decoded_directly_from_original_video",
        "contains_overlay": False,
        "_image": crop_bgr,
    }


def save_ranked_candidates(candidates_by_track: dict, output_dir: Path, best_n: int) -> tuple[list[dict], list[dict]]:
    saved = []
    track_summaries = []
    for track_id in sorted(candidates_by_track):
        type_counts = {}
        saved_by_type = {}
        for crop_type in ("full_body", "torso"):
            candidates = candidates_by_track[track_id].get(crop_type, [])
            candidates.sort(key=lambda row: row["crop_quality_score"], reverse=True)
            selected = candidates[:best_n]
            crop_dir = output_dir / "track_{:03d}".format(track_id) / crop_type
            crop_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for rank, candidate in enumerate(selected, start=1):
                filename = "track_{:03d}_{}_rank_{:02d}_frame_{:03d}_source_{:06d}.png".format(
                    track_id,
                    crop_type,
                    rank,
                    candidate["frame_index"],
                    candidate["source_frame_index"],
                )
                path = crop_dir / filename
                if not cv2.imwrite(str(path), candidate["_image"]):
                    raise RuntimeError("Failed to write crop: {}".format(path))
                record = {key: value for key, value in candidate.items() if key != "_image"}
                record["rank_within_track_and_type"] = rank
                record["crop_path"] = str(path)
                records.append(record)
                saved.append(record)
            type_counts[crop_type] = {"candidates": len(candidates), "saved": len(records)}
            saved_by_type[crop_type] = records
        track_summaries.append(
            {
                "track_id": int(track_id),
                "counts": type_counts,
                "best_full_body_crop": saved_by_type["full_body"][0]["crop_path"] if saved_by_type["full_body"] else None,
                "best_torso_crop": saved_by_type["torso"][0]["crop_path"] if saved_by_type["torso"] else None,
            }
        )
    return saved, track_summaries


def make_contact_sheet(records: list[dict], output_path: Path, thumb_width: int, thumb_height: int, title: str) -> str:
    label_height = 42
    columns = min(6, max(1, len(records)))
    rows = max(1, math.ceil(len(records) / columns))
    title_height = 28
    sheet = Image.new("RGB", (columns * thumb_width, title_height + rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), title, fill=(0, 0, 0))
    for index, record in enumerate(records):
        image = Image.open(record["crop_path"]).convert("RGB")
        image.thumbnail((thumb_width - 12, thumb_height - 12))
        x = (index % columns) * thumb_width
        y = title_height + (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y + 6))
        label = "T{:03d} f{:03d} q{:.3f}".format(
            record["track_id"], record["frame_index"], record["crop_quality_score"]
        )
        draw.text((x + 6, y + thumb_height + 6), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def make_track_strips(saved_records: list[dict], output_dir: Path, thumb_width: int, thumb_height: int) -> list[str]:
    grouped = defaultdict(lambda: defaultdict(list))
    for record in saved_records:
        grouped[record["track_id"]][record["crop_type"]].append(record)
    paths = []
    for track_id in sorted(grouped):
        for crop_type in ("full_body", "torso"):
            records = sorted(grouped[track_id][crop_type], key=lambda row: row["rank_within_track_and_type"])
            if not records:
                continue
            path = output_dir / "track_{:03d}".format(track_id) / "track_{:03d}_{}_strip.png".format(track_id, crop_type)
            paths.append(make_contact_sheet(records, path, thumb_width, thumb_height, "Track {} {} crops".format(track_id, crop_type)))
    return paths


def main() -> int:
    args = parse_args()
    if args.best_n < 1:
        raise ValueError("--best-n must be at least 1")
    if not 0.0 <= args.torso_y_start < args.torso_y_end <= 1.0:
        raise ValueError("Torso y fractions must satisfy 0 <= start < end <= 1")

    video_path = project_path(args.video)
    tracking_path = project_path(args.tracking_metadata)
    output_dir = project_path(args.output_dir)
    if not video_path.is_file():
        raise FileNotFoundError("Video not found: {}".format(video_path))
    if not tracking_path.is_file():
        raise FileNotFoundError("Tracking metadata not found: {}".format(tracking_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = read_json(tracking_path)
    decode_records = {
        int(row["clip_frame_index"]): row for row in tracking.get("decode", {}).get("decoded_frames", [])
    }
    detections_by_frame = defaultdict(list)
    for detection in tracking.get("detections", []):
        detections_by_frame[int(detection["frame_index"])].append(detection)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))
    video_info = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    candidates_by_track = defaultdict(lambda: defaultdict(list))
    decoded_original_frames = 0
    skipped_detections = []
    try:
        for frame_index in sorted(detections_by_frame):
            decode_record = decode_records.get(frame_index)
            if decode_record is None:
                skipped_detections.extend(
                    {"frame_index": frame_index, "track_id": row.get("track_id"), "reason": "missing_decode_mapping"}
                    for row in detections_by_frame[frame_index]
                )
                continue
            source_frame_index = int(decode_record["source_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                skipped_detections.extend(
                    {"frame_index": frame_index, "track_id": row.get("track_id"), "reason": "source_frame_decode_failed"}
                    for row in detections_by_frame[frame_index]
                )
                continue
            decoded_original_frames += 1
            frame_height, frame_width = frame_bgr.shape[:2]
            frame_detections = detections_by_frame[frame_index]
            for detection in frame_detections:
                track_id = detection.get("track_id")
                bbox = normalize_bbox(detection.get("bbox_2d"), frame_width, frame_height)
                if track_id is None or bbox is None:
                    skipped_detections.append(
                        {"frame_index": frame_index, "track_id": track_id, "reason": "missing_track_or_invalid_bbox"}
                    )
                    continue
                overlap = max_overlap_fraction(detection, frame_detections, frame_width, frame_height)
                full_box, full_clipped = padded_box(
                    bbox, frame_width, frame_height, args.full_padding_x, args.full_padding_y
                )
                torso_box_value, torso_clipped = torso_box(
                    bbox,
                    frame_width,
                    frame_height,
                    args.torso_padding_x,
                    args.torso_y_start,
                    args.torso_y_end,
                )
                for crop_type, crop_box_value, clipped in (
                    ("full_body", full_box, full_clipped),
                    ("torso", torso_box_value, torso_clipped),
                ):
                    crop_bgr = crop_image(frame_bgr, crop_box_value)
                    if crop_bgr is None or min(crop_bgr.shape[:2]) < 4:
                        skipped_detections.append(
                            {"frame_index": frame_index, "track_id": track_id, "crop_type": crop_type, "reason": "empty_crop"}
                        )
                        continue
                    candidate = make_candidate(
                        crop_bgr,
                        crop_type,
                        crop_box_value,
                        bbox,
                        detection,
                        decode_record,
                        frame_width,
                        frame_height,
                        overlap,
                        clipped,
                    )
                    candidates_by_track[int(track_id)][crop_type].append(candidate)
    finally:
        capture.release()

    saved_records, track_summaries = save_ranked_candidates(candidates_by_track, output_dir, args.best_n)
    full_records = [row for row in saved_records if row["crop_type"] == "full_body"]
    torso_records = [row for row in saved_records if row["crop_type"] == "torso"]
    best_full = [min((row for row in full_records if row["track_id"] == track_id), key=lambda row: row["rank_within_track_and_type"]) for track_id in sorted({row["track_id"] for row in full_records})]
    best_torso = [min((row for row in torso_records if row["track_id"] == track_id), key=lambda row: row["rank_within_track_and_type"]) for track_id in sorted({row["track_id"] for row in torso_records})]

    full_sheet = make_contact_sheet(
        best_full,
        output_dir / "all_tracks_best_clean_crops.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Best clean full-body crop per track",
    )
    torso_sheet = make_contact_sheet(
        best_torso,
        output_dir / "all_tracks_best_torso_crops.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        "Best clean torso crop per track",
    )
    track_strips = make_track_strips(
        saved_records, output_dir, args.contact_thumb_width, args.contact_thumb_height
    )

    candidate_full_count = sum(len(types.get("full_body", [])) for types in candidates_by_track.values())
    candidate_torso_count = sum(len(types.get("torso", [])) for types in candidates_by_track.values())
    metadata = {
        "status": "complete",
        "stage": "clean_track_crops_for_jersey_ocr",
        "inputs": {"video": str(video_path), "tracking_metadata": str(tracking_path)},
        "video": video_info,
        "parameters": {
            "best_n": args.best_n,
            "full_padding_x": args.full_padding_x,
            "full_padding_y": args.full_padding_y,
            "torso_padding_x": args.torso_padding_x,
            "torso_y_start": args.torso_y_start,
            "torso_y_end": args.torso_y_end,
        },
        "source_integrity": {
            "decoded_directly_from_original_video": True,
            "overlay_frames_used": False,
            "sam_masks_applied": False,
            "annotations_drawn_on_saved_crops": False,
            "output_format": "PNG",
        },
        "quality_method": {
            "weights": {
                "bbox_scale": 0.25,
                "sharpness": 0.24,
                "low_motion_blur": 0.18,
                "resolution": 0.18,
                "low_bbox_overlap": 0.10,
                "not_border_clipped": 0.05,
            },
            "occlusion_is_proxy_only": True,
        },
        "tracks": track_summaries,
        "crops": saved_records,
        "skipped": skipped_detections,
        "artifacts": {
            "all_tracks_best_clean_crops": full_sheet,
            "all_tracks_best_torso_crops": torso_sheet,
            "track_crop_strips": track_strips,
        },
    }
    summary = {
        "status": "complete",
        "stage": "clean_track_crops_for_jersey_ocr_summary",
        "output_dir": str(output_dir),
        "counts": {
            "tracks": len(candidates_by_track),
            "tracking_detections": len(tracking.get("detections", [])),
            "original_video_frames_decoded": decoded_original_frames,
            "full_body_candidates": candidate_full_count,
            "torso_candidates": candidate_torso_count,
            "full_body_crops_saved": len(full_records),
            "torso_crops_saved": len(torso_records),
            "skipped_records": len(skipped_detections),
        },
        "artifacts": {
            "metadata": str(output_dir / "clean_crop_metadata.json"),
            "summary": str(output_dir / "clean_crop_summary.json"),
            "all_tracks_best_clean_crops": full_sheet,
            "all_tracks_best_torso_crops": torso_sheet,
            "track_crop_strips": track_strips,
        },
        "known_limitations": [
            "Torso crops use a fixed bbox-relative region and do not estimate body pose or front/back orientation.",
            "BBox overlap is only a weak occlusion proxy.",
            "Tracking identity switches can place different players in one track folder.",
            "Crop ranking estimates OCR usefulness but does not test number visibility.",
            "No OCR or player identity inference is run by this stage.",
        ],
    }
    write_json(output_dir / "clean_crop_metadata.json", metadata)
    write_json(output_dir / "clean_crop_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
