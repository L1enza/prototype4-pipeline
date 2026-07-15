#!/usr/bin/env python3
"""Build a small manual jersey-digit labeling set from enhanced number-region crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_REGION_MANIFEST = "outputs/nll_test4/enhanced_number_regions/enhanced_number_region_manifest.json"
DEFAULT_REGION_SUMMARY = "outputs/nll_test4/enhanced_number_regions/enhanced_number_region_summary.json"
DEFAULT_CROP_OCR = "outputs/nll_test4/jersey_ocr_enhanced_local_results/crop_ocr_predictions.json"
DEFAULT_TRACK_OCR = "outputs/nll_test4/jersey_ocr_enhanced_local_results/track_jersey_number_predictions.json"
DEFAULT_DEBUG_TRACKS = "outputs/nll_test4/debug_tracks_with_enhanced_ocr.json"
DEFAULT_OUTPUT_DIR = "outputs/nll_test4/manual_jersey_digit_eval_set"
DEFAULT_REVIEW_DIR = "review_exports/nll_test4_manual_jersey_eval"

CSV_FIELDS = [
    "eval_id",
    "track_id",
    "frame_index",
    "timestamp_seconds",
    "source_image_path",
    "copied_image_path",
    "variant_name",
    "ocr_text",
    "ocr_confidence",
    "track_candidate_numbers",
    "manual_label",
    "manual_readable",
    "notes",
]

VARIANT_PRIORITY = {
    "center_contrast_gray_upscaled": 1.0,
    "center_sharpened_rgb_upscaled": 0.92,
    "center_enlarged_rgb": 0.88,
    "center_threshold_binary_upscaled": 0.82,
    "center_threshold_inverted_upscaled": 0.78,
    "center_chest_back_rgb": 0.72,
    "upper_center_torso_rgb": 0.63,
    "wider_torso_rgb": 0.55,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manual jersey digit evaluation set from enhanced number-region crops."
    )
    parser.add_argument("--region-manifest", default=DEFAULT_REGION_MANIFEST)
    parser.add_argument("--region-summary", default=DEFAULT_REGION_SUMMARY)
    parser.add_argument("--crop-ocr-predictions", default=DEFAULT_CROP_OCR)
    parser.add_argument("--track-ocr-predictions", default=DEFAULT_TRACK_OCR)
    parser.add_argument("--debug-tracks", default=DEFAULT_DEBUG_TRACKS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--examples-per-track", type=int, default=8)
    parser.add_argument("--min-examples-per-track", type=int, default=5)
    parser.add_argument("--max-total-images", type=int, default=200)
    parser.add_argument("--max-same-frame-per-track", type=int, default=2)
    parser.add_argument("--contact-thumb-width", type=int, default=150)
    parser.add_argument("--contact-thumb-height", type=int, default=120)
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def prediction_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        safe_int(row.get("track_id"), -1),
        safe_int(row.get("frame_index"), -1),
        str(row.get("number_region_variant") or row.get("variant_name") or ""),
    )


def region_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        safe_int(row.get("track_id"), -1),
        safe_int(row.get("frame_index"), -1),
        str(row.get("variant_name") or ""),
    )


def best_ocr_text(prediction: dict[str, Any] | None) -> str:
    if not prediction:
        return ""
    for key in ("candidate_number", "digit_string", "raw_text"):
        value = prediction.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    attempts = prediction.get("ocr_attempts") or []
    for attempt in attempts:
        for key in ("candidate_number", "digit_string", "raw_text"):
            value = attempt.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def best_ocr_confidence(prediction: dict[str, Any] | None) -> float | None:
    if not prediction:
        return None
    values = []
    for value in (prediction.get("confidence"),):
        if value is not None:
            values.append(safe_float(value))
    for attempt in prediction.get("ocr_attempts") or []:
        if attempt.get("confidence") is not None:
            values.append(safe_float(attempt.get("confidence")))
    return max(values) if values else None


def track_candidate_text(track_row: dict[str, Any] | None, debug_row: dict[str, Any] | None) -> str:
    candidates = []
    if track_row:
        for item in track_row.get("all_candidate_numbers") or []:
            number = item.get("number")
            if number is not None:
                candidates.append(
                    {
                        "number": str(number),
                        "vote_count": item.get("vote_count"),
                        "mean_ocr_confidence": item.get("mean_ocr_confidence"),
                    }
                )
        if track_row.get("best_candidate_number") is not None and not candidates:
            candidates.append({"number": str(track_row.get("best_candidate_number"))})
    if debug_row and not candidates:
        enhanced = debug_row.get("enhanced_jersey_number") or {}
        for item in enhanced.get("candidate_numbers") or []:
            number = item.get("number")
            if number is not None:
                candidates.append({"number": str(number), "vote_count": item.get("vote_count")})
    return json.dumps(candidates, sort_keys=True)


def resolve_region_path(region_root: Path, row: dict[str, Any]) -> Path:
    raw = Path(str(row.get("region_path") or ""))
    if raw.is_absolute() and raw.exists():
        return raw
    candidate = region_root / raw
    if candidate.exists():
        return candidate
    # Some manifests store paths relative to the project root.
    candidate = PROJECT_ROOT / raw
    return candidate


def region_score(region: dict[str, Any], prediction: dict[str, Any] | None) -> float:
    readiness = safe_float(region.get("ocr_readiness_score"))
    width = safe_float(region.get("region_width"))
    height = safe_float(region.get("region_height"))
    area_score = min((width * height) / 30000.0, 1.0)
    variant = str(region.get("variant_name") or "")
    variant_score = VARIANT_PRIORITY.get(variant, 0.4)
    has_ocr_text = 1.0 if best_ocr_text(prediction) else 0.0
    ocr_status = str((prediction or {}).get("status") or "")
    status_score = 0.35 if ocr_status == "uncertain" else 0.15 if ocr_status == "unreadable" else 0.0
    audit_score = 0.2 if region.get("audit_ocr_candidate") else 0.0
    view_score = 0.15 if str(region.get("likely_view") or "") not in {"unknown", ""} else 0.0
    return (
        4.0 * has_ocr_text
        + 2.0 * readiness
        + 1.3 * area_score
        + 1.1 * variant_score
        + status_score
        + audit_score
        + view_score
    )


def build_candidates(
    region_manifest: dict[str, Any],
    crop_predictions: dict[str, Any],
    track_predictions: dict[str, Any],
    debug_tracks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    region_root = project_path(region_manifest.get("output_root") or "outputs/nll_test4/enhanced_number_regions")
    predictions_by_key = {prediction_key(row): row for row in crop_predictions.get("predictions", [])}
    tracks_by_id = {safe_int(row.get("track_id"), -1): row for row in track_predictions.get("tracks", [])}
    debug_by_id = {safe_int(row.get("track_id"), -1): row for row in debug_tracks}

    candidates = []
    for region in region_manifest.get("regions", []):
        track_id = safe_int(region.get("track_id"), -1)
        frame_index = safe_int(region.get("frame_index"), -1)
        variant = str(region.get("variant_name") or "")
        source_path = resolve_region_path(region_root, region)
        if track_id < 0 or frame_index < 0 or not source_path.exists():
            continue
        pred = predictions_by_key.get(region_key(region))
        track_row = tracks_by_id.get(track_id)
        debug_row = debug_by_id.get(track_id)
        ocr_conf = best_ocr_confidence(pred)
        candidates.append(
            {
                "track_id": track_id,
                "frame_index": frame_index,
                "timestamp_seconds": region.get("timestamp_seconds"),
                "source_image_path": source_path,
                "variant_name": variant,
                "region_id": region.get("region_id"),
                "crop_path": region.get("crop_path") or region.get("source_crop_path"),
                "source_frame_index": region.get("source_frame_index"),
                "ocr_readiness_score": region.get("ocr_readiness_score"),
                "region_width": region.get("region_width"),
                "region_height": region.get("region_height"),
                "likely_view": region.get("likely_view"),
                "view_confidence": region.get("view_confidence"),
                "number_region_guess": region.get("number_region_guess"),
                "ocr_text": best_ocr_text(pred),
                "ocr_confidence": ocr_conf,
                "ocr_status": (pred or {}).get("status"),
                "ocr_reason": (pred or {}).get("reason"),
                "track_candidate_numbers": track_candidate_text(track_row, debug_row),
                "selection_score": region_score(region, pred),
            }
        )
    return candidates


def select_examples(
    candidates: list[dict[str, Any]],
    examples_per_track: int,
    min_examples_per_track: int,
    max_total_images: int,
    max_same_frame_per_track: int,
) -> list[dict[str, Any]]:
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_track[int(candidate["track_id"])].append(candidate)

    selected: list[dict[str, Any]] = []
    for track_id in sorted(by_track):
        ranked = sorted(
            by_track[track_id],
            key=lambda row: (
                safe_float(row.get("selection_score")),
                safe_float(row.get("ocr_readiness_score")),
                safe_float(row.get("region_width")) * safe_float(row.get("region_height")),
            ),
            reverse=True,
        )
        track_selected: list[dict[str, Any]] = []
        frame_counts: dict[int, int] = defaultdict(int)
        seen_signatures = set()
        for row in ranked:
            frame = int(row["frame_index"])
            signature = (frame, row.get("variant_name"))
            if signature in seen_signatures:
                continue
            if frame_counts[frame] >= max_same_frame_per_track:
                continue
            track_selected.append(row)
            frame_counts[frame] += 1
            seen_signatures.add(signature)
            if len(track_selected) >= examples_per_track:
                break
        if len(track_selected) < min_examples_per_track:
            for row in ranked:
                signature = (row.get("frame_index"), row.get("variant_name"))
                if signature in seen_signatures:
                    continue
                track_selected.append(row)
                seen_signatures.add(signature)
                if len(track_selected) >= min_examples_per_track:
                    break
        selected.extend(track_selected)

    if len(selected) <= max_total_images:
        return selected

    # Preserve track coverage first, then fill remaining slots by score.
    by_track_selected: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_track_selected[int(row["track_id"])].append(row)
    trimmed: list[dict[str, Any]] = []
    for track_id in sorted(by_track_selected):
        track_rows = sorted(by_track_selected[track_id], key=lambda row: safe_float(row.get("selection_score")), reverse=True)
        keep = min(len(track_rows), max(1, min_examples_per_track))
        trimmed.extend(track_rows[:keep])
    if len(trimmed) > max_total_images:
        return sorted(trimmed, key=lambda row: safe_float(row.get("selection_score")), reverse=True)[:max_total_images]
    used = {id(row) for row in trimmed}
    remaining = sorted(
        [row for row in selected if id(row) not in used],
        key=lambda row: safe_float(row.get("selection_score")),
        reverse=True,
    )
    trimmed.extend(remaining[: max_total_images - len(trimmed)])
    return trimmed


def copy_examples(selected: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(
        sorted(selected, key=lambda item: (int(item["track_id"]), int(item["frame_index"]), str(item["variant_name"]))),
        start=1,
    ):
        eval_id = f"eval_{index:04d}"
        suffix = row["source_image_path"].suffix.lower() or ".png"
        dest_name = f"{eval_id}_track_{int(row['track_id']):03d}_frame_{int(row['frame_index']):03d}_{row['variant_name']}{suffix}"
        dest = images_dir / dest_name
        shutil.copy2(row["source_image_path"], dest)
        rows.append(
            {
                "eval_id": eval_id,
                "track_id": int(row["track_id"]),
                "frame_index": int(row["frame_index"]),
                "timestamp_seconds": row.get("timestamp_seconds"),
                "source_image_path": relpath(Path(row["source_image_path"])),
                "copied_image_path": relpath(dest),
                "variant_name": row.get("variant_name") or "",
                "ocr_text": row.get("ocr_text") or "",
                "ocr_confidence": row.get("ocr_confidence"),
                "track_candidate_numbers": row.get("track_candidate_numbers") or "[]",
                "manual_label": "",
                "manual_readable": "",
                "notes": "",
                "region_id": row.get("region_id"),
                "crop_path": row.get("crop_path"),
                "source_frame_index": row.get("source_frame_index"),
                "ocr_readiness_score": row.get("ocr_readiness_score"),
                "region_width": row.get("region_width"),
                "region_height": row.get("region_height"),
                "likely_view": row.get("likely_view"),
                "view_confidence": row.get("view_confidence"),
                "number_region_guess": row.get("number_region_guess"),
                "ocr_status": row.get("ocr_status"),
                "ocr_reason": row.get("ocr_reason"),
                "selection_score": row.get("selection_score"),
            }
        )
    return rows


def write_label_sheets(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "label_sheet.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in CSV_FIELDS}
            writer.writerow(csv_row)
    write_json(output_dir / "label_sheet.json", rows)


def load_font(size: int = 13) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    for line in text.split("\n"):
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += 15


def create_contact_sheet(rows: list[dict[str, Any]], path: Path, thumb_width: int, thumb_height: int) -> None:
    if not rows:
        return
    cols = 5
    label_height = 58
    margin = 10
    cell_w = thumb_width + 2 * margin
    cell_h = thumb_height + label_height + 2 * margin
    rows_count = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows_count * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x0 = col * cell_w + margin
        y0 = grid_row * cell_h + margin
        img = fit_image(project_path(row["copied_image_path"]), thumb_width, thumb_height)
        sheet.paste(img, (x0, y0))
        text = (
            f"{row['eval_id']}  T{int(row['track_id']):03d} F{int(row['frame_index']):03d}\n"
            f"{row['variant_name'][:24]}\n"
            f"ocr={str(row.get('ocr_text') or '-')[:12]} conf={row.get('ocr_confidence')}"
        )
        draw_multiline(draw, (x0, y0 + thumb_height + 4), text, font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_readme(output_dir: Path) -> None:
    readme = """# Manual Jersey Digit Evaluation Set

This folder contains selected enhanced jersey-number region crops from `nll_test4` for manual digit-label review.

Fill in `label_sheet.csv` or `label_sheet.json`:

- `manual_label`: enter the visible jersey number exactly as seen, for example `4`, `21`, or `42`.
- Use `unknown` when the number is not readable.
- Leave ambiguous cases as `unknown` and explain briefly in `notes` if useful.
- `manual_readable`: use `yes`, `no`, or `partial`.
- Do not enter player names.
- Do not assign player identities.

This dataset is only for evaluating digit recognition. The later identity layer should aggregate jersey-number evidence across a track and only map to roster/player identity when number/team evidence is strong.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def folder_size(path: Path) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    large_files = []
    if not path.exists():
        return 0, []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total += size
        if size > 90 * 1024 * 1024:
            large_files.append({"path": relpath(item), "size_bytes": size})
    return total, large_files


def copy_review_export(output_dir: Path, review_dir: Path) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "images").mkdir(parents=True, exist_ok=True)
    for image in (output_dir / "images").glob("*"):
        if image.is_file():
            shutil.copy2(image, review_dir / "images" / image.name)
    for name in ("label_sheet.csv", "label_sheet.json", "eval_set_manifest.json", "contact_sheet.png", "README.md"):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, review_dir / name)


def write_manifest(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_dir: Path,
    review_dir: Path,
    input_counts: dict[str, Any],
    primary_size: int,
    review_size: int,
    large_files: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks = sorted({int(row["track_id"]) for row in rows})
    ocr_text_rows = [row for row in rows if str(row.get("ocr_text") or "").strip()]
    by_track: dict[str, int] = defaultdict(int)
    for row in rows:
        by_track[f"track_{int(row['track_id']):03d}"] += 1
    manifest = {
        "stage": "manual_jersey_digit_eval_set",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_assignment_performed": False,
        "training_run": False,
        "inputs": {
            "region_manifest": args.region_manifest,
            "region_summary": args.region_summary,
            "crop_ocr_predictions": args.crop_ocr_predictions,
            "track_ocr_predictions": args.track_ocr_predictions,
            "debug_tracks": args.debug_tracks,
        },
        "outputs": {
            "output_dir": relpath(output_dir),
            "review_export_dir": relpath(review_dir),
            "images_dir": relpath(output_dir / "images"),
            "label_sheet_csv": relpath(output_dir / "label_sheet.csv"),
            "label_sheet_json": relpath(output_dir / "label_sheet.json"),
            "contact_sheet": relpath(output_dir / "contact_sheet.png"),
            "readme": relpath(output_dir / "README.md"),
        },
        "selection": {
            "examples_per_track": args.examples_per_track,
            "min_examples_per_track": args.min_examples_per_track,
            "max_total_images": args.max_total_images,
            "max_same_frame_per_track": args.max_same_frame_per_track,
            "strategy": "score enhanced regions by OCR evidence, OCR-readiness, region size, and variant usefulness; limit near-duplicate same-frame variants per track",
        },
        "counts": {
            "images_selected": len(rows),
            "tracks_covered": len(tracks),
            "selected_with_ocr_text": len(ocr_text_rows),
            "input_region_count": input_counts.get("regions"),
            "input_ocr_prediction_count": input_counts.get("ocr_predictions"),
            "input_track_prediction_count": input_counts.get("track_predictions"),
            "examples_by_track": dict(sorted(by_track.items())),
        },
        "size_checks": {
            "output_dir_total_bytes": primary_size,
            "output_dir_total_mb": round(primary_size / (1024 * 1024), 3),
            "review_dir_total_bytes": review_size,
            "review_dir_total_mb": round(review_size / (1024 * 1024), 3),
            "files_over_90mb": large_files,
            "warning": "Files over 90 MB found." if large_files else None,
        },
        "known_limitations": [
            "Manual labels are intentionally blank until reviewed by a human.",
            "Examples may include unreadable, partial, obstructed, or referee crops because the goal is recognizer evaluation.",
            "This set must not be used to assign player identity directly.",
            "OCR text fields are weak local Tesseract evidence only and should not be treated as ground truth.",
        ],
    }
    write_json(output_dir / "eval_set_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    region_manifest_path = project_path(args.region_manifest)
    region_summary_path = project_path(args.region_summary)
    crop_predictions_path = project_path(args.crop_ocr_predictions)
    track_predictions_path = project_path(args.track_ocr_predictions)
    debug_tracks_path = project_path(args.debug_tracks)
    output_dir = project_path(args.output_dir)
    review_dir = project_path(args.review_dir)

    region_manifest = read_json(region_manifest_path)
    region_summary = read_json(region_summary_path, {})
    crop_predictions = read_json(crop_predictions_path)
    track_predictions = read_json(track_predictions_path)
    debug_tracks = read_json(debug_tracks_path, [])

    candidates = build_candidates(region_manifest, crop_predictions, track_predictions, debug_tracks)
    selected = select_examples(
        candidates,
        examples_per_track=args.examples_per_track,
        min_examples_per_track=args.min_examples_per_track,
        max_total_images=args.max_total_images,
        max_same_frame_per_track=args.max_same_frame_per_track,
    )
    if not selected:
        raise RuntimeError("No eligible enhanced number-region images were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = copy_examples(selected, output_dir)
    write_label_sheets(rows, output_dir)
    create_contact_sheet(rows, output_dir / "contact_sheet.png", args.contact_thumb_width, args.contact_thumb_height)
    write_readme(output_dir)

    primary_size, primary_large = folder_size(output_dir)
    copy_review_export(output_dir, review_dir)
    review_size, review_large = folder_size(review_dir)
    large_files = primary_large + review_large

    manifest = write_manifest(
        args,
        rows,
        output_dir,
        review_dir,
        input_counts={
            "regions": len(region_manifest.get("regions", [])),
            "ocr_predictions": len(crop_predictions.get("predictions", [])),
            "track_predictions": len(track_predictions.get("tracks", [])),
            "region_summary_counts": region_summary.get("counts", {}),
        },
        primary_size=primary_size,
        review_size=review_size,
        large_files=large_files,
    )
    # Manifest is copied after it exists so the review export has the final size info.
    copy_review_export(output_dir, review_dir)
    review_size, review_large = folder_size(review_dir)
    if review_large:
        manifest["size_checks"]["files_over_90mb"] = primary_large + review_large
        manifest["size_checks"]["warning"] = "Files over 90 MB found."
        manifest["size_checks"]["review_dir_total_bytes"] = review_size
        manifest["size_checks"]["review_dir_total_mb"] = round(review_size / (1024 * 1024), 3)
        write_json(output_dir / "eval_set_manifest.json", manifest)
        shutil.copy2(output_dir / "eval_set_manifest.json", review_dir / "eval_set_manifest.json")

    print(f"images_selected={manifest['counts']['images_selected']}")
    print(f"tracks_covered={manifest['counts']['tracks_covered']}")
    print(f"output_dir={manifest['outputs']['output_dir']}")
    print(f"review_export_dir={manifest['outputs']['review_export_dir']}")
    print(f"contact_sheet={manifest['outputs']['contact_sheet']}")
    print(f"label_sheet_csv={manifest['outputs']['label_sheet_csv']}")
    print(f"output_dir_total_mb={manifest['size_checks']['output_dir_total_mb']}")
    print(f"review_dir_total_mb={manifest['size_checks']['review_dir_total_mb']}")
    if manifest["size_checks"].get("warning"):
        print(f"warning={manifest['size_checks']['warning']}")
    else:
        print("warning=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
