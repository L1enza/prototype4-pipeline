#!/usr/bin/env python3
"""Generate context-preserving number-region variants without running OCR."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_NAMES = (
    "wider_torso_rgb",
    "upper_center_torso_rgb",
    "center_chest_back_rgb",
    "center_contrast_gray_upscaled",
    "center_threshold_binary_upscaled",
    "center_enlarged_rgb",
    "center_sharpened_rgb_upscaled",
    "center_threshold_inverted_upscaled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced jersey-number regions.")
    parser.add_argument(
        "--clean-crop-metadata",
        default="outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json",
    )
    parser.add_argument(
        "--visibility-predictions",
        default="outputs/nll_test4/jersey_crop_visibility_audit/crop_visibility_predictions.json",
    )
    parser.add_argument(
        "--track-visibility-summary",
        default="outputs/nll_test4/jersey_crop_visibility_audit/track_visibility_summary.json",
    )
    parser.add_argument("--output-dir", default="outputs/nll_test4/enhanced_number_regions")
    parser.add_argument("--enlarge-scale", type=int, choices=[2, 3], default=3)
    parser.add_argument("--contact-thumb-width", type=int, default=180)
    parser.add_argument("--contact-thumb-height", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="Replace only this stage's output directory.")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_key(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def readiness_score(audit: dict, clean: dict) -> float:
    for key in ("ocr_readiness_score", "auto_quality_score", "quality_score"):
        value = audit.get(key)
        if value is not None:
            return float(value)
    return float(clean.get("crop_quality_score") or 0.0)


def join_ocr_ready_crops(clean_payload: dict, audit_payload: dict, track_payload: dict) -> list[dict]:
    clean_by_path = {path_key(row["crop_path"]): row for row in clean_payload.get("crops", [])}
    track_by_id = {int(row["track_id"]): row for row in track_payload.get("tracks", [])}
    joined = []
    missing = []
    for audit in audit_payload.get("crops", audit_payload.get("predictions", [])):
        if not bool(audit.get("ocr_candidate")):
            continue
        crop_path = path_key(audit["crop_path"])
        clean = clean_by_path.get(crop_path)
        if clean is None:
            missing.append(crop_path)
            continue
        if not Path(crop_path).is_file():
            missing.append(crop_path)
            continue
        track_id = int(clean["track_id"])
        row = dict(clean)
        row.update(
            {
                "crop_path": crop_path,
                "ocr_readiness_score": readiness_score(audit, clean),
                "audit_status": audit.get("audit_status"),
                "audit_ocr_candidate": True,
                "likely_view": audit.get("heuristic_likely_view", audit.get("likely_view", "unknown")),
                "view_confidence": audit.get("view_confidence"),
                "number_region_guess": audit.get("number_region_guess", "unknown"),
                "number_region_box": audit.get("number_region_box"),
                "track_visibility_readiness": track_by_id.get(track_id, {}).get("readiness"),
            }
        )
        joined.append(row)
    if missing:
        raise ValueError("{} OCR-ready audit crops could not be joined/read".format(len(missing)))
    if not joined:
        raise ValueError("No OCR-ready crops found")

    grouped = defaultdict(list)
    for row in joined:
        grouped[int(row["track_id"])].append(row)
    for track_id, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                -float(row["ocr_readiness_score"]),
                0 if row.get("crop_type") == "torso" else 1,
                int(row["frame_index"]),
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["ocr_readiness_rank_within_track"] = rank
            row["ocr_ready_crop_count_for_track"] = len(rows)
    return sorted(
        joined,
        key=lambda row: (
            int(row["track_id"]),
            int(row["ocr_readiness_rank_within_track"]),
            int(row["frame_index"]),
        ),
    )


def clamp_box(box: dict, width: int, height: int) -> dict:
    x0 = max(0, min(width - 1, int(math.floor(box["x0"]))))
    y0 = max(0, min(height - 1, int(math.floor(box["y0"]))))
    x1 = max(x0, min(width - 1, int(math.ceil(box["x1"]))))
    y1 = max(y0, min(height - 1, int(math.ceil(box["y1"]))))
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def fractional_box(width: int, height: int, x0: float, y0: float, x1: float, y1: float) -> dict:
    return clamp_box(
        {
            "x0": width * x0,
            "y0": height * y0,
            "x1": width * x1 - 1,
            "y1": height * y1 - 1,
        },
        width,
        height,
    )


def padded_audit_box(box: dict | None, width: int, height: int) -> dict:
    if not box:
        return fractional_box(width, height, 0.12, 0.06, 0.88, 0.72)
    base = clamp_box(box, width, height)
    box_width = base["x1"] - base["x0"] + 1
    box_height = base["y1"] - base["y0"] + 1
    # Keep shoulders and uniform context around a noisy heuristic number box.
    return clamp_box(
        {
            "x0": base["x0"] - 0.16 * box_width,
            "y0": base["y0"] - 0.14 * box_height,
            "x1": base["x1"] + 0.16 * box_width,
            "y1": base["y1"] + 0.18 * box_height,
        },
        width,
        height,
    )


def crop(image: np.ndarray, box: dict) -> np.ndarray:
    region = image[box["y0"] : box["y1"] + 1, box["x0"] : box["x1"] + 1]
    if region.size == 0:
        raise ValueError("Computed an empty number region")
    return region.copy()


def spatial_regions(image: np.ndarray, row: dict) -> dict[str, tuple[np.ndarray, dict]]:
    height, width = image.shape[:2]
    if row.get("crop_type") == "torso":
        wider_box = fractional_box(width, height, 0.00, 0.00, 1.00, 1.00)
        upper_box = fractional_box(width, height, 0.06, 0.00, 0.94, 0.76)
    else:
        wider_box = fractional_box(width, height, 0.04, 0.02, 0.96, 0.74)
        upper_box = fractional_box(width, height, 0.12, 0.02, 0.88, 0.64)
    center_box = padded_audit_box(row.get("number_region_box"), width, height)
    return {
        "wider_torso_rgb": (crop(image, wider_box), wider_box),
        "upper_center_torso_rgb": (crop(image, upper_box), upper_box),
        "center_chest_back_rgb": (crop(image, center_box), center_box),
    }


def enhanced_variants(center_bgr: np.ndarray, scale: int) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(6, 6))
    contrast = clahe.apply(gray)
    enlarged_rgb = cv2.resize(
        center_bgr,
        (center_bgr.shape[1] * scale, center_bgr.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    enlarged_contrast = cv2.resize(
        contrast,
        (contrast.shape[1] * scale, contrast.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    _level, thresholded = cv2.threshold(
        enlarged_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    blurred = cv2.GaussianBlur(enlarged_rgb, (0, 0), sigmaX=1.1)
    sharpened = cv2.addWeighted(enlarged_rgb, 1.8, blurred, -0.8, 0)
    return {
        "center_contrast_gray_upscaled": enlarged_contrast,
        "center_threshold_binary_upscaled": thresholded,
        "center_enlarged_rgb": enlarged_rgb,
        "center_sharpened_rgb_upscaled": sharpened,
        "center_threshold_inverted_upscaled": cv2.bitwise_not(thresholded),
    }


def output_filename(row: dict, variant_name: str) -> str:
    return "track_{:03d}_frame_{:03d}_{}_{}.png".format(
        int(row["track_id"]),
        int(row["frame_index"]),
        row.get("crop_type", "crop"),
        variant_name,
    )


def generate_regions(rows: list[dict], output_dir: Path, scale: int) -> list[dict]:
    records = []
    for row in rows:
        source = cv2.imread(row["crop_path"], cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError("Could not read clean crop: {}".format(row["crop_path"]))
        spatial = spatial_regions(source, row)
        center_image, center_box = spatial["center_chest_back_rgb"]
        generated = {
            name: {"image": image, "source_box": box, "variant_type": "spatial_rgb"}
            for name, (image, box) in spatial.items()
        }
        generated.update(
            {
                name: {
                    "image": image,
                    "source_box": center_box,
                    "variant_type": "center_preprocessing",
                }
                for name, image in enhanced_variants(center_image, scale).items()
            }
        )
        if set(generated) != set(VARIANT_NAMES):
            raise RuntimeError("Enhanced region variant contract mismatch")

        source_slug = "frame_{:03d}_{}".format(int(row["frame_index"]), row.get("crop_type", "crop"))
        region_dir = output_dir / "regions" / "track_{:03d}".format(int(row["track_id"])) / source_slug
        region_dir.mkdir(parents=True, exist_ok=True)
        for variant_name in VARIANT_NAMES:
            generated_row = generated[variant_name]
            path = region_dir / output_filename(row, variant_name)
            if not cv2.imwrite(str(path), generated_row["image"]):
                raise RuntimeError("Could not write enhanced region: {}".format(path))
            image_height, image_width = generated_row["image"].shape[:2]
            records.append(
                {
                    "region_id": "track_{:03d}_frame_{:03d}_{}_{}".format(
                        int(row["track_id"]),
                        int(row["frame_index"]),
                        row.get("crop_type", "crop"),
                        variant_name,
                    ),
                    "track_id": int(row["track_id"]),
                    "frame_index": int(row["frame_index"]),
                    "source_frame_index": row.get("source_frame_index"),
                    "timestamp_seconds": row.get("timestamp_seconds"),
                    "source_crop_path": row["crop_path"],
                    "crop_path": row["crop_path"],
                    "crop_type": row.get("crop_type"),
                    "source_crop_width": int(source.shape[1]),
                    "source_crop_height": int(source.shape[0]),
                    "source_region_box": generated_row["source_box"],
                    "variant_name": variant_name,
                    "variant_type": generated_row["variant_type"],
                    "region_path": path.relative_to(output_dir).as_posix(),
                    "region_width": int(image_width),
                    "region_height": int(image_height),
                    "enlarge_scale": scale if generated_row["variant_type"] == "center_preprocessing" else 1,
                    "ocr_readiness_score": float(row["ocr_readiness_score"]),
                    "ocr_readiness_rank_within_track": int(row["ocr_readiness_rank_within_track"]),
                    "likely_view": row.get("likely_view", "unknown"),
                    "view_confidence": row.get("view_confidence"),
                    "number_region_guess": row.get("number_region_guess", "unknown"),
                    "track_visibility_readiness": row.get("track_visibility_readiness"),
                    "audit_ocr_candidate": True,
                    "manifest_preprocessed_region": True,
                }
            )
    return records


def contact_sheet_records(records: list[dict]) -> list[dict]:
    best = {}
    for row in records:
        key = (int(row["track_id"]), row["variant_name"])
        current = best.get(key)
        if current is None or (
            float(row["ocr_readiness_score"]), -int(row["frame_index"])
        ) > (
            float(current["ocr_readiness_score"]), -int(current["frame_index"])
        ):
            best[key] = row
    return [best[key] for key in sorted(best, key=lambda item: (item[0], VARIANT_NAMES.index(item[1])))]


def make_contact_sheet(
    records: list[dict], output_dir: Path, output_path: Path, thumb_width: int, thumb_height: int
) -> str:
    label_height = 42
    columns = 6
    rows = max(1, math.ceil(len(records) / columns))
    title_height = 30
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, title_height + rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Best enhanced number-region example per track and variant", fill=(0, 0, 0))
    for index, record in enumerate(records):
        image = Image.open(output_dir / record["region_path"]).convert("RGB")
        image.thumbnail((thumb_width - 10, thumb_height - 10))
        x = (index % columns) * thumb_width
        y = title_height + (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y + 5))
        draw.text(
            (x + 5, y + thumb_height + 3),
            "T{:03d} f{:03d}".format(record["track_id"], record["frame_index"]),
            fill=(0, 0, 0),
        )
        draw.text(
            (x + 5, y + thumb_height + 20),
            record["variant_name"],
            fill=(60, 60, 60),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def main() -> int:
    args = parse_args()
    clean_path = project_path(args.clean_crop_metadata)
    audit_path = project_path(args.visibility_predictions)
    track_path = project_path(args.track_visibility_summary)
    output_dir = project_path(args.output_dir)
    for path, label in (
        (clean_path, "clean crop metadata"),
        (audit_path, "visibility predictions"),
        (track_path, "track visibility summary"),
    ):
        if not path.is_file():
            raise FileNotFoundError("Missing {}: {}".format(label, path))
    if output_dir.exists():
        if not args.force:
            raise FileExistsError("Output exists; use --force to replace only this stage: {}".format(output_dir))
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    clean_payload = read_json(clean_path)
    audit_payload = read_json(audit_path)
    track_payload = read_json(track_path)
    rows = join_ocr_ready_crops(clean_payload, audit_payload, track_payload)
    regions = generate_regions(rows, output_dir, args.enlarge_scale)
    representative_records = contact_sheet_records(regions)
    contact_sheet_path = output_dir / "enhanced_number_region_contact_sheet.png"
    make_contact_sheet(
        representative_records,
        output_dir,
        contact_sheet_path,
        args.contact_thumb_width,
        args.contact_thumb_height,
    )

    variant_counts = Counter(row["variant_name"] for row in regions)
    track_region_counts = Counter(int(row["track_id"]) for row in regions)
    track_crop_counts = Counter(int(row["track_id"]) for row in rows)
    manifest_path = output_dir / "enhanced_number_region_manifest.json"
    summary_path = output_dir / "enhanced_number_region_summary.json"
    manifest = {
        "status": "complete",
        "stage": "enhanced_number_region_generation",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_dir),
        "inputs": {
            "clean_crop_metadata": str(clean_path),
            "visibility_predictions": str(audit_path),
            "track_visibility_summary": str(track_path),
        },
        "selection": {
            "requires_ocr_candidate": True,
            "source_crop_order": "descending_ocr_readiness_score",
            "source_crop_count": len(rows),
            "track_count": len(track_crop_counts),
        },
        "variant_order": list(VARIANT_NAMES),
        "variant_count_per_source_crop": len(VARIANT_NAMES),
        "regions": regions,
    }
    summary = {
        "status": "complete",
        "stage": "enhanced_number_region_summary",
        "output_dir": str(output_dir),
        "counts": {
            "ocr_ready_source_crops": len(rows),
            "tracks_covered": len(track_crop_counts),
            "enhanced_regions_generated": len(regions),
            "contact_sheet_tiles": len(representative_records),
            "by_variant": dict(sorted(variant_counts.items())),
            "source_crops_by_track": {str(key): track_crop_counts[key] for key in sorted(track_crop_counts)},
            "enhanced_regions_by_track": {str(key): track_region_counts[key] for key in sorted(track_region_counts)},
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "contact_sheet": str(contact_sheet_path),
            "summary": str(summary_path),
            "regions_dir": str(output_dir / "regions"),
        },
        "ocr_run": False,
        "training_run": False,
        "identity_assignment_performed": False,
        "known_limitations": [
            "Number boxes and body views are heuristic and may not contain visible digits.",
            "All OCR-ready crops are expanded into variants, including crops with unknown view direction.",
            "Upscaling and sharpening cannot recover detail absent from the source crop.",
            "No OCR, jersey-number acceptance, roster matching, or player identity assignment is performed.",
        ],
    }
    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
