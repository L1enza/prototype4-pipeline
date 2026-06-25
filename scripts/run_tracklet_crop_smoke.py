#!/usr/bin/env python3
"""Create crop/contact-sheet smoke artifacts for short player tracklets."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_LIMITATIONS = [
    "Tracklets may contain identity swaps.",
    "Referee may still appear as a tracklet.",
    "Jersey numbers may not be visible in every crop.",
    "This is only a smoke crop stage.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create crop contact sheets for player tracklets.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--tracklet-metadata", default=None, help="Override tracklet metadata path.")
    parser.add_argument("--filtered-dir", default=None, help="Override filtered SAM 3 directory.")
    parser.add_argument("--sampled-frames-dir", default=None, help="Override sampled frames directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--padding", type=float, default=0.28, help="BBox padding as fraction of max bbox dimension.")
    parser.add_argument("--min-padding-px", type=int, default=12, help="Minimum crop padding in pixels.")
    parser.add_argument("--contact-thumb-width", type=int, default=180, help="Per-crop thumbnail width in contact sheets.")
    parser.add_argument("--contact-thumb-height", type=int, default=220, help="Per-crop thumbnail height in contact sheets.")
    parser.add_argument("--apply-mask-highlight", action=argparse.BooleanOptionalAction, default=True, help="Highlight the SAM 3 mask over a dimmed crop background.")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clamp_bbox_with_padding(bbox, width, height, padding_fraction, min_padding_px):
    x0 = float(bbox["x0"])
    y0 = float(bbox["y0"])
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    pad = max(float(min_padding_px), max(x1 - x0 + 1.0, y1 - y0 + 1.0) * float(padding_fraction))
    return {
        "x0": int(max(0, math.floor(x0 - pad))),
        "y0": int(max(0, math.floor(y0 - pad))),
        "x1": int(min(width - 1, math.ceil(x1 + pad))),
        "y1": int(min(height - 1, math.ceil(y1 + pad))),
    }


def crop_area(bbox):
    return int(max(0, bbox["x1"] - bbox["x0"] + 1) * max(0, bbox["y1"] - bbox["y0"] + 1))


def border_margin(bbox, width, height):
    return int(min(bbox["x0"], bbox["y0"], width - 1 - bbox["x1"], height - 1 - bbox["y1"]))


def sharpness_score(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return 0.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(gx.var() + gy.var())


def highlight_crop(image, mask, crop_box):
    crop = image.crop((crop_box["x0"], crop_box["y0"], crop_box["x1"] + 1, crop_box["y1"] + 1)).convert("RGBA")
    mask_crop = mask.crop((crop_box["x0"], crop_box["y0"], crop_box["x1"] + 1, crop_box["y1"] + 1)).convert("L")
    mask_arr = np.asarray(mask_crop) > 0
    base = np.asarray(crop).copy()
    dimmed = base.copy()
    dimmed[..., :3] = (dimmed[..., :3] * 0.42).astype(np.uint8)
    base[~mask_arr] = dimmed[~mask_arr]
    tint = np.zeros_like(base)
    tint[..., 1] = 190
    tint[..., 3] = mask_arr.astype(np.uint8) * 65
    composed = Image.alpha_composite(Image.fromarray(base, mode="RGBA"), Image.fromarray(tint, mode="RGBA"))
    return composed.convert("RGB")


def plain_crop(image, crop_box):
    return image.crop((crop_box["x0"], crop_box["y0"], crop_box["x1"] + 1, crop_box["y1"] + 1)).convert("RGB")


def detection_by_key(detections):
    return {(int(d["frame_index"]), int(d["mask_id"])): d for d in detections}


def track_members_from_metadata(tracklet, detection_lookup):
    members = []
    for member in tracklet.get("members", []):
        key = (int(member["frame_index"]), int(member["mask_id"]))
        det = detection_lookup.get(key)
        if det:
            members.append(det)
    return sorted(members, key=lambda item: (item["frame_index"], item["mask_id"]))


def best_crop_score(record):
    area = float(record.get("bbox_area", 0))
    border = float(max(record.get("border_margin_px", 0), 0))
    has_3d = 1.0 if record.get("has_3d_support") else 0.0
    sharp = float(record.get("sharpness_score", 0.0))
    # Area dominates, with small boosts for non-edge crops, 3D support, and crispness.
    return area + border * 35.0 + has_3d * 2200.0 + min(sharp, 12000.0) * 0.08


def make_contact_sheet(crop_records, output_path, thumb_w, thumb_h, title=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not crop_records:
        sheet = Image.new("RGB", (thumb_w, thumb_h), "white")
        sheet.save(output_path)
        return str(output_path)
    cells = []
    for record in crop_records:
        img = Image.open(record["crop_path"]).convert("RGB")
        img.thumbnail((thumb_w, thumb_h - 34))
        cell = Image.new("RGB", (thumb_w, thumb_h), "white")
        cell.paste(img, ((thumb_w - img.width) // 2, 0))
        draw = ImageDraw.Draw(cell)
        label = "T{} f{} m{}".format(record["track_id"], record["frame_index"], record["mask_id"])
        if record.get("has_3d_support"):
            label += " 3D"
        draw.text((6, thumb_h - 30), label, fill=(0, 0, 0))
        cells.append(cell)
    cols = min(5, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    title_h = 28 if title else 0
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h + title_h), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 7), title, fill=(0, 0, 0))
    for idx, cell in enumerate(cells):
        sheet.paste(cell, ((idx % cols) * thumb_w, title_h + (idx // cols) * thumb_h))
    sheet.save(output_path)
    return str(output_path)


def create_detection_crop(det, track_id, image, args, track_dir):
    bbox = det.get("bbox_2d")
    if not bbox:
        return None
    width, height = image.size
    crop_box = clamp_bbox_with_padding(bbox, width, height, args.padding, args.min_padding_px)
    mask_path = det.get("mask_path")
    mask = Image.open(mask_path).convert("L") if mask_path and Path(mask_path).exists() else Image.new("L", image.size, 0)
    crop = highlight_crop(image, mask, crop_box) if args.apply_mask_highlight else plain_crop(image, crop_box)
    crop_name = "track_{:03d}_frame_{:03d}_mask_{:03d}.png".format(track_id, int(det["frame_index"]), int(det["mask_id"]))
    crop_path = track_dir / crop_name
    crop.save(crop_path)
    record = {
        "track_id": int(track_id),
        "frame_index": int(det["frame_index"]),
        "mask_id": int(det["mask_id"]),
        "crop_path": str(crop_path),
        "source_mask_path": mask_path,
        "bbox": bbox,
        "crop_box": crop_box,
        "bbox_area": int((bbox["x1"] - bbox["x0"] + 1) * (bbox["y1"] - bbox["y0"] + 1)),
        "crop_size": {"width": crop.width, "height": crop.height},
        "centroid_2d": det.get("centroid_2d"),
        "foot_point_2d": det.get("foot_point_2d"),
        "centroid_3d": det.get("centroid_3d"),
        "has_3d_support": det.get("centroid_3d") is not None,
        "fusion_point_count": int(det.get("fusion_point_count") or 0),
        "match_score": det.get("match_score"),
        "match_source": det.get("match_source"),
        "border_margin_px": border_margin(crop_box, width, height),
        "sharpness_score": sharpness_score(crop),
    }
    record["best_crop_score"] = best_crop_score(record)
    return record


def main():
    args = parse_args()
    tracklet_metadata_path = Path(args.tracklet_metadata) if args.tracklet_metadata else PROJECT_ROOT / "outputs" / args.run_id / "player_tracks" / "tracklet_smoke" / "tracklet_metadata.json"
    filtered_dir = Path(args.filtered_dir) if args.filtered_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_filtered"
    sampled_frames_dir = Path(args.sampled_frames_dir) if args.sampled_frames_dir else PROJECT_ROOT / "outputs" / args.run_id / "sampled_frames"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_crops" / "tracklet_crop_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(tracklet_metadata_path)
    detection_lookup = detection_by_key(metadata.get("detections", []))
    all_crop_records = []
    track_records = []
    best_records = []

    for tracklet in sorted(metadata.get("tracklets", []), key=lambda item: int(item["track_id"])):
        track_id = int(tracklet["track_id"])
        members = track_members_from_metadata(tracklet, detection_lookup)
        track_dir = output_dir / "track_{:03d}".format(track_id)
        track_dir.mkdir(parents=True, exist_ok=True)
        crop_records = []
        for det in members:
            frame_meta_path = Path(det["source_frame_metadata"])
            frame_meta = load_json(frame_meta_path)
            image = Image.open(frame_meta["original_path"]).convert("RGB")
            crop_record = create_detection_crop(det, track_id, image, args, track_dir)
            if crop_record:
                crop_records.append(crop_record)
                all_crop_records.append(crop_record)
        crop_records.sort(key=lambda item: (item["frame_index"], item["mask_id"]))
        contact_sheet_path = make_contact_sheet(
            crop_records,
            track_dir / "track_{:03d}_contact_sheet.png".format(track_id),
            args.contact_thumb_width,
            args.contact_thumb_height,
            title="Track {} crops".format(track_id),
        )
        best = max(crop_records, key=best_crop_score) if crop_records else None
        best_path = None
        if best:
            best_img = Image.open(best["crop_path"]).convert("RGB")
            best_path = track_dir / "best_crop.png"
            best_img.save(best_path)
            best = dict(best)
            best["best_crop_path"] = str(best_path)
            best_records.append(best)
        track_record = {
            "track_id": track_id,
            "frames_covered": tracklet.get("frames_covered"),
            "has_3d_support": tracklet.get("has_3d_support"),
            "average_association_score": tracklet.get("average_association_score"),
            "crop_count": len(crop_records),
            "track_dir": str(track_dir),
            "contact_sheet_path": contact_sheet_path,
            "best_crop_path": str(best_path) if best_path else None,
            "best_crop": best,
            "crops": crop_records,
        }
        write_json(track_dir / "track_crop_metadata.json", track_record)
        track_records.append(track_record)

    best_records.sort(key=lambda item: int(item["track_id"]))
    global_sheet_path = make_contact_sheet(
        [{**record, "crop_path": record["best_crop_path"]} for record in best_records],
        output_dir / "all_tracklets_best_crops.png",
        args.contact_thumb_width,
        args.contact_thumb_height,
        title="Best crop per tracklet",
    )

    crop_metadata = {
        "status": "complete",
        "stage": "tracklet_crop_smoke",
        "run_id": args.run_id,
        "inputs": {
            "tracklet_metadata": str(tracklet_metadata_path),
            "filtered_sam3_dir": str(filtered_dir),
            "sampled_frames_dir": str(sampled_frames_dir),
        },
        "parameters": {
            "padding": args.padding,
            "min_padding_px": args.min_padding_px,
            "apply_mask_highlight": bool(args.apply_mask_highlight),
            "contact_thumb_width": args.contact_thumb_width,
            "contact_thumb_height": args.contact_thumb_height,
        },
        "tracklets": track_records,
        "crops": all_crop_records,
        "best_crops": best_records,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "global_best_crop_contact_sheet": global_sheet_path,
            "tracklet_contact_sheets": [record["contact_sheet_path"] for record in track_records],
        },
    }
    summary = {
        "status": "complete",
        "stage": "tracklet_crop_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "crop_metadata.json"),
        "counts": {
            "tracklets": len(track_records),
            "crops": len(all_crop_records),
            "best_crops": len(best_records),
            "tracklets_with_3d_support": sum(1 for record in track_records if record.get("has_3d_support")),
            "multi_crop_tracklets": sum(1 for record in track_records if record.get("crop_count", 0) > 1),
        },
        "artifacts": crop_metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(output_dir / "crop_metadata.json", crop_metadata)
    write_json(output_dir / "crop_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
