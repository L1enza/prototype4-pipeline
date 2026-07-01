#!/usr/bin/env python3
"""Conservative smoke-stage track quality and referee/non-player filtering."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from render_side_by_side_field_tracking_smoke import blend_static_heatmap, compose_frame, draw_field_panel, group_points, load_tracking_frames, write_gif
from run_heatmap_smoke import save_heatmap_images

KNOWN_LIMITATIONS = [
    "This is a smoke-stage heuristic filter, not a trained classifier.",
    "Manual homography is approximate.",
    "Player foot points are noisy.",
    "Tracking IDs may still switch.",
    "Referee appearance detection uses crude color/stripe cues.",
    "Rejected and uncertain tracks are preserved in metadata; raw track data is not deleted.",
]
LABELS = {"likely_player", "likely_referee", "likely_non_player", "low_quality", "uncertain"}
REJECT_LABELS = {"likely_referee", "likely_non_player", "low_quality"}
COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Score and conservatively filter short-clip tracks.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--tracking-metadata", default=None, help="Override stabilized tracking metadata path.")
    parser.add_argument("--tracking-frames-dir", default=None, help="Override stabilized tracking overlay frames directory.")
    parser.add_argument("--projected-points", default=None, help="Override projected field points path.")
    parser.add_argument("--field-template", default="assets/field_templates/nll_field_topdown.png", help="Top-down field template image.")
    parser.add_argument("--crop-dir", default=None, help="Optional existing crop/contact-sheet directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--min-player-frames", type=int, default=4, help="Minimum frames for likely_player label.")
    parser.add_argument("--low-quality-min-frames", type=int, default=3, help="Tracks shorter than this are low quality.")
    parser.add_argument("--max-image-jump", type=float, default=260.0, help="Strong image jump threshold in pixels.")
    parser.add_argument("--max-field-jump", type=float, default=190.0, help="Strong field-template jump threshold in pixels.")
    parser.add_argument("--near-edge-frac", type=float, default=0.70, help="Fraction of projected points near boards/sidelines considered a strong cue.")
    parser.add_argument("--edge-margin-frac", type=float, default=0.08, help="Template edge margin for boards/sideline proximity.")
    parser.add_argument("--heatmap-sigma", type=float, default=18.0, help="Cleaned heatmap blur sigma.")
    parser.add_argument("--heatmap-alpha", type=float, default=0.55, help="Cleaned heatmap overlay alpha.")
    parser.add_argument("--gif-fps", type=float, default=6.0, help="Cleaned side-by-side GIF FPS.")
    parser.add_argument("--trail-length", type=int, default=7, help="Cleaned side-by-side field trail length.")
    parser.add_argument("--field-panel-width", type=int, default=720, help="Cleaned side-by-side field panel width.")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def color_for(track_id):
    return COLORS[(int(track_id) - 1) % len(COLORS)]


def group_detections(detections):
    tracks = {}
    for det in detections:
        tracks.setdefault(int(det["track_id"]), []).append(det)
    for rows in tracks.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return tracks


def group_projected(points):
    tracks = {}
    for point in points:
        tracks.setdefault(int(point["track_id"]), []).append(point)
    for rows in tracks.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return tracks


def bbox_area(bbox):
    if not bbox:
        return None
    return float(max(0, bbox["x1"] - bbox["x0"] + 1) * max(0, bbox["y1"] - bbox["y0"] + 1))


def point_xy(point):
    if not point:
        return None
    return np.asarray([float(point["x"]), float(point["y"])], dtype=np.float32)


def projected_xy(point):
    field = point.get("projected_field_point") or {}
    if "x" not in field or "y" not in field:
        return None
    return np.asarray([float(field["x"]), float(field["y"])], dtype=np.float32)


def distances(values):
    out = []
    for a, b in zip(values[:-1], values[1:]):
        if a is None or b is None:
            continue
        out.append(float(np.linalg.norm(b - a)))
    return out


def safe_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def safe_var(values):
    values = [v for v in values if v is not None]
    return float(np.var(values)) if values else None


def best_detection(detections):
    return max(detections, key=lambda det: bbox_area(det.get("bbox_2d")) or 0.0)


def load_frame_from_detection(det):
    meta_path = Path(det.get("source_frame_metadata") or "")
    if meta_path.exists():
        meta = load_json(meta_path)
        frame_path = Path(meta.get("original_path") or "")
        if frame_path.exists():
            return frame_path
    return None


def crop_detection(det, output_dir, track_id, padding=0.22):
    bbox = det.get("bbox_2d")
    frame_path = load_frame_from_detection(det)
    if not bbox or frame_path is None:
        return None
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    bw = max(1, int(bbox["x1"] - bbox["x0"] + 1))
    bh = max(1, int(bbox["y1"] - bbox["y0"] + 1))
    pad = int(round(max(bw, bh) * padding))
    x0 = max(0, int(bbox["x0"]) - pad)
    y0 = max(0, int(bbox["y0"]) - pad)
    x1 = min(width, int(bbox["x1"]) + pad + 1)
    y1 = min(height, int(bbox["y1"]) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "track_{:03d}_best_crop.png".format(track_id)
    image.crop((x0, y0, x1, y1)).save(out)
    return str(out)


def crop_stats(path):
    if not path or not Path(path).exists():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    mean_sat = float(np.mean(sat))
    median_sat = float(np.median(sat))
    low_sat_fraction = float(np.mean(sat < 45.0))
    high_sat_fraction = float(np.mean(sat > 95.0))
    dark_fraction = float(np.mean(val < 70.0))
    bright_gray_fraction = float(np.mean((val > 155.0) & (sat < 70.0)))
    gray_fraction = float(np.mean(sat < 35.0))
    grayscale_std = float(np.std(gray))
    # Crude stripe cue: low-saturation crop with both dark and bright/gray content plus contrast.
    stripe_score = float(min(1.0, 0.40 * low_sat_fraction + 0.30 * min(1.0, dark_fraction / 0.22) + 0.30 * min(1.0, bright_gray_fraction / 0.26)))
    if grayscale_std < 38.0:
        stripe_score *= 0.65
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "mean_saturation": mean_sat,
        "median_saturation": median_sat,
        "low_saturation_fraction": low_sat_fraction,
        "high_saturation_fraction": high_sat_fraction,
        "dark_fraction": dark_fraction,
        "bright_gray_fraction": bright_gray_fraction,
        "gray_fraction": gray_fraction,
        "grayscale_std": grayscale_std,
        "stripe_score": stripe_score,
        "sharpness": sharpness,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def existing_best_crop(track_id, crop_root):
    path = crop_root / "track_{:03d}".format(track_id) / "best_crop.png"
    return str(path) if path.exists() else None


def edge_fraction(projected_points, template_size, margin_frac):
    if not projected_points:
        return None
    width, height = template_size
    x_margin = width * float(margin_frac)
    y_margin = height * float(margin_frac)
    flags = []
    for point in projected_points:
        xy = projected_xy(point)
        if xy is None:
            continue
        x, y = float(xy[0]), float(xy[1])
        flags.append(x <= x_margin or x >= width - x_margin or y <= y_margin or y >= height - y_margin)
    return float(np.mean(flags)) if flags else None


def score_track(track_id, detections, projected_points, template_size, crop_path, existing_crop_path, args, saturation_baseline):
    frames = [int(det["frame_index"]) for det in detections]
    frame_count = len(frames)
    first_frame = min(frames)
    last_frame = max(frames)
    missing_frames = max(0, (last_frame - first_frame + 1) - frame_count)
    areas = [bbox_area(det.get("bbox_2d")) for det in detections]
    foot_points = [point_xy(det.get("foot_point_2d")) for det in detections]
    centroids = [point_xy(det.get("centroid_2d")) for det in detections]
    foot_dist = distances(foot_points)
    centroid_dist = distances(centroids)
    image_speeds = foot_dist or centroid_dist
    projected_in = [p for p in projected_points if p.get("inside_field_template_bounds")]
    projected_out = [p for p in projected_points if not p.get("inside_field_template_bounds")]
    projected_dist = distances([projected_xy(p) for p in projected_in])
    inside_pct = len(projected_in) / max(1, len(projected_points))
    outside_pct = len(projected_out) / max(1, len(projected_points))
    near_edge = edge_fraction(projected_in, template_size, args.edge_margin_frac)
    appearance = crop_stats(crop_path) or crop_stats(existing_crop_path)

    reasons = []
    quality_flags = []
    referee_flags = []
    non_player_flags = []
    if frame_count < args.low_quality_min_frames:
        quality_flags.append("too_few_frames")
    if outside_pct > 0.50:
        quality_flags.append("mostly_outside_field_bounds")
    if max(image_speeds or [0.0]) > args.max_image_jump:
        quality_flags.append("unrealistic_image_jump")
    if max(projected_dist or [0.0]) > args.max_field_jump:
        quality_flags.append("unrealistic_field_jump")
    if missing_frames >= max(3, frame_count // 2):
        quality_flags.append("many_missing_frames")
    if appearance and appearance.get("sharpness", 0.0) < 18.0:
        quality_flags.append("poor_crop_quality")

    if near_edge is not None and near_edge >= args.near_edge_frac:
        non_player_flags.append("frequent_boards_sideline_region")
    if safe_mean(image_speeds) is not None and safe_mean(image_speeds) < 3.0 and frame_count >= args.min_player_frames:
        non_player_flags.append("very_low_motion")

    if appearance:
        if appearance["stripe_score"] >= 0.62:
            referee_flags.append("striped_low_saturation_crop")
        if appearance["mean_saturation"] < min(55.0, saturation_baseline * 0.65):
            referee_flags.append("low_team_color_saturation")
        if appearance["gray_fraction"] > 0.62 and appearance["dark_fraction"] > 0.12:
            referee_flags.append("black_white_gray_appearance")

    label = "uncertain"
    confidence = 0.45
    if len(quality_flags) >= 2 or ("too_few_frames" in quality_flags and (outside_pct > 0.25 or max(image_speeds or [0.0]) > args.max_image_jump)):
        label = "low_quality"
        confidence = min(0.92, 0.58 + 0.12 * len(quality_flags))
        reasons.extend(quality_flags)
    elif len(referee_flags) >= 2 and not quality_flags:
        label = "likely_referee"
        confidence = min(0.88, 0.56 + 0.10 * len(referee_flags))
        reasons.extend(referee_flags)
        if near_edge is not None and near_edge > 0.45:
            reasons.append("some_boards_sideline_presence")
    elif len(non_player_flags) >= 2:
        label = "likely_non_player"
        confidence = min(0.82, 0.55 + 0.10 * len(non_player_flags))
        reasons.extend(non_player_flags)
    elif frame_count >= args.min_player_frames and inside_pct >= 0.80 and max(image_speeds or [0.0]) <= args.max_image_jump and max(projected_dist or [0.0]) <= args.max_field_jump and len(referee_flags) < 2:
        label = "likely_player"
        confidence = 0.68
        reasons.append("persistent_in_bounds_track")
        if appearance and appearance["high_saturation_fraction"] > 0.12:
            confidence += 0.07
            reasons.append("visible_team_color_saturation")
        if safe_mean(image_speeds) and safe_mean(image_speeds) >= 5.0:
            confidence += 0.05
            reasons.append("plausible_player_motion")
        confidence = min(0.88, confidence)
    else:
        reasons.extend(quality_flags or referee_flags or non_player_flags or ["insufficient_strong_evidence"])
        confidence = 0.45 + min(0.15, 0.03 * frame_count)

    return {
        "track_id": int(track_id),
        "frame_count": int(frame_count),
        "first_frame": int(first_frame),
        "last_frame": int(last_frame),
        "average_bbox_area": safe_mean(areas),
        "bbox_area_variance": safe_var(areas),
        "average_image_speed": safe_mean(image_speeds),
        "average_field_speed": safe_mean(projected_dist),
        "maximum_image_jump_distance": float(max(image_speeds or [0.0])),
        "maximum_field_jump_distance": float(max(projected_dist or [0.0])),
        "missing_frames": int(missing_frames),
        "projected_point_count": int(len(projected_points)),
        "projected_inside_count": int(len(projected_in)),
        "projected_outside_count": int(len(projected_out)),
        "percent_projected_inside_bounds": float(inside_pct),
        "percent_projected_outside_bounds": float(outside_pct),
        "near_boards_sideline_fraction": near_edge,
        "best_crop_path": crop_path,
        "existing_crop_best_path": existing_crop_path,
        "appearance_stats": appearance,
        "quality_flags": quality_flags,
        "referee_flags": referee_flags,
        "non_player_flags": non_player_flags,
        "final_label": label,
        "confidence_score": float(confidence),
        "human_readable_reasons": reasons,
    }


def draw_review_sheet(track_records, output_path, thumb_size=(180, 160), columns=4):
    rows = int(math.ceil(len(track_records) / float(columns))) if track_records else 1
    cell_w = thumb_size[0] + 26
    cell_h = thumb_size[1] + 92
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, record in enumerate(track_records):
        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h
        crop_path = record.get("best_crop_path") or record.get("existing_crop_best_path")
        if crop_path and Path(crop_path).exists():
            thumb = Image.open(crop_path).convert("RGB")
            thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x + 10, y + 10))
        label = record["final_label"]
        color = (32, 130, 55) if label == "likely_player" else (190, 50, 40) if label in REJECT_LABELS else (180, 125, 20)
        draw.rectangle((x + 6, y + 6, x + cell_w - 8, y + cell_h - 8), outline=color, width=3)
        text_y = y + thumb_size[1] + 18
        reason = ", ".join(record.get("human_readable_reasons", [])[:2])[:48]
        draw.text((x + 10, text_y), "T{} {}".format(record["track_id"], label), fill=color)
        draw.text((x + 10, text_y + 18), "conf {:.2f} | frames {}".format(record["confidence_score"], record["frame_count"]), fill=(20, 20, 20))
        draw.text((x + 10, text_y + 36), reason, fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def write_csv(records, output_path):
    fields = [
        "track_id", "final_label", "confidence_score", "frame_count", "first_frame", "last_frame", "average_bbox_area",
        "bbox_area_variance", "average_image_speed", "average_field_speed", "maximum_image_jump_distance",
        "maximum_field_jump_distance", "missing_frames", "projected_point_count", "percent_projected_inside_bounds",
        "percent_projected_outside_bounds", "near_boards_sideline_fraction", "best_crop_path", "human_readable_reasons",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["human_readable_reasons"] = "; ".join(record.get("human_readable_reasons", []))
            writer.writerow(row)
    return str(output_path)


def cleaned_heatmap(points, keep_ids, field_template, output_dir):
    included = [p for p in points if int(p["track_id"]) in keep_ids and p.get("inside_field_template_bounds")]
    template = Image.open(field_template).convert("RGB")
    stats = save_heatmap_images(included, template, output_dir / "cleaned_heatmap_preview.png", output_dir / "cleaned_heatmap_preview_overlay.png", 18.0, 0.55, 1.0)
    return included, stats


def render_clean_side_by_side(run_id, keep_ids, clean_points, field_template, output_dir, fps=6.0, trail_length=7, field_panel_width=720):
    tracking_frames_dir = PROJECT_ROOT / "outputs" / run_id / "short_clip_tracking_stabilized" / "frames"
    heatmap_overlay = output_dir / "cleaned_heatmap_preview_overlay.png"
    side_dir = output_dir / "cleaned_side_by_side_frames"
    side_dir.mkdir(parents=True, exist_ok=True)
    tracking_frames = load_tracking_frames(tracking_frames_dir)
    by_frame, by_track = group_points(clean_points)
    field_base = Image.open(field_template).convert("RGB")
    if heatmap_overlay.exists():
        field_base = blend_static_heatmap(field_base, heatmap_overlay, 0.18)
    rendered = []
    for frame_index, left_path in tracking_frames:
        field_panel, _scale, _count = draw_field_panel(field_base, clean_points, by_track, frame_index, field_panel_width, trail_length, 6, 0.22, 14.0)
        output_path = side_dir / "frame_{:03d}_cleaned_side_by_side.png".format(frame_index)
        rendered.append(Path(compose_frame(left_path, field_panel, output_path, 12)))
    gif = write_gif(rendered, output_dir / "cleaned_side_by_side_preview.gif", fps) if rendered else None
    return {"gif": gif, "frames_dir": str(side_dir), "rendered_frames": [str(path) for path in rendered], "kept_track_ids": sorted(int(tid) for tid in keep_ids)}


def main():
    args = parse_args()
    tracking_metadata_path = Path(args.tracking_metadata) if args.tracking_metadata else PROJECT_ROOT / "outputs" / args.run_id / "short_clip_tracking_stabilized" / "tracking_metadata.json"
    projected_points_path = Path(args.projected_points) if args.projected_points else PROJECT_ROOT / "outputs" / args.run_id / "field_calibration_smoke" / "projected_player_points.json"
    field_template = project_path(args.field_template)
    crop_root = Path(args.crop_dir) if args.crop_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_crops" / "tracklet_crop_smoke"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "track_quality_filter_smoke"
    best_crop_dir = output_dir / "best_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = load_json(tracking_metadata_path)
    projected_payload = load_json(projected_points_path)
    all_projected = projected_payload.get("points", [])
    detections_by_track = group_detections(tracking.get("detections", []))
    projected_by_track = group_projected(all_projected)
    template = Image.open(field_template).convert("RGB")

    temp_records = []
    for track_id, detections in sorted(detections_by_track.items()):
        best_det = best_detection(detections)
        crop_path = crop_detection(best_det, best_crop_dir, track_id)
        stats = crop_stats(crop_path) if crop_path else None
        temp_records.append((track_id, stats))
    saturation_values = [stats["mean_saturation"] for _track_id, stats in temp_records if stats]
    saturation_baseline = float(np.median(saturation_values)) if saturation_values else 80.0

    records = []
    for track_id, detections in sorted(detections_by_track.items()):
        best_det = best_detection(detections)
        crop_path = str(best_crop_dir / "track_{:03d}_best_crop.png".format(track_id))
        if not Path(crop_path).exists():
            crop_path = crop_detection(best_det, best_crop_dir, track_id)
        existing_crop_path = existing_best_crop(track_id, crop_root)
        record = score_track(
            track_id,
            detections,
            projected_by_track.get(track_id, []),
            template.size,
            crop_path,
            existing_crop_path,
            args,
            saturation_baseline,
        )
        records.append(record)
    records.sort(key=lambda item: int(item["track_id"]))

    kept = [r for r in records if r["final_label"] == "likely_player"]
    rejected = [r for r in records if r["final_label"] in REJECT_LABELS]
    uncertain = [r for r in records if r["final_label"] == "uncertain"]
    keep_ids = {int(r["track_id"]) for r in kept}

    clean_points, heatmap_stats = cleaned_heatmap(all_projected, keep_ids, field_template, output_dir)
    side_preview = render_clean_side_by_side(args.run_id, keep_ids, clean_points, field_template, output_dir, fps=args.gif_fps, trail_length=args.trail_length, field_panel_width=args.field_panel_width)
    review_sheet = draw_review_sheet(records, output_dir / "review_contact_sheet.png")
    table_csv = write_csv(records, output_dir / "track_quality_table.csv")

    cleaned_ids = {
        "likely_player_track_ids": sorted(keep_ids),
        "rejected_track_ids": sorted(int(r["track_id"]) for r in rejected),
        "uncertain_track_ids": sorted(int(r["track_id"]) for r in uncertain),
        "hide_from_cleaned_visualizations_track_ids": sorted(int(r["track_id"]) for r in rejected + uncertain),
        "include_in_cleaned_visualizations_track_ids": sorted(keep_ids),
    }
    write_json(output_dir / "kept_tracks.json", {"tracks": kept, "track_ids": cleaned_ids["likely_player_track_ids"]})
    write_json(output_dir / "rejected_tracks.json", {"tracks": rejected, "track_ids": cleaned_ids["rejected_track_ids"]})
    write_json(output_dir / "uncertain_tracks.json", {"tracks": uncertain, "track_ids": cleaned_ids["uncertain_track_ids"]})
    write_json(output_dir / "cleaned_track_ids.json", cleaned_ids)

    label_counts = {label: sum(1 for r in records if r["final_label"] == label) for label in sorted(LABELS)}
    metadata = {
        "status": "complete",
        "stage": "track_quality_filter_smoke",
        "run_id": args.run_id,
        "inputs": {
            "tracking_metadata": str(tracking_metadata_path),
            "projected_points": str(projected_points_path),
            "field_template": str(field_template),
            "crop_root": str(crop_root),
        },
        "parameters": {
            "min_player_frames": args.min_player_frames,
            "low_quality_min_frames": args.low_quality_min_frames,
            "max_image_jump": args.max_image_jump,
            "max_field_jump": args.max_field_jump,
            "near_edge_frac": args.near_edge_frac,
            "edge_margin_frac": args.edge_margin_frac,
            "saturation_baseline": saturation_baseline,
        },
        "track_quality": records,
        "cleaned_track_ids": cleaned_ids,
        "label_counts": label_counts,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "track_quality_table_csv": table_csv,
            "kept_tracks": str(output_dir / "kept_tracks.json"),
            "rejected_tracks": str(output_dir / "rejected_tracks.json"),
            "uncertain_tracks": str(output_dir / "uncertain_tracks.json"),
            "cleaned_track_ids": str(output_dir / "cleaned_track_ids.json"),
            "review_contact_sheet": review_sheet,
            "cleaned_heatmap_preview": heatmap_stats["heatmap"],
            "cleaned_heatmap_preview_overlay": heatmap_stats["overlay"],
            "cleaned_side_by_side_preview_gif": side_preview["gif"],
            "cleaned_side_by_side_frames_dir": side_preview["frames_dir"],
        },
    }
    summary = {
        "status": "complete",
        "stage": "track_quality_filter_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "track_quality_metadata.json"),
        "counts": {
            "total_tracks": len(records),
            "kept_tracks": len(kept),
            "rejected_tracks": len(rejected),
            "uncertain_tracks": len(uncertain),
            "likely_player": label_counts.get("likely_player", 0),
            "likely_referee": label_counts.get("likely_referee", 0),
            "likely_non_player": label_counts.get("likely_non_player", 0),
            "low_quality": label_counts.get("low_quality", 0),
            "cleaned_projected_points": len(clean_points),
        },
        "cleaned_track_ids": cleaned_ids,
        "artifacts": metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(output_dir / "track_quality_metadata.json", metadata)
    write_json(output_dir / "track_quality_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
