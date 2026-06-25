#!/usr/bin/env python3
"""Run a stabilized short-clip SAM 3 active-player tracking smoke stage."""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.sam3_filtering import DEFAULT_FILTER_PROMPT, FILTER_MODES, parse_polygon_json, run_filtered_masks
from render_tracklet_overlay_video import render_frame, write_gif, write_mp4
from run_player_tracklet_smoke import bbox_iou, distance_2d, load_detections, track_summary, write_json
from run_short_clip_tracking_smoke import DEFAULT_SAM3_REPO, decode_clip, flatten_detections

KNOWN_LIMITATIONS = [
    "Referee may still survive active-player filtering.",
    "Player clusters can still cause ID switches.",
    "This is still short-clip smoke tracking, not final tracking.",
    "No full-game scaling yet.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run stabilized SAM 3 tracking on a short contiguous video clip.")
    parser.add_argument("--video", required=True, help="Source lacrosse video path.")
    parser.add_argument("--run-id", required=True, help="Prototype 4 run id.")
    parser.add_argument("--start-time", type=float, default=0.0, help="Clip start time in seconds.")
    parser.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds.")
    parser.add_argument("--device", default="cuda", help="Torch device for SAM 3 smoke inference.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Decode every Nth source frame inside the clip.")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum decoded frames to process.")
    parser.add_argument("--repo", default=DEFAULT_SAM3_REPO, help="Local SAM 3 repo path.")
    parser.add_argument("--prompt", default=DEFAULT_FILTER_PROMPT, help="SAM 3 text prompt.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="SAM 3 smoke dtype.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Allow SAM 3 to download weights if they are not already cached.")
    parser.add_argument("--disable-fused-kernels", action=argparse.BooleanOptionalAction, default=True, help="Disable SAM 3 fused kernels during smoke inference.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--output-fps", type=float, default=None, help="Rendered overlay FPS. Defaults to source_fps/frame_stride.")
    parser.add_argument("--mask-alpha", type=int, default=70, help="Mask fill alpha, 0-255.")
    parser.add_argument("--line-width", type=int, default=4, help="Overlay line width.")

    parser.add_argument("--filter-mode", choices=sorted(FILTER_MODES), default="combined", help="Active-player geometry filter mode.")
    parser.add_argument("--field-y-min", type=float, default=0.38, help="Normalized or pixel y cutoff for playable field lower edge of bench/boards.")
    parser.add_argument("--field-y-max", type=float, default=0.96, help="Normalized or pixel max y field bound.")
    parser.add_argument("--bench-y-cutoff", type=float, default=0.36, help="Normalized or pixel y cutoff for bench strip rejection.")
    parser.add_argument("--field-polygon-json", default=None, help="Optional JSON field polygon as [[x,y], ...] or {points: [...]}.")
    parser.add_argument("--min-mask-pixels", type=int, default=35, help="Minimum mask area to keep.")
    parser.add_argument("--green-sample-radius", type=int, default=5, help="Green foot-point diagnostic radius.")
    parser.add_argument("--green-sample-y-offset", type=int, default=6, help="Green foot-point diagnostic y offset.")

    parser.add_argument("--max-centroid-dist", type=float, default=150.0, help="Predicted 2D centroid distance gate in pixels.")
    parser.add_argument("--max-foot-dist", type=float, default=175.0, help="Predicted bottom/foot point distance gate in pixels.")
    parser.add_argument("--min-iou", type=float, default=0.005, help="Minimum bbox IoU to count as supporting evidence.")
    parser.add_argument("--max-track-age", type=int, default=5, help="Keep unmatched tracks alive for this many frames.")
    parser.add_argument("--min-track-length-to-render", type=int, default=3, help="Hide shorter tracks in rendered overlays only.")
    parser.add_argument("--max-match-cost", type=float, default=1.45, help="Greedy association cost cutoff.")
    parser.add_argument("--weight-centroid", type=float, default=1.2, help="Weight for normalized predicted centroid distance.")
    parser.add_argument("--weight-foot", type=float, default=1.2, help="Weight for normalized predicted foot distance.")
    parser.add_argument("--weight-iou", type=float, default=0.8, help="Weight for 1-IoU bbox term.")
    parser.add_argument("--weight-size", type=float, default=0.45, help="Weight for bbox size similarity.")
    parser.add_argument("--weight-color", type=float, default=0.55, help="Weight for masked crop color similarity.")
    parser.add_argument("--merge-max-gap", type=int, default=8, help="Maximum frame gap for post-pass fragment merging.")
    parser.add_argument("--merge-max-distance", type=float, default=170.0, help="Maximum endpoint/startpoint distance for fragment merging.")
    parser.add_argument("--merge-min-color-similarity", type=float, default=0.58, help="Minimum color similarity for color-supported fragment merges.")
    return parser.parse_args()


def point_to_array(point):
    if not point:
        return None
    return np.asarray([float(point["x"]), float(point["y"])], dtype=np.float32)


def array_to_point(value):
    if value is None:
        return None
    return {"x": float(value[0]), "y": float(value[1])}


def bbox_area(bbox):
    if not bbox:
        return None
    return max(0.0, float(bbox["x1"] - bbox["x0"] + 1) * float(bbox["y1"] - bbox["y0"] + 1))


def size_similarity(a, b):
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    if not area_a or not area_b:
        return None
    return float(min(area_a, area_b) / max(area_a, area_b))


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-9:
        return None
    return float(np.dot(av, bv) / denom)


def color_histogram(image_bgr, mask, bbox):
    if image_bgr is None or not bbox:
        return None
    h, w = image_bgr.shape[:2]
    x0 = max(0, int(bbox["x0"]))
    y0 = max(0, int(bbox["y0"]))
    x1 = min(w - 1, int(bbox["x1"]))
    y1 = min(h - 1, int(bbox["y1"]))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    crop_mask = None
    if mask is not None and mask.shape[:2] == image_bgr.shape[:2]:
        crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255
        if int(crop_mask.sum()) == 0:
            crop_mask = None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], crop_mask, [12, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-9:
        return None
    return (hist / norm).tolist()


def enrich_appearance(detections_by_frame):
    image_cache = {}
    for frame_record in detections_by_frame.values():
        original_path = frame_record.get("original_path")
        image_bgr = None
        if original_path:
            image_bgr = image_cache.get(original_path)
            if image_bgr is None:
                image_bgr = cv2.imread(str(original_path))
                image_cache[original_path] = image_bgr
        for det in frame_record.get("detections", []):
            mask = None
            mask_path = det.get("mask_path")
            if mask_path and Path(mask_path).exists():
                raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    mask = raw > 0
            hist = color_histogram(image_bgr, mask, det.get("bbox_2d"))
            det["appearance_histogram"] = hist
            det["bbox_area"] = bbox_area(det.get("bbox_2d"))


def predict_point(members, frame_index, key):
    points = [(int(m["frame_index"]), point_to_array(m.get(key))) for m in members if point_to_array(m.get(key)) is not None]
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    last_frame, last = points[-1]
    gap = max(1, int(frame_index) - last_frame)
    if len(points) < 2:
        return array_to_point(last)
    prev_frame, prev = points[-2]
    dt = max(1, last_frame - prev_frame)
    velocity = (last - prev) / float(dt)
    return array_to_point(last + velocity * float(gap))


def average_hist(members):
    hists = [np.asarray(m["appearance_histogram"], dtype=np.float32) for m in members if m.get("appearance_histogram") is not None]
    if not hists:
        return None
    avg = np.mean(np.stack(hists, axis=0), axis=0)
    norm = float(np.linalg.norm(avg))
    if norm <= 1e-9:
        return None
    return (avg / norm).tolist()


def association_to_track(track, det, frame_index, args):
    members = track["members"]
    last = members[-1]
    gap = max(1, int(frame_index) - int(last["frame_index"]))
    gate_scale = 1.0 + 0.20 * float(max(0, gap - 1))
    pred_centroid = predict_point(members, frame_index, "centroid_2d")
    pred_foot = predict_point(members, frame_index, "foot_point_2d")
    centroid_dist = distance_2d(pred_centroid, det.get("centroid_2d"))
    foot_dist = distance_2d(pred_foot, det.get("foot_point_2d"))
    iou = bbox_iou(last.get("bbox_2d"), det.get("bbox_2d"))
    size_sim = size_similarity(last.get("bbox_2d"), det.get("bbox_2d"))
    color_sim = cosine_similarity(track.get("appearance_histogram"), det.get("appearance_histogram"))

    centroid_gate = args.max_centroid_dist * gate_scale
    foot_gate = args.max_foot_dist * gate_scale
    has_gate = False
    if centroid_dist is not None and centroid_dist <= centroid_gate:
        has_gate = True
    if foot_dist is not None and foot_dist <= foot_gate:
        has_gate = True
    if iou >= args.min_iou:
        has_gate = True
    if color_sim is not None and color_sim >= 0.82 and (centroid_dist is None or centroid_dist <= centroid_gate * 1.35):
        has_gate = True
    if not has_gate:
        return None

    weighted = []
    sources = []
    if centroid_dist is not None:
        weighted.append((args.weight_centroid, min(centroid_dist / max(centroid_gate, 1e-6), 3.0)))
        sources.append("predicted_centroid")
    if foot_dist is not None:
        weighted.append((args.weight_foot, min(foot_dist / max(foot_gate, 1e-6), 3.0)))
        sources.append("predicted_foot")
    weighted.append((args.weight_iou, 1.0 - iou))
    sources.append("bbox_iou")
    if size_sim is not None:
        weighted.append((args.weight_size, 1.0 - size_sim))
        sources.append("bbox_size")
    if color_sim is not None:
        weighted.append((args.weight_color, 1.0 - max(0.0, min(1.0, color_sim))))
        sources.append("color_histogram")
    weight_sum = sum(weight for weight, _value in weighted)
    cost = float(sum(weight * value for weight, value in weighted) / max(weight_sum, 1e-6))
    if cost > args.max_match_cost:
        return None
    return {
        "cost": cost,
        "score": float(1.0 / (1.0 + cost)),
        "sources": sources,
        "reason": "stabilized_memory_prediction_match",
        "gap": gap,
        "predicted_centroid": pred_centroid,
        "predicted_foot_point": pred_foot,
        "centroid_distance_2d": centroid_dist,
        "foot_distance_2d": foot_dist,
        "bbox_iou": iou,
        "bbox_size_similarity": size_sim,
        "color_similarity": color_sim,
    }


def assign_stabilized_tracklets(detections_by_frame, args):
    next_track_id = 1
    tracks = {}
    active = set()
    match_events = []
    lifecycle = []
    diagnostics = {}
    for frame_index in sorted(detections_by_frame):
        detections = detections_by_frame[frame_index].get("detections", [])
        candidates = []
        candidate_tracks = [tid for tid in sorted(active) if frame_index - tracks[tid]["last_frame"] <= args.max_track_age]
        for track_id in candidate_tracks:
            for det_index, det in enumerate(detections):
                assoc = association_to_track(tracks[track_id], det, frame_index, args)
                if assoc is not None:
                    candidates.append((assoc["cost"], track_id, det_index, assoc))
        candidates.sort(key=lambda item: item[0])
        used_tracks = set()
        used_detections = set()
        frame_new_ids = 0
        frame_matches = 0
        for _cost, track_id, det_index, assoc in candidates:
            if track_id in used_tracks or det_index in used_detections:
                continue
            det = detections[det_index]
            last = tracks[track_id]["members"][-1]
            det["track_id"] = int(track_id)
            det["match_score"] = assoc["score"]
            det["match_source"] = "+".join(assoc["sources"])
            det["match_reason"] = assoc["reason"]
            det["matched_from"] = {"frame_index": last["frame_index"], "mask_id": last["mask_id"], "track_id": int(track_id)}
            det["association_metrics"] = {k: v for k, v in assoc.items() if k not in {"sources", "reason"}}
            tracks[track_id]["members"].append(det)
            tracks[track_id]["last_frame"] = int(frame_index)
            tracks[track_id]["missed_frames"] = 0
            tracks[track_id]["appearance_histogram"] = average_hist(tracks[track_id]["members"])
            used_tracks.add(track_id)
            used_detections.add(det_index)
            frame_matches += 1
            match_events.append({"frame_index": int(frame_index), "track_id": int(track_id), "mask_id": det["mask_id"], "matched_from_mask_id": last["mask_id"], "score": assoc["score"], "cost": assoc["cost"], "gap": assoc["gap"]})

        for det_index, det in enumerate(detections):
            if det_index in used_detections:
                continue
            track_id = next_track_id
            next_track_id += 1
            det["track_id"] = int(track_id)
            det["match_score"] = None
            det["match_source"] = "new_track"
            det["match_reason"] = "unmatched_detection_started_tracklet"
            det["matched_from"] = None
            tracks[track_id] = {
                "track_id": int(track_id),
                "members": [det],
                "start_frame": int(frame_index),
                "last_frame": int(frame_index),
                "missed_frames": 0,
                "appearance_histogram": det.get("appearance_histogram"),
                "status": "active",
            }
            active.add(track_id)
            frame_new_ids += 1
            lifecycle.append({"event": "track_started", "frame_index": int(frame_index), "track_id": int(track_id), "mask_id": int(det["mask_id"])})

        killed = []
        for track_id in list(active):
            if track_id in used_tracks or tracks[track_id]["last_frame"] == frame_index:
                continue
            tracks[track_id]["missed_frames"] = int(frame_index - tracks[track_id]["last_frame"])
            if tracks[track_id]["missed_frames"] > args.max_track_age:
                active.remove(track_id)
                tracks[track_id]["status"] = "killed"
                killed.append(track_id)
                lifecycle.append({"event": "track_killed", "frame_index": int(frame_index), "track_id": int(track_id), "missed_frames": tracks[track_id]["missed_frames"]})

        diagnostics[str(frame_index)] = {
            "frame_index": int(frame_index),
            "detections": len(detections),
            "matches": frame_matches,
            "new_ids": frame_new_ids,
            "unmatched_detections": frame_new_ids,
            "active_tracks_after_frame": len(active),
            "tracks_killed": len(killed),
            "killed_track_ids": [int(tid) for tid in killed],
        }
    return tracks, match_events, lifecycle, diagnostics


def non_overlapping(a, b):
    frames_a = {int(m["frame_index"]) for m in a["members"]}
    frames_b = {int(m["frame_index"]) for m in b["members"]}
    return not bool(frames_a & frames_b)


def merge_candidate(a, b, args):
    if not non_overlapping(a, b):
        return None
    if a["last_frame"] >= b["start_frame"]:
        return None
    gap = int(b["start_frame"] - a["last_frame"])
    if gap < 1 or gap > args.merge_max_gap:
        return None
    endpoint = a["members"][-1]
    start = b["members"][0]
    centroid_dist = distance_2d(endpoint.get("centroid_2d"), start.get("centroid_2d"))
    foot_dist = distance_2d(endpoint.get("foot_point_2d"), start.get("foot_point_2d"))
    pred_centroid = predict_point(a["members"], b["start_frame"], "centroid_2d")
    pred_foot = predict_point(a["members"], b["start_frame"], "foot_point_2d")
    pred_centroid_dist = distance_2d(pred_centroid, start.get("centroid_2d"))
    pred_foot_dist = distance_2d(pred_foot, start.get("foot_point_2d"))
    color_sim = cosine_similarity(a.get("appearance_histogram"), b.get("appearance_histogram"))
    best_dist = min([d for d in [centroid_dist, foot_dist, pred_centroid_dist, pred_foot_dist] if d is not None] or [float("inf")])
    if best_dist > args.merge_max_distance:
        return None
    if color_sim is not None and color_sim < args.merge_min_color_similarity:
        return None
    cost = best_dist / max(args.merge_max_distance, 1e-6) + 0.08 * gap
    if color_sim is not None:
        cost += 0.35 * (1.0 - color_sim)
    return {
        "cost": float(cost),
        "gap": gap,
        "endpoint_start_distance": centroid_dist,
        "foot_distance": foot_dist,
        "predicted_centroid_distance": pred_centroid_dist,
        "predicted_foot_distance": pred_foot_dist,
        "color_similarity": color_sim,
    }


def merge_track_into(target, source):
    target_id = int(target["track_id"])
    for member in sorted(source["members"], key=lambda item: int(item["frame_index"])):
        member["track_id"] = target_id
        member["match_reason"] = "post_pass_fragment_merge"
        member["match_source"] = (member.get("match_source") or "") + "+fragment_merge"
        target["members"].append(member)
    target["members"].sort(key=lambda item: int(item["frame_index"]))
    target["last_frame"] = int(max(m["frame_index"] for m in target["members"]))
    target["start_frame"] = int(min(m["frame_index"] for m in target["members"]))
    target["appearance_histogram"] = average_hist(target["members"])


def merge_fragments(tracks, args):
    merges = []
    removed = set()
    changed = True
    while changed:
        changed = False
        candidates = []
        ids = [tid for tid in sorted(tracks) if tid not in removed]
        for a_id in ids:
            for b_id in ids:
                if a_id == b_id:
                    continue
                candidate = merge_candidate(tracks[a_id], tracks[b_id], args)
                if candidate:
                    candidates.append((candidate["cost"], a_id, b_id, candidate))
        candidates.sort(key=lambda item: item[0])
        for _cost, a_id, b_id, candidate in candidates:
            if a_id in removed or b_id in removed:
                continue
            merge_track_into(tracks[a_id], tracks[b_id])
            removed.add(b_id)
            changed = True
            merges.append({"target_track_id": int(a_id), "merged_track_id": int(b_id), **candidate})
            break
    merged_tracks = {tid: track for tid, track in tracks.items() if tid not in removed}
    return merged_tracks, merges


def tracks_to_detections_by_frame(detections_by_frame, tracks):
    visible_ids = set(tracks)
    for frame_record in detections_by_frame.values():
        frame_record["detections"] = [det for det in frame_record.get("detections", []) if int(det.get("track_id", -1)) in visible_ids]
        frame_record["detections"].sort(key=lambda item: int(item["track_id"]))
    return detections_by_frame


def render_stabilized(detections_by_frame, tracks, output_dir, fps, mask_alpha, line_width, min_track_length):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderable_ids = {int(track_id) for track_id, track in tracks.items() if len(track["members"]) >= min_track_length}
    rendered_frames = []
    frame_records = []
    for frame_index in sorted(detections_by_frame):
        frame_record = detections_by_frame[frame_index]
        frame_path = Path(frame_record.get("original_path", ""))
        if not frame_path.exists():
            frame_records.append({"frame_index": int(frame_index), "status": "skipped", "reason": "missing_filtered_original_frame"})
            continue
        render_detections = [det for det in frame_record.get("detections", []) if int(det.get("track_id", -1)) in renderable_ids]
        output_path = frames_dir / "frame_{:03d}_tracking_overlay.png".format(frame_index)
        rendered = render_frame(frame_path, frame_index, render_detections, output_path, mask_alpha, line_width)
        rendered_frames.append(Path(rendered))
        frame_records.append({
            "frame_index": int(frame_index),
            "status": "rendered",
            "source_frame_path": str(frame_path),
            "rendered_frame_path": rendered,
            "detections_total": len(frame_record.get("detections", [])),
            "detections_rendered": len(render_detections),
            "hidden_short_track_detections": len(frame_record.get("detections", [])) - len(render_detections),
        })
    mp4_path = output_dir / "tracking_overlay.mp4"
    gif_path = output_dir / "tracking_overlay.gif"
    mp4 = None
    mp4_error = None
    gif = None
    if rendered_frames:
        try:
            mp4 = write_mp4(rendered_frames, mp4_path, fps)
        except Exception as exc:
            mp4_error = {"type": exc.__class__.__name__, "message": str(exc)}
        gif = write_gif(rendered_frames, gif_path, fps)
    return {
        "frames_dir": str(frames_dir),
        "rendered_frames": [str(path) for path in rendered_frames],
        "frame_records": frame_records,
        "rendered_track_ids": sorted(renderable_ids),
        "hidden_track_ids": sorted(set(int(tid) for tid in tracks) - renderable_ids),
        "mp4": mp4,
        "mp4_error": mp4_error,
        "gif": gif,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "short_clip_tracking_stabilized"
    decoded_dir = output_dir / "decoded_frames"
    filtered_dir = output_dir / "sam3_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "stage": "short_clip_tracking_stabilized",
        "run_id": args.run_id,
        "inputs": {"video": str(args.video), "sam3_repo": str(args.repo)},
        "known_limitations": KNOWN_LIMITATIONS,
    }
    lifecycle_debug = {"status": "running", "events": [], "per_frame": {}, "fragment_merges": []}
    try:
        decode = decode_clip(Path(args.video), decoded_dir, args.start_time, args.duration, args.frame_stride, args.max_frames)
        frame_paths = [row["frame_path"] for row in decode["decoded_frames"]]
        output_fps = args.output_fps if args.output_fps else max(1.0, float(decode.get("fps") or 0.0) / max(1, args.frame_stride))
        filter_config = {
            "filter_mode": args.filter_mode,
            "bench_y_cutoff": args.bench_y_cutoff,
            "field_y_min": args.field_y_min,
            "field_y_max": args.field_y_max,
            "field_polygon": parse_polygon_json(args.field_polygon_json) if args.field_polygon_json else None,
            "min_mask_pixels": args.min_mask_pixels,
            "green_sample_radius": args.green_sample_radius,
            "green_sample_y_offset": args.green_sample_y_offset,
        }
        sam3_result = run_filtered_masks(
            PROJECT_ROOT,
            args.run_id,
            Path(args.repo),
            filtered_dir,
            args.prompt,
            len(frame_paths),
            args.device,
            args.dtype,
            args.allow_download_weights,
            disable_fused_kernels=args.disable_fused_kernels,
            config=filter_config,
            frame_paths=frame_paths,
            require_allow_download_weights=False,
        )
        if sam3_result.get("status") != "complete":
            raise RuntimeError("SAM 3 filtered stabilized stage failed: {}".format(sam3_result.get("error")))

        detections_by_frame = load_detections(filtered_dir, {})
        valid_frame_indices = set(range(len(frame_paths)))
        detections_by_frame = {idx: row for idx, row in detections_by_frame.items() if idx in valid_frame_indices}
        enrich_appearance(detections_by_frame)
        tracks, match_events, lifecycle_events, per_frame_diag = assign_stabilized_tracklets(detections_by_frame, args)
        merged_tracks, fragment_merges = merge_fragments(tracks, args)
        detections_by_frame = tracks_to_detections_by_frame(detections_by_frame, merged_tracks)
        tracklets = [track_summary(track_id, track["members"]) for track_id, track in sorted(merged_tracks.items())]
        detections = flatten_detections(detections_by_frame)
        render_artifacts = render_stabilized(detections_by_frame, merged_tracks, output_dir, output_fps, args.mask_alpha, args.line_width, args.min_track_length_to_render)

        lifecycle_debug = {
            "status": "complete",
            "events": lifecycle_events,
            "per_frame": per_frame_diag,
            "fragment_merges": fragment_merges,
            "parameters": {
                "max_track_age": args.max_track_age,
                "max_centroid_dist": args.max_centroid_dist,
                "max_foot_dist": args.max_foot_dist,
                "min_iou": args.min_iou,
                "merge_max_gap": args.merge_max_gap,
                "merge_max_distance": args.merge_max_distance,
                "merge_min_color_similarity": args.merge_min_color_similarity,
            },
        }
        metadata.update({
            "status": "complete",
            "parameters": {
                "start_time": args.start_time,
                "duration": args.duration,
                "frame_stride": args.frame_stride,
                "max_frames": args.max_frames,
                "device": args.device,
                "dtype": args.dtype,
                "disable_fused_kernels": args.disable_fused_kernels,
                "allow_download_weights": args.allow_download_weights,
                "prompt": args.prompt,
                "filter_config": filter_config,
                "output_fps": output_fps,
                "matching": "track memory + constant-velocity prediction + greedy weighted association + fragment merge",
                "max_track_age": args.max_track_age,
                "min_track_length_to_render": args.min_track_length_to_render,
                "max_centroid_dist": args.max_centroid_dist,
                "max_foot_dist": args.max_foot_dist,
                "min_iou": args.min_iou,
                "max_match_cost": args.max_match_cost,
            },
            "decode": decode,
            "sam3_filtering": sam3_result,
            "detections": detections,
            "tracklets": tracklets,
            "match_events": match_events,
            "fragment_merges": fragment_merges,
            "render": render_artifacts,
            "artifacts": {
                "decoded_frames_dir": str(decoded_dir),
                "filtered_sam3_dir": str(filtered_dir),
                "rendered_frames_dir": render_artifacts["frames_dir"],
                "tracking_overlay_mp4": render_artifacts["mp4"],
                "tracking_overlay_gif": render_artifacts["gif"],
                "track_lifecycle_debug": str(output_dir / "track_lifecycle_debug.json"),
            },
        })
        summary_status = "complete"
        exit_code = 0
    except Exception as exc:
        metadata.update({"status": "failed", "error": {"type": exc.__class__.__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        lifecycle_debug.update({"status": "failed", "error": metadata["error"]})
        summary_status = "failed"
        exit_code = 1

    write_json(output_dir / "tracking_metadata.json", metadata)
    write_json(output_dir / "track_lifecycle_debug.json", lifecycle_debug)
    tracklets = metadata.get("tracklets", [])
    single_frame = sum(1 for t in tracklets if t.get("frames_covered", 0) == 1)
    rendered_tracks = metadata.get("render", {}).get("rendered_track_ids", [])
    summary = {
        "status": summary_status,
        "stage": "short_clip_tracking_stabilized_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "tracking_metadata.json"),
        "lifecycle_debug": str(output_dir / "track_lifecycle_debug.json"),
        "counts": {
            "decoded_frames": metadata.get("decode", {}).get("decoded_frame_count", 0),
            "frames_rendered": len(metadata.get("render", {}).get("rendered_frames", [])),
            "detections": len(metadata.get("detections", [])),
            "tracklets": len(tracklets),
            "multi_frame_tracklets": sum(1 for t in tracklets if t.get("frames_covered", 0) > 1),
            "single_frame_tracklets": single_frame,
            "match_events": len(metadata.get("match_events", [])),
            "fragment_merges": len(metadata.get("fragment_merges", [])),
            "rendered_tracks": len(rendered_tracks),
            "tracks_killed": sum(row.get("tracks_killed", 0) for row in lifecycle_debug.get("per_frame", {}).values()),
        },
        "new_ids_per_frame": {frame: row.get("new_ids", 0) for frame, row in lifecycle_debug.get("per_frame", {}).items()},
        "unmatched_detections_per_frame": {frame: row.get("unmatched_detections", 0) for frame, row in lifecycle_debug.get("per_frame", {}).items()},
        "artifacts": metadata.get("artifacts", {}),
        "mp4_error": metadata.get("render", {}).get("mp4_error"),
        "known_limitations": KNOWN_LIMITATIONS,
    }
    if metadata.get("error"):
        summary["error"] = metadata["error"]
    write_json(output_dir / "tracking_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
