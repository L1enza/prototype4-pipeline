#!/usr/bin/env python3
"""Build short player tracklets from filtered SAM 3 masks and SAM3+VGGT fusion metadata."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prototype4_matplotlib")

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    (0, 114, 178),
    (213, 94, 0),
]
KNOWN_LIMITATIONS = [
    "Referee may still survive the upstream filter.",
    "Clustered players may cause identity swaps.",
    "Sparse VGGT points may make some 3D matches weak.",
    "This is not final tracking, only a smoke tracklet stage.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build short smoke-test player tracklets across sampled frames.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id.")
    parser.add_argument("--filtered-dir", default=None, help="Override filtered SAM 3 directory.")
    parser.add_argument("--fusion-dir", default=None, help="Override SAM3+VGGT fusion directory.")
    parser.add_argument("--sampled-frames-dir", default=None, help="Override sampled frames directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--max-centroid-distance", type=float, default=130.0, help="2D centroid distance gate in pixels.")
    parser.add_argument("--max-foot-distance", type=float, default=150.0, help="Bottom/foot point distance gate in pixels.")
    parser.add_argument("--max-3d-distance", type=float, default=0.45, help="3D centroid distance gate in raw VGGT units.")
    parser.add_argument("--min-iou", type=float, default=0.01, help="Minimum bbox IoU to count as supporting evidence.")
    parser.add_argument("--max-match-cost", type=float, default=2.85, help="Greedy association cost cutoff.")
    parser.add_argument("--weight-centroid", type=float, default=1.0, help="Weight for normalized 2D centroid distance.")
    parser.add_argument("--weight-foot", type=float, default=1.0, help="Weight for normalized foot-point distance.")
    parser.add_argument("--weight-iou", type=float, default=0.7, help="Weight for 1-IoU bbox term.")
    parser.add_argument("--weight-3d", type=float, default=1.2, help="Weight for normalized 3D centroid distance when available.")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def color_for(track_id):
    return COLORS[(int(track_id) - 1) % len(COLORS)]


def distance_2d(a, b):
    if not a or not b:
        return None
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def distance_3d(a, b):
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        return None
    return float(np.linalg.norm(av - bv))


def bbox_iou(a, b):
    if not a or not b:
        return 0.0
    x0 = max(float(a["x0"]), float(b["x0"]))
    y0 = max(float(a["y0"]), float(b["y0"]))
    x1 = min(float(a["x1"]), float(b["x1"]))
    y1 = min(float(a["y1"]), float(b["y1"]))
    inter_w = max(0.0, x1 - x0 + 1.0)
    inter_h = max(0.0, y1 - y0 + 1.0)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a["x1"] - a["x0"] + 1) * float(a["y1"] - a["y0"] + 1))
    area_b = max(0.0, float(b["x1"] - b["x0"] + 1) * float(b["y1"] - b["y0"] + 1))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def load_fusion_centroids(fusion_dir):
    metadata_path = fusion_dir / "fusion_metadata.json"
    if not metadata_path.exists():
        return {}, None
    data = load_json(metadata_path)
    mapping = {}
    for mask in data.get("masks", []):
        mapping[(int(mask["frame_index"]), int(mask["mask_id"]))] = mask
    return mapping, str(metadata_path)


def load_detections(filtered_dir, fusion_lookup):
    detections_by_frame = {}
    for frame_dir in sorted(filtered_dir.glob("frame_*")):
        metadata_path = frame_dir / "frame_metadata.json"
        if not metadata_path.exists():
            continue
        frame_meta = load_json(metadata_path)
        frame_index = int(frame_meta["frame_index"])
        detections = []
        for kept in frame_meta.get("kept_masks", []):
            mask_id = int(kept["mask_index"])
            fusion = fusion_lookup.get((frame_index, mask_id), {})
            detection = {
                "frame_index": frame_index,
                "mask_id": mask_id,
                "track_id": None,
                "bbox_2d": kept.get("bbox"),
                "centroid_2d": kept.get("centroid"),
                "foot_point_2d": kept.get("bottom_center_point"),
                "centroid_3d": fusion.get("centroid_xyz"),
                "fusion_point_count": fusion.get("point_count", 0),
                "sam_confidence_score": kept.get("confidence_score"),
                "mask_path": kept.get("mask_path"),
                "single_mask_overlay_path": kept.get("single_mask_overlay_path"),
                "source_frame_metadata": str(metadata_path),
                "match_score": None,
                "match_source": "new_track",
                "match_reason": "unmatched_detection_started_tracklet",
                "matched_from": None,
            }
            detections.append(detection)
        detections_by_frame[frame_index] = {
            "frame_index": frame_index,
            "original_path": frame_meta.get("original_path"),
            "detections": detections,
        }
    return detections_by_frame


def association(prev_det, curr_det, args):
    cdist = distance_2d(prev_det.get("centroid_2d"), curr_det.get("centroid_2d"))
    fdist = distance_2d(prev_det.get("foot_point_2d"), curr_det.get("foot_point_2d"))
    dist3 = distance_3d(prev_det.get("centroid_3d"), curr_det.get("centroid_3d"))
    iou = bbox_iou(prev_det.get("bbox_2d"), curr_det.get("bbox_2d"))

    has_2d_gate = (cdist is not None and cdist <= args.max_centroid_distance) or (fdist is not None and fdist <= args.max_foot_distance) or iou >= args.min_iou
    has_3d_gate = dist3 is not None and dist3 <= args.max_3d_distance
    if not (has_2d_gate or has_3d_gate):
        return None

    terms = []
    sources = []
    if cdist is not None:
        terms.append(args.weight_centroid * min(cdist / max(args.max_centroid_distance, 1e-6), 3.0))
        sources.append("2d_centroid")
    if fdist is not None:
        terms.append(args.weight_foot * min(fdist / max(args.max_foot_distance, 1e-6), 3.0))
        sources.append("foot_point")
    terms.append(args.weight_iou * (1.0 - iou))
    sources.append("bbox_iou")
    if dist3 is not None:
        terms.append(args.weight_3d * min(dist3 / max(args.max_3d_distance, 1e-6), 3.0))
        sources.append("3d_centroid")
    else:
        terms.append(args.weight_3d * 0.85)
        sources.append("3d_missing")
    cost = float(sum(terms) / max(sum([args.weight_centroid, args.weight_foot, args.weight_iou, args.weight_3d]), 1e-6))
    if cost > args.max_match_cost:
        return None
    score = float(1.0 / (1.0 + cost))
    return {
        "cost": cost,
        "score": score,
        "centroid_distance_2d": cdist,
        "foot_distance_2d": fdist,
        "bbox_iou": iou,
        "centroid_distance_3d": dist3,
        "sources": sources,
        "reason": "greedy_min_cost_consecutive_frame_match",
    }


def assign_tracklets(detections_by_frame, args):
    next_track_id = 1
    tracks = {}
    prev_detections = []
    events = []
    for frame_index in sorted(detections_by_frame):
        current = detections_by_frame[frame_index]["detections"]
        if not prev_detections:
            for det in current:
                det["track_id"] = next_track_id
                tracks[next_track_id] = [det]
                next_track_id += 1
            prev_detections = current
            continue

        previous_consecutive = [det for det in prev_detections if det["frame_index"] == frame_index - 1]
        candidates = []
        for prev_idx, prev_det in enumerate(previous_consecutive):
            for curr_idx, curr_det in enumerate(current):
                assoc = association(prev_det, curr_det, args)
                if assoc is not None:
                    candidates.append((assoc["cost"], prev_idx, curr_idx, assoc))
        candidates.sort(key=lambda item: item[0])
        used_prev = set()
        used_curr = set()
        for _, prev_idx, curr_idx, assoc in candidates:
            if prev_idx in used_prev or curr_idx in used_curr:
                continue
            prev_det = previous_consecutive[prev_idx]
            curr_det = current[curr_idx]
            track_id = int(prev_det["track_id"])
            curr_det["track_id"] = track_id
            curr_det["match_score"] = assoc["score"]
            curr_det["match_source"] = "+".join(assoc["sources"])
            curr_det["match_reason"] = assoc["reason"]
            curr_det["matched_from"] = {"frame_index": prev_det["frame_index"], "mask_id": prev_det["mask_id"], "track_id": track_id}
            curr_det["association_metrics"] = {k: v for k, v in assoc.items() if k not in {"sources", "reason"}}
            tracks[track_id].append(curr_det)
            used_prev.add(prev_idx)
            used_curr.add(curr_idx)
            events.append({"frame_index": frame_index, "track_id": track_id, "mask_id": curr_det["mask_id"], "matched_from_mask_id": prev_det["mask_id"], "score": assoc["score"], "cost": assoc["cost"]})
        for curr_idx, curr_det in enumerate(current):
            if curr_idx in used_curr:
                continue
            curr_det["track_id"] = next_track_id
            tracks[next_track_id] = [curr_det]
            next_track_id += 1
        prev_detections = current
    return tracks, events


def track_summary(track_id, members):
    scores = [m["match_score"] for m in members if m.get("match_score") is not None]
    centroids_3d = [m.get("centroid_3d") for m in members if m.get("centroid_3d") is not None]
    return {
        "track_id": int(track_id),
        "members": [
            {"frame_index": m["frame_index"], "mask_id": m["mask_id"], "match_score": m.get("match_score"), "match_source": m.get("match_source")}
            for m in members
        ],
        "centroids_2d_by_frame": [{"frame_index": m["frame_index"], "centroid_2d": m.get("centroid_2d")} for m in members],
        "foot_points_by_frame": [{"frame_index": m["frame_index"], "foot_point_2d": m.get("foot_point_2d")} for m in members],
        "centroids_3d_by_frame": [{"frame_index": m["frame_index"], "centroid_3d": m.get("centroid_3d")} for m in members if m.get("centroid_3d") is not None],
        "frames_covered": int(len({m["frame_index"] for m in members})),
        "average_association_score": float(sum(scores) / len(scores)) if scores else None,
        "has_3d_support": bool(centroids_3d),
    }


def save_frame_overlay(frame_record, output_path):
    image = Image.open(frame_record["original_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in frame_record["detections"]:
        color = color_for(det["track_id"])
        bbox = det.get("bbox_2d") or {}
        if bbox:
            draw.rectangle((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]), outline=color, width=4)
            label_xy = (bbox["x0"], max(0, bbox["y0"] - 18))
        else:
            point = det.get("centroid_2d") or {"x": 8, "y": 8}
            label_xy = (point["x"], point["y"])
        label = "T{} M{}".format(det["track_id"], det["mask_id"])
        draw.text(label_xy, label, fill=color)
        foot = det.get("foot_point_2d")
        if foot:
            x = int(round(foot["x"]))
            y = int(round(foot["y"]))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def save_track_plot(path, tracks, axes, title, xlabel, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    any_points = False
    for track_id, members in sorted(tracks.items()):
        pts = [(m["frame_index"], m.get("centroid_3d")) for m in members if m.get("centroid_3d") is not None]
        if not pts:
            continue
        any_points = True
        pts = sorted(pts, key=lambda item: item[0])
        coords = np.asarray([p[1] for p in pts], dtype=np.float32)
        frames = [p[0] for p in pts]
        color = np.asarray(color_for(track_id), dtype=np.float32) / 255.0
        ax.plot(coords[:, axes[0]], coords[:, axes[1]], color=color, marker="o", linewidth=1.6, markersize=4, label="T{}".format(track_id))
        for frame, coord in zip(frames, coords):
            ax.text(coord[axes[0]], coord[axes[1]], str(frame), color=color, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any_points:
        ax.legend(fontsize=7, loc="best", frameon=True)
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main():
    args = parse_args()
    filtered_dir = Path(args.filtered_dir) if args.filtered_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_filtered"
    fusion_dir = Path(args.fusion_dir) if args.fusion_dir else PROJECT_ROOT / "outputs" / args.run_id / "fusion" / "sam3_vggt_smoke"
    sampled_frames_dir = Path(args.sampled_frames_dir) if args.sampled_frames_dir else PROJECT_ROOT / "outputs" / args.run_id / "sampled_frames"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "player_tracks" / "tracklet_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    fusion_lookup, fusion_metadata_path = load_fusion_centroids(fusion_dir)
    detections_by_frame = load_detections(filtered_dir, fusion_lookup)
    tracks, match_events = assign_tracklets(detections_by_frame, args)

    frame_overlay_paths = []
    for frame_index, frame_record in sorted(detections_by_frame.items()):
        overlay_path = output_dir / "frame_{:03d}_tracklets.png".format(frame_index)
        frame_overlay_paths.append(save_frame_overlay(frame_record, overlay_path))

    tracklets = [track_summary(track_id, members) for track_id, members in sorted(tracks.items())]
    topdown_path = save_track_plot(output_dir / "tracklets_topdown_xz.png", tracks, (0, 2), "Tracklet 3D centroids top-down X/Z", "x", "z")
    side_path = save_track_plot(output_dir / "tracklets_side_xy.png", tracks, (0, 1), "Tracklet 3D centroids side X/Y", "x", "y")

    detections = []
    for frame_index in sorted(detections_by_frame):
        for det in detections_by_frame[frame_index]["detections"]:
            detections.append(det)

    metadata = {
        "status": "complete",
        "stage": "player_tracklet_smoke",
        "run_id": args.run_id,
        "inputs": {
            "filtered_sam3_dir": str(filtered_dir),
            "fusion_dir": str(fusion_dir),
            "fusion_metadata": fusion_metadata_path,
            "sampled_frames_dir": str(sampled_frames_dir),
        },
        "parameters": {
            "max_centroid_distance": args.max_centroid_distance,
            "max_foot_distance": args.max_foot_distance,
            "max_3d_distance": args.max_3d_distance,
            "min_iou": args.min_iou,
            "max_match_cost": args.max_match_cost,
            "weight_centroid": args.weight_centroid,
            "weight_foot": args.weight_foot,
            "weight_iou": args.weight_iou,
            "weight_3d": args.weight_3d,
            "matching": "greedy consecutive-frame association",
        },
        "detections": detections,
        "tracklets": tracklets,
        "match_events": match_events,
        "known_limitations": KNOWN_LIMITATIONS,
        "artifacts": {
            "frame_overlays": frame_overlay_paths,
            "topdown_xz": topdown_path,
            "side_xy": side_path,
        },
    }
    summary = {
        "status": "complete",
        "stage": "player_tracklet_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "tracklet_metadata.json"),
        "counts": {
            "frames_processed": len(detections_by_frame),
            "detections": len(detections),
            "tracklets": len(tracklets),
            "multi_frame_tracklets": sum(1 for t in tracklets if t["frames_covered"] > 1),
            "tracklets_with_3d_support": sum(1 for t in tracklets if t["has_3d_support"]),
            "match_events": len(match_events),
        },
        "artifacts": metadata["artifacts"],
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(output_dir / "tracklet_metadata.json", metadata)
    write_json(output_dir / "tracklet_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
