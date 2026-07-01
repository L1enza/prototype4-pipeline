#!/usr/bin/env python3
"""Run a short contiguous SAM 3 active-player tracking smoke stage."""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.sam3_filtering import DEFAULT_FILTER_PROMPT, FILTER_MODES, parse_polygon_json, run_filtered_masks
from run_player_tracklet_smoke import assign_tracklets, load_detections, track_summary, write_json
from render_tracklet_overlay_video import render_frame, write_gif, write_mp4

DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/nll-test1.mp4"
DEFAULT_SAM3_REPO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3"
KNOWN_LIMITATIONS = [
    "Referee may still survive active-player filtering.",
    "Player clusters can cause ID switches.",
    "This is still short-clip smoke tracking, not final tracking.",
    "No full-game scaling yet.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM 3 active-player tracking on a short contiguous video clip.")
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

    parser.add_argument("--max-centroid-distance", type=float, default=130.0, help="2D centroid distance gate in pixels.")
    parser.add_argument("--max-foot-distance", type=float, default=150.0, help="Bottom/foot point distance gate in pixels.")
    parser.add_argument("--max-3d-distance", type=float, default=0.45, help="Unused 3D gate retained for association compatibility.")
    parser.add_argument("--min-iou", type=float, default=0.01, help="Minimum bbox IoU to count as supporting evidence.")
    parser.add_argument("--max-match-cost", type=float, default=2.85, help="Greedy association cost cutoff.")
    parser.add_argument("--weight-centroid", type=float, default=1.0, help="Weight for normalized 2D centroid distance.")
    parser.add_argument("--weight-foot", type=float, default=1.0, help="Weight for normalized foot-point distance.")
    parser.add_argument("--weight-iou", type=float, default=0.7, help="Weight for 1-IoU bbox term.")
    parser.add_argument("--weight-3d", type=float, default=0.0, help="Set to 0 for this 2D-only short-clip smoke stage.")
    return parser.parse_args()


def decode_clip(video_path, output_dir, start_time, duration, frame_stride, max_frames):
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if max_frames < 1:
        raise ValueError("--max-frames must be >= 1")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open source video {}".format(video_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = max(0, int(math.floor(max(0.0, start_time) * fps))) if fps > 0 else 0
    end_frame = start_frame + int(math.ceil(max(0.0, duration) * fps)) if fps > 0 else total_frames
    if total_frames:
        end_frame = min(total_frames, end_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    decoded = []
    source_frame = start_frame
    while len(decoded) < max_frames:
        if total_frames and source_frame >= end_frame:
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if (source_frame - start_frame) % frame_stride == 0:
            clip_index = len(decoded)
            path = output_dir / "frame_{:03d}.jpg".format(clip_index)
            cv2.imwrite(str(path), frame_bgr)
            timestamp = float(source_frame / fps) if fps > 0 else None
            decoded.append({
                "clip_frame_index": clip_index,
                "source_frame_index": int(source_frame),
                "timestamp_seconds": timestamp,
                "frame_path": str(path),
            })
        source_frame += 1
    cap.release()
    return {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": total_frames,
        "width": width,
        "height": height,
        "start_time_seconds": start_time,
        "duration_seconds": duration,
        "start_frame_index": start_frame,
        "end_frame_index_exclusive": end_frame,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "decoded_frame_count": len(decoded),
        "decoded_frames_dir": str(output_dir),
        "decoded_frames": decoded,
    }


def flatten_detections(detections_by_frame):
    detections = []
    for frame_index in sorted(detections_by_frame):
        detections.extend(detections_by_frame[frame_index].get("detections", []))
    return detections


def render_tracklets(detections_by_frame, output_dir, fps, mask_alpha, line_width):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rendered_frames = []
    frame_records = []
    for frame_index in sorted(detections_by_frame):
        frame_record = detections_by_frame[frame_index]
        frame_path = Path(frame_record.get("original_path", ""))
        if not frame_path.exists():
            frame_records.append({"frame_index": frame_index, "status": "skipped", "reason": "missing_filtered_original_frame"})
            continue
        output_path = frames_dir / "frame_{:03d}_tracking_overlay.png".format(frame_index)
        rendered = render_frame(frame_path, frame_index, frame_record.get("detections", []), output_path, mask_alpha, line_width)
        rendered_frames.append(Path(rendered))
        frame_records.append({
            "frame_index": frame_index,
            "status": "rendered",
            "source_frame_path": str(frame_path),
            "rendered_frame_path": rendered,
            "detections": len(frame_record.get("detections", [])),
        })
    mp4_path = output_dir / "tracking_overlay.mp4"
    gif_path = output_dir / "tracking_overlay.gif"
    mp4 = None
    mp4_error = None
    if rendered_frames:
        try:
            mp4 = write_mp4(rendered_frames, mp4_path, fps)
        except Exception as exc:
            mp4_error = {"type": exc.__class__.__name__, "message": str(exc)}
        gif = write_gif(rendered_frames, gif_path, fps)
    else:
        gif = None
    return {
        "frames_dir": str(frames_dir),
        "rendered_frames": [str(path) for path in rendered_frames],
        "frame_records": frame_records,
        "mp4": mp4,
        "mp4_error": mp4_error,
        "gif": gif,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / args.run_id / "short_clip_tracking_smoke"
    decoded_dir = output_dir / "decoded_frames"
    filtered_dir = output_dir / "sam3_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "status": "running",
        "stage": "short_clip_tracking_smoke",
        "run_id": args.run_id,
        "inputs": {"video": str(args.video), "sam3_repo": str(args.repo)},
        "known_limitations": KNOWN_LIMITATIONS,
    }
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
            raise RuntimeError("SAM 3 filtered short-clip stage failed: {}".format(sam3_result.get("error")))

        detections_by_frame = load_detections(filtered_dir, {})
        valid_frame_indices = set(range(len(frame_paths)))
        detections_by_frame = {idx: row for idx, row in detections_by_frame.items() if idx in valid_frame_indices}
        tracks, match_events = assign_tracklets(detections_by_frame, args)
        tracklets = [track_summary(track_id, members) for track_id, members in sorted(tracks.items())]
        detections = flatten_detections(detections_by_frame)
        render_artifacts = render_tracklets(detections_by_frame, output_dir, output_fps, args.mask_alpha, args.line_width)

        metadata.update({
            "status": "complete",
            "inputs": {
                "video": str(args.video),
                "sam3_repo": str(args.repo),
            },
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
                "matching": "greedy consecutive-frame 2D association",
                "max_centroid_distance": args.max_centroid_distance,
                "max_foot_distance": args.max_foot_distance,
                "min_iou": args.min_iou,
                "max_match_cost": args.max_match_cost,
            },
            "decode": decode,
            "sam3_filtering": sam3_result,
            "detections": detections,
            "tracklets": tracklets,
            "match_events": match_events,
            "render": render_artifacts,
            "artifacts": {
                "decoded_frames_dir": str(decoded_dir),
                "filtered_sam3_dir": str(filtered_dir),
                "rendered_frames_dir": render_artifacts["frames_dir"],
                "tracking_overlay_mp4": render_artifacts["mp4"],
                "tracking_overlay_gif": render_artifacts["gif"],
            },
        })
        summary_status = "complete"
        exit_code = 0
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "error": {"type": exc.__class__.__name__, "message": str(exc), "traceback": traceback.format_exc()},
        })
        summary_status = "failed"
        exit_code = 1

    write_json(output_dir / "tracking_metadata.json", metadata)
    counts = {
        "decoded_frames": metadata.get("decode", {}).get("decoded_frame_count", 0),
        "frames_rendered": len(metadata.get("render", {}).get("rendered_frames", [])),
        "detections": len(metadata.get("detections", [])),
        "tracklets": len(metadata.get("tracklets", [])),
        "multi_frame_tracklets": sum(1 for t in metadata.get("tracklets", []) if t.get("frames_covered", 0) > 1),
        "match_events": len(metadata.get("match_events", [])),
    }
    summary = {
        "status": summary_status,
        "stage": "short_clip_tracking_summary",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "tracking_metadata.json"),
        "counts": counts,
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
