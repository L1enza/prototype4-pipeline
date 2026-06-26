#!/usr/bin/env python3
"""Scout a new video and run a bounded stabilized tracking segment test."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/nll_test4.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="Scout a new NLL video and run one short stabilized tracking segment.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Source video path.")
    parser.add_argument("--run-id", default="nll_test4", help="Run id; outputs stay under outputs/<run_id>/.")
    parser.add_argument("--start-time", type=float, default=0.0, help="Segment start time in seconds.")
    parser.add_argument("--duration", type=float, default=3.0, help="Segment duration in seconds.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Tracking frame stride inside the segment.")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum tracking frames in the segment.")
    parser.add_argument("--device", default="cuda", help="Tracking device.")
    parser.add_argument("--segment-tag", default=None, help="Segment output tag. Defaults to segment_<start>s_<duration>s.")
    parser.add_argument("--repo", default="/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3", help="Local SAM 3 repo path.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="SAM 3 smoke dtype.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Pass through to stabilized tracking if needed.")
    parser.add_argument("--disable-fused-kernels", action=argparse.BooleanOptionalAction, default=True, help="Disable SAM 3 fused kernels in tracking smoke path.")
    parser.add_argument("--scout-sample-interval", type=float, default=5.0, help="Scout frame sample interval in seconds.")
    parser.add_argument("--scout-max-frames", type=int, default=24, help="Maximum scout frames for contact sheet.")
    parser.add_argument("--scout-only", action="store_true", help="Only write video scout outputs; do not run tracking.")
    parser.add_argument("--skip-scout", action="store_true", help="Skip scout sampling and only run the segment tracking test.")
    return parser.parse_args()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def segment_tag(start_time, duration):
    def fmt(value):
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return str(value).replace(".", "p")
    return "segment_{}s_{}s".format(fmt(start_time), fmt(duration))


def read_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video {}".format(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = float(frame_count / fps) if fps > 0 else None
    cap.release()
    return {
        "video_path": str(video_path),
        "fps": fps,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def sample_scout_frames(video_path, output_dir, sample_interval, max_frames):
    metadata = read_video_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video {}".format(video_path))
    frames_dir = output_dir / "sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps = metadata["fps"]
    frame_count = metadata["frame_count"]
    duration = metadata["duration_seconds"] or 0.0
    samples = []
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata: fps={} frame_count={}".format(fps, frame_count))
    sample_times = []
    t = 0.0
    while t <= duration and len(sample_times) < max_frames:
        sample_times.append(t)
        t += max(0.1, float(sample_interval))
    for index, timestamp in enumerate(sample_times):
        frame_index = min(frame_count - 1, max(0, int(round(timestamp * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        path = frames_dir / "scout_{:03d}_t{:07.2f}s_f{:06d}.jpg".format(index, timestamp, frame_index)
        cv2.imwrite(str(path), frame)
        samples.append({
            "sample_index": int(index),
            "timestamp_seconds": float(timestamp),
            "frame_index": int(frame_index),
            "frame_path": str(path),
        })
    cap.release()
    return metadata, samples


def create_contact_sheet(samples, output_path, thumb_width=320, columns=4):
    if not samples:
        return None
    thumbs = []
    for sample in samples:
        image = Image.open(sample["frame_path"]).convert("RGB")
        scale = thumb_width / float(image.width)
        thumb = image.resize((thumb_width, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)
        thumbs.append((sample, thumb))
    rows = int(math.ceil(len(thumbs) / float(columns)))
    label_h = 34
    cell_w = thumb_width
    cell_h = max(thumb.height for _sample, thumb in thumbs) + label_h
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, (sample, thumb) in enumerate(thumbs):
        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h
        sheet.paste(thumb, (x, y))
        label = "#{:02d} t={:.1f}s f={}".format(sample["sample_index"], sample["timestamp_seconds"], sample["frame_index"])
        draw.rectangle((x, y + thumb.height, x + cell_w, y + cell_h), fill=(20, 20, 20))
        draw.text((x + 8, y + thumb.height + 9), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def run_scout(args, scout_dir):
    video_path = Path(args.video)
    metadata, samples = sample_scout_frames(video_path, scout_dir, args.scout_sample_interval, args.scout_max_frames)
    contact_sheet = create_contact_sheet(samples, scout_dir / "sampled_frame_contact_sheet.jpg")
    scout_metadata = {
        "status": "complete",
        "stage": "new_video_segment_scout",
        "run_id": args.run_id,
        "video_metadata": metadata,
        "parameters": {
            "scout_sample_interval": args.scout_sample_interval,
            "scout_max_frames": args.scout_max_frames,
        },
        "samples": samples,
        "artifacts": {
            "sampled_frames_dir": str(scout_dir / "sampled_frames"),
            "sampled_frame_contact_sheet": contact_sheet,
        },
    }
    scout_summary = {
        "status": "complete",
        "stage": "new_video_segment_scout_summary",
        "run_id": args.run_id,
        "output_dir": str(scout_dir),
        "metadata": str(scout_dir / "scout_metadata.json"),
        "video_metadata": metadata,
        "counts": {"sampled_frames": len(samples)},
        "artifacts": scout_metadata["artifacts"],
    }
    write_json(scout_dir / "scout_metadata.json", scout_metadata)
    write_json(scout_dir / "scout_summary.json", scout_summary)
    return scout_summary


def run_tracking(args, tag):
    output_dir = PROJECT_ROOT / "outputs" / args.run_id / "segment_tests" / tag
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_short_clip_tracking_stabilized.py"),
        "--video",
        str(args.video),
        "--run-id",
        str(args.run_id),
        "--start-time",
        str(args.start_time),
        "--duration",
        str(args.duration),
        "--frame-stride",
        str(args.frame_stride),
        "--max-frames",
        str(args.max_frames),
        "--max-track-age",
        "5",
        "--min-track-length-to-render",
        "3",
        "--device",
        str(args.device),
        "--repo",
        str(args.repo),
        "--dtype",
        str(args.dtype),
        "--output-dir",
        str(output_dir),
    ]
    if args.allow_download_weights:
        cmd.append("--allow-download-weights")
    if args.disable_fused_kernels:
        cmd.append("--disable-fused-kernels")
    else:
        cmd.append("--no-disable-fused-kernels")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    write_json(output_dir / "segment_command.json", {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    })
    tracking_summary_path = output_dir / "tracking_summary.json"
    tracking_summary = load_json(tracking_summary_path) if tracking_summary_path.exists() else None
    if result.returncode != 0:
        raise RuntimeError("Stabilized tracking segment failed with returncode {}. See {}".format(result.returncode, output_dir / "segment_command.json"))
    return {
        "status": "complete",
        "segment_tag": tag,
        "output_dir": str(output_dir),
        "command_log": str(output_dir / "segment_command.json"),
        "tracking_summary": tracking_summary,
        "artifacts": {
            "tracking_overlay_gif": str(output_dir / "tracking_overlay.gif"),
            "tracking_overlay_mp4": str(output_dir / "tracking_overlay.mp4"),
            "frames_dir": str(output_dir / "frames"),
            "tracking_metadata": str(output_dir / "tracking_metadata.json"),
            "tracking_summary": str(tracking_summary_path),
        },
    }


def main():
    args = parse_args()
    tag = args.segment_tag or segment_tag(args.start_time, args.duration)
    scout_dir = PROJECT_ROOT / "outputs" / args.run_id / "segment_scout"
    summary = {
        "status": "running",
        "stage": "new_video_segment_scout_and_tracking",
        "run_id": args.run_id,
        "video": str(args.video),
        "segment_tag": tag,
    }
    try:
        if args.skip_scout:
            scout_summary = None
        else:
            scout_summary = run_scout(args, scout_dir)
        if args.scout_only:
            tracking_result = None
        else:
            tracking_result = run_tracking(args, tag)
        counts = {}
        if tracking_result and tracking_result.get("tracking_summary"):
            counts = tracking_result["tracking_summary"].get("counts", {})
        summary.update({
            "status": "complete",
            "scout": scout_summary,
            "tracking": tracking_result,
            "counts": counts,
        })
        exit_code = 0
    except Exception as exc:
        summary.update({"status": "failed", "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        exit_code = 1
    run_summary_path = scout_dir / "segment_scout_run_summary.json"
    write_json(run_summary_path, summary)
    summary["run_summary"] = str(run_summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
