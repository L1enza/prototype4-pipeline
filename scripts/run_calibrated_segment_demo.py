#!/usr/bin/env python3
"""Run a calibrated short-segment tracking, projection, heatmap, and demo bundle."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM3_REPO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a calibrated short segment demo from existing homography correspondences.")
    parser.add_argument("--video", required=True, help="Source video path.")
    parser.add_argument("--run-id", required=True, help="Run id, e.g. nll_test4.")
    parser.add_argument("--start-time", type=float, required=True, help="Segment start time in seconds.")
    parser.add_argument("--duration", type=float, required=True, help="Segment duration in seconds.")
    parser.add_argument("--frame-stride", type=int, default=5, help="Tracking frame stride.")
    parser.add_argument("--max-frames", type=int, default=45, help="Maximum decoded tracking frames.")
    parser.add_argument("--device", default="cuda", help="Torch device for SAM 3 tracking smoke path.")
    parser.add_argument("--homography-config", required=True, help="Manual homography config JSON for this camera view.")
    parser.add_argument("--field-template", required=True, help="Top-down field template image.")
    parser.add_argument("--output-tag", required=True, help="Output tag under outputs/<run-id>/calibrated_segment_demos/.")
    parser.add_argument("--repo", default=DEFAULT_SAM3_REPO, help="Local SAM 3 repo path.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32", help="SAM 3 smoke dtype.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Allow SAM 3 weight downloads if not cached.")
    parser.add_argument("--disable-fused-kernels", action=argparse.BooleanOptionalAction, default=True, help="Disable SAM 3 fused kernels in smoke tracking.")
    parser.add_argument("--fps", type=float, default=6.0, help="Output GIF/MP4 FPS for heatmap/side-by-side views.")
    parser.add_argument("--trail-length", type=int, default=7, help="Side-by-side field trail length.")
    return parser.parse_args()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(name, cmd, output_dir):
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record = {
        "name": name,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    write_json(output_dir / "{}_command.json".format(name), record)
    if result.returncode != 0:
        raise RuntimeError("{} failed with returncode {}; see {}".format(name, result.returncode, output_dir / "{}_command.json".format(name)))
    return record


def copy_if_exists(src, dst):
    if Path(src).exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)
    return None


def path_status(path):
    return "written" if path and Path(path).exists() else "missing"


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / "outputs" / args.run_id / "calibrated_segment_demos" / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    homography_config = Path(args.homography_config)
    field_template = Path(args.field_template)
    if not homography_config.is_absolute():
        homography_config = PROJECT_ROOT / homography_config
    if not field_template.is_absolute():
        field_template = PROJECT_ROOT / field_template

    warnings = [
        "Existing homography is reused for this whole segment; verify manually if the broadcast camera cuts, pans, or zooms.",
        "This is a short-clip calibrated demo, not full-game inference.",
    ]
    if args.duration > 3.0:
        warnings.append("Segment duration is longer than the original 3-second calibration check; camera-view validity should be visually inspected.")

    commands = []
    tracking_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_short_clip_tracking_stabilized.py"),
        "--video", str(args.video),
        "--run-id", args.run_id,
        "--start-time", str(args.start_time),
        "--duration", str(args.duration),
        "--frame-stride", str(args.frame_stride),
        "--max-frames", str(args.max_frames),
        "--max-track-age", "5",
        "--min-track-length-to-render", "3",
        "--device", args.device,
        "--repo", args.repo,
        "--dtype", args.dtype,
        "--output-dir", str(output_dir),
    ]
    if args.allow_download_weights:
        tracking_cmd.append("--allow-download-weights")
    tracking_cmd.append("--disable-fused-kernels" if args.disable_fused_kernels else "--no-disable-fused-kernels")
    commands.append(run_command("tracking", tracking_cmd, output_dir))

    tracking_metadata = output_dir / "tracking_metadata.json"
    calibration_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_field_calibration_smoke.py"),
        "--run-id", args.run_id,
        "--config", str(homography_config),
        "--tracking-metadata", str(tracking_metadata),
        "--output-dir", str(output_dir),
    ]
    commands.append(run_command("calibration_projection", calibration_cmd, output_dir))
    calibration_summary_alias = copy_if_exists(output_dir / "calibration_summary.json", output_dir / "calibration_projection_summary.json")

    heatmap_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_heatmap_smoke.py"),
        "--run-id", args.run_id,
        "--projected-points", str(output_dir / "projected_player_points.json"),
        "--field-template", str(field_template),
        "--calibration-metadata", str(output_dir / "calibration_metadata.json"),
        "--output-dir", str(output_dir),
        "--gif-fps", str(args.fps),
        "--mp4-fps", str(args.fps),
    ]
    commands.append(run_command("heatmap", heatmap_cmd, output_dir))

    side_by_side_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "render_side_by_side_field_tracking_smoke.py"),
        "--run-id", args.run_id,
        "--tracking-frames-dir", str(output_dir / "frames"),
        "--tracking-metadata", str(tracking_metadata),
        "--projected-points", str(output_dir / "projected_player_points.json"),
        "--field-template", str(field_template),
        "--heatmap", str(output_dir / "heatmap_all_players_overlay.png"),
        "--output-dir", str(output_dir),
        "--fps", str(args.fps),
        "--trail-length", str(args.trail_length),
    ]
    commands.append(run_command("side_by_side", side_by_side_cmd, output_dir))

    tracking_summary = load_json(output_dir / "tracking_summary.json")
    calibration_summary = load_json(output_dir / "calibration_summary.json")
    heatmap_summary = load_json(output_dir / "heatmap_summary.json")
    side_by_side_summary = load_json(output_dir / "side_by_side_summary.json")
    artifacts = {
        "tracking_overlay_gif": str(output_dir / "tracking_overlay.gif"),
        "tracking_overlay_mp4": str(output_dir / "tracking_overlay.mp4"),
        "projected_tracks_topdown": str(output_dir / "projected_tracks_topdown.png"),
        "heatmap_by_frame_gif": str(output_dir / "heatmap_by_frame.gif"),
        "heatmap_by_frame_mp4": str(output_dir / "heatmap_by_frame.mp4"),
        "heatmap_all_players_overlay": str(output_dir / "heatmap_all_players_overlay.png"),
        "side_by_side_tracking_field_gif": str(output_dir / "side_by_side_tracking_field.gif"),
        "side_by_side_tracking_field_mp4": str(output_dir / "side_by_side_tracking_field.mp4"),
        "tracking_summary": str(output_dir / "tracking_summary.json"),
        "calibration_projection_summary": calibration_summary_alias or str(output_dir / "calibration_projection_summary.json"),
        "heatmap_summary": str(output_dir / "heatmap_summary.json"),
        "side_by_side_summary": str(output_dir / "side_by_side_summary.json"),
    }
    demo_summary = {
        "status": "complete",
        "stage": "calibrated_segment_demo",
        "run_id": args.run_id,
        "output_tag": args.output_tag,
        "output_dir": str(output_dir),
        "inputs": {
            "video": str(args.video),
            "homography_config": str(homography_config),
            "field_template": str(field_template),
        },
        "parameters": {
            "start_time": args.start_time,
            "duration": args.duration,
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "device": args.device,
            "dtype": args.dtype,
        },
        "warnings": warnings,
        "counts": {
            "tracking": tracking_summary.get("counts", {}),
            "calibration_projection": calibration_summary.get("counts", {}),
            "heatmap": heatmap_summary.get("counts", {}),
            "side_by_side": side_by_side_summary.get("counts", {}),
        },
        "artifact_status": {key: path_status(value) for key, value in artifacts.items()},
        "artifacts": artifacts,
        "commands": [record["name"] for record in commands],
    }
    write_json(output_dir / "demo_summary.json", demo_summary)
    print(json.dumps(demo_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
