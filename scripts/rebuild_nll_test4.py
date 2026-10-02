#!/usr/bin/env python3
"""Rebuild every nll_test4 artifact the jersey stage needs, from the source video.

Runs each stage as its own script, in dependency order, and skips a stage whose
output already exists (pass --force to redo it). The run stops before the final
jersey stage until a person has confirmed which colour cluster is which team.

    python scripts/rebuild_nll_test4.py --check     # what is missing, nothing runs
    python scripts/rebuild_nll_test4.py             # run everything that can run
"""

import argparse
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.team_mapping import load_confirmed_mapping  # noqa: E402

RUN = "outputs/nll_test4"
SEGMENT = RUN + "/calibrated_segment_demos/segment_20s_10s_calibrated"
TRACKING = SEGMENT + "/tracking_metadata.json"
JERSEY_CONFIG = "configs/nll_test4_track_jersey_inference.json"
OLLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"
ROSTERS = ["data/rosters/nll_2026/toronto_rock_roster.json", "data/rosters/nll_2026/oshawa_firewolves_roster.json"]
# Which team assignment the jersey stage reads. Colour-only is the default: the hybrid
# ResNet-18 features have not been checked on real footage.
TEAM_SOURCES = {
    "colour": {
        "assignments": RUN + "/team_assignment_demo_v2/track_team_assignments_v2.json",
        "overlay": RUN + "/team_assignment_demo_v2/team_assignment_overlay_v2.mp4",
    },
    "hybrid": {
        "assignments": RUN + "/team_assignment_demo_v2_hybrid/track_team_assignments_v2.json",
        "overlay": RUN + "/team_assignment_demo_v2_hybrid/team_assignment_overlay_v2_hybrid.mp4",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild nll_test4 artifacts from the source video.")
    parser.add_argument("--video", default="data/videos/nll_test4.mp4")
    parser.add_argument("--start-time", type=float, default=20.0, help="Segment start (s). Matches the segment_20s_10s output tag.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None, help="Default: enough frames to cover --duration at the video's own frame rate.")
    parser.add_argument("--detector", choices=["yolo", "sam3"], default="yolo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-model", default="gemma3:12b", help="Ollama model that reads jersey numbers.")
    parser.add_argument("--vision-timeout", type=float, default=600.0, help="Seconds per jersey read; gemma3:12b took 58-100 s per image on CPU.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Let YOLO and ResNet-18 fetch weights if not cached.")
    parser.add_argument("--team-source", choices=sorted(TEAM_SOURCES), default="colour",
                        help="Team assignment the jersey stage uses; hybrid adds unvalidated ResNet-18 features.")
    parser.add_argument("--force", action="append", default=[], metavar="STAGE", help="Re-run this stage even if its output exists. Repeatable.")
    parser.add_argument("--stop-after", default=None, metavar="STAGE", help="Stop after this stage.")
    parser.add_argument("--check", action="store_true", help="Report what each stage needs and has, then exit.")
    return parser.parse_args()


def stages(args):
    py = sys.executable
    weights = ["--allow-download-weights"] if args.allow_download_weights else []
    plan = [
        {
            "name": "roster",
            "why": "Jersey number -> player lookup for TOR and OSH.",
            "output": "outputs/roster_metadata_validation/jersey_number_lookup.json",
            # OSH lists #77 twice; ambiguous numbers never receive a name downstream.
            "cmd": [py, "scripts/validate_roster_metadata.py", *sum([["--roster-json", r] for r in ROSTERS], []),
                    "--output-dir", "outputs/roster_metadata_validation", "--allow-team-duplicates"],
        },
        {
            "name": "tracking",
            "why": "Find, filter and track players, then project feet onto the field.",
            "output": SEGMENT + "/projected_player_points.json",
            "warnings_from": SEGMENT + "/calibration_metadata.json",
            "cmd": [py, "scripts/run_calibrated_segment_demo.py", "--video", args.video, "--run-id", "nll_test4",
                    "--start-time", str(args.start_time), "--duration", str(args.duration),
                    "--frame-stride", str(args.frame_stride), "--max-frames", str(args.max_frames),
                    "--detector", args.detector, "--device", args.device,
                    "--homography-config", "configs/nll_test4_homography_points.json",
                    "--field-template", "assets/field_templates/nll_field_topdown.png",
                    "--output-tag", "segment_20s_10s_calibrated", *weights],
        },
        {
            "name": "crops",
            "why": "Full-body and torso crops per track from the full-resolution video.",
            "output": RUN + "/jersey_ocr_clean_crops/clean_crop_metadata.json",
            "cmd": [py, "scripts/extract_clean_track_crops_for_jersey_ocr.py", "--video", args.video, "--tracking-metadata", TRACKING],
        },
        {
            "name": "teams_color",
            "why": "Split tracks into two teams by shirt colour.",
            "output": RUN + "/team_assignment_demo_v2/track_team_assignments_v2.json",
            "cmd": [py, "scripts/run_team_assignment_and_polygon_demo_v2.py", "--config", "configs/nll_test4_team_assignment_v2.json",
                    "--video", args.video, "--embedding-backend", "none"],
        },
        {
            "name": "teams_hybrid",
            "why": "Same split with ResNet-18 appearance features added (--team-source hybrid).",
            "output": RUN + "/team_assignment_demo_v2_hybrid/track_team_assignments_v2.json",
            "cmd": [py, "scripts/run_team_assignment_and_polygon_demo_v2.py", "--config", "configs/nll_test4_team_assignment_v2.json",
                    "--video", args.video, "--embedding-backend", "auto",
                    "--team-output-dir", RUN + "/team_assignment_demo_v2_hybrid",
                    "--polygon-output-dir", RUN + "/team_polygon_demo_v2_hybrid", *weights],
        },
        {
            "name": "visibility",
            "why": "Score each crop for sharpness, contrast and overlap.",
            "output": RUN + "/jersey_crop_visibility_audit/crop_visibility_predictions.json",
            "cmd": [py, "scripts/audit_jersey_crop_visibility.py"],
        },
        {
            "name": "number_regions",
            "why": "Enlarged, sharpened number-region views of the best crops.",
            "output": RUN + "/enhanced_number_regions/enhanced_number_region_manifest.json",
            "cmd": [py, "scripts/generate_enhanced_number_regions.py", "--force"],
        },
        {
            "name": "jersey",
            "why": "Read numbers per frame with the local vision model and attach names.",
            "output": RUN + "/track_level_jersey_inference/track_jersey_predictions.json",
            "needs_team_confirmation": True,
            "cmd": [py, "scripts/run_track_level_jersey_inference.py", "--config", JERSEY_CONFIG,
                    "--allow-network-vision", "--vision-endpoint", OLLAMA_ENDPOINT, "--vision-model", args.vision_model,
                    "--vision-timeout", str(args.vision_timeout),
                    "--team-assignments", TEAM_SOURCES[args.team_source]["assignments"]],
        },
    ]
    if args.team_source != "hybrid":
        plan = [stage for stage in plan if stage["name"] != "teams_hybrid"]
    return plan


def stage_warnings(stage):
    path = PROJECT_ROOT / stage["warnings_from"] if stage.get("warnings_from") else None
    if not path or not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("warnings") or []


def ollama_has(model):
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
            names = {row.get("name") for row in json.load(response).get("models", [])}
    except OSError:
        return None
    return model in names or "{}:latest".format(model) in names


def team_confirmation_status(assignments_file):
    config = json.loads((PROJECT_ROOT / JERSEY_CONFIG).read_text(encoding="utf-8"))
    inputs = config["inputs"]
    assignments_path = PROJECT_ROOT / assignments_file
    if not assignments_path.exists():
        return False, "team assignments not built yet"
    from prototype4_pipeline.integrations.track_jersey_inference import load_team_assignments

    assignments = load_team_assignments(assignments_path)
    result = load_confirmed_mapping(PROJECT_ROOT / inputs["team_mapping_confirmation"], assignments, {"TOR", "OSH"})
    return result["status"] == "confirmed", result["status"]


def frame_budget(fps, total_frames, start_time, duration, stride):
    """Frames needed to sample the whole segment at this video's real frame rate.

    A fixed cap silently truncates the segment: 60 frames at stride 5 covers 10 s
    of 30 fps video but only 5 s of 59.94 fps broadcast video.
    """
    if not fps or fps <= 0:
        raise ValueError("video reports no frame rate, so the segment length in frames is unknown")
    start = int(math.floor(start_time * fps))
    end = start + int(math.ceil(duration * fps))
    if total_frames and start >= total_frames:
        raise ValueError("segment starts at {:.1f} s but the video is only {:.1f} s long".format(start_time, total_frames / fps))
    if total_frames:
        end = min(end, total_frames)
    return max(1, int(math.ceil((end - start) / float(stride))))


def resolve_frame_budget(args, video):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV cannot open the video")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if args.max_frames is None:
        args.max_frames = frame_budget(fps, total, args.start_time, args.duration, args.frame_stride)
    return "{}x{} at {:.2f} fps; tracking {:.0f}-{:.0f} s as {} frames (every {}th)".format(
        width, height, fps, args.start_time, args.start_time + args.duration, args.max_frames, args.frame_stride
    )


def preflight(args):
    problems = []
    video = PROJECT_ROOT / args.video if not Path(args.video).is_absolute() else Path(args.video)
    if not video.exists():
        problems.append("Source video missing: copy nll_test4.mp4 to {}".format(video))
    else:
        try:
            print("Video: " + resolve_frame_budget(args, video))
        except ValueError as exc:
            problems.append("Cannot plan tracking frames: {}".format(exc))
    weights = PROJECT_ROOT / ".cache" / "models" / "yolo11m-seg.pt"
    if args.detector == "yolo" and not weights.exists() and not args.allow_download_weights:
        problems.append("YOLO weights not cached: add --allow-download-weights (about 45 MB, once)")
    has_model = ollama_has(args.vision_model)
    if has_model is None:
        problems.append("Ollama is not running: start it with `ollama serve`")
    elif not has_model:
        problems.append("Ollama model {} not pulled: run `ollama pull {}`".format(args.vision_model, args.vision_model))
    return video, problems


def main():
    args = parse_args()
    _video, problems = preflight(args)
    plan = stages(args)
    names = [stage["name"] for stage in plan]
    for name in args.force + ([args.stop_after] if args.stop_after else []):
        if name not in names:
            raise SystemExit("Unknown stage {!r}; stages are: {}".format(name, ", ".join(names)))

    if args.check:
        print("Stages (output exists = will be skipped):")
        for stage in plan:
            done = (PROJECT_ROOT / stage["output"]).exists()
            print("  [{}] {:<15} {}".format("x" if done else " ", stage["name"], stage["why"]))
        confirmed, status = team_confirmation_status(TEAM_SOURCES[args.team_source]["assignments"])
        print("Team mapping confirmation: {}".format(status))
        print("Problems:" if problems else "No problems found.")
        for problem in problems:
            print("  - " + problem)
        return 1 if problems else 0

    if problems:
        print("Cannot start:")
        for problem in problems:
            print("  - " + problem)
        return 1

    for stage in plan:
        output = PROJECT_ROOT / stage["output"]
        if output.exists() and stage["name"] not in args.force:
            print("[skip] {} ({} exists)".format(stage["name"], stage["output"]))
        else:
            if stage.get("needs_team_confirmation"):
                confirmed, status = team_confirmation_status(TEAM_SOURCES[args.team_source]["assignments"])
                if not confirmed:
                    print("\n[stop] Team mapping is {}. Before any names are attached, a person must say which".format(status))
                    print("       colour cluster is which team. TOR wears white, OSH dark maroon.")
                    teams = TEAM_SOURCES[args.team_source]
                    print("       1. Watch {}".format(teams["overlay"]))
                    print("       2. python scripts/confirm_team_mapping.py --assignments {}".format(teams["assignments"]))
                    print("          (shows which label looks light or dark, and warns if the split looks wrong)")
                    print("       3. python scripts/confirm_team_mapping.py --assignments {} --team-a TOR --team-b OSH \\".format(teams["assignments"]))
                    print("            --confirmed-by <you> --notes \"<what you saw>\"")
                    print("          (swap TOR/OSH if team_a is the maroon side), then re-run this script.")
                    return 2
            print("[run]  {}: {}".format(stage["name"], stage["why"]), flush=True)
            result = subprocess.run(stage["cmd"], cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print("[fail] {} exited with {}. Fix it, then re-run; finished stages are skipped.".format(stage["name"], result.returncode))
                if output.exists():
                    print("       It still wrote {}, so re-run with --force {} or it will be skipped.".format(stage["output"], stage["name"]))
                return result.returncode
            if not output.exists():
                print("[fail] {} finished but did not write {}".format(stage["name"], stage["output"]))
                return 1
        for warning in stage_warnings(stage):
            print("[warn] {}: {}".format(stage["name"], warning))
        if stage["name"] == args.stop_after:
            print("[stop] --stop-after {}".format(stage["name"]))
            return 0
    print("\nAll stages complete. Jersey results: {}/track_level_jersey_inference/".format(RUN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
