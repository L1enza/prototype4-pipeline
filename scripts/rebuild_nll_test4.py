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


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild nll_test4 artifacts from the source video.")
    parser.add_argument("--video", default="data/videos/nll_test4.mp4")
    parser.add_argument("--start-time", type=float, default=20.0, help="Segment start (s). Matches the segment_20s_10s output tag.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=60, help="10 s at 30 fps with stride 5 is 60 frames.")
    parser.add_argument("--detector", choices=["yolo", "sam3"], default="yolo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vision-model", default="gemma3:12b", help="Ollama model that reads jersey numbers.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Let YOLO and ResNet-18 fetch weights if not cached.")
    parser.add_argument("--force", action="append", default=[], metavar="STAGE", help="Re-run this stage even if its output exists. Repeatable.")
    parser.add_argument("--stop-after", default=None, metavar="STAGE", help="Stop after this stage.")
    parser.add_argument("--check", action="store_true", help="Report what each stage needs and has, then exit.")
    return parser.parse_args()


def stages(args):
    py = sys.executable
    weights = ["--allow-download-weights"] if args.allow_download_weights else []
    return [
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
            "why": "Split tracks into two shirt-colour clusters (colour only).",
            "output": RUN + "/team_assignment_demo_v2/track_team_assignments_v2.json",
            "cmd": [py, "scripts/run_team_assignment_and_polygon_demo_v2.py", "--config", "configs/nll_test4_team_assignment_v2.json",
                    "--video", args.video, "--embedding-backend", "none"],
        },
        {
            "name": "teams_hybrid",
            "why": "Same split with an appearance embedding added; the jersey stage reads this one.",
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
                    "--allow-network-vision", "--vision-endpoint", OLLAMA_ENDPOINT, "--vision-model", args.vision_model],
        },
    ]


def ollama_has(model):
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
            names = {row.get("name") for row in json.load(response).get("models", [])}
    except OSError:
        return None
    return model in names or "{}:latest".format(model) in names


def team_confirmation_status():
    config = json.loads((PROJECT_ROOT / JERSEY_CONFIG).read_text(encoding="utf-8"))
    inputs = config["inputs"]
    assignments_path = PROJECT_ROOT / inputs["team_assignments"]
    if not assignments_path.exists():
        return False, "team assignments not built yet"
    from prototype4_pipeline.integrations.track_jersey_inference import load_team_assignments

    assignments = load_team_assignments(assignments_path)
    result = load_confirmed_mapping(PROJECT_ROOT / inputs["team_mapping_confirmation"], assignments, {"TOR", "OSH"})
    return result["status"] == "confirmed", result["status"]


def preflight(args):
    problems = []
    video = PROJECT_ROOT / args.video if not Path(args.video).is_absolute() else Path(args.video)
    if not video.exists():
        problems.append("Source video missing: copy nll_test4.mp4 to {}".format(video))
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
    plan = stages(args)
    names = [stage["name"] for stage in plan]
    for name in args.force + ([args.stop_after] if args.stop_after else []):
        if name not in names:
            raise SystemExit("Unknown stage {!r}; stages are: {}".format(name, ", ".join(names)))
    video, problems = preflight(args)

    if args.check:
        print("Stages (output exists = will be skipped):")
        for stage in plan:
            done = (PROJECT_ROOT / stage["output"]).exists()
            print("  [{}] {:<15} {}".format("x" if done else " ", stage["name"], stage["why"]))
        confirmed, status = team_confirmation_status()
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
                confirmed, status = team_confirmation_status()
                if not confirmed:
                    print("\n[stop] Team mapping is {}. Before any names are attached, a person must say which".format(status))
                    print("       colour cluster is which team. TOR wears white, OSH dark maroon.")
                    print("       1. Watch {}/team_assignment_demo_v2_hybrid/team_assignment_overlay_v2_hybrid.mp4".format(RUN))
                    print("       2. python scripts/confirm_team_mapping.py            (shows which label looks light or dark)")
                    print("       3. python scripts/confirm_team_mapping.py --team-a TOR --team-b OSH --confirmed-by <you> --notes \"<what you saw>\"")
                    print("          (swap TOR/OSH if team_a is the maroon side), then re-run this script.")
                    return 2
            print("[run]  {}: {}".format(stage["name"], stage["why"]), flush=True)
            result = subprocess.run(stage["cmd"], cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print("[fail] {} exited with {}. Fix it, then re-run; finished stages are skipped.".format(stage["name"], result.returncode))
                return result.returncode
            if not output.exists():
                print("[fail] {} finished but did not write {}".format(stage["name"], stage["output"]))
                return 1
        if stage["name"] == args.stop_after:
            print("[stop] --stop-after {}".format(stage["name"]))
            return 0
    print("\nAll stages complete. Jersey results: {}/track_level_jersey_inference/".format(RUN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
