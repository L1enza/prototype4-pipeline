#!/usr/bin/env python3
"""Run track-level jersey-number inference for nll_test4 crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.track_jersey_inference import (  # noqa: E402
    DEFAULT_CONFIG,
    project_path,
    read_json,
    run_track_level_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate jersey-number evidence at the track level.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vision-backend", choices=["auto", "openai", "none", "existing_ocr"], default=None)
    parser.add_argument(
        "--allow-network-vision",
        action="store_true",
        help="Allow OpenAI-compatible vision requests (needs the configured API key unless api_key_env is null).",
    )
    parser.add_argument("--vision-model", default=None, help="Override vision_backend.model.")
    parser.add_argument(
        "--vision-endpoint",
        default=None,
        help="Override vision_backend.endpoint; a localhost endpoint (e.g. Ollama) runs without an API key.",
    )
    parser.add_argument(
        "--vision-timeout",
        type=float,
        default=None,
        help="Override vision_backend.timeout_seconds. A 12B model on CPU needs 60-100 s per image.",
    )
    parser.add_argument(
        "--team-assignments",
        default=None,
        help="Override inputs.team_assignments; the team mapping confirmation must be made for this file.",
    )
    parser.add_argument("--max-source-frames-per-track", type=int, default=None)
    return parser.parse_args()


def vision_request_counts(predictions: list[dict]) -> dict:
    """Count vision requests and failures, which the stage summary does not report.

    A failed or timed-out read leaves its frame without a number, so a run where
    every request failed looks exactly like one where nothing was legible.
    """
    counts = {"requests": 0, "errors": 0, "timeouts": 0, "missing_views": 0}
    for prediction in predictions:
        result = prediction.get("vision_model_result") or {}
        if result.get("mode") == "per_frame":
            frames = result.get("frames", [])
        elif result.get("status") not in (None, "not_run"):
            frames = [result]
        else:
            frames = []
        for frame in frames:
            status = frame.get("status")
            if status == "missing_view":
                counts["missing_views"] += 1
                continue
            counts["requests"] += 1
            if status == "error":
                counts["errors"] += 1
                if "timed out" in str(frame.get("error", "")).lower() or "timeout" in str(frame.get("error", "")).lower():
                    counts["timeouts"] += 1
    return counts


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = read_json(config_path)
    if args.output_dir:
        config.setdefault("outputs", {})["output_dir"] = args.output_dir
    if args.vision_backend:
        config.setdefault("vision_backend", {})["name"] = args.vision_backend
    if args.allow_network_vision:
        config.setdefault("vision_backend", {})["allow_network"] = True
    if args.vision_model:
        config.setdefault("vision_backend", {})["model"] = args.vision_model
    if args.vision_endpoint:
        backend = config.setdefault("vision_backend", {})
        backend["endpoint"] = args.vision_endpoint
        if "localhost" in args.vision_endpoint or "127.0.0.1" in args.vision_endpoint:
            backend["api_key_env"] = None
    if args.vision_timeout is not None:
        if args.vision_timeout <= 0:
            raise ValueError("--vision-timeout must be positive")
        config.setdefault("vision_backend", {})["timeout_seconds"] = args.vision_timeout
    if args.team_assignments:
        config.setdefault("inputs", {})["team_assignments"] = args.team_assignments
    if args.max_source_frames_per_track is not None:
        if args.max_source_frames_per_track < 1:
            raise ValueError("--max-source-frames-per-track must be positive")
        config.setdefault("selection", {})["max_source_frames_per_track"] = args.max_source_frames_per_track

    summary = run_track_level_inference(config)
    predictions = read_json(Path(summary["outputs"]["track_jersey_predictions"])).get("tracks", [])
    requests = vision_request_counts(predictions)
    summary_path = Path(summary["outputs"]["track_jersey_summary"])
    stored = read_json(summary_path)
    stored["vision_requests"] = requests
    summary_path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**summary["counts"], "vision_requests": requests}, indent=2, sort_keys=True))
    team_mapping = summary["team_mapping"]
    if team_mapping["status"] == "confirmed":
        print(f"team mapping: {team_mapping['mapping']} (confirmed by {team_mapping['confirmed_by']})")
    else:
        print(f"WARNING: team mapping {team_mapping['status']}; no numbers or names were assigned.")
        print(f"  {team_mapping['reason']}")
    print("outputs:")
    for key, value in summary["outputs"].items():
        print(f"  {key}: {value}")
    if requests["errors"]:
        timeout = config.get("vision_backend", {}).get("timeout_seconds", 60)
        print(
            f"WARNING: {requests['errors']} of {requests['requests']} vision requests failed "
            f"({requests['timeouts']} timed out at {timeout} s). Those frames gave no number, so "
            "unresolved tracks may only reflect failed reads; raise --vision-timeout and re-run."
        )
    if requests["requests"] and requests["errors"] == requests["requests"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

