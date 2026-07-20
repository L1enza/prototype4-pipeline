#!/usr/bin/env python3
"""Run track-level jersey-number inference for nll_test4 crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
        help="Allow an OpenAI-compatible vision request if OPENAI_API_KEY is present.",
    )
    parser.add_argument("--max-source-frames-per-track", type=int, default=None)
    return parser.parse_args()


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
    if args.max_source_frames_per_track is not None:
        if args.max_source_frames_per_track < 1:
            raise ValueError("--max-source-frames-per-track must be positive")
        config.setdefault("selection", {})["max_source_frames_per_track"] = args.max_source_frames_per_track

    summary = run_track_level_inference(config)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print("outputs:")
    for key, value in summary["outputs"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

