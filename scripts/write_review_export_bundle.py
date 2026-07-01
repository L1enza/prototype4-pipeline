#!/usr/bin/env python3
"""Create a compact review bundle for the nll_test4 10-second demo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = "outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the curated nll_test4 10-second review export.")
    parser.add_argument("--output-dir", default="review_exports/nll_test4_10s_demo")
    parser.add_argument("--side-by-side-video", default=DEMO_DIR + "/side_by_side_tracking_field.mp4")
    parser.add_argument("--tracking-overlay-video", default=DEMO_DIR + "/tracking_overlay.mp4")
    parser.add_argument("--heatmap-video", default=DEMO_DIR + "/heatmap_by_frame.mp4")
    parser.add_argument("--debug-tracks", default="outputs/nll_test4/debug_tracks.json")
    parser.add_argument("--debug-tracks-summary", default="outputs/nll_test4/debug_tracks_summary.json")
    parser.add_argument("--pipeline-manifest", default="outputs/nll_test4/pipeline_run_manifest.json")
    parser.add_argument("--demo-summary", default=DEMO_DIR + "/demo_summary.json")
    parser.add_argument("--warn-size-mb", type=float, default=90.0)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def human_size(byte_count: int) -> str:
    value = float(byte_count)
    units = ["B", "KiB", "MiB", "GiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{} B".format(byte_count)


def readme_text(included_names: list[str]) -> str:
    videos = [name for name in included_names if name.endswith(".mp4")]
    return """# nll_test4 10-Second Review Export

This is a curated review export, not the full Prototype 4 `outputs/` directory.
It contains only the videos and JSON checkpoint files needed to review the same
10-second `nll_test4` segment.

## Videos

The included MP4s show the 10-second tracking, heatmap, and field-projection
demo. Included video files: {videos}.

## Debug evidence

`debug_tracks.json` is the consolidated per-track evidence export for this exact
segment. `debug_tracks_summary.json` reports coverage and source availability.
The stage-specific source JSON files remain in the main project outputs and are
not duplicated here.

## Current identity status

- Player names and resolved player identities are not assigned.
- Jersey OCR structure and crop selection are prepared, but a usable OCR engine
  was unavailable on the ECE environment for this checkpoint.
- No jersey number was accepted from the current OCR baseline.
- Roster metadata is validated for future constrained matching only.

## Geometry and scope

This demo uses a manually selected homography for the broadcast camera view.
Camera pan or zoom can make the static projection drift. This is a short review
segment, not full-game inference.

The Andrew 3D-world integration is planned and reviewed separately; it is not
part of this export.
""".format(videos=", ".join("`{}`".format(name) for name in videos))


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    sources = [
        ("side_by_side_tracking_field.mp4", resolve_path(args.side_by_side_video), True),
        ("tracking_overlay.mp4", resolve_path(args.tracking_overlay_video), False),
        ("heatmap_by_frame.mp4", resolve_path(args.heatmap_video), False),
        ("debug_tracks.json", resolve_path(args.debug_tracks), True),
        ("debug_tracks_summary.json", resolve_path(args.debug_tracks_summary), True),
        ("pipeline_run_manifest.json", resolve_path(args.pipeline_manifest), True),
        ("demo_summary.json", resolve_path(args.demo_summary), True),
    ]
    missing_required = [portable_path(path) for _name, path, required in sources if required and not path.is_file()]
    missing_optional = [portable_path(path) for _name, path, required in sources if not required and not path.is_file()]
    if missing_required:
        raise FileNotFoundError("Missing required review inputs: {}".format(", ".join(missing_required)))

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for output_name, source, _required in sources:
        if not source.is_file():
            continue
        destination = output_dir / output_name
        shutil.copy2(source, destination)
        copied.append(destination)

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_text([path.name for path in copied]), encoding="utf-8")
    copied.append(readme_path)

    threshold_bytes = int(args.warn_size_mb * 1024 * 1024)
    file_records = []
    oversized = []
    total_bytes = 0
    for path in sorted(copied, key=lambda item: item.name):
        size = path.stat().st_size
        total_bytes += size
        record = {
            "name": path.name,
            "path": portable_path(path),
            "size_bytes": size,
            "size_human": human_size(size),
            "over_warning_threshold": size > threshold_bytes,
        }
        file_records.append(record)
        if record["over_warning_threshold"]:
            oversized.append(record)

    result = {
        "status": "complete_with_size_warnings" if oversized else "complete",
        "output_dir": portable_path(output_dir),
        "files": file_records,
        "file_count": len(file_records),
        "missing_required_inputs": missing_required,
        "missing_optional_inputs": missing_optional,
        "total_size_bytes": total_bytes,
        "total_size_human": human_size(total_bytes),
        "warning_threshold_mb": args.warn_size_mb,
        "files_over_warning_threshold": oversized,
        "notes": [
            "This export contains curated review files only, not the full outputs directory.",
            "No OCR, training, roster resolution, or player-name assignment was run.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
