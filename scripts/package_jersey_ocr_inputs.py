#!/usr/bin/env python3
"""Create a portable jersey-number OCR input package and zip archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "outputs/nll_test4/jersey_ocr_portable_package"
PACKAGE_NAME = "jersey_ocr_portable_package"
NINETY_MB = 90 * 1024 * 1024

REQUIRED_FILES = {
    "crop_visibility_predictions.json": "outputs/nll_test4/jersey_crop_visibility_audit/crop_visibility_predictions.json",
    "track_visibility_summary.json": "outputs/nll_test4/jersey_crop_visibility_audit/track_visibility_summary.json",
    "clean_crop_metadata.json": "outputs/nll_test4/jersey_ocr_clean_crops/clean_crop_metadata.json",
    "jersey_ocr_summary.json": "outputs/nll_test4/jersey_number_ocr_baseline/jersey_ocr_summary.json",
    "debug_tracks.json": "outputs/nll_test4/debug_tracks.json",
    "debug_tracks_summary.json": "outputs/nll_test4/debug_tracks_summary.json",
    "pipeline_run_manifest.json": "outputs/nll_test4/pipeline_run_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package jersey OCR inputs for a portable local run.")
    parser.add_argument(
        "--number-regions-dir",
        default="outputs/nll_test4/jersey_number_ocr_baseline/number_regions",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--progress-manifest",
        default="progress_outputs/nll_test4_checkpoint/json/jersey_ocr_package_manifest.json",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing package directory.")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def package_readme() -> str:
    return """# Portable Jersey OCR Package

This package contains pre-extracted jersey-number regions and metadata for the
`nll_test4` 10-second tracking demo. It contains no player-name assignments and
does not alter the ECE pipeline outputs.

## Mac setup

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python Pillow
```

## Run from a Prototype 4 checkout

Place the unzipped `jersey_ocr_portable_package/` beside the project `scripts/`
folder, then run:

```bash
python scripts/run_jersey_number_ocr_baseline.py \\
  --crop-metadata jersey_ocr_portable_package/clean_crop_metadata.json \\
  --visibility-predictions jersey_ocr_portable_package/crop_visibility_predictions.json \\
  --number-regions-dir jersey_ocr_portable_package/number_regions \\
  --output-dir jersey_ocr_local_results \\
  --engine tesseract
```

## Run using the packaged script

```bash
cd jersey_ocr_portable_package
python scripts/run_jersey_number_ocr_baseline.py \\
  --crop-metadata clean_crop_metadata.json \\
  --visibility-predictions crop_visibility_predictions.json \\
  --number-regions-dir number_regions \\
  --output-dir jersey_ocr_local_results \\
  --engine tesseract
```

Review `crop_ocr_predictions.json`, `track_jersey_number_predictions.json`, and
the generated contact sheets. Predictions remain track-level evidence only;
this stage does not assign player names or roster identities.
"""


def make_zip(package_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            archive.write(path, Path(PACKAGE_NAME) / path.relative_to(package_dir))


def main() -> int:
    args = parse_args()
    number_regions = project_path(args.number_regions_dir)
    output_dir = project_path(args.output_dir)
    progress_manifest = project_path(args.progress_manifest)
    required_sources = {name: project_path(value) for name, value in REQUIRED_FILES.items()}
    runner_source = PROJECT_ROOT / "scripts" / "run_jersey_number_ocr_baseline.py"

    missing = [str(path) for path in [number_regions, runner_source, *required_sources.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required package inputs:\n" + "\n".join(missing))
    if not number_regions.is_dir():
        raise NotADirectoryError(number_regions)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                "Package output already exists; use --force to replace it: {}".format(output_dir)
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    shutil.copytree(number_regions, output_dir / "number_regions")
    for destination_name, source in required_sources.items():
        shutil.copy2(source, output_dir / destination_name)
    packaged_runner = output_dir / "scripts" / runner_source.name
    packaged_runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runner_source, packaged_runner)
    write_text(output_dir / "requirements-local.txt", "numpy\nopencv-python\nPillow")
    write_text(output_dir / "README.md", package_readme())

    number_region_files = sorted((output_dir / "number_regions").rglob("*.png"))
    track_directories = sorted(path for path in (output_dir / "number_regions").iterdir() if path.is_dir())
    payload_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"package_manifest.json", "jersey_ocr_portable_package.zip"}
    )
    payload_inventory = [file_record(path, output_dir) for path in payload_files]
    manifest = {
        "status": "complete",
        "stage": "jersey_ocr_portable_package",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_project_root": str(PROJECT_ROOT),
        "package_root": str(output_dir),
        "archive_name": "jersey_ocr_portable_package.zip",
        "number_region_count": len(number_region_files),
        "number_region_track_directory_count": len(track_directories),
        "includes": {
            "number_regions": True,
            "crop_visibility_predictions": True,
            "track_visibility_summary": True,
            "clean_crop_metadata": True,
            "jersey_ocr_summary": True,
            "debug_tracks": True,
            "debug_tracks_summary": True,
            "pipeline_run_manifest": True,
            "portable_runner": True,
        },
        "source_paths": {
            "number_regions": str(number_regions),
            **{name: str(source) for name, source in required_sources.items()},
            "portable_runner": str(runner_source),
        },
        "payload_file_count": len(payload_inventory),
        "payload_size_bytes": sum(row["size_bytes"] for row in payload_inventory),
        "payload_files": payload_inventory,
        "identity_assignment_performed": False,
        "training_performed": False,
        "large_models_included": False,
    }
    package_manifest_path = output_dir / "package_manifest.json"
    write_json(package_manifest_path, manifest)

    zip_path = output_dir / "jersey_ocr_portable_package.zip"
    with tempfile.TemporaryDirectory(prefix="jersey_ocr_zip_") as temporary_directory:
        temporary_zip = Path(temporary_directory) / zip_path.name
        make_zip(output_dir, temporary_zip)
        shutil.copy2(temporary_zip, zip_path)

    progress_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_manifest_path, progress_manifest)

    all_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    oversized = [
        {"path": str(path), "size_bytes": path.stat().st_size}
        for path in all_files
        if path.stat().st_size > NINETY_MB
    ]
    total_size = sum(path.stat().st_size for path in all_files)
    summary = {
        "status": "complete",
        "package_folder": str(output_dir),
        "zip_path": str(zip_path),
        "number_region_crops_packaged": len(number_region_files),
        "package_manifest": str(package_manifest_path),
        "progress_manifest": str(progress_manifest),
        "includes_debug_tracks": True,
        "includes_pipeline_run_manifest": True,
        "payload_size_bytes": manifest["payload_size_bytes"],
        "zip_size_bytes": zip_path.stat().st_size,
        "total_package_size_bytes": total_size,
        "files_over_90_mb": oversized,
        "local_ocr_command": (
            "python scripts/run_jersey_number_ocr_baseline.py "
            "--crop-metadata jersey_ocr_portable_package/clean_crop_metadata.json "
            "--visibility-predictions jersey_ocr_portable_package/crop_visibility_predictions.json "
            "--number-regions-dir jersey_ocr_portable_package/number_regions "
            "--output-dir jersey_ocr_local_results --engine tesseract"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
