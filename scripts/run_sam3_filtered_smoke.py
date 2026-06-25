#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.sam3_filtering import (
    DEFAULT_FILTER_PROMPT,
    FILTER_MODES,
    parse_polygon,
    parse_polygon_json,
    run_filtered_masks,
)
from prototype4_pipeline.integrations.sam3_smoke import run_import_check, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM 3 active-field mask filtering on sampled frames.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id with sampled frames.")
    parser.add_argument("--frame-count", type=int, default=4, help="Number of sampled frames to evaluate, capped at 8.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Requested device.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="SAM 3 smoke dtype.")
    parser.add_argument("--repo", default="/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3", help="Local SAM 3 repo path.")
    parser.add_argument("--prompt", default=DEFAULT_FILTER_PROMPT, help="Prompt used for SAM 3 masks before filtering.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Allow model construction/inference; may download checkpoints.")
    parser.add_argument("--disable-fused-kernels", action=argparse.BooleanOptionalAction, default=True, help="Disable SAM 3 fused MLP kernels during smoke inference. Defaults to true.")
    parser.add_argument("--filter-mode", default="combined", choices=sorted(FILTER_MODES), help="Filtering mode: green, polygon, y-threshold, or combined geometry.")
    parser.add_argument("--bench-y-cutoff", type=float, default=0.18, help="Bench/boards cutoff as ratio 0-1 or pixel y. Centroids above this are rejected.")
    parser.add_argument("--field-y-min", type=float, default=0.18, help="Playable field minimum y as ratio 0-1 or pixel y. Bottom points above this are rejected.")
    parser.add_argument("--field-y-max", type=float, default=0.96, help="Playable field maximum y as ratio 0-1 or pixel y.")
    parser.add_argument("--bench-y-max-ratio", type=float, default=None, help="Backward-compatible alias for --bench-y-cutoff.")
    parser.add_argument("--field-y-min-ratio", type=float, default=None, help="Backward-compatible alias for --field-y-min.")
    parser.add_argument("--field-y-max-ratio", type=float, default=None, help="Backward-compatible alias for --field-y-max.")
    parser.add_argument("--field-polygon", default=None, help="Optional field polygon as x,y;x,y;... using pixels or 0-1 normalized coordinates.")
    parser.add_argument("--field-polygon-json", default=None, help="Optional field polygon JSON: [[x,y], ...] or {\"points\": [[x,y], ...]}.")
    parser.add_argument("--require-green-surface", action=argparse.BooleanOptionalAction, default=False, help="Only used in green_foot_point mode; combined mode records green but does not reject solely on it.")
    parser.add_argument("--green-sample-radius", type=int, default=5, help="Patch radius for green-surface sampling.")
    parser.add_argument("--green-sample-y-offset", type=int, default=6, help="Pixels below foot point to sample for playing surface.")
    parser.add_argument("--min-mask-pixels", type=int, default=35, help="Reject tiny masks below this pixel count.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count < 1:
        raise SystemExit("--frame-count must be positive")
    if args.frame_count > 8:
        raise SystemExit("--frame-count is capped at 8 for SAM 3 filtered smoke")

    bench_y_cutoff = args.bench_y_max_ratio if args.bench_y_max_ratio is not None else args.bench_y_cutoff
    field_y_min = args.field_y_min_ratio if args.field_y_min_ratio is not None else args.field_y_min
    field_y_max = args.field_y_max_ratio if args.field_y_max_ratio is not None else args.field_y_max
    polygon = parse_polygon_json(args.field_polygon_json) if args.field_polygon_json else parse_polygon(args.field_polygon)

    output_dir = PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_filtered"
    output_dir.mkdir(parents=True, exist_ok=True)
    import_check = run_import_check(args.repo, device=args.device)
    config = {
        "filter_mode": args.filter_mode,
        "bench_y_cutoff": bench_y_cutoff,
        "field_y_min": field_y_min,
        "field_y_max": field_y_max,
        "field_polygon": polygon,
        "require_green_surface": bool(args.require_green_surface),
        "green_sample_radius": args.green_sample_radius,
        "green_sample_y_offset": args.green_sample_y_offset,
        "min_mask_pixels": args.min_mask_pixels,
    }
    if import_check.get("status") == "imports_passed":
        filtered = run_filtered_masks(
            PROJECT_ROOT,
            args.run_id,
            args.repo,
            output_dir,
            args.prompt,
            args.frame_count,
            args.device,
            args.dtype,
            args.allow_download_weights,
            args.disable_fused_kernels,
            config=config,
        )
    else:
        filtered = {
            "status": "blocked",
            "error": {"type": "ImportCheckFailed", "message": "SAM 3 imports/dependencies did not pass."},
        }
    frame_summaries = [
        {
            "frame_index": row.get("frame_index"),
            "all_mask_count": row.get("all_mask_count"),
            "kept_mask_count": row.get("kept_mask_count"),
            "rejected_mask_count": row.get("rejected_mask_count"),
            "all_masks_overlay_path": row.get("all_masks_overlay_path"),
            "filtered_active_player_masks_overlay_path": row.get("filtered_active_player_masks_overlay_path"),
            "filter_debug_overlay_path": row.get("filter_debug_overlay_path"),
            "metadata_path": row.get("metadata_path"),
        }
        for row in filtered.get("frame_metadata", [])
    ]
    metadata = {
        "status": filtered.get("status"),
        "stage": "sam3_filtered_smoke",
        "run_id": args.run_id,
        "repo": args.repo,
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "frame_count_requested": args.frame_count,
        "selected_dtype": args.dtype,
        "disable_fused_kernels": bool(args.disable_fused_kernels),
        "filter_config": config,
        "safety": {
            "did_touch_vggt_outputs": False,
            "did_touch_sam_body4d": False,
            "full_video_inference": False,
            "allow_download_weights_supplied": bool(args.allow_download_weights),
        },
        "import_check": import_check,
        "filtered": filtered,
    }
    summary = {
        "status": filtered.get("status"),
        "stage": "sam3_filtered_summary",
        "run_id": args.run_id,
        "prompt": args.prompt,
        "filter_mode": args.filter_mode,
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "metadata.json"),
        "frame_summaries": frame_summaries,
        "notes": "Geometry-first filter. Tune bench-y-cutoff, field-y-min/max, or field-polygon-json from the debug overlays.",
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "sam3_filtered_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
