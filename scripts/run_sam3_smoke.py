#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.sam3_smoke import run_guarded_inference, run_import_check, sampled_frames, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 dependency/import smoke check for Prototype 4.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id with sampled frames.")
    parser.add_argument("--frame-count", type=int, default=4, help="Number of sampled frames for guarded inference, default 4.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Requested device for future guarded inference.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Smoke inference dtype. Defaults to float32 to avoid SAM 3 bf16/float mismatches.")
    parser.add_argument("--repo", default="/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3", help="Local SAM 3 repo path.")
    parser.add_argument("--prompt", default="active lacrosse player on the field", help="Text prompt for guarded inference.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Allow guarded SAM 3 model construction/inference; may download checkpoints.")
    parser.add_argument("--disable-fused-kernels", action=argparse.BooleanOptionalAction, default=True, help="Disable SAM 3 fused MLP kernels during smoke inference. Defaults to true for older GPUs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count < 1:
        raise SystemExit("--frame-count must be positive")
    if args.frame_count > 8:
        raise SystemExit("--frame-count is capped at 8 for SAM 3 smoke inference")

    output_dir = PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sampled_frames(PROJECT_ROOT, args.run_id, args.frame_count)

    import_check = run_import_check(args.repo, device=args.device)
    inference = {
        "status": "skipped",
        "reason": "Imports/dependencies must pass and --allow-download-weights must be supplied before inference.",
        "allow_download_weights_supplied": bool(args.allow_download_weights),
    }
    if import_check["status"] == "imports_passed":
        inference = run_guarded_inference(
            args.repo,
            frames,
            output_dir,
            args.device,
            args.allow_download_weights,
            args.prompt,
            args.dtype,
            args.disable_fused_kernels,
        )

    overall_status = "imports_passed" if import_check["status"] == "imports_passed" else "blocked"
    if inference.get("status") == "complete":
        overall_status = "complete"
    elif inference.get("status") == "failed":
        overall_status = "inference_failed"

    metadata = {
        "status": overall_status,
        "stage": "sam3_smoke",
        "run_id": args.run_id,
        "repo": args.repo,
        "output_dir": str(output_dir),
        "sampled_frame_dir": str(PROJECT_ROOT / "outputs" / args.run_id / "sampled_frames"),
        "frame_count_requested": args.frame_count,
        "frames": [str(path) for path in frames],
        "device": args.device,
        "selected_dtype": args.dtype,
        "disable_fused_kernels": bool(args.disable_fused_kernels),
        "prompt": args.prompt,
        "safety": {
            "did_touch_vggt_outputs": False,
            "did_touch_sam_body4d": False,
            "full_video_inference": False,
            "allow_download_weights_supplied": bool(args.allow_download_weights),
        },
        "import_check": import_check,
        "inference": inference,
        "artifacts": {
            "metadata": str(output_dir / "metadata.json"),
            "manifest": str(output_dir / "sam3_smoke_manifest.json"),
        },
    }
    manifest = {
        "status": overall_status,
        "stage": "sam3_smoke_manifest",
        "run_id": args.run_id,
        "repo_present": import_check.get("repo_present"),
        "import_status": import_check.get("status"),
        "missing_dependencies": import_check.get("missing_dependencies", []),
        "missing_expected_files": import_check.get("missing_expected_files", []),
        "inference_status": inference.get("status"),
        "selected_dtype": args.dtype,
        "disable_fused_kernels": bool(args.disable_fused_kernels),
        "fused_kernels_disabled": (inference.get("fused_kernel_patch") or {}).get("disabled"),
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "metadata.json"),
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "sam3_smoke_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
