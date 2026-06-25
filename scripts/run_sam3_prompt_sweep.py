#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.sam3_prompt_sweep import DEFAULT_PROMPTS, run_prompt_sweep
from prototype4_pipeline.integrations.sam3_smoke import run_import_check, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small SAM 3 prompt sweep on sampled frames.")
    parser.add_argument("--run-id", default="nll-test1", help="Prototype 4 run id with sampled frames.")
    parser.add_argument("--frame-count", type=int, default=4, help="Number of sampled frames to evaluate, capped at 8.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Requested device.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="SAM 3 smoke dtype.")
    parser.add_argument("--repo", default="/afs/ece.cmu.edu/usr/zllenza/research/prototype4/sam3", help="Local SAM 3 repo path.")
    parser.add_argument("--allow-download-weights", action="store_true", help="Allow model construction/inference; may download checkpoints.")
    parser.add_argument("--disable-fused-kernels", action=argparse.BooleanOptionalAction, default=True, help="Disable SAM 3 fused MLP kernels during smoke inference. Defaults to true.")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Override prompt list. Repeat for multiple prompts.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count < 1:
        raise SystemExit("--frame-count must be positive")
    if args.frame_count > 8:
        raise SystemExit("--frame-count is capped at 8 for SAM 3 prompt sweep")

    prompts = args.prompts or DEFAULT_PROMPTS
    output_dir = PROJECT_ROOT / "outputs" / args.run_id / "player_masks" / "sam3_prompt_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    import_check = run_import_check(args.repo, device=args.device)
    if import_check.get("status") == "imports_passed":
        sweep = run_prompt_sweep(
            PROJECT_ROOT,
            args.run_id,
            args.repo,
            output_dir,
            prompts,
            args.frame_count,
            args.device,
            args.dtype,
            args.allow_download_weights,
            args.disable_fused_kernels,
        )
    else:
        sweep = {
            "status": "blocked",
            "error": {"type": "ImportCheckFailed", "message": "SAM 3 imports/dependencies did not pass."},
            "prompts": prompts,
        }

    metadata = {
        "status": sweep.get("status"),
        "stage": "sam3_prompt_sweep",
        "run_id": args.run_id,
        "repo": args.repo,
        "output_dir": str(output_dir),
        "frame_count_requested": args.frame_count,
        "selected_dtype": args.dtype,
        "disable_fused_kernels": bool(args.disable_fused_kernels),
        "safety": {
            "did_touch_vggt_outputs": False,
            "did_touch_sam_body4d": False,
            "full_video_inference": False,
            "allow_download_weights_supplied": bool(args.allow_download_weights),
        },
        "import_check": import_check,
        "sweep": sweep,
    }
    summary = {
        "status": sweep.get("status"),
        "stage": "sam3_prompt_sweep_summary",
        "run_id": args.run_id,
        "prompts": prompts,
        "prompt_summaries": sweep.get("prompt_summaries", []),
        "contact_sheets": sweep.get("contact_sheets", []),
        "output_dir": str(output_dir),
        "metadata": str(output_dir / "metadata.json"),
        "notes": "Ranking is heuristic until field-boundary filtering and bench/crowd suppression are calibrated.",
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "prompt_sweep_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
