import argparse
from pathlib import Path

from prototype4_pipeline.config import load_config
from prototype4_pipeline.pipeline import PipelineContext, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Prototype 4 lacrosse pipeline.")
    parser.add_argument("--video", required=True, help="Path to raw lacrosse game footage.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument("--run-id", default=None, help="Run id. Defaults to input video stem.")
    parser.add_argument("--dry-run", action="store_true", help="Disable model-backed stages.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(Path(args.config))

    if args.output_root:
        config.setdefault("paths", {})["output_root"] = args.output_root

    if args.dry_run:
        config.setdefault("project", {})["mode"] = "dry_run"
        for model_config in config.get("models", {}).values():
            if isinstance(model_config, dict):
                model_config["enabled"] = False

    video_path = Path(args.video).expanduser().resolve()
    run_id = args.run_id or video_path.stem
    context = PipelineContext.from_config(config, video_path, run_id)
    run_pipeline(context)
    print("Prototype 4 dry-run complete: {}".format(context.run_root))


if __name__ == "__main__":
    main()
