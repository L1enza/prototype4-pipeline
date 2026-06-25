#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype4_pipeline.integrations.vggt_import_check import print_human_summary, run_vggt_import_check


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test local VGGT imports without running inference.")
    parser.add_argument("--repo", default="../vggt", help="Path to the local VGGT repository.")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip optional COLMAP helper imports and test only core model/utility imports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_vggt_import_check(args.repo, include_optional=not args.core_only)
    if args.json_only:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human_summary(result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
