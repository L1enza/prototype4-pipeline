#!/usr/bin/env python3
"""Score a jersey-number legibility classifier on the manually labelled nll_test4 crops.

Readers invent numbers on illegible crops (gemma3:4b, Tesseract and qwen2.5vl:7b were
measured doing so), so a gate that only lets legible crops reach a reader could remove
most wrong reads. This checks whether the hockey-trained ResNet34 legibility classifier
of Koshkina & Elder (CVPR 2024 workshops, github.com/mkoshkina/jersey-number-pipeline,
weights CC BY-NC 3.0) separates our human labels, and how many recorded wrong reads it
would have blocked.

    python scripts/eval_legibility_classifier.py --weights .cache/models/legibility_resnet34_hockey.pth

The eval crops are enhanced chest/back number regions, not the whole-player crops the
classifier was trained on, so results understate what it could do on its own inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.legibility import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    ensure_weights,
    load_model,
    score_images,
)

DEFAULT_LABELS = "review_exports/nll_test4_manual_jersey_eval/label_sheet.json"
DEFAULT_IMAGES = "review_exports/nll_test4_manual_jersey_eval/images"
DEFAULT_READER_DIR = "outputs/jersey_reader_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a legibility classifier against manual jersey labels.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--allow-download-weights", action="store_true", help="Fetch the weights if missing (85 MB, CC BY-NC 3.0).")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES)
    parser.add_argument("--reader-dir", default=DEFAULT_READER_DIR, help="Folder of recorded reader predictions to gate.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Legible above this probability (the paper's default).")
    parser.add_argument("--output", default="outputs/jersey_reader_eval/legibility_hockey/predictions.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def roc_auc(positive: list[float], negative: list[float]) -> float | None:
    """Probability that a random legible crop outscores a random illegible one (ties count half)."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def load_reader_predictions(reader_dir: Path) -> dict[str, dict[str, dict]]:
    readers = {}
    for path in sorted(reader_dir.glob("*/predictions.jsonl")):
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                rows[record["eval_id"]] = record
        readers[path.parent.name] = rows
    return readers


def gate_report(rows: list[dict], legible: dict[str, bool], readers: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Per reader: reads made, and wrong or invented numbers before and after the gate."""
    report = {}
    for reader, predictions in readers.items():
        counts = {"scored_crops": 0, "claims": 0, "wrong_claims": 0, "claims_after_gate": 0, "wrong_claims_after_gate": 0,
                  "correct_claims": 0, "correct_claims_after_gate": 0}
        for row in rows:
            record = predictions.get(row["eval_id"])
            if record is None:
                continue
            counts["scored_crops"] += 1
            number = record.get("number")
            if not number or record.get("visibility") == "none":
                continue
            correct = row["manual_readable"] != "no" and str(number) == str(row["manual_label"])
            passed = legible[row["eval_id"]]
            counts["claims"] += 1
            counts["claims_after_gate"] += passed
            counts["correct_claims" if correct else "wrong_claims"] += 1
            counts["correct_claims_after_gate" if correct else "wrong_claims_after_gate"] += passed and 1 or 0
        report[reader] = counts
    return report


def main() -> int:
    args = parse_args()
    rows = json.loads(project_path(args.labels).read_text(encoding="utf-8"))
    images = project_path(args.images_dir)
    paths = [images / Path(row["copied_image_path"]).name for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} eval images missing, e.g. {missing[0]}")

    scores = score_images(load_model(ensure_weights(project_path(args.weights), args.allow_download_weights)), paths)
    legible = {row["eval_id"]: score > args.threshold for row, score in zip(rows, scores)}
    by_class = {label: [s for row, s in zip(rows, scores) if row["manual_readable"] == label] for label in ("yes", "partial", "no")}
    summary = {
        "threshold": args.threshold,
        "auc_readable_vs_unreadable": roc_auc(by_class["yes"] + by_class["partial"], by_class["no"]),
        "auc_clear_vs_unreadable": roc_auc(by_class["yes"], by_class["no"]),
        "passed_gate": {label: sum(s > args.threshold for s in values) for label, values in by_class.items()},
        "crops": {label: len(values) for label, values in by_class.items()},
        "readers": gate_report(rows, legible, load_reader_predictions(project_path(args.reader_dir))),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "summary": summary,
        "crops": [{"eval_id": row["eval_id"], "manual_readable": row["manual_readable"], "manual_label": row["manual_label"],
                   "variant_name": row.get("variant_name"), "legibility": score} for row, score in zip(rows, scores)],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"AUC readable vs unreadable: {summary['auc_readable_vs_unreadable']:.3f}  (clear only: {summary['auc_clear_vs_unreadable']:.3f})")
    for label in ("yes", "partial", "no"):
        print(f"  {label:<8} {summary['passed_gate'][label]:>3} of {summary['crops'][label]:>3} pass the gate at {args.threshold}")
    for reader, counts in summary["readers"].items():
        print(f"  {reader:<22} wrong reads {counts['wrong_claims']:>3} -> {counts['wrong_claims_after_gate']:>3}   "
              f"correct reads {counts['correct_claims']:>3} -> {counts['correct_claims_after_gate']:>3}   "
              f"({counts['scored_crops']} crops scored)")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
