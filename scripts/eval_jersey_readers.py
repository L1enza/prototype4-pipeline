#!/usr/bin/env python3
"""Score jersey-number readers against the manual jersey digit eval set.

Readers:
- ``tesseract_recorded``: the Tesseract text already stored in the label sheet.
- ``openai:<model>``: any OpenAI-compatible vision endpoint. The default
  endpoint is a local Ollama server, so ``openai:gemma3:4b`` runs offline.

Each reader is scored per crop (correct / wrong / abstained, with wrong reads on
unreadable crops counted as hallucinations) and per track, using the same
"N distinct frames must agree" rule as track-level jersey inference. Predictions
are cached per reader so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Score the exact prompt and number parsing that track-level jersey inference uses per frame.
from prototype4_pipeline.integrations.track_jersey_inference import (  # noqa: E402
    JERSEY_READ_PROMPT as READ_PROMPT,
    clean_number,
    parse_json_object,
)

DEFAULT_LABELS = "review_exports/nll_test4_manual_jersey_eval/label_sheet.json"
DEFAULT_IMAGES = "review_exports/nll_test4_manual_jersey_eval/images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score jersey-number readers against manual labels.")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Labeled label_sheet.json.")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES, help="Folder holding the eval crops.")
    parser.add_argument(
        "--reader",
        action="append",
        required=True,
        help="Reader to score: tesseract_recorded or openai:<model>. Repeat for several.",
    )
    parser.add_argument("--endpoint", default="http://localhost:11434/v1/chat/completions")
    parser.add_argument("--api-key-env", default=None, help="Env var holding an API key; omit for local Ollama.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Seconds per vision request.")
    parser.add_argument("--min-agree", type=int, default=2, help="Distinct frames that must agree on a track number.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N crops (smoke testing).")
    parser.add_argument("--output-dir", default="outputs/jersey_reader_eval")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reader_slug(reader: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", reader)


def image_path(row: dict[str, Any], images_dir: Path) -> Path:
    return images_dir / Path(row["copied_image_path"]).name


def read_with_vision_model(path: Path, model: str, args: argparse.Namespace) -> dict[str, Any]:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": READ_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"
    request = urllib.request.Request(args.endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            text = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
        parsed = parse_json_object(text)
        return {
            "number": clean_number(parsed.get("number")),
            "visibility": parsed.get("visibility"),
            "raw_text": text,
            "seconds": round(time.time() - started, 1),
            "status": "complete",
        }
    except Exception as exc:  # noqa: BLE001 - record backend failures per crop and keep going.
        return {"number": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "complete":
                cache[record["eval_id"]] = record
    return cache


def predict(reader: str, rows: list[dict[str, Any]], args: argparse.Namespace, out_dir: Path) -> dict[str, dict[str, Any]]:
    if reader == "tesseract_recorded":
        return {
            row["eval_id"]: {"eval_id": row["eval_id"], "number": clean_number(row.get("ocr_text")), "status": "complete"}
            for row in rows
        }
    if not reader.startswith("openai:"):
        raise SystemExit(f"Unknown reader {reader!r}; use tesseract_recorded or openai:<model>.")

    model = reader.split(":", 1)[1]
    cache_path = out_dir / "predictions.jsonl"
    cache = load_cache(cache_path)
    images_dir = project_path(args.images_dir)
    todo = [row for row in rows if row["eval_id"] not in cache]
    print(f"[{reader}] {len(cache)} cached, {len(todo)} to run", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(todo, 1):
            result = {"eval_id": row["eval_id"], **read_with_vision_model(image_path(row, images_dir), model, args)}
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if result["status"] == "complete":
                cache[row["eval_id"]] = result
            print(
                f"[{reader}] {index}/{len(todo)} {row['eval_id']} -> {result.get('number')} "
                f"(human: {row['manual_label']}) {result.get('seconds', result.get('error', ''))}",
                flush=True,
            )
    return cache


def crop_outcome(label: str, readable: str, predicted: str | None) -> str:
    if readable == "no":
        return "hallucinated" if predicted else "correct_abstain"
    if not predicted:
        return "abstained"
    if predicted == label:
        return "correct"
    if readable == "partial" and label in predicted:
        return "consistent_with_partial"
    return "wrong"


def vote(numbers_by_frame: list[tuple[int, str]], min_agree: int) -> dict[str, Any]:
    frames: dict[str, set[int]] = defaultdict(set)
    for frame, number in numbers_by_frame:
        if number:
            frames[number].add(frame)
    ranked = sorted(((len(v), k) for k, v in frames.items()), reverse=True)
    if not ranked:
        return {"number": None, "reason": "no_reads"}
    top_count, top = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0
    if top_count < min_agree:
        return {"number": None, "reason": "too_few_agreeing_frames", "leader": top, "leader_frames": top_count}
    if top_count == runner_up:
        return {"number": None, "reason": "tie", "leader": top, "leader_frames": top_count}
    return {"number": top, "frames": top_count, "reason": "accepted"}


def score(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], min_agree: int) -> dict[str, Any]:
    scored = [row for row in rows if row["eval_id"] in predictions]
    crop_counts: Counter[str] = Counter()
    per_crop = []
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        predicted = predictions[row["eval_id"]].get("number")
        outcome = crop_outcome(str(row["manual_label"]), row["manual_readable"], predicted)
        crop_counts[outcome] += 1
        per_crop.append({"eval_id": row["eval_id"], "track_id": row["track_id"], "human": row["manual_label"],
                         "human_readable": row["manual_readable"], "predicted": predicted, "outcome": outcome})
        by_track[row["track_id"]].append({**row, "predicted": predicted})

    track_counts: Counter[str] = Counter()
    per_track = []
    for track_id, items in sorted(by_track.items()):
        truth = vote([(i["frame_index"], str(i["manual_label"])) for i in items if i["manual_readable"] == "yes"], min_agree)
        reader_vote = vote([(i["frame_index"], i["predicted"]) for i in items], min_agree)
        if reader_vote["number"] is None:
            outcome = "abstained"
        elif truth["number"] is None:
            outcome = "unverifiable_claim"
        else:
            outcome = "correct" if reader_vote["number"] == truth["number"] else "wrong"
        track_counts[outcome] += 1
        per_track.append({"track_id": track_id, "human_truth": truth, "reader": reader_vote, "outcome": outcome})

    readable_crops = sum(1 for row in scored if row["manual_readable"] != "no")
    unreadable_crops = len(scored) - readable_crops
    claims = sum(1 for p in per_crop if p["predicted"])
    return {
        "crops_scored": len(scored),
        "crop_outcomes": dict(crop_counts),
        "crop_metrics": {
            "read_rate_on_readable": round(
                (crop_counts["correct"] + crop_counts["consistent_with_partial"]) / readable_crops, 3
            ) if readable_crops else None,
            "hallucination_rate_on_unreadable": round(crop_counts["hallucinated"] / unreadable_crops, 3)
            if unreadable_crops else None,
            "precision_of_claims": round(
                (crop_counts["correct"] + crop_counts["consistent_with_partial"]) / claims, 3
            ) if claims else None,
        },
        "track_outcomes": dict(track_counts),
        "tracks_identifiable_by_human": sum(1 for t in per_track if t["human_truth"]["number"]),
        "per_track": per_track,
        "per_crop": per_crop,
    }


def report_markdown(results: dict[str, dict[str, Any]], min_agree: int) -> str:
    lines = [
        "# Jersey reader evaluation",
        "",
        f"Track rule: at least {min_agree} distinct frames must agree. Human truth uses the same rule on crops labeled clear.",
        "",
        "| Reader | Crops | Read rate (readable) | Hallucination (unreadable) | Precision of claims "
        "| Tracks correct | Tracks wrong | Unverifiable claims | Abstained |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for reader, result in results.items():
        m, t = result["crop_metrics"], result["track_outcomes"]
        fmt = lambda v: "–" if v is None else f"{v:.0%}"  # noqa: E731
        lines.append(
            f"| {reader} | {result['crops_scored']} | {fmt(m['read_rate_on_readable'])} "
            f"| {fmt(m['hallucination_rate_on_unreadable'])} | {fmt(m['precision_of_claims'])} "
            f"| {t.get('correct', 0)} / {result['tracks_identifiable_by_human']} | {t.get('wrong', 0)} "
            f"| {t.get('unverifiable_claim', 0)} | {t.get('abstained', 0)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = json.loads(project_path(args.labels).read_text(encoding="utf-8"))
    unlabeled = [row["eval_id"] for row in rows if not row.get("manual_readable")]
    if unlabeled:
        raise SystemExit(f"{len(unlabeled)} crops have no manual_readable label, e.g. {unlabeled[:3]}")
    rows = sorted(rows, key=lambda row: row["eval_id"])[: args.limit]
    output_root = project_path(args.output_dir)

    results = {}
    for reader in args.reader:
        out_dir = output_root / reader_slug(reader)
        predictions = predict(reader, rows, args, out_dir)
        result = score(rows, predictions, args.min_agree)
        result.update({"reader": reader, "scored_at_utc": datetime.now(timezone.utc).isoformat(),
                       "labels": args.labels, "min_agree": args.min_agree, "limit": args.limit})
        write_json(out_dir / "score.json", result)
        results[reader] = result
        print(json.dumps({"reader": reader, "crop_metrics": result["crop_metrics"],
                          "track_outcomes": result["track_outcomes"]}), flush=True)

    report = report_markdown(results, args.min_agree)
    (output_root / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
