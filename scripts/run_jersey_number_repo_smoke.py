#!/usr/bin/env python3
"""Prepare or explicitly run a small jersey-number recognition smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.jersey_number_repo import (  # noqa: E402
    EAGLE_MODEL_ID,
    aggregate_track_predictions,
    extract_jersey_number_candidates,
    inspect_eagle_repo,
)


PROMPT = (
    "Read the complete jersey number visible on this player's torso or back. "
    "Return only the complete one- or two-digit number. If it is not clearly visible, return none."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or explicitly run the guarded Eagle jersey-number smoke adapter."
    )
    parser.add_argument(
        "--crop-dir",
        default="outputs/nll-test1/player_crops/tracklet_crop_smoke",
        help="Tracklet crop stage directory containing crop_metadata.json.",
    )
    parser.add_argument("--crop-metadata", default=None, help="Override crop metadata JSON path.")
    parser.add_argument("--tracking-metadata", default=None, help="Optional source tracking metadata.")
    parser.add_argument("--roster", default=None, help="Reserved CSV/JSON roster input; not matched yet.")
    parser.add_argument("--output-dir", default="outputs/jersey_number_smoke")
    parser.add_argument("--eagle-repo", default="../Eagle/Embodied")
    parser.add_argument("--model", default=EAGLE_MODEL_ID)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-tracks", type=int, default=5)
    parser.add_argument("--crops-per-track", type=int, default=3)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Instantiate LocateAnything and run the experimental recognition prompt.",
    )
    parser.add_argument(
        "--allow-download-weights",
        action="store_true",
        help="Required with --run-inference when --model is not a local path.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_project_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def crop_rank(crop: dict) -> float:
    area = float(crop.get("bbox_area") or 0)
    sharpness = min(float(crop.get("sharpness_score") or 0), 5000.0)
    border = max(float(crop.get("border_margin_px") or 0), 0.0)
    return area + 10.0 * sharpness + 5.0 * border


def select_crops(metadata: dict, max_tracks: int, crops_per_track: int) -> list[dict]:
    selected = []
    mask_highlight_applied = bool(metadata.get("parameters", {}).get("apply_mask_highlight"))
    tracks = sorted(metadata.get("tracklets", []), key=lambda row: int(row["track_id"]))
    for track in tracks[:max_tracks]:
        track_id = int(track["track_id"])
        seen = set()
        ranked = sorted(track.get("crops", []), key=crop_rank, reverse=True)
        for rank, crop in enumerate(ranked, start=1):
            path_value = crop.get("crop_path")
            if not path_value:
                continue
            crop_path = Path(path_value)
            if not crop_path.exists() or str(crop_path) in seen:
                continue
            seen.add(str(crop_path))
            size = crop.get("crop_size") or {}
            selected.append(
                {
                    "track_id": track_id,
                    "crop_rank": rank,
                    "crop_path": str(crop_path),
                    "frame_index": crop.get("frame_index"),
                    "mask_id": crop.get("mask_id"),
                    "crop_size": size,
                    "bbox": crop.get("bbox"),
                    "bbox_area": crop.get("bbox_area"),
                    "sharpness_score": crop.get("sharpness_score"),
                    "has_3d_support": bool(crop.get("has_3d_support")),
                    "selection_score": crop_rank(crop),
                    "source_mask_highlight_applied": mask_highlight_applied,
                    "visibility_assessment": "unknown_no_pose_or_back_visibility_classifier",
                    "low_resolution_warning": min(int(size.get("width") or 0), int(size.get("height") or 0)) < 128,
                }
            )
            if sum(1 for row in selected if row["track_id"] == track_id) >= crops_per_track:
                break
    return selected


def initial_predictions(selected: list[dict]) -> list[dict]:
    return [
        {
            **crop,
            "status": "not_run",
            "jersey_number": None,
            "number_candidates": [],
            "recognizer_confidence": None,
            "raw_answer": None,
            "text_boxes": [],
            "reason": "Inference is opt-in and was not requested.",
        }
        for crop in selected
    ]


@contextmanager
def prepend_sys_path(path: Path):
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if value in sys.path:
            sys.path.remove(value)


def validate_inference_request(args: argparse.Namespace, repo_status: dict) -> None:
    if not args.run_inference:
        return
    if repo_status["status"] != "ready_for_guarded_import":
        raise RuntimeError(
            "Eagle is not ready: {} (missing files: {}; missing dependencies: {})".format(
                repo_status["status"],
                repo_status["missing_expected_files"],
                repo_status["missing_dependencies"],
            )
        )
    model_path = Path(args.model).expanduser()
    if not model_path.exists() and not args.allow_download_weights:
        raise RuntimeError("Remote model loading requires --allow-download-weights.")


def run_eagle_inference(
    args: argparse.Namespace,
    repo_path: Path,
    repo_status: dict,
    predictions: list[dict],
) -> None:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable in this process.")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    with prepend_sys_path(repo_path):
        from locateanything_worker import LocateAnythingWorker

        repo_status["did_import_eagle"] = True
        worker = LocateAnythingWorker(args.model, device=args.device, dtype=dtype)
        repo_status["did_instantiate_model"] = True
        repo_status["did_download_weights"] = not Path(args.model).expanduser().exists()
        inference_context = torch.inference_mode() if hasattr(torch, "inference_mode") else nullcontext()
        with inference_context:
            for prediction in predictions:
                image = Image.open(prediction["crop_path"]).convert("RGB")
                result = worker.predict(
                    image,
                    PROMPT,
                    generation_mode="slow",
                    max_new_tokens=128,
                    temperature=0.0,
                    top_p=1.0,
                    verbose=False,
                )
                answer = str(result.get("answer") or "")
                candidates = extract_jersey_number_candidates(answer)
                prediction.update(
                    {
                        "status": "predicted" if len(candidates) == 1 else "uncertain",
                        "jersey_number": candidates[0] if len(candidates) == 1 else None,
                        "number_candidates": candidates,
                        "recognizer_confidence": None,
                        "raw_answer": answer,
                        "reason": (
                            "One complete candidate parsed from model text."
                            if len(candidates) == 1
                            else "No unique complete number could be parsed."
                        ),
                    }
                )


def make_contact_sheet(predictions: list[dict], output_path: Path) -> None:
    thumb_w, thumb_h, label_h = 220, 250, 58
    cols = min(4, max(1, len(predictions)))
    rows = max(1, (len(predictions) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, prediction in enumerate(predictions):
        image = Image.open(prediction["crop_path"]).convert("RGB")
        image.thumbnail((thumb_w - 12, thumb_h - 12))
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + label_h)
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 6))
        number = prediction.get("jersey_number") or "?"
        label = "T{} f{}  number {}".format(
            prediction["track_id"], prediction.get("frame_index", "?"), number
        )
        draw.text((x + 6, y + thumb_h + 5), label, fill=(0, 0, 0))
        draw.text((x + 6, y + thumb_h + 25), prediction["status"], fill=(80, 80, 80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    crop_dir = resolve_project_path(args.crop_dir)
    crop_metadata_path = resolve_project_path(args.crop_metadata) if args.crop_metadata else crop_dir / "crop_metadata.json"
    tracking_metadata = resolve_project_path(args.tracking_metadata)
    roster_path = resolve_project_path(args.roster)
    output_dir = resolve_project_path(args.output_dir)
    eagle_repo = resolve_project_path(args.eagle_repo)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(crop_metadata_path)
    selected = select_crops(metadata, args.max_tracks, args.crops_per_track)
    predictions = initial_predictions(selected)
    repo_status = inspect_eagle_repo(eagle_repo)
    inference_error = None
    inference_traceback = None

    try:
        validate_inference_request(args, repo_status)
        if args.run_inference:
            run_eagle_inference(args, eagle_repo, repo_status, predictions)
    except Exception as exc:  # Keep the smoke artifacts useful on setup failures.
        inference_error = f"{type(exc).__name__}: {exc}"
        inference_traceback = traceback.format_exc()
        for prediction in predictions:
            if prediction["status"] == "not_run":
                prediction["status"] = "error"
                prediction["reason"] = inference_error

    tracks = aggregate_track_predictions(predictions, args.minimum_observations)
    contact_sheet = output_dir / "jersey_number_contact_sheet.png"
    make_contact_sheet(predictions, contact_sheet)

    stage_status = "inference_complete" if args.run_inference and not inference_error else "inspection_complete"
    if inference_error:
        stage_status = "inference_failed"
    crop_payload = {
        "status": stage_status,
        "stage": "jersey_number_repo_smoke",
        "backend": "nvlabs_eagle_locateanything",
        "model": args.model,
        "prompt": PROMPT,
        "inference_requested": bool(args.run_inference),
        "predictions": predictions,
    }
    track_payload = {
        "status": stage_status,
        "aggregation": {
            "method": "confidence_weighted_temporal_vote",
            "minimum_observations": args.minimum_observations,
            "acceptance_threshold": 0.67,
            "single_digit_downweight_when_matching_double_digit_exists": 0.25,
        },
        "tracks": tracks,
    }
    summary = {
        "status": stage_status,
        "stage": "jersey_number_repo_smoke_summary",
        "inputs": {
            "crop_dir": str(crop_dir),
            "crop_metadata": str(crop_metadata_path),
            "tracking_metadata": str(tracking_metadata) if tracking_metadata else None,
            "roster": str(roster_path) if roster_path else None,
        },
        "repo_inspection": repo_status,
        "inference": {
            "requested": bool(args.run_inference),
            "allow_download_weights": bool(args.allow_download_weights),
            "device": args.device,
            "dtype": args.dtype,
            "error": inference_error,
            "traceback": inference_traceback,
        },
        "counts": {
            "tracks_selected": len({row["track_id"] for row in selected}),
            "crops_selected": len(selected),
            "crop_predictions": sum(1 for row in predictions if row["jersey_number"]),
            "accepted_track_numbers": sum(1 for row in tracks if row["status"] == "accepted"),
            "uncertain_tracks": sum(1 for row in tracks if row["status"] == "uncertain"),
        },
        "artifacts": {
            "crop_predictions": str(output_dir / "crop_predictions.json"),
            "track_number_predictions": str(output_dir / "track_number_predictions.json"),
            "contact_sheet": str(contact_sheet),
            "summary": str(output_dir / "jersey_number_smoke_summary.json"),
        },
        "limitations": [
            "LocateAnything is a general visual-grounding model, not a dedicated jersey-number recognizer.",
            "The public worker does not expose calibrated per-number confidence.",
            "Back/torso visibility is not classified yet.",
            "Current tracklet crops may contain a green SAM mask highlight; clean RGB crops are preferred for real OCR.",
            "Low-resolution crops and partial digits remain unresolved.",
            "No roster identity is assigned from a single crop or uncertain track consensus.",
        ],
    }
    write_json(output_dir / "crop_predictions.json", crop_payload)
    write_json(output_dir / "track_number_predictions.json", track_payload)
    write_json(output_dir / "jersey_number_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if inference_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
