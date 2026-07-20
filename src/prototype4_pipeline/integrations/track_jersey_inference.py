"""Track-level jersey-number inference from existing Prototype 4 crop artifacts.

This module is intentionally conservative. It can consume existing OCR output,
but it also defines a multimodal vision backend contract so jersey reading is
not locked to Tesseract. The default path does not download models or run
network calls; OpenAI-compatible vision inference is opt-in.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "configs/nll_test4_track_jersey_inference.json"

PREFERRED_ENHANCED_VARIANTS = [
    "center_enlarged_rgb",
    "center_sharpened_rgb_upscaled",
    "wider_torso_rgb",
    "center_chest_back_rgb",
    "upper_center_torso_rgb",
    "center_contrast_gray_upscaled",
    "center_threshold_binary_upscaled",
    "center_threshold_inverted_upscaled",
]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_number(value: Any) -> str | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) in (1, 2):
        return str(int(digits)) if len(digits) == 1 else digits
    return None


def image_stats(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": str(path), "exists": False}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "path": str(path),
        "exists": True,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "pixels": int(image.shape[0] * image.shape[1]),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def variant_rank(name: str | None) -> int:
    if name in PREFERRED_ENHANCED_VARIANTS:
        return PREFERRED_ENHANCED_VARIANTS.index(str(name))
    return len(PREFERRED_ENHANCED_VARIANTS)


def resolve_manifest_region_path(record: dict[str, Any], manifest_path: Path) -> Path:
    raw = Path(str(record.get("region_path", ""))).expanduser()
    if raw.is_absolute():
        return raw
    return manifest_path.parent / raw


def normalize_track_id(value: Any) -> int:
    return int(value)


def source_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row.get("frame_index", -1)),
        str(row.get("source_crop_path") or row.get("crop_path") or ""),
        str(row.get("crop_type") or ""),
    )


def load_clean_crops(path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    rows = payload.get("crops", [])
    output = {}
    for row in rows:
        key = (int(row["track_id"]), int(row["frame_index"]), str(row.get("crop_type") or ""))
        output[key] = row
    return output


def load_team_assignments(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("tracks", rows.get("assignments", []))
    output = {}
    for row in rows:
        if isinstance(row, dict) and "track_id" in row:
            output[int(row["track_id"])] = row
    return output


def roster_number_lookup(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": "missing", "by_number": {}, "by_team_and_number": {}}
    payload = read_json(path)
    if "jersey_number_lookup" in payload:
        return payload["jersey_number_lookup"]
    return payload


def load_ocr_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = read_json(path)
    rows = payload.get("predictions", [])
    return [row for row in rows if isinstance(row, dict)]


def load_manifest_sources(manifest_path: Path, clean_by_key: dict[tuple[int, int, str], dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    payload = read_json(manifest_path)
    grouped: dict[int, dict[tuple[int, str, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in payload.get("regions", []):
        if not isinstance(record, dict):
            continue
        grouped[int(record["track_id"])][source_key(record)].append(record)

    tracks: dict[int, list[dict[str, Any]]] = {}
    for track_id, source_groups in grouped.items():
        sources = []
        for key, variants in source_groups.items():
            frame_index, source_crop_path, crop_type = key
            variants = sorted(
                variants,
                key=lambda row: (
                    variant_rank(row.get("variant_name")),
                    -float(row.get("region_width") or 0) * float(row.get("region_height") or 0),
                ),
            )
            best_variant = variants[0]
            clean = clean_by_key.get((track_id, frame_index, crop_type), {})
            quality = clean.get("quality", {})
            source_path = Path(source_crop_path).expanduser()
            if not source_path.is_absolute():
                source_path = project_path(source_path)
            region_path = resolve_manifest_region_path(best_variant, manifest_path)
            sources.append(
                {
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "source_frame_index": best_variant.get("source_frame_index"),
                    "timestamp_seconds": best_variant.get("timestamp_seconds"),
                    "crop_type": crop_type,
                    "source_crop_path": str(source_path),
                    "best_enhanced_region_path": str(region_path),
                    "best_enhanced_variant": best_variant.get("variant_name"),
                    "enhanced_variant_count": len(variants),
                    "ocr_readiness_score": float(best_variant.get("ocr_readiness_score") or clean.get("crop_quality_score") or 0.0),
                    "likely_view": best_variant.get("likely_view") or "unknown",
                    "view_confidence": best_variant.get("view_confidence") or "none",
                    "source_crop_width": int(best_variant.get("source_crop_width") or quality.get("crop_width") or 0),
                    "source_crop_height": int(best_variant.get("source_crop_height") or quality.get("crop_height") or 0),
                    "quality": quality,
                    "variant_records": variants,
                }
            )
        tracks[track_id] = sources
    return tracks


def source_score(row: dict[str, Any]) -> float:
    crop_type_bonus = 0.12 if row.get("crop_type") == "torso" else 0.0
    area = float(row.get("source_crop_width") or 0) * float(row.get("source_crop_height") or 0)
    area_score = min(1.0, area / 16000.0)
    quality = row.get("quality", {})
    blur_score = float(quality.get("low_motion_blur_score") or 0.45)
    occlusion_score = float(quality.get("occlusion_score") or 0.5)
    readiness = float(row.get("ocr_readiness_score") or 0.0)
    view_bonus = 0.08 if row.get("likely_view") in {"front", "back"} else 0.0
    return 0.36 * readiness + 0.24 * area_score + 0.18 * blur_score + 0.10 * occlusion_score + crop_type_bonus + view_bonus


def select_sources_for_track(sources: list[dict[str, Any]], min_sources: int, max_sources: int) -> list[dict[str, Any]]:
    ranked = sorted(
        sources,
        key=lambda row: (-source_score(row), 0 if row.get("crop_type") == "torso" else 1, int(row.get("frame_index") or 0)),
    )
    selected = []
    used_frames: set[int] = set()
    for row in ranked:
        frame_index = int(row["frame_index"])
        if frame_index in used_frames:
            continue
        selected.append(row)
        used_frames.add(frame_index)
        if len(selected) >= max_sources:
            break
    if len(selected) < min_sources:
        for row in ranked:
            if row in selected:
                continue
            selected.append(row)
            if len(selected) >= min_sources:
                break
    return sorted(selected, key=lambda row: int(row["frame_index"]))


def write_grayscale_copy(input_path: Path, output_path: Path) -> str | None:
    if not input_path.is_file():
        return None
    image = Image.open(input_path).convert("RGB")
    gray = ImageOps.grayscale(image).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gray.save(output_path)
    return str(output_path)


def copy_if_available(input_path: Path, output_path: Path) -> str | None:
    if not input_path.is_file():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return str(output_path)


def preserve_views(selected: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    preserved = []
    for row in selected:
        track_id = int(row["track_id"])
        frame_index = int(row["frame_index"])
        crop_type = str(row.get("crop_type") or "crop")
        base = output_dir / "evidence_views" / f"track_{track_id:03d}" / f"frame_{frame_index:03d}_{crop_type}"
        source_path = Path(str(row["source_crop_path"]))
        enhanced_path = Path(str(row["best_enhanced_region_path"]))
        original_copy = copy_if_available(source_path, base / "original_crop.png")
        grayscale_copy = write_grayscale_copy(source_path, base / "grayscale_crop.png")
        enhanced_copy = copy_if_available(enhanced_path, base / "best_enhanced_region.png")
        row = dict(row)
        row["preserved_views"] = {
            "original_crop": original_copy,
            "grayscale_crop": grayscale_copy,
            "best_enhanced_region": enhanced_copy,
        }
        row["view_image_stats"] = {
            key: image_stats(Path(value)) if value else {"exists": False}
            for key, value in row["preserved_views"].items()
        }
        preserved.append(row)
    return preserved


def collapse_ocr_by_source(predictions: list[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        try:
            key = (int(row["track_id"]), int(row["frame_index"]), str(row.get("crop_type") or ""))
        except (KeyError, TypeError, ValueError):
            continue
        grouped[key].append(row)

    output = {}
    for key, rows in grouped.items():
        usable = [row for row in rows if clean_number(row.get("candidate_number"))]
        if not usable:
            representative = max(rows, key=lambda row: float(row.get("ocr_readiness_score") or 0.0))
            output[key] = {
                "status": "unreadable",
                "candidate_number": None,
                "confidence": None,
                "variant_prediction_count": len(rows),
                "variant_candidate_votes": [],
                "representative_prediction": representative,
            }
            continue
        by_number: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in usable:
            number = clean_number(row.get("candidate_number"))
            if number:
                by_number[number].append(row)
        rankings = []
        for number, number_rows in by_number.items():
            conf_sum = sum(float(row.get("confidence") or 0.25) for row in number_rows)
            rankings.append(
                {
                    "number": number,
                    "variant_vote_count": len(number_rows),
                    "confidence_sum": conf_sum,
                    "mean_confidence": conf_sum / max(len(number_rows), 1),
                }
            )
        supported_two_digits = [row for row in rankings if len(row["number"]) == 2 and row["variant_vote_count"] >= 2]
        pool = supported_two_digits or rankings
        pool.sort(key=lambda row: (-row["variant_vote_count"], -row["confidence_sum"], row["number"]))
        winner = pool[0]["number"]
        representative = max(by_number[winner], key=lambda row: float(row.get("confidence") or 0.0))
        rankings.sort(key=lambda row: (-row["variant_vote_count"], -row["confidence_sum"], row["number"]))
        output[key] = {
            "status": representative.get("status", "candidate"),
            "candidate_number": winner,
            "confidence": representative.get("confidence"),
            "variant_prediction_count": len(rows),
            "variant_candidate_votes": rankings,
            "representative_prediction": representative,
        }
    return output


def vision_backend_status(config: dict[str, Any]) -> dict[str, Any]:
    backend = config.get("vision_backend", {})
    name = backend.get("name", "auto")
    allow_network = bool(backend.get("allow_network", False))
    env_key = backend.get("api_key_env", "OPENAI_API_KEY")
    has_key = bool(os.environ.get(env_key))
    return {
        "requested_backend": name,
        "openai_module_installed": False,
        "httpx_available": True,
        "api_key_env": env_key,
        "api_key_present": has_key,
        "allow_network": allow_network,
        "model": backend.get("model"),
        "will_call_vision_model": name in {"openai", "auto"} and allow_network and has_key,
    }


def encode_image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def run_openai_vision_track(track_id: int, team_label: str | None, selected: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    backend = config.get("vision_backend", {})
    status = vision_backend_status(config)
    if not status["will_call_vision_model"]:
        return {
            "status": "not_run",
            "reason": "Vision backend not enabled or missing API key; used existing OCR evidence only.",
            "backend_status": status,
        }

    endpoint = backend.get("endpoint", "https://api.openai.com/v1/chat/completions")
    model = backend.get("model", "gpt-4o-mini")
    api_key = os.environ[status["api_key_env"]]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are reading NLL lacrosse jersey numbers from multiple crops of the same tracked player. "
                "Do not infer a number from advertisements, boards, overlays, or single threshold artifacts. "
                "Return only JSON with keys: candidate_number, digit_count, visibility, confidence, "
                "evidence_frame_ids, alternative_candidate, conflict_reasons, unreadable_reason. "
                f"Track ID: {track_id}. Team label: {team_label or 'unknown'}."
            ),
        }
    ]
    for row in selected:
        views = row.get("preserved_views", {})
        for view_name in ("original_crop", "grayscale_crop", "best_enhanced_region"):
            view_path = views.get(view_name)
            if not view_path or not Path(view_path).is_file():
                continue
            content.append(
                {
                    "type": "text",
                    "text": f"track {track_id} frame {row['frame_index']} view {view_name}",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(Path(view_path))}})

    request_payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 500,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(backend.get("timeout_seconds", 60))) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        text = response_payload["choices"][0]["message"]["content"]
        parsed = parse_json_object(text)
        return {
            "status": "complete",
            "backend_status": status,
            "raw_response": response_payload,
            "parsed": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - metadata should capture backend failures.
        return {
            "status": "error",
            "backend_status": status,
            "error": f"{type(exc).__name__}: {exc}",
        }


def roster_validate(number: str | None, team_label: str | None, config: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
    if not number:
        return {"status": "no_candidate", "valid_for_assigned_team": False, "valid_any_roster": False, "candidates": []}
    by_number = lookup.get("by_number", {})
    by_team = lookup.get("by_team_and_number", {})
    any_candidates = list(by_number.get(str(int(number)), by_number.get(number, [])))
    label_map = config.get("team_label_to_abbreviation", {})
    team_abbr = label_map.get(team_label) if team_label else None
    if not team_abbr:
        return {
            "status": "team_abbreviation_unknown",
            "team_label": team_label,
            "team_abbreviation": None,
            "jersey_number": number,
            "valid_for_assigned_team": False,
            "valid_any_roster": bool(any_candidates),
            "candidate_count_any_roster": len(any_candidates),
            "candidates": any_candidates,
            "reason": "Team label has not been mapped to a real roster abbreviation.",
        }
    team_abbr = str(team_abbr).upper()
    team_candidates = list(by_team.get(team_abbr, {}).get(str(int(number)), []))
    if not team_candidates:
        status = "number_not_found_for_assigned_team"
    elif len(team_candidates) == 1:
        status = "resolved_for_assigned_team"
    else:
        status = "ambiguous_for_assigned_team"
    return {
        "status": status,
        "team_label": team_label,
        "team_abbreviation": team_abbr,
        "jersey_number": number,
        "valid_for_assigned_team": bool(team_candidates),
        "valid_any_roster": bool(any_candidates),
        "candidate_count_for_assigned_team": len(team_candidates),
        "candidate_count_any_roster": len(any_candidates),
        "candidates": team_candidates,
        "any_roster_candidates": any_candidates,
    }


def candidate_from_vision(result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("status") != "complete":
        return None
    parsed = result.get("parsed") or {}
    number = clean_number(parsed.get("candidate_number"))
    if not number:
        return None
    confidence = parsed.get("confidence")
    if isinstance(confidence, str):
        confidence_value = {"high": 0.9, "medium": 0.65, "low": 0.35}.get(confidence.lower(), 0.5)
    else:
        confidence_value = float(confidence or 0.5)
    frames = parsed.get("evidence_frame_ids") or []
    return {
        "number": number,
        "source": "vision_track",
        "confidence": max(0.0, min(1.0, confidence_value)),
        "evidence_frames": sorted({int(frame) for frame in frames if str(frame).isdigit()}),
        "visibility": parsed.get("visibility"),
        "alternative_candidate": clean_number(parsed.get("alternative_candidate")),
        "raw": parsed,
    }


def aggregate_track(
    track_id: int,
    team_assignment: dict[str, Any] | None,
    selected: list[dict[str, Any]],
    ocr_by_source: dict[tuple[int, int, str], dict[str, Any]],
    vision_result: dict[str, Any],
    config: dict[str, Any],
    lookup: dict[str, Any],
) -> dict[str, Any]:
    team_label = None
    team_confidence = None
    if team_assignment:
        team_label = team_assignment.get("final_class") or team_assignment.get("assigned_class")
        team_confidence = team_assignment.get("confidence")

    evidence_rows = []
    per_frame_candidates = []
    for row in selected:
        key = (track_id, int(row["frame_index"]), str(row.get("crop_type") or ""))
        ocr = ocr_by_source.get(key, {})
        candidate = clean_number(ocr.get("candidate_number"))
        evidence_rows.append(
            {
                "track_id": track_id,
                "frame_index": int(row["frame_index"]),
                "source_frame_index": row.get("source_frame_index"),
                "timestamp_seconds": row.get("timestamp_seconds"),
                "crop_type": row.get("crop_type"),
                "source_crop_path": row.get("source_crop_path"),
                "preserved_views": row.get("preserved_views", {}),
                "selection_score": source_score(row),
                "ocr_readiness_score": row.get("ocr_readiness_score"),
                "best_enhanced_variant": row.get("best_enhanced_variant"),
                "variant_prediction_count": ocr.get("variant_prediction_count", 0),
                "ocr_candidate_number": candidate,
                "ocr_confidence": ocr.get("confidence"),
                "ocr_status": ocr.get("status"),
                "ocr_variant_candidate_votes": ocr.get("variant_candidate_votes", []),
            }
        )
        if candidate:
            confidence = float(ocr.get("confidence") if ocr.get("confidence") is not None else 0.25)
            per_frame_candidates.append(
                {
                    "number": candidate,
                    "source": "existing_ocr_frame",
                    "confidence": confidence,
                    "evidence_frames": [int(row["frame_index"])],
                    "variant_prediction_count": ocr.get("variant_prediction_count", 0),
                }
            )

    vision_candidate = candidate_from_vision(vision_result)
    all_candidates = list(per_frame_candidates)
    if vision_candidate:
        all_candidates.append(vision_candidate)

    numbers = sorted({row["number"] for row in all_candidates})
    candidate_summary = []
    for number in numbers:
        rows = [row for row in all_candidates if row["number"] == number]
        frames = sorted({frame for row in rows for frame in row.get("evidence_frames", [])})
        candidate_summary.append(
            {
                "number": number,
                "sources": sorted({row["source"] for row in rows}),
                "distinct_source_frame_count": len(frames),
                "support_count": len(rows),
                "mean_confidence": float(np.mean([row.get("confidence", 0.0) for row in rows])),
                "evidence_frames": frames,
                "roster_validation": roster_validate(number, team_label, config, lookup),
            }
        )
    candidate_summary.sort(
        key=lambda row: (
            not row["roster_validation"].get("valid_for_assigned_team", False),
            not row["roster_validation"].get("valid_any_roster", False),
            -row["distinct_source_frame_count"],
            -row["mean_confidence"],
            row["number"],
        )
    )

    conflicts = []
    if len({row["number"] for row in candidate_summary}) > 1:
        conflicts.append("multiple_candidate_numbers")
    if vision_candidate and per_frame_candidates and vision_candidate["number"] not in {row["number"] for row in per_frame_candidates}:
        conflicts.append("vision_candidate_conflicts_with_existing_ocr")

    final_number = None
    confidence_label = "unknown"
    reasons = []
    roster_result = {"status": "no_candidate", "valid_for_assigned_team": False, "valid_any_roster": False}
    best = candidate_summary[0] if candidate_summary else None
    if not best:
        reasons.append("no_readable_number_candidate")
    else:
        roster_result = best["roster_validation"]
        valid_team = bool(roster_result.get("valid_for_assigned_team"))
        valid_any = bool(roster_result.get("valid_any_roster"))
        frame_count = int(best["distinct_source_frame_count"])
        mean_conf = float(best["mean_confidence"])
        strong_conflict = bool(conflicts and len(candidate_summary) > 1)
        if valid_team and frame_count >= 2 and not strong_conflict:
            final_number = best["number"]
            confidence_label = "high"
            reasons.append("same_valid_roster_number_supported_by_multiple_distinct_frames")
        elif valid_team and mean_conf >= float(config["aggregation"].get("medium_min_mean_confidence", 0.55)) and not strong_conflict:
            final_number = best["number"]
            confidence_label = "medium"
            reasons.append("single_strong_read_or_compatible_partial_evidence_with_roster_validation")
        elif valid_any and not valid_team:
            reasons.append("candidate_exists_on_roster_but_team_mapping_or_assigned_team_validation_is_missing")
            confidence_label = "low"
        elif strong_conflict:
            reasons.append("conflicting_candidate_numbers")
            confidence_label = "low"
        else:
            reasons.append("invalid_roster_number_or_insufficient_visibility")
            confidence_label = "unknown"

    roster_player = None
    if final_number and roster_result.get("valid_for_assigned_team") and roster_result.get("candidate_count_for_assigned_team") == 1:
        roster_player = roster_result.get("candidates", [None])[0]

    return {
        "track_id": track_id,
        "team_assignment": {
            "team_label": team_label,
            "team_confidence": team_confidence,
            "raw": team_assignment,
        },
        "selected_source_crops": evidence_rows,
        "candidate_numbers": candidate_summary,
        "final_number": final_number,
        "confidence": confidence_label,
        "roster_validation_result": roster_result,
        "supporting_frame_ids": best["evidence_frames"] if best else [],
        "rejection_or_conflict_reasons": reasons + conflicts,
        "vision_model_result": vision_result,
        "roster_player": roster_player if confidence_label in {"high", "medium"} else None,
        "player_name_assigned": bool(roster_player and confidence_label in {"high", "medium"}),
    }


def propagate_to_detections(tracking_path: Path, track_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not tracking_path.is_file():
        return {"status": "missing_tracking_metadata", "detections": []}
    metadata = read_json(tracking_path)
    by_track = {int(row["track_id"]): row for row in track_predictions}
    detections = []
    for detection in metadata.get("detections", []):
        row = dict(detection)
        pred = by_track.get(int(row.get("track_id", -1)))
        if pred:
            row["jersey_number"] = pred["final_number"]
            row["jersey_number_confidence"] = pred["confidence"]
            row["jersey_number_source"] = "track_level_jersey_inference"
            row["roster_player"] = pred.get("roster_player")
        detections.append(row)
    return {
        "status": "complete",
        "source_tracking_metadata": str(tracking_path),
        "detections": detections,
    }


def make_contact_sheet(predictions: list[dict[str, Any]], output_path: Path, thumb_w: int = 168, thumb_h: int = 164) -> str:
    records = []
    for pred in predictions:
        selected = pred.get("selected_source_crops", [])
        best = selected[0] if selected else None
        if not best:
            continue
        views = best.get("preserved_views", {})
        image_path = views.get("original_crop") or views.get("best_enhanced_region") or best.get("source_crop_path")
        if image_path and Path(image_path).is_file():
            records.append((pred, image_path))
    columns = min(5, max(1, len(records)))
    rows = max(1, math.ceil(len(records) / columns))
    label_h = 70
    title_h = 32
    sheet = Image.new("RGB", (columns * thumb_w, title_h + rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Track-level jersey inference review", fill=(0, 0, 0))
    for idx, (pred, image_path) in enumerate(records):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_w - 12, thumb_h - 12))
        x = (idx % columns) * thumb_w
        y = title_h + (idx // columns) * (thumb_h + label_h)
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 6))
        team_label = pred.get("team_assignment", {}).get("team_label")
        number = pred.get("final_number") or "?"
        confidence = pred.get("confidence")
        candidates = ",".join(row["number"] for row in pred.get("candidate_numbers", [])[:3]) or "-"
        reason = "; ".join(pred.get("rejection_or_conflict_reasons", [])[:1])
        draw.text((x + 5, y + thumb_h + 2), f"T{pred['track_id']:03d} {team_label} #{number} {confidence}", fill=(0, 0, 0))
        draw.text((x + 5, y + thumb_h + 20), f"candidates: {candidates}", fill=(60, 60, 60))
        draw.text((x + 5, y + thumb_h + 38), reason[:34], fill=(90, 60, 60))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def run_track_level_inference(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    outputs = config["outputs"]
    output_dir = project_path(outputs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = project_path(inputs["enhanced_number_region_manifest"])
    clean_path = project_path(inputs["clean_crop_metadata"])
    ocr_path = project_path(inputs["crop_ocr_predictions"])
    team_path = project_path(inputs["team_assignments"])
    tracking_path = project_path(inputs["tracking_metadata"])
    roster_path = project_path(inputs["jersey_number_lookup"])

    clean_by_key = load_clean_crops(clean_path)
    track_sources = load_manifest_sources(manifest_path, clean_by_key)
    ocr_by_source = collapse_ocr_by_source(load_ocr_predictions(ocr_path))
    team_assignments = load_team_assignments(team_path)
    roster_lookup = roster_number_lookup(roster_path)
    max_sources = int(config["selection"].get("max_source_frames_per_track", 5))
    min_sources = int(config["selection"].get("min_source_frames_per_track", 3))

    predictions = []
    evidence = []
    for track_id in sorted(track_sources):
        selected = select_sources_for_track(track_sources[track_id], min_sources, max_sources)
        selected = preserve_views(selected, output_dir)
        team_assignment = team_assignments.get(track_id)
        team_label = None
        if team_assignment:
            team_label = team_assignment.get("final_class") or team_assignment.get("assigned_class")
        vision_result = run_openai_vision_track(track_id, team_label, selected, config)
        pred = aggregate_track(
            track_id,
            team_assignment,
            selected,
            ocr_by_source,
            vision_result,
            config,
            roster_lookup,
        )
        predictions.append(pred)
        evidence.append(
            {
                "track_id": track_id,
                "selected_source_crops": pred["selected_source_crops"],
                "vision_model_result": pred["vision_model_result"],
                "candidate_numbers": pred["candidate_numbers"],
            }
        )

    propagated = propagate_to_detections(tracking_path, predictions)
    contact_sheet = make_contact_sheet(predictions, output_dir / "track_jersey_review_contact_sheet.png")
    prediction_path = output_dir / "track_jersey_predictions.json"
    evidence_path = output_dir / "track_jersey_evidence.json"
    summary_path = output_dir / "track_jersey_summary.json"
    propagated_path = output_dir / "detections_with_track_jersey_numbers.json"

    high = sum(1 for row in predictions if row["confidence"] == "high")
    medium = sum(1 for row in predictions if row["confidence"] == "medium")
    unresolved = sum(1 for row in predictions if row["final_number"] is None)
    roster_valid = sum(
        1
        for row in predictions
        if any(candidate.get("roster_validation", {}).get("valid_any_roster") for candidate in row.get("candidate_numbers", []))
    )
    conflicting = sum(1 for row in predictions if "conflicting_candidate_numbers" in row.get("rejection_or_conflict_reasons", []) or "multiple_candidate_numbers" in row.get("rejection_or_conflict_reasons", []))
    summary = {
        "stage": "track_level_jersey_inference",
        "status": "complete",
        "generated_at_utc": now_utc(),
        "run_id": config.get("run_id", "nll_test4"),
        "inputs": {key: str(project_path(value)) for key, value in inputs.items()},
        "outputs": {
            "track_jersey_predictions": str(prediction_path),
            "track_jersey_evidence": str(evidence_path),
            "track_jersey_summary": str(summary_path),
            "track_jersey_review_contact_sheet": contact_sheet,
            "detections_with_track_jersey_numbers": str(propagated_path),
        },
        "backend": vision_backend_status(config),
        "team_label_to_abbreviation": config.get("team_label_to_abbreviation", {}),
        "team_label_metadata": config.get("team_label_metadata", {}),
        "confidence_policy": config.get("confidence_policy"),
        "counts": {
            "tracks_processed": len(predictions),
            "selected_source_crops": sum(len(row["selected_source_crops"]) for row in predictions),
            "high_confidence_numbers": high,
            "medium_confidence_numbers": medium,
            "unresolved_tracks": unresolved,
            "roster_valid_candidates": roster_valid,
            "conflicting_candidates": conflicting,
            "propagated_detection_count": len(propagated.get("detections", [])),
            "player_names_assigned": sum(1 for row in predictions if row.get("player_name_assigned")),
        },
        "known_limitations": [
            "Vision-model inference is opt-in and was skipped unless enabled with API credentials.",
            "Existing OCR predictions are treated as weak supporting evidence, not final identity.",
            "Team labels require a confirmed mapping to roster abbreviations before player names are assigned.",
            "A track can remain unresolved even when a plausible number exists.",
            "Thresholded/enhanced variants from the same source crop are collapsed into one frame of evidence.",
        ],
    }

    write_json(prediction_path, {"stage": "track_level_jersey_inference", "status": "complete", "tracks": predictions})
    write_json(evidence_path, {"stage": "track_level_jersey_evidence", "status": "complete", "tracks": evidence})
    write_json(propagated_path, propagated)
    write_json(summary_path, summary)
    return summary
