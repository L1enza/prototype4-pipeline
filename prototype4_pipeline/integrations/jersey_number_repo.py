"""Inspection and parsing helpers for the optional Eagle jersey-number adapter."""

from __future__ import annotations

import importlib.util
import re
from collections import defaultdict
from pathlib import Path


EAGLE_REPO_URL = "https://github.com/NVlabs/Eagle.git"
EAGLE_MODEL_ID = "nvidia/LocateAnything-3B"

EAGLE_EXPECTED_FILES = {
    "readme": "README.md",
    "pyproject": "pyproject.toml",
    "worker": "locateanything_worker.py",
    "model_license": "LICENSE_MODEL",
    "package": "eaglevl",
}

EAGLE_IMPORT_MODULES = {
    "torch": "torch",
    "Pillow": "PIL",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "peft": "peft",
    "decord": "decord",
    "deepspeed": "deepspeed",
    "timm": "timm",
}


def inspect_eagle_repo(repo_path: Path) -> dict:
    """Inspect the local checkout and imports without importing Eagle itself."""
    repo_path = repo_path.resolve()
    expected_files = {}
    missing_files = []
    for name, relative_path in EAGLE_EXPECTED_FILES.items():
        path = repo_path / relative_path
        present = path.exists()
        expected_files[name] = {"path": str(path), "present": present}
        if not present:
            missing_files.append(relative_path)

    dependencies = {}
    missing_dependencies = []
    for package, module in EAGLE_IMPORT_MODULES.items():
        available = importlib.util.find_spec(module) is not None
        dependencies[package] = {"module": module, "available": available}
        if not available:
            missing_dependencies.append(package)

    repo_present = repo_path.is_dir()
    if not repo_present:
        status = "repo_missing"
    elif missing_files:
        status = "repo_incomplete"
    elif missing_dependencies:
        status = "dependencies_missing"
    else:
        status = "ready_for_guarded_import"

    return {
        "status": status,
        "repo_path": str(repo_path),
        "repo_present": repo_present,
        "expected_files": expected_files,
        "missing_expected_files": missing_files,
        "dependencies": dependencies,
        "missing_dependencies": missing_dependencies,
        "clone_command_if_missing": "git clone https://github.com/NVlabs/Eagle.git ../Eagle",
        "install_subdirectory": "../Eagle/Embodied",
        "model_id": EAGLE_MODEL_ID,
        "did_import_eagle": False,
        "did_instantiate_model": False,
        "did_download_weights": False,
    }


def extract_jersey_number_candidates(answer: str) -> list[str]:
    """Extract only explicit 1-2 digit text labels, never box coordinates."""
    if not answer:
        return []

    candidates = []
    labels = re.findall(r"<ref>\s*([^<]+?)\s*</ref>", answer, flags=re.IGNORECASE)
    for label in labels:
        candidates.extend(re.findall(r"(?<!\d)(\d{1,2})(?!\d)", label))

    plain_patterns = [
        r"jersey\s+(?:number\s+)?(?:is|:)?\s*#?\s*(\d{1,2})(?!\d)",
        r"(?:number|no\.)\s*(?:is|:)?\s*#?\s*(\d{1,2})(?!\d)",
        r"^\s*#?\s*(\d{1,2})\s*$",
    ]
    stripped = re.sub(r"<box>.*?</box>", "", answer, flags=re.IGNORECASE)
    for pattern in plain_patterns:
        candidates.extend(re.findall(pattern, stripped, flags=re.IGNORECASE))

    unique = []
    for candidate in candidates:
        normalized = str(int(candidate)) if candidate.isdigit() else candidate
        if normalized not in unique:
            unique.append(normalized)
    return unique


def aggregate_track_predictions(crop_predictions: list[dict], minimum_observations: int = 2) -> list[dict]:
    """Build conservative temporal consensus records from per-crop predictions."""
    grouped = defaultdict(list)
    for prediction in crop_predictions:
        grouped[int(prediction["track_id"])].append(prediction)

    results = []
    for track_id in sorted(grouped):
        rows = grouped[track_id]
        usable = [row for row in rows if row.get("jersey_number")]
        votes = defaultdict(float)
        two_digit_values = {row["jersey_number"] for row in usable if len(row["jersey_number"]) == 2}
        for row in usable:
            number = row["jersey_number"]
            weight = row.get("recognizer_confidence")
            weight = float(weight) if weight is not None else 1.0
            if len(number) == 1 and any(number in value for value in two_digit_values):
                weight *= 0.25
            votes[number] += weight

        ranking = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        total_vote = sum(votes.values())
        best_number = ranking[0][0] if ranking else None
        consensus = ranking[0][1] / total_vote if ranking and total_vote else 0.0
        enough_evidence = len(usable) >= minimum_observations and consensus >= 0.67
        results.append(
            {
                "track_id": track_id,
                "status": "accepted" if enough_evidence else "uncertain",
                "jersey_number": best_number if enough_evidence else None,
                "leading_candidate": best_number,
                "temporal_consensus": round(consensus, 6),
                "recognizer_confidence": None,
                "recognizer_confidence_note": "LocateAnything does not expose calibrated per-number confidence.",
                "crop_count": len(rows),
                "usable_prediction_count": len(usable),
                "candidate_votes": [
                    {"jersey_number": number, "weighted_vote": round(weight, 6)}
                    for number, weight in ranking
                ],
                "partial_digit_protection_applied": bool(two_digit_values),
                "roster_match": None,
                "evidence": rows,
            }
        )
    return results
