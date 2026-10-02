"""Human confirmation of which team-assignment label is which real team.

Team assignment names its two colour clusters `team_a` (the larger cluster) and
`team_b`, so the labels carry no fixed meaning: on another clip, or a re-run with
different tracks, `team_a` can be the other team. Jersey inference therefore only
maps labels to roster teams through a confirmation a person wrote for one specific
team-assignment result. The confirmation stores a fingerprint of the per-track
labels it was made against; if the labels change, it no longer applies.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "team_mapping_confirmation_v1"
TEAM_LABELS = ("team_a", "team_b")


def assignment_fingerprint(assignments: dict[int, dict[str, Any]]) -> str:
    """Hash of every track's team label, so any relabelling invalidates a confirmation."""
    lines = [
        f"{track_id}:{row.get('final_class') or row.get('assigned_class')}"
        for track_id, row in sorted(assignments.items())
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def lightness_from_opencv_lab(lab: list[float]) -> float:
    """True L* (0-100) from OpenCV's 8-bit Lab, which stores L scaled to 0-255.

    Team assignment computes `representative_median_lab` with cv2.COLOR_RGB2LAB on
    uint8 crops, so a and b are offset by 128 and L must be rescaled before any
    threshold on real lightness applies.
    """
    return float(lab[0]) * 100.0 / 255.0


def describe_lab(lab: list[float]) -> str:
    lightness = lightness_from_opencv_lab(lab)
    if lightness >= 65:
        return "light (e.g. white uniforms)"
    if lightness <= 40:
        return "dark (e.g. maroon/black uniforms)"
    return "mid-tone"


def label_color_summary(assignments: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Per-label track count and average shirt colour, to help a reviewer tell the teams apart."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments.values():
        by_label[str(row.get("final_class") or row.get("assigned_class"))].append(row)

    summary = {}
    for label in sorted(by_label):
        rows = by_label[label]
        labs = [row["representative_median_lab"] for row in rows if row.get("representative_median_lab")]
        mean_lab = [round(sum(values) / len(values), 1) for values in zip(*labs)] if labs else None
        lightness = [lightness_from_opencv_lab(lab) for lab in labs]
        summary[label] = {
            "track_count": len(rows),
            "track_ids": sorted(int(row["track_id"]) for row in rows),
            "mean_shirt_lab": mean_lab,
            "mean_lightness": round(lightness_from_opencv_lab(mean_lab)) if mean_lab else None,
            "lightness_range": [round(min(lightness)), round(max(lightness))] if lightness else None,
            # Same thresholds as describe_lab: one label holding both kinds of shirt is two teams merged.
            "mixed_light_and_dark": bool(lightness) and max(lightness) >= 65 and min(lightness) <= 40,
            "looks": describe_lab(mean_lab) if mean_lab else "no colour evidence",
        }
    return summary


def split_warnings(summary: dict[str, Any], min_team_share: float = 0.25) -> list[str]:
    """Signs that team_a/team_b are not cleanly the two teams, shown before anyone confirms.

    A single odd track can take a whole cluster and push both teams into the other,
    and one team on screen still yields two clusters; either would send one team's
    numbers to the other team's roster.
    """
    teams = {label: info for label, info in summary.items() if label in TEAM_LABELS}
    warnings = []
    missing = [label for label in TEAM_LABELS if label not in teams]
    if missing:
        warnings.append(f"no tracks are labelled {', '.join(missing)}; the clip may show one team or the split failed")
    for label, info in teams.items():
        if info.get("mixed_light_and_dark"):
            low, high = info["lightness_range"]
            warnings.append(f"{label} mixes light and dark shirts (lightness {low}-{high}/100); both teams may be merged in it")
    if len(teams) == 2:
        counts = [teams[label]["track_count"] for label in TEAM_LABELS]
        if sum(counts) and min(counts) / sum(counts) < min_team_share:
            warnings.append(f"lopsided split: only {min(counts)} of {sum(counts)} team tracks are in the smaller label")
        tones = {info["looks"].split(" (")[0] for info in teams.values()}
        if len(tones) == 1 and "no colour evidence" not in tones:
            warnings.append(f"team_a and team_b both look {tones.pop()}; they may not be the two teams")
    return warnings


def load_confirmed_mapping(
    confirmation_path: Path | None,
    assignments: dict[int, dict[str, Any]],
    known_abbreviations: set[str] | None = None,
) -> dict[str, Any]:
    """Return the label -> roster abbreviation mapping only if a valid confirmation applies.

    `status` is "confirmed" or explains why no mapping is used; `mapping` is empty
    unless confirmed, which keeps every track's team unmapped and every name withheld.
    """
    current = assignment_fingerprint(assignments) if assignments else None
    result: dict[str, Any] = {
        "status": "missing",
        "mapping": {},
        "confirmation_path": str(confirmation_path) if confirmation_path else None,
        "current_assignment_fingerprint": current,
    }
    if not assignments:
        return {**result, "status": "no_team_assignments", "reason": "No team assignments were loaded."}
    if not confirmation_path or not confirmation_path.is_file():
        return {
            **result,
            "reason": "No team mapping confirmation exists for these team assignments; run scripts/confirm_team_mapping.py.",
        }

    try:
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**result, "status": "invalid", "reason": f"Could not read confirmation: {type(exc).__name__}: {exc}"}

    result.update(
        {
            "confirmed_by": confirmation.get("confirmed_by"),
            "confirmed_at_utc": confirmation.get("confirmed_at_utc"),
            "confirmed_assignment_fingerprint": confirmation.get("assignment_fingerprint"),
        }
    )
    mapping = confirmation.get("mapping") or {}
    problems = []
    if confirmation.get("schema") != SCHEMA:
        problems.append(f"schema is {confirmation.get('schema')!r}, expected {SCHEMA!r}")
    if not confirmation.get("confirmed_by"):
        problems.append("confirmed_by is empty")
    if set(mapping) != set(TEAM_LABELS):
        problems.append(f"mapping must cover exactly {list(TEAM_LABELS)}")
    abbreviations = [str(value).upper() for value in mapping.values() if value]
    if len(abbreviations) != len(set(abbreviations)) or len(abbreviations) != len(mapping):
        problems.append("each label must map to a different, non-empty team")
    if known_abbreviations is not None:
        unknown = sorted(set(abbreviations) - known_abbreviations)
        if unknown:
            problems.append(f"teams not in the roster lookup: {unknown}")
    if problems:
        return {**result, "status": "invalid", "reason": "; ".join(problems)}

    if confirmation.get("assignment_fingerprint") != current:
        return {
            **result,
            "status": "stale",
            "reason": (
                "Team labels changed since this mapping was confirmed (team_a/team_b may have swapped); "
                "re-run scripts/confirm_team_mapping.py after reviewing the new assignments."
            ),
        }

    return {
        **result,
        "status": "confirmed",
        "mapping": {label: str(value).upper() for label, value in mapping.items()},
        "reason": "Mapping confirmed for exactly these team assignments.",
    }
