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


def describe_lab(lab: list[float]) -> str:
    lightness = lab[0]
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
        summary[label] = {
            "track_count": len(rows),
            "track_ids": sorted(int(row["track_id"]) for row in rows),
            "mean_shirt_lab": mean_lab,
            "looks": describe_lab(mean_lab) if mean_lab else "no colour evidence",
        }
    return summary


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
