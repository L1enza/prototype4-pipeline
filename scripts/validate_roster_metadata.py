#!/usr/bin/env python3
"""Validate NLL roster JSON files and export deterministic lookup tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.roster_metadata import (  # noqa: E402
    RosterValidationError,
    build_roster_metadata,
    load_roster_files,
    resolve_team_jersey,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate roster metadata and build roster lookup JSON files.")
    parser.add_argument(
        "--roster-json",
        action="append",
        required=True,
        help="Roster JSON path. Repeat for multiple teams.",
    )
    parser.add_argument("--output-dir", default="outputs/roster_metadata_validation")
    parser.add_argument(
        "--allow-team-duplicates",
        action="store_true",
        help="Retain all candidates when one team lists the same jersey number more than once.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_warnings(metadata: dict) -> None:
    ambiguities = metadata["ambiguous_numbers"]
    for duplicate in ambiguities["within_team_duplicates"]:
        names = ", ".join(candidate["name"] for candidate in duplicate["candidates"])
        print(
            "WARNING: {} has duplicate jersey #{}: {}".format(
                duplicate["team_abbreviation"], duplicate["jersey_number"], names
            ),
            file=sys.stderr,
        )
    for shared in ambiguities["shared_across_teams"]:
        mappings = ", ".join(
            "{}:{}".format(candidate["team_abbreviation"], candidate["name"])
            for candidate in shared["candidates"]
        )
        print(
            "WARNING: jersey #{} is shared across teams: {}".format(shared["jersey_number"], mappings),
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    roster_paths = [project_path(value) for value in args.roster_json]
    output_dir = project_path(args.output_dir)
    try:
        rosters = load_roster_files(roster_paths)
        metadata = build_roster_metadata(rosters, allow_team_duplicates=args.allow_team_duplicates)
    except RosterValidationError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    print_warnings(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    jersey_lookup_path = output_dir / "jersey_number_lookup.json"
    ambiguity_path = output_dir / "ambiguous_numbers.json"
    team_lookup_path = output_dir / "team_roster_lookup.json"
    summary_path = output_dir / "roster_validation_summary.json"

    write_json(jersey_lookup_path, metadata["jersey_number_lookup"])
    write_json(ambiguity_path, metadata["ambiguous_numbers"])
    write_json(team_lookup_path, metadata["team_roster_lookup"])

    duplicate_numbers_by_team = {
        roster["team"]["abbreviation"]: [] for roster in metadata["rosters"]
    }
    for row in metadata["ambiguous_numbers"]["within_team_duplicates"]:
        duplicate_numbers_by_team[row["team_abbreviation"]].append(row["jersey_number"])

    summary = {
        "status": "valid_with_warnings" if metadata["ambiguous_numbers"]["within_team_duplicates"] or metadata["ambiguous_numbers"]["shared_across_teams"] else "valid",
        "stage": "roster_metadata_validation",
        "identity_pipeline": "track -> team assignment -> jersey number prediction -> roster lookup -> player name",
        "player_identity_assignment_performed": False,
        "inputs": [str(path.resolve()) for path in roster_paths],
        "parameters": {"allow_team_duplicates": bool(args.allow_team_duplicates)},
        "counts": {
            "teams": len(metadata["rosters"]),
            "players": sum(len(roster["players"]) for roster in metadata["rosters"]),
            "within_team_duplicate_numbers": len(metadata["ambiguous_numbers"]["within_team_duplicates"]),
            "shared_numbers_across_teams": len(metadata["ambiguous_numbers"]["shared_across_teams"]),
        },
        "duplicate_numbers_by_team": duplicate_numbers_by_team,
        "shared_jersey_numbers": metadata["ambiguous_numbers"]["shared_jersey_numbers"],
        "lookup_examples": {
            "TOR_21": resolve_team_jersey(metadata, "TOR", 21),
            "OSH_21": resolve_team_jersey(metadata, "OSH", 21),
        },
        "artifacts": {
            "roster_validation_summary": str(summary_path),
            "jersey_number_lookup": str(jersey_lookup_path),
            "ambiguous_numbers": str(ambiguity_path),
            "team_roster_lookup": str(team_lookup_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
