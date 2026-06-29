"""Validation and lookup helpers for team roster metadata."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


class RosterValidationError(ValueError):
    """Raised when roster metadata cannot be indexed safely."""


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def normalize_jersey_number(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a jersey number")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise ValueError("jersey number must be an integer or digit string")
    if not 0 <= number <= 99:
        raise ValueError("jersey number must be between 0 and 99")
    return number


def validate_roster_document(document: object, source_path: Path) -> dict:
    """Validate and normalize one roster JSON document."""
    errors = []
    if not isinstance(document, dict):
        raise RosterValidationError(f"{source_path}: roster root must be an object")

    schema = document.get("schema")
    if not _nonempty_string(schema):
        errors.append("schema must be a non-empty string")

    team = document.get("team")
    if not isinstance(team, dict):
        errors.append("team must be an object")
        team = {}
    for field in ("name", "abbreviation", "source_url"):
        if not _nonempty_string(team.get(field)):
            errors.append(f"team.{field} must be a non-empty string")

    players = document.get("players")
    if not isinstance(players, list):
        errors.append("players must be an array")
        players = []

    normalized_players = []
    for index, player in enumerate(players):
        prefix = f"players[{index}]"
        if not isinstance(player, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if not _nonempty_string(player.get("name")):
            errors.append(f"{prefix}.name must be a non-empty string")
        if not _nonempty_string(player.get("position")):
            errors.append(f"{prefix}.position must be a non-empty string")
        try:
            number = normalize_jersey_number(player.get("number"))
        except ValueError as exc:
            errors.append(f"{prefix}.number {exc}")
            continue
        normalized = dict(player)
        normalized["name"] = str(player.get("name", "")).strip()
        normalized["number"] = number
        normalized["position"] = str(player.get("position", "")).strip()
        normalized_players.append(normalized)

    if errors:
        joined = "\n  - ".join(errors)
        raise RosterValidationError(f"{source_path}: validation failed:\n  - {joined}")

    normalized_team = dict(team)
    normalized_team["name"] = team["name"].strip()
    normalized_team["abbreviation"] = team["abbreviation"].strip().upper()
    normalized_team["source_url"] = team["source_url"].strip()
    return {
        "schema": schema.strip(),
        "team": normalized_team,
        "players": normalized_players,
        "field_notes": document.get("field_notes"),
        "source_file": str(source_path.resolve()),
    }


def load_roster_files(paths: Iterable[Path]) -> list[dict]:
    """Load, validate, and normalize one or more roster files."""
    rosters = []
    seen_paths = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.is_file():
            raise RosterValidationError(f"Roster file not found: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RosterValidationError(f"{path}: invalid JSON: {exc}") from exc
        rosters.append(validate_roster_document(document, path))
    if not rosters:
        raise RosterValidationError("At least one roster JSON file is required")
    return rosters


def _player_reference(roster: dict, player: dict) -> dict:
    return {
        "team_name": roster["team"]["name"],
        "team_abbreviation": roster["team"]["abbreviation"],
        "name": player["name"],
        "number": player["number"],
        "position": player["position"],
        "roster_source_url": roster["team"]["source_url"],
        "roster_source_file": roster["source_file"],
    }


def build_roster_metadata(rosters: list[dict], allow_team_duplicates: bool = False) -> dict:
    """Build roster indexes and report number ambiguities."""
    by_abbreviation = {}
    by_name = {}
    by_number = defaultdict(list)
    by_team_and_number = {}
    within_team_duplicates = []

    for roster in rosters:
        team = roster["team"]
        abbreviation = team["abbreviation"]
        team_name = team["name"]
        if abbreviation in by_abbreviation:
            raise RosterValidationError(f"Duplicate team abbreviation: {abbreviation}")
        name_key = team_name.casefold()
        if name_key in {name.casefold() for name in by_name}:
            raise RosterValidationError(f"Duplicate team name: {team_name}")

        team_payload = {
            "schema": roster["schema"],
            "team": team,
            "players": roster["players"],
            "player_count": len(roster["players"]),
            "source_file": roster["source_file"],
        }
        by_abbreviation[abbreviation] = team_payload
        by_name[team_name] = team_payload

        team_numbers = defaultdict(list)
        for player in roster["players"]:
            reference = _player_reference(roster, player)
            number_key = str(player["number"])
            team_numbers[number_key].append(reference)
            by_number[number_key].append(reference)
        by_team_and_number[abbreviation] = dict(sorted(team_numbers.items(), key=lambda item: int(item[0])))

        for number_key, candidates in team_numbers.items():
            if len(candidates) > 1:
                within_team_duplicates.append(
                    {
                        "team_name": team_name,
                        "team_abbreviation": abbreviation,
                        "jersey_number": int(number_key),
                        "candidate_count": len(candidates),
                        "candidates": candidates,
                    }
                )

    shared_across_teams = []
    for number_key, candidates in sorted(by_number.items(), key=lambda item: int(item[0])):
        teams = sorted({candidate["team_abbreviation"] for candidate in candidates})
        if len(teams) > 1:
            shared_across_teams.append(
                {
                    "jersey_number": int(number_key),
                    "team_abbreviations": teams,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                }
            )

    if within_team_duplicates and not allow_team_duplicates:
        descriptions = [
            "{} #{} ({})".format(
                row["team_abbreviation"],
                row["jersey_number"],
                ", ".join(candidate["name"] for candidate in row["candidates"]),
            )
            for row in within_team_duplicates
        ]
        raise RosterValidationError(
            "Duplicate jersey numbers within a team: {}. "
            "Pass --allow-team-duplicates to retain all candidates.".format("; ".join(descriptions))
        )

    jersey_number_lookup = {
        "schema": "nll_jersey_number_lookup_v1",
        "by_number": dict(sorted(by_number.items(), key=lambda item: int(item[0]))),
        "by_team_and_number": by_team_and_number,
    }
    team_roster_lookup = {
        "schema": "nll_team_roster_lookup_v1",
        "by_abbreviation": by_abbreviation,
        "by_name": by_name,
    }
    ambiguous_numbers = {
        "schema": "nll_roster_ambiguities_v1",
        "within_team_duplicates": within_team_duplicates,
        "shared_across_teams": shared_across_teams,
        "shared_jersey_numbers": [row["jersey_number"] for row in shared_across_teams],
    }
    return {
        "rosters": rosters,
        "jersey_number_lookup": jersey_number_lookup,
        "team_roster_lookup": team_roster_lookup,
        "ambiguous_numbers": ambiguous_numbers,
    }


def resolve_team_jersey(metadata: dict, team_abbreviation: str, jersey_number: object) -> dict:
    """Resolve a team abbreviation and jersey number to zero or more roster candidates."""
    abbreviation = str(team_abbreviation).strip().upper()
    try:
        number = normalize_jersey_number(jersey_number)
    except ValueError as exc:
        return {
            "team_abbreviation": abbreviation,
            "jersey_number": jersey_number,
            "status": "invalid_number",
            "confidence": "none",
            "candidates": [],
            "reason": str(exc),
        }

    lookup = metadata.get("jersey_number_lookup", metadata)
    by_team = lookup.get("by_team_and_number", {})
    if abbreviation not in by_team:
        return {
            "team_abbreviation": abbreviation,
            "jersey_number": number,
            "status": "team_not_found",
            "confidence": "none",
            "candidates": [],
        }
    candidates = list(by_team[abbreviation].get(str(number), []))
    if not candidates:
        status = "number_not_found"
        confidence = "none"
    elif len(candidates) == 1:
        status = "resolved"
        confidence = "exact_roster_lookup"
    else:
        status = "ambiguous"
        confidence = "ambiguous"
    return {
        "team_abbreviation": abbreviation,
        "jersey_number": number,
        "status": status,
        "confidence": confidence,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
