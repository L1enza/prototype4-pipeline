#!/usr/bin/env python3
"""Confirm which team-assignment label (team_a / team_b) is which real team.

Team assignment names clusters by size, not by team, so a person must confirm the
mapping for each team-assignment result before jersey inference assigns names.

Review first (prints each label's track count and average shirt colour):
    python scripts/confirm_team_mapping.py

Then confirm, after checking the team assignment overlay or contact sheets:
    python scripts/confirm_team_mapping.py --team-a TOR --team-b OSH --confirmed-by "your name"

If team assignment is re-run and the labels change, jersey inference reports the
confirmation as stale and withholds names until this is run again.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype4_pipeline.integrations.team_mapping import (  # noqa: E402
    SCHEMA,
    assignment_fingerprint,
    label_color_summary,
)
from prototype4_pipeline.integrations.track_jersey_inference import (  # noqa: E402
    DEFAULT_CONFIG,
    load_team_assignments,
    project_path,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and confirm the team_a/team_b to real-team mapping.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Jersey inference config to read input paths from.")
    parser.add_argument("--assignments", default=None, help="Team assignments JSON. Defaults to the config's input.")
    parser.add_argument("--output", default=None, help="Confirmation file. Defaults to the config's input.")
    parser.add_argument("--team-a", default=None, help="Roster abbreviation for team_a, e.g. TOR.")
    parser.add_argument("--team-b", default=None, help="Roster abbreviation for team_b, e.g. OSH.")
    parser.add_argument("--confirmed-by", default=None, help="Who reviewed the footage.")
    parser.add_argument("--notes", default="", help="What was checked, e.g. uniform colours seen.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(project_path(args.config))
    inputs = config.get("inputs", {})
    assignments_path = project_path(args.assignments or inputs["team_assignments"])
    output_path = project_path(args.output or inputs["team_mapping_confirmation"])

    assignments = load_team_assignments(assignments_path)
    if not assignments:
        print(f"No team assignments found at {assignments_path}", file=sys.stderr)
        return 1
    summary = label_color_summary(assignments)
    fingerprint = assignment_fingerprint(assignments)

    print(f"Team assignments: {assignments_path}")
    print(f"Fingerprint:      {fingerprint}\n")
    for label, info in summary.items():
        print(f"  {label:<10} {info['track_count']:>3} tracks   looks {info['looks']}   mean L*a*b {info['mean_shirt_lab']}")
    print()

    if not (args.team_a or args.team_b):
        print("Review only; nothing written. Check the team assignment overlay or contact sheets, then re-run with")
        print('  --team-a <ABBR> --team-b <ABBR> --confirmed-by "<name>"')
        return 0
    if not (args.team_a and args.team_b and args.confirmed_by):
        print("Confirming needs --team-a, --team-b and --confirmed-by together.", file=sys.stderr)
        return 2
    if args.team_a.upper() == args.team_b.upper():
        print("team_a and team_b must be different teams.", file=sys.stderr)
        return 2

    confirmation = {
        "schema": SCHEMA,
        "team_assignments": str(assignments_path),
        "assignment_fingerprint": fingerprint,
        "mapping": {"team_a": args.team_a.upper(), "team_b": args.team_b.upper()},
        "confirmed_by": args.confirmed_by,
        "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        "label_color_summary_at_confirmation": summary,
    }
    write_json(output_path, confirmation)
    print(f"Wrote {output_path}")
    print(json.dumps(confirmation["mapping"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
