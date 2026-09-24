"""Team labels reach the roster only through a human confirmation for the exact assignments."""

import json
import subprocess
import sys

from conftest import PROJECT_ROOT
from prototype4_pipeline.integrations.team_mapping import (
    SCHEMA,
    assignment_fingerprint,
    label_color_summary,
    load_confirmed_mapping,
)
from prototype4_pipeline.integrations.track_jersey_inference import run_track_level_inference

WHITE = [85.0, 0.0, 2.0]
MAROON = [25.0, 30.0, 10.0]
ASSIGNMENTS = {
    5: {"track_id": 5, "final_class": "team_a", "representative_median_lab": WHITE},
    6: {"track_id": 6, "final_class": "team_b", "representative_median_lab": MAROON},
}
KNOWN = {"TOR", "OSH"}


def confirmation(assignments=ASSIGNMENTS, **overrides):
    payload = {
        "schema": SCHEMA,
        "assignment_fingerprint": assignment_fingerprint(assignments),
        "mapping": {"team_a": "TOR", "team_b": "OSH"},
        "confirmed_by": "reviewer",
        "confirmed_at_utc": "2026-09-24T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def swapped(assignments):
    flip = {"team_a": "team_b", "team_b": "team_a"}
    return {tid: {**row, "final_class": flip[row["final_class"]]} for tid, row in assignments.items()}


def test_missing_confirmation_maps_nothing(tmp_path):
    result = load_confirmed_mapping(tmp_path / "absent.json", ASSIGNMENTS, KNOWN)
    assert result["status"] == "missing"
    assert result["mapping"] == {}


def test_matching_confirmation_is_used(tmp_path):
    path = write(tmp_path / "c.json", confirmation())
    result = load_confirmed_mapping(path, ASSIGNMENTS, KNOWN)
    assert result["status"] == "confirmed"
    assert result["mapping"] == {"team_a": "TOR", "team_b": "OSH"}


def test_swapped_labels_make_confirmation_stale(tmp_path):
    path = write(tmp_path / "c.json", confirmation())
    result = load_confirmed_mapping(path, swapped(ASSIGNMENTS), KNOWN)
    assert result["status"] == "stale"
    assert result["mapping"] == {}


def test_bad_confirmations_are_rejected(tmp_path):
    bad = [
        confirmation(mapping={"team_a": "TOR", "team_b": "TOR"}),
        confirmation(mapping={"team_a": "TOR"}),
        confirmation(mapping={"team_a": "TOR", "team_b": "XYZ"}),
        confirmation(confirmed_by=""),
        confirmation(schema="something_else"),
    ]
    for index, payload in enumerate(bad):
        result = load_confirmed_mapping(write(tmp_path / f"c{index}.json", payload), ASSIGNMENTS, KNOWN)
        assert result["status"] == "invalid", payload
        assert result["mapping"] == {}
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{not json", encoding="utf-8")
    assert load_confirmed_mapping(garbled, ASSIGNMENTS, KNOWN)["status"] == "invalid"


def test_color_summary_tells_light_from_dark():
    summary = label_color_summary(ASSIGNMENTS)
    assert summary["team_a"]["looks"].startswith("light")
    assert summary["team_b"]["looks"].startswith("dark")


def build_stage_inputs(tmp_path, assignments):
    """Smallest set of files the jersey stage reads: track 5 read as 42 on two frames."""
    regions = [
        {"track_id": 5, "frame_index": f, "crop_type": "torso", "variant_name": "center_enlarged_rgb", "region_path": f"r{f}.png"}
        for f in (1, 2, 3)
    ]
    ocr = [{"track_id": 5, "frame_index": f, "crop_type": "torso", "candidate_number": "42", "confidence": 0.8} for f in (1, 2)]
    lookup = {
        "by_number": {"42": [{"name": "Mark Matthews"}]},
        "by_team_and_number": {"TOR": {"42": [{"name": "Mark Matthews"}]}, "OSH": {}},
    }
    inputs = {
        "enhanced_number_region_manifest": write(tmp_path / "manifest.json", {"regions": regions}),
        "clean_crop_metadata": tmp_path / "absent_clean.json",
        "crop_ocr_predictions": write(tmp_path / "ocr.json", {"predictions": ocr}),
        "tracking_metadata": tmp_path / "absent_tracking.json",
        "team_assignments": write(tmp_path / "teams.json", list(assignments.values())),
        "jersey_number_lookup": write(tmp_path / "lookup.json", lookup),
        "team_mapping_confirmation": tmp_path / "confirmation.json",
    }
    return {
        "inputs": {key: str(value) for key, value in inputs.items()},
        "outputs": {"output_dir": str(tmp_path / "out")},
        "selection": {"min_source_frames_per_track": 3, "max_source_frames_per_track": 5},
        "vision_backend": {"allow_network": False},
        "aggregation": {"min_agreeing_frames": 2},
        # A mapping typed into the config must be ignored; only the confirmation counts.
        "team_label_to_abbreviation": {"team_a": "TOR", "team_b": "OSH"},
    }


def track_5(tmp_path):
    predictions = json.loads((tmp_path / "out" / "track_jersey_predictions.json").read_text(encoding="utf-8"))
    return next(row for row in predictions["tracks"] if row["track_id"] == 5)


def test_stage_withholds_names_until_confirmed(tmp_path):
    config = build_stage_inputs(tmp_path, ASSIGNMENTS)

    summary = run_track_level_inference(config)
    assert summary["team_mapping"]["status"] == "missing"
    assert track_5(tmp_path)["final_number"] is None
    assert summary["counts"]["player_names_assigned"] == 0

    write(tmp_path / "confirmation.json", confirmation())
    summary = run_track_level_inference(config)
    assert summary["team_mapping"]["status"] == "confirmed"
    assert track_5(tmp_path)["final_number"] == "42"
    assert track_5(tmp_path)["roster_player"] == {"name": "Mark Matthews"}


def test_stage_withholds_names_when_labels_swap_after_confirmation(tmp_path):
    write(tmp_path / "confirmation.json", confirmation())
    config = build_stage_inputs(tmp_path, swapped(ASSIGNMENTS))
    summary = run_track_level_inference(config)
    assert summary["team_mapping"]["status"] == "stale"
    assert track_5(tmp_path)["roster_player"] is None


def test_confirm_script_reviews_then_writes(tmp_path):
    config = build_stage_inputs(tmp_path, ASSIGNMENTS)
    config_path = write(tmp_path / "config.json", config)
    script = [sys.executable, str(PROJECT_ROOT / "scripts" / "confirm_team_mapping.py"), "--config", str(config_path)]

    review = subprocess.run(script, capture_output=True, text=True, check=True)
    assert "light" in review.stdout and "dark" in review.stdout
    assert not (tmp_path / "confirmation.json").exists()

    subprocess.run(script + ["--team-a", "tor", "--team-b", "osh", "--confirmed-by", "reviewer"], check=True, capture_output=True)
    written = json.loads((tmp_path / "confirmation.json").read_text(encoding="utf-8"))
    assert written["mapping"] == {"team_a": "TOR", "team_b": "OSH"}
    assert load_confirmed_mapping(tmp_path / "confirmation.json", ASSIGNMENTS, KNOWN)["status"] == "confirmed"
