"""Aggregation rules for track-level jersey inference.

The cases mirror failure modes measured on the nll_test4 manual eval set: readers
repeat confident wrong numbers, and one model call must not pose as several frames.
"""

import pytest

from prototype4_pipeline.integrations.track_jersey_inference import (
    aggregate_track,
    clean_number,
    collapse_ocr_by_source,
    parse_json_object,
    resolve_duplicate_numbers,
    source_score,
    vision_backend_status,
)

LOOKUP = {
    "by_number": {"42": [{"name": "Mark Matthews"}], "17": [{"name": "OSH player"}]},
    "by_team_and_number": {"TOR": {"42": [{"name": "Mark Matthews"}]}, "OSH": {"17": [{"name": "OSH player"}]}},
}
CONFIG = {
    "team_label_to_abbreviation": {"team_a": "TOR", "team_b": "OSH"},
    "aggregation": {"min_agreeing_frames": 2, "allow_single_frame_medium": False, "medium_min_mean_confidence": 0.55},
}
TEAM_A = {"final_class": "team_a", "confidence": 0.9}


def selected(*frames):
    return [{"frame_index": f, "crop_type": "torso"} for f in frames]


def per_frame(*reads):
    """reads: (frame_index, number, visibility) tuples."""
    return {
        "status": "complete",
        "mode": "per_frame",
        "frames": [
            {"frame_index": f, "status": "complete", "parsed": {"number": n, "visibility": v}} for f, n, v in reads
        ],
    }


def aggregate(vision, team=TEAM_A, frames=(1, 2, 3), ocr=None):
    return aggregate_track(7, team, selected(*frames), ocr or {}, vision, CONFIG, LOOKUP)


def test_two_independent_frames_on_team_roster_assign_player():
    pred = aggregate(per_frame((1, "42", "full"), (2, "42", "full")))
    assert pred["final_number"] == "42"
    assert pred["confidence"] == "high"
    assert pred["roster_player"] == {"name": "Mark Matthews"}


def test_single_read_is_never_enough():
    pred = aggregate(per_frame((1, "42", "full")))
    assert pred["final_number"] is None
    assert pred["player_name_assigned"] is False
    assert "too_few_independent_frames_for_identity" in pred["rejection_or_conflict_reasons"]


def test_track_mode_claimed_frames_are_not_independent():
    vision = {
        "status": "complete",
        "parsed": {"candidate_number": "42", "confidence": "high", "evidence_frame_ids": [1, 2, 3]},
    }
    pred = aggregate(vision)
    assert pred["final_number"] is None
    assert pred["candidate_numbers"][0]["distinct_source_frame_count"] == 0


def test_visibility_none_reads_are_ignored():
    pred = aggregate(per_frame((1, "42", "none"), (2, "42", "none")))
    assert pred["candidate_numbers"] == []
    assert pred["final_number"] is None


def test_conflicting_reads_are_not_assigned():
    pred = aggregate(per_frame((1, "42", "full"), (2, "42", "full"), (3, "47", "full")))
    assert pred["final_number"] is None


def test_number_missing_from_assigned_team_roster_is_not_assigned():
    # 17 exists on OSH but not TOR; a white (TOR) track reading 17 twice must stay unresolved.
    pred = aggregate(per_frame((1, "17", "full"), (2, "17", "full")))
    assert pred["final_number"] is None
    assert pred["roster_validation_result"]["valid_any_roster"] is True


def test_unmapped_team_never_gets_a_name():
    pred = aggregate(per_frame((1, "42", "full"), (2, "42", "full")), team={"final_class": "unknown"})
    assert pred["final_number"] is None
    assert pred["roster_player"] is None


def _assigned(track_id, number="42"):
    return {
        "track_id": track_id,
        "final_number": number,
        "confidence": "high",
        "roster_player": {"name": "Mark Matthews"},
        "player_name_assigned": True,
        "roster_validation_result": {"team_abbreviation": "TOR"},
        "rejection_or_conflict_reasons": [],
    }


def test_overlapping_same_team_duplicates_are_withdrawn():
    preds = [_assigned(2), _assigned(18)]
    withdrawn = resolve_duplicate_numbers(preds, {2: (0, 40), 18: (35, 50)})
    assert withdrawn == 2
    assert all(p["final_number"] is None and p["withdrawn_number"] == "42" for p in preds)
    assert all(not p["player_name_assigned"] for p in preds)


def test_non_overlapping_duplicates_are_kept():
    # A fragmented track (same player re-tracked later) legitimately repeats the number.
    preds = [_assigned(2), _assigned(18)]
    assert resolve_duplicate_numbers(preds, {2: (0, 30), 18: (35, 50)}) == 0
    assert all(p["final_number"] == "42" for p in preds)


def test_local_endpoint_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    local = {"vision_backend": {"name": "auto", "allow_network": True, "api_key_env": None}}
    cloud = {"vision_backend": {"name": "auto", "allow_network": True, "api_key_env": "OPENAI_API_KEY"}}
    assert vision_backend_status(local)["will_call_vision_model"] is True
    assert vision_backend_status(cloud)["will_call_vision_model"] is False


def test_leading_zero_is_the_same_number():
    assert clean_number("07") == clean_number("7") == "7"
    assert clean_number("123") is None
    pred = aggregate(per_frame((1, "42", "full"), (2, "042", "full"), (3, "42", "full")))
    assert pred["final_number"] == "42"
    lookup_seven = aggregate_track(
        7,
        {"final_class": "team_b"},
        selected(1, 2),
        {},
        per_frame((1, "07", "full"), (2, "7", "full")),
        CONFIG,
        {"by_number": {"7": [{"name": "OSH seven"}]}, "by_team_and_number": {"OSH": {"7": [{"name": "OSH seven"}]}}},
    )
    assert lookup_seven["final_number"] == "7"
    assert lookup_seven["confidence"] == "high"


def test_zero_quality_scores_are_not_replaced_by_defaults():
    worst = {"quality": {"low_motion_blur_score": 0.0, "occlusion_score": 0.0}}
    missing = {"quality": {}}
    assert source_score(worst) < source_score(missing)


def test_zero_ocr_confidence_is_not_replaced_by_default():
    rows = [{"track_id": 1, "frame_index": 1, "candidate_number": "4", "confidence": 0.0}] * 2
    votes = collapse_ocr_by_source(rows)[(1, 1, "")]["variant_candidate_votes"]
    assert votes[0]["confidence_sum"] == 0.0


@pytest.mark.parametrize("text", ["27", '"27"', "[27]", "null"])
def test_model_reply_that_is_not_an_object_is_rejected(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_model_reply_wrapped_in_prose_still_parses():
    assert parse_json_object('Sure! {"number": "42", "visibility": "full"}') == {"number": "42", "visibility": "full"}
