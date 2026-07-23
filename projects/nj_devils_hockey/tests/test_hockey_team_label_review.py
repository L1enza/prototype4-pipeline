"""Direct tests for the downstream hockey team-label review pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_hockey_team_labels as review


def features(**updates) -> dict:
    base = {
        "white": 0.05, "red": 0.02, "orange": 0.0, "black": 0.08,
        "stripe": 0.05, "red_hue_consistency": 0.0, "red_saturation": 0.0,
        "white_brightness": 0.0, "quality": 0.9, "blur_variance": 150.0,
        "crop_box": [0, 0, 10, 10],
    }
    return base | updates


def assignment(team: str = "unknown", legacy_votes: dict | None = None) -> dict:
    return {
        "team": team,
        "confidence": 0.60,
        "legacy_team_votes": legacy_votes or {},
    }


def samples(feature: dict, count: int = 5) -> list[dict]:
    return [{"frame_index": index * 5, "features": dict(feature)} for index in range(count)]


def test_upper_torso_excludes_black_pants() -> None:
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[:105] = (0, 0, 230)
    result = review.upper_torso_features(frame, [0, 0, 99, 199])
    assert result["red"] > 0.90
    assert result["black"] < 0.03


def test_multiple_independent_red_crops_suggest_devils() -> None:
    red = features(red=0.42, red_hue_consistency=0.98, red_saturation=0.92)
    result = review.suggest_track_label(samples(red), assignment())
    assert result["label"] == "devils"
    assert result["crop_votes"]["devils"] == 5


def test_multiple_independent_white_crops_suggest_flyers() -> None:
    white = features(white=0.79, black=0.06, white_brightness=0.91)
    result = review.suggest_track_label(samples(white), assignment())
    assert result["label"] == "flyers"
    assert result["crop_votes"]["flyers"] == 5


def test_black_alone_remains_unknown() -> None:
    black = features(white=0.02, red=0.0, black=0.82)
    result = review.suggest_track_label(samples(black), assignment())
    assert result["label"] == "unknown"


def test_white_board_without_uniform_detail_remains_unknown() -> None:
    board = features(white=0.91, red=0.0, orange=0.08, black=0.004, white_brightness=0.96)
    result = review.suggest_track_label(samples(board), assignment(legacy_votes={"flyers": 9}))
    assert result["label"] == "unknown"


def test_known_v2_label_is_not_destabilized_by_noisy_crops() -> None:
    noisy = features(white=0.02, red=0.40, red_hue_consistency=0.98, red_saturation=0.90)
    result = review.suggest_track_label(samples(noisy), assignment(team="flyers", legacy_votes={"flyers": 9}))
    assert result["label"] == "flyers"


def test_reliable_original_official_plus_stripes_suggests_official() -> None:
    striped = features(white=0.45, black=0.28, stripe=0.58, white_brightness=0.85)
    result = review.suggest_track_label(samples(striped), assignment(legacy_votes={"official": 9, "unknown": 1}))
    assert result["label"] == "official"


def test_blurred_legacy_official_artifact_remains_unknown() -> None:
    artifact = features(white=0.72, black=0.04, stripe=0.26, quality=0.45, white_brightness=0.9)
    result = review.suggest_track_label(samples(artifact), assignment(legacy_votes={"official": 10}))
    assert result["label"] == "unknown"


def test_diverse_crops_are_unique_and_capped() -> None:
    rows = [{"frame_index": index, "features": features(quality=0.3 + index / 100)} for index in range(20)]
    selected = review.diverse_best_samples(rows)
    assert 6 <= len(selected) <= 10
    assert len({row["frame_index"] for row in selected}) == len(selected)


def test_override_parser_requires_all_accepted_tracks(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"overrides": [{"track_id": 1, "label": "devils"}]}))
    try:
        review.load_overrides(path, {1, 2})
    except ValueError as error:
        assert "missing" in str(error).lower()
    else:
        raise AssertionError("Incomplete override file was accepted")


def test_override_parser_accepts_complete_list(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"overrides": [
        {"track_id": 1, "label": "devils", "reviewed": True},
        {"track_id": 2, "label": "flyers", "reviewed": True},
    ]}))
    assert review.load_overrides(path, {1, 2}) == {1: "devils", 2: "flyers"}


def test_override_parser_rejects_unreviewed_export(tmp_path: Path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"overrides": [
        {"track_id": 1, "label": "devils", "reviewed": False},
    ]}))
    try:
        review.load_overrides(path, {1})
    except ValueError as error:
        assert "unreviewed" in str(error).lower()
    else:
        raise AssertionError("Unreviewed browser export was accepted")


def test_reviewer_html_has_all_four_track_level_buttons() -> None:
    record = {
        "track_id": 7, "current_v2_label": "unknown", "current_v2_confidence": 0.6,
        "legacy_label": None, "legacy_match_fraction": 0.0, "legacy_votes": {},
        "observation_count": 3, "frame_start": 2, "frame_end": 4,
        "torso_crop_count": 3, "short_track_crop_notice": "Only 3 distinct observations.",
        "crops": [], "context_path": "assets/context.jpg", "context_frame_index": 3,
        "suggestion": {
            "label": "unknown", "confidence": 0.6, "basis": "visual review required",
            "crop_votes": {label: 0 for label in review.VALID_LABELS},
            "evidence": {key: 0.0 for key in (
                "white", "red", "orange", "black", "stripe", "red_hue_consistency",
                "red_saturation", "white_brightness", "quality", "blur_variance",
            )},
            "high_quality_crops_used": 0, "legacy_reliable_label": None,
            "legacy_reliable_fraction": 0.0,
        },
    }
    document = review.reviewer_html([record])
    assert document.count('class="track-card"') == 1
    for label in review.VALID_LABELS:
        assert f'data-label="{label}"' in document
    assert "team_label_overrides.json" in document


def test_final_contact_sheet_handles_equal_candidate_scores() -> None:
    frame = np.full((120, 120, 3), 220, dtype=np.uint8)
    rows = {
        0: [
            {"track_id": 1, "team": "flyers", "team_confidence": 0.8, "predicted": False, "bbox": [10, 10, 40, 90]},
            {"track_id": 2, "team": "flyers", "team_confidence": 0.8, "predicted": False, "bbox": [60, 10, 90, 90]},
        ]
    }
    sheet = review.final_contact_sheet([frame], rows)
    assert sheet.shape == (680, 1140, 3)
