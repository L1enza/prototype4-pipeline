import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_hockey_polygon_demo_v2.py"
SPEC = importlib.util.spec_from_file_location("hockey_v2", SCRIPT)
V2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V2)


def test_torso_crop_excludes_lower_body():
    box = [100, 100, 200, 300]
    crop = V2.torso_crop_box(box, 1280, 720)
    assert crop[1] >= 100
    assert crop[3] < 220
    assert crop[0] > 100 and crop[2] < 200


def test_black_alone_does_not_choose_devils():
    features = {
        "white": 0.02, "red": 0.0, "orange": 0.0, "black": 0.80,
        "stripe": 0.0, "lab_red": 0.0,
    }
    probabilities = V2.base_crop_probabilities(features)
    assert probabilities["devils"] < 0.40
    assert max(probabilities, key=probabilities.get) != "devils"


def test_red_torso_prefers_devils_and_white_torso_prefers_flyers():
    red = {"white": 0.08, "red": 0.42, "orange": 0.02, "black": 0.28, "stripe": 0.05, "lab_red": 0.6}
    white = {"white": 0.62, "red": 0.02, "orange": 0.08, "black": 0.16, "stripe": 0.08, "lab_red": 0.0}
    assert max(V2.base_crop_probabilities(red), key=V2.base_crop_probabilities(red).get) == "devils"
    assert max(V2.base_crop_probabilities(white), key=V2.base_crop_probabilities(white).get) == "flyers"


def test_stitching_never_merges_overlapping_fragments():
    def detection(frame, x):
        probabilities = {"devils": 0.8, "flyers": 0.05, "official": 0.05, "unknown": 0.1}
        return {
            "frame_index": frame, "bbox": [x, 10, x + 30, 80],
            "torso_histogram": [1.0] + [0.0] * 71,
            "team_probabilities": probabilities,
        }
    tracks = {
        1: {"detections": [detection(0, 10), detection(5, 20)]},
        2: {"detections": [detection(5, 22), detection(10, 32)]},
    }
    stitched, events, _remaining = V2.stitch_track_fragments(tracks, max_gap=15)
    assert len(stitched) == 2
    assert events == []


def test_internal_gap_predictions_are_marked_and_labels_locked():
    def row(frame, box):
        return {
            "frame_index": frame, "bbox": box, "score": 0.9, "source": "test",
            "torso_v2": {"quality": 1.0}, "raw_crop_label": "devils",
            "team_probabilities": {"devils": 0.9, "flyers": 0.03, "official": 0.02, "unknown": 0.05},
        }
    tracks = {1: {"detections": [row(0, [10, 10, 40, 80]), row(3, [16, 10, 46, 80])]}}
    assignments = {1: {"renderable_on_ice_track": True, "team": "devils", "confidence": 0.9}}
    by_frame, stats = V2.smooth_and_fill_tracks(tracks, assignments, (128, 96), max_gap=15)
    assert by_frame[1][0]["predicted"] and by_frame[2][0]["predicted"]
    assert not by_frame[0][0]["predicted"] and not by_frame[3][0]["predicted"]
    assert all(by_frame[index][0]["team"] == "devils" for index in range(4))
    assert stats["short_detection_gap_events_bridged"] == 1


def test_sustained_legacy_devils_vote_requires_and_uses_red_torso_evidence():
    detections = []
    for frame in range(5):
        detections.append({
            "frame_index": frame,
            "bbox": [10 + frame, 10, 50 + frame, 90],
            "legacy_team": "devils",
            "raw_crop_label": "flyers",
            "team_probabilities": {"devils": 0.30, "flyers": 0.56, "official": 0.04, "unknown": 0.10},
            "torso_v2": {
                "white": 0.52, "red": 0.22, "orange": 0.01, "black": 0.16,
                "stripe": 0.12, "quality": 0.8, "blur_variance": 100.0,
            },
        })
    tracks = {1: {"track_id": 1, "raw_track_ids": [1], "detections": detections}}
    assignments, stats = V2.assign_stable_teams(tracks, minimum_detections=3)
    assert assignments[1]["team"] == "devils"
    assert assignments[1]["stabilized_label_transitions"] == 0
    assert stats["raw_per_crop_label_transitions"] == 0


def test_red_heavy_legacy_devils_track_wins_despite_white_overlap():
    detections = []
    for frame in range(6):
        detections.append({
            "frame_index": frame,
            "bbox": [10 + frame, 10, 55 + frame, 95],
            "legacy_team": "devils" if frame < 3 else "unknown",
            "raw_crop_label": "flyers",
            "team_probabilities": {"devils": 0.31, "flyers": 0.46, "official": 0.08, "unknown": 0.15},
            "torso_v2": {
                "white": 0.50, "red": 0.24, "orange": 0.01, "black": 0.14,
                "stripe": 0.10, "quality": 0.75, "blur_variance": 100.0,
            },
        })
    tracks = {1: {"track_id": 1, "raw_track_ids": [1], "detections": detections}}
    assignments, _stats = V2.assign_stable_teams(tracks, minimum_detections=3)
    assert assignments[1]["team"] == "devils"


def test_small_stripe_like_artifact_is_not_promoted_to_official():
    detections = []
    for frame in range(5):
        detections.append({
            "frame_index": frame,
            "bbox": [10, 10, 45, 58],
            "legacy_team": "official",
            "raw_crop_label": "official",
            "team_probabilities": {"devils": 0.03, "flyers": 0.08, "official": 0.79, "unknown": 0.10},
            "torso_v2": {
                "white": 0.50, "red": 0.0, "orange": 0.0, "black": 0.25,
                "stripe": 0.60, "quality": 0.8, "blur_variance": 100.0,
            },
        })
    tracks = {1: {"track_id": 1, "raw_track_ids": [1], "detections": detections}}
    assignments, _stats = V2.assign_stable_teams(tracks, minimum_detections=3)
    assert assignments[1]["team"] == "unknown"


def test_white_orange_board_without_uniform_detail_is_unknown():
    detections = []
    for frame in range(5):
        detections.append({
            "frame_index": frame,
            "bbox": [10, 10, 80, 120],
            "legacy_team": "flyers",
            "raw_crop_label": "flyers",
            "team_probabilities": {"devils": 0.01, "flyers": 0.88, "official": 0.03, "unknown": 0.08},
            "torso_v2": {
                "white": 0.80, "red": 0.0, "orange": 0.10, "black": 0.005,
                "stripe": 0.01, "quality": 0.9, "blur_variance": 120.0,
            },
        })
    tracks = {1: {"track_id": 1, "raw_track_ids": [1], "detections": detections}}
    assignments, _stats = V2.assign_stable_teams(tracks, minimum_detections=3)
    assert assignments[1]["team"] == "unknown"
