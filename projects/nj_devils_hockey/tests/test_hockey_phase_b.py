import importlib.util
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_hockey_rink_registration_and_polygons.py"
SPEC = importlib.util.spec_from_file_location("hockey_phase_b", SCRIPT)
PHASE_B = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PHASE_B)


def test_player_projection_uses_stored_broadcast_to_canonical_h_directly():
    matrix = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
    assert PHASE_B.project_broadcast_point(matrix, (5.0, 7.0)) == (20.0, 41.0)


def test_rink_geometry_validation_rejects_nonfinite_and_accepts_identity():
    line_mask = np.zeros((100, 160), dtype=np.uint8)
    cv2.rectangle(line_mask, (10, 10), (150, 90), 255, 3)
    ok, reason, stats = PHASE_B.validate_matrix_geometry(
        np.eye(3), line_mask, (160, 100), (160, 100)
    )
    assert ok and reason is None
    assert stats["visible_projected_rink_line_pixels"] >= 200
    invalid = np.eye(3)
    invalid[0, 0] = np.nan
    ok, reason, _stats = PHASE_B.validate_matrix_geometry(
        invalid, line_mask, (160, 100), (160, 100)
    )
    assert not ok and reason == "homography_nonfinite"
    ill_conditioned = np.asarray([[20.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ok, reason, stats = PHASE_B.validate_matrix_geometry(
        ill_conditioned, line_mask, (160, 100), (160, 100)
    )
    assert not ok and reason == "homography_condition_too_high"
    assert stats["normalized_condition_number"] > PHASE_B.MAX_NORMALIZED_HOMOGRAPHY_CONDITION


def test_reference_estimation_preserves_image_to_canonical_direction():
    image = np.asarray([[20.0, 10.0], [120.0, 10.0], [120.0, 80.0], [20.0, 80.0]])
    canonical = np.asarray([[10.0, 10.0], [150.0, 10.0], [150.0, 90.0], [10.0, 90.0]])
    points = [
        {"image_xy": source.tolist(), "canonical_xy": target.tolist()}
        for source, target in zip(image, canonical)
    ]
    line_mask = np.zeros((100, 160), dtype=np.uint8)
    cv2.rectangle(line_mask, (10, 10), (150, 90), 255, 3)
    reference = PHASE_B.estimate_reference_homography(
        0, points, line_mask, (160, 100), (160, 100)
    )
    assert reference["usable"]
    projected = PHASE_B.project_broadcast_point(reference["homography_matrix"], tuple(image[2]))
    assert np.allclose(projected, canonical[2], atol=1e-4)


def test_rejected_status_is_not_projection_eligible():
    records = [
        {"status": "ok", "homography_matrix": np.eye(3).tolist()},
        {"status": "homography_low_confidence", "homography_matrix": None},
        {"status": "registration_unavailable", "homography_matrix": None},
    ]
    assert [row["status"] == "ok" and row["homography_matrix"] is not None for row in records] == [True, False, False]


def test_team_polygon_requires_stable_geometry_in_both_views():
    broadcast = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]
    rink_collinear = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    rink_stable = [(10.0, 10.0), (110.0, 10.0), (10.0, 110.0)]
    assert not PHASE_B.team_polygon_is_stable(broadcast, rink_collinear)
    assert PHASE_B.team_polygon_is_stable(broadcast, rink_stable)
