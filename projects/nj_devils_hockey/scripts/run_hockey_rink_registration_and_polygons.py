#!/usr/bin/env python3
"""Run hockey Phase B2/B3 from manual rink points and Phase A tracks.

Stored homographies always map broadcast-image pixels to natural canonical-rink
pixels.  The inverse is used only to draw canonical rink markings on broadcast
frames.  Registration-rejected frames never project detections or draw polygons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_hockey_polygon_demo as phase_a
from prototype4_pipeline.field_registration.homography import contains_near_collinear_triple


DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/njdevils.mp4"
DEFAULT_OUTPUT = "outputs/njdevils/hockey_polygon_demo"
DEFAULT_ANNOTATIONS = f"{DEFAULT_OUTPUT}/rink_registration_annotations/rink_keypoints.json"
DEFAULT_RINK = "assets/hockey/icerink.jpg"
REFERENCE_DIAGNOSTIC_FRAMES = [0, 70, 138, 210, 276]
REGISTRATION_CONTACT_FRAMES = [0, 35, 70, 105, 138, 175, 210, 245, 276]
MIN_REGISTRATION_CONFIDENCE = 0.50
MAX_NORMALIZED_HOMOGRAPHY_CONDITION = 15.0

TEAM_COLORS = {
    "devils": (45, 45, 235),
    "flyers": (30, 145, 255),
    "official": (40, 220, 220),
    "unknown": (160, 160, 160),
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--rink", default=DEFAULT_RINK)
    parser.add_argument("--tracking", default=f"{DEFAULT_OUTPUT}/tracking_results.json")
    parser.add_argument("--assignments", default=f"{DEFAULT_OUTPUT}/team_assignments.json")
    parser.add_argument("--min-registration-confidence", type=float, default=MIN_REGISTRATION_CONFIDENCE)
    return parser.parse_args()


def point_hull_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    return float(cv2.contourArea(cv2.convexHull(points))) if len(points) >= 3 else 0.0


def normalized_matrix_condition(
    matrix: np.ndarray,
    image_size: tuple[int, int],
    canonical_size: tuple[int, int],
) -> float:
    """Condition H after normalizing both coordinate systems to [-1, 1]."""
    image_width, image_height = image_size
    canonical_width, canonical_height = canonical_size
    image_norm = np.asarray(
        [[2.0 / max(1, image_width - 1), 0.0, -1.0],
         [0.0, 2.0 / max(1, image_height - 1), -1.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    canonical_norm = np.asarray(
        [[2.0 / max(1, canonical_width - 1), 0.0, -1.0],
         [0.0, 2.0 / max(1, canonical_height - 1), -1.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = canonical_norm @ np.asarray(matrix, dtype=np.float64) @ np.linalg.inv(image_norm)
    return float(np.linalg.cond(normalized))


def perspective_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(source, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)


def canonical_line_mask(rink_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rink_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.uint8(gray < 110) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def rink_playable_contour(rink_bgr: np.ndarray) -> np.ndarray:
    mask = canonical_line_mask(rink_bgr)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Canonical rink image contains no closed outer contour")
    return max(contours, key=cv2.contourArea)


def point_inside_rink(contour: np.ndarray, xy: tuple[float, float]) -> bool:
    if not np.all(np.isfinite(np.asarray(xy, dtype=np.float64))):
        return False
    return cv2.pointPolygonTest(contour, (float(xy[0]), float(xy[1])), False) >= 0


def visible_rink_line_pixels(
    matrix: np.ndarray,
    line_mask: np.ndarray,
    image_size: tuple[int, int],
) -> int:
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
        warped = cv2.warpPerspective(line_mask, inverse, image_size, flags=cv2.INTER_NEAREST)
    except (cv2.error, np.linalg.LinAlgError):
        return 0
    return int(np.count_nonzero(warped))


def validate_matrix_geometry(
    matrix: np.ndarray,
    line_mask: np.ndarray,
    image_size: tuple[int, int],
    canonical_size: tuple[int, int],
) -> tuple[bool, str | None, dict]:
    stats: dict[str, float | int] = {}
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return False, "homography_nonfinite", stats
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return False, "homography_singular", stats
    try:
        condition = normalized_matrix_condition(matrix, image_size, canonical_size)
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return False, "homography_inverse_failed", stats
    stats["normalized_condition_number"] = condition
    if not math.isfinite(condition) or condition > MAX_NORMALIZED_HOMOGRAPHY_CONDITION:
        return False, "homography_condition_too_high", stats
    visible_pixels = visible_rink_line_pixels(matrix, line_mask, image_size)
    stats["visible_projected_rink_line_pixels"] = visible_pixels
    if visible_pixels < 200:
        return False, "projected_rink_geometry_not_visible", stats
    return True, None, stats


def grouped_annotations(document: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in document.get("annotations", []):
        if row.get("visibility", "visible") != "visible":
            continue
        image_xy = row.get("image_xy")
        canonical_xy = row.get("canonical_xy")
        if not isinstance(image_xy, list) or len(image_xy) != 2:
            continue
        if not isinstance(canonical_xy, list) or len(canonical_xy) != 2:
            continue
        grouped[int(row["frame_index"])].append(row)
    return dict(grouped)


def estimate_reference_homography(
    frame_index: int,
    points: list[dict],
    line_mask: np.ndarray,
    image_size: tuple[int, int],
    canonical_size: tuple[int, int],
) -> dict:
    row = {
        "frame_index": int(frame_index),
        "view_name": f"frame_{frame_index:06d}",
        "point_count": len(points),
        "usable": False,
        "homography_matrix": None,
        "failure_reason": None,
    }
    if len(points) < 4:
        row["failure_reason"] = "needs_at_least_4_points"
        return row
    image_points = np.asarray([item["image_xy"] for item in points], dtype=np.float64)
    canonical_points = np.asarray([item["canonical_xy"] for item in points], dtype=np.float64)
    image_hull = point_hull_area(image_points)
    canonical_hull = point_hull_area(canonical_points)
    row.update({
        "image_hull_area": image_hull,
        "canonical_hull_area": canonical_hull,
        "image_hull_area_ratio": image_hull / float(image_size[0] * image_size[1]),
        "canonical_hull_area_ratio": canonical_hull / float(canonical_size[0] * canonical_size[1]),
    })
    if image_hull < 0.008 * image_size[0] * image_size[1]:
        row["failure_reason"] = "image_hull_area_too_small"
        return row
    if canonical_hull < 0.015 * canonical_size[0] * canonical_size[1]:
        row["failure_reason"] = "canonical_hull_area_too_small"
        return row
    if len(points) == 4 and contains_near_collinear_triple(canonical_points):
        row["failure_reason"] = "canonical_points_contain_collinear_triple"
        return row
    if len(points) == 4 and contains_near_collinear_triple(image_points):
        row["failure_reason"] = "image_points_contain_collinear_triple"
        return row
    if np.linalg.matrix_rank(image_points - image_points.mean(axis=0), tol=1.0) < 2:
        row["failure_reason"] = "image_points_collinear"
        return row
    if np.linalg.matrix_rank(canonical_points - canonical_points.mean(axis=0), tol=1.0) < 2:
        row["failure_reason"] = "canonical_points_collinear"
        return row
    matrix, inlier_mask = cv2.findHomography(
        image_points.astype(np.float32),
        canonical_points.astype(np.float32),
        cv2.RANSAC,
        6.0,
    )
    if matrix is None or inlier_mask is None:
        row["failure_reason"] = "cv2_findHomography_failed"
        return row
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / len(points))
    row.update({"inlier_count": inlier_count, "inlier_ratio": inlier_ratio, "inlier_mask": inliers.astype(int).tolist()})
    if inlier_count < 4:
        row["failure_reason"] = "insufficient_ransac_inliers"
        return row
    if inlier_ratio < 0.40:
        row["failure_reason"] = "ransac_inlier_ratio_too_low"
        return row
    try:
        projected_canonical = perspective_points(matrix, image_points[inliers])
        inverse = np.linalg.inv(matrix)
        projected_image = perspective_points(inverse, canonical_points[inliers])
    except (cv2.error, np.linalg.LinAlgError):
        row["failure_reason"] = "bidirectional_reprojection_failed"
        return row
    forward_errors = np.linalg.norm(projected_canonical - canonical_points[inliers], axis=1)
    inverse_errors = np.linalg.norm(projected_image - image_points[inliers], axis=1)
    canonical_diagonal = math.hypot(*canonical_size)
    image_diagonal = math.hypot(*image_size)
    row.update({
        "mean_forward_reprojection_error_px": float(np.mean(forward_errors)),
        "max_forward_reprojection_error_px": float(np.max(forward_errors)),
        "mean_inverse_reprojection_error_px": float(np.mean(inverse_errors)),
        "max_inverse_reprojection_error_px": float(np.max(inverse_errors)),
        "mean_normalized_bidirectional_error": float(
            0.5 * (np.mean(forward_errors) / canonical_diagonal + np.mean(inverse_errors) / image_diagonal)
        ),
    })
    if not np.all(np.isfinite(forward_errors)) or not np.all(np.isfinite(inverse_errors)):
        row["failure_reason"] = "bidirectional_reprojection_nonfinite"
        return row
    if float(np.mean(inverse_errors)) > 25.0 or float(np.max(inverse_errors)) > 80.0:
        row["failure_reason"] = "inverse_reprojection_error_too_high"
        return row
    if float(np.mean(forward_errors)) > 20.0 or float(np.max(forward_errors)) > 50.0:
        row["failure_reason"] = "forward_reprojection_error_too_high"
        return row
    geometry_ok, reason, geometry = validate_matrix_geometry(matrix, line_mask, image_size, canonical_size)
    row.update(geometry)
    if not geometry_ok:
        row["failure_reason"] = reason
        return row
    confidence = min(
        0.99,
        0.50 + 0.20 * min(1.0, inlier_ratio / 0.70) + 0.18 * min(1.0, inlier_count / 6.0)
        + 0.11 * max(0.0, 1.0 - row["mean_normalized_bidirectional_error"] / 0.02),
    )
    row.update({
        "usable": True,
        "failure_reason": None,
        "confidence": float(confidence),
        "homography_matrix": np.asarray(matrix, dtype=float).tolist(),
        "projection_direction": "broadcast_image_to_canonical_rink_uses_H",
    })
    return row


def optical_flow_update(
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    source_homography: np.ndarray,
) -> tuple[np.ndarray | None, dict, str | None]:
    height, width = source_gray.shape
    mask = np.zeros_like(source_gray)
    mask[int(round(height * 0.30)) :, :] = 255
    source_points = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=600,
        qualityLevel=0.008,
        minDistance=6,
        mask=mask,
        blockSize=5,
    )
    stats: dict[str, float | int] = {"detected_features": 0, "tracked_features": 0, "flow_inliers": 0}
    if source_points is None or len(source_points) < 20:
        return None, stats, "insufficient_optical_flow_features"
    stats["detected_features"] = int(len(source_points))
    lk = dict(
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    target_points, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(
        source_gray, target_gray, source_points, None, **lk
    )
    if target_points is None or forward_status is None:
        return None, stats, "optical_flow_forward_failed"
    back_points, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
        target_gray, source_gray, target_points, None, **lk
    )
    if back_points is None or backward_status is None:
        return None, stats, "optical_flow_backward_failed"
    forward_backward = np.linalg.norm(back_points - source_points, axis=2).reshape(-1)
    good = (
        (forward_status.reshape(-1) > 0)
        & (backward_status.reshape(-1) > 0)
        & np.isfinite(forward_backward)
        & (forward_backward < 1.5)
    )
    source_good = source_points.reshape(-1, 2)[good]
    target_good = target_points.reshape(-1, 2)[good]
    stats["tracked_features"] = int(len(source_good))
    if len(source_good) < 20:
        return None, stats, "insufficient_bidirectional_flow_tracks"
    image_motion, inlier_mask = cv2.findHomography(source_good, target_good, cv2.RANSAC, 2.5)
    if image_motion is None or inlier_mask is None:
        return None, stats, "flow_findHomography_failed"
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / len(source_good))
    stats.update({"flow_inliers": inlier_count, "flow_inlier_ratio": inlier_ratio})
    if inlier_count < 20 or inlier_ratio < 0.45:
        return None, stats, "optical_flow_inliers_insufficient"
    predicted = perspective_points(image_motion, source_good[inliers])
    errors = np.linalg.norm(predicted - target_good[inliers], axis=1)
    median_error = float(np.median(errors))
    stats["median_flow_reprojection_error_px"] = median_error
    if not math.isfinite(median_error) or median_error > 1.75:
        return None, stats, "optical_flow_reprojection_error_too_high"
    try:
        target_homography = np.asarray(source_homography, dtype=np.float64) @ np.linalg.inv(image_motion)
        target_homography /= target_homography[2, 2]
    except (np.linalg.LinAlgError, FloatingPointError):
        return None, stats, "optical_flow_homography_composition_failed"
    return target_homography, stats, None


def build_dynamic_registration(
    frames: list[np.ndarray],
    references: list[dict],
    line_mask: np.ndarray,
    canonical_size: tuple[int, int],
    min_confidence: float = MIN_REGISTRATION_CONFIDENCE,
) -> list[dict]:
    if not frames:
        return []
    height, width = frames[0].shape[:2]
    image_size = (width, height)
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    usable = {int(row["frame_index"]): row for row in references if row.get("usable")}
    records: dict[int, dict] = {}
    for frame_index, reference in usable.items():
        records[frame_index] = {
            "frame_index": frame_index,
            "status": "ok",
            "confidence": float(reference["confidence"]),
            "homography_matrix": reference["homography_matrix"],
            "candidate_homography_matrix": reference["homography_matrix"],
            "reference_view": reference["view_name"],
            "reference_frame_index": frame_index,
            "reference_frame_distance": 0,
            "propagation_method": "manual_reference_reacquisition",
            "failure_reason": None,
            "projection_direction": "broadcast_image_to_canonical_rink_uses_H",
            "visible_projected_rink_line_pixels": reference.get("visible_projected_rink_line_pixels", 0),
        }
    if not usable:
        return [{
            "frame_index": index,
            "status": "registration_unavailable",
            "confidence": 0.0,
            "homography_matrix": None,
            "candidate_homography_matrix": None,
            "failure_reason": "no_usable_manual_reference",
            "propagation_method": "none",
        } for index in range(len(frames))]

    usable_indices = sorted(usable)
    anchor_for_frame = {
        frame_index: min(usable_indices, key=lambda anchor: (abs(anchor - frame_index), anchor))
        for frame_index in range(len(frames))
    }
    for anchor in usable_indices:
        for direction in (-1, 1):
            candidate = np.asarray(usable[anchor]["homography_matrix"], dtype=np.float64)
            minimum_flow_quality = 1.0
            previous = anchor
            target = anchor + direction
            while 0 <= target < len(frames) and anchor_for_frame[target] == anchor:
                updated, flow_stats, flow_error = optical_flow_update(
                    gray_frames[previous], gray_frames[target], candidate
                )
                distance = abs(target - anchor)
                if updated is None:
                    records[target] = {
                        "frame_index": target,
                        "status": "registration_unavailable",
                        "confidence": 0.0,
                        "homography_matrix": None,
                        "candidate_homography_matrix": None,
                        "reference_view": usable[anchor]["view_name"],
                        "reference_frame_index": anchor,
                        "reference_frame_distance": distance,
                        "propagation_method": "optical_flow_update_failed",
                        "flow_source_frame_index": previous,
                        "failure_reason": flow_error,
                        **flow_stats,
                    }
                    break
                candidate = updated
                geometry_ok, geometry_error, geometry_stats = validate_matrix_geometry(
                    candidate, line_mask, image_size, canonical_size
                )
                ratio = float(flow_stats.get("flow_inlier_ratio", 0.0))
                median_error = float(flow_stats.get("median_flow_reprojection_error_px", 99.0))
                flow_quality = min(1.0, ratio / 0.70) * math.exp(-median_error / 4.0)
                minimum_flow_quality = min(minimum_flow_quality, flow_quality)
                confidence = (
                    float(usable[anchor]["confidence"])
                    * math.exp(-distance / 80.0)
                    * (0.78 + 0.22 * minimum_flow_quality)
                )
                status = "ok" if geometry_ok and confidence >= min_confidence else "homography_low_confidence"
                reason = geometry_error if not geometry_ok else (
                    "propagated_confidence_below_threshold" if confidence < min_confidence else None
                )
                records[target] = {
                    "frame_index": target,
                    "status": status,
                    "confidence": float(confidence),
                    "homography_matrix": candidate.astype(float).tolist() if status == "ok" else None,
                    "candidate_homography_matrix": candidate.astype(float).tolist(),
                    "reference_view": usable[anchor]["view_name"],
                    "reference_frame_index": anchor,
                    "reference_frame_distance": distance,
                    "propagation_method": "sparse_bidirectional_optical_flow",
                    "flow_source_frame_index": previous,
                    "failure_reason": reason,
                    "projection_direction": "broadcast_image_to_canonical_rink_uses_H",
                    **flow_stats,
                    **geometry_stats,
                }
                previous = target
                target += direction
    return [records.get(index, {
        "frame_index": index,
        "status": "registration_unavailable",
        "confidence": 0.0,
        "homography_matrix": None,
        "candidate_homography_matrix": None,
        "failure_reason": "optical_flow_chain_unavailable",
        "propagation_method": "none",
    }) for index in range(len(frames))]


def project_broadcast_point(matrix: list[list[float]], xy: tuple[float, float]) -> tuple[float, float] | None:
    """Project a player footpoint with stored broadcast->canonical H directly."""
    try:
        projected = perspective_points(np.asarray(matrix, dtype=np.float64), np.asarray([xy], dtype=np.float64))[0]
    except (cv2.error, ValueError, TypeError):
        return None
    if not np.all(np.isfinite(projected)):
        return None
    return float(projected[0]), float(projected[1])


def draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.52,
) -> None:
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(
        image,
        (x, max(0, y - text_height - 7)),
        (min(image.shape[1] - 1, x + text_width + 8), min(image.shape[0] - 1, y + baseline + 2)),
        (12, 12, 12),
        -1,
    )
    cv2.putText(image, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def registration_header(
    image: np.ndarray,
    record: dict,
    devils_count: int = 0,
    flyers_count: int = 0,
    title: str = "HOCKEY DYNAMIC RINK REGISTRATION",
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 82), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.80, image, 0.20, 0.0, image)
    status = str(record.get("status", "registration_unavailable"))
    status_color = (80, 235, 80) if status == "ok" else ((0, 185, 255) if status == "homography_low_confidence" else (60, 60, 240))
    cv2.putText(image, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"frame {int(record.get('frame_index', -1)):03d} | status={status} | confidence={float(record.get('confidence') or 0.0):.3f}",
        (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.57, status_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"ref={record.get('reference_frame_index')} | {record.get('propagation_method')} | H: broadcast -> rink | DEV {devils_count}  PHI {flyers_count}",
        (16, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (235, 235, 235), 1, cv2.LINE_AA,
    )


def rejection_banner(image: np.ndarray, record: dict) -> None:
    status = record.get("status")
    label = "LOW CONFIDENCE - REGISTRATION REJECTED" if status == "homography_low_confidence" else "REGISTRATION UNAVAILABLE"
    y0, y1 = image.shape[0] - 86, image.shape[0] - 16
    overlay = image.copy()
    cv2.rectangle(overlay, (24, y0), (image.shape[1] - 24, y1), (0, 0, 210), -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0.0, image)
    cv2.putText(image, label, (48, y0 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(
        image, f"No player projection or team polygon | reason={record.get('failure_reason')}",
        (48, y0 + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (255, 255, 255), 2, cv2.LINE_AA,
    )


def overlay_projected_rink_lines(
    frame: np.ndarray,
    matrix: list[list[float]],
    line_mask: np.ndarray,
) -> int:
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
        warped = cv2.warpPerspective(
            line_mask,
            inverse,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
    except (cv2.error, np.linalg.LinAlgError):
        return 0
    warped = cv2.dilate(warped, np.ones((3, 3), np.uint8))
    visible = warped > 0
    if np.any(visible):
        tint = np.zeros_like(frame)
        tint[:, :] = (40, 255, 40)
        frame[visible] = np.uint8(0.18 * frame[visible].astype(np.float32) + 0.82 * tint[visible].astype(np.float32))
    return int(np.count_nonzero(visible))


def draw_team_polygon(image: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int]) -> bool:
    if len(points) < 3:
        return False
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)).astype(np.int32)
    if len(hull) < 3 or cv2.contourArea(hull) < 30.0:
        return False
    overlay = image.copy()
    cv2.fillPoly(overlay, [hull], color)
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0, image)
    cv2.polylines(image, [hull], True, color, 4, cv2.LINE_AA)
    return True


def team_polygon_is_stable(
    broadcast_points: list[tuple[float, float]],
    rink_points: list[tuple[float, float]],
) -> bool:
    """Require a nondegenerate hull in both views before drawing either polygon."""
    if len(broadcast_points) < 3 or len(rink_points) < 3:
        return False
    broadcast_hull = cv2.convexHull(np.asarray(broadcast_points, dtype=np.float32).reshape(-1, 1, 2))
    rink_hull = cv2.convexHull(np.asarray(rink_points, dtype=np.float32).reshape(-1, 1, 2))
    return bool(
        len(broadcast_hull) >= 3
        and len(rink_hull) >= 3
        and cv2.contourArea(broadcast_hull) >= 30.0
        and cv2.contourArea(rink_hull) >= 30.0
    )


def draw_visible_region(rink_panel: np.ndarray, matrix: list[list[float]], broadcast_size: tuple[int, int]) -> None:
    width, height = broadcast_size
    corners = np.asarray([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])
    try:
        projected = perspective_points(np.asarray(matrix, dtype=np.float64), corners)
    except cv2.error:
        return
    if np.all(np.isfinite(projected)):
        polygon = np.round(projected).astype(np.int32).reshape(-1, 1, 2)
        overlay = rink_panel.copy()
        cv2.fillPoly(overlay, [polygon], (255, 220, 80))
        cv2.addWeighted(overlay, 0.10, rink_panel, 0.90, 0.0, rink_panel)
        cv2.polylines(rink_panel, [polygon], True, (220, 120, 20), 3, cv2.LINE_AA)


def montage(images: list[np.ndarray], labels: list[str], columns: int = 3, tile_size: tuple[int, int] = (640, 360)) -> np.ndarray:
    if not images:
        return np.full((360, 640, 3), 245, dtype=np.uint8)
    tile_width, tile_height = tile_size
    rows = int(math.ceil(len(images) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        thumb = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * tile_height
        canvas[y : y + tile_height, x : x + tile_width] = thumb
        cv2.rectangle(canvas, (x, y), (x + tile_width - 1, y + 36), (10, 10, 10), -1)
        cv2.putText(canvas, labels[index], (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def rink_heatmap(
    rink: np.ndarray,
    points: list[tuple[float, float]],
    title: str,
    color_map: int,
    playable_contour: np.ndarray,
) -> np.ndarray:
    height, width = rink.shape[:2]
    accumulation = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if point_inside_rink(playable_contour, (x, y)):
            cv2.circle(accumulation, (int(round(x)), int(round(y))), 26, 1.0, -1, cv2.LINE_AA)
    accumulation = cv2.GaussianBlur(accumulation, (0, 0), 28)
    if float(accumulation.max()) > 0:
        accumulation /= float(accumulation.max())
    colored = cv2.applyColorMap(np.uint8(np.clip(accumulation * 255.0, 0, 255)), color_map)
    alpha = np.clip(accumulation[:, :, None] * 0.82, 0.0, 0.82)
    result = np.uint8(rink.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha)
    cv2.rectangle(result, (0, 0), (width, 62), (12, 12, 12), -1)
    cv2.putText(result, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def decode_video(path: Path) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    metadata = {
        "path": str(path),
        "fps": fps,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    metadata["decoded_frame_count"] = len(frames)
    metadata["duration_seconds"] = len(frames) / fps if fps else None
    return frames, metadata


def verify_video(path: Path, expected_frames: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    first_shape = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        if first_shape is None:
            first_shape = list(frame.shape)
    cap.release()
    return {
        "path": str(path),
        "decoded_frames": decoded,
        "expected_frames": expected_frames,
        "fully_decodable": decoded == expected_frames,
        "frame_shape": first_shape,
    }


def main() -> int:
    args = parse_args()
    video_path = project_path(args.video)
    output_dir = project_path(args.output_dir)
    annotations_path = project_path(args.annotations)
    rink_path = project_path(args.rink)
    tracking_path = project_path(args.tracking)
    assignments_path = project_path(args.assignments)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "rink_registration_full_resolution_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    frames, video_metadata = decode_video(video_path)
    if len(frames) != 277:
        raise ValueError(f"Expected all 277 source frames, decoded {len(frames)}")
    rink = cv2.imread(str(rink_path), cv2.IMREAD_COLOR)
    if rink is None:
        raise FileNotFoundError(rink_path)
    canonical_size = (rink.shape[1], rink.shape[0])
    if canonical_size != (1568, 980):
        raise ValueError(f"Expected natural icerink.jpg dimensions 1568x980, got {canonical_size}")
    image_size = (frames[0].shape[1], frames[0].shape[0])
    line_mask = canonical_line_mask(rink)
    playable_contour = rink_playable_contour(rink)

    annotations = load_json(annotations_path)
    grouped = grouped_annotations(annotations)
    references = [
        estimate_reference_homography(frame_index, points, line_mask, image_size, canonical_size)
        for frame_index, points in sorted(grouped.items())
    ]
    records = build_dynamic_registration(
        frames, references, line_mask, canonical_size, args.min_registration_confidence
    )
    if len(records) != len(frames):
        raise RuntimeError("Registration did not produce exactly one record per source frame")

    tracking = load_json(tracking_path)
    assignments_payload = load_json(assignments_path)
    assignments = {int(row["track_id"]): row for row in assignments_payload.get("assignments", [])}
    detections_by_frame: dict[int, list[dict]] = defaultdict(list)
    for detection in tracking.get("detections", []):
        detections_by_frame[int(detection["source_frame_index"])].append(detection)

    fps = float(video_metadata["fps"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    broadcast_path = output_dir / "broadcast_team_polygons_rink_registered.mp4"
    rink_video_path = output_dir / "rink_team_polygons.mp4"
    diagnostic_video_path = output_dir / "rink_registration_diagnostic_overlay.mp4"
    side_path = output_dir / "side_by_side_team_polygons.mp4"
    broadcast_writer = cv2.VideoWriter(str(broadcast_path), fourcc, fps, image_size)
    diagnostic_writer = cv2.VideoWriter(str(diagnostic_video_path), fourcc, fps, image_size)
    rink_writer = cv2.VideoWriter(str(rink_video_path), fourcc, fps, canonical_size)
    canonical_panel_width = int(round(canonical_size[0] * image_size[1] / canonical_size[1]))
    side_size = (image_size[0] + canonical_panel_width, image_size[1])
    side_writer = cv2.VideoWriter(str(side_path), fourcc, fps, side_size)
    if not all(writer.isOpened() for writer in (broadcast_writer, diagnostic_writer, rink_writer, side_writer)):
        raise RuntimeError("Could not open one or more Phase B video writers")

    observations: list[dict] = []
    observation_points = {"devils": [], "flyers": []}
    stats = Counter()
    stats["total_frames"] = len(frames)
    stats["total_detections"] = len(tracking.get("detections", []))
    record_by_frame = {int(row["frame_index"]): row for row in records}
    reference_points_by_frame = grouped
    registration_diagnostic_images: dict[int, np.ndarray] = {}
    registered_images: dict[int, np.ndarray] = {}
    rink_polygon_images: dict[int, np.ndarray] = {}
    rejected_polygon_count = 0

    csv_rows = []
    for frame_index, raw in enumerate(frames):
        record = record_by_frame[frame_index]
        status = record["status"]
        stats[f"registration_{status}_frames"] += 1
        if record.get("propagation_method") == "manual_reference_reacquisition":
            stats["frames_reacquired_from_manual_references"] += 1
        elif record.get("propagation_method") == "sparse_bidirectional_optical_flow":
            stats["frames_using_optical_flow_updates"] += 1

        diagnostic = raw.copy()
        broadcast = raw.copy()
        rink_panel = rink.copy()
        visible_line_pixels = 0
        if status == "ok":
            visible_line_pixels = overlay_projected_rink_lines(diagnostic, record["homography_matrix"], line_mask)
            draw_visible_region(rink_panel, record["homography_matrix"], image_size)
        else:
            rejection_banner(diagnostic, record)

        for annotation in reference_points_by_frame.get(frame_index, []):
            image_xy = annotation["image_xy"]
            point = (int(round(image_xy[0])), int(round(image_xy[1])))
            cv2.circle(diagnostic, point, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(diagnostic, point, 11, (255, 255, 255), 2, cv2.LINE_AA)

        team_broadcast_points: dict[str, list[tuple[float, float]]] = {"devils": [], "flyers": []}
        team_rink_points: dict[str, list[tuple[float, float]]] = {"devils": [], "flyers": []}
        frame_detections = detections_by_frame.get(frame_index, [])
        for detection in frame_detections:
            track_id = int(detection["track_id"])
            assignment = assignments.get(track_id, {})
            team = str(assignment.get("team", detection.get("team", "unknown")))
            renderable = bool(assignment.get("renderable_on_ice_track", False))
            if not renderable:
                stats["nonrenderable_detections_excluded"] += 1
                continue
            color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
            box = [int(value) for value in detection["bbox"]]
            cv2.rectangle(broadcast, (box[0], box[1]), (box[2], box[3]), color, 2, cv2.LINE_AA)
            draw_text_box(
                broadcast,
                f"T{track_id} {team.upper()} {float(assignment.get('confidence') or 0.0):.2f}",
                (box[0], max(20, box[1])),
                color,
                0.43,
            )
            if team == "official":
                stats["official_observations_excluded"] += 1
                continue
            if team not in ("devils", "flyers"):
                stats["unknown_observations_excluded"] += 1
                continue
            stats["eligible_team_detections_considered"] += 1
            if status != "ok":
                stats["detections_rejected_by_registration"] += 1
                continue
            foot = (float(detection["footpoint"][0]), float(detection["footpoint"][1]))
            projected = project_broadcast_point(record["homography_matrix"], foot)
            if projected is None:
                stats["nonfinite_projections"] += 1
                continue
            if not point_inside_rink(playable_contour, projected):
                stats["out_of_rink_projections"] += 1
                continue
            stats[f"{team}_projected_observations"] += 1
            team_broadcast_points[team].append(foot)
            team_rink_points[team].append(projected)
            observation_points[team].append(projected)
            cv2.circle(broadcast, (int(round(foot[0])), int(round(foot[1]))), 5, color, -1, cv2.LINE_AA)
            cv2.circle(rink_panel, (int(round(projected[0])), int(round(projected[1]))), 8, color, -1, cv2.LINE_AA)
            observation = {
                "source_frame_index": frame_index,
                "track_id": track_id,
                "team": team,
                "team_confidence": float(assignment.get("confidence") or 0.0),
                "broadcast_footpoint": [foot[0], foot[1]],
                "canonical_rink_xy": [projected[0], projected[1]],
                "registration_status": status,
                "registration_confidence": float(record["confidence"]),
                "reference_frame_index": int(record["reference_frame_index"]),
                "projection_direction": "broadcast_footpoint_to_canonical_rink_uses_stored_H_directly",
            }
            observations.append(observation)
            csv_rows.append(observation)

        frame_polygons = 0
        for team in ("devils", "flyers"):
            polygon_stable = status == "ok" and team_polygon_is_stable(
                team_broadcast_points[team], team_rink_points[team]
            )
            if polygon_stable:
                broadcast_drawn = draw_team_polygon(broadcast, team_broadcast_points[team], TEAM_COLORS[team])
                rink_drawn = draw_team_polygon(rink_panel, team_rink_points[team], TEAM_COLORS[team])
                if not broadcast_drawn or not rink_drawn:
                    raise RuntimeError(f"Stable polygon failed to render at frame {frame_index} for {team}")
                stats[f"frames_with_{team}_polygon"] += 1
                frame_polygons += 1
            else:
                stats[f"frames_with_insufficient_{team}_points_for_polygon"] += 1
                if status == "ok":
                    stats[f"ok_frames_with_insufficient_{team}_points_for_polygon"] += 1
        if status != "ok" and frame_polygons:
            rejected_polygon_count += frame_polygons

        if status != "ok":
            rejection_banner(broadcast, record)
            rejection_banner(rink_panel, record)
        registration_header(diagnostic, record, len(team_rink_points["devils"]), len(team_rink_points["flyers"]))
        registration_header(
            broadcast, record, len(team_rink_points["devils"]), len(team_rink_points["flyers"]),
            "RINK-REGISTERED TEAM POLYGONS",
        )
        registration_header(
            rink_panel, record, len(team_rink_points["devils"]), len(team_rink_points["flyers"]),
            "CANONICAL RINK TEAM POLYGONS",
        )
        record["renderer_visible_rink_line_pixels"] = visible_line_pixels
        record["devils_projected_player_count"] = len(team_rink_points["devils"])
        record["flyers_projected_player_count"] = len(team_rink_points["flyers"])
        record["team_polygons_drawn"] = frame_polygons

        diagnostic_writer.write(diagnostic)
        broadcast_writer.write(broadcast)
        rink_writer.write(rink_panel)
        rink_resized = cv2.resize(rink_panel, (canonical_panel_width, image_size[1]), interpolation=cv2.INTER_AREA)
        side = np.hstack([broadcast, rink_resized])
        side_writer.write(side)

        if frame_index in REGISTRATION_CONTACT_FRAMES:
            registration_diagnostic_images[frame_index] = diagnostic.copy()
        if frame_index in REFERENCE_DIAGNOSTIC_FRAMES:
            registered_images[frame_index] = broadcast.copy()
            rink_polygon_images[frame_index] = rink_panel.copy()
            cv2.imwrite(str(diagnostics_dir / f"frame_{frame_index:06d}_registration_overlay.png"), diagnostic)
            cv2.imwrite(str(diagnostics_dir / f"frame_{frame_index:06d}_broadcast_polygons.png"), broadcast)
            cv2.imwrite(str(diagnostics_dir / f"frame_{frame_index:06d}_rink_polygons.png"), rink_panel)

    for writer in (broadcast_writer, diagnostic_writer, rink_writer, side_writer):
        writer.release()

    if rejected_polygon_count != 0:
        raise RuntimeError(f"Drew {rejected_polygon_count} polygons on rejected-registration frames")
    stats["polygons_drawn_on_rejected_registration_frames"] = rejected_polygon_count

    devils_heatmap_path = output_dir / "devils_rink_heatmap.png"
    flyers_heatmap_path = output_dir / "flyers_rink_heatmap.png"
    cv2.imwrite(
        str(devils_heatmap_path),
        rink_heatmap(rink, observation_points["devils"], "NEW JERSEY DEVILS - CANONICAL RINK HEATMAP", cv2.COLORMAP_HOT, playable_contour),
    )
    cv2.imwrite(
        str(flyers_heatmap_path),
        rink_heatmap(rink, observation_points["flyers"], "PHILADELPHIA FLYERS - CANONICAL RINK HEATMAP", cv2.COLORMAP_TURBO, playable_contour),
    )

    registration_contact_path = output_dir / "rink_registration_contact_sheet.png"
    registration_diagnostics_path = output_dir / "rink_registration_diagnostics.png"
    contact_indices = sorted(registration_diagnostic_images)
    contact = montage(
        [registration_diagnostic_images[index] for index in contact_indices],
        [f"frame {index} | {record_by_frame[index]['status']} | ref {record_by_frame[index].get('reference_frame_index')}" for index in contact_indices],
    )
    cv2.imwrite(str(registration_contact_path), contact)
    cv2.imwrite(str(registration_diagnostics_path), contact)
    representative_registered_path = output_dir / "representative_registered_frames.png"
    representative_rink_path = output_dir / "representative_rink_polygon_frames.png"
    diagnostic_indices = sorted(registered_images)
    cv2.imwrite(
        str(representative_registered_path),
        montage([registered_images[index] for index in diagnostic_indices], [f"broadcast frame {index}" for index in diagnostic_indices]),
    )
    cv2.imwrite(
        str(representative_rink_path),
        montage([rink_polygon_images[index] for index in diagnostic_indices], [f"canonical frame {index}" for index in diagnostic_indices]),
    )

    projected_json_path = output_dir / "projected_rink_observations.json"
    projected_csv_path = output_dir / "projected_rink_observations.csv"
    write_json(projected_json_path, {
        "coordinate_space": "natural pixels of assets/hockey/icerink.jpg",
        "homography_direction": "broadcast image -> canonical rink; player footpoints use stored H directly",
        "observations": observations,
    })
    with projected_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source_frame_index", "track_id", "team", "team_confidence", "broadcast_foot_x", "broadcast_foot_y",
            "canonical_rink_x", "canonical_rink_y", "registration_status", "registration_confidence", "reference_frame_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({
                "source_frame_index": row["source_frame_index"],
                "track_id": row["track_id"],
                "team": row["team"],
                "team_confidence": row["team_confidence"],
                "broadcast_foot_x": row["broadcast_footpoint"][0],
                "broadcast_foot_y": row["broadcast_footpoint"][1],
                "canonical_rink_x": row["canonical_rink_xy"][0],
                "canonical_rink_y": row["canonical_rink_xy"][1],
                "registration_status": row["registration_status"],
                "registration_confidence": row["registration_confidence"],
                "reference_frame_index": row["reference_frame_index"],
            })

    per_frame_path = output_dir / "per_frame_rink_homographies.json"
    registration_summary_path = output_dir / "rink_registration_summary.json"
    phase_b_summary_path = output_dir / "phase_b_summary.json"
    write_json(per_frame_path, {
        "homography_direction": "broadcast image coordinates -> canonical rink coordinates",
        "canonical_rink": str(rink_path),
        "canonical_size": list(canonical_size),
        "frames": records,
    })
    status_counts = Counter(row["status"] for row in records)
    registration_summary = {
        "stage": "hockey_phase_b2_dynamic_rink_registration",
        "status": "complete" if status_counts["ok"] else "failed",
        "annotation_source": str(annotations_path),
        "annotation_count": sum(len(rows) for rows in grouped.values()),
        "reference_homographies": references,
        "counts": {
            "total_frames": len(records),
            "reference_views": len(references),
            "usable_reference_homographies": sum(bool(row.get("usable")) for row in references),
            "valid_registration_frames": status_counts["ok"],
            "low_confidence_frames": status_counts["homography_low_confidence"],
            "unavailable_frames": status_counts["registration_unavailable"],
            "frames_using_optical_flow_updates": stats["frames_using_optical_flow_updates"],
            "frames_reacquired_from_manual_references": stats["frames_reacquired_from_manual_references"],
        },
        "homography_direction": {
            "stored_H": "broadcast image coordinates -> canonical rink coordinates",
            "player_footpoint_projection": "use H directly",
            "canonical_lines_on_broadcast": "use inverse(H)",
        },
        "validation": {
            "minimum_correspondences": 4,
            "ransac_threshold_canonical_px": 6.0,
            "minimum_ransac_inliers": 4,
            "minimum_inlier_ratio": 0.40,
            "maximum_normalized_matrix_condition": MAX_NORMALIZED_HOMOGRAPHY_CONDITION,
            "minimum_visible_projected_line_pixels": 200,
            "registration_confidence_threshold": args.min_registration_confidence,
        },
    }
    write_json(registration_summary_path, registration_summary)

    video_verification = [
        verify_video(path, len(frames))
        for path in (diagnostic_video_path, broadcast_path, rink_video_path, side_path)
    ]
    stats["valid_registration_frames"] = status_counts["ok"]
    stats["low_confidence_frames"] = status_counts["homography_low_confidence"]
    stats["unavailable_frames"] = status_counts["registration_unavailable"]
    stats["rejected_registration_frames"] = (
        status_counts["homography_low_confidence"] + status_counts["registration_unavailable"]
    )
    stats["usable_reference_homographies"] = sum(bool(row.get("usable")) for row in references)
    stats["official_tracks_excluded"] = sum(
        1 for row in assignments.values() if row.get("renderable_on_ice_track") and row.get("team") == "official"
    )
    stats["unknown_tracks_excluded"] = sum(
        1 for row in assignments.values() if row.get("renderable_on_ice_track") and row.get("team") == "unknown"
    )
    stats["devils_tracks"] = sum(
        1 for row in assignments.values() if row.get("renderable_on_ice_track") and row.get("team") == "devils"
    )
    stats["flyers_tracks"] = sum(
        1 for row in assignments.values() if row.get("renderable_on_ice_track") and row.get("team") == "flyers"
    )
    for required_counter in (
        "nonfinite_projections",
        "detections_rejected_by_registration",
        "registration_homography_low_confidence_frames",
        "registration_registration_unavailable_frames",
    ):
        stats[required_counter] += 0
    phase_b_summary = {
        "stage": "hockey_polygon_demo_phase_b2_b3",
        "status": "complete" if all(row["fully_decodable"] for row in video_verification) else "video_verification_failed",
        "video": video_metadata,
        "canonical_rink": {"path": str(rink_path), "width": canonical_size[0], "height": canonical_size[1]},
        "inputs": {
            "annotations": str(annotations_path),
            "phase_a_tracking": str(tracking_path),
            "phase_a_team_assignments": str(assignments_path),
        },
        "counts": dict(stats),
        "video_verification": video_verification,
        "safety_checks": {
            "registration_rejected_frames_project_no_players": True,
            "registration_rejected_frames_draw_no_polygons": rejected_polygon_count == 0,
            "stale_coordinates_reused": False,
            "player_projection_direction": "stored broadcast-to-canonical H used directly",
        },
        "outputs": {
            "rink_registration_diagnostic_overlay": str(diagnostic_video_path),
            "broadcast_team_polygons_rink_registered": str(broadcast_path),
            "rink_team_polygons": str(rink_video_path),
            "side_by_side_team_polygons": str(side_path),
            "devils_rink_heatmap": str(devils_heatmap_path),
            "flyers_rink_heatmap": str(flyers_heatmap_path),
            "rink_registration_contact_sheet": str(registration_contact_path),
            "rink_registration_diagnostics": str(registration_diagnostics_path),
            "representative_registered_frames": str(representative_registered_path),
            "representative_rink_polygon_frames": str(representative_rink_path),
            "full_resolution_diagnostics": str(diagnostics_dir),
            "per_frame_rink_homographies": str(per_frame_path),
            "rink_registration_summary": str(registration_summary_path),
            "projected_rink_observations_json": str(projected_json_path),
            "projected_rink_observations_csv": str(projected_csv_path),
            "phase_b_summary": str(phase_b_summary_path),
        },
    }
    write_json(phase_b_summary_path, phase_b_summary)
    print(json.dumps(phase_b_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
