#!/usr/bin/env python3
"""V2 team assignment and team-polygon rendering from existing nll_test4 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prototype4_pipeline.integrations.team_appearance_features import load_appearance_backend, resnet18_preflight_metadata
import run_team_assignment_and_polygon_demo as v1


TEAM_COLORS = {
    "team_a": (220, 40, 40),
    "team_b": (30, 105, 230),
    "official": (245, 210, 35),
    "unknown": (145, 145, 145),
}

KNOWN_LIMITATIONS = [
    "No OCR, player names, or player identities are assigned.",
    "Pretrained appearance embeddings are used only when cached or explicitly allowed.",
    "Color-only fallback may still confuse white uniforms, officials, goalies, shadows, and similar uniforms.",
    "Officials and unknown tracks are excluded from team polygons.",
    "Static homography drift and track ID switches still affect field analytics.",
    "Scorebug priors are optional and disabled by default.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run V2 team assignment and team polygon demo.")
    parser.add_argument("--config", default="configs/nll_test4_team_assignment_v2.json")
    parser.add_argument("--tracking-metadata", default=None)
    parser.add_argument("--projected-points", default=None)
    parser.add_argument("--clean-crop-metadata", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--field-template", default=None)
    parser.add_argument("--team-output-dir", default=None)
    parser.add_argument("--polygon-output-dir", default=None)
    parser.add_argument("--embedding-backend", choices=["auto", "torchvision_resnet18", "none"], default=None)
    parser.add_argument("--allow-download-weights", action="store_true")
    parser.add_argument("--preflight-weights", action="store_true", help="Print ResNet-18 model/cache/training metadata and exit.")
    parser.add_argument("--color-only-assignments", default="outputs/nll_test4/team_assignment_demo_v2/track_team_assignments_v2.json")
    parser.add_argument("--color-only-summary", default="outputs/nll_test4/team_assignment_demo_v2/team_assignment_summary_v2.json")
    return parser.parse_args()


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_config(args):
    cfg = load_json(project_path(args.config))
    inputs = cfg.setdefault("inputs", {})
    outputs = cfg.setdefault("outputs", {})
    if args.tracking_metadata:
        inputs["tracking_metadata"] = args.tracking_metadata
    if args.projected_points:
        inputs["projected_points"] = args.projected_points
    if args.clean_crop_metadata:
        inputs["clean_crop_metadata"] = args.clean_crop_metadata
    if args.video:
        inputs["video"] = args.video
    if args.field_template:
        inputs["field_template"] = args.field_template
    if args.team_output_dir:
        outputs["team_assignment_dir"] = args.team_output_dir
    if args.polygon_output_dir:
        outputs["team_polygon_dir"] = args.polygon_output_dir
    if args.embedding_backend:
        cfg.setdefault("appearance", {})["embedding_backend"] = args.embedding_backend
    if args.allow_download_weights:
        cfg.setdefault("appearance", {})["allow_download_weights"] = True
    return cfg


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(1.0 - float(np.dot(a, b)) / denom)


def normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def entropy_from_hist(hist):
    hist = np.asarray(hist, dtype=np.float32)
    total = float(np.sum(hist))
    if total <= 1e-9:
        return 0.0
    p = hist / total
    p = p[p > 1e-9]
    return float(-np.sum(p * np.log2(p)))


def dominant_lab_colors(lab_pixels, weights=None, k=3, max_iter=20):
    if lab_pixels.size == 0:
        return []
    pixels = lab_pixels.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 2500:
        idx = np.linspace(0, len(pixels) - 1, 2500).astype(np.int64)
        pixels = pixels[idx]
    k = min(k, len(pixels))
    if k <= 0:
        return []
    centers = [pixels[0]]
    for _ in range(1, k):
        dists = np.min(np.linalg.norm(pixels[:, None, :] - np.asarray(centers)[None, :, :], axis=2), axis=1)
        centers.append(pixels[int(np.argmax(dists))])
    centers = np.asarray(centers, dtype=np.float32)
    labels = np.zeros(len(pixels), dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for c in range(k):
            members = pixels[new_labels == c]
            if len(members):
                new_centers[c] = np.median(members, axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels = new_labels
        centers = new_centers
    out = []
    for c in range(k):
        frac = float(np.mean(labels == c))
        out.append({"lab": [float(x) for x in centers[c].tolist()], "fraction": frac})
    out.sort(key=lambda item: item["fraction"], reverse=True)
    return out


def detection_score(det, clean_quality_by_key):
    key = (int(det["track_id"]), int(det["frame_index"]), "torso")
    clean_q = clean_quality_by_key.get(key, 0.0)
    area = float(det.get("bbox_area") or 0.0)
    area_score = min(1.0, area / 9000.0)
    sam = float(det.get("sam_confidence_score") or 0.0)
    return 0.35 * sam + 0.35 * area_score + 0.30 * clean_q


def select_diverse_observations(rows, clean_quality_by_key, max_count, min_spacing):
    ranked = sorted(rows, key=lambda det: detection_score(det, clean_quality_by_key), reverse=True)
    selected = []
    used_frames = []
    for det in ranked:
        frame_index = int(det["frame_index"])
        if any(abs(frame_index - other) < min_spacing for other in used_frames):
            continue
        selected.append(det)
        used_frames.append(frame_index)
        if len(selected) >= max_count:
            break
    if len(selected) < min(max_count, len(ranked)):
        for det in ranked:
            if det in selected:
                continue
            selected.append(det)
            if len(selected) >= max_count:
                break
    selected.sort(key=lambda det: int(det["frame_index"]))
    return selected


def build_clean_crop_index(clean_crop_metadata_path):
    index = {}
    if not clean_crop_metadata_path or not Path(clean_crop_metadata_path).exists():
        return index
    payload = load_json(clean_crop_metadata_path)
    for crop in payload.get("crops", []):
        key = (int(crop["track_id"]), int(crop["frame_index"]), crop.get("crop_type"))
        current = index.get(key)
        if current is None or float(crop.get("crop_quality_score") or 0.0) > float(current.get("crop_quality_score") or 0.0):
            index[key] = crop
    return index


def crop_torso_from_frame(frame_rgb, bbox, fractions):
    h, w = frame_rgb.shape[:2]
    x0, y0, x1, y1 = v1.clamp_box(bbox, w, h)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    tx0 = x0 + int(round(float(fractions["x_start"]) * bw))
    tx1 = x0 + int(round(float(fractions["x_end"]) * bw))
    ty0 = y0 + int(round(float(fractions["y_start"]) * bh))
    ty1 = y0 + int(round(float(fractions["y_end"]) * bh))
    tx0, ty0, tx1, ty1 = v1.clamp_box({"x0": tx0, "y0": ty0, "x1": tx1, "y1": ty1}, w, h)
    return frame_rgb[ty0 : ty1 + 1, tx0 : tx1 + 1], {"x0": tx0, "y0": ty0, "x1": tx1, "y1": ty1}


def mask_for_torso(det, torso_box):
    mask_path = det.get("mask_path")
    if not mask_path or not Path(mask_path).exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    x0, y0, x1, y1 = torso_box["x0"], torso_box["y0"], torso_box["x1"], torso_box["y1"]
    if y1 >= mask.shape[0] or x1 >= mask.shape[1]:
        mask = cv2.resize(mask, (max(x1 + 1, mask.shape[1]), max(y1 + 1, mask.shape[0])), interpolation=cv2.INTER_NEAREST)
    return mask[y0 : y1 + 1, x0 : x1 + 1] > 0


def valid_uniform_pixels(crop_rgb, crop_mask):
    if crop_rgb.size == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros(crop_rgb.shape[:2], dtype=bool)
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    greenish_floor = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 95) & (sat > 55) & (val < 190)
    valid = (val > 28) & (val < 248) & (~greenish_floor)
    if crop_mask is not None and crop_mask.shape == valid.shape:
        valid &= crop_mask
    if np.count_nonzero(valid) < max(20, int(0.04 * valid.size)):
        valid = (val > 25) & (val < 252)
        if crop_mask is not None and crop_mask.shape == valid.shape:
            valid &= crop_mask
    pixels = crop_rgb[valid]
    return pixels, valid


def color_features_from_crop(crop_rgb, crop_mask):
    pixels, valid = valid_uniform_pixels(crop_rgb, crop_mask)
    if len(pixels) < 12:
        return None
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    valid_u8 = valid.astype(np.uint8)
    hist_h = cv2.calcHist([hsv], [0], valid_u8, [24], [0, 180]).flatten().astype(np.float32)
    hist_s = cv2.calcHist([hsv], [1], valid_u8, [12], [0, 256]).flatten().astype(np.float32)
    hist_lab_a = cv2.calcHist([lab], [1], valid_u8, [12], [0, 256]).flatten().astype(np.float32)
    hist_lab_b = cv2.calcHist([lab], [2], valid_u8, [12], [0, 256]).flatten().astype(np.float32)
    hist = np.concatenate([hist_h, hist_s, hist_lab_a, hist_lab_b]).astype(np.float32)
    hist = normalize_vector(hist)
    hsv_pixels = hsv[valid]
    lab_pixels = lab[valid]
    val = hsv_pixels[:, 2].astype(np.float32)
    sat = hsv_pixels[:, 1].astype(np.float32)
    median_hsv = np.median(hsv_pixels, axis=0).astype(np.float32)
    median_lab = np.median(lab_pixels, axis=0).astype(np.float32)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.mean(edges[valid] > 0)) if np.count_nonzero(valid) else 0.0
    return {
        "color_vector": hist,
        "median_hsv": median_hsv,
        "median_lab": median_lab,
        "dominant_lab_colors": dominant_lab_colors(lab_pixels, k=3),
        "white_fraction": float(np.mean((sat < 45) & (val > 170))),
        "dark_fraction": float(np.mean(val < 65)),
        "neutral_fraction": float(np.mean(sat < 45)),
        "color_entropy": entropy_from_hist(hist),
        "valid_pixel_coverage": float(np.count_nonzero(valid) / max(1, valid.size)),
        "edge_density": edge_density,
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def observation_quality(det, feats, clean_crop):
    area = float(det.get("bbox_area") or 0.0)
    area_score = min(1.0, area / 9000.0)
    sam = float(det.get("sam_confidence_score") or 0.0)
    valid = float(feats.get("valid_pixel_coverage") or 0.0)
    sharp = min(1.0, math.log1p(float(feats.get("sharpness") or 0.0)) / math.log1p(1200.0))
    clean_q = float(clean_crop.get("crop_quality_score") or 0.0) if clean_crop else 0.0
    return max(0.0, min(1.0, 0.25 * sam + 0.24 * area_score + 0.24 * valid + 0.17 * sharp + 0.10 * clean_q))


def build_track_observations(track_id, rows, frame_lookup, frames_by_source, clean_index, backend, cfg):
    obs_cfg = cfg["observation_selection"]
    fractions = cfg["torso_crop_fractions"]
    clean_quality_by_key = {key: float(value.get("crop_quality_score") or 0.0) for key, value in clean_index.items()}
    selected = select_diverse_observations(rows, clean_quality_by_key, int(obs_cfg["max_observations_per_track"]), int(obs_cfg["min_frame_spacing"]))
    observations = []
    for det in selected:
        frame_index = int(det["frame_index"])
        source_info = frame_lookup.get(frame_index)
        if not source_info:
            continue
        frame = frames_by_source.get(int(source_info["source_frame_index"]))
        if frame is None:
            continue
        clean_crop = clean_index.get((track_id, frame_index, "torso"))
        crop_rgb, torso_box = crop_torso_from_frame(frame, det["bbox_2d"], fractions)
        crop_mask = mask_for_torso(det, torso_box)
        feats = color_features_from_crop(crop_rgb, crop_mask)
        if feats is None:
            continue
        embedding = backend.embed_rgb(crop_rgb)
        quality = observation_quality(det, feats, clean_crop)
        observations.append({
            "track_id": int(track_id),
            "frame_index": frame_index,
            "source_frame_index": int(source_info["source_frame_index"]),
            "mask_id": int(det.get("mask_id", -1)),
            "bbox_2d": det.get("bbox_2d"),
            "torso_box": torso_box,
            "sam_confidence_score": float(det.get("sam_confidence_score") or 0.0),
            "clean_crop_path": clean_crop.get("crop_path") if clean_crop else None,
            "color_vector": feats["color_vector"],
            "median_hsv": feats["median_hsv"],
            "median_lab": feats["median_lab"],
            "dominant_lab_colors": feats["dominant_lab_colors"],
            "white_fraction": feats["white_fraction"],
            "dark_fraction": feats["dark_fraction"],
            "neutral_fraction": feats["neutral_fraction"],
            "color_entropy": feats["color_entropy"],
            "valid_pixel_coverage": feats["valid_pixel_coverage"],
            "edge_density": feats["edge_density"],
            "observation_quality_score": float(quality),
            "appearance_embedding": embedding,
            "crop_rgb": crop_rgb,
        })
    return observations


def reject_outliers(observations, cfg, appearance_available):
    if len(observations) < 4:
        return observations, []
    color_stack = np.stack([obs["color_vector"] for obs in observations])
    color_center = normalize_vector(np.median(color_stack, axis=0))
    color_dist = np.array([cosine_distance(obs["color_vector"], color_center) for obs in observations], dtype=np.float32)
    app_dist = np.zeros(len(observations), dtype=np.float32)
    if appearance_available and all(obs["appearance_embedding"] is not None for obs in observations):
        emb_stack = np.stack([obs["appearance_embedding"] for obs in observations])
        emb_center = normalize_vector(np.median(emb_stack, axis=0))
        app_dist = np.array([cosine_distance(obs["appearance_embedding"], emb_center) for obs in observations], dtype=np.float32)
    reject_cfg = cfg["outlier_rejection"]
    reject = color_dist > float(reject_cfg["color_cosine_distance_threshold"])
    if appearance_available:
        reject |= app_dist > float(reject_cfg["appearance_cosine_distance_threshold"])
    max_reject = int(math.floor(len(observations) * float(reject_cfg["max_reject_fraction"])))
    if int(np.count_nonzero(reject)) > max_reject:
        combined = color_dist + app_dist
        keep_count = max(1, len(observations) - max_reject)
        keep_idx = set(int(i) for i in np.argsort(combined)[:keep_count])
        reject = np.array([i not in keep_idx for i in range(len(observations))], dtype=bool)
    kept = [obs for obs, r in zip(observations, reject) if not r]
    rejected = []
    for obs, r, cd, ad in zip(observations, reject, color_dist, app_dist):
        if r:
            rejected.append({"frame_index": obs["frame_index"], "color_distance": float(cd), "appearance_distance": float(ad)})
    return kept, rejected


def aggregate_signature(track_id, observations, rejected, cfg, evidence_mode):
    if not observations:
        return None
    weights = np.array([max(0.05, obs["observation_quality_score"]) for obs in observations], dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), 1e-8)
    color_stack = np.stack([obs["color_vector"] for obs in observations])
    color_signature = normalize_vector(np.sum(color_stack * weights[:, None], axis=0))
    median_hsv = np.sum(np.stack([obs["median_hsv"] for obs in observations]) * weights[:, None], axis=0)
    median_lab = np.sum(np.stack([obs["median_lab"] for obs in observations]) * weights[:, None], axis=0)
    appearance_available = observations[0].get("appearance_embedding") is not None
    appearance_signature = None
    if appearance_available:
        emb_stack = np.stack([obs["appearance_embedding"] for obs in observations])
        appearance_signature = normalize_vector(np.sum(emb_stack * weights[:, None], axis=0))
    color_distances = [cosine_distance(obs["color_vector"], color_signature) for obs in observations]
    app_distances = []
    if appearance_signature is not None:
        app_distances = [cosine_distance(obs["appearance_embedding"], appearance_signature) for obs in observations]
    color_consistency = float(max(0.0, 1.0 - statistics.mean(color_distances))) if color_distances else 0.0
    appearance_consistency = float(max(0.0, 1.0 - statistics.mean(app_distances))) if app_distances else None
    quality = float(min(1.0, 0.45 * (len(observations) / max(1, cfg["observation_selection"]["max_observations_per_track"])) + 0.35 * statistics.mean([obs["observation_quality_score"] for obs in observations]) + 0.20 * color_consistency))
    official_score = official_likelihood(observations)
    dominant = merge_dominant_colors(observations)
    feature_weights = cfg["feature_weights"]
    if appearance_signature is not None:
        combined = np.concatenate([
            appearance_signature * float(feature_weights["appearance"]),
            color_signature * float(feature_weights["color"]),
        ])
    else:
        combined = color_signature
    return {
        "track_id": int(track_id),
        "observations_selected": len(observations) + len(rejected),
        "observations_used": len(observations),
        "observations_rejected": len(rejected),
        "rejected_observations": rejected,
        "color_signature": color_signature,
        "appearance_signature": appearance_signature,
        "combined_signature": normalize_vector(combined),
        "median_hsv": median_hsv,
        "median_lab": median_lab,
        "dominant_lab_colors": dominant,
        "white_fraction": float(np.mean([obs["white_fraction"] for obs in observations])),
        "dark_fraction": float(np.mean([obs["dark_fraction"] for obs in observations])),
        "neutral_fraction": float(np.mean([obs["neutral_fraction"] for obs in observations])),
        "color_entropy": float(np.mean([obs["color_entropy"] for obs in observations])),
        "valid_pixel_coverage": float(np.mean([obs["valid_pixel_coverage"] for obs in observations])),
        "color_consistency": color_consistency,
        "appearance_consistency": appearance_consistency,
        "evidence_quality": quality,
        "official_score": official_score,
        "evidence_mode": evidence_mode,
        "best_crop_rgb": max(observations, key=lambda obs: obs["observation_quality_score"])["crop_rgb"],
    }


def merge_dominant_colors(observations):
    colors = []
    for obs in observations:
        for item in obs["dominant_lab_colors"]:
            colors.append((item["lab"], item["fraction"] * obs["observation_quality_score"]))
    colors.sort(key=lambda item: item[1], reverse=True)
    total = sum(w for _lab, w in colors[:6]) or 1.0
    return [{"lab": [float(x) for x in lab], "fraction": float(w / total)} for lab, w in colors[:3]]


def official_likelihood(observations):
    neutral = float(np.mean([obs["neutral_fraction"] for obs in observations]))
    white = float(np.mean([obs["white_fraction"] for obs in observations]))
    dark = float(np.mean([obs["dark_fraction"] for obs in observations]))
    edge = float(np.mean([obs["edge_density"] for obs in observations]))
    dark_white_mix = min(1.0, min(dark, white) / 0.16)
    balanced_bw = min(1.0, (dark + white) / 0.62) * dark_white_mix
    neutral_body = min(1.0, neutral / 0.78)
    stripe_like = min(1.0, edge / 0.18)
    return float(max(0.0, min(1.0, 0.52 * balanced_bw + 0.26 * stripe_like + 0.22 * neutral_body * dark_white_mix)))


def standardize_features(features):
    arr = np.asarray(features, dtype=np.float32)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std[std < 1e-5] = 1.0
    return (arr - mean) / std, mean, std


def diagonal_gmm_two(features, seed=17, iterations=80):
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=np.float32)
    n, d = x.shape
    if n < 2:
        return np.zeros(n, dtype=np.int32), np.ones((n, 1), dtype=np.float32), x.copy(), np.ones((1, d), dtype=np.float32)
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
    means = np.stack([x[i], x[j]], axis=0)
    variances = np.stack([np.var(x, axis=0) + 1e-3, np.var(x, axis=0) + 1e-3], axis=0)
    priors = np.array([0.5, 0.5], dtype=np.float32)
    resp = np.zeros((n, 2), dtype=np.float32)
    for _ in range(iterations):
        logp = []
        for k in range(2):
            var = np.maximum(variances[k], 1e-4)
            ll = -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(((x - means[k]) ** 2) / var, axis=1))
            logp.append(np.log(max(priors[k], 1e-6)) + ll)
        logp = np.stack(logp, axis=1)
        logp -= np.max(logp, axis=1, keepdims=True)
        prob = np.exp(logp)
        resp = prob / np.maximum(np.sum(prob, axis=1, keepdims=True), 1e-8)
        nk = np.sum(resp, axis=0) + 1e-6
        priors = nk / n
        for k in range(2):
            means[k] = np.sum(resp[:, k:k+1] * x, axis=0) / nk[k]
            variances[k] = np.sum(resp[:, k:k+1] * ((x - means[k]) ** 2), axis=0) / nk[k] + 1e-3
    labels = np.argmax(resp, axis=1).astype(np.int32)
    return labels, resp, means, variances


def assign_tracks(signatures, cfg, v1_assignments):
    min_obs = int(cfg["observation_selection"]["min_observations_per_track"])
    official_cfg = cfg["official"]
    prelim = {}
    cluster_ids = []
    cluster_feats = []
    for track_id, sig in sorted(signatures.items()):
        v1_cls = v1_assignments.get(track_id, {}).get("assigned_class")
        official_score = float(sig["official_score"])
        if v1_cls == "official":
            official_score = min(1.0, official_score + float(official_cfg.get("v1_official_bonus", 0.0)))
        sig["official_score_with_v1"] = official_score
        if sig["observations_used"] < min_obs:
            prelim[track_id] = make_assignment(sig, "unknown", 0.0, "too_few_usable_observations")
        elif v1_cls == "official" and official_score >= 0.45:
            prelim[track_id] = make_assignment(sig, "official", official_score, "v1_official_confirmed_by_neutral_torso_evidence")
        elif official_score >= float(official_cfg["score_threshold"]):
            prelim[track_id] = make_assignment(sig, "official", official_score, "official_neutral_dark_white_track_evidence")
        else:
            cluster_ids.append(track_id)
            cluster_feats.append(sig["combined_signature"])
    if len(cluster_ids) < 2:
        for track_id in cluster_ids:
            prelim[track_id] = make_assignment(signatures[track_id], "unknown", 0.0, "not_enough_tracks_for_two_team_gmm")
        return prelim, {"status": "skipped_not_enough_tracks"}
    x, mean, std = standardize_features(cluster_feats)
    labels, posterior, centers, variances = diagonal_gmm_two(x, int(cfg["clustering"]["gmm_random_seed"]))
    counts = [int(np.sum(labels == 0)), int(np.sum(labels == 1))]
    if counts[1] > counts[0]:
        label_to_team = {1: "team_a", 0: "team_b"}
    else:
        label_to_team = {0: "team_a", 1: "team_b"}
    for idx, track_id in enumerate(cluster_ids):
        sig = signatures[track_id]
        label = int(labels[idx])
        post = float(posterior[idx, label])
        other = 1 - label
        margin = float(abs(posterior[idx, label] - posterior[idx, other]))
        dists = [float(np.linalg.norm(x[idx] - centers[0])), float(np.linalg.norm(x[idx] - centers[1]))]
        reason = "gmm_posterior_team_assignment"
        cls = label_to_team[label]
        confidence = float(max(0.0, min(1.0, 0.58 * post + 0.25 * margin + 0.17 * sig["evidence_quality"])))
        if post < float(cfg["clustering"]["unknown_posterior_threshold"]):
            cls, reason = "unknown", "low_gmm_posterior"
        elif margin < float(cfg["clustering"]["assignment_margin_threshold"]):
            cls, reason = "unknown", "low_assignment_margin"
        elif sig["evidence_quality"] < float(cfg["clustering"]["min_evidence_quality"]):
            cls, reason = "unknown", "low_evidence_quality"
        prelim[track_id] = make_assignment(sig, cls, confidence, reason)
        prelim[track_id].update({
            "automatic_cluster": label,
            "posterior_probability": post,
            "distance_to_team_a_center": dists[[k for k, v in label_to_team.items() if v == "team_a"][0]],
            "distance_to_team_b_center": dists[[k for k, v in label_to_team.items() if v == "team_b"][0]],
            "assignment_margin": margin,
        })
    diagnostics = {
        "status": "complete",
        "method": "internal_diagonal_gaussian_mixture",
        "sklearn_gaussian_mixture_available": False,
        "cluster_counts": {"cluster_0": counts[0], "cluster_1": counts[1]},
        "label_to_team": {str(k): v for k, v in label_to_team.items()},
        "centers_standardized": centers.tolist(),
        "variances_standardized": variances.tolist(),
        "feature_standardization": {"mean": mean.tolist(), "std": std.tolist()},
    }
    return prelim, diagnostics


def make_assignment(sig, cls, confidence, reason):
    return {
        "track_id": int(sig["track_id"]),
        "final_class": cls,
        "assigned_class": cls,
        "confidence": float(confidence),
        "posterior_probability": None,
        "automatic_cluster": None,
        "distance_to_team_a_center": None,
        "distance_to_team_b_center": None,
        "assignment_margin": None,
        "color_consistency": float(sig["color_consistency"]),
        "appearance_consistency": sig["appearance_consistency"],
        "evidence_mode": sig["evidence_mode"],
        "evidence_quality": float(sig["evidence_quality"]),
        "observations_used": int(sig["observations_used"]),
        "observations_selected": int(sig["observations_selected"]),
        "observations_rejected": int(sig["observations_rejected"]),
        "official_score": float(sig["official_score"]),
        "official_score_with_v1": float(sig.get("official_score_with_v1", sig["official_score"])),
        "representative_median_hsv": [float(x) for x in sig["median_hsv"].tolist()],
        "representative_median_lab": [float(x) for x in sig["median_lab"].tolist()],
        "dominant_lab_colors": sig["dominant_lab_colors"],
        "reason": reason,
        "reason_for_unknown": reason if cls == "unknown" else None,
    }


def save_contact_sheet_for_class(signatures, assignments, cls, output_path):
    rows = [a for a in assignments if a["assigned_class"] == cls]
    if cls == "low_confidence":
        rows = [a for a in assignments if a["confidence"] < 0.65 or a["assigned_class"] == "unknown"]
    tiles = []
    for a in sorted(rows, key=lambda item: int(item["track_id"])):
        sig = signatures[int(a["track_id"])]
        crop = Image.fromarray(sig["best_crop_rgb"].astype(np.uint8)).convert("RGB")
        crop.thumbnail((120, 140), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (180, 220), (245, 245, 245))
        tile.paste(crop, ((180 - crop.width) // 2, 8))
        draw = ImageDraw.Draw(tile)
        color = TEAM_COLORS.get(a["assigned_class"], TEAM_COLORS["unknown"])
        draw.rectangle((0, 0, 179, 219), outline=color, width=4)
        lines = [
            "T{} {}".format(a["track_id"], a["assigned_class"]),
            "conf {:.2f}".format(a["confidence"]),
            "obs {}".format(a["observations_used"]),
            a["evidence_mode"].replace("_", " "),
        ]
        y = 150
        for line in lines:
            draw.text((8, y), line[:24], fill=color if y == 150 else (0, 0, 0))
            y += 16
        dom = a.get("dominant_lab_colors") or []
        x = 8
        for item in dom[:3]:
            lab = np.uint8([[item["lab"]]])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]
            draw.rectangle((x, 204, x + 22, 216), fill=tuple(int(v) for v in rgb.tolist()))
            x += 26
        tiles.append(tile)
    cols = max(1, min(5, len(tiles) or 1))
    rows_n = max(1, int(math.ceil(len(tiles) / cols)))
    sheet = Image.new("RGB", (cols * 180, rows_n * 220), (235, 235, 235))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 180, (idx // cols) * 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def save_palette_preview(assignments, output_path):
    tile_w, tile_h = 210, 70
    image = Image.new("RGB", (tile_w * 2, tile_h * max(1, math.ceil(len(assignments) / 2))), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for idx, a in enumerate(sorted(assignments, key=lambda item: int(item["track_id"]))):
        x = (idx % 2) * tile_w
        y = (idx // 2) * tile_h
        color = TEAM_COLORS.get(a["assigned_class"], TEAM_COLORS["unknown"])
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=color, width=3)
        draw.text((x + 8, y + 8), "T{} {} {:.2f}".format(a["track_id"], a["assigned_class"], a["confidence"]), fill=color)
        px = x + 8
        for item in (a.get("dominant_lab_colors") or [])[:3]:
            lab = np.uint8([[item["lab"]]])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0, 0]
            draw.rectangle((px, y + 34, px + 44, y + 58), fill=tuple(int(v) for v in rgb.tolist()), outline=(0, 0, 0))
            px += 50
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def save_embedding_plot(signatures, assignments, output_path):
    points = []
    for a in assignments:
        sig = signatures[int(a["track_id"])]
        vec = sig["combined_signature"]
        points.append((a, vec))
    if not points:
        return None
    x = np.stack([vec for _a, vec in points])
    x, _mean, _std = standardize_features(x)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2] if u.shape[1] >= 2 else np.pad(u[:, :1], ((0, 0), (0, 1)))
    minv = coords.min(axis=0)
    maxv = coords.max(axis=0)
    span = np.maximum(maxv - minv, 1e-6)
    image = Image.new("RGB", (720, 520), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), "V2 track signatures projected with PCA/SVD", fill=(0, 0, 0))
    for (a, _vec), xy in zip(points, coords):
        px = int(50 + (xy[0] - minv[0]) / span[0] * 620)
        py = int(470 - (xy[1] - minv[1]) / span[1] * 400)
        color = TEAM_COLORS.get(a["assigned_class"], TEAM_COLORS["unknown"])
        draw.ellipse((px - 9, py - 9, px + 9, py + 9), fill=color, outline=(255, 255, 255), width=2)
        draw.text((px + 11, py - 7), "T{}".format(a["track_id"]), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def class_counts(assignments):
    counts = {"team_a": 0, "team_b": 0, "official": 0, "unknown": 0}
    for a in assignments:
        counts[a["assigned_class"]] = counts.get(a["assigned_class"], 0) + 1
    return counts


def assignment_dict(assignments):
    return {int(a["track_id"]): a for a in assignments}


def grouped_projected_counts(points, assignments_by_id):
    counts = {"team_a": 0, "team_b": 0, "official": 0, "unknown": 0}
    for p in points:
        if not p.get("inside_field_template_bounds"):
            continue
        cls = assignments_by_id.get(int(p["track_id"]), {}).get("assigned_class", "unknown")
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def write_v1_v2_comparison(v1_rows, v2_rows, projected_points, output_dir):
    v1_by_id = {int(r["track_id"]): r for r in v1_rows}
    v2_by_id = {int(r["track_id"]): r for r in v2_rows}
    all_ids = sorted(set(v1_by_id) | set(v2_by_id))
    rows = []
    changed = []
    for track_id in all_ids:
        v1r = v1_by_id.get(track_id, {})
        v2r = v2_by_id.get(track_id, {})
        v1_cls = v1r.get("assigned_class")
        v2_cls = v2r.get("assigned_class")
        did_change = v1_cls != v2_cls
        if did_change:
            changed.append(track_id)
        reason = "unchanged"
        if did_change:
            reason = v2r.get("reason") or "v2_evidence_changed_assignment"
        rows.append({
            "track_id": track_id,
            "v1_assignment": v1_cls,
            "v1_confidence": v1r.get("confidence"),
            "v2_assignment": v2_cls,
            "v2_confidence": v2r.get("confidence"),
            "assignment_changed": did_change,
            "likely_reason_for_change": reason,
        })
    v1_counts = class_counts(list(v1_by_id.values()))
    v2_counts = class_counts(list(v2_by_id.values()))
    v1_proj = grouped_projected_counts(projected_points, v1_by_id)
    v2_proj = grouped_projected_counts(projected_points, v2_by_id)
    imbalance_v1 = abs(v1_counts.get("team_a", 0) - v1_counts.get("team_b", 0))
    imbalance_v2 = abs(v2_counts.get("team_a", 0) - v2_counts.get("team_b", 0))
    explanation = "V2 suggests the original imbalance is partly track fragmentation plus conservative unknown handling."
    if imbalance_v2 < imbalance_v1:
        explanation = "V2 reduced the team-count imbalance, suggesting V1 had some classification leakage."
    elif imbalance_v2 >= imbalance_v1:
        explanation = "V2 did not materially reduce the imbalance; this likely reflects fragmented visible tracks and uneven team visibility more than pure color clustering error."
    payload = {
        "tracks": rows,
        "summary": {
            "v1_class_counts": v1_counts,
            "v2_class_counts": v2_counts,
            "v1_projected_observation_counts_by_team": v1_proj,
            "v2_projected_observation_counts_by_team": v2_proj,
            "tracks_changed": changed,
            "newly_marked_unknown": [r["track_id"] for r in rows if r["v1_assignment"] != "unknown" and r["v2_assignment"] == "unknown"],
            "newly_identified_official": [r["track_id"] for r in rows if r["v1_assignment"] != "official" and r["v2_assignment"] == "official"],
            "imbalance_reduced": imbalance_v2 < imbalance_v1,
            "imbalance_explanation": explanation,
        },
    }
    write_json(output_dir / "v1_v2_comparison.json", payload)
    csv_path = output_dir / "v1_v2_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return payload


def write_color_only_vs_hybrid(color_rows, hybrid_rows, output_dir):
    color_by_id = {int(row["track_id"]): row for row in color_rows}
    hybrid_by_id = {int(row["track_id"]): row for row in hybrid_rows}
    all_ids = sorted(set(color_by_id) | set(hybrid_by_id))
    rows = []
    changed = []
    audited_tracks = sorted(set([3, 5, 11, 21, 12, 13, 15, 20, 10]))
    for track_id in all_ids:
        c = color_by_id.get(track_id, {})
        h = hybrid_by_id.get(track_id, {})
        c_cls = c.get("assigned_class")
        h_cls = h.get("assigned_class")
        did_change = c_cls != h_cls
        if did_change:
            changed.append(track_id)
        reason = "unchanged"
        if did_change:
            reason = h.get("reason") or "hybrid_feature_assignment_changed"
        rows.append({
            "track_id": track_id,
            "color_only_assignment": c_cls,
            "color_only_confidence": c.get("confidence"),
            "hybrid_assignment": h_cls,
            "hybrid_confidence": h.get("confidence"),
            "assignment_changed": did_change,
            "appearance_consistency": h.get("appearance_consistency"),
            "color_consistency": h.get("color_consistency"),
            "observations_used": h.get("observations_used"),
            "distance_to_hybrid_team_a_center": h.get("distance_to_team_a_center"),
            "distance_to_hybrid_team_b_center": h.get("distance_to_team_b_center"),
            "reason_for_change": reason,
            "explicit_audit_track": track_id in audited_tracks,
        })
    color_counts = class_counts(list(color_by_id.values()))
    hybrid_counts = class_counts(list(hybrid_by_id.values()))
    payload = {
        "tracks": rows,
        "summary": {
            "color_only_class_counts": color_counts,
            "hybrid_class_counts": hybrid_counts,
            "tracks_changed_from_color_only_v2": changed,
            "changed_v1_to_v2_tracks_audited": [3, 5, 11, 21],
            "low_observation_tracks_audited": [12, 13, 15, 20, 21],
            "official_track_audited": 10,
            "posterior_note": "GMM posterior is a clustering diagnostic, not calibrated real-world accuracy.",
        },
    }
    write_json(output_dir / "color_only_vs_hybrid.json", payload)
    write_json(output_dir / "color_only_vs_hybrid_summary.json", payload["summary"])
    csv_path = output_dir / "color_only_vs_hybrid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return payload


def save_contact_sheet_for_track_ids(signatures, assignments_by_id, track_ids, output_path, title):
    tiles = []
    for track_id in track_ids:
        if int(track_id) not in signatures:
            continue
        sig = signatures[int(track_id)]
        assignment = assignments_by_id.get(int(track_id), {"assigned_class": "unknown", "confidence": 0.0, "observations_used": 0})
        crop = Image.fromarray(sig["best_crop_rgb"].astype(np.uint8)).convert("RGB")
        crop.thumbnail((120, 140), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (190, 225), (245, 245, 245))
        tile.paste(crop, ((190 - crop.width) // 2, 8))
        draw = ImageDraw.Draw(tile)
        color = TEAM_COLORS.get(assignment.get("assigned_class", "unknown"), TEAM_COLORS["unknown"])
        draw.rectangle((0, 0, 189, 224), outline=color, width=4)
        lines = [
            "T{} {}".format(track_id, assignment.get("assigned_class", "unknown")),
            "conf {:.2f}".format(float(assignment.get("confidence") or 0.0)),
            "obs {}".format(assignment.get("observations_used", 0)),
            assignment.get("evidence_mode", "")[:20],
        ]
        y = 150
        for line in lines:
            draw.text((8, y), line, fill=color if y == 150 else (0, 0, 0))
            y += 16
        tiles.append(tile)
    cols = max(1, min(5, len(tiles) or 1))
    rows_n = max(1, int(math.ceil(len(tiles) / cols)))
    sheet = Image.new("RGB", (cols * 190, rows_n * 225 + 28), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill=(0, 0, 0))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 190, 28 + (idx // cols) * 225))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def copy_alias(src, dst):
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def detections_by_frame(detections):
    out = {}
    for det in detections:
        out.setdefault(int(det["frame_index"]), []).append(det)
    return out


def render_overlay_v2(frames_by_source, frame_lookup, det_by_frame, assignments_by_id, output_dir, fps):
    frames_dir = output_dir / "overlay_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for frame_index in sorted(det_by_frame):
        source = frame_lookup.get(frame_index, {}).get("source_frame_index")
        if source not in frames_by_source:
            continue
        image = v1.draw_tracking_overlay(frames_by_source[source], det_by_frame[frame_index], assignments_by_id)
        path = frames_dir / "frame_{:03d}_team_assignment_v2.png".format(frame_index)
        image.save(path)
        rendered.append(path)
    mp4, mp4_error = v1.write_mp4(rendered, output_dir / "team_assignment_overlay_v2.mp4", fps)
    gif = v1.write_gif(rendered, output_dir / "team_assignment_overlay_v2.gif", fps)
    return {"frames_dir": str(frames_dir), "frames": [str(p) for p in rendered], "mp4": mp4, "mp4_error": mp4_error, "gif": gif}


def point_xy(point):
    xy = point["projected_field_point"]
    return float(xy["x"]), float(xy["y"])


def group_points(points):
    by_frame, by_track = {}, {}
    for p in points:
        if not p.get("inside_field_template_bounds"):
            continue
        frame = int(p["frame_index"])
        track = int(p["track_id"])
        by_frame.setdefault(frame, []).append(p)
        by_track.setdefault(track, []).append(p)
    for rows in by_frame.values():
        rows.sort(key=lambda p: int(p["track_id"]))
    for rows in by_track.values():
        rows.sort(key=lambda p: int(p["frame_index"]))
    return by_frame, by_track


def draw_field_v2(template, by_frame, by_track, assignments_by_id, frame_index, trail_length, opacity):
    image = template.convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image)
    current = by_frame.get(frame_index, [])
    metrics = {
        "frame_index": frame_index,
        "team_a_visible_count": 0,
        "team_b_visible_count": 0,
        "official_count": 0,
        "unknown_count": 0,
        "team_a_polygon_area": 0.0,
        "team_b_polygon_area": 0.0,
        "team_a_centroid": None,
        "team_b_centroid": None,
        "centroid_distance": None,
        "team_a_width": 0.0,
        "team_a_depth": 0.0,
        "team_b_width": 0.0,
        "team_b_depth": 0.0,
        "projected_points_excluded": 0,
    }
    for track_id, rows in sorted(by_track.items()):
        cls = assignments_by_id.get(track_id, {}).get("assigned_class", "unknown")
        if cls not in ("team_a", "team_b"):
            continue
        hist = [p for p in rows if frame_index - trail_length <= int(p["frame_index"]) <= frame_index]
        if len(hist) >= 2:
            coords = [point_xy(p) for p in hist]
            draw.line(coords, fill=TEAM_COLORS[cls], width=3)
    team_points = {"team_a": [], "team_b": []}
    other_points = []
    for p in current:
        track_id = int(p["track_id"])
        cls = assignments_by_id.get(track_id, {}).get("assigned_class", "unknown")
        x, y = point_xy(p)
        if cls in team_points:
            team_points[cls].append((x, y, track_id))
        elif cls == "official":
            metrics["official_count"] += 1
            other_points.append((x, y, track_id, cls))
        else:
            metrics["unknown_count"] += 1
            other_points.append((x, y, track_id, "unknown"))
    alpha = int(max(0, min(1, opacity)) * 255)
    for cls in ("team_a", "team_b"):
        coords = [(x, y) for x, y, _ in team_points[cls]]
        metrics["{}_visible_count".format(cls)] = len(coords)
        if coords:
            xs = [x for x, _ in coords]
            ys = [y for _, y in coords]
            metrics["{}_width".format(cls)] = float(max(xs) - min(xs))
            metrics["{}_depth".format(cls)] = float(max(ys) - min(ys))
            metrics["{}_centroid".format(cls)] = {"x": float(statistics.mean(xs)), "y": float(statistics.mean(ys))}
        if len(coords) >= 3:
            hull = cv2.convexHull(np.asarray(coords, dtype=np.float32)).reshape(-1, 2)
            poly = [(float(x), float(y)) for x, y in hull]
            odraw.polygon(poly, fill=TEAM_COLORS[cls] + (alpha,), outline=TEAM_COLORS[cls] + (220,))
            metrics["{}_polygon_area".format(cls)] = float(cv2.contourArea(hull.reshape(-1, 1, 2)))
        elif len(coords) == 2:
            draw.line(coords, fill=TEAM_COLORS[cls], width=5)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    for cls, rows in team_points.items():
        color = TEAM_COLORS[cls]
        for x, y, track_id in rows:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255), width=2)
            draw.text((x + 9, y - 8), "T{}".format(track_id), fill=color)
        c = metrics["{}_centroid".format(cls)]
        if c:
            draw.rectangle((c["x"] - 6, c["y"] - 6, c["x"] + 6, c["y"] + 6), fill=(255, 255, 255), outline=color, width=3)
    for x, y, track_id, cls in other_points:
        color = TEAM_COLORS[cls]
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255))
        draw.text((x + 7, y - 7), "T{}".format(track_id), fill=color)
    ca, cb = metrics["team_a_centroid"], metrics["team_b_centroid"]
    if ca and cb:
        metrics["centroid_distance"] = float(math.hypot(ca["x"] - cb["x"], ca["y"] - cb["y"]))
        draw.line((ca["x"], ca["y"], cb["x"], cb["y"]), fill=(40, 40, 40), width=2)
    draw.rectangle((8, 8, 390, 40), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((14, 16), "V2 frame {:03d} | A {} B {} O {} U {}".format(frame_index, metrics["team_a_visible_count"], metrics["team_b_visible_count"], metrics["official_count"], metrics["unknown_count"]), fill=(0, 0, 0))
    return image, metrics


def render_polygon_v2(frames_by_source, frame_lookup, det_by_frame, projected_points, assignments_by_id, template, output_dir, cfg):
    render_cfg = cfg["rendering"]
    frames_root = output_dir / "frames"
    field_dir = frames_root / "field"
    side_dir = frames_root / "side_by_side"
    field_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)
    in_bounds = [p for p in projected_points if p.get("inside_field_template_bounds")]
    by_frame, by_track = group_points(in_bounds)
    frame_indices = sorted(set(det_by_frame) | set(by_frame))
    field_frames, side_frames, metrics = [], [], []
    for frame_index in frame_indices:
        field, row = draw_field_v2(template, by_frame, by_track, assignments_by_id, frame_index, int(render_cfg["trail_length"]), float(render_cfg["polygon_opacity"]))
        field_resized = v1.resize_with_aspect(field, target_width=int(render_cfg["field_panel_width"]))
        field_path = field_dir / "frame_{:03d}_team_polygon_field_v2.png".format(frame_index)
        field_resized.save(field_path)
        field_frames.append(field_path)
        source = frame_lookup.get(frame_index, {}).get("source_frame_index")
        if source in frames_by_source:
            left = v1.draw_tracking_overlay(frames_by_source[source], det_by_frame.get(frame_index, []), assignments_by_id)
            side_path = side_dir / "frame_{:03d}_team_polygon_side_by_side_v2.png".format(frame_index)
            v1.compose_side_by_side(left, field_resized, side_path)
            side_frames.append(side_path)
        metrics.append(row)
    fps = float(render_cfg["fps"])
    field_mp4, field_mp4_error = v1.write_mp4(field_frames, output_dir / "team_polygon_field_v2.mp4", fps)
    field_gif = v1.write_gif(field_frames, output_dir / "team_polygon_field_v2.gif", fps)
    side_mp4, side_mp4_error = v1.write_mp4(side_frames, output_dir / "team_polygon_side_by_side_v2.mp4", fps)
    side_gif = v1.write_gif(side_frames, output_dir / "team_polygon_side_by_side_v2.gif", fps)
    return metrics, {
        "field_frames_dir": str(field_dir),
        "side_by_side_frames_dir": str(side_dir),
        "team_polygon_field_v2_mp4": field_mp4,
        "team_polygon_field_v2_mp4_error": field_mp4_error,
        "team_polygon_field_v2_gif": field_gif,
        "team_polygon_side_by_side_v2_mp4": side_mp4,
        "team_polygon_side_by_side_v2_mp4_error": side_mp4_error,
        "team_polygon_side_by_side_v2_gif": side_gif,
    }


def save_heatmap(points, assignments_by_id, template, output_dir, cfg):
    render_cfg = cfg["rendering"]
    sigma = float(render_cfg["heatmap_sigma"])
    alpha = float(render_cfg["heatmap_alpha"])
    artifacts = {}
    for cls, name in (("team_a", "team_a_heatmap_v2"), ("team_b", "team_b_heatmap_v2")):
        rows = [p for p in points if p.get("inside_field_template_bounds") and assignments_by_id.get(int(p["track_id"]), {}).get("assigned_class") == cls]
        heat, overlay, _nonzero = v1.density_heatmap(rows, template, output_dir / "{}.png".format(name), output_dir / "{}_overlay.png".format(name), sigma, alpha)
        artifacts[name] = heat
        artifacts["{}_overlay".format(name)] = overlay
    combined = template.convert("RGB")
    draw = ImageDraw.Draw(combined, "RGBA")
    for p in points:
        if not p.get("inside_field_template_bounds"):
            continue
        cls = assignments_by_id.get(int(p["track_id"]), {}).get("assigned_class")
        if cls not in ("team_a", "team_b"):
            continue
        x, y = point_xy(p)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=TEAM_COLORS[cls] + (75,))
    combined_path = output_dir / "team_specific_heatmap_v2.png"
    combined.save(combined_path)
    artifacts["team_specific_heatmap_v2"] = str(combined_path)
    return artifacts


def main():
    args = parse_args()
    if args.preflight_weights:
        print(json.dumps(resnet18_preflight_metadata(), indent=2, sort_keys=True))
        return 0
    cfg = read_config(args)
    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    tracking_path = project_path(inputs["tracking_metadata"])
    projected_path = project_path(inputs["projected_points"])
    clean_path = project_path(inputs["clean_crop_metadata"])
    video_path = Path(inputs["video"])
    field_template_path = project_path(inputs["field_template"])
    team_output = project_path(outputs["team_assignment_dir"])
    polygon_output = project_path(outputs["team_polygon_dir"])
    team_output.mkdir(parents=True, exist_ok=True)
    polygon_output.mkdir(parents=True, exist_ok=True)

    tracking = load_json(tracking_path)
    projected = load_json(projected_path).get("points", [])
    detections = tracking.get("detections", [])
    tracks = {}
    for det in detections:
        tracks.setdefault(int(det["track_id"]), []).append(det)
    frame_lookup = v1.decoded_frame_map(tracking)
    det_by_frame = detections_by_frame(detections)
    needed_sources = [frame_lookup[int(det["frame_index"])]["source_frame_index"] for det in detections if int(det["frame_index"]) in frame_lookup]
    frames_by_source = v1.read_needed_frames(video_path, needed_sources)
    clean_index = build_clean_crop_index(clean_path)

    appearance_cfg = cfg["appearance"]
    backend = load_appearance_backend(appearance_cfg.get("embedding_backend", "auto"), bool(appearance_cfg.get("allow_download_weights", False)))
    backend_meta = backend.metadata or {}
    evidence_mode = backend_meta.get("evidence_mode") or ("appearance_plus_color" if backend.active_backend != "none" else "color_only")
    if backend.active_backend == "none":
        cfg["feature_weights"]["appearance"] = 0.0
        cfg["feature_weights"]["color"] = 1.0
    write_json(team_output / "feature_backend_metadata.json", backend_meta)

    observations_by_track = {}
    signatures = {}
    for track_id, rows in sorted(tracks.items()):
        raw_obs = build_track_observations(track_id, rows, frame_lookup, frames_by_source, clean_index, backend, cfg)
        kept, rejected = reject_outliers(raw_obs, cfg, backend.active_backend != "none")
        observations_by_track[track_id] = {"raw": raw_obs, "kept": kept, "rejected": rejected}
        sig = aggregate_signature(track_id, kept, rejected, cfg, evidence_mode)
        if sig is not None:
            signatures[track_id] = sig

    v1_path = project_path(inputs.get("v1_assignments", "outputs/nll_test4/team_assignment_demo/track_team_assignments.json"))
    v1_rows = load_json(v1_path) if v1_path.exists() else []
    v1_by_id = {int(row["track_id"]): row for row in v1_rows}
    assignments_by_id, cluster_diag = assign_tracks(signatures, cfg, v1_by_id)
    for track_id in sorted(tracks):
        if track_id not in assignments_by_id:
            assignments_by_id[track_id] = {
                "track_id": int(track_id),
                "assigned_class": "unknown",
                "final_class": "unknown",
                "confidence": 0.0,
                "reason": "no_usable_v2_signature",
                "reason_for_unknown": "no_usable_v2_signature",
                "observations_used": 0,
                "observations_selected": 0,
                "observations_rejected": 0,
                "evidence_mode": evidence_mode,
            }
    assignments = [assignments_by_id[k] for k in sorted(assignments_by_id)]
    counts = class_counts(assignments)
    confidences = [float(a.get("confidence") or 0.0) for a in assignments]
    confidence_dist = {
        "min": min(confidences) if confidences else None,
        "max": max(confidences) if confidences else None,
        "mean": float(statistics.mean(confidences)) if confidences else None,
        "median": float(statistics.median(confidences)) if confidences else None,
    }

    signatures_json = []
    for track_id, sig in sorted(signatures.items()):
        signatures_json.append({
            "track_id": int(track_id),
            "observations_selected": sig["observations_selected"],
            "observations_used": sig["observations_used"],
            "observations_rejected": sig["observations_rejected"],
            "color_consistency": sig["color_consistency"],
            "appearance_consistency": sig["appearance_consistency"],
            "evidence_quality": sig["evidence_quality"],
            "evidence_mode": sig["evidence_mode"],
            "median_hsv": [float(x) for x in sig["median_hsv"].tolist()],
            "median_lab": [float(x) for x in sig["median_lab"].tolist()],
            "dominant_lab_colors": sig["dominant_lab_colors"],
            "white_fraction": sig["white_fraction"],
            "dark_fraction": sig["dark_fraction"],
            "neutral_fraction": sig["neutral_fraction"],
            "color_entropy": sig["color_entropy"],
            "valid_pixel_coverage": sig["valid_pixel_coverage"],
            "official_score": sig["official_score"],
        })
    write_json(team_output / "track_feature_signatures.json", signatures_json)
    write_json(team_output / "cluster_centers.json", cluster_diag)
    write_json(team_output / "track_team_assignments_v2.json", assignments)

    contact_artifacts = {}
    for cls in ("team_a", "team_b", "official", "unknown", "low_confidence"):
        contact_artifacts["{}_contact_sheet".format(cls)] = save_contact_sheet_for_class(signatures, assignments, cls, team_output / "{}_contact_sheet.png".format(cls))
    palette = save_palette_preview(assignments, team_output / "color_palette_preview.png")
    emb_plot = save_embedding_plot(signatures, assignments, team_output / "embedding_cluster_plot.png")
    overlay = render_overlay_v2(frames_by_source, frame_lookup, det_by_frame, assignments_by_id, team_output, float(cfg["rendering"]["fps"]))
    comparison = write_v1_v2_comparison(v1_rows, assignments, projected, team_output)
    hybrid_mode = "hybrid" in str(team_output)
    color_hybrid_comparison = None
    color_only_rows = []
    if hybrid_mode:
        color_path = project_path(args.color_only_assignments)
        if color_path.exists():
            color_only_rows = load_json(color_path)
            color_hybrid_comparison = write_color_only_vs_hybrid(color_only_rows, assignments, team_output)
        color_by_id = {int(row["track_id"]): row for row in color_only_rows}
        changed_track_ids = []
        if color_hybrid_comparison:
            changed_track_ids = color_hybrid_comparison["summary"]["tracks_changed_from_color_only_v2"]
        save_contact_sheet_for_track_ids(
            signatures,
            assignments_by_id,
            changed_track_ids,
            team_output / "color_only_vs_hybrid_changed_tracks.png",
            "Color-only V2 vs hybrid changed tracks",
        )
        save_contact_sheet_for_track_ids(
            signatures,
            assignments_by_id,
            [12, 13, 15, 20, 21],
            team_output / "low_observation_tracks.png",
            "Low-observation audit tracks",
        )
        for cls in ("team_a", "team_b", "official", "unknown"):
            src = team_output / "{}_contact_sheet.png".format(cls)
            dst = team_output / "hybrid_{}_contact_sheet.png".format(cls)
            copy_alias(str(src), dst)
        copy_alias(overlay["mp4"], team_output / "team_assignment_overlay_v2_hybrid.mp4")
        copy_alias(overlay["gif"], team_output / "team_assignment_overlay_v2_hybrid.gif")

    scorebug_cfg = cfg.get("scorebug_priors", {})
    scorebug_used = bool(scorebug_cfg.get("enabled", False) and project_path(scorebug_cfg.get("path", "")).exists())
    summary = {
        "status": "complete",
        "stage": "team_assignment_demo_v2",
        "run_id": cfg.get("run_id", "nll_test4"),
        "output_dir": str(team_output),
        "counts": {
            "tracks_total": len(assignments),
            "team_a": counts.get("team_a", 0),
            "team_b": counts.get("team_b", 0),
            "official": counts.get("official", 0),
            "unknown": counts.get("unknown", 0),
            "tracks_with_v2_signatures": len(signatures),
            "detections_used": len(detections),
        },
        "confidence_distribution": confidence_dist,
        "embedding_backend_used": backend.active_backend,
        "weights_cached_or_downloaded": {
            "cached_before_load": backend_meta.get("weights_cached_before_load"),
            "downloaded": backend_meta.get("weights_downloaded"),
        },
        "color_only_fallback_used": backend.active_backend == "none",
        "scorebug_priors_used": scorebug_used,
        "scorebug_note": "disabled or unavailable; cluster names remain team_a/team_b" if not scorebug_used else "scorebug priors were available but do not override torso evidence",
        "v1_v2_comparison_summary": comparison["summary"],
        "color_only_vs_hybrid_summary": color_hybrid_comparison["summary"] if color_hybrid_comparison else None,
        "artifacts": {
            "track_team_assignments_v2": str(team_output / "track_team_assignments_v2.json"),
            "team_assignment_summary_v2": str(team_output / "team_assignment_summary_v2.json"),
            "track_feature_signatures": str(team_output / "track_feature_signatures.json"),
            "cluster_centers": str(team_output / "cluster_centers.json"),
            "feature_backend_metadata": str(team_output / "feature_backend_metadata.json"),
            "color_palette_preview": palette,
            "embedding_cluster_plot": emb_plot,
            "team_assignment_overlay_v2_mp4": overlay["mp4"],
            "team_assignment_overlay_v2_gif": overlay["gif"],
            "team_assignment_overlay_v2_hybrid_mp4": str(team_output / "team_assignment_overlay_v2_hybrid.mp4") if hybrid_mode else None,
            "team_assignment_overlay_v2_hybrid_gif": str(team_output / "team_assignment_overlay_v2_hybrid.gif") if hybrid_mode else None,
            "v1_v2_comparison_json": str(team_output / "v1_v2_comparison.json"),
            "v1_v2_comparison_csv": str(team_output / "v1_v2_comparison.csv"),
            "color_only_vs_hybrid_json": str(team_output / "color_only_vs_hybrid.json") if hybrid_mode else None,
            "color_only_vs_hybrid_csv": str(team_output / "color_only_vs_hybrid.csv") if hybrid_mode else None,
            "color_only_vs_hybrid_summary": str(team_output / "color_only_vs_hybrid_summary.json") if hybrid_mode else None,
            "color_only_vs_hybrid_changed_tracks": str(team_output / "color_only_vs_hybrid_changed_tracks.png") if hybrid_mode else None,
            "low_observation_tracks": str(team_output / "low_observation_tracks.png") if hybrid_mode else None,
            "hybrid_team_a_contact_sheet": str(team_output / "hybrid_team_a_contact_sheet.png") if hybrid_mode else None,
            "hybrid_team_b_contact_sheet": str(team_output / "hybrid_team_b_contact_sheet.png") if hybrid_mode else None,
            "hybrid_official_contact_sheet": str(team_output / "hybrid_official_contact_sheet.png") if hybrid_mode else None,
            "hybrid_unknown_contact_sheet": str(team_output / "hybrid_unknown_contact_sheet.png") if hybrid_mode else None,
            **contact_artifacts,
        },
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(team_output / "team_assignment_summary_v2.json", summary)

    template = Image.open(field_template_path).convert("RGB")
    metrics, polygon_artifacts = render_polygon_v2(frames_by_source, frame_lookup, det_by_frame, projected, assignments_by_id, template, polygon_output, cfg)
    heatmap_artifacts = save_heatmap(projected, assignments_by_id, template, polygon_output, cfg)
    polygon_artifacts.update(heatmap_artifacts)
    if hybrid_mode:
        polygon_artifacts["team_polygon_field_v2_hybrid_mp4"] = copy_alias(
            polygon_artifacts.get("team_polygon_field_v2_mp4"),
            polygon_output / "team_polygon_field_v2_hybrid.mp4",
        )
        polygon_artifacts["team_polygon_field_v2_hybrid_gif"] = copy_alias(
            polygon_artifacts.get("team_polygon_field_v2_gif"),
            polygon_output / "team_polygon_field_v2_hybrid.gif",
        )
        polygon_artifacts["team_polygon_side_by_side_v2_hybrid_mp4"] = copy_alias(
            polygon_artifacts.get("team_polygon_side_by_side_v2_mp4"),
            polygon_output / "team_polygon_side_by_side_v2_hybrid.mp4",
        )
        polygon_artifacts["team_polygon_side_by_side_v2_hybrid_gif"] = copy_alias(
            polygon_artifacts.get("team_polygon_side_by_side_v2_gif"),
            polygon_output / "team_polygon_side_by_side_v2_hybrid.gif",
        )
        polygon_artifacts["team_specific_heatmap_v2_hybrid"] = copy_alias(
            heatmap_artifacts.get("team_specific_heatmap_v2"),
            polygon_output / "team_specific_heatmap_v2_hybrid.png",
        )
    excluded_total = len([p for p in projected if not p.get("inside_field_template_bounds")])
    for row in metrics:
        row["projected_points_excluded"] = len([p for p in projected if int(p["frame_index"]) == row["frame_index"] and (not p.get("inside_field_template_bounds") or assignments_by_id.get(int(p["track_id"]), {}).get("assigned_class") not in ("team_a", "team_b"))])
    polygon_summary = {
        "status": "complete",
        "stage": "team_polygon_demo_v2",
        "run_id": cfg.get("run_id", "nll_test4"),
        "output_dir": str(polygon_output),
        "counts": {
            "frames_rendered": len(metrics),
            "projected_points_total": len(projected),
            "projected_points_in_bounds": len([p for p in projected if p.get("inside_field_template_bounds")]),
            "projected_points_outside_bounds": excluded_total,
            "team_a_tracks": counts.get("team_a", 0),
            "team_b_tracks": counts.get("team_b", 0),
            "official_tracks": counts.get("official", 0),
            "unknown_tracks": counts.get("unknown", 0),
            "team_a_projected_points": grouped_projected_counts(projected, assignments_by_id).get("team_a", 0),
            "team_b_projected_points": grouped_projected_counts(projected, assignments_by_id).get("team_b", 0),
        },
        "artifacts": polygon_artifacts,
        "known_limitations": KNOWN_LIMITATIONS,
    }
    write_json(polygon_output / "team_polygon_metrics_v2.json", {"status": "complete", "frames": metrics, "known_limitations": KNOWN_LIMITATIONS})
    write_json(polygon_output / "team_polygon_summary_v2.json", polygon_summary)
    final = {"team_assignment_v2": summary, "team_polygon_v2": polygon_summary}
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
