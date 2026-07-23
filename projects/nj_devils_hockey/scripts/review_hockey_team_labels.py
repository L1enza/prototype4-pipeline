#!/usr/bin/env python3
"""Review and finalize stable track-level hockey team labels.

This stage is deliberately downstream of V2.  It never reruns tracking or
registration: the smoothed/predicted boxes and the broadcast-to-rink
homographies are read unchanged.  Review assets are written beneath the V2
directory; final media is written only when an explicit override JSON exists.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_hockey_polygon_demo_v2 as v2
import run_hockey_rink_registration_and_polygons as phase_b


DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/njdevils.mp4"
DEFAULT_V1 = "outputs/njdevils/hockey_polygon_demo"
DEFAULT_V2 = "outputs/njdevils/hockey_polygon_demo_v2"
DEFAULT_REVIEW = f"{DEFAULT_V2}/team_track_review"
DEFAULT_FINAL = "outputs/njdevils/hockey_polygon_demo_final"
DEFAULT_RINK = "assets/hockey/icerink.jpg"
VALID_LABELS = ("devils", "flyers", "official", "unknown")
REPRESENTATIVE_FRAMES = (0, 70, 138, 210, 276)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "review", "final"), default="auto")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--v1-dir", default=DEFAULT_V1)
    parser.add_argument("--v2-dir", default=DEFAULT_V2)
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW)
    parser.add_argument("--final-dir", default=DEFAULT_FINAL)
    parser.add_argument("--rink", default=DEFAULT_RINK)
    return parser.parse_args()


def decode_all_frames(video_path: Path) -> tuple[list[np.ndarray], dict]:
    frames, metadata = phase_b.decode_video(video_path)
    if len(frames) != 277:
        raise ValueError(f"Expected 277 source frames, decoded {len(frames)}")
    return frames, metadata


def upper_torso_features(frame: np.ndarray, box: list[int]) -> dict:
    """Measure jersey evidence only in the upper half of the tracked person.

    The crop deliberately ends at 49% of box height.  Black is reported for
    review, but is not positive team evidence because both teams wear black
    pants and a small amount can still enter a torso crop.
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in box]
    bw = x1 - x0 + 1.0
    bh = y1 - y0 + 1.0
    crop_box = v2.phase_a.clamp_box(
        [x0 + 0.14 * bw, y0 + 0.08 * bh, x1 - 0.14 * bw, y0 + 0.49 * bh],
        width,
        height,
    )
    tx0, ty0, tx1, ty1 = crop_box
    crop = frame[ty0 : ty1 + 1, tx0 : tx1 + 1]
    if crop.size == 0:
        return {
            "white": 0.0, "red": 0.0, "orange": 0.0, "black": 0.0,
            "stripe": 0.0, "red_hue_consistency": 0.0,
            "red_saturation": 0.0, "white_brightness": 0.0,
            "quality": 0.0, "blur_variance": 0.0, "crop_box": crop_box,
        }
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    white_mask = (saturation < 66) & (value > 145)
    red_mask = ((hue < 9) | (hue > 171)) & (saturation > 90) & (value > 52)
    orange_mask = (hue >= 9) & (hue <= 28) & (saturation > 88) & (value > 65)
    black_mask = value < 80

    red_hues = hue[red_mask]
    if red_hues.size:
        angles = red_hues * (2.0 * math.pi / 180.0)
        red_hue_consistency = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
        red_saturation = float(np.mean(saturation[red_mask]) / 255.0)
    else:
        red_hue_consistency = 0.0
        red_saturation = 0.0
    white_brightness = float(np.mean(value[white_mask]) / 255.0) if np.any(white_mask) else 0.0

    column_white = np.mean(white_mask, axis=0)
    column_black = np.mean(black_mask, axis=0)
    stripe_signal = column_white - column_black
    transitions = float(np.mean(np.abs(np.diff(stripe_signal)) > 0.20)) if stripe_signal.size > 1 else 0.0
    balanced = 2.0 * min(float(np.mean(white_mask)), float(np.mean(black_mask)))
    stripe = float(min(1.0, 2.8 * transitions + 0.62 * balanced))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    pixel_quality = min(1.0, crop.shape[0] * crop.shape[1] / 1500.0)
    blur_quality = min(1.0, blur_variance / 105.0)
    quality = float(pixel_quality * (0.22 + 0.78 * blur_quality))
    return {
        "white": float(np.mean(white_mask)),
        "red": float(np.mean(red_mask)),
        "orange": float(np.mean(orange_mask)),
        "black": float(np.mean(black_mask)),
        "stripe": stripe,
        "red_hue_consistency": red_hue_consistency,
        "red_saturation": red_saturation,
        "white_brightness": white_brightness,
        "quality": quality,
        "blur_variance": blur_variance,
        "crop_box": crop_box,
    }


def weighted_average(rows: list[dict], key: str) -> float:
    weights = np.asarray([max(0.05, float(row["features"]["quality"])) for row in rows], dtype=np.float64)
    values = np.asarray([float(row["features"][key]) for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights)) if len(rows) else 0.0


def classify_evidence_crop(features: dict) -> str:
    red = float(features["red"])
    white = float(features["white"])
    orange = float(features["orange"])
    black = float(features["black"])
    stripe = float(features["stripe"])
    if stripe >= 0.35 and white >= 0.18 and black >= 0.07 and red + orange < 0.12:
        return "official"
    if red >= 0.10 and features["red_hue_consistency"] >= 0.72 and features["red_saturation"] >= 0.48:
        return "devils"
    if white >= 0.40 and features["white_brightness"] >= 0.68 and red < 0.20:
        return "flyers"
    return "unknown"


def reliable_legacy_label(legacy_votes: dict) -> tuple[str | None, float]:
    usable = {str(label): int(count) for label, count in legacy_votes.items() if label in VALID_LABELS}
    total = sum(usable.values())
    if total < 3:
        return None, 0.0
    label, count = max(usable.items(), key=lambda item: item[1])
    fraction = count / total
    if label == "unknown" or fraction < 0.70:
        return None, float(fraction)
    return label, float(fraction)


def suggest_track_label(samples: list[dict], assignment: dict) -> dict:
    """Return a review suggestion, never a silent final override."""
    usable = sorted(samples, key=lambda row: row["features"]["quality"], reverse=True)[:10]
    evidence = {
        key: weighted_average(usable, key)
        for key in (
            "white", "red", "orange", "black", "stripe", "red_hue_consistency",
            "red_saturation", "white_brightness", "quality", "blur_variance",
        )
    }
    aspects = [
        (float(row["bbox"][3]) - float(row["bbox"][1]) + 1.0)
        / max(1.0, float(row["bbox"][2]) - float(row["bbox"][0]) + 1.0)
        for row in usable if "bbox" in row
    ]
    evidence["median_bbox_aspect"] = float(np.median(aspects)) if aspects else 2.0
    votes = Counter(
        classify_evidence_crop(row["features"])
        for row in usable
        if row["features"]["quality"] >= 0.24
    )
    independent = sum(votes.values())
    legacy_label, legacy_fraction = reliable_legacy_label(assignment.get("legacy_team_votes", {}))
    current = str(assignment["team"])
    suggested = current
    confidence = float(assignment["confidence"])
    reasons: list[str] = []

    # Several independent crops can lower the unknown threshold.  Black never
    # provides positive team evidence; it is only shown to the reviewer.
    devils_agree = votes["devils"] >= 3 and votes["devils"] >= max(1, votes["flyers"] + 2)
    flyers_agree = votes["flyers"] >= 3 and votes["flyers"] >= max(1, votes["devils"] + 2)
    if current != "unknown":
        # The purpose of this pass is to recover conservative unknowns, not to
        # destabilize labels that V2 already locked.  Every known track still
        # has four manual buttons if the reviewer sees a genuine error.
        suggested = current
        reasons.append("retained stable non-unknown V2 label for manual confirmation")
    elif evidence["black"] < 0.018 and evidence["red"] < 0.025:
        # White boards, ice, blue lines, and glass generated several confident
        # Phase A Flyer labels.  White/orange alone is not enough to recover an
        # unknown track without dark or red uniform detail.
        suggested = "unknown"
        reasons.append("white background lacks person-like dark/red uniform detail")
    elif (
        votes["official"] >= 3 and evidence["stripe"] >= 0.30 and evidence["black"] >= 0.07
        and evidence["quality"] >= 0.72 and evidence["red"] < 0.035
        and evidence["median_bbox_aspect"] >= 1.10
    ):
        # Match the sharp, balanced black/white torso signature of the two V2
        # officials.  This rejects legacy "official" matches that are actually
        # lettering, glass supports, sticks, or blurred board fragments.
        suggested = "official"
        confidence = min(0.94, 0.58 + 0.035 * votes["official"] + 0.20 * evidence["stripe"])
        reasons.append(f"{votes['official']} sharp striped-torso crops match successful officials")
    elif (
        legacy_label == "official" and legacy_fraction >= 0.82 and votes["official"] >= 3
        and evidence["stripe"] >= 0.28 and evidence["black"] >= 0.06 and evidence["quality"] >= 0.65
    ):
        suggested = "official"
        confidence = min(0.93, 0.60 + 0.25 * legacy_fraction + 0.15 * evidence["stripe"])
        reasons.append("strong Phase A official match plus stripe evidence")
    elif legacy_label == "official":
        suggested = "unknown"
        reasons.append("legacy official match rejected because sharp striped-person evidence is insufficient")
    elif devils_agree and evidence["red"] >= 0.085:
        suggested = "devils"
        confidence = min(0.97, 0.58 + 0.035 * votes["devils"] + 0.35 * evidence["red"])
        reasons.append(f"{votes['devils']} independent red-torso crops agree")
    elif flyers_agree and evidence["white"] >= 0.36 and evidence["black"] >= 0.025:
        suggested = "flyers"
        confidence = min(0.97, 0.56 + 0.03 * votes["flyers"] + 0.27 * evidence["white"])
        reasons.append(f"{votes['flyers']} independent white-torso crops agree")
    elif legacy_label == "devils" and legacy_fraction >= 0.78 and evidence["red"] >= 0.065:
        suggested = "devils"
        confidence = min(0.92, 0.56 + 0.24 * legacy_fraction + 0.42 * evidence["red"])
        reasons.append("strong Phase A match plus red-torso evidence")
    elif (
        legacy_label == "flyers" and legacy_fraction >= 0.78 and evidence["white"] >= 0.38
        and evidence["black"] >= 0.025
    ):
        suggested = "flyers"
        confidence = min(0.92, 0.55 + 0.23 * legacy_fraction + 0.22 * evidence["white"])
        reasons.append("strong Phase A match plus white-torso evidence")
    elif current != "unknown":
        reasons.append("retained stable V2 label; evidence did not justify changing it")
    else:
        suggested = "unknown"
        confidence = max(0.50, min(0.90, float(assignment["confidence"])))
        reasons.append("ambiguous evidence; visual review required")

    if current == "unknown" and suggested == "devils" and evidence["red"] < 0.055:
        suggested = "unknown"
        reasons.append("rejected Devils suggestion because red torso evidence is absent")
    if current == "unknown" and suggested == "flyers" and evidence["white"] < 0.32:
        suggested = "unknown"
        reasons.append("rejected Flyers suggestion because white torso evidence is absent")
    return {
        "label": suggested,
        "confidence": float(confidence),
        "basis": "; ".join(reasons),
        "crop_votes": {label: int(votes[label]) for label in VALID_LABELS},
        "evidence": evidence,
        "high_quality_crops_used": int(independent),
        "legacy_reliable_label": legacy_label,
        "legacy_reliable_fraction": legacy_fraction,
    }


def diverse_best_samples(samples: list[dict], maximum: int = 10) -> list[dict]:
    ranked = sorted(samples, key=lambda row: row["features"]["quality"], reverse=True)
    selected: list[dict] = []
    for minimum_gap in (5, 2, 0):
        for row in ranked:
            if row in selected:
                continue
            if all(abs(int(row["frame_index"]) - int(other["frame_index"])) >= minimum_gap for other in selected):
                selected.append(row)
            if len(selected) >= maximum:
                return sorted(selected, key=lambda row: int(row["frame_index"]))
    return sorted(selected, key=lambda row: int(row["frame_index"]))


def full_frame_context(frame: np.ndarray, box: list[int]) -> np.ndarray:
    context = frame.copy()
    x0, y0, x1, y1 = [int(value) for value in box]
    cv2.rectangle(context, (x0, y0), (x1, y1), (0, 255, 255), 5, cv2.LINE_AA)
    cv2.circle(context, ((x0 + x1) // 2, (y0 + y1) // 2), 8, (0, 255, 255), -1, cv2.LINE_AA)
    return context


def save_jpeg(path: Path, image: np.ndarray, width: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if width and image.shape[1] > width:
        height = max(1, int(round(image.shape[0] * width / image.shape[1])))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"Could not write {path}")


def build_review_records(frames: list[np.ndarray], v2_dir: Path, review_dir: Path) -> list[dict]:
    assignment_rows = read_json(v2_dir / "team_assignments_v2.json")["assignments"]
    accepted = {int(row["track_id"]): row for row in assignment_rows if row["renderable_on_ice_track"]}
    if len(accepted) != 60:
        raise ValueError(f"Expected 60 accepted V2 tracks, found {len(accepted)}")
    rows_by_track: dict[int, list[dict]] = defaultdict(list)
    for row in read_json(v2_dir / "stabilized_tracking_results.json")["tracks"]:
        track_id = int(row["track_id"])
        if track_id in accepted and not bool(row["predicted"]):
            features = upper_torso_features(frames[int(row["frame_index"])], row["bbox"])
            rows_by_track[track_id].append({**row, "features": features})

    asset_dir = review_dir / "assets"
    records = []
    for track_id in sorted(accepted):
        assignment = accepted[track_id]
        samples = rows_by_track[track_id]
        suggestion = suggest_track_label(samples, assignment)
        selected = diverse_best_samples(samples, maximum=10)
        crop_entries = []
        for index, sample in enumerate(selected, 1):
            frame_index = int(sample["frame_index"])
            x0, y0, x1, y1 = sample["features"]["crop_box"]
            crop = frames[frame_index][y0 : y1 + 1, x0 : x1 + 1]
            relative = Path("assets") / f"track_{track_id:06d}_torso_{index:02d}_frame_{frame_index:06d}.jpg"
            save_jpeg(review_dir / relative, crop, width=180)
            crop_entries.append({
                "path": relative.as_posix(),
                "frame_index": frame_index,
                "quality": float(sample["features"]["quality"]),
            })
        context_sample = max(samples, key=lambda row: row["features"]["quality"])
        context_frame = int(context_sample["frame_index"])
        context_path = Path("assets") / f"track_{track_id:06d}_context_frame_{context_frame:06d}.jpg"
        save_jpeg(review_dir / context_path, full_frame_context(frames[context_frame], context_sample["bbox"]), width=640)
        legacy_label, legacy_fraction = reliable_legacy_label(assignment.get("legacy_team_votes", {}))
        records.append({
            "track_id": track_id,
            "current_v2_label": assignment["team"],
            "current_v2_confidence": float(assignment["confidence"]),
            "legacy_label": legacy_label,
            "legacy_match_fraction": legacy_fraction,
            "legacy_votes": assignment.get("legacy_team_votes", {}),
            "observation_count": int(assignment["detection_count"]),
            "frame_start": int(assignment["frame_start"]),
            "frame_end": int(assignment["frame_end"]),
            "torso_crop_count": len(crop_entries),
            "short_track_crop_notice": (
                None if len(crop_entries) >= 6
                else f"Only {len(crop_entries)} distinct fresh observations exist; no crops were duplicated."
            ),
            "crops": crop_entries,
            "context_path": context_path.as_posix(),
            "context_frame_index": context_frame,
            "suggestion": suggestion,
        })
    return records


def reviewer_html(records: list[dict]) -> str:
    data = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    cards = []
    for record in records:
        tid = int(record["track_id"])
        suggestion = record["suggestion"]
        evidence = suggestion["evidence"]
        crop_html = "".join(
            f'<figure><img src="{html.escape(crop["path"])}" alt="Track {tid} torso frame {crop["frame_index"]}">'
            f'<figcaption>f{crop["frame_index"]} · q {crop["quality"]:.2f}</figcaption></figure>'
            for crop in record["crops"]
        )
        legacy = record["legacy_label"] or "none reliable"
        notice = f'<p class="notice">{html.escape(record["short_track_crop_notice"])}</p>' if record["short_track_crop_notice"] else ""
        buttons = "".join(
            f'<button type="button" data-label="{label}" onclick="choose({tid},\'{label}\')">{label.title()}</button>'
            for label in VALID_LABELS
        )
        cards.append(f"""
        <article class="track-card" id="track-{tid}" data-current="{record['current_v2_label']}" data-suggested="{suggestion['label']}">
          <header><h2>Track {tid}</h2><span class="choice" id="choice-{tid}">Suggestion: {suggestion['label'].title()}</span></header>
          <div class="meta">
            <span>V2: <b>{record['current_v2_label']}</b> {record['current_v2_confidence']:.2f}</span>
            <span>Suggestion: <b>{suggestion['label']}</b> {suggestion['confidence']:.2f}</span>
            <span>Legacy: <b>{legacy}</b> {record['legacy_match_fraction']:.0%}</span>
            <span>Observations: <b>{record['observation_count']}</b></span>
            <span>Frames: <b>{record['frame_start']}–{record['frame_end']}</b></span>
          </div>
          <p class="basis">{html.escape(suggestion['basis'])}</p>
          <div class="evidence">
            <span>white {evidence['white']:.3f}</span><span>red {evidence['red']:.3f}</span>
            <span>orange {evidence['orange']:.3f}</span><span>black {evidence['black']:.3f}</span>
            <span>stripe {evidence['stripe']:.3f}</span><span>red consistency {evidence['red_hue_consistency']:.3f}</span>
          </div>
          <div class="review-grid">
            <div><h3>On-ice context · frame {record['context_frame_index']}</h3><img class="context" src="{record['context_path']}" alt="Track {tid} full-frame context crop"></div>
            <div><h3>Best distinct upper-torso observations</h3><div class="crops">{crop_html}</div>{notice}</div>
          </div>
          <div class="buttons" role="group" aria-label="Choose label for track {tid}">{buttons}</div>
        </article>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NJ Devils V2 Team Track Review</title>
<style>
:root{{--bg:#0d1015;--panel:#171c24;--line:#303947;--text:#edf2f7;--muted:#aab4c2;--red:#ef4444;--orange:#f59e0b;--yellow:#facc15;--gray:#9ca3af}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.42 system-ui,sans-serif}}
.toolbar{{position:sticky;top:0;z-index:5;background:#0d1015f2;border-bottom:1px solid var(--line);padding:14px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.toolbar h1{{font-size:19px;margin:0 18px 0 0}}.toolbar button,.toolbar label{{background:#26303d;color:white;border:1px solid #465468;border-radius:7px;padding:8px 12px;cursor:pointer}}
.toolbar .primary{{background:#166534;border-color:#22c55e}}#progress{{color:var(--muted)}}main{{max-width:1500px;margin:auto;padding:18px}}
.warning{{background:#3a2410;border:1px solid #b76913;padding:12px 16px;border-radius:8px;margin-bottom:16px}}
.track-card{{background:var(--panel);border:2px solid var(--line);border-radius:12px;margin:0 0 18px;padding:16px}}
.track-card.reviewed{{border-color:#22c55e}}.track-card header{{display:flex;align-items:center;justify-content:space-between}}h2,h3{{margin:0 0 9px}}h3{{font-size:14px;color:var(--muted)}}
.choice{{font-weight:700;padding:6px 10px;background:#252d38;border-radius:6px}}.meta,.evidence{{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);margin:7px 0}}
.basis{{margin:7px 0;color:#cdd6e3}}.evidence span{{font-family:ui-monospace,monospace;background:#10141a;padding:4px 7px;border-radius:4px}}
.review-grid{{display:grid;grid-template-columns:minmax(270px,36%) 1fr;gap:14px;margin-top:14px}}.context{{width:100%;max-height:300px;object-fit:contain;background:#050607}}
.crops{{display:grid;grid-template-columns:repeat(5,minmax(80px,1fr));gap:7px}}figure{{margin:0;background:#090b0e}}figure img{{width:100%;height:130px;object-fit:contain}}figcaption{{font-size:12px;color:var(--muted);padding:3px 5px}}
.notice{{color:#fbbf24}}.buttons{{display:flex;gap:8px;margin-top:14px}}.buttons button{{border:2px solid transparent;border-radius:7px;padding:9px 18px;background:#343d4a;color:white;font-weight:700;cursor:pointer}}
.buttons button[data-label=devils]{{color:#fecaca}}.buttons button[data-label=flyers]{{color:#fed7aa}}.buttons button[data-label=official]{{color:#fef08a}}.buttons button.selected{{border-color:white;box-shadow:0 0 0 2px #22c55e}}
@media(max-width:800px){{.review-grid{{grid-template-columns:1fr}}.crops{{grid-template-columns:repeat(3,1fr)}}}}
</style></head><body>
<div class="toolbar"><h1>NJ Devils Hockey · Stable Track Team Review</h1>
<span id="progress"></span><button onclick="showOnly('all')">All</button><button onclick="showOnly('unknown')">V2 unknown</button>
<label>Import overrides <input id="import" type="file" accept="application/json" hidden onchange="importOverrides(this.files[0])"></label>
<button class="primary" onclick="downloadOverrides()">Download team_label_overrides.json</button></div>
<main><p class="warning"><b>Review the player, not the pants:</b> Devils have predominantly red torsos; Flyers predominantly white torsos with possible orange/black accents; officials have black-and-white striped torsos. Black pants are not team evidence. Suggestions are not final until visually reviewed.</p>
{''.join(cards)}</main>
<script id="track-data" type="application/json">{data}</script>
<script>
const tracks=JSON.parse(document.getElementById('track-data').textContent);const selections={{}};const reviewed={{}};
for(const t of tracks){{selections[t.track_id]=t.suggestion.label;reviewed[t.track_id]=false;paint(t.track_id)}}
function paint(id){{const card=document.getElementById(`track-${{id}}`);card.classList.toggle('reviewed',reviewed[id]);card.querySelectorAll('button[data-label]').forEach(b=>b.classList.toggle('selected',b.dataset.label===selections[id]));document.getElementById(`choice-${{id}}`).textContent=`${{reviewed[id]?'Reviewed':'Suggestion'}}: ${{selections[id][0].toUpperCase()+selections[id].slice(1)}}`;updateProgress()}}
function choose(id,label){{selections[id]=label;reviewed[id]=true;paint(id);localStorage.setItem('njdevils-team-review',JSON.stringify({{selections,reviewed}}))}}
function updateProgress(){{document.getElementById('progress').textContent=`${{Object.values(reviewed).filter(Boolean).length}} / ${{tracks.length}} reviewed`}}
function showOnly(mode){{document.querySelectorAll('.track-card').forEach(card=>card.style.display=(mode==='all'||card.dataset.current===mode)?'block':'none')}}
function payload(){{return{{schema_version:1,source:'hockey_polygon_demo_v2/team_track_review/team_track_reviewer.html',all_accepted_tracks_reviewed:Object.values(reviewed).every(Boolean),overrides:tracks.map(t=>({{track_id:t.track_id,label:selections[t.track_id],reviewed:reviewed[t.track_id],previous_v2_label:t.current_v2_label,automatic_suggestion:t.suggestion.label}}))}}}}
function downloadOverrides(){{if(!Object.values(reviewed).every(Boolean)){{alert('Review all 60 tracks before downloading overrides.');return}}const blob=new Blob([JSON.stringify(payload(),null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='team_label_overrides.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}}
async function importOverrides(file){{if(!file)return;const data=JSON.parse(await file.text());const rows=Array.isArray(data.overrides)?data.overrides:Object.entries(data.overrides||{{}}).map(([track_id,label])=>({{track_id:+track_id,label,reviewed:true}}));for(const row of rows){{if(selections[row.track_id]!==undefined&&{json.dumps(list(VALID_LABELS))}.includes(row.label)){{selections[row.track_id]=row.label;reviewed[row.track_id]=row.reviewed!==false;paint(row.track_id)}}}}}}
const saved=localStorage.getItem('njdevils-team-review');if(saved){{try{{const state=JSON.parse(saved);for(const t of tracks){{if(state.selections?.[t.track_id])selections[t.track_id]=state.selections[t.track_id];if(state.reviewed?.[t.track_id])reviewed[t.track_id]=true;paint(t.track_id)}}}}catch(e){{console.warn(e)}}}}
</script></body></html>"""


def generate_reviewer(frames: list[np.ndarray], v2_dir: Path, review_dir: Path) -> list[dict]:
    records = build_review_records(frames, v2_dir, review_dir)
    write_json(review_dir / "team_label_suggestions.json", {
        "schema_version": 1,
        "accepted_track_count": len(records),
        "note": "Suggestions require visual review and are not final overrides.",
        "tracks": records,
    })
    (review_dir / "team_track_reviewer.html").write_text(reviewer_html(records), encoding="utf-8")
    write_json(review_dir / "review_manifest.json", {
        "accepted_track_count": len(records),
        "current_v2_labels": dict(Counter(row["current_v2_label"] for row in records)),
        "automatic_suggestions": dict(Counter(row["suggestion"]["label"] for row in records)),
        "tracks_with_fewer_than_six_distinct_observations": [
            row["track_id"] for row in records if row["torso_crop_count"] < 6
        ],
        "override_path_expected": str(review_dir / "team_label_overrides.json"),
        "final_render_ready": (review_dir / "team_label_overrides.json").exists(),
    })
    return records


def load_overrides(path: Path, accepted_track_ids: set[int]) -> dict[int, str]:
    payload = read_json(path)
    raw = payload.get("overrides", payload)
    if isinstance(raw, list):
        unreviewed = sorted(int(row["track_id"]) for row in raw if row.get("reviewed") is False)
        if unreviewed:
            raise ValueError(f"Reviewer export contains unreviewed tracks: {unreviewed}")
        overrides = {int(row["track_id"]): str(row["label"]).lower() for row in raw}
    elif isinstance(raw, dict):
        overrides = {int(key): str(value).lower() for key, value in raw.items()}
    else:
        raise ValueError("team_label_overrides.json must contain a list or object named overrides")
    invalid_ids = sorted(set(overrides) - accepted_track_ids)
    invalid_labels = {track_id: label for track_id, label in overrides.items() if label not in VALID_LABELS}
    missing = sorted(accepted_track_ids - set(overrides))
    if invalid_ids:
        raise ValueError(f"Overrides contain unknown accepted track IDs: {invalid_ids}")
    if invalid_labels:
        raise ValueError(f"Overrides contain invalid labels: {invalid_labels}")
    if missing:
        raise ValueError(f"Overrides must review all 60 accepted tracks; missing: {missing}")
    return overrides


def final_contact_sheet(frames: list[np.ndarray], rows_by_frame: dict[int, list[dict]]) -> np.ndarray:
    candidates: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    for frame_index, rows in rows_by_frame.items():
        for row in rows:
            if row["predicted"]:
                continue
            features = upper_torso_features(frames[frame_index], row["bbox"])
            candidates[row["team"]].append((features["quality"] * row["team_confidence"], frame_index, row))
    labels = ("devils", "flyers", "official", "unknown")
    canvas = np.full((4 * 170, 6 * 190, 3), 238, dtype=np.uint8)
    for row_index, label in enumerate(labels):
        used_tracks = set()
        column = 0
        ordered = sorted(
            candidates[label],
            key=lambda item: (float(item[0]), -int(item[1]), -int(item[2]["track_id"])),
            reverse=True,
        )
        for _score, frame_index, row in ordered:
            if row["track_id"] in used_tracks:
                continue
            used_tracks.add(row["track_id"])
            tile = v2.crop_tile(frames[frame_index], {**row, "torso_v2": upper_torso_features(frames[frame_index], row["bbox"])}, 190, 132)
            y0, x0 = row_index * 170, column * 190
            canvas[y0 : y0 + 132, x0 : x0 + 190] = tile
            cv2.putText(canvas, f"T{row['track_id']} {label.upper()}", (x0 + 5, y0 + 153), cv2.FONT_HERSHEY_SIMPLEX, 0.43, v2.TEAM_COLORS[label], 1, cv2.LINE_AA)
            column += 1
            if column == 6:
                break
        cv2.putText(canvas, label.upper(), (4, row_index * 170 + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def render_final(
    frames: list[np.ndarray], metadata: dict, v1_dir: Path, v2_dir: Path,
    review_dir: Path, final_dir: Path, rink_path: Path,
) -> dict:
    assignments_payload = read_json(v2_dir / "team_assignments_v2.json")
    accepted_rows = [row for row in assignments_payload["assignments"] if row["renderable_on_ice_track"]]
    accepted_ids = {int(row["track_id"]) for row in accepted_rows}
    overrides_path = review_dir / "team_label_overrides.json"
    overrides = load_overrides(overrides_path, accepted_ids)
    final_dir.mkdir(parents=True, exist_ok=True)

    assignments = {}
    for row in accepted_rows:
        track_id = int(row["track_id"])
        assignments[track_id] = {
            **row,
            "previous_v2_label": row["team"],
            "team": overrides[track_id],
            "manual_override_applied": overrides[track_id] != row["team"],
            "method": "manual_track_review_override_on_immutable_v2_tracking",
        }

    rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    for source in read_json(v2_dir / "stabilized_tracking_results.json")["tracks"]:
        track_id = int(source["track_id"])
        if track_id not in assignments:
            continue
        row = {**source, "team": assignments[track_id]["team"], "team_confidence": 1.0}
        rows_by_frame[int(row["frame_index"])].append(row)

    registrations = {
        int(row["frame_index"]): row
        for row in read_json(v1_dir / "per_frame_rink_homographies.json")["frames"]
    }
    if sorted(registrations) != list(range(277)):
        raise ValueError("Registration records do not cover source frames 0..276")
    rink = cv2.imread(str(rink_path), cv2.IMREAD_COLOR)
    if rink is None:
        raise FileNotFoundError(rink_path)
    playable = phase_b.rink_playable_contour(rink)
    frame_size = (int(metadata["width"]), int(metadata["height"]))
    rink_size = (rink.shape[1], rink.shape[0])
    panel_width = int(round(rink.shape[1] * frame_size[1] / rink.shape[0]))
    fps = float(metadata["fps"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    paths = {
        "showcase": final_dir / "final_hockey_showcase.mp4",
        "boxes": final_dir / "broadcast_player_boxes_final.mp4",
        "broadcast_polygons": final_dir / "broadcast_team_polygons_final.mp4",
        "rink_polygons": final_dir / "rink_team_polygons_final.mp4",
        "side_by_side": final_dir / "side_by_side_team_polygons_final.mp4",
    }
    writers = {
        "showcase": cv2.VideoWriter(str(paths["showcase"]), fourcc, fps, frame_size),
        "boxes": cv2.VideoWriter(str(paths["boxes"]), fourcc, fps, frame_size),
        "broadcast_polygons": cv2.VideoWriter(str(paths["broadcast_polygons"]), fourcc, fps, frame_size),
        "rink_polygons": cv2.VideoWriter(str(paths["rink_polygons"]), fourcc, fps, rink_size),
        "side_by_side": cv2.VideoWriter(str(paths["side_by_side"]), fourcc, fps, (frame_size[0] + panel_width, frame_size[1])),
    }
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("Could not open final video writers")

    stats = Counter()
    heat_points = {"devils": [], "flyers": []}
    projected_rows = []
    for frame_index, raw in enumerate(frames):
        record = registrations[frame_index]
        rows = sorted(rows_by_frame.get(frame_index, []), key=lambda row: row["bbox"][3])
        boxes_frame = raw.copy()
        broadcast = raw.copy()
        showcase = raw.copy()
        rink_frame = rink.copy()
        counts = Counter(row["team"] for row in rows)
        team_image = {"devils": [], "flyers": []}
        team_rink = {"devils": [], "flyers": []}
        registration_ok = record.get("status") == "ok" and record.get("homography_matrix") is not None
        if registration_ok:
            phase_b.draw_visible_region(rink_frame, record["homography_matrix"], frame_size)
            stats["registration_ok_frames"] += 1
        else:
            stats["registration_rejected_frames"] += 1
        for row in rows:
            v2.draw_track(boxes_frame, row)
            v2.draw_track(broadcast, row)
            v2.draw_track(showcase, row)
            if row["predicted"]:
                stats["predicted_boxes_drawn"] += 1
                continue
            team = row["team"]
            if team == "official":
                stats["official_observations_excluded"] += 1
                continue
            if team not in ("devils", "flyers"):
                stats["unknown_observations_excluded"] += 1
                continue
            if not registration_ok:
                stats["fresh_team_observations_rejected_by_registration"] += 1
                continue
            foot = tuple(float(value) for value in row["footpoint"])
            projected = phase_b.project_broadcast_point(record["homography_matrix"], foot)
            if projected is None:
                stats["nonfinite_projections"] += 1
                continue
            if not phase_b.point_inside_rink(playable, projected):
                stats["out_of_rink_projections"] += 1
                continue
            team_image[team].append(foot)
            team_rink[team].append(projected)
            heat_points[team].append(projected)
            stats[f"{team}_projected_observations"] += 1
            cv2.circle(broadcast, tuple(int(round(v)) for v in foot), 5, v2.TEAM_COLORS[team], -1, cv2.LINE_AA)
            cv2.circle(showcase, tuple(int(round(v)) for v in foot), 5, v2.TEAM_COLORS[team], -1, cv2.LINE_AA)
            cv2.circle(rink_frame, tuple(int(round(v)) for v in projected), 8, v2.TEAM_COLORS[team], -1, cv2.LINE_AA)
            projected_rows.append({
                "source_frame_index": frame_index, "track_id": int(row["track_id"]), "team": team,
                "broadcast_footpoint": list(foot), "canonical_rink_xy": list(projected),
                "registration_status": "ok", "registration_confidence": float(record.get("confidence") or 0.0),
                "reference_frame_index": record.get("reference_frame_index"),
                "predicted_track_box": False,
                "projection_direction": "fresh_smoothed_broadcast_footpoint_to_rink_uses_stored_H_directly",
            })
        if registration_ok:
            for team in ("devils", "flyers"):
                if phase_b.team_polygon_is_stable(team_image[team], team_rink[team]):
                    phase_b.draw_team_polygon(broadcast, team_image[team], v2.TEAM_COLORS[team])
                    phase_b.draw_team_polygon(showcase, team_image[team], v2.TEAM_COLORS[team])
                    phase_b.draw_team_polygon(rink_frame, team_rink[team], v2.TEAM_COLORS[team])
                    stats[f"frames_with_{team}_polygon"] += 1
                else:
                    stats[f"frames_with_insufficient_{team}_points"] += 1
        else:
            phase_b.rejection_banner(broadcast, record)
            phase_b.rejection_banner(showcase, record)
            phase_b.rejection_banner(rink_frame, record)
        v2.v2_header(boxes_frame, frame_index, counts, "FINAL REVIEWED TEAM LABELS", "solid=fresh | dashed/dim=prediction")
        v2.v2_header(broadcast, frame_index, counts, "FINAL BROADCAST TEAM POLYGONS", "manual track-level review | predictions never projected")
        v2.v2_header(rink_frame, frame_index, counts, "FINAL CANONICAL RINK POLYGONS", "blue polygon = current broadcast footprint")
        v2.v2_header(showcase, frame_index, counts, "NJ DEVILS HOCKEY - FINAL REVIEWED LABELS", "stable track labels | dynamic rink registration")
        writers["boxes"].write(boxes_frame)
        writers["broadcast_polygons"].write(broadcast)
        writers["rink_polygons"].write(rink_frame)
        writers["side_by_side"].write(np.hstack([broadcast, cv2.resize(rink_frame, (panel_width, frame_size[1]), interpolation=cv2.INTER_AREA)]))
        writers["showcase"].write(showcase)
    for writer in writers.values():
        writer.release()

    devils_heatmap = final_dir / "devils_rink_heatmap_final.png"
    flyers_heatmap = final_dir / "flyers_rink_heatmap_final.png"
    contact_sheet = final_dir / "final_team_assignment_contact_sheet.png"
    cv2.imwrite(str(devils_heatmap), phase_b.rink_heatmap(rink, heat_points["devils"], "NEW JERSEY DEVILS - FINAL RINK HEATMAP", cv2.COLORMAP_HOT, playable))
    cv2.imwrite(str(flyers_heatmap), phase_b.rink_heatmap(rink, heat_points["flyers"], "PHILADELPHIA FLYERS - FINAL RINK HEATMAP", cv2.COLORMAP_TURBO, playable))
    cv2.imwrite(str(contact_sheet), final_contact_sheet(frames, rows_by_frame))
    final_assignment_path = final_dir / "final_team_assignments.json"
    write_json(final_assignment_path, {
        "method": "manual_track_review_override_on_immutable_v2_tracking",
        "source_overrides": str(overrides_path),
        "assignments": [assignments[key] for key in sorted(assignments)],
    })
    write_json(final_dir / "projected_rink_observations_final.json", {"observations": projected_rows})
    checks = [phase_b.verify_video(path, 277) for path in paths.values()]
    summary = {
        "stage": "hockey_polygon_demo_final_team_review",
        "status": "complete" if all(check["fully_decodable"] for check in checks) else "video_verification_failed",
        "source_frame_count": len(frames),
        "tracking_reused_unchanged": True,
        "registration_reused_unchanged": True,
        "blue_visible_rink_polygon_preserved": True,
        "predicted_boxes_projected": False,
        "stale_coordinates_reused": False,
        "team_track_counts": dict(Counter(row["team"] for row in assignments.values())),
        "manual_label_changes": sum(row["manual_override_applied"] for row in assignments.values()),
        "rendering": dict(stats),
        "video_verification": checks,
        "outputs": {key: str(path) for key, path in paths.items()} | {
            "devils_heatmap": str(devils_heatmap), "flyers_heatmap": str(flyers_heatmap),
            "contact_sheet": str(contact_sheet), "assignments": str(final_assignment_path),
        },
    }
    write_json(final_dir / "final_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    video_path = project_path(args.video)
    v1_dir = project_path(args.v1_dir)
    v2_dir = project_path(args.v2_dir)
    review_dir = project_path(args.review_dir)
    final_dir = project_path(args.final_dir)
    rink_path = project_path(args.rink)
    frames, metadata = decode_all_frames(video_path)
    records = generate_reviewer(frames, v2_dir, review_dir)
    overrides_path = review_dir / "team_label_overrides.json"
    print(json.dumps({
        "reviewer": str(review_dir / "team_track_reviewer.html"),
        "accepted_tracks": len(records),
        "v2_labels": dict(Counter(row["current_v2_label"] for row in records)),
        "automatic_suggestions": dict(Counter(row["suggestion"]["label"] for row in records)),
        "overrides_present": overrides_path.exists(),
    }, indent=2, sort_keys=True))
    if args.mode == "final" and not overrides_path.exists():
        raise FileNotFoundError(f"Manual overrides do not exist: {overrides_path}")
    if args.mode == "review" or (args.mode == "auto" and not overrides_path.exists()):
        print("Reviewer generated; final rendering intentionally skipped until manual overrides exist.")
        return 0
    summary = render_final(frames, metadata, v1_dir, v2_dir, review_dir, final_dir, rink_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
