#!/usr/bin/env python3
"""Generate the separate browser annotator for NJ Devils rink registration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import annotate_field_keypoints as base
from prototype4_pipeline.field_registration.annotations import empty_annotation_document, load_annotations
from prototype4_pipeline.field_registration.landmarks import load_landmark_config
from prototype4_pipeline.field_registration.shot_detection import read_video_metadata, save_frame_image


DEFAULT_VIDEO = "/afs/ece.cmu.edu/usr/zllenza/research/prototype4/videos/njdevils.mp4"
DEFAULT_RINK = "assets/hockey/icerink.jpg"
DEFAULT_LANDMARKS = "configs/canonical_hockey_rink_landmarks.json"
DEFAULT_OUTPUT_DIR = "outputs/njdevils/hockey_polygon_demo/rink_registration_annotations"
DEFAULT_FRAMES = [0, 35, 70, 105, 138, 175, 210, 245, 276]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--canonical-rink-image", default=DEFAULT_RINK)
    parser.add_argument("--landmarks", default=DEFAULT_LANDMARKS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", default=",".join(str(value) for value in DEFAULT_FRAMES))
    return parser.parse_args()


def parse_frames(raw: str, frame_count: int) -> list[int]:
    frames = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    invalid = [value for value in frames if value < 0 or value >= frame_count]
    if invalid:
        raise ValueError(f"Frame indices outside [0, {frame_count - 1}]: {invalid}")
    return frames


def extract_frames(video: Path, output_dir: Path, metadata, frame_indices: list[int]) -> list[dict]:
    frames_dir = output_dir / "representative_frames"
    rows = []
    for frame_index in frame_indices:
        frame_path = frames_dir / f"frame_{frame_index:06d}.png"
        if not frame_path.exists():
            save_frame_image(video, frame_index, frame_path)
        with Image.open(frame_path) as image:
            width, height = image.size
        rows.append({
            "frame_index": frame_index,
            "timestamp_seconds": frame_index / metadata.fps if metadata.fps else 0.0,
            "frame_path": str(frame_path.relative_to(PROJECT_ROOT)),
            "filename": frame_path.name,
            "source_width": width,
            "source_height": height,
        })
    return rows


def replace_function(source: str, name: str, replacement: str, next_name: str) -> str:
    start = source.index(f"function {name}(")
    try:
        end = source.index(f"function {next_name}(", start)
    except ValueError:
        if next_name != "renderLandmarks":
            raise
        end = source.index("\nrenderLandmarks();", start)
    return source[:start] + replacement.rstrip() + "\n" + source[end:]


def write_preset_diagnostic(rink_image: Path, landmarks: list[dict], output_path: Path) -> None:
    image = Image.open(rink_image).convert("RGB")
    legend_width = 650
    canvas = Image.new("RGB", (image.width + legend_width, max(image.height, 24 * len(landmarks) + 24)), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(landmarks, start=1):
        x, y = (float(value) for value in row["template_xy"])
        radius = 8
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ff1744", outline="white", width=2)
        draw.text((x + 10, y - 8), str(index), fill="#0057d9", stroke_width=2, stroke_fill="white")
        draw.text((image.width + 12, 12 + (index - 1) * 24), f"{index:02d}  {row['id']}  ({x:.1f}, {y:.1f})", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def customize_hockey_html(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    replacements = {
        "Prototype 4 Field Landmark Annotator": "NJ Devils Hockey Rink Keypoint Annotator",
        "Prototype 4 Dynamic Field Registration": "NJ Devils Dynamic Hockey Rink Registration",
        "Canonical Top-Down Field": "Canonical Top-Down Hockey Rink",
        "canonical top-down lacrosse field": "canonical top-down hockey rink",
        "Only label an exact physical point visible in both views. Do not use players, shadows, or approximate unmarked turf locations.":
            "Only label an exact physical point visible in both views. Do not use players, sticks, shadows, logos without an exact known canonical location, or approximate points on featureless ice.",
        "Only label landmarks that are genuinely visible. Do not guess hidden field corners or landmarks covered by players, refs, graphics, or boards.":
            "Only label exact rink markings that are genuinely visible. Do not guess hidden points or use players, sticks, shadows, broadcast graphics, or approximate featureless ice.",
        "left attacking-zone landmarks": "left-zone hockey landmarks",
        "right attacking-zone landmarks": "right-zone hockey landmarks",
        "restraining lines": "blue lines",
        "restraining-line": "blue-line",
        "Restraining-line": "Blue-line",
        "left/right restraining line at top and bottom boundaries": "left/right blue line at top and bottom boards",
        "field-line": "rink-line",
        "field line": "rink line",
        "field_keypoints.json": "rink_keypoints.json",
        "live homography preview": "live projected-rink preview",
        "category === 'restraining_line' || id.indexOf('restraining') >= 0": "category === 'blue_line' || id.indexOf('blue_line') >= 0",
        "row.category === 'center_line' || row.category === 'goal_line' || row.category === 'restraining_line'":
            "row.category === 'center_line' || row.category === 'goal_line' || row.category === 'blue_line'",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    html = html.replace("height: 285px", "height: 371.25px")
    html = html.replace("{width: 1509, height: 724}", "{width: 1568, height: 980}")
    html = html.replace("Number(templateSize.width) || 1509", "Number(templateSize.width) || 1568")
    html = html.replace("Number(templateSize.height) || 724", "Number(templateSize.height) || 980")
    html = html.replace(
        "setStatus('Free canonical point selected at (' + pendingCanonicalPoint[0].toFixed(1) + ', ' + pendingCanonicalPoint[1].toFixed(1) + '). Now click the exact corresponding physical point in the broadcast frame.');",
        "setStatus('Click the matching point in the broadcast frame');",
    )
    html = html.replace(
        "setStatus(annotationMode.value === 'free' ? 'Click an exact physical point on the canonical field first.' : 'Select a preset landmark, then click its exact broadcast-frame correspondence.');",
        "selectedPoint = null; setStatus(annotationMode.value === 'free' ? 'Click a point on the canonical rink' : 'Select a preset landmark, then click its exact broadcast-frame correspondence.');",
    )
    html = html.replace(
        "setStatus('Select the landmark on the canonical field first, then click the same physical landmark in the broadcast image.');",
        "setStatus(annotationMode.value === 'free' ? 'Click a point on the canonical rink' : 'Select the landmark on the canonical rink first, then click the same physical landmark in the broadcast image.');",
    )
    html = html.replace(
        "ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, templateSize.height); ctx.stroke();",
        "ctx.beginPath(); ctx.moveTo(x, 155); ctx.lineTo(x, 824); ctx.stroke();",
    )

    html = replace_function(html, "canonicalLineSegments", """
function canonicalLineSegments() {
  const segments = [[[260, 155], [1307, 155]], [[260, 824], [1307, 824]]];
  const seen = {};
  landmarks.filter(function(row) { return ['center_line', 'blue_line', 'goal_line'].indexOf(row.category) >= 0; }).forEach(function(row) {
    const x = Number(row.template_xy[0]);
    const key = x.toFixed(2);
    if (!seen[key]) { segments.push([[x, 155], [x, 824]]); seen[key] = true; }
  });
  return segments;
}
""", "canonicalCirclePoints")

    html = replace_function(html, "drawProjectedFieldPreview", """
function drawProjectedFieldPreview(ctx, H) {
  ctx.save();
  ctx.globalAlpha = 0.92;
  ctx.setLineDash([14, 8]);
  canonicalLineSegments().forEach(function(seg) { drawProjectedPolyline(ctx, H, seg, '#ffea00', 4); });
  ctx.setLineDash([]);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('center_spot', 'center_circle_top', 0, Math.PI * 2, 72), '#00e5ff', 4);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('left_top_faceoff_center', 'left_top_faceoff_circle_top', 0, Math.PI * 2, 64), '#ff65d8', 3);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('left_bottom_faceoff_center', 'left_bottom_faceoff_circle_top', 0, Math.PI * 2, 64), '#ff65d8', 3);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('right_top_faceoff_center', 'right_top_faceoff_circle_top', 0, Math.PI * 2, 64), '#ff65d8', 3);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('right_bottom_faceoff_center', 'right_bottom_faceoff_circle_top', 0, Math.PI * 2, 64), '#ff65d8', 3);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('left_crease_center', 'left_crease_top', -Math.PI / 2, Math.PI / 2, 36), '#6dff80', 3);
  drawProjectedPolyline(ctx, H, canonicalCirclePoints('right_crease_center', 'right_crease_top', Math.PI / 2, 3 * Math.PI / 2, 36), '#6dff80', 3);
  ctx.font = '18px Arial'; ctx.lineWidth = 5; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.fillStyle = '#ffea00';
  ctx.strokeText('live projected-rink preview (canonical to broadcast)', 18, 30);
  ctx.fillText('live projected-rink preview (canonical to broadcast)', 18, 30);
  ctx.restore();
}
""", "drawCanonicalField")

    html = replace_function(html, "paintedLineSegments", """
function paintedLineSegments() {
  const segments = canonicalLineSegments().slice();
  [
    canonicalCirclePoints('center_spot', 'center_circle_top', 0, Math.PI * 2, 72),
    canonicalCirclePoints('left_top_faceoff_center', 'left_top_faceoff_circle_top', 0, Math.PI * 2, 64),
    canonicalCirclePoints('left_bottom_faceoff_center', 'left_bottom_faceoff_circle_top', 0, Math.PI * 2, 64),
    canonicalCirclePoints('right_top_faceoff_center', 'right_top_faceoff_circle_top', 0, Math.PI * 2, 64),
    canonicalCirclePoints('right_bottom_faceoff_center', 'right_bottom_faceoff_circle_top', 0, Math.PI * 2, 64),
    canonicalCirclePoints('left_crease_center', 'left_crease_top', -Math.PI / 2, Math.PI / 2, 36),
    canonicalCirclePoints('right_crease_center', 'right_crease_top', Math.PI / 2, 3 * Math.PI / 2, 36),
  ].forEach(function(polyline) {
    for (let i = 0; i + 1 < polyline.length; i += 1) { segments.push([polyline[i], polyline[i + 1]]); }
  });
  return segments;
}
""", "nearestPointOnSegment")

    html = replace_function(html, "saveCorrespondenceAt", """
function saveCorrespondenceAt(x, y) {
  const frame = currentFrame();
  if (annotationMode.value === 'free') {
    if (!pendingCanonicalPoint) { setStatus('Click a point on the canonical rink'); return; }
    let replaceIndex = null;
    if (selectedPoint !== null && selectedPoint >= 0 && selectedPoint < annotations.length && isFreeRow(annotations[selectedPoint]) && Number(annotations[selectedPoint].frame_index) === Number(frame.frame_index)) {
      replaceIndex = selectedPoint;
    } else {
      const duplicate = visibleRowsForView().find(function(row) { return isFreeRow(row) && distanceSquared(row.canonical_xy, pendingCanonicalPoint) < 0.01; });
      if (duplicate) { replaceIndex = duplicate.__index; }
    }
    const existing = replaceIndex === null ? null : annotations[replaceIndex];
    const freeRow = {
      view_name: currentViewName(), frame_index: frame.frame_index, timestamp_seconds: frame.timestamp_seconds,
      frame_path: frame.frame_path, source_frame_width: frameImage.naturalWidth || frame.source_width,
      source_frame_height: frameImage.naturalHeight || frame.source_height,
      landmark_id: existing ? existing.landmark_id : nextFreeLandmarkId(frame.frame_index),
      image_xy: [x, y], canonical_xy: pendingCanonicalPoint.slice(), visibility: existing ? (existing.visibility || 'visible') : 'visible',
      confidence: existing ? (existing.confidence || 'manual') : 'manual', note: existing ? (existing.note || '') : ''
    };
    if (replaceIndex === null) { annotations.push(freeRow); } else { annotations[replaceIndex] = freeRow; }
    pendingCanonicalPoint = null;
    selectedPoint = null;
    updateLandmarkInfo(); renderPointList(); updateJsonBox(); drawOverlay(); drawCanonicalField();
    setStatus('Click a point on the canonical rink');
    return;
  }
  const landmark = selectedLandmark();
  if (!landmark) { return; }
  const duplicate = visibleRowsForView().find(function(row) { return row.landmark_id === landmark.id; });
  const row = {
    view_name: currentViewName(), frame_index: frame.frame_index, timestamp_seconds: frame.timestamp_seconds,
    frame_path: frame.frame_path, source_frame_width: frameImage.naturalWidth, source_frame_height: frameImage.naturalHeight,
    landmark_id: landmark.id, image_xy: [x, y], canonical_xy: landmark.template_xy.slice(), visibility: 'visible', confidence: 'manual', note: duplicate ? (duplicate.note || '') : ''
  };
  if (duplicate) { annotations[duplicate.__index] = row; selectedPoint = duplicate.__index; }
  else { annotations.push(row); selectedPoint = annotations.length - 1; }
  renderPointList(); updateJsonBox(); drawOverlay(); drawCanonicalField();
  setStatus('Saved ' + landmark.id + '. Select another preset or switch to Free correspondence.');
}
""", "deleteSelectedPoint")

    html = replace_function(html, "runBrowserSelfTest", """
function runBrowserSelfTest() {
  const initialCount = annotations.length;
  annotationMode.value = 'free'; canonicalSnap.value = 'none'; selectedPoint = null;
  const rect = fieldCanvas.getBoundingClientRect();
  function clickCanonicalNatural(x, y) {
    const bounds = fieldCanvas.getBoundingClientRect();
    fieldCanvas.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: bounds.left + x * bounds.width / fieldCanvas.width, clientY: bounds.top + y * bounds.height / fieldCanvas.height}));
  }
  function clickBroadcastNatural(x, y) {
    const bounds = frameImage.getBoundingClientRect();
    frameImage.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: bounds.left + x * bounds.width / frameImage.naturalWidth, clientY: bounds.top + y * bounds.height / frameImage.naturalHeight}));
  }
  if (fieldCanvas.width !== 1568 || fieldCanvas.height !== 980 || fieldImage.naturalWidth !== 1568 || fieldImage.naturalHeight !== 980) { throw new Error('natural canonical dimensions are wrong'); }
  clickCanonicalNatural(392, 245);
  if (!pendingCanonicalPoint || Math.abs(pendingCanonicalPoint[0] - 392) > 3 || Math.abs(pendingCanonicalPoint[1] - 245) > 3) { throw new Error('canonical display-to-natural coordinate scaling failed'); }
  pendingCanonicalPoint = null;
  const expectedIds = [];
  for (let i = 0; i < 5; i += 1) {
    clickCanonicalNatural(300 + i * 130, 240 + i * 105);
    clickBroadcastNatural(360 + i * 120, 250 + i * 70);
    expectedIds.push('free_frame_' + String(currentFrame().frame_index).padStart(6, '0') + '_' + String(i + 1).padStart(3, '0'));
    if (pendingCanonicalPoint !== null || selectedPoint !== null || annotationMode.value !== 'free') { throw new Error('free-pair transient state was not reset'); }
  }
  let freeRows = annotations.slice(initialCount);
  const ids = freeRows.map(function(row) { return row.landmark_id; });
  if (freeRows.length !== 5 || new Set(ids).size !== 5 || expectedIds.some(function(id) { return ids.indexOf(id) < 0; })) { throw new Error('five distinct sequential free IDs were not created'); }
  if (freeRows.filter(function(row) { return row.canonical_xy && row.image_xy; }).length !== 5) { throw new Error('five canonical/broadcast marker pairs were not retained'); }
  updateJsonBox();
  const exported = JSON.parse(jsonBox.value).annotations.slice(initialCount);
  if (exported.length !== 5) { throw new Error('JSON export does not contain all five free correspondences'); }
  const movingId = expectedIds[2];
  const unchangedIds = expectedIds.filter(function(id) { return id !== movingId; });
  const beforeOthers = {};
  unchangedIds.forEach(function(id) { beforeOthers[id] = JSON.stringify(annotations.find(function(row) { return row.landmark_id === id; })); });
  const movingIndex = annotations.findIndex(function(row) { return row.landmark_id === movingId; });
  selectedPoint = movingIndex; clickCanonicalNatural(777, 444); clickBroadcastNatural(888, 555);
  const moved = annotations.find(function(row) { return row.landmark_id === movingId; });
  if (!moved || Math.abs(moved.canonical_xy[0] - 777) > 3 || Math.abs(moved.image_xy[0] - 888) > 2) { throw new Error('selected free correspondence was not moved/replaced'); }
  unchangedIds.forEach(function(id) { if (JSON.stringify(annotations.find(function(row) { return row.landmark_id === id; })) !== beforeOthers[id]) { throw new Error('moving one point changed another point'); } });
  annotations.find(function(row) { return row.landmark_id === expectedIds[0]; }).note = 'browser workflow note';
  selectedPoint = annotations.findIndex(function(row) { return row.landmark_id === movingId; }); deleteSelectedPoint();
  if (annotations.slice(initialCount).length !== 4 || unchangedIds.some(function(id) { return !annotations.find(function(row) { return row.landmark_id === id; }); })) { throw new Error('deleting one point did not leave the other four intact'); }
  clickCanonicalNatural(1010, 470); clickBroadcastNatural(1030, 480);
  freeRows = annotations.slice(initialCount);
  if (freeRows.length !== 5 || new Set(freeRows.map(function(row) { return row.landmark_id; })).size !== 5) { throw new Error('new point could not be created after deletion'); }
  canonicalSnap.value = 'line';
  const lineSnap = snapCanonicalPoint([590, 300]);
  canonicalSnap.value = 'intersection';
  const intersectionSnap = snapCanonicalPoint([590, 155]);
  if (![lineSnap[0], lineSnap[1], intersectionSnap[0], intersectionSnap[1]].every(Number.isFinite)) { throw new Error('snapping failed'); }
  canonicalSnap.value = 'none';
  annotationMode.value = 'preset'; landmarkSelect.value = 'center_spot'; selectedPoint = null; clickBroadcastNatural(520, 430);
  const preset = annotations.find(function(row) { return row.landmark_id === 'center_spot'; });
  if (!preset || preset.canonical_xy[0] !== 783.5 || preset.canonical_xy[1] !== 489.5) { throw new Error('preset workflow failed'); }
  selectedPoint = annotations.indexOf(preset); deleteSelectedPoint();
  annotationMode.value = 'free'; selectedPoint = null; pendingCanonicalPoint = null; updateLandmarkInfo(); updateJsonBox(); drawOverlay(); drawCanonicalField(); setStatus('Click a point on the canonical rink');
  if (visibleRowsForView().filter(function(row) { return row.canonical_xy && row.image_xy; }).length !== 5) { throw new Error('final canonical/broadcast marker count is not five'); }
  const result = document.createElement('div');
  result.id = 'browser-self-test-result'; result.dataset.freeCount = '5';
  result.style.cssText = 'position:fixed;left:20px;bottom:20px;z-index:9999;background:#167c3a;color:white;padding:16px;font-weight:bold';
  result.textContent = 'BROWSER WORKFLOW PASS: 5 free pairs + preset, scaling, move, replace, note, snapping, delete, JSON';
  document.body.appendChild(result); document.title = 'PASS - Unlimited hockey correspondence workflow';
}
""", "renderLandmarks")
    html_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    video = project_path(args.video)
    rink_image = project_path(args.canonical_rink_image)
    landmarks_path = project_path(args.landmarks)
    output_dir = project_path(args.output_dir)
    html_path = output_dir / "hockey_rink_keypoint_annotator.html"
    output_json = output_dir / "rink_keypoints.json"
    if not rink_image.exists():
        raise FileNotFoundError(f"Canonical rink image not found: {rink_image}")
    metadata = read_video_metadata(video)
    frame_indices = parse_frames(args.frames, metadata.frame_count)
    frames = extract_frames(video, output_dir, metadata, frame_indices)
    landmark_config = load_landmark_config(landmarks_path)
    if landmark_config.get("template_size_px") != {"width": 1568, "height": 980}:
        raise ValueError("Hockey landmark config does not match the natural icerink.jpg dimensions")
    default_doc = empty_annotation_document("njdevils", str(video), str(landmarks_path.relative_to(PROJECT_ROOT)))
    default_doc["notes"] = [
        "Manual hockey rink correspondences only. Do not fabricate points.",
        "Click an exact canonical marking and the identical physical point in the broadcast frame.",
        "Stored homographies will map broadcast image coordinates to canonical rink coordinates.",
    ]
    annotation_doc = load_annotations(output_json, default=default_doc)
    with Image.open(rink_image) as image:
        rink_width, rink_height = image.size
    bundled_rink = output_dir / "canonical_rink" / "icerink.jpg"
    bundled_rink.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rink_image, bundled_rink)
    preset_diagnostic = output_dir / "preset_landmark_diagnostic.png"
    write_preset_diagnostic(rink_image, landmark_config["landmarks"], preset_diagnostic)
    rink_asset = {
        "source_path": str(rink_image),
        "stable_repo_path": str(rink_image.relative_to(PROJECT_ROOT)),
        "bundled_output_path": str(bundled_rink.relative_to(PROJECT_ROOT)),
        "browser_url": Path(os.path.relpath(bundled_rink, html_path.parent)).as_posix(),
        "width": rink_width,
        "height": rink_height,
    }
    base.write_html(html_path, output_json, annotation_doc, landmark_config, frames, rink_asset)
    customize_hockey_html(html_path)
    summary = {
        "status": "awaiting_manual_annotations" if not output_json.exists() else "annotations_present",
        "html": str(html_path),
        "annotation_json": str(output_json),
        "annotation_json_exists": output_json.exists(),
        "canonical_rink_image": rink_asset,
        "video_metadata": metadata.__dict__,
        "extracted_frames": frame_indices,
        "landmark_count": len(landmark_config["landmarks"]),
        "preset_landmark_diagnostic": str(preset_diagnostic),
        "next_step": "Open the HTML, label only exact visible pairs, then download/copy the generated JSON to annotation_json.",
    }
    (output_dir / "hockey_annotator_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
