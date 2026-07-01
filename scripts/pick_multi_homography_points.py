#!/usr/bin/env python3
"""Create an interactive multi-frame homography calibration picker."""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/nll_test4_multi_homography_points.json"
DEFAULT_HTML = "outputs/nll_test4/multi_homography_demos/segment_20s_10s_multi_homography/multi_frame_point_picker.html"
DEFAULT_TRACKING_FRAMES_DIR = "outputs/nll_test4/calibrated_segment_demos/segment_20s_10s_calibrated/frames"
DEFAULT_FIELD_TEMPLATE = "assets/field_templates/nll_field_topdown.png"
DEFAULT_RUN_ID = "nll_test4"
DEFAULT_POINT_NAMES = [
    "center_faceoff_dot",
    "center_circle_edge_1",
    "center_circle_edge_2",
    "visible_white_line_intersection_1",
    "visible_white_line_intersection_2",
    "restraining_line_intersection_1",
    "restraining_line_intersection_2",
    "boards_or_boundary_intersection",
    "visible_white_line_reference_1",
    "visible_white_line_reference_2",
]
COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Write one interactive multi-frame homography point picker HTML.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Multi-homography config JSON to create/update.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id for the config.")
    parser.add_argument("--field-template", default=DEFAULT_FIELD_TEMPLATE, help="Top-down field template image.")
    parser.add_argument("--tracking-frames-dir", default=DEFAULT_TRACKING_FRAMES_DIR, help="Directory containing frame_*_tracking_overlay.png files.")
    parser.add_argument("--html-output", default=DEFAULT_HTML, help="Output interactive HTML picker path.")
    parser.add_argument("--mode", choices=["html"], default="html", help="Only HTML mode is supported.")
    parser.add_argument("--point-names", default=",".join(DEFAULT_POINT_NAMES), help="Comma-separated default point names for new point pairs.")
    parser.add_argument("--add-default-mid-frame", type=int, default=30, help="Frame index to seed as an empty mid-zone keyframe when missing.")
    return parser.parse_args()


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def browser_rel_path(path, html_output):
    rel = os.path.relpath(str(path.resolve()), str(html_output.parent.resolve()))
    return Path(rel).as_posix()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_data_uri(path):
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), image.size


def color_for(index):
    return COLORS[index % len(COLORS)]


def frame_index_from_path(path):
    match = re.search(r"frame_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_tracking_frames(frames_dir, html_output):
    frames = []
    for path in sorted(frames_dir.glob("frame_*_tracking_overlay.png")):
        frame_index = frame_index_from_path(path)
        if frame_index is None:
            continue
        frames.append({
            "frame_index": frame_index,
            "filename": path.name,
            "frame_path": rel_path(path),
            "url": browser_rel_path(path, html_output),
        })
    if not frames:
        raise FileNotFoundError("No frame_*_tracking_overlay.png files found in {}".format(frames_dir))
    return frames


def load_existing_attack_points():
    single_config = project_path("configs/nll_test4_homography_points.json")
    if single_config.exists():
        payload = load_json(single_config)
        return payload.get("points", [])
    return []


def split_points(points):
    video_points = []
    template_points = []
    point_names = []
    for index, point in enumerate(points or []):
        if point.get("video_xy") is None or point.get("field_xy") is None:
            continue
        video_points.append(point["video_xy"])
        template_points.append(point["field_xy"])
        point_names.append(point.get("name") or "point_{:02d}".format(index + 1))
    return video_points, template_points, point_names


def normalize_keyframe(raw):
    frame_index = int(raw.get("frame_index", 0))
    name = raw.get("name") or "frame_{:03d}".format(frame_index)
    if name in {"attack_zone", "mid_zone"}:
        name = "frame_{:03d}".format(frame_index)
    frame_path = raw.get("frame_path", "")
    video_points = raw.get("video_points")
    template_points = raw.get("template_points") or raw.get("field_points")
    point_names = raw.get("point_names")
    if video_points is None or template_points is None:
        video_points, template_points, point_names_from_points = split_points(raw.get("points", []))
        point_names = point_names or point_names_from_points
    point_names = point_names or ["point_{:02d}".format(index + 1) for index in range(min(len(video_points), len(template_points)))]
    return {
        "name": name,
        "frame_index": frame_index,
        "frame_path": frame_path,
        "video_points": video_points or [],
        "template_points": template_points or [],
        "point_names": point_names,
    }


def frame_path_for_index(frames, frame_index):
    for frame in frames:
        if int(frame["frame_index"]) == int(frame_index):
            return frame["frame_path"]
    return None


def default_config(run_id, field_template, frames, add_default_mid_frame):
    attack_points = load_existing_attack_points()
    video_points, template_points, point_names = split_points(attack_points)
    attack_frame = frame_path_for_index(frames, 4) or "outputs/nll_test4/segment_tests/segment_20s_3s/frames/frame_004_tracking_overlay.png"
    keyframes = [{
        "name": "frame_004",
        "frame_index": 4,
        "frame_path": attack_frame,
        "video_points": video_points,
        "template_points": template_points,
        "point_names": point_names,
    }]
    mid_path = frame_path_for_index(frames, add_default_mid_frame)
    if mid_path:
        keyframes.append({
            "name": "frame_{:03d}".format(add_default_mid_frame),
            "frame_index": int(add_default_mid_frame),
            "frame_path": mid_path,
            "video_points": [],
            "template_points": [],
            "point_names": [],
        })
    return {
        "_comment": "Multi-keyframe homography config. Add only calibration keyframes where the broadcast camera view changes meaningfully. Use stable visible white field markings only. Nearest-keyframe assignment is used for this smoke stage; interpolation comes later.",
        "run_id": run_id,
        "field_template": field_template,
        "assignment_mode": "nearest_keyframe",
        "keyframes": keyframes,
    }


def normalize_config(args, frames):
    config_path = project_path(args.config)
    if config_path.exists():
        raw = load_json(config_path)
    else:
        raw = default_config(args.run_id, args.field_template, frames, args.add_default_mid_frame)
    config = {
        "_comment": raw.get("_comment", "Multi-keyframe homography config. Add only keyframes where the camera view changes meaningfully."),
        "run_id": raw.get("run_id") or args.run_id,
        "field_template": raw.get("field_template") or args.field_template,
        "assignment_mode": "nearest_keyframe",
        "keyframes": [normalize_keyframe(row) for row in raw.get("keyframes", [])],
    }
    if not config["keyframes"]:
        config = default_config(args.run_id, args.field_template, frames, args.add_default_mid_frame)
    existing_frames = {int(row["frame_index"]) for row in config["keyframes"]}
    if args.add_default_mid_frame is not None and int(args.add_default_mid_frame) not in existing_frames:
        mid_path = frame_path_for_index(frames, args.add_default_mid_frame)
        if mid_path:
            config["keyframes"].append({
                "name": "frame_{:03d}".format(args.add_default_mid_frame),
                "frame_index": int(args.add_default_mid_frame),
                "frame_path": mid_path,
                "video_points": [],
                "template_points": [],
                "point_names": [],
            })
    for row in config["keyframes"]:
        current_frame_path = frame_path_for_index(frames, int(row["frame_index"]))
        if current_frame_path:
            row["frame_path"] = current_frame_path
    config["keyframes"].sort(key=lambda row: int(row["frame_index"]))
    return config_path, config


def save_debug_image(image_path, keyframe, key, output_path):
    points = []
    for index, (video_xy, template_xy) in enumerate(zip(keyframe.get("video_points", []), keyframe.get("template_points", []))):
        points.append({
            "name": keyframe.get("point_names", [])[index] if index < len(keyframe.get("point_names", [])) else "point_{:02d}".format(index + 1),
            "video_xy": video_xy,
            "field_xy": template_xy,
        })
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, point in enumerate(points):
        x, y = point[key]
        color = color_for(index)
        x = int(round(float(x)))
        y = int(round(float(y)))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255), width=2)
        draw.text((x + 9, y - 7), point["name"], fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def write_multi_frame_html(config, frames, field_template, html_output, names, config_path):
    field_uri, field_size = image_data_uri(field_template)
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Multi-Frame Homography Calibration Picker</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 18px; background: #f7f7f4; color: #1c1c1c; }
    .topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-bottom: 12px; }
    .wrap { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; align-items: start; }
    .pane { background: white; border: 1px solid #ccc; padding: 10px; }
    .imgbox { position: relative; width: 100%; max-height: 78vh; overflow: auto; border: 1px solid #ddd; cursor: crosshair; background: #222; }
    img { display: block; max-width: none; cursor: crosshair; user-select: none; }
    canvas { position: absolute; left: 0; top: 0; pointer-events: none; z-index: 2; }
    button, select { padding: 6px 10px; }
    input[type=range] { width: min(680px, 45vw); }
    textarea { width: 100%; min-height: 300px; font-family: ui-monospace, monospace; }
    code { background: #eee; padding: 1px 4px; }
    #status { font-weight: 700; }
    #pointList, #keyframeList { background: #fff; border: 1px solid #ccc; padding: 10px 14px; min-height: 42px; }
    #pointList li, #keyframeList li { margin: 4px 0; font-family: ui-monospace, monospace; }
    .muted { color: #555; }
  </style>
</head>
<body>
  <h1>Multi-Frame Homography Calibration Picker</h1>
  <p>Scroll the tracking frames, add keyframes only when the camera view changes, then click video/template point pairs for the selected keyframe. Use visible white field markings, not players, refs, logos, labels, or shadows. Copy the generated full config into <code>__CONFIG_PATH__</code>.</p>
  <div class=\"topbar\">
    <button id=\"prevFrame\">Previous frame</button>
    <input id=\"frameSlider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\">
    <button id=\"nextFrame\">Next frame</button>
    <span id=\"frameLabel\"></span>
    <button id=\"addKeyframe\">Add calibration keyframe from current frame</button>
  </div>
  <div class=\"topbar\">
    <label>Selected keyframe <select id=\"keyframeSelect\"></select></label>
    <button id=\"useCurrentFrameForKeyframe\">Move selected keyframe to current frame</button>
    <button id=\"undoPoint\">Undo point</button>
    <button id=\"deletePoint\">Delete selected point</button>
    <button id=\"deleteKeyframe\">Delete selected keyframe</button>
    <button id=\"copy\">Copy JSON</button>
    <span id=\"status\">Click video point</span>
  </div>
  <div class=\"wrap\">
    <div class=\"pane\">
      <h2>Broadcast/tracking frame</h2>
      <div id=\"videoBox\" class=\"imgbox\"><img id=\"videoImg\" src=\"\"><canvas id=\"videoCanvas\"></canvas></div>
    </div>
    <div class=\"pane\">
      <h2>Fixed NLL field template</h2>
      <div id=\"fieldBox\" class=\"imgbox\"><img id=\"fieldImg\" src=\"__FIELD_URI__\" width=\"__FIELD_W__\" height=\"__FIELD_H__\"><canvas id=\"fieldCanvas\" width=\"__FIELD_W__\" height=\"__FIELD_H__\"></canvas></div>
    </div>
  </div>
  <h2>Calibration keyframes</h2>
  <ol id=\"keyframeList\"></ol>
  <h2>Point pairs for selected keyframe</h2>
  <ol id=\"pointList\"></ol>
  <h2>Generated full config JSON</h2>
  <textarea id=\"jsonOut\"></textarea>
  <script>
    const frames = __FRAMES_JSON__;
    const defaultNames = __NAMES_JSON__;
    let config = __CONFIG_JSON__;
    const fieldTemplate = __FIELD_TEMPLATE_JSON__;
    const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
    let frameCursor = 0;
    let selectedKeyframe = 0;
    let pendingVideo = null;
    let selectedPoint = null;
    const videoImg = document.getElementById('videoImg');
    const videoCanvas = document.getElementById('videoCanvas');
    const fieldCanvas = document.getElementById('fieldCanvas');
    const frameSlider = document.getElementById('frameSlider');
    const frameLabel = document.getElementById('frameLabel');
    const keyframeSelect = document.getElementById('keyframeSelect');
    const statusEl = document.getElementById('status');
    const jsonOut = document.getElementById('jsonOut');
    const pointList = document.getElementById('pointList');
    const keyframeList = document.getElementById('keyframeList');
    frameSlider.max = String(Math.max(0, frames.length - 1));
    function setStatus(text) { statusEl.textContent = text; }
    function currentFrame() { return frames[frameCursor]; }
    function currentKeyframe() { return config.keyframes[selectedKeyframe] || null; }
    function pointCount(kf) { return Math.min((kf.video_points || []).length, (kf.template_points || []).length); }
    function keyframeNameForFrame(frameIndex) { return 'frame_' + String(frameIndex).padStart(3, '0'); }
    function findFrameCursor(frameIndex) {
      for (let i = 0; i < frames.length; i++) if (Number(frames[i].frame_index) === Number(frameIndex)) return i;
      return Math.max(0, Math.min(frames.length - 1, Number(frameIndex) || 0));
    }
    function ensureKeyframeShape(kf) {
      if (!kf.video_points) kf.video_points = [];
      if (!kf.template_points) kf.template_points = [];
      if (!kf.point_names) kf.point_names = [];
      return kf;
    }
    function localXY(event, img) {
      const rect = img.getBoundingClientRect();
      const sx = img.naturalWidth / rect.width;
      const sy = img.naturalHeight / rect.height;
      const x = (event.clientX - rect.left) * sx;
      const y = (event.clientY - rect.top) * sy;
      if (x < 0 || y < 0 || x > img.naturalWidth || y > img.naturalHeight) return null;
      return [Number(x.toFixed(2)), Number(y.toFixed(2))];
    }
    function resizeVideoCanvas() {
      videoCanvas.width = videoImg.naturalWidth || 1;
      videoCanvas.height = videoImg.naturalHeight || 1;
      videoCanvas.style.width = videoImg.width + 'px';
      videoCanvas.style.height = videoImg.height + 'px';
    }
    function drawCanvas(canvas, points, key) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      points.forEach(function(p, i) {
        const xy = p[key];
        if (!xy) return;
        ctx.fillStyle = colors[i % colors.length];
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(xy[0], xy[1], 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        if (selectedPoint === i) { ctx.strokeStyle = 'black'; ctx.lineWidth = 3; ctx.strokeRect(xy[0] - 14, xy[1] - 14, 28, 28); }
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.font = 'bold 14px sans-serif';
        const n = String(i + 1);
        ctx.strokeText(n, xy[0] - 4, xy[1] + 5);
        ctx.fillText(n, xy[0] - 4, xy[1] + 5);
        ctx.fillStyle = colors[i % colors.length];
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 3;
        ctx.strokeText(p.name, xy[0] + 13, xy[1] - 10);
        ctx.fillText(p.name, xy[0] + 13, xy[1] - 10);
      });
    }
    function selectedPointObjects() {
      const kf = currentKeyframe();
      if (!kf) return [];
      ensureKeyframeShape(kf);
      const n = pointCount(kf);
      const rows = [];
      for (let i = 0; i < n; i++) {
        rows.push({name: kf.point_names[i] || defaultNames[i] || ('point_' + String(i + 1).padStart(2, '0')), video_xy: kf.video_points[i], template_xy: kf.template_points[i]});
      }
      return rows;
    }
    function renderKeyframeControls() {
      keyframeSelect.innerHTML = '';
      config.keyframes.forEach(function(kf, i) {
        ensureKeyframeShape(kf);
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = kf.name + ' | frame ' + String(kf.frame_index).padStart(3, '0') + ' | points ' + pointCount(kf);
        keyframeSelect.appendChild(opt);
      });
      if (selectedKeyframe >= config.keyframes.length) selectedKeyframe = Math.max(0, config.keyframes.length - 1);
      keyframeSelect.value = String(selectedKeyframe);
      keyframeList.innerHTML = '';
      config.keyframes.forEach(function(kf, i) {
        const li = document.createElement('li');
        li.textContent = (i === selectedKeyframe ? '* ' : '') + kf.name + ': frame ' + kf.frame_index + ', pairs ' + pointCount(kf) + ', path ' + kf.frame_path;
        keyframeList.appendChild(li);
      });
    }
    function renderPointList() {
      pointList.innerHTML = '';
      const rows = selectedPointObjects();
      if (!currentKeyframe()) {
        const li = document.createElement('li'); li.textContent = 'No keyframe selected.'; pointList.appendChild(li); return;
      }
      if (rows.length === 0) {
        const li = document.createElement('li'); li.textContent = 'No point pairs for this keyframe yet.'; pointList.appendChild(li); return;
      }
      rows.forEach(function(p, i) {
        const li = document.createElement('li');
        li.textContent = String(i + 1) + '. ' + p.name + ': video [' + p.video_xy[0] + ', ' + p.video_xy[1] + '] -> field [' + p.template_xy[0] + ', ' + p.template_xy[1] + ']';
        li.onclick = function() { selectedPoint = i; redraw(); };
        if (selectedPoint === i) li.style.fontWeight = '700';
        pointList.appendChild(li);
      });
    }
    function fullExportConfig() {
      const cfg = JSON.parse(JSON.stringify(config));
      cfg.field_template = fieldTemplate;
      cfg.assignment_mode = 'nearest_keyframe';
      cfg.keyframes = cfg.keyframes.map(function(kf) {
        ensureKeyframeShape(kf);
        return {name: kf.name, frame_index: Number(kf.frame_index), frame_path: kf.frame_path, video_points: kf.video_points, template_points: kf.template_points, point_names: kf.point_names};
      }).sort(function(a, b) { return Number(a.frame_index) - Number(b.frame_index); });
      return cfg;
    }
    function redraw() {
      const frame = currentFrame();
      if (videoImg.getAttribute('src') !== frame.url) videoImg.src = frame.url;
      frameSlider.value = String(frameCursor);
      frameLabel.textContent = 'frame ' + String(frame.frame_index).padStart(3, '0') + ' | ' + frame.filename;
      const rows = selectedPointObjects();
      drawCanvas(videoCanvas, rows, 'video_xy');
      drawCanvas(fieldCanvas, rows, 'template_xy');
      if (pendingVideo) {
        const ctx = videoCanvas.getContext('2d');
        ctx.strokeStyle = 'black'; ctx.fillStyle = 'white'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(pendingVideo[0], pendingVideo[1], 10, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.font = 'bold 14px sans-serif'; ctx.fillStyle = 'black'; ctx.fillText('pending', pendingVideo[0] + 14, pendingVideo[1] - 10);
      }
      renderKeyframeControls();
      renderPointList();
      jsonOut.value = JSON.stringify(fullExportConfig(), null, 2) + '\\n';
      setStatus(pendingVideo ? 'Click matching template point' : 'Click video point for selected keyframe');
    }
    function gotoFrameCursor(cursor) {
      frameCursor = Math.max(0, Math.min(frames.length - 1, Number(cursor)));
      pendingVideo = null;
      selectedPoint = null;
      redraw();
    }
    function addKeyframeFromCurrentFrame() {
      const frame = currentFrame();
      let existing = -1;
      config.keyframes.forEach(function(kf, i) { if (Number(kf.frame_index) === Number(frame.frame_index)) existing = i; });
      if (existing >= 0) { selectedKeyframe = existing; redraw(); return; }
      config.keyframes.push({name: keyframeNameForFrame(frame.frame_index), frame_index: Number(frame.frame_index), frame_path: frame.frame_path, video_points: [], template_points: [], point_names: []});
      config.keyframes.sort(function(a, b) { return Number(a.frame_index) - Number(b.frame_index); });
      selectedKeyframe = config.keyframes.findIndex(function(kf) { return Number(kf.frame_index) === Number(frame.frame_index); });
      redraw();
    }
    function moveSelectedKeyframeToCurrentFrame() {
      const kf = currentKeyframe();
      if (!kf) return;
      const frame = currentFrame();
      kf.name = keyframeNameForFrame(frame.frame_index);
      kf.frame_index = Number(frame.frame_index);
      kf.frame_path = frame.frame_path;
      config.keyframes.sort(function(a, b) { return Number(a.frame_index) - Number(b.frame_index); });
      selectedKeyframe = config.keyframes.findIndex(function(row) { return Number(row.frame_index) === Number(frame.frame_index); });
      redraw();
    }
    function handleVideoClick(event) {
      const kf = currentKeyframe();
      if (!kf) { setStatus('Add or select a keyframe first'); return; }
      const xy = localXY(event, videoImg);
      if (!xy) return;
      pendingVideo = xy;
      redraw();
    }
    function handleFieldClick(event) {
      const kf = currentKeyframe();
      if (!kf) { setStatus('Add or select a keyframe first'); return; }
      if (!pendingVideo) { setStatus('Click video point first'); return; }
      const fieldXY = localXY(event, document.getElementById('fieldImg'));
      if (!fieldXY) return;
      ensureKeyframeShape(kf);
      const idx = pointCount(kf);
      kf.video_points.push(pendingVideo);
      kf.template_points.push(fieldXY);
      kf.point_names.push(defaultNames[idx] || ('point_' + String(idx + 1).padStart(2, '0')));
      pendingVideo = null;
      selectedPoint = idx;
      redraw();
    }
    videoImg.onload = function() { resizeVideoCanvas(); redraw(); };
    document.getElementById('videoBox').addEventListener('click', handleVideoClick);
    document.getElementById('fieldBox').addEventListener('click', handleFieldClick);
    frameSlider.oninput = function() { gotoFrameCursor(frameSlider.value); };
    document.getElementById('prevFrame').onclick = function() { gotoFrameCursor(frameCursor - 1); };
    document.getElementById('nextFrame').onclick = function() { gotoFrameCursor(frameCursor + 1); };
    document.getElementById('addKeyframe').onclick = addKeyframeFromCurrentFrame;
    document.getElementById('useCurrentFrameForKeyframe').onclick = moveSelectedKeyframeToCurrentFrame;
    keyframeSelect.onchange = function() { selectedKeyframe = Number(keyframeSelect.value); const kf = currentKeyframe(); frameCursor = findFrameCursor(kf.frame_index); pendingVideo = null; selectedPoint = null; redraw(); };
    document.getElementById('undoPoint').onclick = function() { const kf = currentKeyframe(); if (!kf) return; if (pendingVideo) { pendingVideo = null; } else { ensureKeyframeShape(kf); kf.video_points.pop(); kf.template_points.pop(); kf.point_names.pop(); selectedPoint = null; } redraw(); };
    document.getElementById('deletePoint').onclick = function() { const kf = currentKeyframe(); if (!kf || selectedPoint === null) return; ensureKeyframeShape(kf); kf.video_points.splice(selectedPoint, 1); kf.template_points.splice(selectedPoint, 1); kf.point_names.splice(selectedPoint, 1); selectedPoint = null; redraw(); };
    document.getElementById('deleteKeyframe').onclick = function() { if (config.keyframes.length <= 1) { setStatus('Keep at least one keyframe'); return; } config.keyframes.splice(selectedKeyframe, 1); selectedKeyframe = Math.max(0, selectedKeyframe - 1); pendingVideo = null; selectedPoint = null; redraw(); };
    document.getElementById('copy').onclick = async function() { try { await navigator.clipboard.writeText(jsonOut.value); setStatus('Copied JSON to clipboard.'); } catch (err) { jsonOut.focus(); jsonOut.select(); setStatus('Clipboard blocked; JSON selected for manual copy.'); } };
    if (config.keyframes.length > 0) frameCursor = findFrameCursor(config.keyframes[0].frame_index);
    redraw();
  </script>
</body>
</html>
"""
    replacements = {
        "__CONFIG_PATH__": rel_path(config_path),
        "__FIELD_URI__": field_uri,
        "__FIELD_W__": str(field_size[0]),
        "__FIELD_H__": str(field_size[1]),
        "__FRAMES_JSON__": json.dumps(frames),
        "__NAMES_JSON__": json.dumps(names),
        "__CONFIG_JSON__": json.dumps(config),
        "__FIELD_TEMPLATE_JSON__": json.dumps(rel_path(field_template)),
    }
    for token, value in replacements.items():
        html = html.replace(token, value)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(html, encoding="utf-8")
    return str(html_output)


def main():
    args = parse_args()
    html_output = project_path(args.html_output)
    frames_dir = project_path(args.tracking_frames_dir)
    field_template = project_path(args.field_template)
    names = [item.strip() for item in args.point_names.split(",") if item.strip()]
    if not frames_dir.exists():
        raise SystemExit("Missing tracking frames directory: {}".format(frames_dir))
    if not field_template.exists():
        raise SystemExit("Missing field template: {}".format(field_template))
    frames = load_tracking_frames(frames_dir, html_output)
    config_path, config = normalize_config(args, frames)
    write_json(config_path, config)
    picker = write_multi_frame_html(config, frames, field_template, html_output, names, config_path)
    debug_dir = html_output.parent / "point_picker_debug"
    for keyframe in config.get("keyframes", []):
        if not keyframe.get("video_points"):
            continue
        frame_path = project_path(keyframe["frame_path"])
        if frame_path.exists():
            save_debug_image(frame_path, keyframe, "video_xy", debug_dir / "{}_video_debug.png".format(keyframe["name"]))
            save_debug_image(field_template, keyframe, "field_xy", debug_dir / "{}_template_debug.png".format(keyframe["name"]))
    payload = {
        "status": "html_written",
        "config": str(config_path),
        "html": picker,
        "tracking_frames_dir": str(frames_dir),
        "frame_count": len(frames),
        "keyframes": [{"name": row["name"], "frame_index": row["frame_index"], "point_pairs": min(len(row.get("video_points", [])), len(row.get("template_points", [])))} for row in config.get("keyframes", [])],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
