#!/usr/bin/env python3
"""Pick matching video/template points for the field homography config."""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_VIDEO_FRAME = "outputs/nll-test1/short_clip_tracking_stabilized/frames/frame_005_tracking_overlay.png"
DEFAULT_FIELD_TEMPLATE = "assets/field_templates/nll_field_topdown.png"
DEFAULT_OUTPUT_CONFIG = "configs/nll_field_homography_points.json"
DEFAULT_HTML = "outputs/nll-test1/field_calibration_smoke/point_picker.html"
DEFAULT_POINT_NAMES = [
    "center_faceoff_dot",
    "center_circle_left_edge",
    "center_circle_right_edge",
    "white_line_intersection_1",
    "white_line_intersection_2",
    "restraining_line_intersection_1",
    "restraining_line_intersection_2",
    "boards_or_boundary_intersection",
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
    parser = argparse.ArgumentParser(description="Pick matching points for configs/nll_field_homography_points.json.")
    parser.add_argument("--config", default=None, help="Existing homography config JSON to load video_frame, field_template, point names, and output path from.")
    parser.add_argument("--video-frame", default=DEFAULT_VIDEO_FRAME, help="Broadcast/video frame image to click.")
    parser.add_argument("--field-template", default=DEFAULT_FIELD_TEMPLATE, help="Top-down field template image to click.")
    parser.add_argument("--output-config", default=DEFAULT_OUTPUT_CONFIG, help="Output homography config JSON.")
    parser.add_argument("--video-debug", default="configs/nll_field_homography_points_video_debug.png", help="Diagnostic video-frame points image.")
    parser.add_argument("--template-debug", default="configs/nll_field_homography_points_template_debug.png", help="Diagnostic template points image.")
    parser.add_argument("--html-output", default=DEFAULT_HTML, help="Fallback HTML point picker path.")
    parser.add_argument("--mode", choices=["auto", "gui", "html"], default="auto", help="Use matplotlib GUI or write HTML fallback.")
    parser.add_argument("--min-points", type=int, default=4, help="Minimum point pairs before GUI save is allowed.")
    parser.add_argument("--point-names", default=",".join(DEFAULT_POINT_NAMES), help="Comma-separated default names for picked points.")
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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def color_for(index):
    return COLORS[index % len(COLORS)]


def make_config(video_frame, field_template, points):
    return {
        "_comment": "Manual homography correspondences. Use stable white field lines/painted markings, not the center logo. At least 4 non-collinear points are required; 6 to 8 spread across the visible field is preferred.",
        "video_frame": rel_path(video_frame),
        "field_template": rel_path(field_template),
        "points": points,
    }


def save_debug_image(image_path, points, key, output_path):
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


def save_outputs(video_frame, field_template, output_config, video_debug, template_debug, points):
    write_json(output_config, make_config(video_frame, field_template, points))
    save_debug_image(video_frame, points, "video_xy", video_debug)
    save_debug_image(field_template, points, "field_xy", template_debug)


def point_name(names, index):
    if index < len(names) and names[index]:
        return names[index]
    return "point_{:02d}".format(index + 1)


def run_gui(video_frame, field_template, output_config, video_debug, template_debug, names, min_points):
    import matplotlib.pyplot as plt

    video_img = Image.open(video_frame).convert("RGB")
    field_img = Image.open(field_template).convert("RGB")
    points = []
    pending_video = {"xy": None}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(video_img)
    axes[0].set_title("1. Click video frame landmark")
    axes[1].imshow(field_img)
    axes[1].set_title("2. Click matching field-template landmark")
    for ax in axes:
        ax.set_axis_off()
    status = fig.text(0.02, 0.02, "Click video point, then matching template point. Keys: s=save, u=undo, q=quit", fontsize=10)
    markers = []

    def redraw_status(message):
        status.set_text(message)
        fig.canvas.draw_idle()

    def draw_marker(ax, xy, label, color):
        marker = ax.scatter([xy[0]], [xy[1]], s=52, c=[tuple(c / 255.0 for c in color)], edgecolors="white", linewidths=1.2)
        text = ax.text(xy[0] + 8, xy[1] - 8, label, color=tuple(c / 255.0 for c in color), fontsize=9, weight="bold")
        markers.extend([marker, text])

    def clear_markers():
        while markers:
            artist = markers.pop()
            artist.remove()
        for index, pair in enumerate(points):
            color = color_for(index)
            draw_marker(axes[0], pair["video_xy"], pair["name"], color)
            draw_marker(axes[1], pair["field_xy"], pair["name"], color)
        if pending_video["xy"] is not None:
            draw_marker(axes[0], pending_video["xy"], "pending", (255, 255, 255))
        fig.canvas.draw_idle()

    def save_if_valid():
        if len(points) < min_points:
            redraw_status("Need at least {} point pairs before saving; currently {}.".format(min_points, len(points)))
            return
        save_outputs(video_frame, field_template, output_config, video_debug, template_debug, points)
        redraw_status("Saved {} point pairs to {}".format(len(points), output_config))

    def on_click(event):
        if event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return
        xy = [round(float(event.xdata), 2), round(float(event.ydata), 2)]
        if event.inaxes == axes[0]:
            pending_video["xy"] = xy
            clear_markers()
            redraw_status("Video point selected at {}. Now click the matching field-template point.".format(xy))
            return
        if pending_video["xy"] is None:
            redraw_status("Click the video-frame landmark first, then the matching template landmark.")
            return
        index = len(points)
        pair = {"name": point_name(names, index), "video_xy": pending_video["xy"], "field_xy": xy}
        points.append(pair)
        pending_video["xy"] = None
        clear_markers()
        redraw_status("Added {}. Total pairs: {}. Press s to save when ready.".format(pair["name"], len(points)))

    def on_key(event):
        if event.key in {"s", "enter", "return"}:
            save_if_valid()
        elif event.key in {"u", "backspace", "delete"}:
            if pending_video["xy"] is not None:
                pending_video["xy"] = None
            elif points:
                points.pop()
            clear_markers()
            redraw_status("Undo complete. Total pairs: {}.".format(len(points)))
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return points


def image_data_uri(path):
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), image.size


def write_html_picker(video_frame, field_template, output_config, html_output, names):
    video_uri, video_size = image_data_uri(video_frame)
    field_uri, field_size = image_data_uri(field_template)
    names_json = json.dumps(names)
    template_config = make_config(video_frame, field_template, [])
    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>NLL Field Homography Point Picker</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 18px; background: #f7f7f4; color: #1c1c1c; }
    .wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
    .pane { background: white; border: 1px solid #ccc; padding: 10px; }
    .imgbox { position: relative; width: 100%; overflow: auto; border: 1px solid #ddd; cursor: crosshair; }
    img { display: block; max-width: none; cursor: crosshair; user-select: none; }
    canvas { position: absolute; left: 0; top: 0; pointer-events: none; z-index: 2; }
    button { margin: 8px 8px 8px 0; padding: 6px 10px; }
    textarea { width: 100%; min-height: 240px; font-family: ui-monospace, monospace; }
    code { background: #eee; padding: 1px 4px; }
    #status { display: inline-block; margin-left: 10px; font-weight: 700; }
    #pointList { background: #fff; border: 1px solid #ccc; padding: 10px 14px; min-height: 42px; }
    #pointList li { margin: 4px 0; font-family: ui-monospace, monospace; }
  </style>
</head>
<body>
  <h1>NLL Field Homography Point Picker</h1>
  <p>Click one landmark on the video frame, then the matching landmark on the top-down template. Use stable white field lines and painted markings, not the logo. Copy the JSON into <code>__OUTPUT_CONFIG__</code>.</p>
  <div><button id=\"undo\">Undo</button><button id=\"clear\">Clear</button><button id=\"copy\">Copy JSON</button><span id=\"status\">Click video point</span></div>
  <div class=\"wrap\">
    <div class=\"pane\"><h2>Video frame</h2><div id=\"videoBox\" class=\"imgbox\"><img id=\"videoImg\" src=\"__VIDEO_URI__\" width=\"__VIDEO_W__\" height=\"__VIDEO_H__\"><canvas id=\"videoCanvas\" width=\"__VIDEO_W__\" height=\"__VIDEO_H__\"></canvas></div></div>
    <div class=\"pane\"><h2>Field template</h2><div id=\"fieldBox\" class=\"imgbox\"><img id=\"fieldImg\" src=\"__FIELD_URI__\" width=\"__FIELD_W__\" height=\"__FIELD_H__\"><canvas id=\"fieldCanvas\" width=\"__FIELD_W__\" height=\"__FIELD_H__\"></canvas></div></div>
  </div>
  <h2>Point pairs</h2>
  <ol id=\"pointList\"></ol>
  <h2>Generated config JSON</h2>
  <textarea id=\"jsonOut\"></textarea>
  <script>
    const names = __NAMES_JSON__;
    const baseConfig = __BASE_CONFIG__;
    const points = [];
    let pendingVideo = null;
    const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
    const statusEl = document.getElementById('status');
    const jsonOut = document.getElementById('jsonOut');
    const pointList = document.getElementById('pointList');
    function setStatus(text) { statusEl.textContent = text; }
    function localXY(event, img) {
      const rect = img.getBoundingClientRect();
      const sx = img.naturalWidth / rect.width;
      const sy = img.naturalHeight / rect.height;
      const x = (event.clientX - rect.left) * sx;
      const y = (event.clientY - rect.top) * sy;
      if (x < 0 || y < 0 || x > img.naturalWidth || y > img.naturalHeight) return null;
      return [Number(x.toFixed(2)), Number(y.toFixed(2))];
    }
    function drawCanvas(canvas, entries, key) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      entries.forEach((p, i) => {
        const xy = p[key];
        ctx.fillStyle = colors[i % colors.length];
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(xy[0], xy[1], 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
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
    function renderPointList() {
      pointList.innerHTML = '';
      if (points.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No point pairs yet.';
        pointList.appendChild(li);
        return;
      }
      points.forEach(function(p, i) {
        const li = document.createElement('li');
        li.textContent = String(i + 1) + '. ' + p.name + ': video [' + p.video_xy[0] + ', ' + p.video_xy[1] + '] -> field [' + p.field_xy[0] + ', ' + p.field_xy[1] + ']';
        pointList.appendChild(li);
      });
    }
    function redraw() {
      drawCanvas(document.getElementById('videoCanvas'), points, 'video_xy');
      drawCanvas(document.getElementById('fieldCanvas'), points, 'field_xy');
      if (pendingVideo) {
        const ctx = document.getElementById('videoCanvas').getContext('2d');
        ctx.strokeStyle = 'black'; ctx.fillStyle = 'white'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(pendingVideo[0], pendingVideo[1], 10, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.font = 'bold 14px sans-serif';
        ctx.fillStyle = 'black'; ctx.fillText('pending', pendingVideo[0] + 14, pendingVideo[1] - 10);
      }
      const cfg = Object.assign({}, baseConfig, {points: points});
      jsonOut.value = JSON.stringify(cfg, null, 2) + "\\n";
      renderPointList();
      setStatus(pendingVideo ? 'Click matching template point' : 'Click video point');
    }
    function handleVideoClick(event) {
      const xy = localXY(event, document.getElementById('videoImg'));
      if (!xy) return;
      pendingVideo = xy;
      redraw();
    }
    function handleFieldClick(event) {
      if (!pendingVideo) { setStatus('Click video point'); return; }
      const fieldXY = localXY(event, document.getElementById('fieldImg'));
      if (!fieldXY) return;
      const name = names[points.length] || ('point_' + String(points.length + 1).padStart(2, '0'));
      points.push({name, video_xy: pendingVideo, field_xy: fieldXY});
      pendingVideo = null;
      redraw();
    }
    document.getElementById('videoBox').addEventListener('click', handleVideoClick);
    document.getElementById('fieldBox').addEventListener('click', handleFieldClick);
    document.getElementById('undo').onclick = () => { if (pendingVideo) pendingVideo = null; else points.pop(); redraw(); };
    document.getElementById('clear').onclick = () => { pendingVideo = null; points.splice(0, points.length); redraw(); };
    document.getElementById('copy').onclick = async () => {
      try {
        await navigator.clipboard.writeText(jsonOut.value);
        setStatus('Copied JSON to clipboard.');
      } catch (err) {
        jsonOut.focus(); jsonOut.select();
        setStatus('Clipboard blocked; JSON selected for manual copy.');
      }
    };
    redraw();
  </script>
</body>
</html>
"""
    html = html.replace("__OUTPUT_CONFIG__", rel_path(output_config))
    html = html.replace("__VIDEO_URI__", video_uri)
    html = html.replace("__FIELD_URI__", field_uri)
    html = html.replace("__VIDEO_W__", str(video_size[0]))
    html = html.replace("__VIDEO_H__", str(video_size[1]))
    html = html.replace("__FIELD_W__", str(field_size[0]))
    html = html.replace("__FIELD_H__", str(field_size[1]))
    html = html.replace("__NAMES_JSON__", names_json)
    html = html.replace("__BASE_CONFIG__", json.dumps(template_config))
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(html, encoding="utf-8")
    return str(html_output)


def main():
    args = parse_args()
    config_path = project_path(args.config) if args.config else None
    config_payload = None
    if config_path:
        config_payload = load_json(config_path)
        if args.video_frame == DEFAULT_VIDEO_FRAME and config_payload.get("video_frame"):
            args.video_frame = config_payload["video_frame"]
        if args.field_template == DEFAULT_FIELD_TEMPLATE and config_payload.get("field_template"):
            args.field_template = config_payload["field_template"]
        if args.output_config == DEFAULT_OUTPUT_CONFIG:
            args.output_config = str(config_path)
        if args.html_output == DEFAULT_HTML:
            stem = config_path.stem
            if "nll_test4" in stem:
                args.html_output = "outputs/nll_test4/field_calibration_smoke/point_picker.html"
            elif "nll_field" in stem:
                args.html_output = DEFAULT_HTML
        if args.video_debug == "configs/nll_field_homography_points_video_debug.png" and "nll_test4" in config_path.stem:
            args.video_debug = "configs/nll_test4_homography_points_video_debug.png"
        if args.template_debug == "configs/nll_field_homography_points_template_debug.png" and "nll_test4" in config_path.stem:
            args.template_debug = "configs/nll_test4_homography_points_template_debug.png"
        config_names = [point.get("name") for point in config_payload.get("points", []) if point.get("name")]
        if args.point_names == ",".join(DEFAULT_POINT_NAMES) and config_names:
            args.point_names = ",".join(config_names)
    video_frame = project_path(args.video_frame)
    field_template = project_path(args.field_template)
    output_config = project_path(args.output_config)
    video_debug = project_path(args.video_debug)
    template_debug = project_path(args.template_debug)
    html_output = project_path(args.html_output)
    names = [item.strip() for item in args.point_names.split(",") if item.strip()]
    if not video_frame.exists():
        raise SystemExit("Missing video frame: {}".format(video_frame))
    if not field_template.exists():
        raise SystemExit("Missing field template: {}".format(field_template))

    if args.mode == "html":
        print(json.dumps({"status": "html_written", "html": write_html_picker(video_frame, field_template, output_config, html_output, names)}, indent=2))
        return 0
    if args.mode in {"auto", "gui"}:
        try:
            run_gui(video_frame, field_template, output_config, video_debug, template_debug, names, args.min_points)
            return 0
        except Exception as exc:
            if args.mode == "gui":
                raise
            html = write_html_picker(video_frame, field_template, output_config, html_output, names)
            print(json.dumps({"status": "gui_failed_html_written", "error": "{}: {}".format(exc.__class__.__name__, exc), "html": html}, indent=2))
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
