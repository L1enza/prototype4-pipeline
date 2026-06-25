import json
import traceback
from pathlib import Path

from prototype4_pipeline.integrations.sam3_prompt_sweep import image_stats, slugify_prompt, tensor_to_list
from prototype4_pipeline.integrations.sam3_smoke import (
    cast_tensors,
    dtype_diagnostics,
    force_module_dtype,
    sampled_frames,
    set_image_with_dtype,
    set_text_prompt_with_dtype,
    smoke_autocast_policy,
    smoke_fused_kernel_policy,
    temporary_sys_path,
    torch_dtype,
    write_json,
)

DEFAULT_FILTER_PROMPT = "player on the field"
FILTER_MODES = {"green_foot_point", "field_polygon", "y_threshold", "combined"}


def tensor_masks_to_numpy(masks):
    import numpy as np

    if masks is None or not hasattr(masks, "detach"):
        return np.zeros((0, 1, 1), dtype=bool)
    masks_cpu = masks.detach().to("cpu")
    if len(masks_cpu.shape) == 4 and int(masks_cpu.shape[1]) == 1:
        masks_cpu = masks_cpu[:, 0]
    if len(masks_cpu.shape) == 2:
        masks_cpu = masks_cpu.unsqueeze(0)
    return masks_cpu.bool().numpy()


def bbox_from_mask(mask):
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return None
    return {
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()),
        "y1": int(ys.max()),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
    }


def centroid_from_mask(mask):
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return None
    return {"x": float(xs.mean()), "y": float(ys.mean())}


def foot_point_from_mask(mask):
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return None
    bottom_y = int(ys.max())
    bottom_xs = xs[ys >= max(0, bottom_y - 2)]
    if len(bottom_xs) == 0:
        bottom_xs = xs[ys == bottom_y]
    return {"x": float(bottom_xs.mean()), "y": float(bottom_y)}


def score_at(scores, index):
    values = tensor_to_list(scores)
    if index < len(values):
        try:
            return float(values[index])
        except Exception:
            return None
    return None


def point_in_polygon(point, polygon):
    if not polygon or point is None:
        return None
    x = point["x"]
    y = point["y"]
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def parse_polygon(raw):
    if not raw:
        return None
    points = []
    for pair in raw.split(";"):
        if not pair.strip():
            continue
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    return points or None


def parse_polygon_json(raw):
    if not raw:
        return None
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("points") or payload.get("polygon")
    if not isinstance(payload, list):
        raise ValueError("field polygon JSON must be a list of [x, y] points or an object with points/polygon")
    points = []
    for point in payload:
        if isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        else:
            points.append((float(point[0]), float(point[1])))
    return points or None


def coord_to_pixel(value, extent):
    if value is None:
        return None
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value * extent
    return value


def polygon_to_pixels(polygon, width, height):
    if not polygon:
        return None
    return [(coord_to_pixel(x, width), coord_to_pixel(y, height)) for x, y in polygon]


def green_surface_stats(image_array, point, radius=4, y_offset=4):
    if point is None:
        return {"sampled": False, "is_green": False}
    height, width = image_array.shape[:2]
    x = int(round(point["x"]))
    y = int(round(point["y"])) + int(y_offset)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return {"sampled": False, "is_green": False}
    patch = image_array[y0:y1, x0:x1].astype("float32")
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)
    r, g, b = [float(v) for v in mean_rgb]
    is_green = bool(g >= r * 0.82 and g >= b * 0.68 and g >= 35.0)
    return {
        "sampled": True,
        "sample_box": [int(x0), int(y0), int(x1), int(y1)],
        "mean_rgb": [r, g, b],
        "is_green": is_green,
    }


def mask_touches_bench(mask, bench_y_cutoff):
    ys, _ = mask.nonzero()
    if len(ys) == 0:
        return False
    return bool(ys.min() <= bench_y_cutoff)


def classify_mask(mask, index, scores, image_array, config, field_polygon_pixels=None):
    height, width = image_array.shape[:2]
    bbox = bbox_from_mask(mask)
    centroid = centroid_from_mask(mask)
    foot = foot_point_from_mask(mask)
    pixel_count = int(mask.sum())
    filter_mode = config["filter_mode"]
    bench_y_cutoff = coord_to_pixel(config["bench_y_cutoff"], height)
    field_y_min = coord_to_pixel(config["field_y_min"], height)
    field_y_max = coord_to_pixel(config["field_y_max"], height)
    touches_bench = mask_touches_bench(mask, bench_y_cutoff)
    green = green_surface_stats(image_array, foot, radius=int(config["green_sample_radius"]), y_offset=int(config["green_sample_y_offset"]))
    bottom_in_y_band = bool(foot is not None and field_y_min <= foot["y"] <= field_y_max)
    centroid_in_y_band = bool(centroid is not None and field_y_min <= centroid["y"] <= field_y_max)
    bottom_in_polygon = point_in_polygon(foot, field_polygon_pixels) if field_polygon_pixels else None
    centroid_in_polygon = point_in_polygon(centroid, field_polygon_pixels) if field_polygon_pixels else None
    centroid_in_bench_strip = bool(centroid is not None and centroid["y"] < bench_y_cutoff)
    bottom_above_cutoff = bool(foot is not None and foot["y"] < field_y_min)
    if field_polygon_pixels:
        bottom_in_field_geometry = bool(bottom_in_polygon)
        centroid_in_field_geometry = bool(centroid_in_polygon)
    else:
        bottom_in_field_geometry = bottom_in_y_band
        centroid_in_field_geometry = centroid_in_y_band

    reasons = []
    if bbox is None or foot is None or centroid is None:
        reasons.append("empty_mask")
    if pixel_count < int(config["min_mask_pixels"]):
        reasons.append("too_small")
    if centroid_in_bench_strip:
        reasons.append("centroid_in_bench_strip")
    if filter_mode in {"y_threshold", "combined"} and bottom_above_cutoff:
        reasons.append("bottom_point_above_field_cutoff")
    if filter_mode in {"field_polygon", "combined"}:
        if field_polygon_pixels and not bottom_in_polygon:
            reasons.append("bottom_point_outside_field_polygon")
        elif not field_polygon_pixels and not bottom_in_y_band:
            reasons.append("bottom_point_outside_field_y_band")
    if filter_mode == "green_foot_point" and not green.get("is_green"):
        reasons.append("foot_point_not_on_green_surface")
    kept = len(reasons) == 0
    return {
        "mask_index": int(index),
        "kept": kept,
        "rejection_reasons": reasons,
        "bbox": bbox,
        "centroid": centroid,
        "bottom_center_point": foot,
        "mask_area": pixel_count,
        "mask_pixel_count": pixel_count,
        "confidence_score": score_at(scores, index),
        "touches_bench_or_boards_region": touches_bench,
        "green_foot_check_passed": bool(green.get("is_green")),
        "centroid_in_field_polygon": centroid_in_polygon,
        "bottom_point_in_field_polygon": bottom_in_polygon,
        "centroid_in_field_geometry": centroid_in_field_geometry,
        "bottom_point_in_field_geometry": bottom_in_field_geometry,
        "centroid_in_bench_strip": centroid_in_bench_strip,
        "bottom_point_above_field_cutoff": bottom_above_cutoff,
        "field_region": {
            "filter_mode": filter_mode,
            "field_y_min": field_y_min,
            "field_y_max": field_y_max,
            "bench_y_cutoff": bench_y_cutoff,
            "bottom_point_in_y_band": bottom_in_y_band,
            "centroid_in_y_band": centroid_in_y_band,
            "bottom_point_in_polygon": bottom_in_polygon,
            "centroid_in_polygon": centroid_in_polygon,
            "bottom_point_on_green_surface": green,
        },
    }


def save_individual_mask(mask, image, output_dir, frame_index, mask_index, kept):
    import numpy as np
    from PIL import Image

    subdir = output_dir / ("kept_masks" if kept else "rejected_masks")
    subdir.mkdir(parents=True, exist_ok=True)
    mask_path = subdir / "frame_{:03d}_mask_{:03d}.png".format(frame_index, mask_index)
    overlay_path = subdir / "frame_{:03d}_mask_{:03d}_overlay.png".format(frame_index, mask_index)
    mask_u8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_u8, mode="L").save(mask_path)
    rgba = image.convert("RGBA")
    color = (20, 220, 80, 125) if kept else (255, 40, 40, 100)
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = mask.astype(np.uint8) * color[3]
    Image.alpha_composite(rgba, Image.fromarray(overlay, mode="RGBA")).convert("RGB").save(overlay_path)
    return str(mask_path), str(overlay_path)


def save_multi_overlay(image, masks, records, output_path, kept_only=False):
    import numpy as np
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    colors = [(20, 220, 80, 120), (0, 170, 255, 120), (255, 220, 40, 120), (255, 80, 200, 120), (140, 90, 255, 120)]
    for idx, record in enumerate(records):
        if kept_only and not record["kept"]:
            continue
        color = colors[idx % len(colors)] if record["kept"] else (255, 35, 35, 90)
        overlay[masks[idx].astype(bool)] = color
    composed = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")
    draw = ImageDraw.Draw(composed)
    for record in records:
        if kept_only and not record["kept"]:
            continue
        point = record.get("bottom_center_point")
        if not point:
            continue
        x = int(round(point["x"]))
        y = int(round(point["y"]))
        color = (0, 255, 0) if record["kept"] else (255, 0, 0)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x + 5, y - 8), str(record["mask_index"]), fill=color)
    composed.save(output_path)
    return str(output_path)


def save_filter_debug_overlay(image, masks, records, output_path, field_polygon_pixels, config):
    import numpy as np
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    height = image.height
    bench_y = coord_to_pixel(config["bench_y_cutoff"], height)
    field_y_min = coord_to_pixel(config["field_y_min"], height)
    field_y_max = coord_to_pixel(config["field_y_max"], height)
    if field_polygon_pixels:
        pts = [(int(round(x)), int(round(y))) for x, y in field_polygon_pixels]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=(0, 255, 255), width=4)
            draw.text((pts[0][0] + 6, pts[0][1] + 6), "field polygon", fill=(0, 255, 255))
    draw.line((0, int(round(bench_y)), image.width, int(round(bench_y))), fill=(255, 180, 0), width=3)
    draw.text((8, int(round(bench_y)) + 4), "bench cutoff", fill=(255, 180, 0))
    draw.line((0, int(round(field_y_min)), image.width, int(round(field_y_min))), fill=(0, 255, 0), width=3)
    draw.text((8, int(round(field_y_min)) + 4), "field y min", fill=(0, 255, 0))
    draw.line((0, int(round(field_y_max)), image.width, int(round(field_y_max))), fill=(0, 160, 0), width=2)
    for idx, record in enumerate(records):
        point = record.get("bottom_center_point")
        centroid = record.get("centroid")
        color = (0, 255, 0) if record["kept"] else (255, 0, 0)
        if point:
            x = int(round(point["x"]))
            y = int(round(point["y"]))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            reason = "kept" if record["kept"] else ",".join(record["rejection_reasons"][:2])
            draw.text((x + 7, y - 11), "{} {}".format(record["mask_index"], reason[:34]), fill=color)
        if centroid:
            cx = int(round(centroid["x"]))
            cy = int(round(centroid["y"]))
            draw.rectangle((cx - 3, cy - 3, cx + 3, cy + 3), outline=color, width=2)
    canvas.save(output_path)
    return str(output_path)


def run_filtered_masks(project_root, run_id, repo_path, output_dir, prompt, frame_count, device, dtype_name, allow_download_weights, disable_fused_kernels=True, config=None):
    result = {
        "status": "not_run",
        "run_id": run_id,
        "repo_path": str(repo_path),
        "output_dir": str(output_dir),
        "prompt": prompt,
        "prompt_slug": slugify_prompt(prompt),
        "frame_count_requested": frame_count,
        "device": device,
        "selected_dtype": dtype_name,
        "allow_download_weights_supplied": bool(allow_download_weights),
        "disable_fused_kernels": bool(disable_fused_kernels),
        "frames": [],
        "frame_metadata": [],
        "fused_kernel_patch": {"disabled": bool(disable_fused_kernels)},
        "dtype_diagnostics": [],
        "error": None,
    }
    defaults = {
        "filter_mode": "combined",
        "bench_y_cutoff": 0.18,
        "field_y_min": 0.18,
        "field_y_max": 0.96,
        "field_polygon": None,
        "require_green_surface": False,
        "green_sample_radius": 5,
        "green_sample_y_offset": 6,
        "min_mask_pixels": 35,
    }
    if config:
        defaults.update(config)
    if defaults["filter_mode"] not in FILTER_MODES:
        raise ValueError("filter_mode must be one of {}".format(sorted(FILTER_MODES)))
    result["filter_config"] = defaults
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sampled_frames(project_root, run_id, frame_count)
    result["frames"] = [str(path) for path in frames]
    if not allow_download_weights:
        result["status"] = "skipped"
        result["error"] = {"type": "RuntimeError", "message": "Filtered mask smoke requires --allow-download-weights."}
        return result
    if not frames:
        result["status"] = "failed"
        result["error"] = {"type": "ValueError", "message": "No sampled frames were found."}
        return result

    current_stage = "startup"
    torch = None
    selected_dtype = None
    model = None
    processor = None
    state = None
    try:
        import numpy as np
        import torch
        from PIL import Image

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        selected_dtype = torch_dtype(torch, dtype_name)
        repo = Path(repo_path)
        with temporary_sys_path(repo):
            current_stage = "import_sam3_builder"
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            with torch.inference_mode(), smoke_autocast_policy(torch, device, dtype_name), smoke_fused_kernel_policy(
                disable_fused_kernels, selected_dtype, result["fused_kernel_patch"]
            ):
                current_stage = "model_construction"
                result["dtype_diagnostics"].append(dtype_diagnostics(torch, "before_model_construction", selected_dtype))
                model = build_sam3_image_model(device=device)
                current_stage = "force_model_dtype"
                model = force_module_dtype(model, dtype_name, selected_dtype)
                model.eval()
                processor = Sam3Processor(model, device=device)
                result["dtype_diagnostics"].append(dtype_diagnostics(torch, "after_processor_construction", selected_dtype, model=model, processor=processor))

                for frame_index, frame_path in enumerate(frames):
                    current_stage = "frame_{}".format(frame_index)
                    image = Image.open(frame_path).convert("RGB")
                    image_array = np.asarray(image.convert("RGB"))
                    field_polygon_pixels = polygon_to_pixels(defaults.get("field_polygon"), image.width, image.height)
                    frame_dir = output_dir / "frame_{:03d}".format(frame_index)
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    original_path = frame_dir / "original.jpg"
                    image.save(original_path)
                    state = set_image_with_dtype(
                        processor,
                        image,
                        selected_dtype,
                        diagnostics=result["dtype_diagnostics"],
                        torch_module=torch,
                        selected_dtype=selected_dtype,
                        model=model,
                    )
                    state = cast_tensors(state, selected_dtype)
                    output = set_text_prompt_with_dtype(
                        processor,
                        prompt,
                        state,
                        selected_dtype,
                        diagnostics=result["dtype_diagnostics"],
                        torch_module=torch,
                        selected_dtype=selected_dtype,
                        model=model,
                    )
                    output = cast_tensors(output, selected_dtype)
                    masks_tensor = output.get("masks") if isinstance(output, dict) else None
                    boxes = output.get("boxes") if isinstance(output, dict) else None
                    scores = output.get("scores") if isinstance(output, dict) else None
                    masks = tensor_masks_to_numpy(masks_tensor)
                    records = []
                    kept = []
                    rejected = []
                    for mask_index, mask in enumerate(masks):
                        record = classify_mask(mask, mask_index, scores, image_array, defaults, field_polygon_pixels=field_polygon_pixels)
                        mask_path, overlay_path = save_individual_mask(mask, image, frame_dir, frame_index, mask_index, record["kept"])
                        record["mask_path"] = mask_path
                        record["single_mask_overlay_path"] = overlay_path
                        records.append(record)
                        if record["kept"]:
                            kept.append(record)
                        else:
                            rejected.append(record)
                    all_overlay_path = save_multi_overlay(image, masks, records, frame_dir / "all_sam3_masks_overlay.png", kept_only=False)
                    filtered_overlay_path = save_multi_overlay(image, masks, records, frame_dir / "filtered_active_player_masks_overlay.png", kept_only=True)
                    debug_overlay_path = save_filter_debug_overlay(image, masks, records, frame_dir / "filter_debug_overlay.png", field_polygon_pixels, defaults)
                    frame_record = {
                        "frame_index": frame_index,
                        "frame_path": str(frame_path),
                        "original_path": str(original_path),
                        "prompt": prompt,
                        "image_stats": image_stats(image),
                        "all_masks_overlay_path": all_overlay_path,
                        "filtered_active_player_masks_overlay_path": filtered_overlay_path,
                        "filter_debug_overlay_path": debug_overlay_path,
                        "all_mask_count": len(records),
                        "kept_mask_count": len(kept),
                        "rejected_mask_count": len(rejected),
                        "all_masks": records,
                        "kept_masks": kept,
                        "rejected_masks": rejected,
                        "boxes": tensor_to_list(boxes),
                        "scores": tensor_to_list(scores),
                    }
                    metadata_path = frame_dir / "frame_metadata.json"
                    write_json(metadata_path, frame_record)
                    frame_record["metadata_path"] = str(metadata_path)
                    result["frame_metadata"].append(frame_record)
                result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "failure_stage": current_stage,
        }
        if torch is not None and selected_dtype is not None:
            try:
                result["dtype_diagnostics"].append(dtype_diagnostics(torch, "exception_at_{}".format(current_stage), selected_dtype, model=model, processor=processor, state=state))
            except Exception as diag_exc:
                result["diagnostic_error"] = "{}: {}".format(diag_exc.__class__.__name__, diag_exc)
    return result
