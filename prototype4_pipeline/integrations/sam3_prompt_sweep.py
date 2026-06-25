import re
import traceback
from pathlib import Path

from prototype4_pipeline.integrations.sam3_smoke import (
    cast_tensors,
    dtype_diagnostics,
    force_module_dtype,
    run_import_check,
    safe_shape,
    sampled_frames,
    save_mask_overlay,
    set_image_with_dtype,
    set_text_prompt_with_dtype,
    smoke_autocast_policy,
    smoke_fused_kernel_policy,
    temporary_sys_path,
    torch_dtype,
    write_json,
)

DEFAULT_PROMPTS = [
    "person",
    "player",
    "lacrosse player",
    "athlete",
    "sports player",
    "active lacrosse player",
    "player on the field",
]


def slugify_prompt(prompt):
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug or "prompt"


def tensor_to_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
        if hasattr(value, "tolist"):
            return value.tolist()
    return []


def mask_pixel_counts(masks):
    if masks is None or not hasattr(masks, "detach"):
        return []
    masks_cpu = masks.detach().to("cpu")
    if len(masks_cpu.shape) == 4 and int(masks_cpu.shape[1]) == 1:
        masks_cpu = masks_cpu[:, 0]
    if len(masks_cpu.shape) < 3:
        return [int(masks_cpu.bool().sum().item())]
    return [int(mask.bool().sum().item()) for mask in masks_cpu]


def mask_centroids(masks):
    if masks is None or not hasattr(masks, "detach"):
        return []
    masks_cpu = masks.detach().to("cpu")
    if len(masks_cpu.shape) == 4 and int(masks_cpu.shape[1]) == 1:
        masks_cpu = masks_cpu[:, 0]
    if len(masks_cpu.shape) < 3:
        masks_cpu = masks_cpu.unsqueeze(0)
    centroids = []
    for mask in masks_cpu.bool():
        ys, xs = mask.nonzero(as_tuple=True)
        if len(xs) == 0:
            centroids.append(None)
        else:
            centroids.append({"x": float(xs.float().mean().item()), "y": float(ys.float().mean().item())})
    return centroids


def image_stats(image):
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return {
        "width": int(image.width),
        "height": int(image.height),
        "mean_rgb": [float(v) for v in arr.mean(axis=(0, 1))],
        "std_rgb": [float(v) for v in arr.std(axis=(0, 1))],
    }


def active_field_count(centroids, image_width, image_height):
    count = 0
    for point in centroids:
        if not point:
            continue
        x_ratio = point["x"] / max(image_width, 1)
        y_ratio = point["y"] / max(image_height, 1)
        # Temporary broadcast-lacrosse heuristic: ignore top boards/crowd band and extreme edges.
        if 0.04 <= x_ratio <= 0.96 and 0.18 <= y_ratio <= 0.94:
            count += 1
    return count


def likely_bench_or_crowd_count(centroids, image_width, image_height):
    count = 0
    for point in centroids:
        if not point:
            continue
        x_ratio = point["x"] / max(image_width, 1)
        y_ratio = point["y"] / max(image_height, 1)
        if y_ratio < 0.18 or x_ratio < 0.04 or x_ratio > 0.96:
            count += 1
    return count


def prompt_score(record):
    total_masks = int(record["total_masks"])
    active = int(record["masks_on_active_field_area"])
    off_field = int(record["likely_bench_crowd_masks"])
    mean_score = float(record.get("mean_score") or 0.0)
    return (active * 4.0) + (total_masks * 0.7) + mean_score - (off_field * 2.0)


def create_contact_sheets(frame_records, prompts, output_dir):
    from PIL import Image, ImageDraw, ImageFont

    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    by_frame = {}
    for record in frame_records:
        by_frame.setdefault(record["frame_index"], {})[record["prompt"]] = record

    sheets = []
    for frame_index, prompt_records in sorted(by_frame.items()):
        cells = []
        for prompt in prompts:
            record = prompt_records.get(prompt)
            if not record:
                continue
            if not record.get("overlay_path"):
                continue
            overlay_path = Path(record["overlay_path"])
            if not overlay_path.exists():
                continue
            image = Image.open(overlay_path).convert("RGB")
            image.thumbnail((360, 220))
            canvas = Image.new("RGB", (360, 260), "white")
            canvas.paste(image, ((360 - image.width) // 2, 0))
            draw = ImageDraw.Draw(canvas)
            label = "{} | masks {}".format(prompt, record["mask_count"])
            draw.text((8, 226), label[:52], fill=(0, 0, 0))
            cells.append(canvas)
        if not cells:
            continue
        cols = min(4, len(cells))
        rows = (len(cells) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 360, rows * 260), "white")
        for idx, cell in enumerate(cells):
            sheet.paste(cell, ((idx % cols) * 360, (idx // cols) * 260))
        sheet_path = contact_dir / "frame_{:03d}_prompt_comparison.png".format(frame_index)
        sheet.save(sheet_path)
        sheets.append(str(sheet_path))
    return sheets


def run_prompt_sweep(project_root, run_id, repo_path, output_dir, prompts, frame_count, device, dtype_name, allow_download_weights, disable_fused_kernels=True):
    result = {
        "status": "not_run",
        "run_id": run_id,
        "repo_path": str(repo_path),
        "output_dir": str(output_dir),
        "prompts": prompts,
        "frame_count_requested": frame_count,
        "device": device,
        "selected_dtype": dtype_name,
        "allow_download_weights_supplied": bool(allow_download_weights),
        "disable_fused_kernels": bool(disable_fused_kernels),
        "frames": [],
        "per_frame_metadata": [],
        "prompt_summaries": [],
        "contact_sheets": [],
        "fused_kernel_patch": {"disabled": bool(disable_fused_kernels)},
        "dtype_diagnostics": [],
        "error": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sampled_frames(project_root, run_id, frame_count)
    result["frames"] = [str(path) for path in frames]
    if not allow_download_weights:
        result["status"] = "skipped"
        result["error"] = {"type": "RuntimeError", "message": "Prompt sweep requires --allow-download-weights."}
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

                prompt_totals = {
                    prompt: {"prompt": prompt, "total_masks": 0, "masks_on_active_field_area": 0, "likely_bench_crowd_masks": 0, "scores": []}
                    for prompt in prompts
                }
                for frame_index, frame_path in enumerate(frames):
                    image = Image.open(frame_path).convert("RGB")
                    stats = image_stats(image)
                    original_by_prompt = {}
                    state = None
                    current_stage = "set_image_frame_{}".format(frame_index)
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
                    for prompt in prompts:
                        slug = slugify_prompt(prompt)
                        prompt_dir = output_dir / slug
                        prompt_dir.mkdir(parents=True, exist_ok=True)
                        original_path = prompt_dir / "frame_{:03d}_original.jpg".format(frame_index)
                        if not original_path.exists():
                            image.save(original_path)
                        current_stage = "prompt_{}_frame_{}".format(slug, frame_index)
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
                        masks = output.get("masks") if isinstance(output, dict) else None
                        boxes = output.get("boxes") if isinstance(output, dict) else None
                        scores = output.get("scores") if isinstance(output, dict) else None
                        mask_count = int(masks.shape[0]) if masks is not None and hasattr(masks, "shape") and len(masks.shape) > 0 else 0
                        overlay_paths = save_mask_overlay(image, masks, prompt_dir, frame_index)
                        overlay_path = overlay_paths[0] if overlay_paths else None
                        combined_mask_path = overlay_paths[1] if len(overlay_paths) > 1 else None
                        pixels = mask_pixel_counts(masks)
                        centroids = mask_centroids(masks)
                        active_count = active_field_count(centroids, image.width, image.height)
                        bench_count = likely_bench_or_crowd_count(centroids, image.width, image.height)
                        score_values = [float(v) for v in tensor_to_list(scores)]
                        prompt_totals[prompt]["total_masks"] += mask_count
                        prompt_totals[prompt]["masks_on_active_field_area"] += active_count
                        prompt_totals[prompt]["likely_bench_crowd_masks"] += bench_count
                        prompt_totals[prompt]["scores"].extend(score_values)
                        record = {
                            "prompt": prompt,
                            "prompt_slug": slug,
                            "frame_index": frame_index,
                            "frame_path": str(frame_path),
                            "original_path": str(original_path),
                            "overlay_path": overlay_path,
                            "combined_mask_path": combined_mask_path,
                            "mask_count": mask_count,
                            "boxes": tensor_to_list(boxes),
                            "scores": score_values,
                            "mask_pixel_counts": pixels,
                            "mask_centroids": centroids,
                            "masks_on_active_field_area": active_count,
                            "likely_bench_crowd_masks": bench_count,
                            "image_stats": stats,
                            "output_shapes": {
                                "masks": safe_shape(masks),
                                "boxes": safe_shape(boxes),
                                "scores": safe_shape(scores),
                            },
                        }
                        metadata_path = prompt_dir / "frame_{:03d}_metadata.json".format(frame_index)
                        write_json(metadata_path, record)
                        record["metadata_path"] = str(metadata_path)
                        result["per_frame_metadata"].append(record)
                        original_by_prompt[prompt] = str(original_path)
                summaries = []
                for prompt, summary in prompt_totals.items():
                    scores = summary.pop("scores")
                    summary["mean_score"] = float(sum(scores) / len(scores)) if scores else 0.0
                    summary["visual_usefulness"] = prompt_score(summary)
                    summary["ranking_notes"] = "Heuristic only: field/crowd estimates use mask centroids before field-boundary calibration."
                    summaries.append(summary)
                summaries.sort(key=lambda item: item["visual_usefulness"], reverse=True)
                for rank, summary in enumerate(summaries, start=1):
                    summary["rank"] = rank
                result["prompt_summaries"] = summaries
                result["contact_sheets"] = create_contact_sheets(result["per_frame_metadata"], prompts, output_dir)
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
