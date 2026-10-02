"""CPU-friendly player masks from an Ultralytics YOLO segmentation model.

Produces the same per-frame layout as ``sam3_filtering.run_filtered_masks``
(``frame_XXX/frame_metadata.json`` plus kept/rejected mask PNGs), so the
stabilized tracker and every later stage consume it unchanged. YOLO only
knows "person", so the shared field/bench geometry filter is what removes
crowd and bench detections, exactly as it does for SAM 3.
"""

import traceback
from pathlib import Path

from prototype4_pipeline.integrations.sam3_filtering import resolve_filter_config, write_filtered_frame

DEFAULT_YOLO_MODEL = "yolo11m-seg.pt"
PERSON_CLASS_ID = 0


def resolve_weights_path(project_root, model):
    """Bare model names live under .cache/models; explicit paths are kept."""
    path = Path(model)
    if path.parent == Path("."):
        return Path(project_root) / ".cache" / "models" / path.name
    return path if path.is_absolute() else Path(project_root) / path


def result_to_arrays(result, image_size):
    """Return (masks[N,H,W] bool, scores, boxes) for one YOLO result.

    Scores and boxes stay CPU tensors because the shared frame writer reads
    them through sam3_prompt_sweep.tensor_to_list, which ignores plain lists.
    """
    import numpy as np

    width, height = image_size
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, height, width), dtype=bool), None, None
    masks = result.masks.data.detach().to("cpu").numpy() > 0.5
    if masks.shape[1:] != (height, width):
        import cv2

        masks = np.stack([cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
    return masks, result.boxes.conf.detach().to("cpu"), result.boxes.xyxy.detach().to("cpu")


def run_yolo_filtered_masks(project_root, run_id, output_dir, frame_paths, device="cpu", model=DEFAULT_YOLO_MODEL, allow_download_weights=False, conf=0.25, imgsz=1280, config=None):
    weights_path = resolve_weights_path(project_root, model)
    defaults = resolve_filter_config(config)
    result = {
        "status": "not_run",
        "detector": "yolo_seg",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "prompt": "yolo class person",
        "model": str(model),
        "weights_path": str(weights_path),
        "device": device,
        "conf": conf,
        "imgsz": imgsz,
        "allow_download_weights_supplied": bool(allow_download_weights),
        "filter_config": defaults,
        "frames": [str(path) for path in frame_paths],
        "frame_metadata": [],
        "error": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not frame_paths:
        result["status"] = "failed"
        result["error"] = {"type": "ValueError", "message": "No decoded frames were supplied."}
        return result
    if not weights_path.exists() and not allow_download_weights:
        result["status"] = "failed"
        result["error"] = {
            "type": "RuntimeError",
            "message": "YOLO weights not found at {}; pass --allow-download-weights to fetch them.".format(weights_path),
        }
        return result

    current_stage = "startup"
    try:
        from PIL import Image
        from ultralytics import YOLO

        current_stage = "model_construction"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        yolo = YOLO(str(weights_path))
        for frame_index, frame_path in enumerate(frame_paths):
            current_stage = "frame_{}".format(frame_index)
            image = Image.open(frame_path).convert("RGB")
            prediction = yolo.predict(
                source=str(frame_path),
                device=device,
                classes=[PERSON_CLASS_ID],
                conf=conf,
                imgsz=imgsz,
                retina_masks=True,
                verbose=False,
            )[0]
            masks, scores, boxes = result_to_arrays(prediction, image.size)
            frame_record = write_filtered_frame(image, frame_path, frame_index, masks, scores, boxes, output_dir, defaults, result["prompt"])
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
    return result
