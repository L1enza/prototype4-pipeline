from prototype4_pipeline.artifacts import write_json


def run_video_ingestion(context):
    if not context.video_path.exists():
        raise FileNotFoundError("Input video does not exist: {}".format(context.video_path))

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video ingestion. Install `opencv-python` in the dry-run environment."
        ) from exc

    capture = cv2.VideoCapture(str(context.video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open input video: {}".format(context.video_path))

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    payload = {
        "status": "complete",
        "stage": "video_ingestion",
        "output_dir": str(context.dirs["metadata"]),
        "video_path": str(context.video_path),
        "video_metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": frame_count / fps if fps > 0 else None,
        },
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps > 0 else None,
    }
    write_json(context.dirs["metadata"] / "video_ingestion.json", payload)
    return payload
