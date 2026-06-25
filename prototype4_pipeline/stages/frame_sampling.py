from prototype4_pipeline.artifacts import frame_record, write_json


def run_frame_sampling(context, ingestion):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for frame sampling. Install `opencv-python` in the dry-run environment."
        ) from exc

    video_cfg = context.config.get("video", {})
    sample_fps = float(video_cfg.get("sample_fps", 2.0))
    max_frames = int(video_cfg.get("max_frames", 48))
    image_ext = video_cfg.get("image_ext", "jpg")
    source_fps = float(ingestion.get("fps") or 0.0)
    step = max(1, round(source_fps / sample_fps)) if source_fps > 0 and sample_fps > 0 else 1

    capture = cv2.VideoCapture(str(context.video_path))
    records = []
    source_index = 0
    sampled_index = 0

    while len(records) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % step == 0:
            timestamp_sec = source_index / source_fps if source_fps > 0 else float(sampled_index)
            frame_path = context.dirs["frames"] / "frame_{:06d}.{}".format(sampled_index, image_ext)
            cv2.imwrite(str(frame_path), frame)
            height, width = frame.shape[:2]
            records.append(
                frame_record(
                    sampled_index,
                    timestamp_sec,
                    source_index,
                    str(frame_path),
                    width,
                    height,
                )
            )
            sampled_index += 1
        source_index += 1

    capture.release()
    timestamps = [record["timestamp_sec"] for record in records]
    manifest = {
        "status": "complete",
        "stage": "frame_sampling",
        "input_video": str(context.video_path),
        "output_dir": str(context.dirs["frames"]),
        "sampling": {
            "requested_sample_fps": sample_fps,
            "source_fps": source_fps,
            "source_frame_step": step,
            "max_frames": max_frames,
            "image_ext": image_ext,
        },
        "counts": {
            "sampled_frames": len(records),
            "source_frame_count": ingestion.get("frame_count"),
        },
        "timestamp_range_sec": {
            "start": timestamps[0] if timestamps else None,
            "end": timestamps[-1] if timestamps else None,
        },
        "frames": records,
    }
    write_json(context.dirs["metadata"] / "sampled_frames.json", manifest)
    return records
