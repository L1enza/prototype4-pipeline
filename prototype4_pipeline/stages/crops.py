from prototype4_pipeline.artifacts import write_json


def run_player_crop_extraction(context, frames, active_tracks):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for player crop extraction. Install `opencv-python` in the dry-run environment."
        ) from exc

    frame_by_index = {frame["frame_index"]: frame for frame in frames}
    crop_records = []

    for track in active_tracks.get("tracks", []):
        track_id = track["track_id"]
        track_dir = context.dirs["crops"] / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        for observation in track.get("observations", []):
            frame = frame_by_index.get(observation["frame_index"])
            bbox = observation.get("bbox_xyxy")
            if not frame or not bbox:
                continue
            image = cv2.imread(frame["path"])
            if image is None:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            height, width = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            crop_path = track_dir / "frame_{:06d}.jpg".format(observation["frame_index"])
            cv2.imwrite(str(crop_path), crop)
            crop_records.append(
                {
                    "track_id": track_id,
                    "frame_index": observation["frame_index"],
                    "timestamp_sec": observation.get("timestamp_sec"),
                    "path": str(crop_path),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "source_frame": frame["path"],
                }
            )

    result = {
        "status": "complete",
        "stage": "player_crop_extraction",
        "input_frame_count": len(frames),
        "active_track_count": len(active_tracks.get("tracks", [])),
        "output_dir": str(context.dirs["crops"]),
        "crop_count": len(crop_records),
        "crops": crop_records,
    }
    write_json(context.dirs["crops"] / "crops_manifest.json", result)
    return result
