import json


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def frame_record(frame_index, timestamp_sec, source_frame_index, path, width, height):
    return {
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "source_frame_index": source_frame_index,
        "path": path,
        "width": width,
        "height": height,
    }
