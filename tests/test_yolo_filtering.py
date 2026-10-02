"""YOLO masks reach the tracker through the same per-frame layout and field filter as SAM 3."""

import sys

import numpy as np
from PIL import Image

from conftest import PROJECT_ROOT
from prototype4_pipeline.integrations.sam3_filtering import resolve_filter_config, write_filtered_frame
from prototype4_pipeline.integrations.yolo_filtering import resolve_weights_path, result_to_arrays, run_yolo_filtered_masks

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_player_tracklet_smoke import load_detections  # noqa: E402

WIDTH, HEIGHT = 200, 100


class FakeTensor:
    """Stands in for a torch tensor: the code only calls detach/to/numpy/tolist."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def to(self, _device):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class FakeBoxes:
    def __init__(self, conf, xyxy):
        self.conf = FakeTensor(conf)
        self.xyxy = FakeTensor(xyxy)

    def __len__(self):
        return len(self.conf.values)


class FakeMasks:
    def __init__(self, data):
        self.data = FakeTensor(data)


class FakeResult:
    def __init__(self, masks=None, boxes=None):
        self.masks = masks
        self.boxes = boxes


def box_mask(x0, y0, x1, y1, height=HEIGHT, width=WIDTH):
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def green_image():
    return Image.new("RGB", (WIDTH, HEIGHT), (40, 140, 40))


def test_write_filtered_frame_keeps_field_players_and_rejects_bench(tmp_path):
    frame_path = tmp_path / "frame_000.jpg"
    image = green_image()
    image.save(frame_path)
    on_field = box_mask(50, 60, 60, 90)  # feet at y=89, inside the default field band
    on_bench = box_mask(120, 2, 130, 12)  # centroid above the bench cutoff
    masks = np.stack([on_field, on_bench])
    scores = FakeTensor([0.9, 0.8])
    config = resolve_filter_config(None)

    record = write_filtered_frame(image, frame_path, 0, masks, scores, None, tmp_path, config, "yolo class person")

    assert record["kept_mask_count"] == 1
    assert record["kept_masks"][0]["mask_index"] == 0
    assert record["kept_masks"][0]["confidence_score"] == 0.9
    assert "centroid_in_bench_strip" in record["rejected_masks"][0]["rejection_reasons"]

    detections = load_detections(tmp_path, {})
    assert list(detections) == [0]
    [det] = detections[0]["detections"]
    assert det["mask_id"] == 0
    assert det["foot_point_2d"]["y"] == 89.0
    assert det["sam_confidence_score"] == 0.9


def test_result_to_arrays_handles_frames_with_no_people():
    masks, scores, boxes = result_to_arrays(FakeResult(), (WIDTH, HEIGHT))
    assert masks.shape == (0, HEIGHT, WIDTH)
    assert scores is None and boxes is None


def test_result_to_arrays_thresholds_masks_and_keeps_tensor_scores():
    data = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    data[0, 10:20, 10:20] = 0.9
    data[0, 30:40, 30:40] = 0.2  # below the 0.5 mask threshold
    result = FakeResult(FakeMasks(data), FakeBoxes([0.7], [[10, 10, 20, 20]]))

    masks, scores, boxes = result_to_arrays(result, (WIDTH, HEIGHT))

    assert masks.dtype == bool
    assert int(masks[0].sum()) == 100
    assert scores.tolist() == [0.7]
    assert boxes.tolist() == [[10, 10, 20, 20]]


def test_missing_weights_without_download_flag_fails_before_loading(tmp_path):
    frame_path = tmp_path / "frame_000.jpg"
    green_image().save(frame_path)

    result = run_yolo_filtered_masks(tmp_path, "run", tmp_path / "out", [frame_path], model="not-there-seg.pt")

    assert result["status"] == "failed"
    assert "--allow-download-weights" in result["error"]["message"]
    assert not (tmp_path / "out" / "frame_000").exists()


def test_bare_model_names_are_cached_under_project(tmp_path):
    assert resolve_weights_path(tmp_path, "yolo11m-seg.pt") == tmp_path / ".cache" / "models" / "yolo11m-seg.pt"
    assert resolve_weights_path(tmp_path, "weights/custom.pt") == tmp_path / "weights" / "custom.pt"
