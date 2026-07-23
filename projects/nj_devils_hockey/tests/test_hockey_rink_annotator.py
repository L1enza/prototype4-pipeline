import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "annotate_hockey_rink_keypoints.py"
SPEC = importlib.util.spec_from_file_location("hockey_annotator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_requested_reference_frames_are_valid():
    assert MODULE.parse_frames("0,35,70,105,138,175,210,245,276", 277) == [0, 35, 70, 105, 138, 175, 210, 245, 276]


def test_hockey_landmarks_match_supplied_rink_dimensions():
    payload = json.loads((ROOT / "configs" / "canonical_hockey_rink_landmarks.json").read_text(encoding="utf-8"))
    assert payload["field_template"] == "assets/hockey/icerink.jpg"
    assert payload["template_size_px"] == {"width": 1568, "height": 980}
    ids = {row["id"] for row in payload["landmarks"]}
    assert {"center_spot", "left_blue_line_top_boards", "right_blue_line_bottom_boards", "left_crease_edge"} <= ids
    for row in payload["landmarks"]:
        x, y = row["template_xy"]
        assert 0 <= x < 1568
        assert 0 <= y < 980


def test_generator_uses_only_new_canonical_asset():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'DEFAULT_RINK = "assets/hockey/icerink.jpg"' in source
    assert "njdevils_rink.png" not in source


def test_hockey_customization_contains_required_workflow():
    source = (ROOT / "scripts" / "annotate_field_keypoints.py").read_text(encoding="utf-8")
    # Exercise the deterministic textual substitutions against function-bearing HTML-like source.
    assert "function canonicalLineSegments" in source
    assert "function paintedLineSegments" in source
    required = ("Free correspondence", "snap to nearest painted line", "snap to nearest line intersection", "Select / move", "Delete")
    for text in required:
        assert text in source
