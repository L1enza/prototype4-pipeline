"""Team clustering leaves ambiguous tracks unknown, and the confirm step warns about bad splits."""

import copy
import json
import sys

import numpy as np
import pytest

from conftest import PROJECT_ROOT
from prototype4_pipeline.integrations.team_mapping import label_color_summary, split_warnings

pytest.importorskip("cv2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import confirm_team_mapping  # noqa: E402
import run_team_assignment_and_polygon_demo_v2 as v2  # noqa: E402

CONFIG = json.loads((PROJECT_ROOT / "configs" / "nll_test4_team_assignment_v2.json").read_text(encoding="utf-8"))
WHITE, BLUE, MAROON, TAN = (235, 235, 238), (30, 60, 160), (95, 20, 35), (200, 160, 110)


def shirt(rng, base, trim=None, occluder=None, stripes=False, h=60, w=40):
    """A synthetic upper-torso crop: uniform colour, sleeve trim and a shadow band."""
    img = np.clip(rng.normal(base, 10, (h, w, 3)), 0, 255).astype(np.uint8)
    if trim is not None:
        img[:, :5] = trim
        img[:, -5:] = trim
    if occluder is not None:
        img[:, w // 2:] = np.clip(rng.normal(occluder, 10, (h, w - w // 2, 3)), 0, 255)
    if stripes:
        for x in range(0, w, 8):
            img[:, x:x + 4] = (20, 20, 20)
    img[40:, :] = (img[40:, :] * rng.uniform(0.6, 0.9)).astype(np.uint8)
    return img


def assign(spec, weights=None, seed=0):
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(CONFIG)
    cfg["feature_weights"] = weights or {"appearance": 0.0, "color": 1.0}
    signatures, truth = {}, {}
    for track_id, (name, kwargs) in enumerate(spec, start=1):
        observations = []
        for frame in range(8):
            crop = shirt(rng, **kwargs)
            observations.append({**v2.color_features_from_crop(crop, None), "observation_quality_score": 0.8,
                                 "appearance_embedding": None, "crop_rgb": crop, "frame_index": frame})
        signatures[track_id] = v2.aggregate_signature(track_id, observations, [], cfg, "color_only")
        truth[track_id] = name
    assignments, diagnostics = v2.assign_tracks(signatures, cfg, {})
    groups = {}
    for track_id, row in assignments.items():
        groups.setdefault(row["final_class"], []).append(truth[track_id])
    return {label: sorted(names) for label, names in groups.items()}, assignments, diagnostics


WHITE_TRACK = ("white", {"base": WHITE, "trim": BLUE})
MAROON_TRACK = ("maroon", {"base": MAROON, "trim": TAN})
TEAMS = [WHITE_TRACK] * 10 + [MAROON_TRACK] * 9


def test_clean_clip_splits_white_from_maroon():
    groups, _rows, diagnostics = assign(TEAMS)
    assert groups == {"team_a": ["white"] * 10, "team_b": ["maroon"] * 9}
    assert diagnostics["method"] == "robust_two_means"


def test_half_hidden_player_is_unknown_not_put_on_the_other_team():
    groups, rows, _diag = assign(TEAMS + [("occluded", {"base": WHITE, "trim": BLUE, "occluder": MAROON})])
    assert groups["unknown"] == ["occluded"]
    assert rows[20]["final_class"] == "unknown"


def test_one_odd_track_cannot_take_a_whole_cluster():
    # The old Gaussian mixture made a lone trainer "team_b" and merged both teams into "team_a".
    groups, _rows, diagnostics = assign(TEAMS + [("trainer", {"base": (40, 200, 60)})])
    assert groups == {"team_a": ["white"] * 10, "team_b": ["maroon"] * 9, "unknown": ["trainer"]}
    assert diagnostics["outlier_track_ids"] == [20]


def test_striped_referee_is_not_put_on_a_team():
    groups, _rows, _diag = assign(TEAMS + [("referee", {"base": (235, 235, 235), "stripes": True})])
    assert groups["unknown"] == ["referee"]


def test_one_team_on_screen_gives_no_split_instead_of_two_fake_teams():
    groups, _rows, diagnostics = assign([WHITE_TRACK] * 10 + [MAROON_TRACK])
    assert set(groups) == {"unknown"}
    assert diagnostics["status"] == "no_clear_two_team_split"


def test_posteriors_are_no_longer_always_certain():
    _groups, rows, _diag = assign(TEAMS + [("occluded", {"base": WHITE, "trim": BLUE, "occluder": MAROON})])
    assert 0.4 < rows[20]["posterior_probability"] < 0.65
    assert all(rows[t]["posterior_probability"] > 0.9 for t in range(1, 20))


def test_feature_weights_now_change_the_features():
    rng = np.random.default_rng(1)
    signatures = {
        t: {"color_signature": v2.normalize_vector(rng.normal(0, 1, 60)), "appearance_signature": v2.normalize_vector(rng.normal(0, 1, 512))}
        for t in range(1, 9)
    }
    ids = list(signatures)
    colour_only = v2.team_feature_matrix(signatures, ids, {"appearance": 0.0, "color": 1.0})
    mostly_appearance = v2.team_feature_matrix(signatures, ids, {"appearance": 0.9, "color": 0.1})
    assert colour_only.shape == mostly_appearance.shape == (8, 3)
    assert not np.allclose(np.abs(colour_only), np.abs(mostly_appearance))


# OpenCV 8-bit Lab, as team assignment writes it.
LIGHT, DARK = [217.0, 128.0, 130.0], [64.0, 158.0, 138.0]


def rows(*labels_and_labs):
    return {i: {"track_id": i, "final_class": label, "representative_median_lab": lab} for i, (label, lab) in enumerate(labels_and_labs, start=1)}


def test_clean_split_has_no_warnings():
    summary = label_color_summary(rows(*[("team_a", LIGHT)] * 5, *[("team_b", DARK)] * 4))
    assert split_warnings(summary) == []


def test_warns_when_one_label_holds_both_teams():
    summary = label_color_summary(rows(*[("team_a", LIGHT)] * 10, *[("team_a", DARK)] * 9, ("team_b", [150.0, 90.0, 180.0])))
    warnings = split_warnings(summary)
    assert any("team_a mixes light and dark shirts" in w for w in warnings)
    assert any("lopsided split: only 1 of 20" in w for w in warnings)


def test_warns_when_both_labels_look_the_same_or_one_is_missing():
    assert any("both look light" in w for w in split_warnings(label_color_summary(rows(*[("team_a", LIGHT)] * 4, *[("team_b", LIGHT)] * 4))))
    assert any("no tracks are labelled team_b" in w for w in split_warnings(label_color_summary(rows(*[("team_a", LIGHT)] * 4))))


def run_confirm(tmp_path, monkeypatch, assignments, *argv):
    teams = tmp_path / "teams.json"
    teams.write_text(json.dumps(list(assignments.values())), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"inputs": {"team_assignments": str(teams), "team_mapping_confirmation": str(tmp_path / "confirmation.json")}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["confirm_team_mapping.py", "--config", str(config), *argv])
    return confirm_team_mapping.main(), tmp_path / "confirmation.json"


CONFIRM = ("--team-a", "TOR", "--team-b", "OSH", "--confirmed-by", "tester")


def test_confirm_refuses_a_suspicious_split_without_accept_warnings(tmp_path, monkeypatch, capsys):
    merged = rows(*[("team_a", LIGHT)] * 10, *[("team_a", DARK)] * 9, ("team_b", [150.0, 90.0, 180.0]))
    code, written = run_confirm(tmp_path, monkeypatch, merged, *CONFIRM)
    assert code == 2 and not written.exists()
    assert "WARNING: team_a mixes light and dark shirts" in capsys.readouterr().out

    code, written = run_confirm(tmp_path, monkeypatch, merged, *CONFIRM, "--accept-warnings")
    assert code == 0
    assert json.loads(written.read_text(encoding="utf-8"))["warnings_accepted"]


def test_confirm_writes_a_clean_split(tmp_path, monkeypatch):
    clean = rows(*[("team_a", LIGHT)] * 5, *[("team_b", DARK)] * 4)
    code, written = run_confirm(tmp_path, monkeypatch, clean, *CONFIRM)
    assert code == 0
    assert json.loads(written.read_text(encoding="utf-8"))["warnings_accepted"] == []
