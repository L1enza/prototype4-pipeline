"""The legibility gate keeps illegible crops away from number readers."""

import sys

import pytest

from conftest import PROJECT_ROOT
from prototype4_pipeline.integrations import legibility

pytest.importorskip("cv2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import audit_jersey_crop_visibility as audit  # noqa: E402
import generate_enhanced_number_regions as regions  # noqa: E402
import rebuild_nll_test4 as rebuild  # noqa: E402


def record(path, candidate=True, status="ocr_ready", width=True):
    row = {"crop_path": path, "ocr_candidate": candidate, "audit_status": status, "rejection_reasons": [], "rejection_reason": None}
    if width:
        row["crop_width"] = 40
    return row


def test_gate_rejects_illegible_crops_and_keeps_legible_ones():
    records = [record("legible.png"), record("illegible.png"), record("borderline.png", False, "borderline"), record("unreadable.png", False, "rejected", width=False)]
    counts = audit.apply_legibility_gate(records, {"legible.png": 0.9, "illegible.png": 0.2, "borderline.png": 0.1}, 0.5)

    legible, illegible, borderline, unreadable = records
    assert legible["ocr_candidate"] and legible["legible_by_classifier"] is True
    assert not illegible["ocr_candidate"] and illegible["audit_status"] == "rejected"
    assert illegible["rejection_reasons"] == ["not_legible_by_classifier"]
    assert borderline["audit_status"] == "rejected"
    assert unreadable["legibility_score"] is None and unreadable["audit_status"] == "rejected"
    assert counts == {"crops_scored": 3, "crops_legible": 1, "ocr_ready_before_gate": 2, "ocr_ready_after_gate": 1}


def test_score_exactly_at_threshold_is_not_legible():
    records = [record("edge.png")]
    audit.apply_legibility_gate(records, {"edge.png": 0.5}, 0.5)
    assert not records[0]["ocr_candidate"]


def test_audit_warns_when_gate_is_off_or_nothing_passes():
    assert any("gate disabled" in w for w in audit.audit_warnings([{"x": 1}], {"enabled": False}))
    assert any("No crop passed" in w for w in audit.audit_warnings([], {"enabled": True}))
    assert audit.audit_warnings([{"x": 1}], {"enabled": True}) == []


def test_number_regions_accept_a_clip_with_no_candidates():
    # Before the gate, an empty candidate list stopped the whole rebuild.
    assert regions.join_ocr_ready_crops({"crops": []}, {"crops": [{"ocr_candidate": False, "crop_path": "x.png"}]}, {"tracks": []}) == []


def test_missing_weights_need_explicit_download_permission(tmp_path):
    with pytest.raises(FileNotFoundError, match="--allow-download-weights"):
        legibility.ensure_weights(tmp_path / "missing.pth", allow_download=False)


def test_weights_with_the_wrong_checksum_are_refused(tmp_path):
    fake = tmp_path / "weights.pth"
    fake.write_bytes(b"not the published weights")
    with pytest.raises(ValueError, match="SHA-256"):
        legibility.ensure_weights(fake)


def test_model_loads_a_state_dict_and_scores_probabilities(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from torch import nn
    from torchvision import models

    backbone = models.resnet34(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 1)
    weights = tmp_path / "random.pth"
    torch.save({f"model_ft.{key}": value for key, value in backbone.state_dict().items()}, weights)

    scores = legibility.score_images(legibility.load_model(weights), [Image.new("RGB", (40, 60), (200, 30, 40))] * 3)
    assert len(scores) == 3 and all(0.0 <= s <= 1.0 for s in scores)


def test_rebuild_runs_the_gate_and_flags_missing_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rebuild_nll_test4.py", "--max-frames", "60"])
    plan = {stage["name"]: stage for stage in rebuild.stages(rebuild.parse_args())}
    assert "--legibility-weights" in plan["visibility"]["cmd"]

    monkeypatch.setattr(rebuild, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rebuild, "ollama_has", lambda model: True)
    _video, problems = rebuild.preflight(rebuild.parse_args())
    assert any("Legibility classifier weights not cached" in problem for problem in problems)
