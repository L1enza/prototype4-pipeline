"""Scoring rules of scripts/eval_jersey_readers.py, checked against the committed labels."""

import importlib.util
import json

from conftest import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("eval_jersey_readers", PROJECT_ROOT / "scripts" / "eval_jersey_readers.py")
evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluator)

LABELS = PROJECT_ROOT / "review_exports" / "nll_test4_manual_jersey_eval" / "label_sheet.json"


def test_crop_outcomes():
    assert evaluator.crop_outcome("unknown", "no", None) == "correct_abstain"
    assert evaluator.crop_outcome("unknown", "no", "17") == "hallucinated"
    assert evaluator.crop_outcome("42", "yes", "42") == "correct"
    assert evaluator.crop_outcome("42", "yes", "47") == "wrong"
    assert evaluator.crop_outcome("42", "yes", None) == "abstained"
    assert evaluator.crop_outcome("4", "partial", "42") == "consistent_with_partial"


def test_vote_counts_each_frame_once():
    # Two preprocessing variants of one frame are not two frames of evidence.
    assert evaluator.vote([(5, "16"), (5, "16")], 2)["number"] is None
    assert evaluator.vote([(5, "16"), (6, "16")], 2)["number"] == "16"
    assert evaluator.vote([(1, "42"), (2, "42"), (3, "47"), (4, "47")], 2)["reason"] == "tie"


def test_every_crop_is_labeled():
    rows = json.loads(LABELS.read_text(encoding="utf-8"))
    assert len(rows) == 130
    assert all(row["manual_readable"] in {"yes", "partial", "no"} for row in rows)


def test_human_labels_identify_expected_tracks():
    rows = json.loads(LABELS.read_text(encoding="utf-8"))
    perfect = {row["eval_id"]: {"number": None if row["manual_readable"] == "no" else row["manual_label"]} for row in rows}
    result = evaluator.score(rows, perfect, min_agree=2)
    truth = {t["track_id"]: t["human_truth"]["number"] for t in result["per_track"] if t["human_truth"]["number"]}
    assert truth == {2: "42", 8: "77", 19: "1"}
    assert result["track_outcomes"].get("wrong", 0) == 0
