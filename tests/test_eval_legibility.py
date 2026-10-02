"""Scoring for the legibility-gate evaluation."""

import sys

from conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_legibility_classifier import gate_report, roc_auc  # noqa: E402


def test_roc_auc_counts_pairwise_wins_and_ties():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5
    assert roc_auc([], [0.5]) is None


def test_gate_report_separates_wrong_and_correct_reads_before_and_after_the_gate():
    rows = [
        {"eval_id": "a", "manual_readable": "yes", "manual_label": "22"},
        {"eval_id": "b", "manual_readable": "no", "manual_label": "unknown"},
        {"eval_id": "c", "manual_readable": "yes", "manual_label": "7"},
        {"eval_id": "d", "manual_readable": "no", "manual_label": "unknown"},
    ]
    legible = {"a": True, "b": False, "c": True, "d": False}
    readers = {"r": {
        "a": {"number": "22", "visibility": "full"},     # correct, passes
        "b": {"number": "11", "visibility": "full"},     # invented, blocked
        "c": {"number": "1", "visibility": "partial"},   # misread, passes
        "d": {"number": None, "visibility": "none"},     # abstained
    }}
    counts = gate_report(rows, legible, readers)["r"]
    assert counts == {"scored_crops": 4, "claims": 3, "wrong_claims": 2, "claims_after_gate": 2,
                      "wrong_claims_after_gate": 1, "correct_claims": 1, "correct_claims_after_gate": 1}
