"""Failed vision reads are counted and reported instead of looking like illegible jerseys."""

import json
import sys

from conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_track_level_jersey_inference as runner  # noqa: E402

TIMEOUT = {"status": "error", "error": "TimeoutError: timed out"}
REFUSED = {"status": "error", "error": "URLError: connection refused"}
READ = {"status": "complete", "parsed": {"number": "22", "visibility": "full"}}


def per_frame(*frames):
    return {"vision_model_result": {"status": "complete", "mode": "per_frame", "frames": list(frames)}}


def test_counts_requests_errors_timeouts_and_missing_views():
    predictions = [
        per_frame(READ, TIMEOUT, {"status": "missing_view"}),
        per_frame(REFUSED),
        {"vision_model_result": {"status": "error", "mode": "track", "error": "TimeoutError: timed out"}},
        {"vision_model_result": {"status": "not_run"}},
        {},
    ]
    assert runner.vision_request_counts(predictions) == {"requests": 4, "errors": 3, "timeouts": 2, "missing_views": 1}


def run_main(tmp_path, monkeypatch, predictions, *argv):
    seen = {}

    def fake_run(config):
        seen["config"] = config
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        (out / "track_jersey_predictions.json").write_text(json.dumps({"tracks": predictions}))
        (out / "track_jersey_summary.json").write_text(json.dumps({"counts": {}}))
        return {
            "counts": {"tracks_processed": len(predictions)},
            "team_mapping": {"status": "confirmed", "mapping": {"team_a": "TOR"}, "confirmed_by": "t"},
            "outputs": {
                "track_jersey_predictions": str(out / "track_jersey_predictions.json"),
                "track_jersey_summary": str(out / "track_jersey_summary.json"),
            },
        }

    monkeypatch.setattr(runner, "run_track_level_inference", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_track_level_jersey_inference.py", *argv])
    code = runner.main()
    stored = json.loads((tmp_path / "out" / "track_jersey_summary.json").read_text())
    return code, seen["config"], stored


def test_every_request_failing_is_a_failed_run(tmp_path, monkeypatch, capsys):
    code, _config, stored = run_main(tmp_path, monkeypatch, [per_frame(TIMEOUT, TIMEOUT)])
    assert code == 3
    assert stored["vision_requests"]["timeouts"] == 2
    assert "2 of 2 vision requests failed (2 timed out" in capsys.readouterr().out


def test_some_failures_warn_but_succeed(tmp_path, monkeypatch, capsys):
    code, _config, _stored = run_main(tmp_path, monkeypatch, [per_frame(READ, TIMEOUT)])
    assert code == 0
    assert "WARNING: 1 of 2 vision requests failed" in capsys.readouterr().out


def test_timeout_and_team_assignment_overrides_reach_the_stage(tmp_path, monkeypatch):
    _code, config, _stored = run_main(
        tmp_path, monkeypatch, [per_frame(READ)], "--vision-timeout", "600", "--team-assignments", "teams/colour.json"
    )
    assert config["vision_backend"]["timeout_seconds"] == 600.0
    assert config["inputs"]["team_assignments"] == "teams/colour.json"
