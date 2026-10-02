"""The rebuild runs stages in order, resumes from finished ones, and never names players unconfirmed."""

import sys

import pytest

from conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import rebuild_nll_test4 as rebuild  # noqa: E402


def touch_stage(tmp_path, name, output, **extra):
    """A stage whose command writes its output file and logs that it ran."""
    log = tmp_path / "ran.log"
    code = "import pathlib; pathlib.Path(r'{}').parent.mkdir(parents=True, exist_ok=True); pathlib.Path(r'{}').write_text('x'); open(r'{}', 'a').write('{}\\n')".format(
        tmp_path / output, tmp_path / output, log, name
    )
    return {"name": name, "why": name, "output": output, "cmd": [sys.executable, "-c", code], **extra}


def fail_stage(name, output):
    return {"name": name, "why": name, "output": output, "cmd": [sys.executable, "-c", "raise SystemExit(3)"]}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rebuild, "preflight", lambda args: (None, []))
    monkeypatch.setattr(rebuild, "team_confirmation_status", lambda: (False, "missing"))

    def run(plan, *argv):
        monkeypatch.setattr(rebuild, "stages", lambda args: plan)
        monkeypatch.setattr(sys, "argv", ["rebuild_nll_test4.py", *argv])
        return rebuild.main()

    def ran():
        log = tmp_path / "ran.log"
        return log.read_text().split() if log.exists() else []

    return run, ran


def test_runs_stages_in_order_and_skips_finished_ones(tmp_path, harness):
    run, ran = harness
    plan = [touch_stage(tmp_path, "a", "out/a.json"), touch_stage(tmp_path, "b", "out/b.json")]

    assert run(plan) == 0
    assert ran() == ["a", "b"]

    assert run(plan) == 0
    assert ran() == ["a", "b"], "finished stages must not run again"

    assert run(plan, "--force", "b") == 0
    assert ran() == ["a", "b", "b"]


def test_failure_stops_the_run_and_resume_continues_from_it(tmp_path, harness):
    run, ran = harness
    good_a = touch_stage(tmp_path, "a", "out/a.json")

    assert run([good_a, fail_stage("b", "out/b.json"), touch_stage(tmp_path, "c", "out/c.json")]) == 3
    assert ran() == ["a"]

    assert run([good_a, touch_stage(tmp_path, "b", "out/b.json"), touch_stage(tmp_path, "c", "out/c.json")]) == 0
    assert ran() == ["a", "b", "c"]


def test_stage_that_exits_cleanly_without_output_is_a_failure(tmp_path, harness):
    run, _ran = harness
    silent = {"name": "a", "why": "a", "output": "out/a.json", "cmd": [sys.executable, "-c", "pass"]}

    assert run([silent]) == 1


def test_stops_before_naming_players_until_teams_are_confirmed(tmp_path, harness, monkeypatch, capsys):
    run, ran = harness
    plan = [touch_stage(tmp_path, "teams", "out/teams.json"), touch_stage(tmp_path, "jersey", "out/jersey.json", needs_team_confirmation=True)]

    assert run(plan) == 2
    assert ran() == ["teams"]
    assert "confirm_team_mapping.py" in capsys.readouterr().out

    monkeypatch.setattr(rebuild, "team_confirmation_status", lambda: (True, "confirmed"))
    assert run(plan) == 0
    assert ran() == ["teams", "jersey"]


def test_stop_after_and_unknown_stage_names(tmp_path, harness):
    run, ran = harness
    plan = [touch_stage(tmp_path, "a", "out/a.json"), touch_stage(tmp_path, "b", "out/b.json")]

    assert run(plan, "--stop-after", "a") == 0
    assert ran() == ["a"]
    with pytest.raises(SystemExit, match="Unknown stage"):
        run(plan, "--force", "typo")


def test_stage_warnings_are_read_from_the_stage_report(tmp_path, harness):
    (tmp_path / "report.json").write_text('{"warnings": ["points unverified"]}')
    assert rebuild.stage_warnings({"warnings_from": "report.json"}) == ["points unverified"]
    assert rebuild.stage_warnings({"warnings_from": "missing.json"}) == []
    assert rebuild.stage_warnings({}) == []


def test_real_plan_puts_team_confirmation_before_the_only_naming_stage(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rebuild_nll_test4.py"])
    plan = rebuild.stages(rebuild.parse_args())
    gated = [stage["name"] for stage in plan if stage.get("needs_team_confirmation")]
    assert gated == ["jersey"]
    assert plan[-1]["name"] == "jersey"
    jersey_inputs = " ".join(plan[-1]["cmd"])
    assert "--allow-network-vision" in jersey_inputs and "localhost:11434" in jersey_inputs


class TestCalibrationPointCheck:
    @pytest.fixture(autouse=True)
    def module(self):
        pytest.importorskip("cv2")
        import run_field_calibration_smoke

        self.check = run_field_calibration_smoke.check_point_frame_size

    POINTS = [{"name": "a", "video_xy": [31.75, 360.7]}, {"name": "b", "video_xy": [1265.25, 633.7]}]

    def test_points_that_fit_but_unrecorded_size_warn(self):
        [warning] = self.check({}, self.POINTS, {"width": 1280, "height": 720})
        assert '"video_frame_size": [1280, 720]' in warning

    def test_recorded_matching_size_is_silent(self):
        assert self.check({"video_frame_size": [1280, 720]}, self.POINTS, {"width": 1280, "height": 720}) == []

    def test_recorded_size_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="picked on a 1280x720 frame but the video decodes at 1920x1080"):
            self.check({"video_frame_size": [1280, 720]}, self.POINTS, {"width": 1920, "height": 1080})

    def test_points_outside_a_smaller_frame_are_an_error(self):
        with pytest.raises(ValueError, match="outside the 960x540 decoded frame \\(b\\)"):
            self.check({}, self.POINTS, {"width": 960, "height": 540})
