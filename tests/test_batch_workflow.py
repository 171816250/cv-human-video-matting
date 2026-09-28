"""Session cancellation and the complete browser callback contract."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cv_human_video_matting.cli import ConsoleProgress
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.ui import gradio_app as ui
from cv_human_video_matting.ui.jobs import BatchJobs


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path / "results")
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {})
    monkeypatch.setattr(ui, "_BATCH_JOBS", BatchJobs())
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.full((48, 64, 3), 100, np.uint8))
    writer.release()
    return video


def test_jobs_are_isolated_and_cleaned_after_errors():
    jobs = BatchJobs()
    with pytest.raises(RuntimeError):
        with jobs.start("one") as one, jobs.start("two") as two:
            assert jobs.cancel("one")
            assert one.is_set() and not two.is_set()
            assert not jobs.cancel("unknown")
            with pytest.raises(ValueError):
                with jobs.start("one"):
                    pass
            raise RuntimeError("worker failed")
    assert not jobs.cancel("one")
    with jobs.start("one") as reused:
        assert not reused.is_set()


@pytest.mark.parametrize("mode", ["green", "transparent"])
def test_cancel_browser_session_cleans_partial_outputs(source, monkeypatch, mode):
    entered, release = Event(), Event()
    class SlowRunner(DummyMattingRunner):
        def predict(self, frame):
            entered.set()
            assert release.wait(5)
            return super().predict(frame)
    monkeypatch.setattr(ui, "create_matting_runner", lambda **kwargs: SlowRunner())
    request = SimpleNamespace(session_hash="owner")
    with ThreadPoolExecutor() as executor:
        future = executor.submit(ui.run_demo_for_ui, str(source), None, "dummy", mode, 0,
                                 "cpu", progress=None, request=request)
        try:
            assert entered.wait(5)
            assert "没有" in ui.cancel_batch_for_ui(SimpleNamespace(session_hash="other"))
            assert "收到停止" in ui.cancel_batch_for_ui(request)
        finally:
            release.set()
        result = future.result(timeout=10)
    assert "已停止" in result[2]
    assert not list(ui.OUTPUT_DIR.iterdir())
    assert not ui._BATCH_JOBS.cancel("owner")


def test_ui_downloads_report_and_records_export_controls(source):
    events = []
    result = ui.run_demo_for_ui(str(source), None, "dummy", "green", 0, "cpu",
        preserve_audio=False, resize_short=24, background_color="#123456",
        progress=lambda value, **kwargs: events.append((value, kwargs)),
        request=SimpleNamespace(session_hash="success"))
    assert len(result) == 4
    report = Path(result[3]["value"])
    assert result[3]["visible"] and report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["result"]["frames"] == 8
    assert data["options"]["resize_short"] == 24
    assert data["options"]["preserve_audio"] is False
    assert data["options"]["background_color"] == "#123456"
    assert any(value == (8, 8) for value, _ in events)
    assert not list(ui.OUTPUT_DIR.glob(".batch-*"))


def test_custom_output_directory_is_used(source, tmp_path, monkeypatch):
    custom = tmp_path / "custom-results"
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {"paths": {"output_dir": str(custom)}})
    output, _ = ui.run_demo(str(source), None, "dummy", "green", 1, "cpu")
    assert Path(output).parent.parent == custom
    assert Path(output).with_suffix(".json").is_file()


def test_console_progress_is_bounded_and_keeps_completion(capsys, monkeypatch):
    from cv_human_video_matting import cli
    monkeypatch.setattr(cli, "perf_counter", lambda: 10)
    progress = ConsoleProgress()
    for frame in range(101):
        progress(frame, 100)
    lines = capsys.readouterr().err.splitlines()
    assert lines == ["Processing frames: 0/100", "Processing frames: 100/100"]


@pytest.mark.parametrize("mode", ["green", "transparent"])
def test_report_failure_does_not_publish_partial_job(source, monkeypatch, mode):
    from cv_human_video_matting import reporting
    def fail_report(*args, **kwargs):
        raise OSError("report storage unavailable")
    monkeypatch.setattr(reporting, "write_run_report", fail_report)
    with pytest.raises(ui.gr.Error):
        ui.run_demo_for_ui(str(source), None, "dummy", mode, 2, "cpu", progress=None)
    assert not list(ui.OUTPUT_DIR.iterdir())


def test_empty_frame_limit_means_whole_video(source):
    result = ui.run_demo_for_ui(str(source), None, "dummy", "green", None, "cpu", progress=None)
    data = json.loads(Path(result[3]["value"]).read_text(encoding="utf-8"))
    assert data["result"]["frames"] == 8


def test_summary_cli_accepts_patterns(source, tmp_path):
    from cv_human_video_matting.cli import main
    for _ in range(2):
        ui.run_demo(str(source), None, "dummy", "green", 2, "cpu")
    html = tmp_path / "summary.html"
    main(["summarize", "--reports", str(ui.OUTPUT_DIR / "*" / "result.json"), "--output", str(html)])
    assert html.is_file() and "<!doctype html>" in html.read_text(encoding="utf-8").lower()
