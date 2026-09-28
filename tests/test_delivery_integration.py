"""User-entry regressions for delivery failures and effective configuration."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_human_video_matting import cli
from cv_human_video_matting.io.video import probe_video


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(np.full((48, 64, 3), 90, np.uint8))
    finally:
        writer.release()


def test_cli_effective_config_and_export_report(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "result.mp4"
    make_video(source)
    config = tmp_path / "config.yaml"
    config.write_text("matting:\n  model: dummy\ndevice:\n  mode: cpu\ninput:\n  resize_short: 24\n"
                      "render:\n  background_color: '#336699'\n  preserve_audio: false\n", encoding="utf-8")
    report = tmp_path / "run.json"
    cli.main(["--config", str(config), "matte", "--input", str(source), "--output", str(target),
              "--report", str(report), "--max-frames", "2"])
    info = probe_video(target)
    assert (info.width, info.height, info.frame_count) == (32, 24, 2)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["options"]["background_color"] == "#336699"
    assert data["options"]["preserve_audio"] is False
    assert data["result"]["export_seconds"] > 0
    assert Path(data["result"]["output_path"]) == target
    assert data["result"]["total_seconds"] >= data["result"]["elapsed_seconds"] + data["result"]["export_seconds"]
    assert not list(tmp_path.glob(".matting-*"))


def test_cli_keeps_existing_delivery_on_model_failure(tmp_path, monkeypatch):
    source, target = tmp_path / "source.mp4", tmp_path / "result.mp4"
    make_video(source)
    target.write_bytes(b"previous-good-result")
    class FailingRunner:
        def predict(self, frame):
            raise RuntimeError("inference failed")
    monkeypatch.setattr(cli, "create_matting_runner", lambda **kwargs: FailingRunner())
    with pytest.raises(RuntimeError, match="inference failed"):
        cli.main(["matte", "--model", "dummy", "--input", str(source), "--output", str(target)])
    assert target.read_bytes() == b"previous-good-result"
    assert not list(tmp_path.glob(".matting-*"))


def test_ui_saves_effective_parameters_and_delivery_report(tmp_path, monkeypatch):
    from cv_human_video_matting.ui import gradio_app as ui
    source = tmp_path / "source.mp4"
    make_video(source)
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path / "ui-results")
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {"input": {"resize_short": 24},
                                            "render": {"preserve_audio": False, "background_color": "#FF0000"}})
    result, _ = ui.run_demo(str(source), None, "dummy", "green", 2, "cpu")
    info = probe_video(result)
    assert (info.width, info.height, info.frame_count) == (32, 24, 2)
    report = json.loads(Path(result).with_suffix(".json").read_text(encoding="utf-8"))
    assert report["options"]["background_color"] == "#FF0000"
    assert report["options"]["preserve_audio"] is False
    assert report["result"]["export_seconds"] > 0


def test_cli_rejects_report_overwriting_transparent_frame_before_processing(tmp_path):
    source, target = tmp_path / "source.mp4", tmp_path / "result.mp4"
    make_video(source)
    frame_dir = tmp_path / "result_frames"
    with pytest.raises(SystemExit) as error:
        cli.main(["matte", "--model", "dummy", "--mode", "transparent", "--input", str(source),
                  "--output", str(target), "--report", str(frame_dir / "frame_000000.png")])
    assert error.value.code == 2
    assert not frame_dir.exists()
