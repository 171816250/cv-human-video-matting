"""Regression cases for the incubation project's reproducibility requirements."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_human_video_matting.cli import main
from cv_human_video_matting.depth import DepthEstimator, draw_distance_label
from cv_human_video_matting.matting.base import MattingResult
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.pipeline import process_video, render_matting_frame


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.full((24, 32, 3), 60, np.uint8))
    finally:
        writer.release()


def test_distance_uint8_mask_excludes_faint_background():
    estimator = object.__new__(DepthEstimator)
    alpha = np.full((40, 40), 10, np.uint8)
    alpha[10:20, 10:20] = 255
    depth = np.full((40, 40), 0.1, np.float32)
    depth[10:20, 10:20] = 0.9
    distance, label = estimator.person_distance(depth, alpha)
    assert label == "Near"
    assert distance == pytest.approx(0.1)
    frame = np.zeros((40, 40, 3), np.uint8)
    assert np.array_equal(
        draw_distance_label(frame, alpha, distance, label),
        draw_distance_label(frame, alpha.astype(np.float32) / 255, distance, label),
    )


def test_distance_nonfinite_depth_is_not_classified():
    estimator = object.__new__(DepthEstimator)
    distance, label = estimator.person_distance(
        np.full((20, 20), np.nan), np.ones((20, 20), np.float32)
    )
    assert distance is None
    assert label == "No person detected"


@pytest.mark.parametrize("mode", ["green", "foreground", "image", "blur", "transparent", "side_by_side"])
def test_composition_uses_model_foreground(mode):
    class Runner:
        def predict(self, frame):
            return MattingResult(np.ones(frame.shape[:2], np.float32), np.full_like(frame, 200))
    frame = np.full((16, 20, 3), 50, np.uint8)
    rendered, _ = render_matting_frame(frame, Runner(), mode, background_bgr=frame)
    if mode == "side_by_side":
        assert np.all(rendered[:, :20] == 50)
        rendered = rendered[:, 20:]
    assert np.all(rendered[:, :, :3] == 200)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_leaves_output_untouched(tmp_path, limit):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source)
    output.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="max_frames"):
        process_video(source, output, DummyMattingRunner(), "green", max_frames=limit)
    assert output.read_bytes() == b"existing output"


def test_same_input_output_is_rejected(tmp_path):
    source = tmp_path / "in.mp4"
    make_video(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="different"):
        process_video(source, source, DummyMattingRunner(), "green")
    assert source.read_bytes() == original


def test_capture_is_released_on_first_prediction_failure(tmp_path, monkeypatch):
    from cv_human_video_matting import pipeline
    source = tmp_path / "in.mp4"
    make_video(source)
    real_capture = cv2.VideoCapture
    captures = []
    def tracked_capture(*args):
        capture = real_capture(*args)
        captures.append(capture)
        return capture
    class BrokenRunner:
        def predict(self, frame):
            raise RuntimeError("inference failed")
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", tracked_capture)
    with pytest.raises(RuntimeError, match="inference failed"):
        process_video(source, tmp_path / "out.mp4", BrokenRunner(), "green")
    assert captures and all(not capture.isOpened() for capture in captures)


def test_transparent_rerun_preserves_existing_frames(tmp_path):
    source = tmp_path / "in.mp4"
    make_video(source)
    output = tmp_path / "out.mp4"
    stats = process_video(source, output, DummyMattingRunner(), "transparent")
    first = stats.output_path / "frame_000000.png"
    previous = first.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        process_video(source, output, DummyMattingRunner(), "transparent")
    assert first.read_bytes() == previous


def test_cli_report_contains_reproducible_conditions(tmp_path):
    source = tmp_path / "in.mp4"
    make_video(source)
    report = tmp_path / "report.json"
    main(["matte", "--model", "dummy", "--device", "cpu", "--input", str(source),
          "--output", str(tmp_path / "out.mp4"), "--max-frames", "2", "--report", str(report)])
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["input"]["width"] == 32
    assert data["options"]["model"] == "dummy"
    assert data["result"]["frames"] == 2
    assert data["result"]["total_seconds"] >= data["result"]["elapsed_seconds"] > 0
    assert data["quality_metrics"] is None


def test_switching_model_does_not_reuse_default_checkpoint(tmp_path, monkeypatch):
    from cv_human_video_matting import cli
    source = tmp_path / "in.mp4"
    make_video(source)
    config = tmp_path / "config.yaml"
    config.write_text("matting:\n  model: rvm_mobilenetv3\n  checkpoint: wrong.pth\n")
    received = []
    def factory(**kwargs):
        received.append(kwargs)
        return DummyMattingRunner()
    monkeypatch.setattr(cli, "create_matting_runner", factory)
    main(["--config", str(config), "matte", "--model", "rvm_resnet50", "--device", "cpu",
          "--input", str(source), "--output", str(tmp_path / "out.mp4")])
    assert received[0]["checkpoint"] is None


def test_live_sessions_keep_independent_models_and_depth_intervals(monkeypatch):
    from cv_human_video_matting.ui import gradio_app as ui
    calls = []
    class FakeDepth:
        def __init__(self, device, **kwargs):
            self.counter = 0
        def estimate(self, frame):
            self.counter += 1
            return np.ones(frame.shape[:2], np.float32)
        def person_distance(self, depth, alpha):
            return 0.0, "Near"
    monkeypatch.setattr(ui, "DepthEstimator", FakeDepth)
    first, second = ui.LiveRunnerCache(), ui.LiveRunnerCache()
    frame = np.zeros((48, 64, 3), np.uint8)
    def render(cache, image=frame):
        return ui.render_live_frame(image, None, "dummy", "green", "cpu", 64,
                                    temporal_smooth=True, enable_depth=True,
                                    depth_interval=2, cache=cache)
    for _ in range(3):
        render(first)
    render(second)
    assert first.runner is not second.runner
    assert first.depth_estimator.counter == 2
    assert second.depth_estimator.counter == 1
    assert (first.frames, second.frames) == (3, 1)
    output, _ = render(first, np.zeros((32, 40, 3), np.uint8))
    assert output.shape == (32, 40, 3)
    assert first.depth_map.shape == (32, 40)
