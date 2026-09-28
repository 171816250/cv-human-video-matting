"""Adaptive controls, model asset selection and preflight at the UI boundary."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_human_video_matting import reporting
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.ui import gradio_app as ui


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path / "results")
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {})
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.full((24, 32, 3), 100, np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("problem", [
    "missing", "corrupt", "background", "edge_feather", "mask_open", "temporal_mode",
    "temporal_motion_threshold", "nonfinite_threshold", "temporal_blend", "depth_interval",
])
def test_invalid_request_fails_before_model_loading(source, monkeypatch, problem):
    def forbidden_model(**kwargs):
        pytest.fail("An invalid request must not load or download a model")
    monkeypatch.setattr(ui, "create_matting_runner", forbidden_model)
    path, background, mode = source, None, "green"
    settings = {}
    if problem == "missing":
        path = source.with_name("missing.mp4")
    elif problem == "corrupt":
        path.write_bytes(b"not a video")
    elif problem == "background":
        mode = "image"
        background = source.with_name("invalid.png")
        background.write_bytes(b"not an image")
    else:
        settings = {
            "edge_feather": {"edge_feather": -1},
            "mask_open": {"mask_open": 1.5},
            "temporal_mode": {"temporal_mode": "unknown"},
            "temporal_motion_threshold": {"temporal_motion_threshold": 0},
            "nonfinite_threshold": {"temporal_motion_threshold": float("nan")},
            "temporal_blend": {"temporal_blend": 1.1},
            "depth_interval": {"depth_interval": 0},
        }[problem]
    with pytest.raises(ui.gr.Error):
        ui.run_demo_for_ui(str(path), str(background) if background else None,
                           "rvm_mobilenetv3", mode, 0, "cpu", progress=None, **settings)
    assert not ui.OUTPUT_DIR.exists() or not list(ui.OUTPUT_DIR.iterdir())


@pytest.mark.parametrize("mode,codec", [("green", "FFV1"), ("transparent", "PNG")])
def test_adaptive_config_is_used_and_reported(source, monkeypatch, mode, codec):
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {"postprocess": {
        "temporal_mode": "adaptive", "temporal_motion_threshold": 0.23}})
    result = ui._run_batch_demo(str(source), None, "dummy", mode, 2, "cpu",
                                temporal_smooth=True)
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["options"]["temporal_mode"] == "adaptive"
    assert report["options"]["temporal_motion_threshold"] == 0.23
    assert report["options"]["temporal_smooth"] is True
    assert report["options"]["intermediate_codec"] == codec
    assert not list(ui.OUTPUT_DIR.rglob("*.mkv"))
    assert not list(ui.OUTPUT_DIR.glob(".batch-*"))


@pytest.mark.parametrize("selected,expected", [("rvm_mobilenetv3", True), ("rvm_resnet50", False)])
def test_batch_bundle_selection_and_loaded_identity_reach_report(source, tmp_path, monkeypatch,
                                                                 selected, expected):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    identity = {"content_sha256": "fixture", "model_name": "mobilenetv3"}
    received = []
    def factory(**kwargs):
        received.append(kwargs)
        runner = DummyMattingRunner()
        runner.bundle_info = identity if expected else None
        return runner
    report_calls = []
    def write_report(path, input_path, stats, options, total_seconds, **kwargs):
        report_calls.append((options, kwargs))
        Path(path).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ui, "create_matting_runner", factory)
    monkeypatch.setattr(reporting, "write_run_report", write_report)
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {"matting": {
        "model": "rvm_mobilenetv3", "bundle": str(bundle), "downsample_ratio": 0.5}})
    ui._run_batch_demo(str(source), None, selected, "green", 1, "cpu")
    assert received[0]["bundle_path"] == (str(bundle) if expected else None)
    assert received[0]["checkpoint"] is None
    assert received[0]["downsample_ratio"] == 0.5
    assert report_calls[0][0]["model_bundle"] == (str(bundle) if expected else None)
    assert report_calls[0][1]["model_bundle_info"] == (identity if expected else None)


def test_ui_output_cannot_be_inside_model_bundle(source, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output = bundle / "results"
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {"paths": {"output_dir": str(output)},
        "matting": {"model": "rvm_mobilenetv3", "bundle": str(bundle)}})
    monkeypatch.setattr(ui, "create_matting_runner", lambda **kwargs: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="outside"):
        ui._run_batch_demo(str(source), None, "rvm_mobilenetv3", "green", 0, "cpu")
    assert not output.exists()
    assert not list(bundle.iterdir())


@pytest.mark.parametrize("setting", ["checkpoint", "bundle", "downsample_ratio", "temporal_mode",
                                     "temporal_motion_threshold", "temporal_blend", "temporal_smooth"])
def test_live_cache_changes_with_every_model_and_temporal_setting(monkeypatch, setting):
    config = {"matting": {"model": "rvm_mobilenetv3", "downsample_ratio": "auto"}}
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", config)
    created = []
    def factory(**kwargs):
        created.append(kwargs)
        return DummyMattingRunner()
    monkeypatch.setattr(ui, "create_matting_runner", factory)
    cache = ui.LiveRunnerCache()
    args = dict(model="rvm_mobilenetv3", device="cpu", temporal_smooth=True,
                temporal_blend=0.75, cache=cache, temporal_mode="ema",
                temporal_motion_threshold=0.15)
    ui._live_cache_for(**args)
    runner, smoother = cache.runner, cache.temporal_smoother
    ui._live_cache_for(**args)
    assert cache.runner is runner and cache.temporal_smoother is smoother
    if setting in ("checkpoint", "bundle", "downsample_ratio"):
        config["matting"][setting] = 0.5 if setting == "downsample_ratio" else "changed-path"
    else:
        args[setting] = {"temporal_mode": "adaptive", "temporal_motion_threshold": 0.25,
                         "temporal_blend": 0.7501, "temporal_smooth": False}[setting]
    ui._live_cache_for(**args)
    assert cache.runner is not runner and cache.temporal_smoother is not smoother
    assert len(created) == 2
    assert cache.frames == 0


def test_live_bundle_is_only_forwarded_to_configured_model(monkeypatch):
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {"matting": {
        "model": "rvm_mobilenetv3", "bundle": "mobile-bundle"}})
    received = []
    def factory(**kwargs):
        received.append(kwargs)
        return DummyMattingRunner()
    monkeypatch.setattr(ui, "create_matting_runner", factory)
    ui._live_cache_for("rvm_mobilenetv3", "cpu", False, 0.75, ui.LiveRunnerCache())
    ui._live_cache_for("dummy", "cpu", False, 0.75, ui.LiveRunnerCache())
    assert [item["bundle_path"] for item in received] == ["mobile-bundle", None]


def test_new_live_controls_preserve_legacy_cache_position(monkeypatch):
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {})
    frame = np.full((24, 32, 3), 100, np.uint8)
    old_controls = (frame, None, "dummy", "green", "cpu", 32, 0, 0, 0, True, 0.75, False, 5)
    legacy, adaptive = ui.LiveRunnerCache(), ui.LiveRunnerCache()
    ui.render_live_frame(*old_controls, legacy)
    output, _, returned_cache = ui.render_live_frame_for_ui(*old_controls, "adaptive", 0.23, adaptive)
    assert output.shape == frame.shape and returned_cache is adaptive
    assert legacy.temporal_smoother.mode == "ema"
    assert adaptive.temporal_smoother.mode == "adaptive"
    assert adaptive.temporal_smoother.motion_threshold == 0.23


def test_temporal_controls_read_config_and_default_smoothing_off(monkeypatch):
    monkeypatch.setattr(ui, "_ACTIVE_CONFIG", {})
    demo = ui.build_demo({"postprocess": {"temporal_mode": "adaptive", "temporal_motion_threshold": 0.23}})
    try:
        controls = [item["props"] for item in demo.config["components"]]
        assert [item["value"] for item in controls if item.get("label") == "平滑方式"] == ["adaptive"] * 2
        assert [item["value"] for item in controls if item.get("label") == "自适应变化阈值"] == [0.23] * 2
        assert [item["value"] for item in controls if item.get("label") == "启用时间平滑"] == [False] * 2
    finally:
        demo.close()
