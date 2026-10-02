"""Delivery settings must affect actual frames, with no network model downloads."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from cv_human_video_matting import pipeline
from cv_human_video_matting.depth import DepthEstimator
from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.matting.base import MattingResult
from cv_human_video_matting.utils.config import (
    load_config,
    parse_background_color,
    validate_project_config,
)


class RecordingRunner:
    def __init__(self):
        self.shapes = []

    def predict(self, frame):
        self.shapes.append(frame.shape)
        return MattingResult(alpha=np.zeros(frame.shape[:2], dtype=np.uint8))


def make_video(path, size=(100, 60), frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((size[1], size[0], 3), 20 + index, dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize("mode,width_factor", [("green", 1), ("side_by_side", 2)])
def test_resize_and_custom_color_reach_encoded_video(tmp_path, mode, width_factor):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source)
    runner = RecordingRunner()
    stats = pipeline.process_video(
        source, output, runner, mode=mode, resize_short=31, max_frames=3,
        background_color_bgr=parse_background_color("#204080"),
    )
    assert runner.shapes == [(30, 50, 3)] * 3
    info = probe_video(output)
    assert (info.width, info.height, stats.frames) == (50 * width_factor, 30, 3)
    capture = cv2.VideoCapture(str(output))
    try:
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
        assert len(decoded) == 3
        for frame in decoded:
            assert frame.shape == (30, 50 * width_factor, 3)
            # MP4 compression can shift channels slightly; assess the solid output region.
            actual = frame[10:20, -20:-10].mean(axis=(0, 1))
            np.testing.assert_allclose(actual, (128, 64, 32), atol=8)
    finally:
        capture.release()


@pytest.mark.parametrize("limit", [None, 120])
def test_resize_never_upscales_and_default_preserves_resolution(tmp_path, limit):
    source = tmp_path / "in.mp4"
    make_video(source)
    runner = RecordingRunner()
    stats = pipeline.process_video(source, tmp_path / "out.mp4", runner, "transparent",
                                   max_frames=2, resize_short=limit)
    assert runner.shapes == [(60, 100, 3)] * 2
    assert cv2.imread(str(stats.output_path / "frame_000001.png"), -1).shape == (60, 100, 4)


def test_depth_settings_and_interval_are_applied_to_resized_frames(tmp_path, monkeypatch):
    source = tmp_path / "in.mp4"
    make_video(source)
    settings, estimated_shapes, alpha_ranges = [], [], []

    class FakeDepth:
        def __init__(self, **kwargs):
            settings.append(kwargs)

        def estimate(self, frame):
            estimated_shapes.append(frame.shape)
            return np.full(frame.shape[:2], 0.75, np.float32)

        def person_distance(self, depth, alpha):
            assert depth.shape == alpha.shape
            alpha_ranges.append((alpha.dtype, alpha.min(), alpha.max()))
            return None, "No person detected"

    monkeypatch.setattr(pipeline, "DepthEstimator", FakeDepth)
    pipeline.process_video(
        source, tmp_path / "out.mp4", RecordingRunner(), "green", resize_short=30,
        enable_depth=True, depth_interval=2, depth_model="DPT_Hybrid",
        depth_near=0.8, depth_mid=0.2,
    )
    assert settings == [{"device": "cpu", "model_type": "DPT_Hybrid",
                         "near_threshold": 0.8, "mid_threshold": 0.2}]
    assert estimated_shapes == [(30, 50, 3)] * 3
    assert len(alpha_ranges) == 5
    assert all(dtype == np.uint8 and minimum == maximum == 0
               for dtype, minimum, maximum in alpha_ranges)


def test_local_depth_anything_checkpoint_reaches_video_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "in.mp4"
    make_video(source, frames=2)
    checkpoint = tmp_path / "depth-anything"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"depth_anything","depth_estimation_type":"relative",'
        '"backbone":null,"backbone_config":{"model_type":"dinov2",'
        '"hidden_size":384,"patch_size":14}}',
        encoding="utf-8",
    )
    (checkpoint / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"local-test-weights")
    loaded = []

    class FakeDepth:
        def __init__(self, **kwargs):
            loaded.append(kwargs)

        def estimate(self, frame):
            return np.full(frame.shape[:2], 0.75, np.float32)

        def person_distance(self, depth, alpha):
            return 0.25, "Near"

    monkeypatch.setattr(pipeline, "DepthEstimator", FakeDepth)
    stats = pipeline.process_video(
        source, tmp_path / "out.mp4", RecordingRunner(), "green",
        enable_depth=True, depth_model="DepthAnythingV2_small",
        depth_checkpoint=checkpoint,
    )
    assert stats.frames == 2
    assert loaded[0]["checkpoint_path"] == checkpoint
    assert loaded[0]["model_type"] == "DepthAnythingV2_small"


def test_depth_anything_preflight_rejects_missing_model_before_output(tmp_path):
    output = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError, match="No model is downloaded"):
        pipeline.validate_video_request(
            tmp_path / "missing-input.mp4", output, "green", enable_depth=True,
            depth_model="DepthAnythingV2_small",
            depth_checkpoint=tmp_path / "missing-depth-model",
        )
    assert not output.exists()


def test_live_frame_uses_custom_color_with_uint8_alpha():
    frame = np.full((10, 10, 3), 200, dtype=np.uint8)
    output, _ = pipeline.render_matting_frame(
        frame, RecordingRunner(), "green", background_color_bgr=(15, 40, 250),
    )
    np.testing.assert_array_equal(output, np.full_like(frame, (15, 40, 250)))


def test_depth_thresholds_change_classification_and_normalize_uint8_mask(monkeypatch):
    loaded = []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

    def fake_load(repo, model, **kwargs):
        loaded.append(model)
        if model == "transforms":
            return SimpleNamespace(small_transform=None, dpt_transform=None)
        return Model()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    estimator = DepthEstimator(model_type="DPT_Hybrid", near_threshold=0.9, mid_threshold=0.7)
    alpha = np.zeros((20, 20), dtype=np.uint8)
    alpha[5:15, 5:15] = 255
    depth = np.full((20, 20), 0.99, dtype=np.float32)
    depth[5:15, 5:15] = 0.8
    distance, label = estimator.person_distance(depth, alpha)
    assert label == "Mid"
    assert distance == pytest.approx(0.2)
    assert loaded == ["DPT_Hybrid", "transforms"]


@pytest.mark.parametrize("kwargs", [
    {"model_type": "unknown"}, {"near_threshold": 0.3, "mid_threshold": 0.4},
    {"mid_threshold": float("nan")}, {"near_threshold": float("inf")},
    {"mid_threshold": -0.1}, {"near_threshold": True},
])
def test_invalid_depth_options_fail_before_loading_model(kwargs):
    # conftest blocks all model downloads; ValueError must be raised before hub.load.
    with pytest.raises(ValueError):
        DepthEstimator(**kwargs)


@pytest.mark.parametrize("config", [
    {"input": {"resize_short": 0}}, {"input": {"resize_short": 1.5}},
    {"input": {"resize_short": True}}, {"depth": {"interval": 0}},
    {"depth": {"interval": "5"}}, {"depth": {"enabled": "false"}},
    {"depth": {"model": "unknown"}}, {"depth": {"thresholds": {"near": 0.2, "mid": 0.3}}},
    {"depth": {"thresholds": {"near": float("nan")}}},
    {"postprocess": {"temporal_alpha": 2}}, {"postprocess": {"mask_open": -1}},
    {"postprocess": {"edge_feather": 1.5}}, {"matting": {"downsample_ratio": 0}},
    {"matting": {"downsample_ratio": "typo"}}, {"render": {"mode": "invisible"}},
    {"render": {"background_color": "#GGHHII"}}, {"depth": []},
    {"depth": {"thresholds": None}},
])
def test_explicit_invalid_config_is_not_silently_replaced(config):
    with pytest.raises(ValueError):
        validate_project_config(config)


@pytest.mark.parametrize("text", ["false", "[]", "0", "depth:\n  interval: 0\n"])
def test_load_config_rejects_invalid_root_or_values(tmp_path, text):
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_shipped_config_loads_and_color_order_is_bgr():
    config = load_config("configs/default.yaml")
    assert config["input"]["resize_short"] is None
    assert "source" not in config["input"]
    assert "label_mode" not in config["depth"]
    assert parse_background_color("#aB10Ff") == (255, 16, 171)
    validate_project_config({"device": {"mode": "cuda:0"}})


@pytest.mark.parametrize("kwargs", [
    {"resize_short": 0}, {"background_color_bgr": (1, 2, 256)},
    {"enable_depth": True, "depth_interval": 0}, {"depth_mid": 0.9, "depth_near": 0.4},
    {"max_frames": 1.5}, {"max_frames": True},
])
def test_bad_pipeline_options_fail_without_touching_existing_output(tmp_path, kwargs):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep existing output")
    with pytest.raises(ValueError):
        pipeline.process_video(tmp_path / "not-opened.mp4", output, RecordingRunner(),
                               "green", **kwargs)
    assert output.read_bytes() == b"keep existing output"
