"""Shared input preflight, truncated decode and lossless intermediate regressions."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from cv_human_video_matting import pipeline
from cv_human_video_matting.matting.base import MattingResult
from cv_human_video_matting.matting.dummy import DummyMattingRunner


@pytest.fixture
def source_video(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (32, 24))
    assert writer.isOpened()
    try:
        for value in (30, 80, 140):
            writer.write(np.full((24, 32, 3), value, np.uint8))
    finally:
        writer.release()
    return path


def test_preflight_returns_prepared_input_without_creating_output(source_video, tmp_path):
    background = tmp_path / "background.png"
    expected = np.full((16, 20, 3), (12, 70, 160), np.uint8)
    assert cv2.imwrite(str(background), expected)
    output = tmp_path / "new-directory" / "result.mp4"

    prepared = pipeline.validate_video_request(
        source_video, output, "image", background_path=background, max_frames=2,
    )

    assert prepared.info.frame_count == 3
    assert prepared.info.path == source_video
    assert prepared.output_path == output
    assert prepared.total_frames == 2
    np.testing.assert_array_equal(prepared.background_bgr, expected)
    assert not output.parent.exists()


def test_preflight_resolves_transparent_output_and_preserves_existing_frames(source_video, tmp_path):
    output = tmp_path / "out.mp4"
    prepared = pipeline.validate_video_request(source_video, output, "transparent")
    assert prepared.output_path == tmp_path / "out_frames"
    assert not prepared.output_path.exists()
    prepared.output_path.mkdir()
    marker = prepared.output_path / "frame_000000.png"
    marker.write_bytes(b"keep existing frame")
    with pytest.raises(ValueError, match="already exists"):
        pipeline.validate_video_request(source_video, output, "transparent")
    assert marker.read_bytes() == b"keep existing frame"


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"max_frames": 0}, {"max_frames": True}, {"edge_feather": -1},
    {"mask_close": 1.5}, {"temporal_blend": float("nan")}, {"resize_short": 1},
    {"background_color_bgr": (0, 256, 0)}, {"depth_interval": 0}, {"depth_model": "unknown"},
    {"depth_mid": 0.8, "depth_near": 0.2}, {"enable_depth": "false"},
    {"temporal_smooth": "false"}, {"temporal_mode": "unknown"},
    {"temporal_motion_threshold": 0}, {"temporal_motion_threshold": float("inf")},
])
def test_bad_options_fail_before_opening_input(tmp_path, monkeypatch, kwargs):
    def forbidden_probe(*args):
        raise AssertionError("Validate request options before opening the video")

    monkeypatch.setattr(pipeline, "probe_video", forbidden_probe)
    values = {"mode": "green", **kwargs}
    with pytest.raises(ValueError):
        pipeline.validate_video_request(tmp_path / "missing.mp4", tmp_path / "out.mp4", **values)
    assert not (tmp_path / "out.mp4").exists()


def test_missing_input_and_background_are_rejected(source_video, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pipeline.validate_video_request(tmp_path / "missing.mp4", tmp_path / "out.mp4", "green")
    with pytest.raises(ValueError, match="requires a background"):
        pipeline.validate_video_request(source_video, tmp_path / "out.mp4", "image")
    with pytest.raises(ValueError, match="Failed to read background"):
        pipeline.validate_video_request(
            source_video, tmp_path / "out.mp4", "image", background_path=tmp_path / "missing.png",
        )


def test_invalid_destination_rejected_before_input_probe(source_video, tmp_path, monkeypatch):
    def forbidden_probe(*args):
        raise AssertionError("Reject colliding destination before opening video")

    monkeypatch.setattr(pipeline, "probe_video", forbidden_probe)
    with pytest.raises(ValueError, match="different"):
        pipeline.validate_video_request(source_video, source_video, "green")
    with pytest.raises(ValueError, match="not a file"):
        pipeline.validate_video_request(source_video, tmp_path, "green")
    with pytest.raises(ValueError, match="parent is not a directory"):
        pipeline.validate_video_request(source_video, source_video / "out.mp4", "green")


def _metadata(monkeypatch, source, frame_count):
    info = pipeline.probe_video(source)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: replace(info, frame_count=frame_count))


@pytest.mark.parametrize("mode", ["green", "transparent"])
def test_known_frame_count_premature_eof_is_failure(source_video, tmp_path, monkeypatch, mode):
    _metadata(monkeypatch, source_video, frame_count=10)
    with pytest.raises(ValueError, match="decoding ended early: decoded 3 of 10"):
        pipeline.process_video(source_video, tmp_path / "out.mp4", DummyMattingRunner(), mode)
    assert not (tmp_path / "out_frames").exists()
    assert not list(tmp_path.glob(".out_frames-*"))


@pytest.mark.parametrize("count,limit,expected", [(10, 2, 2), (3, 8, 3), (0, None, 3), (0, 8, 3)])
def test_intentional_limits_and_unknown_metadata_are_not_truncated_errors(
    source_video, tmp_path, monkeypatch, count, limit, expected,
):
    _metadata(monkeypatch, source_video, frame_count=count)
    stats = pipeline.process_video(
        source_video, tmp_path / "out.mp4", DummyMattingRunner(), "green", max_frames=limit,
    )
    assert stats.frames == expected


def test_zero_decodable_frames_keeps_existing_error(source_video, tmp_path, monkeypatch):
    info = pipeline.probe_video(source_video)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: info)

    class EmptyCapture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda path: EmptyCapture())
    with pytest.raises(ValueError, match="no readable frames"):
        pipeline.process_video(source_video, tmp_path / "out.mp4", DummyMattingRunner(), "green")
    assert not (tmp_path / "out.mp4").exists()


def test_pipeline_passes_adaptive_smoothing_options(source_video, tmp_path, monkeypatch):
    received = []
    real_smoother = pipeline.TemporalAlphaSmoother

    def smoother(**kwargs):
        received.append(kwargs)
        return real_smoother(**kwargs)

    monkeypatch.setattr(pipeline, "TemporalAlphaSmoother", smoother)
    stats = pipeline.process_video(
        source_video, tmp_path / "out.mp4", DummyMattingRunner(), "green",
        temporal_smooth=True, temporal_blend=0.6, temporal_mode="adaptive",
        temporal_motion_threshold=0.2,
    )
    assert stats.frames == 3
    assert received == [{"blend": 0.6, "mode": "adaptive", "motion_threshold": 0.2}]


def test_mkv_intermediate_preserves_decoded_pixels(source_video, tmp_path):
    class IdentityRunner:
        def predict(self, frame):
            return MattingResult(np.full(frame.shape[:2], 255, np.uint8), frame.copy())

    output = tmp_path / "lossless.mkv"
    stats = pipeline.process_video(source_video, output, IdentityRunner(), "green")
    source, rendered = cv2.VideoCapture(str(source_video)), cv2.VideoCapture(str(output))
    try:
        for _ in range(stats.frames):
            ok_source, source_frame = source.read()
            ok_output, output_frame = rendered.read()
            assert ok_source and ok_output
            np.testing.assert_array_equal(source_frame, output_frame)
    finally:
        source.release()
        rendered.release()
