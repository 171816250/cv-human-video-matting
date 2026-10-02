"""Submission validation, lossless intermediates, and bundle report identity."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_human_video_matting import cli, reporting
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.pipeline import process_video
from cv_human_video_matting.utils.config import validate_project_config


def make_video(path):
    frames = np.random.default_rng(28).integers(0, 256, (4, 48, 64, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (64, 48))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return frames


@pytest.mark.parametrize("args", [
    ["--max-frames", "0"], ["--edge-feather", "-1"], ["--mask-open", "-1"],
    ["--temporal-blend", "nan"], ["--temporal-motion-threshold", "0"],
    ["--downsample-ratio", "nan"], ["--downsample-ratio", "0"],
    ["--mode", "image", "--background", "missing-background.png"],
    ["--checkpoint", "missing-weights.pth"],
])
def test_cli_rejects_invalid_submission_before_loading_model(tmp_path, monkeypatch, args):
    source = tmp_path / "source.mkv"
    make_video(source)
    called = []
    monkeypatch.setattr(cli, "create_matting_runner", lambda **kwargs: called.append(kwargs))
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.main(["matte", "--device", "cpu", "--input", str(source),
                  "--output", str(tmp_path / "out.mp4"), *args])
    assert not called
    assert not (tmp_path / "out.mp4").exists()
    assert not list(tmp_path.glob(".matting-*"))


def test_cli_missing_video_is_rejected_before_model_load(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "create_matting_runner", lambda **kwargs: called.append(kwargs))
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.main(["matte", "--device", "cpu", "--input", str(tmp_path / "missing.mp4"),
                  "--output", str(tmp_path / "out.mp4")])
    assert not called


def test_pipeline_ffv1_preserves_rendered_pixels_exactly(tmp_path):
    from cv_human_video_matting.matting.base import MattingResult

    class OpaqueRunner:
        def predict(self, frame):
            return MattingResult(alpha=np.full(frame.shape[:2], 255, np.uint8), foreground=frame)

    source, output = tmp_path / "source.mkv", tmp_path / "out.mkv"
    frames = make_video(source)
    stats = process_video(source, output, OpaqueRunner(), "foreground")
    capture = cv2.VideoCapture(str(output))
    try:
        for frame in frames:
            success, decoded = capture.read()
            assert success
            np.testing.assert_array_equal(decoded, frame)
        assert not capture.read()[0]
    finally:
        capture.release()
    assert stats.frames == 4


def test_cli_records_adaptive_settings_and_lossless_intermediate(tmp_path):
    source = tmp_path / "source.mkv"
    make_video(source)
    report = tmp_path / "run.json"
    cli.main(["matte", "--model", "dummy", "--device", "cpu", "--input", str(source),
              "--output", str(tmp_path / "out.mp4"), "--report", str(report),
              "--temporal-smooth", "--temporal-mode", "adaptive",
              "--temporal-motion-threshold", "0.2", "--downsample-ratio", "0.5"])
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["options"]["intermediate_codec"] == "FFV1"
    assert data["options"]["temporal_mode"] == "adaptive"
    assert data["options"]["temporal_motion_threshold"] == 0.2
    assert data["options"]["downsample_ratio"] == 0.5
    assert data["output"]["frame_count"] == 4


@pytest.mark.parametrize("config", [
    {"postprocess": {"temporal_mode": "unknown"}},
    {"postprocess": {"temporal_motion_threshold": 0}},
    {"postprocess": {"temporal_motion_threshold": float("nan")}},
    {"matting": {"checkpoint": "a.pth", "bundle": "bundle"}},
    {"matting": {"bundle": True}}, {"matting": {"bundle": ""}},
])
def test_invalid_new_config_fails_early(config):
    with pytest.raises(ValueError):
        validate_project_config(config)


def test_report_rejects_bundle_changed_after_loading(tmp_path, monkeypatch):
    from cv_human_video_matting.utils import model_bundle

    source = tmp_path / "source.mkv"
    make_video(source)
    stats = process_video(source, tmp_path / "out.mkv", DummyMattingRunner(), "alpha")
    monkeypatch.setattr(model_bundle, "verify_model_bundle", lambda path: {
        "content_sha256": "a" * 64, "model_name": "mobilenetv3", "weights_path": "weights/rvm.pth",
        "files": [{"path": "weights/rvm.pth", "sha256": "c" * 64, "size_bytes": 12}],
    })
    options = {"model_bundle": "bundle", "device": "cpu"}
    target = tmp_path / "report.json"
    target.write_text("previous report", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after"):
        reporting.write_run_report(target, source, stats, options, 1,
            model_bundle_info={"content_sha256": "b" * 64, "model_name": "mobilenetv3"})
    assert target.read_text(encoding="utf-8") == "previous report"
    reporting.write_run_report(target, source, stats, options, 1,
        model_bundle_info={"content_sha256": "a" * 64, "model_name": "mobilenetv3"})
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["assets"]["model_bundle"]["load_identity_confirmed"] is True
    assert data["assets"]["model_bundle"]["content_sha256"] == "a" * 64


@pytest.mark.parametrize("target", ["manifest.json", "code/hubconf.py", "weights/model.pth"])
def test_publication_paths_cannot_overwrite_model_bundle(tmp_path, target):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    destination = bundle / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"protected model asset")
    for report, output in [(destination, tmp_path / "out.mp4"), (None, destination)]:
        with pytest.raises(ValueError):
            reporting.validate_artifact_paths(report, tmp_path / "source.mp4", output, [bundle])
    assert destination.read_bytes() == b"protected model asset"


def test_publication_paths_expand_home_for_local_depth_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    bundle = tmp_path / "depth-small"
    bundle.mkdir()
    config = bundle / "config.json"
    config.write_text("protected model config", encoding="utf-8")
    assert Path("~/depth-small").expanduser() == bundle
    with pytest.raises(ValueError):
        reporting.validate_artifact_paths(
            config, tmp_path / "source.mp4", tmp_path / "out.mp4",
            asset_paths=["~/depth-small"],
        )
    assert config.read_text(encoding="utf-8") == "protected model config"


def test_unknown_source_metadata_is_explicit_and_excluded_from_strict_groups(tmp_path, monkeypatch):
    from dataclasses import replace
    from cv_human_video_matting import evaluation
    from cv_human_video_matting.io import video

    source = tmp_path / "source.mkv"
    make_video(source)
    stats = process_video(source, tmp_path / "out.mkv", DummyMattingRunner(), "alpha")
    real_probe = video.probe_video
    monkeypatch.setattr(video, "probe_video", lambda path:
        replace(real_probe(path), frame_count=0) if Path(path) == source else real_probe(path))
    target = reporting.write_run_report(tmp_path / "unknown.json", source, stats,
                                        {"model": "dummy", "device": "cpu"}, 2)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["input"]["frame_count"] is None
    assert data["input"]["duration_seconds"] is None
    assert data["output"]["frame_count"] == 4
    assert evaluation.summarize_reports([target])["group_count"] == 0
