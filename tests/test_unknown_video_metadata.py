"""Unknown input length is supported; missing dimensions/FPS are not invented."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from cv_human_video_matting import cli, evaluation, pipeline
from cv_human_video_matting.io import video
from cv_human_video_matting.matting.dummy import DummyMattingRunner


@pytest.fixture
def source_video(tmp_path):
    import imageio_ffmpeg

    source = tmp_path / "audio-source.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=32x24:rate=12:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
         "-c:v", "mpeg4", "-c:a", "aac", str(source)],
        capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0
    return source


def unknown_source_count(monkeypatch, source, count=0):
    original_probe = video.probe_video

    def probe(path):
        info = original_probe(path)
        return replace(info, frame_count=count) if Path(path).resolve() == source.resolve() else info

    monkeypatch.setattr(video, "probe_video", probe)
    monkeypatch.setattr(pipeline, "probe_video", probe)
    return original_probe


@pytest.mark.parametrize("count", [0, -1])
def test_unknown_input_length_remains_valid_for_processing(source_video, tmp_path, monkeypatch, count):
    unknown_source_count(monkeypatch, source_video, count)
    prepared = pipeline.validate_video_request(source_video, tmp_path / "out.mkv", "green")
    assert prepared.total_frames is None
    updates = []
    stats = pipeline.process_video(
        source_video, tmp_path / "out.mkv", DummyMattingRunner(), "green", max_frames=3,
        progress_callback=lambda done, total: updates.append((done, total)),
    )
    assert stats.frames == 3
    assert updates[-1] == (3, None)
    assert video.probe_video(stats.output_path).fps == 12


@pytest.mark.parametrize("field,value", [
    ("fps", 0), ("fps", -1), ("fps", float("nan")), ("fps", float("inf")),
    ("width", 0), ("width", -1), ("width", float("nan")), ("width", float("inf")),
    ("height", 0), ("height", -1),
])
def test_invalid_fps_or_dimensions_fail_preflight_without_creating_outputs(
    source_video, tmp_path, monkeypatch, field, value,
):
    bad = replace(video.probe_video(source_video), **{field: value})
    monkeypatch.setattr(pipeline, "probe_video", lambda path: bad)
    output = tmp_path / "new-output-directory" / "result.mp4"
    with pytest.raises(ValueError, match="invalid dimensions or frame rate"):
        pipeline.validate_video_request(source_video, output, "green")
    assert not output.parent.exists()


def test_unknown_count_exception_applies_only_to_original_audio_source(source_video, monkeypatch):
    unknown_source_count(monkeypatch, source_video)
    assert video._checked_video(source_video, allow_unknown_frames=True).frame_count == 0
    with pytest.raises(ValueError, match="usable frames"):
        video._checked_video(source_video)
    with pytest.raises(RuntimeError, match="browser-compatible"):
        video.ensure_browser_playable_video(source_video, strict=True)


def _assert_delivered_video_and_report(output, report_path):
    import imageio_ffmpeg

    output = Path(output)
    info = video.probe_video(output)
    assert (info.width, info.height, info.frame_count) == (32, 24, 3)
    assert info.fps == 12
    assert info.duration_seconds == pytest.approx(0.25)
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(output), "-map", "0:a:0", "-ac", "1", "-ar", "48000",
         "-f", "s16le", "pipe:1"], capture_output=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    samples = np.frombuffer(result.stdout, dtype="<i2")
    assert 0.24 <= len(samples) / 48000 <= 0.30
    assert np.max(np.abs(samples)) > 100
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert data["input"]["frame_count"] is None
    assert data["input"]["duration_seconds"] is None
    assert data["output"]["frame_count"] == data["result"]["frames"] == 3
    assert data["output"]["duration_seconds"] == pytest.approx(0.25)
    assert Path(data["output"]["path"]) == output
    summary = evaluation.summarize_reports([report_path])
    assert summary["group_count"] == 0
    assert summary["unverified_report_count"] == 1


def test_cli_unknown_count_completes_matting_audio_export_and_report(source_video, tmp_path, monkeypatch):
    unknown_source_count(monkeypatch, source_video)
    output, report = tmp_path / "cli.mp4", tmp_path / "cli.json"
    cli.main(["matte", "--model", "dummy", "--device", "cpu", "--input", str(source_video),
              "--output", str(output), "--report", str(report), "--max-frames", "3"])
    _assert_delivered_video_and_report(output, report)
    assert not list(tmp_path.glob(".matting-*"))


def test_ui_unknown_count_completes_matting_audio_export_and_report(source_video, tmp_path, monkeypatch):
    from cv_human_video_matting.ui import gradio_app

    unknown_source_count(monkeypatch, source_video)
    monkeypatch.setattr(gradio_app, "_ACTIVE_CONFIG", {})
    monkeypatch.setattr(gradio_app, "_timestamped_output_path", lambda mode: tmp_path / "ui.mp4")
    result = gradio_app._run_batch_demo(
        str(source_video), None, "dummy", "green", 3, "cpu", preserve_audio=True,
    )
    _assert_delivered_video_and_report(result.video_path, result.report_path)
    assert not list(tmp_path.glob(".batch-*"))


def test_output_with_unknown_count_remains_rejected(source_video, tmp_path, monkeypatch):
    unknown_source_count(monkeypatch, source_video)
    with pytest.raises(ValueError, match="usable frames"):
        video.export_video_for_delivery(source_video, tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists()


def test_early_eof_message_includes_possible_metadata_inaccuracy(source_video, tmp_path, monkeypatch):
    info = video.probe_video(source_video)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: replace(info, frame_count=100))
    with pytest.raises(ValueError, match="inaccurate frame-count metadata; verify/re-encode source"):
        pipeline.process_video(source_video, tmp_path / "out.mp4", DummyMattingRunner(), "green")


def test_writer_silent_frame_loss_prevents_cli_publication_and_preserves_old_results(
    source_video, tmp_path, monkeypatch,
):
    original_writer = pipeline.cv2.VideoWriter

    class DroppingWriter:
        def __init__(self, *args):
            self.writer = original_writer(*args)
            self.saved_first = False

        def isOpened(self):
            return self.writer.isOpened()

        def write(self, frame):
            if not self.saved_first:
                self.writer.write(frame)
                self.saved_first = True

        def release(self):
            self.writer.release()

    monkeypatch.setattr(pipeline.cv2, "VideoWriter", DroppingWriter)
    output, report = tmp_path / "existing.mp4", tmp_path / "existing.json"
    output.write_bytes(b"keep previous delivery")
    report.write_bytes(b"keep previous report")
    with pytest.raises(RuntimeError, match="Written video frame count mismatch: expected 3, found 1"):
        cli.main(["matte", "--model", "dummy", "--device", "cpu", "--input", str(source_video),
                  "--output", str(output), "--report", str(report), "--max-frames", "3"])
    assert output.read_bytes() == b"keep previous delivery"
    assert report.read_bytes() == b"keep previous report"
    assert not list(tmp_path.glob(".matting-*"))


def test_valid_but_truncated_ffmpeg_output_is_not_published(source_video, tmp_path, monkeypatch):
    processed = tmp_path / "processed.mkv"
    pipeline.process_video(source_video, processed, DummyMattingRunner(), "green", max_frames=3)
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep previous delivery")
    original_encoder = video._run_encoder

    def truncate(command, should_cancel):
        original_encoder(command[:-1] + ["-frames:v", "1", command[-1]], should_cancel)

    monkeypatch.setattr(video, "_run_encoder", truncate)
    with pytest.raises(RuntimeError, match="FFmpeg output frame count mismatch: expected 3, found 1"):
        video.export_video_for_delivery(processed, output, source_video)
    assert output.read_bytes() == b"keep previous delivery"
    assert not list(tmp_path.glob(".existing-*.mp4"))
