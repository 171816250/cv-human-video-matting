"""Delivery checks use short, locally generated videos and tone audio only."""

from pathlib import Path
import os
import subprocess

import numpy as np
import pytest

from cv_human_video_matting.io import video


@pytest.fixture(scope="module")
def ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(ffmpeg, *arguments):
    return subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-y", *map(str, arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _make_video(ffmpeg, path, *, duration=0.5, audio=False, odd=False):
    dimensions = "33x25" if odd else "32x24"
    arguments = [
        "-f", "lavfi", "-i",
        f"testsrc=size={dimensions}:rate=12:duration={duration}",
    ]
    if audio:
        arguments.extend([
            "-f", "lavfi", "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            "-c:a", "aac",
        ])
    arguments.extend(["-c:v", "ffv1" if odd else "mpeg4", str(path)])
    _run(ffmpeg, *arguments)
    return path


def _audio_samples(ffmpeg, path):
    result = _run(ffmpeg, "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "48000",
                  "-f", "s16le", "pipe:1")
    return np.frombuffer(result.stdout, dtype="<i2")


def _stream_description(ffmpeg, path):
    result = _run(ffmpeg, "-i", path, "-map", "0", "-f", "null", "-")
    return result.stderr.decode("utf-8", errors="replace")


def test_export_keeps_original_audio_and_trims_it_to_processed_frames(tmp_path, ffmpeg):
    original = _make_video(ffmpeg, tmp_path / "original.mp4", duration=2, audio=True)
    processed = _make_video(ffmpeg, tmp_path / "processed.mp4")
    output = tmp_path / "delivery.mp4"
    original_bytes = original.read_bytes()

    assert video.export_video_for_delivery(processed, output, original) == output

    info = video.probe_video(output)
    assert info.frame_count == 6
    assert info.duration_seconds == pytest.approx(0.5, abs=0.01)
    samples = _audio_samples(ffmpeg, output)
    assert 0.49 <= len(samples) / 48000 <= 0.54  # AAC may pad its final packet.
    assert np.max(np.abs(samples)) > 100
    description = _stream_description(ffmpeg, output)
    assert "Video: h264" in description
    assert "yuv420p" in description
    assert "Audio: aac" in description
    assert original.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".delivery-*.mp4"))


def test_export_accepts_original_without_audio(tmp_path, ffmpeg):
    original = _make_video(ffmpeg, tmp_path / "original.mp4", duration=1)
    processed = _make_video(ffmpeg, tmp_path / "processed.mp4")
    output = video.export_video_for_delivery(processed, tmp_path / "silent.mp4", original)

    assert video.probe_video(output).frame_count == 6
    assert "Audio:" not in _stream_description(ffmpeg, output)


def test_export_can_disable_audio(tmp_path, ffmpeg):
    original = _make_video(ffmpeg, tmp_path / "original.mp4", audio=True)
    output = video.export_video_for_delivery(
        original, tmp_path / "silent.mp4", preserve_audio=False
    )

    assert "Audio:" not in _stream_description(ffmpeg, output)


def test_export_in_place_preserves_its_own_audio(tmp_path, ffmpeg):
    processed = _make_video(ffmpeg, tmp_path / "processed.mp4", audio=True)

    assert video.export_video_for_delivery(processed, processed) == processed

    assert video.probe_video(processed).frame_count == 6
    assert len(_audio_samples(ffmpeg, processed)) > 0
    assert "Video: h264" in _stream_description(ffmpeg, processed)


def test_export_pads_odd_dimensions_without_dropping_frames(tmp_path, ffmpeg):
    processed = _make_video(ffmpeg, tmp_path / "odd.avi", odd=True)
    assert video.probe_video(processed).width == 33
    output = video.export_video_for_delivery(processed, tmp_path / "even.mp4")

    info = video.probe_video(output)
    assert (info.width, info.height, info.frame_count) == (34, 26, 6)


def test_export_refuses_overwriting_original_source(tmp_path, ffmpeg):
    original = _make_video(ffmpeg, tmp_path / "original.mp4")
    processed = _make_video(ffmpeg, tmp_path / "processed.mp4")
    original_bytes = original.read_bytes()

    with pytest.raises(ValueError, match="original source"):
        video.export_video_for_delivery(processed, original, original)

    assert original.read_bytes() == original_bytes


@pytest.mark.parametrize("failure", ["missing", "nonzero", "empty", "subprocess"])
def test_export_failure_preserves_existing_output_and_removes_temporary(
    tmp_path, ffmpeg, monkeypatch, failure
):
    processed = _make_video(ffmpeg, tmp_path / "processed.mp4")
    output = tmp_path / "delivery.mp4"
    output.write_bytes(b"existing user output")

    def fail(command, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("FFmpeg executable unavailable")
        if failure == "subprocess":
            raise subprocess.SubprocessError("FFmpeg launch interrupted")
        if failure == "nonzero":
            Path(command[-1]).write_bytes(b"partial conversion")
            return subprocess.CompletedProcess(command, 1, b"", b"encoder failed")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        video.export_video_for_delivery(processed, output)

    assert output.read_bytes() == b"existing user output"
    assert not list(tmp_path.glob(".delivery-*.mp4"))


def test_browser_preview_preserves_audio(tmp_path, ffmpeg):
    source = _make_video(ffmpeg, tmp_path / "input.mp4", audio=True)

    preview = video.ensure_browser_playable_video(source, strict=True)

    assert preview.name == "input_browser.mp4"
    assert "Video: h264" in _stream_description(ffmpeg, preview)
    assert len(_audio_samples(ffmpeg, preview)) > 0


def test_browser_preview_rebuilds_empty_cache(tmp_path, ffmpeg):
    source = _make_video(ffmpeg, tmp_path / "input.mp4")
    cached = tmp_path / "input_browser.mp4"
    cached.touch()
    os.utime(cached, (source.stat().st_mtime + 10, source.stat().st_mtime + 10))

    assert video.ensure_browser_playable_video(source, strict=True) == cached
    assert video.probe_video(cached).frame_count == 6


def test_browser_preview_reuses_valid_cache(tmp_path, ffmpeg):
    source = _make_video(ffmpeg, tmp_path / "input.mp4")
    cached = video.ensure_browser_playable_video(source, strict=True)

    def unexpected_conversion(source, target):
        raise AssertionError("The valid cache should be reused")

    assert video.ensure_browser_playable_video(source, unexpected_conversion, strict=True) == cached


def test_browser_conversion_failure_is_strict_or_backward_compatible(tmp_path, ffmpeg):
    source = _make_video(ffmpeg, tmp_path / "input.mp4")
    cached = tmp_path / "input_browser.mp4"
    cached.write_bytes(b"previous output")

    def fail(source, target):
        target.write_bytes(b"partial output")
        raise OSError("encoder unavailable")

    assert video.ensure_browser_playable_video(source, fail) == source
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        video.ensure_browser_playable_video(source, fail, strict=True)
    assert cached.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".input_browser-*.mp4"))


def test_browser_strict_mode_rejects_missing_converter(tmp_path, ffmpeg):
    source = _make_video(ffmpeg, tmp_path / "input.mp4")

    with pytest.raises(RuntimeError, match="No browser-compatible"):
        video.ensure_browser_playable_video(source, converter=None, strict=True)
