"""Progress, cancellation, publication and child-process cleanup regressions."""

from dataclasses import replace
import subprocess
import sys

import cv2
import numpy as np
import pytest

from cv_human_video_matting import pipeline
from cv_human_video_matting.io import video
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.utils.progress import ProcessingCancelled


def make_video(path, frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (32, 24))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((24, 32, 3), 30 + index, np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("limit,total", [(None, 5), (3, 3), (8, 5)])
def test_progress_starts_at_zero_and_counts_each_completed_frame(tmp_path, limit, total):
    source = make_video(tmp_path / "source.mp4")
    progress = []

    stats = pipeline.process_video(
        source, tmp_path / "output.mp4", DummyMattingRunner(), "green", max_frames=limit,
        progress_callback=lambda done, count: progress.append((done, count)),
    )

    assert stats.frames == total
    assert progress == [(index, total) for index in range(total + 1)]


def test_unknown_metadata_reports_unknown_total_and_honors_frame_limit(tmp_path, monkeypatch):
    source = make_video(tmp_path / "source.mp4")
    info = video.probe_video(source)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: replace(info, frame_count=0))
    progress = []

    stats = pipeline.process_video(
        source, tmp_path / "out.mp4", DummyMattingRunner(), "green", max_frames=2,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert stats.frames == 2
    assert progress == [(0, None), (1, None), (2, None)]


def test_precancelled_job_stops_before_opening_input_or_modifying_output(tmp_path):
    output = tmp_path / "old.mp4"
    output.write_bytes(b"previous result")

    with pytest.raises(ProcessingCancelled):
        pipeline.process_video(
            tmp_path / "missing.mp4", output, DummyMattingRunner(), "green",
            should_cancel=lambda: True,
        )
    assert output.read_bytes() == b"previous result"


def test_cancelled_video_releases_capture_and_writer(tmp_path, monkeypatch):
    source = make_video(tmp_path / "source.mp4")
    captures, writers, progress = [], [], []
    real_capture, real_writer = cv2.VideoCapture, cv2.VideoWriter

    def capture(*args):
        instance = real_capture(*args)
        captures.append(instance)
        return instance

    def writer(*args):
        instance = real_writer(*args)
        writers.append(instance)
        return instance

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", writer)
    with pytest.raises(ProcessingCancelled):
        pipeline.process_video(
            source, tmp_path / "intermediate.mp4", DummyMattingRunner(), "green",
            progress_callback=lambda done, total: progress.append(done),
            should_cancel=lambda: bool(progress) and progress[-1] >= 2,
        )

    assert progress == [0, 1, 2]
    assert captures and all(not item.isOpened() for item in captures)
    assert writers and all(not item.isOpened() for item in writers)


@pytest.mark.parametrize("failure", ["cancel", "prediction", "callback"])
@pytest.mark.parametrize("existing_directory", [False, True])
def test_transparent_failure_removes_partial_frames_and_preserves_other_files(
    tmp_path, failure, existing_directory
):
    source = make_video(tmp_path / "source.mp4")
    output_dir = tmp_path / "transparent_frames"
    if existing_directory:
        output_dir.mkdir()
        (output_dir / "notes.txt").write_text("keep notes", encoding="utf-8")
    progress = []

    class Runner(DummyMattingRunner):
        def predict(self, frame):
            if failure == "prediction" and progress[-1] >= 2:
                raise RuntimeError("prediction failed")
            return super().predict(frame)

    def callback(done, total):
        progress.append(done)
        assert not list(output_dir.glob("frame_*.png"))
        if failure == "callback" and done == 2:
            raise RuntimeError("progress callback failed")

    error = ProcessingCancelled if failure == "cancel" else RuntimeError
    with pytest.raises(error):
        pipeline.process_video(
            source, tmp_path / "transparent.mp4", Runner(), "transparent",
            progress_callback=callback,
            should_cancel=lambda: failure == "cancel" and bool(progress) and progress[-1] >= 2,
        )

    assert progress == [0, 1, 2]
    assert not list(output_dir.glob("frame_*.png"))
    assert not list(tmp_path.glob(".transparent_frames-*"))
    if existing_directory:
        assert (output_dir / "notes.txt").read_text(encoding="utf-8") == "keep notes"
    else:
        assert not output_dir.exists()


def test_transparent_success_merges_into_directory_without_touching_notes(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    output_dir = tmp_path / "out_frames"
    output_dir.mkdir()
    (output_dir / "notes.txt").write_text("keep notes", encoding="utf-8")

    stats = pipeline.process_video(
        source, tmp_path / "out.mp4", DummyMattingRunner(), "transparent", max_frames=2,
    )

    assert stats.frames == 2
    assert len(list(output_dir.glob("frame_*.png"))) == 2
    assert (output_dir / "notes.txt").read_text(encoding="utf-8") == "keep notes"
    assert not list(tmp_path.glob(".out_frames-*"))


def test_cancellation_during_png_publication_rolls_back_only_new_frames(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    output_dir = tmp_path / "out_frames"
    output_dir.mkdir()
    (output_dir / "notes.txt").write_bytes(b"keep notes")

    with pytest.raises(ProcessingCancelled):
        pipeline.process_video(
            source, tmp_path / "out.mp4", DummyMattingRunner(), "transparent",
            should_cancel=lambda: (output_dir / "frame_000000.png").exists(),
        )

    assert sorted(item.name for item in output_dir.iterdir()) == ["notes.txt"]
    assert (output_dir / "notes.txt").read_bytes() == b"keep notes"
    assert not list(tmp_path.glob(".out_frames-*"))


def test_competing_frame_file_is_not_overwritten_or_removed(tmp_path, monkeypatch):
    source = make_video(tmp_path / "source.mp4")
    output_dir = tmp_path / "out_frames"
    output_dir.mkdir()
    competing = output_dir / "frame_000001.png"
    real_copy = pipeline.shutil.copyfileobj

    def competing_write(incoming, destination):
        real_copy(incoming, destination)
        competing.write_bytes(b"another job owns this file")

    monkeypatch.setattr(pipeline.shutil, "copyfileobj", competing_write)
    with pytest.raises(FileExistsError):
        pipeline.process_video(source, tmp_path / "out.mp4", DummyMattingRunner(), "transparent")

    assert competing.read_bytes() == b"another job owns this file"
    assert sorted(item.name for item in output_dir.iterdir()) == [competing.name]
    assert not list(tmp_path.glob(".out_frames-*"))


@pytest.mark.parametrize("interrupt", [False, True])
def test_export_cancellation_reaps_actual_ffmpeg_and_preserves_old_output(
    tmp_path, monkeypatch, interrupt
):
    processed = make_video(tmp_path / "processed.mp4")
    output = tmp_path / "delivery.mp4"
    output.write_bytes(b"previous successful output")
    processes = []
    real_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        assert kwargs["stderr"] != subprocess.PIPE
        child = real_popen(*args, **kwargs)
        processes.append(child)
        return child

    def cancel():
        if processes and interrupt:
            raise KeyboardInterrupt()
        return bool(processes)

    monkeypatch.setattr(video.subprocess, "Popen", tracked_popen)
    with pytest.raises(KeyboardInterrupt if interrupt else ProcessingCancelled):
        video.export_video_for_delivery(processed, output, should_cancel=cancel)

    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert output.read_bytes() == b"previous successful output"
    assert not list(tmp_path.glob(".delivery-*.mp4"))


def test_stubborn_encoder_is_killed_and_waited_after_terminate_timeout(tmp_path, monkeypatch):
    processed = make_video(tmp_path / "processed.mp4")
    output = tmp_path / "delivery.mp4"
    output.write_bytes(b"previous successful output")

    class StubbornEncoder:
        def __init__(self):
            self.waits = []
            self.terminated = False
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if not self.killed:
                raise subprocess.TimeoutExpired("mock ffmpeg", timeout)
            return -9

    encoder = StubbornEncoder()
    monkeypatch.setattr(video.subprocess, "Popen", lambda *args, **kwargs: encoder)

    with pytest.raises(ProcessingCancelled):
        video.export_video_for_delivery(
            processed, output, should_cancel=lambda: bool(encoder.waits),
        )

    assert encoder.terminated and encoder.killed
    assert encoder.waits == [0.1, 2.0, None]
    assert output.read_bytes() == b"previous successful output"
    assert not list(tmp_path.glob(".delivery-*.mp4"))


def test_precancelled_export_does_not_spawn_encoder_or_touch_output(tmp_path, monkeypatch):
    output = tmp_path / "old.mp4"
    output.write_bytes(b"previous result")

    def unexpected_process(*args, **kwargs):
        raise AssertionError("Cancelled work must not start FFmpeg")

    monkeypatch.setattr(video.subprocess, "Popen", unexpected_process)
    with pytest.raises(ProcessingCancelled):
        video.export_video_for_delivery(tmp_path / "missing.mp4", output, should_cancel=lambda: True)
    assert output.read_bytes() == b"previous result"


def test_encoder_large_stderr_does_not_block_and_reports_only_tail():
    command = [sys.executable, "-c",
               "import sys; sys.stderr.write('x' * 262144 + 'ENCODER_TAIL'); sys.exit(7)"]

    with pytest.raises(RuntimeError, match="ENCODER_TAIL") as error:
        video._run_encoder(command, should_cancel=None)

    assert len(str(error.value)) < 2100


def test_cancellation_after_encoding_does_not_publish_the_result(tmp_path, monkeypatch):
    processed = make_video(tmp_path / "processed.mp4")
    output = tmp_path / "delivery.mp4"
    output.write_bytes(b"keep delivered file")
    cancelled = False

    def complete_encoding(command, should_cancel):
        from pathlib import Path

        nonlocal cancelled
        Path(command[-1]).write_bytes(processed.read_bytes())
        cancelled = True

    monkeypatch.setattr(video, "_run_encoder", complete_encoding)
    with pytest.raises(ProcessingCancelled):
        video.export_video_for_delivery(processed, output, should_cancel=lambda: cancelled)

    assert output.read_bytes() == b"keep delivered file"
    assert not list(tmp_path.glob(".delivery-*.mp4"))
