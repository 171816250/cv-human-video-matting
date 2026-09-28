"""CLI media/report publication must preserve prior artifacts on any failure."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_human_video_matting import cli, reporting
from cv_human_video_matting.utils import publication


@pytest.fixture
def source_video(tmp_path):
    source = tmp_path / "source.mkv"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"FFV1"), 10, (32, 24))
    assert writer.isOpened()
    try:
        for value in (50, 100, 150):
            writer.write(np.full((24, 32, 3), value, np.uint8))
    finally:
        writer.release()
    return source


def run_cli(source, output, report=None, mode="green"):
    args = ["matte", "--model", "dummy", "--device", "cpu", "--input", str(source),
            "--output", str(output), "--mode", mode, "--no-progress"]
    if report is not None:
        args.extend(["--report", str(report)])
    cli.main(args)


def fail_publication_to(monkeypatch, destination):
    real_replace = Path.replace
    real_link = publication.os.link

    def replace(source, target):
        if Path(target) == destination:
            raise OSError("simulated second artifact publication failure")
        return real_replace(source, target)

    def link(source, target, *args, **kwargs):
        if Path(target) == destination:
            raise OSError("simulated second artifact publication failure")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(publication.os, "link", link)


def assert_no_staging(root):
    assert not list(root.rglob(".publish-*"))
    assert not list(root.rglob(".matting-*"))


def test_report_directory_fails_before_model_load(source_video, tmp_path, monkeypatch):
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous video")
    report = tmp_path / "report.json"
    report.mkdir()
    calls = []
    monkeypatch.setattr(cli, "create_matting_runner", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="[Rr]eport"):
        run_cli(source_video, output, report)
    assert not calls
    assert output.read_bytes() == b"previous video"
    assert report.is_dir()
    assert_no_staging(tmp_path)


def test_report_generation_failure_preserves_existing_media_and_report(source_video, tmp_path, monkeypatch):
    output, report = tmp_path / "out.mp4", tmp_path / "report.json"
    output.write_bytes(b"previous video")
    report.write_bytes(b"previous report")

    def changed_bundle(*args, **kwargs):
        raise ValueError("Model bundle changed after model initialization")

    monkeypatch.setattr(reporting, "write_run_report", changed_bundle)
    with pytest.raises(ValueError, match="bundle changed"):
        run_cli(source_video, output, report)
    assert output.read_bytes() == b"previous video"
    assert report.read_bytes() == b"previous report"
    assert_no_staging(tmp_path)


@pytest.mark.parametrize("existing", [True, False])
def test_cross_directory_report_publication_failure_rolls_back_video(
    source_video, tmp_path, monkeypatch, existing,
):
    output = tmp_path / "video-dir" / "out.mp4"
    report = tmp_path / "reports" / "report.json"
    output.parent.mkdir()
    report.parent.mkdir()
    if existing:
        output.write_bytes(b"previous video")
        report.write_bytes(b"previous report")
    fail_publication_to(monkeypatch, report)
    with pytest.raises(OSError, match="second artifact publication"):
        run_cli(source_video, output, report)
    if existing:
        assert output.read_bytes() == b"previous video"
        assert report.read_bytes() == b"previous report"
    else:
        assert not output.exists()
        assert not report.exists()
    assert_no_staging(tmp_path)


@pytest.mark.parametrize("existing_directory", [False, True])
def test_transparent_report_failure_rolls_back_frames_and_keeps_notes(
    source_video, tmp_path, monkeypatch, existing_directory,
):
    output = tmp_path / "transparent.mp4"
    frames = tmp_path / "transparent_frames"
    report = tmp_path / "report.json"
    report.write_bytes(b"previous report")
    if existing_directory:
        frames.mkdir()
        (frames / "notes.txt").write_bytes(b"keep notes")
    fail_publication_to(monkeypatch, report)
    with pytest.raises(OSError, match="second artifact publication"):
        run_cli(source_video, output, report, "transparent")
    assert report.read_bytes() == b"previous report"
    if existing_directory:
        assert sorted(path.name for path in frames.iterdir()) == ["notes.txt"]
        assert (frames / "notes.txt").read_bytes() == b"keep notes"
    else:
        assert not frames.exists()
    assert_no_staging(tmp_path)


@pytest.mark.parametrize("mode", ["green", "transparent"])
def test_successful_report_records_final_location_and_artifact_bytes(source_video, tmp_path, mode):
    output, report = tmp_path / "out.mp4", tmp_path / "reports" / "run.json"
    if mode == "transparent":
        (tmp_path / "out_frames").mkdir()
        (tmp_path / "out_frames" / "notes.txt").write_bytes(b"keep notes")
    run_cli(source_video, output, report, mode)
    data = json.loads(report.read_text(encoding="utf-8"))
    actual = tmp_path / "out_frames" if mode == "transparent" else output
    assert Path(data["output"]["path"]) == actual
    assert Path(data["result"]["output_path"]) == actual
    assert data["output"]["frame_count"] == data["result"]["frames"] == 3
    if mode == "transparent":
        assert len(list(actual.glob("frame_*.png"))) == 3
        assert (actual / "notes.txt").read_bytes() == b"keep notes"
    assert_no_staging(tmp_path)


def test_exclusive_frame_publication_preserves_competing_file(tmp_path, monkeypatch):
    staged = tmp_path / "staged.png"
    staged.write_bytes(b"our frame")
    target = tmp_path / "frame_000000.png"
    real_link = publication.os.link

    def competitor(source, destination, *args, **kwargs):
        target.write_bytes(b"concurrent frame")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(publication.os, "link", competitor)
    with pytest.raises(FileExistsError):
        publication.publish_artifacts([publication.PublicationItem(staged, target, False)])
    assert target.read_bytes() == b"concurrent frame"
    assert_no_staging(tmp_path)


def test_publication_copy_failure_leaves_existing_files_intact(tmp_path, monkeypatch):
    staged, target = tmp_path / "staged", tmp_path / "target"
    staged.write_bytes(b"new")
    target.write_bytes(b"old")

    def fail_copy(*args, **kwargs):
        raise OSError("storage full")

    monkeypatch.setattr(publication.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="storage full"):
        publication.publish_artifacts([publication.PublicationItem(staged, target)])
    assert target.read_bytes() == b"old"
    assert_no_staging(tmp_path)


def test_failed_rollback_preserves_original_backup_for_recovery(tmp_path, monkeypatch):
    staged_video, staged_report = tmp_path / "new-video", tmp_path / "new-report"
    video, report = tmp_path / "video", tmp_path / "report"
    staged_video.write_bytes(b"new video")
    staged_report.write_bytes(b"new report")
    video.write_bytes(b"old video")
    report.write_bytes(b"old report")
    real_replace = Path.replace

    def fail_publication_and_restore(source, target):
        if Path(target) == report or (Path(target) == video and source.name == "previous"):
            raise OSError("simulated inaccessible target")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_restore)
    with pytest.raises(RuntimeError, match="rollback was incomplete.*recovery"):
        publication.publish_artifacts([
            publication.PublicationItem(staged_video, video),
            publication.PublicationItem(staged_report, report),
        ])
    backups = list(tmp_path.glob(".publish-*/previous"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old video"
    assert report.read_bytes() == b"old report"


def test_cleanup_failure_does_not_report_successful_publication_as_failed(tmp_path, monkeypatch, caplog):
    staged, target = tmp_path / "staged", tmp_path / "target"
    staged.write_bytes(b"new")
    target.write_bytes(b"old")

    def failed_cleanup(*args, **kwargs):
        raise PermissionError("temporary directory is busy")

    monkeypatch.setattr(publication.shutil, "rmtree", failed_cleanup)
    publication.publish_artifacts([publication.PublicationItem(staged, target)])
    assert target.read_bytes() == b"new"
    assert "Could not clean publication temporary directory" in caplog.text
