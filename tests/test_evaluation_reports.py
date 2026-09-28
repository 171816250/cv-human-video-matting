"""Artifact hashes, strict condition matching, and safe offline report output."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cv_human_video_matting import evaluation, reporting
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.pipeline import process_video


def make_video(path, size=(64, 48), frames=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((size[1], size[0], 3), 30 + index * 20, np.uint8))
    finally:
        writer.release()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def real_run(tmp_path, monkeypatch):
    source, output = tmp_path / "source.mp4", tmp_path / "processed.mp4"
    make_video(source)
    stats = process_video(source, output, DummyMattingRunner(), "green", max_frames=3,
                          resize_short=24)
    monkeypatch.setattr(reporting, "_code_provenance", lambda: {
        "version": "0.2.0rc1", "git_commit": "a" * 40, "git_dirty": False,
        "source_sha256": "b" * 64,
    })
    monkeypatch.setattr(reporting, "_environment", lambda device: {
        "python": "3.11.0", "platform": "test-platform", "packages": {"torch": "test"},
        "device": device, "gpu": None,
    })
    return source, stats


def test_run_report_preserves_legacy_fields_and_identifies_scaled_output(real_run, tmp_path):
    source, stats = real_run
    target = reporting.write_run_report(tmp_path / "run.json", source, stats,
                                        {"device": "cpu", "model": "dummy"}, 5.0, 0.4)
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["schema_version"] == 3
    assert report["input"]["sha256"] == sha(source)
    assert report["input"]["width"] == 64
    assert report["output"]["sha256"] == sha(stats.output_path)
    assert (report["output"]["width"], report["output"]["height"]) == (32, 24)
    assert report["output"]["frame_count"] == report["result"]["frames"] == 3
    assert report["output"]["fps"] == 10
    assert report["output"]["duration_seconds"] == pytest.approx(0.3)
    assert report["result"]["output_path"] == str(stats.output_path)
    assert report["result"]["total_seconds"] == 5.0
    assert report["result"]["total_fps"] == pytest.approx(0.6)
    assert report["result"]["metadata_seconds"] >= 0
    assert "Not included in total_seconds" in report["timing_scope"]["metadata_seconds"]
    assert report["code"]["version"] == "0.2.0rc1"
    assert report["quality_metrics"] is None
    assert evaluation.summarize_reports([target])["group_count"] == 1


def test_png_manifest_and_asset_hashes_are_reproducible(real_run, tmp_path):
    source, _ = real_run
    stats = process_video(source, tmp_path / "transparent.mp4", DummyMattingRunner(),
                          "transparent", max_frames=2, resize_short=24)
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"fixture checkpoint identity")
    background = tmp_path / "background.png"
    assert cv2.imwrite(str(background), np.zeros((8, 8, 3), np.uint8))
    options = {"device": "cpu", "model": "dummy", "checkpoint": str(checkpoint),
               "background": str(background)}
    target = reporting.write_run_report(tmp_path / "png.json", source, stats, options, 5)
    report = json.loads(target.read_text(encoding="utf-8"))
    output = report["output"]
    assert output["type"] == "png_sequence"
    assert (output["width"], output["height"], output["frame_count"]) == (32, 24, 2)
    assert output["manifest_sha256"] == reporting._canonical_hash(output["files"])
    assert [item["name"] for item in output["files"]] == ["frame_000000.png", "frame_000001.png"]
    for item in output["files"]:
        assert item["sha256"] == sha(stats.output_path / item["name"])
    assert report["assets"]["checkpoint"]["sha256"] == sha(checkpoint)
    assert report["assets"]["background"]["sha256"] == sha(background)
    assert evaluation.summarize_reports(target)["group_count"] == 1


def test_staged_output_hash_uses_actual_file_but_records_published_path(real_run, tmp_path):
    source, stats = real_run
    published = tmp_path / "published-task" / "result.mp4"
    target = reporting.write_run_report(
        tmp_path / "staged-report.json", source, stats, {"device": "cpu", "model": "dummy"},
        5, published_output_path=published,
    )
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["output"]["path"] == report["result"]["output_path"] == str(published)
    assert report["output"]["sha256"] == sha(stats.output_path)
    assert report["output"]["width"] == 32
    assert not published.exists()


def test_published_output_path_cannot_claim_input_or_report_path(real_run, tmp_path):
    source, stats = real_run
    report = tmp_path / "run.json"
    for published in (source, report):
        with pytest.raises(ValueError):
            reporting.write_run_report(report, source, stats, {}, 5,
                                       published_output_path=published)
    assert not report.exists()


def test_hashing_uses_bounded_reads_for_large_files(tmp_path, monkeypatch):
    path = tmp_path / "large.bin"
    contents = b"test-data" * 300000
    path.write_bytes(contents)
    original_open = Path.open
    read_sizes = []

    class Reader:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, size):
            read_sizes.append(size)
            return self.file.read(size)

    def open_file(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        return Reader(handle) if self == path else handle

    monkeypatch.setattr(Path, "open", open_file)
    record = reporting._file_record(path)
    assert record == {"sha256": hashlib.sha256(contents).hexdigest(), "size_bytes": len(contents)}
    assert len(read_sizes) >= 3
    assert all(size == 1024 * 1024 for size in read_sizes)


@pytest.mark.parametrize("which", ["input", "output", "png_child"])
def test_report_cannot_overwrite_media_or_live_inside_png_output(real_run, tmp_path, which):
    source, stats = real_run
    if which == "input":
        target = source.with_suffix(".json")
        source.rename(target)
        source = target
    elif which == "output":
        target = stats.output_path.with_suffix(".json")
        stats.output_path.rename(target)
        stats = replace(stats, output_path=target)
    else:
        output = tmp_path / "sequence"
        output.mkdir()
        target = output / "run.json"
        target.write_bytes(b"previous")
        stats = replace(stats, output_path=output)
    before = target.read_bytes()
    with pytest.raises(ValueError):
        reporting.write_run_report(target, source, stats, {"model": "dummy", "device": "cpu"}, 5)
    assert target.read_bytes() == before


def test_atomic_failure_keeps_existing_report_and_removes_temporary(real_run, tmp_path, monkeypatch):
    source, stats = real_run
    target = tmp_path / "run.json"
    target.write_bytes(b"old report")
    original_replace = Path.replace

    def fail_replace(self, destination):
        if Path(destination) == target:
            raise PermissionError("report open in another application")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError):
        reporting.write_run_report(target, source, stats, {"model": "dummy", "device": "cpu"}, 5)
    assert target.read_bytes() == b"old report"
    assert list(tmp_path.glob(".run.json-*.tmp")) == []


@pytest.mark.parametrize("has_git,fails", [(False, False), (True, False), (True, True)])
def test_code_identity_works_without_git_and_uses_scoped_safe_directory(tmp_path, monkeypatch, has_git, fails):
    root = tmp_path / "source-export"
    source = root / "src" / "cv_human_video_matting" / "reporting.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source fixture", encoding="utf-8")
    monkeypatch.setattr(reporting, "__file__", str(source))
    if has_git:
        (root / ".git").mkdir()
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5
        assert command[:5] == ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
        if fails:
            raise subprocess.TimeoutExpired(command, 5)
        return SimpleNamespace(stdout="a" * 40 if "rev-parse" in command else " M src/changed.py\n")

    monkeypatch.setattr(reporting.subprocess, "run", git)
    code = reporting._code_provenance()
    assert len(code["source_sha256"]) == 64
    assert code["version"]
    assert code["git_commit"] == ("a" * 40 if has_git and not fails else None)
    assert code["git_dirty"] is (True if has_git and not fails else None)
    assert bool(calls) is has_git


def sample_report():
    return {
        "schema_version": 3, "kind": "run_report", "created_at": "2026-09-28T00:00:00Z",
        "input": {"path": "input.mp4", "sha256": "a" * 64, "width": 64, "height": 48,
                  "fps": 10, "frame_count": 100, "duration_seconds": 10},
        "output": {"type": "video", "path": "output.mp4", "sha256": "c" * 64,
                   "width": 32, "height": 24, "fps": 10, "frame_count": 20, "duration_seconds": 2},
        "code": {"version": "0.2.0rc1", "git_commit": "d" * 40, "git_dirty": False,
                 "source_sha256": "b" * 64},
        "options": {"model": "rvm_mobilenetv3", "device": "cpu", "resize_short": 24,
                    "enable_depth": False, "downsample_ratio": "auto"},
        "environment": {"python": "3.11.0", "platform": "Windows-test",
                        "packages": {"torch": "2.0"}, "device": "cpu", "gpu": None},
        "result": {"frames": 20, "elapsed_seconds": 2.0, "total_seconds": 4.0,
                   "fps": 10.0, "total_fps": 5.0, "metadata_seconds": 0.1},
        "assets": {}, "timing_scope": {"total_seconds": "excludes metadata"},
        "quality_metrics": None,
    }


def put_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def test_repeats_group_and_medians_are_measured_not_pooled(tmp_path):
    paths = []
    for index, seconds in enumerate([2, 4, 5]):
        report = sample_report()
        report["input"]["path"] = f"different-local-copy-{index}.mp4"
        report["result"].update(elapsed_seconds=seconds, total_seconds=seconds * 2,
                                fps=20 / seconds, total_fps=10 / seconds)
        paths.append(put_report(tmp_path / f"run-{index}.json", report))
    summary = evaluation.summarize_reports(paths + [paths[0]])
    assert summary["report_count"] == 3
    assert summary["group_count"] == 1
    group = summary["groups"][0]
    assert group["count"] == 3
    assert group["processing_fps"] == {"median": 5, "min": 4, "max": 10}
    assert group["total_fps"] == {"median": 2.5, "min": 2, "max": 5}
    assert len(group["runs"]) == 3
    assert summary["quality_metrics"] is None


@pytest.mark.parametrize("section,key,value", [
    ("input", "sha256", "f" * 64), ("options", "model", "birefnet"),
    ("options", "resize_short", 30), ("options", "enable_depth", True),
    ("options", "downsample_ratio", 0.5), ("environment", "device", "cuda:0"),
    ("environment", "gpu", "Different GPU"), ("environment", "packages", {"torch": "different"}),
    ("output", "width", 64), ("output", "fps", 20),
    ("code", "source_sha256", "f" * 64), ("code", "version", "next-version"),
    ("result", "frames", 10),
])
def test_different_input_config_frames_environment_or_code_never_mix(tmp_path, section, key, value):
    first, second = sample_report(), sample_report()
    second[section][key] = value
    if section == "result":
        second["result"]["fps"] = value / second["result"]["elapsed_seconds"]
        second["result"]["total_fps"] = value / second["result"]["total_seconds"]
    paths = [put_report(tmp_path / "one.json", first), put_report(tmp_path / "two.json", second)]
    summary = evaluation.summarize_reports(paths)
    assert summary["group_count"] == 2
    assert all(group["count"] == 1 for group in summary["groups"])


@pytest.mark.parametrize("section,key", [
    ("input", "sha256"), ("options", "model"), ("environment", "device"),
    ("environment", "packages"), ("code", "source_sha256"), ("output", "sha256"),
    ("output", "frame_count"), ("result", "elapsed_seconds"),
])
def test_missing_identity_fields_are_listed_but_never_grouped(tmp_path, section, key):
    report = sample_report()
    del report[section][key]
    summary = evaluation.summarize_reports(put_report(tmp_path / "missing.json", report))
    assert summary["group_count"] == 0
    assert summary["unverified_report_count"] == 1
    assert summary["unverified"][0]["reasons"]


def test_legacy_report_stays_unverified_even_if_input_hash_added(tmp_path):
    report = sample_report()
    report["schema_version"] = 2
    summary = evaluation.summarize_reports(put_report(tmp_path / "legacy.json", report))
    assert summary["comparable_report_count"] == 0
    assert "旧版" in summary["unverified"][0]["reasons"][0]


@pytest.mark.parametrize("content", ["{bad json", "[]", '{"schema_version": 3, "result": NaN}',
                                      '{"schema_version": 3, "input": []}'])
def test_bad_report_fails_with_its_file_name_and_keeps_existing_summary(tmp_path, content):
    source = tmp_path / "bad-source.json"
    source.write_text(content, encoding="utf-8")
    target = tmp_path / "summary.json"
    target.write_bytes(b"previous summary")
    with pytest.raises(ValueError, match="bad-source.json"):
        evaluation.write_evaluation_summary([source], target)
    assert target.read_bytes() == b"previous summary"


def test_inconsistent_reported_fps_is_rejected(tmp_path):
    report = sample_report()
    report["result"]["fps"] = 9999
    with pytest.raises(ValueError, match="tampered.json"):
        evaluation.summarize_reports(put_report(tmp_path / "tampered.json", report))


def test_html_is_self_contained_and_escapes_all_external_content(tmp_path):
    report = sample_report()
    payload = '<script>alert("XSS")</script><img src="https://bad.invalid/track" onerror="boom">'
    report["options"]["model"] = payload
    report["input"]["path"] = payload
    report["environment"]["platform"] = payload
    source = put_report(tmp_path / "source.json", report)
    target = evaluation.write_evaluation_summary([source], tmp_path / "summary.html")
    html = target.read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in html
    assert "&lt;script&gt;" in html
    assert "<script" not in html and "<img" not in html
    assert 'href="http' not in html and 'src="http' not in html
    assert "default-src 'none'" in html
    assert "中位数" in html and "参数与报告来源" in html


def test_summary_does_not_overwrite_source_report(tmp_path):
    source = put_report(tmp_path / "run.json", sample_report())
    previous = source.read_bytes()
    with pytest.raises(ValueError):
        evaluation.write_evaluation_summary([source], source)
    assert source.read_bytes() == previous


def test_summary_json_round_trip_and_no_heavy_imports(tmp_path):
    source = put_report(tmp_path / "run.json", sample_report())
    target = evaluation.write_evaluation_summary([source], tmp_path / "summary.json")
    summary = json.loads(target.read_text(encoding="utf-8"))
    assert summary["kind"] == "evaluation_summary"
    assert summary["group_count"] == 1
    # Standard library only: no packages such as torch, cv2, numpy or PyYAML are available.
    package_source = Path(__file__).resolve().parents[1] / "src"
    code = "import sys;sys.path.insert(0,sys.argv[1]);from cv_human_video_matting.evaluation import summarize_reports;assert summarize_reports(sys.argv[2])['group_count']==1"
    result = subprocess.run([sys.executable, "-S", "-c", code, str(package_source), str(source)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_atomic_summary_replace_failure_keeps_old_file(tmp_path, monkeypatch):
    source = put_report(tmp_path / "run.json", sample_report())
    target = tmp_path / "summary.html"
    target.write_bytes(b"old summary")

    def fail_replace(self, destination):
        raise PermissionError("read-only destination")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError):
        evaluation.write_evaluation_summary([source], target)
    assert target.read_bytes() == b"old summary"
    assert list(tmp_path.glob(".summary.html-*.tmp")) == []
