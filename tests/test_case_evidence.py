"""Private case evidence must preserve identity and same-frame pairing."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import collect_case_evidence as cases
from cv_human_video_matting import reporting
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.pipeline import process_video


def make_video(path, offset=0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(3):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 30 + offset + index * 35
        image[:, :, 1] = 60 + index * 40
        image[:, :, 2] = 90 + index * 45
        writer.write(image)
    writer.release()


def make_run(tmp_path, *, transparent=False):
    source = tmp_path / "source.mp4"
    make_video(source)
    if transparent:
        result = tmp_path / "result_frames"
        result.mkdir()
        files = []
        for index in range(3):
            frame = np.zeros((24, 32, 4), dtype=np.uint8)
            frame[:, :, :3] = 50 + index * 30
            frame[:, :, 3] = 80 + index * 60
            path = result / "frame_{:06d}.png".format(index)
            assert cv2.imwrite(str(path), frame)
            files.append({"name": path.name, **cases._file_record(path)})
        output = {"type": "png_sequence", "path": str(result), "files": files,
                  "manifest_sha256": cases._canonical_hash(files), "frame_count": 3}
    else:
        result = tmp_path / "result.mp4"
        make_video(result, offset=25)
        output = {"type": "video", "path": str(result),
                  **cases._file_record(result), "frame_count": 3}
    report = tmp_path / "run.json"
    report.write_text(json.dumps({
        "schema_version": 3, "kind": "run_report",
        "input": {"path": str(source), **cases._file_record(source), "frame_count": 3},
        "output": output,
        "result": {"frames": 3},
        "options": {"model": "dummy", "mode": "transparent" if transparent else "green",
                    "edge_feather": 1},
        "quality_metrics": None,
    }), encoding="utf-8")
    permission = tmp_path / "permission.txt"
    permission.write_text("Private processing and screenshot review authorized.", encoding="utf-8")
    return source, result, report, permission


def collect(tmp_path, run, *, private_name="private", frames=(0, 2)):
    source, result, report, permission = run
    return cases.collect_case_evidence(
        case_id="lesson_01", scenario="moving_person", authorization_record=permission,
        authorized_use="private teaching review", report_path=report, source_path=source,
        result_path=result, frame_indices=frames, private_dir=tmp_path / private_name)


def test_collect_and_verify_video_pairs_are_repeatable_and_private(tmp_path):
    run = make_run(tmp_path)
    first = collect(tmp_path, run)
    case_dir = first.parent
    manifest = json.loads(first.read_text(encoding="utf-8"))
    assert manifest["human_quality_assessment"] is None
    assert manifest["conditions"]["edge_feather"] == 1
    assert [item["index"] for item in manifest["frames"]] == [0, 2]
    assert sorted(path.name for path in case_dir.iterdir()) == [
        "frame_000000_result.png", "frame_000000_source.png",
        "frame_000002_result.png", "frame_000002_source.png", "manifest.json"]
    assert cases.verify_case_evidence(case_dir) == manifest
    second = collect(tmp_path, run, private_name="another-private")
    assert second.read_bytes() == first.read_bytes()
    assert cases.REPOSITORY not in case_dir.parents


def test_collects_actual_schema_three_run_report(tmp_path):
    source = tmp_path / "actual-source.mp4"
    make_video(source)
    stats = process_video(source, tmp_path / "actual-result.mp4", DummyMattingRunner(),
                          "green", max_frames=2)
    report = reporting.write_run_report(
        tmp_path / "actual-run.json", source, stats,
        {"model": "dummy", "device": "cpu", "mode": "green"}, total_seconds=1.0)
    permission = tmp_path / "actual-permission.txt"
    permission.write_text("Private review authorized", encoding="utf-8")
    manifest_path = collect(tmp_path, (source, stats.output_path, report, permission),
                            frames=(0, 1))
    manifest = cases.verify_case_evidence(manifest_path.parent)
    assert manifest["result"]["sha256"] == json.loads(
        report.read_text(encoding="utf-8"))["output"]["sha256"]


def test_transparent_sequence_preserves_actual_alpha_channel(tmp_path):
    run = make_run(tmp_path, transparent=True)
    manifest_path = collect(tmp_path, run, frames=(1,))
    manifest = cases.verify_case_evidence(manifest_path.parent)
    pair = manifest["frames"][0]
    assert "alpha_image" in pair
    alpha = cv2.imread(str(manifest_path.parent / pair["alpha_image"]["name"]),
                       cv2.IMREAD_UNCHANGED)
    result = cv2.imread(str(manifest_path.parent / pair["result_image"]["name"]),
                        cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(alpha, result[:, :, 3])


def test_report_hash_mismatch_fails_before_creating_evidence(tmp_path):
    run = make_run(tmp_path)
    with run[0].open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Input SHA256"):
        collect(tmp_path, run)
    assert not (tmp_path / "private").exists()


def test_unselected_png_frame_change_invalidates_whole_result(tmp_path):
    run = make_run(tmp_path, transparent=True)
    frame = run[1] / "frame_000002.png"
    frame.write_bytes(frame.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="PNG frame SHA256"):
        collect(tmp_path, run, frames=(0,))
    assert not (tmp_path / "private").exists()


def test_missing_authorization_and_public_repo_destination_are_rejected(tmp_path):
    run = make_run(tmp_path)
    run[3].write_bytes(b"")
    with pytest.raises(ValueError, match="Authorization record is empty"):
        collect(tmp_path, run)
    run[3].write_text("Permission confirmed", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the public repository"):
        cases.collect_case_evidence(
            case_id="lesson_01", scenario="moving_person", authorization_record=run[3],
            authorized_use="private review", report_path=run[2], source_path=run[0],
            result_path=run[1], frame_indices=[0],
            private_dir=Path(__file__).resolve().parents[1] / "data" / "evidence")


def test_new_relative_destination_inside_repository_is_rejected(monkeypatch):
    monkeypatch.chdir(cases.REPOSITORY)
    with pytest.raises(ValueError, match="outside the public repository"):
        cases._private_target("data/new-private-case-evidence", "lesson_01")


def test_out_of_range_frame_is_rejected_before_writing(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(ValueError, match="exceeds the processed output"):
        collect(tmp_path, run, frames=(3,))
    assert not (tmp_path / "private").exists()


def test_verify_detects_swapped_same_frame_screenshot(tmp_path):
    run = make_run(tmp_path)
    manifest_path = collect(tmp_path, run)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["frames"][0]["source_image"], data["frames"][1]["source_image"] = (
        data["frames"][1]["source_image"], data["frames"][0]["source_image"])
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Source screenshot differs from frame"):
        cases.verify_case_evidence(manifest_path.parent)


def test_verify_detects_changed_authorization_record(tmp_path):
    run = make_run(tmp_path)
    manifest_path = collect(tmp_path, run)
    run[3].write_text("different permission note", encoding="utf-8")
    with pytest.raises(ValueError, match="Authorization SHA256"):
        cases.verify_case_evidence(manifest_path.parent)
