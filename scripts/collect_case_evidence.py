"""Collect and verify private, same-frame evidence from a CLI run report.

This script never downloads media or copies an entire source/result video. It
requires an output directory outside this repository because extracted source
frames and local paths may identify a person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SCENARIOS = ("clear_single", "complex_background", "moving_person", "other")


def _file_record(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Required file is missing: {}".format(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed while hashing: {}".format(path))
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _no_constant(value):
    raise ValueError("JSON contains a non-finite number: {}".format(value))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _load_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=_no_constant, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return data


def _check_file(path, claimed, label):
    if not isinstance(claimed, dict) or not SHA256.fullmatch(str(claimed.get("sha256", ""))):
        raise ValueError("{} lacks a SHA256 identity".format(label))
    actual = _file_record(path)
    if actual["sha256"] != claimed["sha256"]:
        raise ValueError("{} SHA256 differs from the run report: {}".format(label, path))
    if "size_bytes" in claimed and actual["size_bytes"] != claimed["size_bytes"]:
        raise ValueError("{} size differs from the run report: {}".format(label, path))
    return actual


def _canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_run(report_path, source_path, result_path):
    report_path = Path(report_path).resolve()
    source_path = Path(source_path).resolve()
    result_path = Path(result_path).resolve()
    report_record = _file_record(report_path)
    report = _load_json(report_path)
    if _file_record(report_path) != report_record:
        raise ValueError("Run report changed while reading: {}".format(report_path))
    if report.get("schema_version") != 3 or report.get("kind") != "run_report":
        raise ValueError("Expected a CLI schema 3 run_report: {}".format(report_path))
    if report.get("quality_metrics", "missing") is not None:
        raise ValueError("This collector expects unmeasured quality_metrics=null")
    if not isinstance(report.get("options"), dict):
        raise ValueError("Run report has no parameter object")
    if not isinstance(report.get("input"), dict) or not isinstance(report.get("output"), dict):
        raise ValueError("Run report has no input/output records")
    result = report.get("result")
    if not isinstance(result, dict) or isinstance(result.get("frames"), bool) or \
            not isinstance(result.get("frames"), int) or result["frames"] <= 0:
        raise ValueError("Run report has no positive processed frame count")
    frames = result["frames"]
    if report["output"].get("frame_count") != frames:
        raise ValueError("Output frame count differs from processed frames")
    input_frames = report["input"].get("frame_count")
    if input_frames is not None and (isinstance(input_frames, bool) or
                                     not isinstance(input_frames, int) or input_frames < frames):
        raise ValueError("Input frame count is smaller than processed frames")
    source_record = _check_file(source_path, report["input"], "Input")
    output = report["output"]
    output_type = output.get("type")
    if output_type == "video":
        result_record = _check_file(result_path, output, "Output")
    elif output_type == "png_sequence":
        if not result_path.is_dir():
            raise ValueError("PNG result directory is missing: {}".format(result_path))
        files = output.get("files")
        if not isinstance(files, list) or len(files) != frames or any(
                not isinstance(item, dict) for item in files):
            raise ValueError("PNG report has an incomplete frame manifest")
        if output.get("manifest_sha256") != _canonical_hash(files):
            raise ValueError("PNG manifest SHA256 differs from the run report")
        expected = ["frame_{:06d}.png".format(index) for index in range(frames)]
        actual = sorted(path.name for path in result_path.glob("frame_*.png"))
        if actual != expected or [item.get("name") for item in files] != expected:
            raise ValueError("PNG frame names are missing, extra, or out of order")
        for item in files:
            _check_file(result_path / item["name"], item, "PNG frame")
        result_record = {"manifest_sha256": output["manifest_sha256"],
                         "frame_count": frames}
    else:
        raise ValueError("Unsupported run report output type: {}".format(output_type))
    return report, report_record, source_record, result_record


def _video_frames(path, indices):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Cannot open video: {}".format(path))
    selected = {}
    wanted = set(indices)
    try:
        for index in range(max(indices) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Cannot decode frame {} of {}".format(index, path))
            if index in wanted:
                selected[index] = frame
    finally:
        capture.release()
    return selected


def _read_png(path):
    frame = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 4:
        raise ValueError("Expected a transparent BGRA PNG: {}".format(path))
    return frame


def _result_frames(path, output_type, indices):
    if output_type == "video":
        return _video_frames(path, indices)
    return {index: _read_png(Path(path) / "frame_{:06d}.png".format(index))
            for index in indices}


def _write_png(path, frame):
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("Could not encode evidence image: {}".format(path))
    Path(path).write_bytes(encoded.tobytes())
    return {"name": Path(path).name, **_file_record(path)}


def _private_target(private_dir, case_id):
    if not CASE_ID.fullmatch(case_id):
        raise ValueError("Case ID must use 1-64 ASCII letters, digits, _ or -")
    # On supported Windows/Python 3.9 installations, resolving a path that
    # does not exist yet can retain its relative form. Anchor it to cwd first
    # so an ignored directory inside the repository cannot bypass this check.
    private_dir = Path(private_dir).absolute().resolve()
    if private_dir == REPOSITORY or REPOSITORY in private_dir.parents:
        raise ValueError("Private evidence directory must be outside the public repository")
    target = private_dir / case_id
    if target.exists() or target.is_symlink():
        raise ValueError("Case evidence already exists; choose a new private directory: {}".format(target))
    return private_dir, target


def collect_case_evidence(*, case_id, scenario, authorization_record, authorized_use,
                          report_path, source_path, result_path, frame_indices, private_dir):
    """Validate original artifacts, then atomically publish a private case folder."""
    if scenario not in SCENARIOS:
        raise ValueError("Scenario must be one of: {}".format(", ".join(SCENARIOS)))
    if not isinstance(authorized_use, str) or not authorized_use.strip():
        raise ValueError("Describe the authorized use of this material")
    private_dir, target = _private_target(private_dir, case_id)
    indices = sorted(set(frame_indices))
    if not indices or any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                          for index in indices):
        raise ValueError("Provide one or more non-negative, zero-based frame indices")
    authorization_record = Path(authorization_record).resolve()
    report_path = Path(report_path).resolve()
    source_path = Path(source_path).resolve()
    result_path = Path(result_path).resolve()
    permission_record = _file_record(authorization_record)
    if permission_record["size_bytes"] == 0:
        raise ValueError("Authorization record is empty")
    report, report_record, source_record, result_record = _validate_run(
        report_path, source_path, result_path)
    if indices[-1] >= report["result"]["frames"]:
        raise ValueError("Requested frame exceeds the processed output frame count")
    source_frames = _video_frames(source_path, indices)
    result_frames = _result_frames(result_path, report["output"]["type"], indices)

    private_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".{}-".format(case_id), dir=private_dir))
    try:
        frame_records = []
        for index in indices:
            stem = "frame_{:06d}".format(index)
            pair = {
                "index": index,
                "source_image": _write_png(staging / (stem + "_source.png"), source_frames[index]),
                "result_image": _write_png(staging / (stem + "_result.png"), result_frames[index]),
            }
            if report["output"]["type"] == "png_sequence":
                pair["alpha_image"] = _write_png(
                    staging / (stem + "_alpha.png"), result_frames[index][:, :, 3])
            frame_records.append(pair)
        manifest = {
            "schema_version": 1, "kind": "case_evidence", "case_id": case_id,
            "scenario": scenario, "authorized_use": authorized_use.strip(),
            "authorization_record": {"path": str(authorization_record), **permission_record},
            "run_report": {"path": str(report_path), **report_record},
            "source": {"path": str(source_path), **source_record},
            "result": {"path": str(result_path), "type": report["output"]["type"],
                       **result_record},
            "conditions": report["options"],
            "frame_pairing": "zero_based_frame_index_from_start_of_source_and_result",
            "frames": frame_records,
            "human_quality_assessment": None,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise ValueError("Case evidence appeared while collecting: {}".format(target))
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target / "manifest.json"


def _manifest_image(case_dir, record):
    if not isinstance(record, dict):
        raise ValueError("Evidence image record is missing")
    name = record.get("name")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".png"):
        raise ValueError("Evidence image name must be a local PNG basename")
    path = case_dir / name
    _check_file(path, record, "Evidence image")
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot decode evidence image: {}".format(path))
    return image


def verify_case_evidence(case_dir):
    """Recheck artifacts and that each saved screenshot decodes to its named frame."""
    case_dir = Path(case_dir).resolve()
    manifest = _load_json(case_dir / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "case_evidence":
        raise ValueError("Expected a case_evidence schema 1 manifest")
    auth, run, source, result = (manifest.get(key) for key in
                                 ("authorization_record", "run_report", "source", "result"))
    for label, item in (("Authorization", auth), ("Run report", run),
                        ("Source", source), ("Result", result)):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("{} path is missing from case manifest".format(label))
    _check_file(auth["path"], auth, "Authorization")
    _check_file(run["path"], run, "Run report")
    report, _, source_record, result_record = _validate_run(
        run["path"], source["path"], result["path"])
    if source_record != {key: source.get(key) for key in source_record}:
        raise ValueError("Case source identity changed")
    if result.get("type") != report["output"]["type"] or any(
            result.get(key) != value for key, value in result_record.items()):
        raise ValueError("Case result identity changed")
    if manifest.get("conditions") != report["options"]:
        raise ValueError("Case conditions differ from run report")
    pairs = manifest.get("frames")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Case manifest has no frame pairs")
    indices = [pair.get("index") for pair in pairs if isinstance(pair, dict)]
    if len(indices) != len(pairs) or any(isinstance(index, bool) or not isinstance(index, int)
                                        for index in indices) or indices != sorted(set(indices)) \
            or indices[0] < 0 or indices[-1] >= report["result"]["frames"]:
        raise ValueError("Case manifest has invalid frame indices")
    source_frames = _video_frames(source["path"], indices)
    result_frames = _result_frames(result["path"], result["type"], indices)
    for pair in pairs:
        index = pair["index"]
        if not np.array_equal(_manifest_image(case_dir, pair.get("source_image")),
                              source_frames[index]):
            raise ValueError("Source screenshot differs from frame {}".format(index))
        if not np.array_equal(_manifest_image(case_dir, pair.get("result_image")),
                              result_frames[index]):
            raise ValueError("Result screenshot differs from frame {}".format(index))
        if result["type"] == "png_sequence":
            if not np.array_equal(_manifest_image(case_dir, pair.get("alpha_image")),
                                  result_frames[index][:, :, 3]):
                raise ValueError("Alpha screenshot differs from frame {}".format(index))
        elif "alpha_image" in pair:
            raise ValueError("Video result cannot claim an alpha screenshot")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Collect a private case from a CLI run report")
    collect.add_argument("--case-id", required=True)
    collect.add_argument("--scenario", required=True, choices=SCENARIOS)
    collect.add_argument("--authorization-record", required=True)
    collect.add_argument("--authorized-use", required=True)
    collect.add_argument("--report", required=True)
    collect.add_argument("--source", required=True)
    collect.add_argument("--result", required=True)
    collect.add_argument("--frame", type=int, action="append", required=True)
    collect.add_argument("--private-dir", default=str(Path.home() /
                         "cv-human-video-matting-case-evidence"),
                         help="Private root outside the repository (default: user home)")
    verify = commands.add_parser("verify", help="Verify a collected case and its source artifacts")
    verify.add_argument("--case-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            path = collect_case_evidence(
                case_id=args.case_id, scenario=args.scenario,
                authorization_record=args.authorization_record, authorized_use=args.authorized_use,
                report_path=args.report, source_path=args.source, result_path=args.result,
                frame_indices=args.frame, private_dir=args.private_dir)
            print(path)
        else:
            verify_case_evidence(args.case_dir)
            print("Verified: {}".format(Path(args.case_dir).resolve() / "manifest.json"))
    except (OSError, ValueError) as exc:
        parser.exit(2, "Case evidence error: {}\n".format(exc))


if __name__ == "__main__":
    main()
