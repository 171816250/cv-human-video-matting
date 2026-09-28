"""Failure cleanup and provenance checks for the standalone local UI rehearsal."""

import importlib.util
import json
from pathlib import Path
import platform
import socket
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_local_demo.py"
spec = importlib.util.spec_from_file_location("demo_rehearsal", SCRIPT)
rehearsal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearsal)


class Process:
    pid = 12345
    returncode = None

    def __init__(self, hung=False):
        self.hung = hung
        self.terminated = self.killed = False
        self.waits = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.hung:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waits += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired("owned-service", timeout)
        return self.returncode


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    from cv_human_video_matting import reporting
    from cv_human_video_matting.io import video
    from cv_human_video_matting.utils import model_bundle

    source = tmp_path / "input.mp4"
    source.write_bytes(b"fixture")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    identity = {"model_name": "mobilenetv3", "content_sha256": "c" * 64, "bundle_path": str(bundle),
                "weights_path": "weights/model.pth", "files": [
                    {"path": "weights/model.pth", "sha256": "w" * 64, "size_bytes": 4}]}
    monkeypatch.setattr(model_bundle, "verify_model_bundle", lambda path: identity)
    monkeypatch.setattr(video, "probe_video", lambda path: SimpleNamespace(frame_count=3))
    monkeypatch.setattr(reporting, "_code_provenance", lambda: {"source_sha256": "s" * 64, "version": "test"})
    monkeypatch.setattr(rehearsal, "_free_port", lambda: 12345)
    args = SimpleNamespace(bundle=str(bundle), input=str(source), output_dir=str(tmp_path / "rehearsal"),
        device="cpu", max_frames=8, startup_timeout=1, request_timeout=1)
    return args, identity


@pytest.mark.parametrize("failure", ["spawn", "startup_exit", "startup_timeout", "request_timeout", "bad_report", "interrupt"])
def test_failures_leave_diagnostics_and_reap_only_owned_service(setup_run, monkeypatch, failure):
    from gradio_client import Client
    import gradio_client

    args, _ = setup_run
    process = Process(hung=failure == "request_timeout")
    client = SimpleNamespace(closed=False, shutdown=False)
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        assert command[0] == rehearsal.sys.executable
        assert "--no-share" in command and "--model-bundle" in command
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        if failure == "spawn":
            raise OSError("service executable unavailable")
        return process
    monkeypatch.setattr(rehearsal.subprocess, "Popen", popen)
    def ready(*unused):
        if failure == "startup_exit":
            process.returncode = 7
            raise RuntimeError("service exited")
        if failure == "startup_timeout":
            raise TimeoutError("service readiness timeout")
    monkeypatch.setattr(rehearsal, "_wait_ready", ready)
    def create_client(url, verbose, download_files):
        downloads = Path(download_files)
        downloads.mkdir()
        video, report = downloads / "result.mp4", downloads / "result.json"
        video.write_bytes(b"downloaded")
        report.write_text("{}")
        def result(timeout):
            if failure == "request_timeout":
                raise TimeoutError("request timeout")
            if failure == "interrupt":
                raise KeyboardInterrupt()
            return ({"value": {"video": str(video)}}, None, "done", {"value": str(report)})
        pending = SimpleNamespace(result=result, done=lambda: failure in ("bad_report",))
        stop = SimpleNamespace(result=lambda timeout: None)
        client.submit = lambda *values, **kwargs: stop if kwargs["api_name"] == "/stop_processing" else pending
        client.close = lambda: setattr(client, "closed", True)
        client.executor = SimpleNamespace(shutdown=lambda **kwargs: setattr(client, "shutdown", True))
        return client
    monkeypatch.setattr(gradio_client, "Client", create_client)
    monkeypatch.setattr(gradio_client, "handle_file", lambda path: path)
    expected = KeyboardInterrupt if failure == "interrupt" else Exception
    with pytest.raises(expected):
        rehearsal.run_rehearsal(args)
    output = Path(args.output_dir)
    evidence = json.loads((output / "rehearsal.json").read_text(encoding="utf-8"))
    assert evidence["passed"] is False and evidence["error_type"]
    assert (output / "server.log").is_file() and (output / "client.log").is_file()
    assert not list(output.glob("runtime-*"))
    assert len(calls) == 1
    if failure == "spawn":
        assert not process.terminated and not process.killed
    else:
        assert process.waits >= 1 and evidence["service_reaped"] is True
        if failure == "startup_exit":
            assert not process.terminated
        else:
            assert process.terminated
    if failure == "request_timeout":
        assert process.killed and process.waits == 2
    if failure in ("bad_report", "request_timeout", "interrupt"):
        assert client.closed and client.shutdown


def test_existing_output_directory_is_not_modified(setup_run):
    args, _ = setup_run
    output = Path(args.output_dir)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        rehearsal.run_rehearsal(args)
    assert marker.read_text() == "existing evidence"
    assert list(output.iterdir()) == [marker]


def test_outputs_cannot_pollute_bundle(setup_run):
    args, _ = setup_run
    args.output_dir = str(Path(args.bundle) / "rehearsal")
    with pytest.raises(ValueError, match="outside"):
        rehearsal.run_rehearsal(args)
    assert not Path(args.output_dir).exists()


def test_local_guard_blocks_remote_resolution_and_connections():
    with rehearsal.local_network_only():
        assert socket.getaddrinfo("127.0.0.1", 80)
        with pytest.raises(RuntimeError, match="non-loopback"):
            socket.getaddrinfo("example.invalid", 443)
        with socket.socket() as channel:
            with pytest.raises(RuntimeError, match="non-loopback"):
                channel.connect(("192.0.2.1", 443))
            with pytest.raises(RuntimeError, match="non-loopback"):
                channel.connect_ex(("192.0.2.1", 443))


@pytest.mark.parametrize("problem", [None, "frames", "output_hash", "model_hash", "code_hash", "environment", "model_unconfirmed",
    "option:mode", "option:max_frames", "option:background", "option:checkpoint",
    "option:edge_feather", "option:mask_open", "option:mask_close", "option:temporal_smooth",
    "option:temporal_blend", "option:temporal_mode", "option:temporal_motion_threshold",
    "option:enable_depth", "option:depth_interval", "option:preserve_audio", "option:resize_short",
    "option:background_color", "option:model", "option:device", "option:model_bundle",
    "option:intermediate_codec", "missing:background", "missing:resize_short"])
def test_downloaded_report_must_match_actual_bytes_and_loaded_identity(tmp_path, monkeypatch, problem):
    output = tmp_path / "server-output"
    output.mkdir()
    published = output / "result.mp4"
    writer = cv2.VideoWriter(str(published), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((24, 32, 3), 120, np.uint8))
    writer.release()
    downloaded = tmp_path / "downloaded.mp4"
    downloaded.write_bytes(published.read_bytes())
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")
    weights = {"path": "weights/model.pth", "sha256": "w" * 64, "size_bytes": 4}
    bundle = {"content_sha256": "b" * 64, "model_name": "mobilenetv3", "bundle_path": str(tmp_path / "bundle"),
              "weights_path": weights["path"], "files": [weights]}
    code = {"source_sha256": "c" * 64, "version": "test"}
    monkeypatch.setattr(rehearsal, "version", lambda name: "1.0")
    report = {"schema_version": 3, "kind": "run_report", "input": {"sha256": rehearsal._sha256(source)},
        "output": {"sha256": rehearsal._sha256(downloaded), "size_bytes": downloaded.stat().st_size,
                   "frame_count": 3, "width": 32, "height": 24},
        "result": {"frames": 3, "output_path": str(published)}, "code": code.copy(),
        "assets": {"model_bundle": {"load_identity_confirmed": True, "content_sha256": bundle["content_sha256"],
                    "model_name": "mobilenetv3", "weights": weights}},
        "options": {"model": "rvm_mobilenetv3", "checkpoint": None, "device": "cpu", "mode": "side_by_side",
            "model_bundle": bundle["bundle_path"], "max_frames": 8, "background": None,
            "edge_feather": 0, "mask_open": 0, "mask_close": 0, "temporal_smooth": True, "temporal_blend": 0.75,
            "temporal_mode": "adaptive", "temporal_motion_threshold": 0.15, "intermediate_codec": "FFV1",
            "enable_depth": False, "depth_interval": 5, "resize_short": None,
            "background_color": "#00FF00", "preserve_audio": True},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
            "packages": {name: "1.0" for name in ("torch", "torchvision", "numpy", "opencv-python", "gradio", "imageio-ffmpeg")}}}
    if problem == "frames":
        report["result"]["frames"] = 2
    elif problem == "output_hash":
        report["output"]["sha256"] = "0" * 64
    elif problem == "model_hash":
        report["assets"]["model_bundle"]["content_sha256"] = "0" * 64
    elif problem == "code_hash":
        report["code"]["source_sha256"] = "0" * 64
    elif problem == "environment":
        report["environment"]["packages"]["torch"] = "other"
    elif problem == "model_unconfirmed":
        report["assets"]["model_bundle"]["load_identity_confirmed"] = False
    elif problem and problem.startswith("option:"):
        report["options"][problem.split(":", 1)[1]] = "wrong-control-value"
    elif problem and problem.startswith("missing:"):
        del report["options"][problem.split(":", 1)[1]]
    args = (report, downloaded, source, SimpleNamespace(frame_count=3), bundle, code, "cpu", 8, output)
    if problem:
        with pytest.raises(ValueError, match="verification failed"):
            rehearsal._check_report(*args)
    else:
        assert rehearsal._check_report(*args)["frames"] == 3


def test_download_must_be_inside_client_directory(tmp_path):
    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        rehearsal._download_path({"value": str(foreign)}, ".json", tmp_path / "downloads")


def test_relative_new_output_is_absolute_before_child_changes_cwd(setup_run, monkeypatch):
    args, _ = setup_run
    caller = Path(args.input).parent
    monkeypatch.chdir(caller)
    args.input, args.bundle, args.output_dir = "input.mp4", "bundle", "new evidence"
    original_resolve = Path.resolve

    def legacy_resolve(path, *values, **kwargs):
        # Windows Python 3.9 can return nonexistent relative paths unchanged.
        if not path.is_absolute() and not path.exists():
            return path
        return original_resolve(path, *values, **kwargs)

    monkeypatch.setattr(Path, "resolve", legacy_resolve)
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        raise OSError("stop before any service is launched")

    monkeypatch.setattr(rehearsal.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="stop before"):
        rehearsal.run_rehearsal(args)
    command, options = calls[0]
    config = Path(command[command.index("--config") + 1])
    assert config == caller / "new evidence/config.yaml"
    assert config.is_absolute() and config.is_file()
    assert Path(command[command.index("--model-bundle") + 1]) == caller / "bundle"
    assert options["cwd"] == rehearsal.ROOT
    assert (caller / "new evidence/rehearsal.json").is_file()
