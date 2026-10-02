"""Protocol metrics and owned-process cleanup without loading model weights."""

import asyncio
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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_realtime.py"
spec = importlib.util.spec_from_file_location("realtime_rehearsal", SCRIPT)
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
            raise subprocess.TimeoutExpired("owned-realtime-service", timeout)
        return self.returncode


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    from cv_human_video_matting import reporting
    from cv_human_video_matting.utils import model_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    identity = {"model_name": "mobilenetv3", "content_sha256": "b" * 64, "bundle_path": str(bundle),
                "weights_path": "weights/model.pth", "files": [
                    {"path": "weights/model.pth", "sha256": "w" * 64, "size_bytes": 4}]}
    code = {"source_sha256": "s" * 64, "version": "test"}
    monkeypatch.setattr(model_bundle, "verify_model_bundle", lambda path: identity)
    monkeypatch.setattr(reporting, "_code_provenance", lambda: code)
    monkeypatch.setattr(rehearsal, "_free_port", lambda: 12345)
    args = SimpleNamespace(bundle=str(bundle), input=None, output_dir=str(tmp_path / "rehearsal"),
                           device="cpu", frames=120, width=320, warmup=10, startup_timeout=1, request_timeout=1)
    return args, identity, code


def handshake(bundle, code):
    shared = {"protocol": 1, "model": "rvm_mobilenetv3", "device": "cpu",
              "bundle_content_sha256": bundle["content_sha256"]}
    def installed(name):
        try:
            return rehearsal.version(name)
        except rehearsal.PackageNotFoundError:
            return None

    environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
                   "device_error": None, "packages": {name: installed(name) for name in
                   ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless")}}
    return ({"type": "hello", **shared}, {"type": "ready", **shared,
        "package_version": code["version"], "source_sha256": code["source_sha256"],
        "model_content_sha256": bundle["content_sha256"], "weights": bundle["files"][0],
        "model_load_ms": 12.0, "environment": environment})


class WebSocket:
    def __init__(self, bundle, code, failure=None):
        self.messages = [json.dumps(message) for message in handshake(bundle, code)]
        self.closed = False
        self.outstanding = False
        self.sent_frames = 0
        self.failure = failure

    async def send(self, value):
        if isinstance(value, str):
            configuration = json.loads(value)
            assert configuration == {"type": "configure", "mode": "green", "max_width": 320}
            self.messages.append(json.dumps({"type": "configured", "mode": "green", "max_width": 320}))
            return
        assert not self.outstanding, "must receive a complete JPEG before sending the next frame"
        self.outstanding = True
        self.sent_frames += 1
        seq = self.sent_frames
        decoded = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
        height, width = decoded.shape[:2]
        metrics = {"type": "frame", "seq": seq, "width": width, "height": height,
                   "processing_ms": 100.0 if seq <= 10 else 10.0, "cold": seq == 1,
                   "warmup": seq <= 10, "device": "cpu", "cuda_peak_allocated_bytes": None,
                   "cuda_peak_reserved_bytes": None}
        if self.failure == "sequence":
            metrics["seq"] = 99
        if self.failure == "nan":
            metrics["processing_ms"] = float("nan")
        if self.failure == "dimensions":
            metrics["width"] += 1
        if self.failure == "memory":
            metrics["cuda_peak_allocated_bytes"] = 100
        self.messages.append(json.dumps(metrics))
        self.messages.append(b"not a JPEG" if self.failure == "bad_jpeg" else value)

    async def recv(self):
        if self.failure == "timeout" and self.sent_frames:
            raise asyncio.TimeoutError("frame timed out")
        value = self.messages.pop(0)
        if isinstance(value, bytes):
            self.outstanding = False
        return value

    async def close(self):
        self.closed = True


def exercise_fixture(setup_run, monkeypatch, failure=None):
    from websockets.legacy import client

    args, bundle, code = setup_run
    connection = WebSocket(bundle, code, failure)

    async def connect(url, **kwargs):
        assert url == "ws://127.0.0.1:12345/ws"
        assert kwargs["origin"] == "http://127.0.0.1:12345"
        return connection

    monkeypatch.setattr(client, "connect", connect)
    monkeypatch.setattr(rehearsal, "_source_frames", lambda source, count, evidence:
                        (np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(count)))
    evidence = {"frames": [], "source": {}}
    call = rehearsal._exercise(Process(), "ws://127.0.0.1:12345", args, bundle, code,
                               evidence, rehearsal.time.perf_counter())
    return call, connection, evidence


def test_single_frame_in_flight_excludes_warmup_from_metrics(setup_run, monkeypatch):
    call, connection, evidence = exercise_fixture(setup_run, monkeypatch)
    asyncio.run(call)
    assert connection.closed and connection.sent_frames == 120
    assert len(evidence["frames"]) == 120
    assert sum(frame["excluded_warmup"] for frame in evidence["frames"]) == 10
    metrics = evidence["metrics"]
    assert metrics["warm_frame_count"] == 110
    assert metrics["warm_service_fps"] == 100.0
    assert metrics["warm_processing_ms"]["p50"] == metrics["warm_processing_ms"]["p95"] == 10.0
    assert metrics["warm_round_trip_ms"]["count"] == 110
    assert metrics["warm_delivered_fps"] > 0
    assert metrics["cuda_peak_allocated_bytes"] is None
    assert evidence["first_frame_round_trip_ms"] > 0
    assert evidence["startup_to_first_result_seconds"] >= evidence["startup_to_ready_seconds"]


@pytest.mark.parametrize("failure", ["sequence", "nan", "dimensions", "memory", "bad_jpeg", "timeout"])
def test_bad_frames_or_timeout_close_websocket_and_never_publish_metrics(setup_run, monkeypatch, failure):
    call, connection, evidence = exercise_fixture(setup_run, monkeypatch, failure)
    with pytest.raises((ValueError, asyncio.TimeoutError)):
        asyncio.run(call)
    assert connection.closed
    assert "metrics" not in evidence
    assert evidence["frames"] == []


@pytest.mark.parametrize("changed", ["model_content_sha256", "source_sha256", "weights", "package_version", "environment"])
def test_ready_must_confirm_loaded_source_weights_and_environment(setup_run, changed):
    _, bundle, code = setup_run
    hello, ready = handshake(bundle, code)
    ready[changed] = {} if changed == "environment" else "wrong"
    with pytest.raises(ValueError, match="mismatch|environment"):
        rehearsal._check_ready(hello, ready, bundle, code, "cpu")


def test_headless_opencv_is_recorded_and_accepted(setup_run, monkeypatch):
    from cv_human_video_matting import reporting

    actual_client_version = rehearsal.version
    actual_server_version = reporting.version

    def headless_version(actual):
        def lookup(name):
            if name == "opencv-python":
                raise rehearsal.PackageNotFoundError(name)
            if name == "opencv-python-headless":
                return "4.10.0.84"
            return actual(name)
        return lookup

    monkeypatch.setattr(rehearsal, "version", headless_version(actual_client_version))
    monkeypatch.setattr(reporting, "version", headless_version(actual_server_version))
    _, bundle, code = setup_run
    hello, ready = handshake(bundle, code)
    packages = ready["environment"]["packages"]
    assert packages["opencv-python"] is None
    assert packages["opencv-python-headless"] == "4.10.0.84"
    assert reporting._environment("cpu")["packages"]["opencv-python-headless"] == "4.10.0.84"
    rehearsal._check_ready(hello, ready, bundle, code, "cpu")


@pytest.mark.parametrize("failure", ["spawn", "startup_exit", "startup_timeout", "request_timeout", "bad_frame", "interrupt"])
def test_failure_evidence_and_owned_process_cleanup(setup_run, monkeypatch, failure):
    args, _, _ = setup_run
    process = Process(hung=failure == "request_timeout")
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        assert command[0] == rehearsal.sys.executable
        assert "realtime" in command and "--model-bundle" in command
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["GRADIO_ANALYTICS_ENABLED"] == "False"
        assert not Path(kwargs["env"]["TORCH_HOME"]).exists()
        if failure == "spawn":
            raise OSError("spawn failed")
        return process

    async def exercise(*unused):
        if failure == "startup_exit":
            process.returncode = 7
            raise RuntimeError("service exited")
        if failure in ("startup_timeout", "request_timeout"):
            raise TimeoutError(failure)
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise ValueError("bad JPEG result")

    monkeypatch.setattr(rehearsal.subprocess, "Popen", popen)
    monkeypatch.setattr(rehearsal, "_exercise", exercise)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else Exception):
        rehearsal.run_rehearsal(args)
    output = Path(args.output_dir)
    report = json.loads((output / "rehearsal.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["error_type"]
    assert (output / "server.log").is_file() and (output / "client.log").is_file()
    assert not list(output.glob("runtime-*"))
    assert len(commands) == 1
    if failure == "spawn":
        assert process.waits == 0
    else:
        assert report["service_reaped"] is True and process.waits >= 1
    if failure == "request_timeout":
        assert process.killed


def test_existing_outputs_and_model_directory_are_protected(setup_run):
    args, _, _ = setup_run
    output = Path(args.output_dir)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old evidence")
    with pytest.raises(FileExistsError):
        rehearsal.run_rehearsal(args)
    assert list(output.iterdir()) == [marker] and marker.read_text() == "old evidence"
    args.output_dir = str(Path(args.bundle) / "new-evidence")
    with pytest.raises(ValueError, match="outside"):
        rehearsal.run_rehearsal(args)
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize("name,value", [("frames", 119), ("warmup", 9), ("warmup", 120),
                                       ("width", 123), ("device", "auto"), ("request_timeout", float("nan"))])
def test_invalid_requests_do_not_create_outputs(setup_run, name, value):
    args, _, _ = setup_run
    setattr(args, name, value)
    with pytest.raises(ValueError):
        rehearsal.run_rehearsal(args)
    assert not Path(args.output_dir).exists()


def test_pre_resize_never_upscales_and_preserves_browser_rounding():
    payload, width, height = rehearsal._encode_frame(np.zeros((481, 1000, 3), np.uint8), 640)
    assert (width, height) == (640, 308)
    assert cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (308, 640)
    _, width, height = rehearsal._encode_frame(np.zeros((24, 32, 3), np.uint8), 320)
    assert (width, height) == (32, 24)


def test_short_video_repetition_is_explicit_and_capture_released(tmp_path):
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.write(np.full((24, 32, 3), 100, np.uint8))
    writer.release()
    evidence = {"source": {}}
    frames = list(rehearsal._source_frames(path, 5, evidence))
    assert len(frames) == 5 and evidence["source"]["loop_count"] == 2
    assert np.array_equal(frames[0], frames[2])
    path.unlink()  # Windows refuses removal while a capture handle remains open.


def test_network_guard_records_and_blocks_external_connections(tmp_path):
    log = tmp_path / "blocked.jsonl"
    with rehearsal.local_network_only(log):
        assert socket.getaddrinfo("127.0.0.1", 80)
        with pytest.raises(RuntimeError, match="non-loopback"):
            socket.getaddrinfo("example.invalid", 443)
        with socket.socket() as channel:
            with pytest.raises(RuntimeError, match="non-loopback"):
                channel.connect(("192.0.2.1", 443))
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_percentiles_are_interpolated_and_reject_nonfinite_metrics():
    values = rehearsal._summary([1.0, 2.0, 3.0, 4.0])
    assert values["p50"] == 2.5 and values["p95"] == pytest.approx(3.85)
    with pytest.raises(ValueError):
        rehearsal._summary([float("nan")])
