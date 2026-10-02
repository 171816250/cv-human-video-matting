"""Benchmark the real local WebSocket service with one JPEG frame in flight.

This is an unpaced protocol/service rehearsal, not a browser webcam test, a
real-person quality evaluation, or an operating-system firewall verification.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from verify_local_demo import _free_port, _stop_process  # noqa: E402


@contextmanager
def local_network_only(log_path=None, *, guard_downloads=False):
    """Block non-loopback Python TCP/DNS and optionally Torch Hub downloads."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_resolve = socket.getaddrinfo

    def reject(detail):
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"blocked": str(detail)}, ensure_ascii=False) + "\n")
        raise RuntimeError(f"Realtime rehearsal blocked non-loopback networking/download: {detail}")

    def allow(host):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if host == "localhost":
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        reject(host)

    def connect(channel, address):
        allow(address[0])
        return original_connect(channel, address)

    def connect_ex(channel, address):
        allow(address[0])
        return original_connect_ex(channel, address)

    def resolve(host, *args, **kwargs):
        allow(host)
        return original_resolve(host, *args, **kwargs)

    with patch.object(socket.socket, "connect", connect), \
            patch.object(socket.socket, "connect_ex", connect_ex), \
            patch.object(socket, "getaddrinfo", resolve):
        if guard_downloads:
            import torch
            with patch.object(torch.hub, "download_url_to_file", lambda *a, **k: reject("torch.hub download")):
                yield
        else:
            yield


def _file_identity(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _summary(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Metrics require nonempty finite, nonnegative measurements")
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)

    return {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
            "p50": percentile(0.5), "p95": percentile(0.95), "mean": sum(ordered) / len(ordered)}


def _encode_frame(frame, width):
    import cv2

    height, original_width = frame.shape[:2]
    scaled_width = min(width, original_width)
    scaled_height = max(1, int(height * scaled_width / original_width + 0.5))
    if (scaled_width, scaled_height) != (original_width, height):
        frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not encode the input JPEG frame")
    payload = encoded.tobytes()
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("Encoded input frame exceeds the service's 2 MiB limit")
    return payload, scaled_width, scaled_height


def _source_frames(source, count, evidence):
    """Loop a short input explicitly; synthetic frames are deterministic."""
    import cv2
    import numpy as np

    capture = None
    try:
        if source is not None:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError(f"Cannot decode source video: {source}")
        evidence["source"]["loop_count"] = 0
        for index in range(count):
            if capture is None:
                frame = np.full((720, 1280, 3), (45, 80, 110), dtype=np.uint8)
                x = (index * 17) % 1120
                cv2.rectangle(frame, (x, 160), (x + 160, 560), (120, 190, 220), -1)
                cv2.circle(frame, (640, 160 + index % 320), 65, (200, 80, 90), -1)
            else:
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    capture = cv2.VideoCapture(str(source))
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Source video has no decodable frames: {source}")
                    evidence["source"]["loop_count"] += 1
            yield frame
    finally:
        if capture is not None:
            capture.release()


async def _json_message(websocket, expected, timeout):
    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    if not isinstance(raw, str):
        raise ValueError(f"Expected {expected} JSON before binary JPEG")
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError(f"Expected {expected} JSON object")
    if message.get("type") == "error":
        raise RuntimeError(f"Service error {message.get('code')}: {message.get('message')}")
    if message.get("type") != expected:
        raise ValueError(f"Expected {expected}, received {message.get('type')}")
    return message


def _check_ready(hello, ready, bundle, code, requested_device):
    expected_model = "rvm_" + bundle["model_name"]
    for message in (hello, ready):
        if message.get("protocol") != 1 or message.get("model") not in (expected_model, bundle["model_name"]):
            raise ValueError("Service protocol/model identity mismatch")
        if message.get("bundle_content_sha256") != bundle["content_sha256"]:
            raise ValueError("Service model bundle hash mismatch")
    device = ready.get("device")
    if not isinstance(device, str) or device != hello.get("device"):
        raise ValueError("Service device identity mismatch")
    if not (device == requested_device or requested_device == "cuda" and device.startswith("cuda:")):
        raise ValueError(f"Requested device {requested_device}, service resolved {device}")
    weights = next(item for item in bundle["files"] if item["path"] == bundle["weights_path"])
    expected = {"package_version": code["version"], "source_sha256": code["source_sha256"],
                "model_content_sha256": bundle["content_sha256"], "weights": weights}
    for name, value in expected.items():
        if ready.get(name) != value:
            raise ValueError(f"Service loaded provenance mismatch: {name}")
    environment = ready.get("environment", {})
    if environment.get("device") != device or environment.get("device_error") is not None:
        raise ValueError("Service environment does not confirm the inference device")
    if environment.get("python") != platform.python_version() or environment.get("platform") != platform.platform():
        raise ValueError("Service environment differs from the selected interpreter/platform")
    for name in ("torch", "torchvision", "numpy"):
        if environment.get("packages", {}).get(name) != version(name):
            raise ValueError(f"Service package version mismatch: {name}")
    opencv = {}
    for name in ("opencv-python", "opencv-python-headless"):
        try:
            opencv[name] = version(name)
        except PackageNotFoundError:
            opencv[name] = None
        if environment.get("packages", {}).get(name) != opencv[name]:
            raise ValueError(f"Service package version mismatch: {name}")
    if not any(opencv.values()):
        raise ValueError("Neither supported OpenCV distribution is installed")


def _check_frame(message, payload, sequence, width, height, device):
    import cv2
    import numpy as np

    if message.get("seq") != sequence or isinstance(message.get("seq"), bool):
        raise ValueError("Frame sequence mismatch; one frame must be in flight")
    if message.get("cold") is not (sequence == 1) or message.get("warmup") is not (sequence <= 10):
        raise ValueError("Service cold/warmup markers disagree with protocol 1")
    if message.get("device") != device:
        raise ValueError("Frame device differs from the ready handshake")
    duration = message.get("processing_ms")
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid service processing_ms")
    if not isinstance(payload, bytes) or not payload.startswith(b"\xff\xd8"):
        raise ValueError("Expected a binary JPEG result")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError("Returned JPEG dimensions differ from the submitted green frame")
    if (message.get("width"), message.get("height")) != (width, height):
        raise ValueError("Frame dimensions differ from the decoded JPEG")
    for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = message.get(name)
        if name not in message or (device == "cpu" and value is not None) or (
            device != "cpu" and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
        ):
            raise ValueError(f"Invalid peak memory metric: {name}")
    return {"seq": sequence, "width": width, "height": height, "processing_ms": float(duration),
            "cuda_peak_allocated_bytes": message["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": message["cuda_peak_reserved_bytes"],
            "output_jpeg_sha256": hashlib.sha256(payload).hexdigest(), "output_bytes": len(payload)}


async def _exercise(process, url, args, bundle, code, evidence, started):
    from websockets.legacy.client import connect

    deadline = time.perf_counter() + args.startup_timeout
    websocket = None
    frames = None
    try:
        while websocket is None:
            if process.poll() is not None:
                raise RuntimeError(f"Service exited before readiness: {process.returncode}; see server.log")
            if time.perf_counter() >= deadline:
                raise TimeoutError("WebSocket startup timeout; see server.log")
            try:
                websocket = await connect(url + "/ws", origin=url.replace("ws://", "http://"),
                    open_timeout=min(1, max(0.01, deadline - time.perf_counter())), close_timeout=2,
                    max_size=8 * 1024 * 1024, compression=None, ping_interval=None)
            except (OSError, asyncio.TimeoutError):
                await asyncio.sleep(0.1)
        hello = await _json_message(websocket, "hello", max(0.01, deadline - time.perf_counter()))
        ready = await _json_message(websocket, "ready", max(0.01, deadline - time.perf_counter()))
        _check_ready(hello, ready, bundle, code, args.device)
        evidence.update(ready=ready, startup_to_ready_seconds=time.perf_counter() - started)
        await asyncio.wait_for(websocket.send(json.dumps(
            {"type": "configure", "mode": "green", "max_width": args.width})), args.request_timeout)
        configured = await _json_message(websocket, "configured", args.request_timeout)
        if configured.get("mode") != "green" or configured.get("max_width") != args.width:
            raise ValueError("Service did not apply requested green mode/width")
        frames = _source_frames(args.input, args.frames, evidence)
        warm_started = None
        for index in range(args.frames):
            if index == args.warmup:
                warm_started = time.perf_counter()
            prepared = time.perf_counter()
            payload, width, height = _encode_frame(next(frames), args.width)
            preparation_ms = (time.perf_counter() - prepared) * 1000
            sent = time.perf_counter()
            await asyncio.wait_for(websocket.send(payload), args.request_timeout)
            remaining = args.request_timeout - (time.perf_counter() - sent)
            if remaining <= 0:
                raise TimeoutError("Frame send timeout")
            message = await _json_message(websocket, "frame", remaining)
            remaining = args.request_timeout - (time.perf_counter() - sent)
            if remaining <= 0:
                raise TimeoutError("Frame round-trip timeout")
            returned = await asyncio.wait_for(websocket.recv(), remaining)
            frame = _check_frame(message, returned, index + 1, width, height, ready["device"])
            completed = time.perf_counter()
            frame.update(round_trip_ms=(completed - sent) * 1000, client_prepare_ms=preparation_ms,
                         excluded_warmup=index < args.warmup, input_jpeg_sha256=hashlib.sha256(payload).hexdigest(),
                         input_bytes=len(payload))
            evidence["frames"].append(frame)
            if index == 0:
                evidence.update(first_frame_round_trip_ms=frame["round_trip_ms"],
                                startup_to_first_result_seconds=completed - started)
        warm_seconds = time.perf_counter() - warm_started
        warm = evidence["frames"][args.warmup:]
        evidence["metrics"] = {
            "warm_frame_count": len(warm), "warm_wall_seconds": warm_seconds,
            "warm_delivered_fps": len(warm) / warm_seconds,
            "warm_service_fps": 1000 * len(warm) / sum(item["processing_ms"] for item in warm),
            "warm_processing_ms": _summary([item["processing_ms"] for item in warm]),
            "warm_round_trip_ms": _summary([item["round_trip_ms"] for item in warm]),
            "cuda_peak_allocated_bytes": max((item["cuda_peak_allocated_bytes"] for item in evidence["frames"]
                                              if item["cuda_peak_allocated_bytes"] is not None), default=None),
            "cuda_peak_reserved_bytes": max((item["cuda_peak_reserved_bytes"] for item in evidence["frames"]
                                             if item["cuda_peak_reserved_bytes"] is not None), default=None),
        }
    finally:
        if frames is not None:
            frames.close()
        if websocket is not None:
            await websocket.close()


def run_rehearsal(args):
    from cv_human_video_matting.reporting import _atomic_write_text, _code_provenance
    from cv_human_video_matting.utils.model_bundle import verify_model_bundle

    if args.frames < 120 or args.warmup < 10 or args.warmup >= args.frames:
        raise ValueError("Use at least 120 frames and 10 warmup frames, with warmup < frames")
    if args.width not in (320, 640, 960, 1280):
        raise ValueError("Width must be 320, 640, 960 or 1280")
    if args.device != "cpu" and not (args.device == "cuda" or args.device.startswith("cuda:") and args.device[5:].isdigit()):
        raise ValueError("Device must be cpu, cuda or cuda:<index>")
    if any(not math.isfinite(value) or value <= 0 for value in (args.startup_timeout, args.request_timeout)):
        raise ValueError("Timeouts must be finite and positive")
    bundle_path, output = (Path(value).absolute().resolve() for value in (args.bundle, args.output_dir))
    if output.is_relative_to(bundle_path):
        raise ValueError("Output directory must be outside the model bundle")
    bundle, code = verify_model_bundle(bundle_path), _code_provenance()
    source = Path(args.input).absolute().resolve() if args.input is not None else None
    source_identity = ({"kind": "video", **_file_identity(source), "repeat_policy": "loop on EOF"} if source else
                       {"kind": "synthetic", "generator": "moving-shapes-v1", "width": 1280, "height": 720})
    output.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless",
                 "websockets", "fastapi", "uvicorn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    evidence = {"schema_version": 1, "kind": "realtime_service_rehearsal", "passed": False,
        "created_at": datetime.now(timezone.utc).isoformat(), "source": source_identity, "code": code,
        "verifier": _file_identity(Path(__file__).resolve()), "bundle_content_sha256": bundle["content_sha256"],
        "environment": {"python": platform.python_version(), "python_executable": sys.executable,
                        "platform": platform.platform(), "packages": packages},
        "requested": {"frames": args.frames, "warmup": args.warmup, "width": args.width,
                      "device": args.device, "mode": "green", "jpeg_quality": 80, "in_flight": 1},
        "scope": "Unpaced synthetic/video WebSocket service benchmark; not browser webcam latency or real-person quality.",
        "timing_scope": "processing_ms: backend decode through JPEG encode; round_trip_ms: send through client JPEG decode. Warm delivered FPS also includes source reading and client resize/encode. Startup and model loading are separate.",
        "network_scope": "Python TCP/DNS restricted to loopback and Torch Hub download entry blocked; not an OS firewall.",
        "frames": []}
    process = None
    error = None
    network_log = output / "blocked-network.jsonl"
    try:
        with TemporaryDirectory(prefix="runtime-", dir=output) as runtime, \
                (output / "server.log").open("w", encoding="utf-8") as log:
            env = {key: value for key, value in os.environ.items()
                   if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
            torch_home = Path(runtime) / "torch"
            env.update({"NO_PROXY": "127.0.0.1,localhost,::1", "GRADIO_ANALYTICS_ENABLED": "False",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "TORCH_HOME": str(torch_home), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                        "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
            bootstrap = ("import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                         "from verify_realtime import local_network_only; log=sys.argv.pop(1); "
                         "sys.argv=sys.argv[1:]; guard=local_network_only(log,guard_downloads=True); "
                         "guard.__enter__(); runpy.run_path(sys.argv[0],run_name='__main__')")
            port = _free_port()
            url = f"ws://127.0.0.1:{port}"
            command = [sys.executable, "-u", "-c", bootstrap, str(ROOT / "scripts"), str(network_log),
                       str(ROOT / "app.py"), "realtime", "--model-bundle", str(bundle_path),
                       "--device", args.device, "--host", "127.0.0.1", "--port", str(port)]
            started = time.perf_counter()
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                evidence.update(service_pid=process.pid, url=url)
                with local_network_only(network_log):
                    asyncio.run(_exercise(process, url, args, bundle, code, evidence, started))
                if verify_model_bundle(bundle_path)["content_sha256"] != bundle["content_sha256"]:
                    raise ValueError("Model bundle changed during the rehearsal")
                if _code_provenance()["source_sha256"] != code["source_sha256"]:
                    raise ValueError("Service source code changed during the rehearsal")
                if source is not None and _file_identity(source)["sha256"] != source_identity["sha256"]:
                    raise ValueError("Source video changed during the rehearsal")
                evidence["empty_torch_cache_entries"] = sorted(p.relative_to(torch_home).as_posix()
                                                               for p in torch_home.rglob("*"))
                if evidence["empty_torch_cache_entries"]:
                    raise ValueError("Service wrote into its empty Torch cache")
                evidence["passed"] = True
            finally:
                if process is not None:
                    evidence["service_exit_code"] = _stop_process(process)
                    evidence["service_reaped"] = process.poll() is not None
        if network_log.exists():
            evidence["blocked_attempts"] = network_log.read_text(encoding="utf-8").splitlines()
            raise RuntimeError("Network/download attempts were blocked during the rehearsal")
        evidence["blocked_attempts"] = []
    except BaseException as exc:
        evidence.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        (output / "client.log").write_text(traceback.format_exc(), encoding="utf-8")
        error = exc
    finally:
        evidence["blocked_attempts"] = (network_log.read_text(encoding="utf-8").splitlines()
                                        if network_log.exists() else [])
        if not (output / "client.log").exists():
            (output / "client.log").write_text("Protocol, decoded frames and provenance checks passed.\n", encoding="utf-8")
        _atomic_write_text(output / "rehearsal.json", json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if error is not None:
        raise error
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", help="Optional video; short videos repeat and repetition is recorded")
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths are refused")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640, choices=(320, 640, 960, 1280))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=30)
    args = parser.parse_args(argv)
    run_rehearsal(args)
    print(f"Realtime service rehearsal passed: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
