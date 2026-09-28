"""Rehearse the actual local Gradio service with an existing offline RVM bundle.

This checks the supplied sample, downloads and provenance. It does not certify
real-person matting quality, webcam behavior, or an operating-system firewall.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
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


@contextmanager
def local_network_only():
    """Guard Python TCP connections and name resolution; not an OS firewall."""
    connect, connect_ex, resolve = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

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
        raise RuntimeError(f"Local demo rehearsal blocked non-loopback networking: {host}")

    def guarded_connect(channel, address):
        allow(address[0])
        return connect(channel, address)

    def guarded_connect_ex(channel, address):
        allow(address[0])
        return connect_ex(channel, address)

    def guarded_resolve(host, *args, **kwargs):
        allow(host)
        return resolve(host, *args, **kwargs)

    with patch.object(socket.socket, "connect", guarded_connect), \
            patch.object(socket.socket, "connect_ex", guarded_connect_ex), \
            patch.object(socket, "getaddrinfo", guarded_resolve):
        yield


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _free_port():
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        return channel.getsockname()[1]


def _wait_ready(process, url, timeout):
    import httpx

    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=1) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"UI service exited before readiness (code {process.returncode}); see server.log")
            try:
                response = client.get(url + "/config")
                if response.status_code == 200 and isinstance(response.json(), dict):
                    if process.poll() is None:
                        return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.2)
    raise TimeoutError(f"UI service did not become ready within {timeout}s; see server.log")


def _stop_process(process):
    """Only stop and reap the exact Popen instance created by this rehearsal."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    else:
        process.wait(timeout=5)
    return process.returncode


def _download_path(value, suffix, downloads):
    candidates = set()

    def visit(item):
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str) and item.lower().endswith(suffix):
            path = Path(item)
            if path.is_file():
                path = path.resolve()
                if not path.is_relative_to(downloads.resolve()):
                    raise ValueError("Client returned a file outside its download directory")
                candidates.add(path)

    visit(value)
    if len(candidates) != 1:
        raise ValueError(f"Expected one downloaded {suffix} file, found {len(candidates)}")
    return candidates.pop()


def _check_report(report, video, source, source_info, bundle, code, device, limit, server_output):
    import cv2

    def require(condition, description):
        if not condition:
            raise ValueError(f"Downloaded report verification failed: {description}")

    require(report.get("schema_version") == 3 and report.get("kind") == "run_report", "schema/kind")
    source_record, output = report.get("input", {}), report.get("output", {})
    result, environment = report.get("result", {}), report.get("environment", {})
    options, recorded_code = report.get("options", {}), report.get("code", {})
    require(source_record.get("sha256") == _sha256(source), "input SHA256")
    require(output.get("sha256") == _sha256(video), "downloaded video SHA256")
    require(output.get("size_bytes") == video.stat().st_size, "downloaded video size")
    published = Path(result.get("output_path", "")).resolve()
    require(published.is_relative_to(server_output.resolve()) and published.is_file(), "service-owned output path")
    require(_sha256(published) == output["sha256"], "published/downloaded identity")
    capture = cv2.VideoCapture(str(video))
    frames = 0
    try:
        require(capture.isOpened(), "downloaded video decodes")
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while capture.read()[0]:
            frames += 1
    finally:
        capture.release()
    expected = min(limit, source_info.frame_count) if source_info.frame_count > 0 else None
    require(frames > 0 and frames == result.get("frames") == output.get("frame_count"), "decoded frame count")
    require(frames <= limit and (expected is None or frames == expected), "requested frame limit")
    require((width, height) == (output.get("width"), output.get("height")), "decoded dimensions")
    loaded = report.get("assets", {}).get("model_bundle", {})
    require(loaded.get("load_identity_confirmed") is True, "confirmed loaded model identity")
    require(loaded.get("content_sha256") == bundle["content_sha256"], "model content SHA256")
    require(loaded.get("model_name") == bundle["model_name"], "model architecture")
    weights = next(item for item in bundle["files"] if item["path"] == bundle["weights_path"])
    require(loaded.get("weights") == weights, "model weights identity")
    # Verify every submitted control, including values that happen to equal a UI
    # default. Presence matters: a missing nullable option is not a confirmed None.
    requested_options = {
        "model": "rvm_" + bundle["model_name"], "checkpoint": None,
        "mode": "side_by_side", "max_frames": limit, "background": None,
        "edge_feather": 0, "mask_open": 0, "mask_close": 0,
        "temporal_smooth": True, "temporal_blend": 0.75,
        "enable_depth": False, "depth_interval": 5, "preserve_audio": True,
        "resize_short": None, "background_color": "#00FF00",
        "temporal_mode": "adaptive", "temporal_motion_threshold": 0.15,
        "intermediate_codec": "FFV1",
    }
    for name, requested in requested_options.items():
        actual = options.get(name)
        matches = actual == requested
        if isinstance(requested, bool) or requested is None:
            matches = actual is requested
        elif isinstance(requested, (int, float)):
            matches = not isinstance(actual, bool) and matches
        require(name in options and matches, f"requested option {name}={requested!r}")
    recorded_bundle = options.get("model_bundle")
    require(isinstance(recorded_bundle, str) and
            Path(recorded_bundle).absolute().resolve() == Path(bundle["bundle_path"]).absolute().resolve(),
            "selected model bundle path")
    require(recorded_code.get("source_sha256") == code["source_sha256"], "package source SHA256")
    require(recorded_code.get("version") == code["version"], "package version")
    require(environment.get("python") == platform.python_version(), "Python interpreter version")
    require(environment.get("platform") == platform.platform(), "operating system")
    for name in ("torch", "torchvision", "numpy", "opencv-python", "gradio", "imageio-ffmpeg"):
        require(environment.get("packages", {}).get(name) == version(name), f"installed {name} version")
    actual_device = environment.get("device")
    require(environment.get("device_error") is None, "resolved device")
    require(actual_device == device or (device in ("auto", "cuda") and
            (actual_device == "cpu" and device == "auto" or str(actual_device).startswith("cuda:"))), "actual device")
    require(options.get("device") == actual_device or
            (options.get("device") == "cuda" and str(actual_device).startswith("cuda:")),
            "requested device reached pipeline")
    return {"frames": frames, "expected_frames": expected, "output_sha256": output["sha256"],
            "model_content_sha256": loaded["content_sha256"], "package_source_sha256": recorded_code["source_sha256"],
            "environment": environment, "report_code": recorded_code}


def run_rehearsal(args):
    from cv_human_video_matting.io.video import probe_video
    from cv_human_video_matting.reporting import _code_provenance
    from cv_human_video_matting.utils.model_bundle import verify_model_bundle
    import yaml

    source, model_path, output = (Path(value).absolute().resolve()
                                for value in (args.input, args.bundle, args.output_dir))
    if args.max_frames < 1 or any(not math.isfinite(value) or value <= 0
            for value in (args.startup_timeout, args.request_timeout)):
        raise ValueError("Frame limit and timeouts must be positive")
    bundle = verify_model_bundle(model_path)
    source_info, code = probe_video(source), _code_provenance()
    if output.is_relative_to(model_path):
        raise ValueError("Rehearsal output must be outside the model bundle")
    output.mkdir(parents=True, exist_ok=False)
    downloads, server_output = output / "downloads", output / "server-output"
    model = "rvm_" + bundle["model_name"]
    config = {"matting": {"model": model, "bundle": str(model_path), "checkpoint": None},
              "device": {"mode": args.device}, "depth": {"enabled": False},
              "paths": {"output_dir": str(server_output)},
              "postprocess": {"enable_temporal_smooth": False}}
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    evidence = {"schema_version": 1, "kind": "local_demo_rehearsal", "passed": False,
                "created_at": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
                "input": str(source), "bundle": str(model_path), "model": model, "max_frames": args.max_frames,
                "network_scope": "Python TCP/DNS guarded to loopback; no share link. Not an OS firewall test.",
                "scope": "Supplied sample only; does not certify human matting quality or webcam operation."}
    process = client = job = None
    error = None
    try:
        with TemporaryDirectory(prefix="runtime-", dir=output) as runtime, \
                (output / "server.log").open("w", encoding="utf-8") as server_log:
            env = {name: value for name, value in os.environ.items() if name.lower() not in
                   {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
            env.update({"NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1",
                        "GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                        "PYTHONDONTWRITEBYTECODE": "1", "TORCH_HOME": str(Path(runtime) / "torch"),
                        "GRADIO_TEMP_DIR": str(Path(runtime) / "gradio"), "MPLBACKEND": "Agg"})
            port = _free_port()
            url = f"http://127.0.0.1:{port}"
            # Execute the real CLI after installing guards before Gradio imports.
            bootstrap = ("import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                         "from verify_local_demo import local_network_only; sys.argv=sys.argv[1:]; "
                         "guard=local_network_only(); guard.__enter__(); "
                         "runpy.run_path(sys.argv[0],run_name='__main__')")
            command = [sys.executable, "-u", "-c", bootstrap, str(ROOT / "scripts"), str(ROOT / "app.py"),
                       "--config", str(config_path), "ui", "--model-bundle", str(model_path),
                       "--host", "127.0.0.1", "--port", str(port), "--no-share"]
            evidence["url"] = url
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=server_log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                evidence["service_pid"] = process.pid
                with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost,::1",
                        "no_proxy": "127.0.0.1,localhost,::1", "GRADIO_ANALYTICS_ENABLED": "False"}), local_network_only():
                    from gradio_client import Client, handle_file
                    _wait_ready(process, url, args.startup_timeout)
                    client = Client(url, verbose=False, download_files=str(downloads))
                    job = client.submit(handle_file(str(source)), None, model, "side_by_side",
                        args.max_frames, args.device, 0, 0, 0, True, 0.75, False, 5, True, 0, "#00FF00",
                        "adaptive", 0.15, api_name="/process_video")
                    result = job.result(timeout=args.request_timeout)
                    if not isinstance(result, (tuple, list)) or len(result) != 4:
                        raise ValueError("Unexpected process_video response; expected four outputs")
                    video = _download_path(result[0], ".mp4", downloads)
                    report_path = _download_path(result[3], ".json", downloads)
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    current_bundle = verify_model_bundle(model_path)
                    if current_bundle["content_sha256"] != bundle["content_sha256"]:
                        raise ValueError("Model bundle changed during the rehearsal")
                    checks = _check_report(report, video, source, source_info, bundle, code,
                                           args.device, args.max_frames, server_output)
                    evidence.update({"passed": True, "checks": checks, "downloaded_video": str(video),
                                     "downloaded_report": str(report_path), "status": str(result[2])})
                    (output / "client.log").write_text(str(result[2]) + "\n", encoding="utf-8")
            finally:
                # Ask the service to cancel a pending inference before terminating it.
                if client is not None and job is not None and not job.done():
                    try:
                        with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost,::1",
                                "no_proxy": "127.0.0.1,localhost,::1"}), local_network_only():
                            client.submit(api_name="/stop_processing").result(timeout=3)
                    except Exception:
                        pass
                if process is not None:
                    evidence["service_exit_code"] = _stop_process(process)
                    evidence["service_reaped"] = process.poll() is not None
                if client is not None:
                    client.close()
                    client.executor.shutdown(wait=False, cancel_futures=True)
    except BaseException as exc:
        evidence.update({"passed": False, "error_type": type(exc).__name__, "error": str(exc)})
        (output / "client.log").write_text(traceback.format_exc(), encoding="utf-8")
        error = exc
    finally:
        (output / "rehearsal.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    if error is not None:
        raise error
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; existing paths refused")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    run_rehearsal(args)
    print(f"Local demo rehearsal passed: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
