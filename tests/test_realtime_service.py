"""Realtime protocol and ownership tests use tiny JPEGs and fake offline runners."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
from threading import Event, get_ident
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cv_human_video_matting.matting.base import MattingResult
from cv_human_video_matting.realtime import service


def jpeg(width=64, height=48):
    stream = BytesIO()
    Image.new("RGB", (width, height), (80, 120, 160)).save(stream, format="JPEG")
    return stream.getvalue()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    bundle = {"model_name": "mobilenetv3", "bundle_path": str(tmp_path / "bundle"),
              "content_sha256": "b" * 64, "weights_path": "weights/rvm_mobilenetv3.pth",
              "files": [{"path": "weights/rvm_mobilenetv3.pth", "sha256": "w" * 64, "size_bytes": 7}]}
    runners, factory_calls = [], []

    class Runner:
        def __init__(self):
            self.bundle_info = deepcopy(bundle)
            self.frames = 0
            self.resets = 0
            self.thread_ids = []

        def predict(self, frame):
            self.thread_ids.append(get_ident())
            self.frames += 1
            return MattingResult(alpha=np.full(frame.shape[:2], 200, np.uint8), foreground=frame)

        def reset(self):
            self.resets += 1
            self.frames = 0

    def create(**kwargs):
        factory_calls.append(kwargs)
        runner = Runner()
        runners.append(runner)
        return runner

    monkeypatch.setattr(service, "verify_model_bundle", lambda path: deepcopy(bundle))
    monkeypatch.setattr(service, "resolve_device", lambda device: device)
    monkeypatch.setattr(service, "create_matting_runner", create)
    monkeypatch.setattr(service, "_code_provenance", lambda: {"version": "test", "source_sha256": "c" * 64})
    monkeypatch.setattr(service, "_environment", lambda device: {"device": device, "gpu": None})
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>local camera</title>")
    app = service.create_realtime_app(bundle["bundle_path"], frontend_dir=static)
    return SimpleNamespace(app=app, bundle=bundle, runners=runners, calls=factory_calls,
                           static=static, Runner=Runner)


def ready(websocket):
    hello = websocket.receive_json()
    status = websocket.receive_json()
    assert hello["type"] == "hello" and status["type"] == "ready"
    return hello, status


def frame(websocket, payload=None):
    websocket.send_bytes(payload or jpeg())
    metadata = websocket.receive_json()
    encoded = websocket.receive_bytes()
    assert metadata["type"] == "frame"
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert (decoded.shape[1], decoded.shape[0]) == (metadata["width"], metadata["height"])
    return metadata


def test_no_model_loaded_until_connection_and_page_has_no_store(runtime):
    assert not runtime.calls
    with TestClient(runtime.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/static/index.html").headers["cache-control"] == "no-store"
        assert client.get("/openapi.json").status_code == 404
    assert not runtime.calls


def test_ready_provenance_and_one_model_per_session(runtime):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
            hello, status = ready(ws)
            assert status["model"] == "rvm_mobilenetv3"
            assert status["device"] == "cpu"
            assert status["source_sha256"] == "c" * 64
            assert status["model_content_sha256"] == status["bundle_content_sha256"] == "b" * 64
            assert status["weights"] == runtime.bundle["files"][0]
            assert status["model_load_ms"] >= 0
            assert hello["max_jpeg_bytes"] == 2 * 1024 * 1024
            assert frame(ws)["seq"] == 1
            assert frame(ws)["seq"] == 2
            assert len(runtime.calls) == 1
            assert runtime.calls[0] == {"model": "rvm_mobilenetv3", "device": "cpu",
                "bundle_path": runtime.bundle["bundle_path"], "checkpoint": None, "downsample_ratio": "auto"}
    assert runtime.runners[0].resets >= 1
    assert not runtime.app.state.session_lock.locked()


@pytest.mark.parametrize("origin", ["null", "https://testserver", "http://foreign.example", "http://testserver:99",
                                    "http://user@testserver", "http://testserver/path"])
def test_foreign_or_invalid_origin_rejected_without_loading(runtime, origin):
    with TestClient(runtime.app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/ws", headers={"origin": origin}):
                pass
    assert exc.value.code == 1008
    assert not runtime.calls


def test_native_client_may_omit_origin(runtime):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            ready(ws)


def test_https_origin_matches_secure_websocket(runtime):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("wss://testserver/ws", headers={"origin": "https://testserver"}) as ws:
            ready(ws)


def test_second_session_rejected_and_next_connection_gets_new_runner(runtime):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as first:
            ready(first)
            with client.websocket_connect("/ws") as second:
                assert second.receive_json()["code"] == "busy"
                with pytest.raises(WebSocketDisconnect) as exc:
                    second.receive_json()
                assert exc.value.code == 1013
            assert len(runtime.runners) == 1
        # Disconnect only queues a message; model cleanup is intentionally async.
        async def idle():
            while runtime.app.state.session_lock.locked():
                await asyncio.sleep(0.005)
        client.portal.call(asyncio.wait_for, idle(), 3)
        with client.websocket_connect("/ws") as third:
            ready(third)
            assert frame(third)["seq"] == 1
    assert len(runtime.runners) == 2
    assert all(runner.resets >= 1 for runner in runtime.runners)


@pytest.mark.parametrize("mode,width,expected_width", [
    ("green", 320, 320), ("blur", 640, 640), ("alpha", 960, 800),
    ("foreground", 1280, 800), ("side_by_side", 320, 640),
])
def test_modes_resize_and_reset_ack(runtime, mode, width, expected_width):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            ready(ws)
            ws.send_json({"type": "configure", "mode": mode, "max_width": width})
            assert ws.receive_json() == {"type": "configured", "mode": mode, "max_width": width, "seq": 0}
            first = frame(ws, jpeg(800, 480))
            assert first["seq"] == 1 and first["width"] == expected_width
            assert first["cold"] is True and first["cold_reason"] == "model_first_frame"
            assert first["cuda_peak_allocated_bytes"] is None
            assert first["cuda_peak_reserved_bytes"] is None
            assert first["processing_ms"] > 0
            ws.send_json({"type": "reset"})
            assert ws.receive_json()["type"] == "reset"
            restarted = frame(ws)
            assert restarted["seq"] == 1 and restarted["cold_reason"] == "state_reset"


def test_warmup_is_marked_separately_from_cold_first_frame(runtime):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            ready(ws)
            stats = [frame(ws) for _ in range(11)]
    assert [item["seq"] for item in stats] == list(range(1, 12))
    assert [item["cold"] for item in stats] == [True] + [False] * 10
    assert [item["warmup"] for item in stats] == [True] * 10 + [False]


@pytest.mark.parametrize("control", [
    {"type": "configure", "mode": "image", "max_width": 640},
    {"type": "configure", "mode": "green", "max_width": 641},
    {"type": "configure", "mode": "green", "max_width": True},
    {"type": "configure", "mode": "green", "max_width": 640, "path": "private.mp4"},
    {"type": "reset", "checkpoint": "arbitrary.pth"}, {"type": "depth"}, [], None,
])
def test_control_parameters_cannot_add_paths_models_or_depth(runtime, control):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            ready(ws)
            ws.send_text(json.dumps(control))
            assert ws.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008
    assert not runtime.app.state.session_lock.locked()


@pytest.mark.parametrize("payload,close_code", [
    (b"not a jpeg", 1008), (b"", 1009), (b"x" * (service.MAX_JPEG_BYTES + 1), 1009),
    (b"\xff\xd8not-valid\xff\xd9", 1008), (jpeg()[:-2], 1008),
], ids=["not-jpeg", "empty", "over-limit", "corrupt", "truncated"])
def test_invalid_or_oversized_frame_closes_and_releases(runtime, payload, close_code):
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            ready(ws)
            ws.send_bytes(payload)
            assert ws.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_bytes()
            assert exc.value.code == close_code
    assert runtime.runners[0].frames == 0
    assert not runtime.app.state.session_lock.locked()


@pytest.mark.parametrize("size", [(4097, 2), (2049, 2049), (1, 100)])
def test_dimension_header_checked_before_image_decode(monkeypatch, size):
    class Header:
        format = "JPEG"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def load(self):
            pytest.fail("oversized image was decoded")
    header = Header()
    header.size = size
    monkeypatch.setattr(service.Image, "open", lambda stream: header)
    with pytest.raises(service.ProtocolError, match="dimensions"):
        service.decode_jpeg(b"\xff\xd8fixture\xff\xd9")


def test_changed_bundle_identity_fails_ready_and_releases_runner(runtime, monkeypatch):
    def changed(**kwargs):
        runner = runtime.Runner()
        runner.bundle_info["content_sha256"] = "changed"
        runtime.runners.append(runner)
        return runner
    monkeypatch.setattr(service, "create_matting_runner", changed)
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            error = ws.receive_json()
            assert error["code"] == "processing_failed" and "identity" in error["message"]
    assert runtime.runners[0].resets == 1
    assert not runtime.app.state.session_lock.locked()


def test_disconnect_keeps_seat_until_processing_thread_finishes(runtime, monkeypatch):
    started, finish = Event(), Event()
    original = runtime.Runner.predict

    def slow(self, frame):
        started.set()
        assert finish.wait(timeout=5)
        return original(self, frame)

    monkeypatch.setattr(runtime.Runner, "predict", slow)
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as first:
            ready(first)
            first.send_bytes(jpeg())
            assert started.wait(timeout=3)
            first.close()
            try:
                with client.websocket_connect("/ws") as other:
                    assert other.receive_json()["code"] == "busy"
                assert runtime.app.state.session_lock.locked()
                assert runtime.runners[0].resets == 0
            finally:
                finish.set()
    assert runtime.runners[0].resets == 1
    assert not runtime.app.state.session_lock.locked()


def test_worker_cancellation_waits_for_thread_before_returning():
    async def scenario():
        started, finish = Event(), Event()
        def slow():
            started.set()
            assert finish.wait(timeout=3)
        task = asyncio.create_task(service._worker(slow))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())


def test_cuda_metrics_and_cleanup_do_not_require_a_real_gpu(runtime, monkeypatch):
    calls = []
    fake_cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda device: calls.append(("peak-reset", device)),
        synchronize=lambda device: calls.append(("synchronize", device)),
        max_memory_allocated=lambda device: 1234,
        max_memory_reserved=lambda device: 5678,
        device=lambda device: nullcontext(),
        empty_cache=lambda: calls.append(("empty-cache",)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    session = service.RealtimeSession(runtime.bundle, "cuda:0", "green", 640)
    session.load()
    stats, _ = session.process(jpeg())
    assert stats["cuda_peak_allocated_bytes"] == 1234
    assert stats["cuda_peak_reserved_bytes"] == 5678
    session.close()
    assert session.runner is None
    assert calls == [("peak-reset", "cuda:0"), ("synchronize", "cuda:0"), ("empty-cache",)]


def test_failed_inference_closes_session_and_releases_runner(runtime, monkeypatch):
    def fail(self, image):
        raise RuntimeError("simulated inference failure")
    monkeypatch.setattr(runtime.Runner, "predict", fail)
    with TestClient(runtime.app) as client:
        with client.websocket_connect("/ws") as ws:
            ready(ws)
            ws.send_bytes(jpeg())
            assert ws.receive_json()["code"] == "processing_failed"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_bytes()
            assert exc.value.code == 1011
    assert runtime.runners[0].resets == 1
    assert not runtime.app.state.session_lock.locked()
