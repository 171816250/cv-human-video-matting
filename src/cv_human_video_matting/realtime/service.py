"""独立于 Gradio 状态的单会话 JPEG 实时预览服务。

协议 1：hello → ready；客户端发送 JPEG，服务端依次返回 frame JSON 和 JPEG。
客户端收完上一帧再发送下一帧；传输队列上限由 CLI 设置。
"""

from __future__ import annotations

import asyncio
import gc
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cv_human_video_matting.matting.factory import create_matting_runner
from cv_human_video_matting.pipeline import render_matting_frame
from cv_human_video_matting.reporting import _code_provenance, _environment
from cv_human_video_matting.utils.device import resolve_device
from cv_human_video_matting.utils.model_bundle import verify_model_bundle


MODES = frozenset({"green", "blur", "alpha", "foreground", "side_by_side"})
WIDTHS = frozenset({320, 640, 960, 1280})
MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_INPUT_PIXELS = 4 * 1024 * 1024
MAX_INPUT_SIDE = 4096
MAX_CONTROL_BYTES = 4096
WARMUP_FRAMES = 10
JPEG_QUALITY = 85


class ProtocolError(ValueError):
    def __init__(self, message: str, code: str = "invalid_request", close_code: int = 1008):
        super().__init__(message)
        self.code = code
        self.close_code = close_code


def validate_settings(mode, max_width) -> None:
    if not isinstance(mode, str) or mode not in MODES:
        raise ProtocolError("Choose green, blur, alpha, foreground, or side_by_side.")
    if type(max_width) is not int or max_width not in WIDTHS:
        raise ProtocolError("max_width must be 320, 640, 960, or 1280.")


def _origin_allowed(websocket: WebSocket) -> bool:
    """Browsers must be same-origin; native protocol clients may omit Origin."""
    value = websocket.headers.get("origin")
    if value is None:
        return True
    try:
        origin = urlsplit(value)
        host = urlsplit("http://" + websocket.headers.get("host", ""))
        if (origin.scheme not in {"http", "https"} or not origin.hostname
                or origin.username is not None or origin.password is not None
                or origin.path not in {"", "/"} or origin.query or origin.fragment):
            return False
        expected_scheme = "https" if websocket.url.scheme == "wss" else "http"
        return (origin.scheme == expected_scheme and origin.hostname == host.hostname
                and (origin.port or (443 if origin.scheme == "https" else 80))
                == (host.port or (443 if expected_scheme == "https" else 80)))
    except (TypeError, ValueError):
        return False


def decode_jpeg(payload: bytes) -> np.ndarray:
    """先检查 JPEG 头部尺寸，防止解码过大的图片。"""
    if not payload or len(payload) > MAX_JPEG_BYTES:
        raise ProtocolError("JPEG frame exceeds the 2 MiB limit or is empty.", "frame_size", 1009)
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ProtocolError("Each binary message must contain one complete JPEG.", "invalid_jpeg")
    try:
        with Image.open(BytesIO(payload)) as picture:
            if picture.format != "JPEG":
                raise ProtocolError("Only JPEG frames are accepted.", "invalid_jpeg")
            width, height = picture.size
            if (width < 2 or height < 2 or max(width, height) > MAX_INPUT_SIDE
                    or width * height > MAX_INPUT_PIXELS):
                raise ProtocolError("JPEG dimensions exceed 4096 per side / 4 megapixels, or are below 2.",
                                    "frame_dimensions", 1009)
            picture.load()
            return cv2.cvtColor(np.asarray(picture.convert("RGB")), cv2.COLOR_RGB2BGR)
    except ProtocolError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ProtocolError("JPEG could not be decoded completely.", "invalid_jpeg") from exc


class RealtimeSession:
    """One model and recurrent state, accessed serially from worker threads."""

    def __init__(self, bundle: dict, device: str, mode: str, max_width: int):
        validate_settings(mode, max_width)
        self.bundle, self.device = bundle, device
        self.model = "rvm_" + bundle["model_name"]
        self.mode, self.max_width = mode, max_width
        self.runner = None
        self.seq = 0
        self.ever_processed = False
        self.model_load_ms = None

    def load(self) -> dict:
        import torch

        started = perf_counter()
        if self.device.startswith("cuda:"):
            torch.cuda.reset_peak_memory_stats(self.device)
        self.runner = create_matting_runner(model=self.model, device=self.device,
            bundle_path=self.bundle["bundle_path"], checkpoint=None, downsample_ratio="auto")
        loaded = getattr(self.runner, "bundle_info", None)
        if (not isinstance(loaded, dict) or loaded.get("content_sha256") != self.bundle["content_sha256"]
                or loaded.get("model_name") != self.bundle["model_name"]):
            raise RuntimeError("Loaded model identity does not match the verified offline bundle.")
        if self.device.startswith("cuda:"):
            torch.cuda.synchronize(self.device)
        self.model_load_ms = (perf_counter() - started) * 1000
        provenance = _code_provenance()
        weight = next(item for item in self.bundle["files"] if item["path"] == self.bundle["weights_path"])
        return {"type": "ready", "protocol": 1, "model": self.model, "device": self.device,
                "mode": self.mode, "max_width": self.max_width,
                "bundle_content_sha256": self.bundle["content_sha256"],
                "model_content_sha256": self.bundle["content_sha256"], "weights": weight,
                "package_version": provenance["version"], "source_sha256": provenance["source_sha256"],
                "environment": _environment(self.device), "model_load_ms": self.model_load_ms,
                "warmup_frames": WARMUP_FRAMES, "output_jpeg_quality": JPEG_QUALITY,
                "processing_scope": "JPEG decode, server resize, inference, composite and JPEG encode; "
                                    "excludes model loading, network transit and browser display.",
                "cuda_memory_scope": "PyTorch allocator peak since session model load, including warmup; "
                                     "not total device memory usage."}

    def configure(self, mode: str, max_width: int) -> dict:
        validate_settings(mode, max_width)
        self.reset()
        self.mode, self.max_width = mode, max_width
        return {"type": "configured", "mode": mode, "max_width": max_width, "seq": 0}

    def reset(self) -> dict:
        if self.runner is not None:
            self.runner.reset()
        self.seq = 0
        return {"type": "reset", "seq": 0, "warmup_frames": WARMUP_FRAMES}

    def process(self, payload: bytes) -> tuple[dict, bytes]:
        if self.runner is None:
            raise RuntimeError("Session model is not ready.")
        started = perf_counter()
        frame = decode_jpeg(payload)
        if frame.shape[1] > self.max_width:
            height = max(2, int(frame.shape[0] * self.max_width / frame.shape[1] + 0.5))
            frame = cv2.resize(frame, (self.max_width, height), interpolation=cv2.INTER_AREA)
        rendered, _ = render_matting_frame(frame, self.runner, self.mode)
        success, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not success:
            raise RuntimeError("Could not encode the processed frame as JPEG.")
        result = encoded.tobytes()
        elapsed_ms = (perf_counter() - started) * 1000
        self.seq += 1
        metrics = {"type": "frame", "seq": self.seq, "width": int(rendered.shape[1]),
                   "height": int(rendered.shape[0]), "processing_ms": elapsed_ms,
                   "cold": self.seq == 1, "warmup": self.seq <= WARMUP_FRAMES,
                   "cold_reason": ("model_first_frame" if not self.ever_processed else "state_reset")
                                  if self.seq == 1 else None,
                   "device": self.device, "mode": self.mode, "max_width": self.max_width,
                   "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
        if self.device.startswith("cuda:"):
            import torch
            metrics.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                           cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)))
        self.ever_processed = True
        return metrics, result

    def close(self) -> None:
        runner, self.runner = self.runner, None
        try:
            if runner is not None:
                runner.reset()
        finally:
            del runner
            gc.collect()
            if self.device.startswith("cuda:"):
                import torch
                # Explicit context affects only this worker and restores the
                # previous device; no process-wide default is changed.
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()


async def _worker(function, *args):
    """取消请求后仍等待工作线程结束，避免提前释放会话名额。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            # Retrieve a failed worker's exception to avoid an unobserved task.
            task.exception()
        raise


async def _error(websocket: WebSocket, code: str, message: str, close_code: int):
    try:
        await websocket.send_json({"type": "error", "code": code, "message": message})
        await websocket.close(code=close_code)
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass


def create_realtime_app(bundle_path, device="cpu", max_width=640, mode="green", frontend_dir=None) -> FastAPI:
    """创建实时服务；取得会话名额后才加载模型。

    建议绑定本机并使用单个 ASGI worker；ws_max_size=2*1024*1024，ws_max_queue=1。
    默认关闭代理头处理，受信反代由启动参数显式开启。
    """
    validate_settings(mode, max_width)
    bundle = verify_model_bundle(bundle_path)
    actual_device = resolve_device(device)
    if actual_device == "cuda":
        import torch
        actual_device = f"cuda:{torch.cuda.current_device()}"
    app = FastAPI(title="Local realtime matting", docs_url=None, redoc_url=None, openapi_url=None)
    seat = Lock()
    app.state.session_lock = seat

    @app.middleware("http")
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    static = (Path(frontend_dir) if frontend_dir is not None else Path(__file__).parent / "static").resolve()
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def preview(websocket: WebSocket):
        if not _origin_allowed(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if not seat.acquire(blocking=False):
            await _error(websocket, "busy", "Another realtime session is active. Stop it before connecting.", 1013)
            return
        session = RealtimeSession(bundle, actual_device, mode, max_width)
        try:
            await websocket.send_json({"type": "hello", "protocol": 1, "model": session.model,
                "device": actual_device, "mode": mode, "max_width": max_width,
                "bundle_content_sha256": bundle["content_sha256"],
                "max_jpeg_bytes": MAX_JPEG_BYTES, "max_input_pixels": MAX_INPUT_PIXELS,
                "max_input_side": MAX_INPUT_SIDE, "modes": sorted(MODES), "widths": sorted(WIDTHS)})
            await websocket.send_json(await _worker(session.load))
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is not None:
                    metadata, jpeg = await _worker(session.process, payload)
                    await websocket.send_json(metadata)
                    await websocket.send_bytes(jpeg)
                    continue
                text = message.get("text", "")
                if len(text.encode("utf-8")) > MAX_CONTROL_BYTES:
                    raise ProtocolError("Control message is too large.", "control_size", 1009)
                try:
                    control = json.loads(text)
                except (ValueError, RecursionError) as exc:
                    raise ProtocolError("Control message must be JSON.") from exc
                if not isinstance(control, dict):
                    raise ProtocolError("Control message must be an object.")
                kind = control.get("type")
                if kind == "configure" and set(control) == {"type", "mode", "max_width"}:
                    result = await _worker(session.configure, control["mode"], control["max_width"])
                elif kind == "reset" and set(control) == {"type"}:
                    result = await _worker(session.reset)
                else:
                    raise ProtocolError("Only configure(mode,max_width) and reset controls are accepted.")
                await websocket.send_json(result)
        except ProtocolError as exc:
            await _error(websocket, exc.code, str(exc), exc.close_code)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await _error(websocket, "processing_failed", str(exc), 1011)
        finally:
            try:
                await _worker(session.close)
            finally:
                seat.release()

    return app
