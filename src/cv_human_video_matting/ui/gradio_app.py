"""Gradio workbench for batch and live human video matting."""

from __future__ import annotations

import os

# Gradio wraps each request in a MatplotlibBackendManager that probes
# matplotlib's backend; with PyQt5 present, matplotlib may pick the Qt backend
# and crash on some version combos, breaking live-camera streaming. Force the
# headless Agg backend before gradio (and thus matplotlib) is imported.
os.environ.setdefault("MPLBACKEND", "Agg")
try:  # pragma: no cover - defensive; Agg is always available with matplotlib
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    pass

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np

from cv_human_video_matting.depth import DepthEstimator
from cv_human_video_matting.io.video import ensure_browser_playable_video
from cv_human_video_matting.matting.base import MattingRunner
from cv_human_video_matting.matting.factory import create_matting_runner
from cv_human_video_matting.pipeline import (
    load_background_image,
    process_video,
    render_matting_frame,
)
from cv_human_video_matting.postprocess import TemporalAlphaSmoother
from cv_human_video_matting.utils.config import config_value
from cv_human_video_matting.utils.device import resolve_device
from cv_human_video_matting.utils.paths import ensure_dir


OUTPUT_DIR = Path("data/output")
BATCH_MODE_CHOICES = [
    "green",
    "blur",
    "transparent",
    "image",
    "alpha",
    "foreground",
    "side_by_side",
]
LIVE_MODE_CHOICES = ["green", "blur", "image", "alpha", "foreground", "side_by_side"]
MODEL_CHOICES = [
    ("RVM MobileNetV3 (Fast)", "rvm_mobilenetv3"),
    ("RVM ResNet50 (Balanced)", "rvm_resnet50"),
    ("BiRefNet (High Quality)", "birefnet"),
]
MODEL_VALUES = [choice[1] for choice in MODEL_CHOICES]

# Active project config, set by build_demo so batch/live callbacks can resolve
# config-only settings (e.g. matting.checkpoint) the same way the CLI does.
_ACTIVE_CONFIG: dict = {}


@dataclass
class LiveRunnerCache:
    key: tuple[str, str, bool, float] | None = None
    runner: MattingRunner | None = None
    temporal_smoother: TemporalAlphaSmoother | None = None
    frames: int = 0
    started_at: float = 0.0
    depth_estimator: DepthEstimator | None = None
    depth_map: np.ndarray | None = None


_LIVE_CACHE = LiveRunnerCache()


@dataclass(frozen=True)
class BatchDemoResult:
    video_path: str | None
    artifact_path: str | None
    status: str


def _timestamped_output_path(mode: str) -> Path:
    ensure_dir(OUTPUT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"gradio_{timestamp}_{uuid4().hex[:12]}_{mode}.mp4"


def _zip_png_sequence(sequence_dir: str | Path) -> Path:
    source = Path(sequence_dir)
    if not source.is_dir():
        raise gr.Error(f"Transparent output directory was not created: {source}")

    frame_paths = sorted(source.glob("frame_*.png"))
    if not frame_paths:
        raise gr.Error(f"Transparent output directory has no PNG frames: {source}")

    archive_path = source.with_suffix(".zip")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for frame_path in frame_paths:
            archive.write(frame_path, arcname=frame_path.name)
    return archive_path


def _format_mode(mode: str) -> str:
    return mode.replace("_", " ").title()


def _choice_default(value: object, choices: list[str], fallback: str) -> str:
    return str(value) if value in choices else fallback


def _resize_rgb_for_live(frame_rgb: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame_rgb.shape[1] <= max_width:
        return frame_rgb
    scale = max_width / frame_rgb.shape[1]
    height = max(int(round(frame_rgb.shape[0] * scale)), 1)
    return cv2.resize(frame_rgb, (max_width, height), interpolation=cv2.INTER_AREA)


def _live_cache_for(
    model: str,
    device: str,
    temporal_smooth: bool,
    temporal_blend: float,
    cache: LiveRunnerCache | None = None,
) -> LiveRunnerCache:
    cache = cache if cache is not None else _LIVE_CACHE
    key = (model, device, bool(temporal_smooth), round(float(temporal_blend), 3))
    if cache.runner is None or cache.key != key:
        runner = create_matting_runner(model=model, device=device,
            downsample_ratio=config_value(_ACTIVE_CONFIG, "matting.downsample_ratio", "auto"))
        cache.key = key
        cache.runner = runner
        cache.temporal_smoother = (
            TemporalAlphaSmoother(blend=temporal_blend) if temporal_smooth else None
        )
        cache.depth_estimator = None
        cache.depth_map = None
        cache.frames = 0
        cache.started_at = perf_counter()
    return cache


def reset_live_state() -> str:
    _LIVE_CACHE.key = None
    _LIVE_CACHE.runner = None
    _LIVE_CACHE.temporal_smoother = None
    _LIVE_CACHE.frames = 0
    _LIVE_CACHE.started_at = 0.0
    _LIVE_CACHE.depth_estimator = None
    _LIVE_CACHE.depth_map = None
    return "Live state reset. The next frame will reload the selected model."


def prepare_input_preview(video_path: str | None) -> str | None:
    if not video_path:
        return None
    return str(ensure_browser_playable_video(video_path))


def _run_batch_demo(
    video_path: str | None,
    background_path: str | None,
    model: str,
    mode: str,
    max_frames: int,
    device: str,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smooth: bool = False,
    temporal_blend: float = 0.75,
    enable_depth: bool = False,
    depth_interval: int = 5,
) -> BatchDemoResult:
    if not video_path:
        raise gr.Error("Please upload an input video first.")
    if mode == "image" and not background_path:
        raise gr.Error("Image background mode requires a background image.")

    actual_device = resolve_device(device)
    output_path = _timestamped_output_path(mode)
    checkpoint = (
        config_value(_ACTIVE_CONFIG, "matting.checkpoint")
        if model == config_value(_ACTIVE_CONFIG, "matting.model") else None
    )
    runner = create_matting_runner(model=model, checkpoint=checkpoint, device=actual_device,
        downsample_ratio=config_value(_ACTIVE_CONFIG, "matting.downsample_ratio", "auto"))
    stats = process_video(
        input_path=video_path,
        output_path=output_path,
        runner=runner,
        mode=mode,
        max_frames=max_frames if max_frames > 0 else None,
        background_path=background_path,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smooth=temporal_smooth,
        temporal_blend=temporal_blend,
        enable_depth=enable_depth,
        depth_interval=depth_interval,
        device=actual_device,
    )
    if mode == "transparent":
        archive_path = _zip_png_sequence(stats.output_path)
        output_label = f"PNG frames: {stats.output_path}\nDownload: {archive_path}"
        video_path_out = None
        artifact_path = str(archive_path)
    else:
        playable_output = ensure_browser_playable_video(stats.output_path)
        output_label = f"Output: {playable_output}"
        video_path_out = str(playable_output)
        artifact_path = None

    depth_label = f"Enabled (interval={depth_interval})" if enable_depth else "Disabled"
    depth_status = f"\nDepth Labeling: {depth_label}"
    status = (
        f"{output_label}\n"
        f"Model: {model}\n"
        f"Device: {actual_device}\n"
        f"Mode: {_format_mode(mode)}\n"
        f"Frames: {stats.frames}\n"
        f"Elapsed: {stats.elapsed_seconds:.3f}s\n"
        f"FPS: {stats.fps:.2f}"
        + depth_status
    )
    return BatchDemoResult(video_path=video_path_out, artifact_path=artifact_path, status=status)


def run_demo(
    video_path: str | None,
    background_path: str | None,
    model: str,
    mode: str,
    max_frames: int,
    device: str,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smooth: bool = False,
    temporal_blend: float = 0.75,
    enable_depth: bool = False,
    depth_interval: int = 5,
) -> tuple[str | None, str]:
    result = _run_batch_demo(
        video_path=video_path,
        background_path=background_path,
        model=model,
        mode=mode,
        max_frames=max_frames,
        device=device,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smooth=temporal_smooth,
        temporal_blend=temporal_blend,
        enable_depth=enable_depth,
        depth_interval=depth_interval,
    )
    return result.video_path or result.artifact_path, result.status


def run_demo_for_ui(
    video_path: str | None,
    background_path: str | None,
    model: str,
    mode: str,
    max_frames: int,
    device: str,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smooth: bool = False,
    temporal_blend: float = 0.75,
    enable_depth: bool = False,
    depth_interval: int = 5,
):
    result = _run_batch_demo(
        video_path=video_path,
        background_path=background_path,
        model=model,
        mode=mode,
        max_frames=max_frames,
        device=device,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smooth=temporal_smooth,
        temporal_blend=temporal_blend,
        enable_depth=enable_depth,
        depth_interval=depth_interval,
    )
    return (
        gr.update(value=result.video_path, visible=result.video_path is not None),
        gr.update(value=result.artifact_path, visible=result.artifact_path is not None),
        result.status,
    )


def render_live_frame(
    frame_rgb: np.ndarray | None,
    background_path: str | None,
    model: str,
    mode: str,
    device: str,
    max_live_width: int,
    edge_feather: int = 0,
    mask_open: int = 0,
    mask_close: int = 0,
    temporal_smooth: bool = False,
    temporal_blend: float = 0.75,
    enable_depth: bool = False,
    depth_interval: int = 5,
    cache: LiveRunnerCache | None = None,
) -> tuple[np.ndarray | None, str]:
    if frame_rgb is None:
        return None, "Waiting for camera frame."
    if mode == "image" and not background_path:
        return None, "Image mode needs a background image."

    actual_device = resolve_device(device)
    frame_rgb = _resize_rgb_for_live(frame_rgb, int(max_live_width))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    background_bgr = load_background_image(background_path)
    cache = _live_cache_for(
        model=model,
        device=actual_device,
        temporal_smooth=temporal_smooth,
        temporal_blend=temporal_blend,
        cache=cache,
    )
    if cache.runner is None:
        return None, "Live model failed to initialize."

    started_at = perf_counter()
    if enable_depth and mode != "alpha":
        if cache.depth_estimator is None:
            cache.depth_estimator = DepthEstimator(device=actual_device)
        if (cache.depth_map is None or cache.depth_map.shape != frame_bgr.shape[:2]
                or cache.frames % max(1, int(depth_interval)) == 0):
            cache.depth_map = cache.depth_estimator.estimate(frame_bgr)
    else:
        cache.depth_estimator = None
        cache.depth_map = None
    rendered_bgr, _ = render_matting_frame(
        frame_bgr,
        runner=cache.runner,
        mode=mode,
        background_bgr=background_bgr,
        edge_feather=edge_feather,
        mask_open=mask_open,
        mask_close=mask_close,
        temporal_smoother=cache.temporal_smoother,
        depth_estimator=cache.depth_estimator,
        depth_map=cache.depth_map,
    )
    elapsed = perf_counter() - started_at
    cache.frames += 1
    total_elapsed = perf_counter() - cache.started_at if cache.started_at else elapsed
    session_fps = cache.frames / total_elapsed if total_elapsed > 0 else 0.0
    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
    status = (
        f"Mode: {_format_mode(mode)}\n"
        f"Frame: {cache.frames}\n"
        f"Last frame: {elapsed * 1000:.1f} ms\n"
        f"Session FPS: {session_fps:.2f}\n"
        f"Device: {actual_device}"
    )
    return rendered_rgb, status


def render_live_frame_for_ui(*args):
    # gr.State supplies an independent cache for each browser session.
    cache = args[-1]
    image, status = render_live_frame(*args[:-1], cache=cache)
    return image, status, cache


def reset_live_session():
    return "Live state reset.", LiveRunnerCache()


def _panel_header(title: str, meta: str) -> str:
    return f"""
    <div class="panel-heading">
        <h2>{title}</h2>
        <span>{meta}</span>
    </div>
    """


def _build_css() -> str:
    return """
    :root {
        --matte-bg: #eef2f6;
        --matte-surface: #ffffff;
        --matte-surface-soft: #f8fafc;
        --matte-ink: #101722;
        --matte-muted: #667085;
        --matte-border: #d8e0ea;
        --matte-accent: #0f76d1;
        --matte-accent-strong: #0b5cad;
        --matte-accent-soft: #e8f3ff;
        --matte-good: #087f5b;
    }

    body,
    .gradio-container {
        background:
            linear-gradient(180deg, #eaf0f7 0%, #f7f9fc 46%, #eef2f6 100%) !important;
        color: var(--matte-ink) !important;
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

    .app-shell {
        max-width: 1460px;
        margin: 0 auto;
        padding: 18px 22px 30px;
    }

    .topbar {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 20px;
        align-items: center;
        min-height: 86px;
        padding: 18px 20px;
        border: 1px solid #1c2733;
        border-radius: 8px;
        background:
            linear-gradient(135deg, #111827 0%, #172233 54%, #0d3d66 100%);
        box-shadow: 0 18px 45px rgb(16 24 40 / 0.18);
    }

    .brand-lockup {
        display: grid;
        grid-template-columns: 48px minmax(0, 1fr);
        gap: 14px;
        align-items: center;
    }

    .brand-mark {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        background:
            linear-gradient(135deg, #29b6f6 0%, #0f76d1 52%, #0a2f54 100%);
        box-shadow:
            inset 0 1px 0 rgb(255 255 255 / 0.32),
            0 14px 24px rgb(15 118 209 / 0.26);
        position: relative;
    }

    .brand-mark::after {
        content: "";
        position: absolute;
        inset: 12px;
        border: 2px solid rgb(255 255 255 / 0.82);
        border-radius: 999px;
        box-shadow: 0 0 0 6px rgb(255 255 255 / 0.12);
    }

    .brand-title {
        margin: 0;
        color: #ffffff !important;
        font-size: 26px;
        line-height: 1.05;
        font-weight: 780;
        letter-spacing: 0;
    }

    .brand-subtitle {
        margin-top: 6px;
        color: #c7d7e8;
        font-size: 13px;
        line-height: 1.45;
    }

    .run-meta {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 8px;
    }

    .meta-chip {
        min-height: 32px;
        display: inline-flex;
        align-items: center;
        border: 1px solid rgb(255 255 255 / 0.18);
        border-radius: 999px;
        padding: 0 12px;
        background: rgb(255 255 255 / 0.08);
        color: #edf7ff;
        font-size: 12px;
        font-weight: 720;
        white-space: nowrap;
    }

    .workspace-tabs {
        margin-top: 18px;
    }

    .gradio-container .tabs {
        background: transparent !important;
        border: 0 !important;
    }

    .gradio-container .tab-nav {
        gap: 8px !important;
        border: 0 !important;
        margin-bottom: 12px !important;
    }

    .gradio-container .tab-nav button {
        border-radius: 8px !important;
        border: 1px solid var(--matte-border) !important;
        background: #ffffff !important;
        color: var(--matte-muted) !important;
        min-height: 40px !important;
        font-weight: 760 !important;
    }

    .gradio-container .tab-nav button.selected {
        background: var(--matte-accent) !important;
        color: #ffffff !important;
        border-color: var(--matte-accent) !important;
        box-shadow: 0 12px 24px rgb(15 118 209 / 0.18);
    }

    .workspace-row {
        align-items: stretch;
        gap: 18px !important;
    }

    .media-column,
    .control-column {
        min-width: 0;
    }

    .panel,
    .control-panel {
        border-radius: 8px !important;
        border: 1px solid var(--matte-border) !important;
        background: var(--matte-surface) !important;
        box-shadow: 0 15px 34px rgb(16 24 40 / 0.08);
        overflow: hidden;
    }

    .control-panel {
        position: sticky;
        top: 14px;
    }

    .panel-heading {
        min-height: 48px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 14px;
        border-bottom: 1px solid var(--matte-border);
        background: linear-gradient(180deg, #ffffff, #f8fafc);
    }

    .panel-heading h2 {
        margin: 0;
        color: var(--matte-ink) !important;
        font-size: 13px;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .panel-heading span {
        color: var(--matte-muted);
        font-size: 12px;
        white-space: nowrap;
    }

    .media-grid,
    .control-stack,
    .live-grid {
        padding: 14px;
        gap: 14px !important;
    }

    .live-hint {
        margin: 0 14px 14px;
        padding: 10px 14px;
        border-radius: 10px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a5f;
        font-size: 13px;
        line-height: 1.6;
    }

    .result-grid {
        margin-top: 18px;
        gap: 18px !important;
    }

    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .wrap,
    .gradio-container .contain {
        border-radius: 8px !important;
    }

    .panel .block,
    .panel .form,
    .panel .wrap,
    .control-panel .block,
    .control-panel .form,
    .control-panel .wrap {
        background: var(--matte-surface-soft) !important;
        border-color: var(--matte-border) !important;
    }

    .gradio-container label > span,
    .gradio-container .label-wrap span {
        background: transparent !important;
        color: var(--matte-muted) !important;
        border-radius: 0 !important;
        padding: 0 !important;
        font-size: 12px !important;
        font-weight: 740 !important;
        letter-spacing: 0.01em;
    }

    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        color: var(--matte-ink) !important;
        background: #ffffff !important;
        border-color: #c9d4e2 !important;
    }

    .gradio-container button,
    .gradio-container .button {
        border-radius: 8px !important;
        font-weight: 760 !important;
        letter-spacing: 0 !important;
    }

    .primary-action button {
        min-height: 48px !important;
        background: var(--matte-accent) !important;
        border: 1px solid var(--matte-accent-strong) !important;
        color: #ffffff !important;
        box-shadow: 0 13px 24px rgb(15 118 209 / 0.22);
    }

    .primary-action button:hover {
        background: var(--matte-accent-strong) !important;
    }

    .secondary-action button {
        min-height: 42px !important;
        background: #ffffff !important;
        border: 1px solid #b9c7d8 !important;
        color: #233142 !important;
    }

    .status-box textarea {
        min-height: 210px !important;
        background: #0f172a !important;
        color: #d7e8f8 !important;
        border: 1px solid #1e293b !important;
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace !important;
        font-size: 13px !important;
        line-height: 1.55 !important;
    }

    .live-status textarea {
        min-height: 160px !important;
    }

    .gradio-container input[type="range"],
    .gradio-container input[type="checkbox"] {
        accent-color: var(--matte-accent);
    }

    footer {
        display: none !important;
    }

    @media (max-width: 980px) {
        .app-shell {
            padding: 12px;
        }

        .topbar {
            grid-template-columns: 1fr;
        }

        .run-meta {
            justify-content: flex-start;
        }

        .control-panel {
            position: static;
        }

        .panel-heading {
            align-items: flex-start;
            flex-direction: column;
            gap: 4px;
        }
    }
    """


def build_demo(config: dict | None = None) -> gr.Blocks:
    config = config or {}
    # Publish config so batch/live callbacks resolve config-only settings
    # (e.g. matting.checkpoint) the same way the CLI does.
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    batch_model_default = _choice_default(
        config_value(config, "matting.model", "rvm_resnet50"),
        MODEL_VALUES,
        "rvm_resnet50",
    )
    live_model_default = _choice_default(
        config_value(config, "matting.model", "rvm_mobilenetv3"),
        ["dummy", "rvm_mobilenetv3"],
        "rvm_mobilenetv3",
    )
    device_default = _choice_default(
        config_value(config, "device.mode", "auto"),
        ["auto", "cpu", "cuda"],
        "auto",
    )
    batch_mode_default = _choice_default(
        config_value(config, "render.mode", "green"),
        BATCH_MODE_CHOICES,
        "green",
    )
    live_mode_default = _choice_default(
        config_value(config, "render.mode", "green"),
        LIVE_MODE_CHOICES,
        "green",
    )
    edge_feather_default = int(config_value(config, "postprocess.edge_feather", 0))
    mask_open_default = int(config_value(config, "postprocess.mask_open", 0))
    mask_close_default = int(config_value(config, "postprocess.mask_close", 0))
    temporal_smooth_default = bool(
        config_value(config, "postprocess.enable_temporal_smooth", False)
    )
    temporal_blend_default = float(config_value(config, "postprocess.temporal_alpha", 0.75))
    enable_depth_default = bool(config_value(config, "depth.enabled", False))
    depth_interval_default = int(config_value(config, "depth.interval", 5))

    with gr.Blocks(
        title="Human Video Matting",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=_build_css(),
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <header class="topbar">
                    <div class="brand-lockup">
                        <div class="brand-mark"></div>
                        <div>
                            <h1 class="brand-title">Human Video Matting</h1>
                            <div class="brand-subtitle">RVM + BiRefNet + Distance Labeling | Batch & Live Camera</div>
                        </div>
                    </div>
                    <div class="run-meta">
                        <span class="meta-chip">CUDA ready</span>
                        <span class="meta-chip">3 Models</span>
                        <span class="meta-chip">7 Modes</span>
                        <span class="meta-chip">⭐ Depth</span>
                    </div>
                </header>
                """
            )

            with gr.Tabs(elem_classes=["workspace-tabs"]):
                with gr.Tab("Batch Render"):
                    with gr.Row(elem_classes=["workspace-row"]):
                        with gr.Column(scale=7, elem_classes=["media-column"]):
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML(_panel_header("Source", "input video and background"))
                                with gr.Row(elem_classes=["media-grid"]):
                                    video_input = gr.File(
                                        label="Input Video File",
                                        file_types=["video"],
                                        type="filepath",
                                    )
                                    input_preview = gr.Video(
                                        label="Input Preview",
                                        format="mp4",
                                        include_audio=False,
                                        height=280,
                                        interactive=False,
                                    )
                                    background_input = gr.Image(
                                        label="Background Image",
                                        sources=["upload"],
                                        type="filepath",
                                    )

                            with gr.Row(elem_classes=["result-grid"]):
                                with gr.Group(elem_classes=["panel"]):
                                    gr.HTML(_panel_header("Output", "rendered result"))
                                    output_video = gr.Video(
                                        label="Output Video",
                                        format="mp4",
                                        include_audio=False,
                                        height=320,
                                    )
                                    output_file = gr.File(
                                        label="Output Files",
                                        visible=False,
                                    )
                                with gr.Group(elem_classes=["panel"]):
                                    gr.HTML(_panel_header("Run Log", "frames and timing"))
                                    status_output = gr.Textbox(
                                        label="Status",
                                        lines=8,
                                        elem_classes=["status-box"],
                                        show_copy_button=True,
                                    )

                        with gr.Column(scale=4, elem_classes=["control-column"]):
                            with gr.Group(elem_classes=["control-panel"]):
                                gr.HTML(_panel_header("Batch Controls", "offline render"))
                                with gr.Column(elem_classes=["control-stack"]):
                                    model_input = gr.Dropdown(
                                        label="Model",
                                        choices=MODEL_VALUES,
                                        value=batch_model_default,
                                        info="BiRefNet offers best edge quality but slower",
                                    )
                                    device_input = gr.Dropdown(
                                        label="Device",
                                        choices=["auto", "cpu", "cuda"],
                                        value=device_default,
                                    )
                                    mode_input = gr.Dropdown(
                                        label="Mode",
                                        choices=BATCH_MODE_CHOICES,
                                        value=batch_mode_default,
                                        info="blur=background bokeh, transparent=BGRA PNG",
                                    )
                                    max_frames_input = gr.Slider(
                                        label="Max Frames",
                                        minimum=0,
                                        maximum=300,
                                        value=120,
                                        step=1,
                                        info="0 means whole video.",
                                    )
                                    with gr.Accordion("⭐ Distance Labeling (NEW)", open=True):
                                        enable_depth_input = gr.Checkbox(
                                            label="Enable Distance Labeling",
                                            value=enable_depth_default,
                                            info="Show Near/Mid/Far distance labels on person",
                                        )
                                        depth_interval_input = gr.Slider(
                                            label="Depth Interval",
                                            minimum=1,
                                            maximum=30,
                                            value=depth_interval_default,
                                            step=1,
                                            info="Estimate depth every N frames",
                                        )
                                    with gr.Accordion("Alpha Controls", open=True):
                                        edge_feather_input = gr.Slider(
                                            label="Edge Feather",
                                            minimum=0,
                                            maximum=9,
                                            value=edge_feather_default,
                                            step=1,
                                        )
                                        mask_open_input = gr.Slider(
                                            label="Mask Open",
                                            minimum=0,
                                            maximum=9,
                                            value=mask_open_default,
                                            step=1,
                                        )
                                        mask_close_input = gr.Slider(
                                            label="Mask Close",
                                            minimum=0,
                                            maximum=9,
                                            value=mask_close_default,
                                            step=1,
                                        )
                                    with gr.Accordion("Temporal Controls", open=True):
                                        temporal_smooth_input = gr.Checkbox(
                                            label="Temporal Smooth",
                                            value=temporal_smooth_default,
                                        )
                                        temporal_blend_input = gr.Slider(
                                            label="Temporal Blend",
                                            minimum=0.0,
                                            maximum=0.95,
                                            value=temporal_blend_default,
                                            step=0.05,
                                        )
                                    run_button = gr.Button(
                                        "Run Batch Matting",
                                        variant="primary",
                                        elem_classes=["primary-action"],
                                    )

                with gr.Tab("Live Camera"):
                    with gr.Row(elem_classes=["workspace-row"]):
                        with gr.Column(scale=7, elem_classes=["media-column"]):
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML(_panel_header("Camera", "browser webcam input"))
                                with gr.Row(elem_classes=["live-grid"]):
                                    live_camera = gr.Image(
                                        label="Camera Frame",
                                        sources=["webcam"],
                                        streaming=True,
                                        type="numpy",
                                        mirror_webcam=True,
                                        height=420,
                                    )
                                    live_output = gr.Image(
                                        label="Matting Preview",
                                        type="numpy",
                                        height=420,
                                        interactive=False,
                                    )
                                gr.HTML(
                                    "<div class=\"live-hint\">👉 点击摄像头下方的红色 ● 按钮开始实时抠图。"
                                    "首帧需加载模型（约 10 秒），之后实时更新。"
                                    "<br/>👉 Click the red ● button under the camera to start. "
                                    "The first frame loads the model (~10s), then streams live.</div>"
                                )
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML(_panel_header("Live Status", "current frame timing"))
                                live_status = gr.Textbox(
                                    label="Status",
                                    lines=6,
                                    elem_classes=["status-box", "live-status"],
                                    show_copy_button=True,
                                )

                        with gr.Column(scale=4, elem_classes=["control-column"]):
                            with gr.Group(elem_classes=["control-panel"]):
                                gr.HTML(_panel_header("Live Controls", "stream inference"))
                                with gr.Column(elem_classes=["control-stack"]):
                                    live_model = gr.Dropdown(
                                        label="Model",
                                        choices=["dummy", "rvm_mobilenetv3"],
                                        value=live_model_default,
                                    )
                                    live_device = gr.Dropdown(
                                        label="Device",
                                        choices=["auto", "cpu", "cuda"],
                                        value=device_default,
                                    )
                                    live_mode = gr.Dropdown(
                                        label="Mode",
                                        choices=LIVE_MODE_CHOICES,
                                        value=live_mode_default,
                                    )
                                    live_width = gr.Slider(
                                        label="Max Live Width",
                                        minimum=240,
                                        maximum=960,
                                        value=640,
                                        step=40,
                                    )
                                    live_background = gr.Image(
                                        label="Background Image",
                                        sources=["upload"],
                                        type="filepath",
                                    )
                                    with gr.Accordion("Alpha Controls", open=True):
                                        live_edge_feather = gr.Slider(
                                            label="Edge Feather",
                                            minimum=0,
                                            maximum=9,
                                            value=edge_feather_default,
                                            step=1,
                                        )
                                        live_mask_open = gr.Slider(
                                            label="Mask Open",
                                            minimum=0,
                                            maximum=9,
                                            value=mask_open_default,
                                            step=1,
                                        )
                                        live_mask_close = gr.Slider(
                                            label="Mask Close",
                                            minimum=0,
                                            maximum=9,
                                            value=mask_close_default,
                                            step=1,
                                        )
                                    with gr.Accordion("Temporal Controls", open=True):
                                        live_temporal_smooth = gr.Checkbox(
                                            label="Temporal Smooth",
                                            value=temporal_smooth_default,
                                        )
                                        live_temporal_blend = gr.Slider(
                                            label="Temporal Blend",
                                            minimum=0.0,
                                            maximum=0.95,
                                            value=temporal_blend_default,
                                            step=0.05,
                                        )
                                    live_depth = gr.Checkbox(label="Relative Distance (Near / Mid / Far)", value=False)
                                    live_depth_interval = gr.Slider(label="Depth Update Interval", minimum=1, maximum=30, value=5, step=1)
                                    live_session = gr.State(LiveRunnerCache())
                                    reset_live_button = gr.Button(
                                        "Reset Live State",
                                        elem_classes=["secondary-action"],
                                    )

            video_input.change(
                fn=prepare_input_preview,
                inputs=[video_input],
                outputs=[input_preview],
            )

            run_button.click(
                fn=run_demo_for_ui,
                inputs=[
                    video_input,
                    background_input,
                    model_input,
                    mode_input,
                    max_frames_input,
                    device_input,
                    edge_feather_input,
                    mask_open_input,
                    mask_close_input,
                    temporal_smooth_input,
                    temporal_blend_input,
                    enable_depth_input,
                    depth_interval_input,
                ],
                outputs=[output_video, output_file, status_output],
            )

            live_camera.stream(
                fn=render_live_frame_for_ui,
                inputs=[
                    live_camera,
                    live_background,
                    live_model,
                    live_mode,
                    live_device,
                    live_width,
                    live_edge_feather,
                    live_mask_open,
                    live_mask_close,
                    live_temporal_smooth,
                    live_temporal_blend,
                    live_depth,
                    live_depth_interval,
                    live_session,
                ],
                outputs=[live_output, live_status, live_session],
                trigger_mode="always_last",
                concurrency_limit=1,
            )

            reset_live_button.click(
                fn=reset_live_session,
                inputs=None,
                outputs=[live_status, live_session],
            )

    demo.queue(default_concurrency_limit=2)
    return demo


def launch_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    config: dict | None = None,
) -> None:
    demo = build_demo(config=config)
    demo.launch(server_name=server_name, server_port=server_port, share=share)
