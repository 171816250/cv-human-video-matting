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

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np

from cv_human_video_matting.depth import DepthEstimator
from cv_human_video_matting.io.video import ensure_browser_playable_video, export_video_for_delivery
from cv_human_video_matting.matting.base import MattingRunner
from cv_human_video_matting.matting.factory import create_matting_runner
from cv_human_video_matting.pipeline import (
    load_background_image,
    process_video,
    render_matting_frame,
    validate_video_request,
)
from cv_human_video_matting.postprocess import TemporalAlphaSmoother
from cv_human_video_matting.utils.config import config_value, parse_background_color, validate_project_config
from cv_human_video_matting.utils.device import resolve_device
from cv_human_video_matting.utils.paths import ensure_dir
from cv_human_video_matting.utils.progress import ProcessingCancelled
from cv_human_video_matting.ui.jobs import BatchJobs


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
    ("RVM MobileNetV3 · 轻量", "rvm_mobilenetv3"),
    ("RVM ResNet50 · 大模型", "rvm_resnet50"),
    ("BiRefNet · 实验后端", "birefnet"),
]
MODEL_VALUES = [choice[1] for choice in MODEL_CHOICES]
MODE_LABELS = {"green": "纯色背景", "blur": "背景虚化", "transparent": "透明PNG序列",
               "image": "图片背景", "alpha": "遮罩预览", "foreground": "黑底前景",
               "side_by_side": "原片与结果对比"}

# Active project config, set by build_demo so batch/live callbacks can resolve
# config-only settings (e.g. matting.checkpoint) the same way the CLI does.
_ACTIVE_CONFIG: dict = {}
_BATCH_JOBS = BatchJobs()


@dataclass
class LiveRunnerCache:
    key: tuple | None = None
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
    report_path: str | None = None


def _timestamped_output_path(mode: str) -> Path:
    output_dir = Path(config_value(_ACTIVE_CONFIG, "paths.output_dir", OUTPUT_DIR))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"gradio_{timestamp}_{uuid4().hex[:12]}_{mode}.mp4"


def _zip_png_sequence(sequence_dir: str | Path, should_cancel=None) -> Path:
    source = Path(sequence_dir)
    if not source.is_dir():
        raise gr.Error(f"Transparent output directory was not created: {source}")

    frame_paths = sorted(source.glob("frame_*.png"))
    if not frame_paths:
        raise gr.Error(f"Transparent output directory has no PNG frames: {source}")

    archive_path = source.with_suffix(".zip")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for frame_path in frame_paths:
            if should_cancel and should_cancel():
                raise ProcessingCancelled("已停止透明帧打包。")
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
    temporal_mode: str | None = None,
    temporal_motion_threshold: float | None = None,
) -> LiveRunnerCache:
    cache = cache if cache is not None else _LIVE_CACHE
    temporal_mode, temporal_motion_threshold = _temporal_settings(
        temporal_mode, temporal_motion_threshold)
    configured_model = config_value(_ACTIVE_CONFIG, "matting.model")
    checkpoint = (config_value(_ACTIVE_CONFIG, "matting.checkpoint")
                  if model == configured_model else None)
    bundle_path = (config_value(_ACTIVE_CONFIG, "matting.bundle")
                   if model == configured_model else None)
    ratio = config_value(_ACTIVE_CONFIG, "matting.downsample_ratio", "auto")
    key = (model, device, checkpoint, bundle_path, ratio, bool(temporal_smooth),
           float(temporal_blend), temporal_mode, temporal_motion_threshold,
           config_value(_ACTIVE_CONFIG, "depth.model", "MiDaS_small"),
           config_value(_ACTIVE_CONFIG, "depth.thresholds.near", 0.66),
           config_value(_ACTIVE_CONFIG, "depth.thresholds.mid", 0.33))
    if cache.runner is None or cache.key != key:
        # Validate temporal settings before loading a potentially large model.
        smoother = TemporalAlphaSmoother(blend=temporal_blend, mode=temporal_mode,
                                        motion_threshold=temporal_motion_threshold)
        runner = create_matting_runner(model=model, device=device, checkpoint=checkpoint,
            downsample_ratio=ratio, bundle_path=bundle_path)
        cache.key = key
        cache.runner = runner
        cache.temporal_smoother = smoother if temporal_smooth else None
        cache.depth_estimator = None
        cache.depth_map = None
        cache.frames = 0
        cache.started_at = perf_counter()
    return cache


def _temporal_settings(mode, motion_threshold):
    if mode is None:
        mode = config_value(_ACTIVE_CONFIG, "postprocess.temporal_mode", "ema")
    if motion_threshold is None:
        motion_threshold = config_value(_ACTIVE_CONFIG, "postprocess.temporal_motion_threshold", 0.15)
    return mode, motion_threshold


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
    try:
        return str(ensure_browser_playable_video(video_path, strict=True))
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(f"Input preview could not be prepared: {exc}") from exc


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
    preserve_audio: bool | None = None,
    resize_short: int | None = None,
    background_color: str | None = None,
    progress=None,
    should_cancel=None,
    temporal_mode: str | None = None,
    temporal_motion_threshold: float | None = None,
) -> BatchDemoResult:
    if not video_path:
        raise gr.Error("Please upload an input video first.")
    if mode == "image" and not background_path:
        raise gr.Error("Image background mode requires a background image.")
    max_frames = 0 if max_frames is None else max_frames
    if max_frames < 0 or int(max_frames) != max_frames:
        raise gr.Error("处理帧数须为非负整数，0表示完整视频。")
    max_frames = int(max_frames)
    if resize_short is None:
        resize_short = config_value(_ACTIVE_CONFIG, "input.resize_short")
    elif resize_short == 0:
        resize_short = None
    if resize_short is not None:
        if resize_short < 2 or int(resize_short) != resize_short:
            raise gr.Error("输出短边须为不小于2的整数，0表示原始尺寸。")
        resize_short = int(resize_short)
    if preserve_audio is None:
        preserve_audio = config_value(_ACTIVE_CONFIG, "render.preserve_audio", True)
    background_color = background_color or config_value(_ACTIVE_CONFIG, "render.background_color", "#00FF00")
    color_bgr = parse_background_color(background_color)
    temporal_mode, temporal_motion_threshold = _temporal_settings(
        temporal_mode, temporal_motion_threshold)

    def check_cancelled():
        if should_cancel and should_cancel():
            raise ProcessingCancelled("已停止处理。")

    def frame_progress(processed, total):
        if progress is not None:
            value = (processed, total) if total else None
            progress(value, desc=f"正在处理：{processed} / {total or '?'} 帧")

    check_cancelled()
    actual_device = resolve_device(device)
    output_path = _timestamped_output_path(mode)
    final_directory = output_path.with_suffix("")
    checkpoint = (
        config_value(_ACTIVE_CONFIG, "matting.checkpoint")
        if model == config_value(_ACTIVE_CONFIG, "matting.model") else None
    )
    bundle_path = (config_value(_ACTIVE_CONFIG, "matting.bundle")
                   if model == config_value(_ACTIVE_CONFIG, "matting.model") else None)
    from cv_human_video_matting.reporting import validate_artifact_paths
    validate_artifact_paths(final_directory / "result.json", video_path, final_directory,
                            asset_paths=(background_path, checkpoint, bundle_path))
    ensure_dir(output_path.parent)
    started_at = perf_counter()
    with TemporaryDirectory(prefix=".batch-", dir=output_path.parent) as staging:
        processing_args = dict(
            input_path=video_path,
            output_path=Path(staging) / "processed.mkv",
            mode=mode,
            max_frames=max_frames if max_frames > 0 else None,
            background_path=background_path,
            edge_feather=edge_feather, mask_open=mask_open, mask_close=mask_close,
            temporal_smooth=temporal_smooth, temporal_blend=temporal_blend,
            temporal_mode=temporal_mode, temporal_motion_threshold=temporal_motion_threshold,
            enable_depth=enable_depth, depth_interval=depth_interval,
            resize_short=resize_short,
            depth_model=config_value(_ACTIVE_CONFIG, "depth.model", "MiDaS_small"),
            depth_near=config_value(_ACTIVE_CONFIG, "depth.thresholds.near", 0.66),
            depth_mid=config_value(_ACTIVE_CONFIG, "depth.thresholds.mid", 0.33),
            background_color_bgr=color_bgr,
        )
        validate_video_request(**processing_args)
        check_cancelled()
        if progress is not None:
            progress(0, desc="正在准备模型，首次加载可能需要较长时间…")
        runner = create_matting_runner(model=model, checkpoint=checkpoint, device=actual_device,
            downsample_ratio=config_value(_ACTIVE_CONFIG, "matting.downsample_ratio", "auto"),
            bundle_path=bundle_path)
        check_cancelled()
        stats = process_video(**processing_args, runner=runner, device=actual_device,
                              progress_callback=frame_progress, should_cancel=should_cancel)
        export_started = perf_counter()
        check_cancelled()
        if progress is not None:
            progress(None, desc="正在打包透明帧…" if mode == "transparent" else "正在导出视频与音频…")
        if mode == "transparent":
            archive_path = _zip_png_sequence(stats.output_path, should_cancel=should_cancel)
            final_frames = final_directory / stats.output_path.name
            final_archive = final_directory / archive_path.name
            published_output = final_frames
            output_label = f"PNG frames: {final_frames}\nDownload: {final_archive}"
            video_path_out = None
            artifact_path = str(final_archive)
        else:
            playable_output = export_video_for_delivery(stats.output_path, Path(staging) / "result.mp4",
                source_path=video_path, preserve_audio=preserve_audio, should_cancel=should_cancel)
            stats.output_path.unlink()
            stats = replace(stats, output_path=playable_output)
            published_output = final_directory / playable_output.name
            output_label = f"Output: {published_output}"
            video_path_out = str(published_output)
            artifact_path = None

        export_seconds = perf_counter() - export_started
        total_seconds = perf_counter() - started_at
        if progress is not None:
            progress(None, desc="视频已处理完成，正在保存可追溯运行报告…")
        from cv_human_video_matting.reporting import write_run_report
        write_run_report(Path(staging) / "result.json", video_path, stats, {
            "model": model, "checkpoint": checkpoint, "device": actual_device, "mode": mode,
            "model_bundle": str(bundle_path) if bundle_path is not None else None,
            "max_frames": max_frames if max_frames > 0 else None, "background": background_path,
            "edge_feather": edge_feather, "mask_open": mask_open, "mask_close": mask_close,
            "temporal_smooth": temporal_smooth, "temporal_blend": temporal_blend,
            "temporal_mode": temporal_mode, "temporal_motion_threshold": temporal_motion_threshold,
            "intermediate_codec": "PNG" if mode == "transparent" else "FFV1",
            "enable_depth": enable_depth, "depth_interval": depth_interval,
            "downsample_ratio": config_value(_ACTIVE_CONFIG, "matting.downsample_ratio", "auto"),
            "resize_short": resize_short,
            "depth_model": config_value(_ACTIVE_CONFIG, "depth.model", "MiDaS_small"),
            "depth_near": config_value(_ACTIVE_CONFIG, "depth.thresholds.near", 0.66),
            "depth_mid": config_value(_ACTIVE_CONFIG, "depth.thresholds.mid", 0.33),
            "background_color": background_color,
            "preserve_audio": preserve_audio and mode != "transparent",
        }, total_seconds, export_seconds=export_seconds, published_output_path=published_output,
           model_bundle_info=getattr(runner, "bundle_info", None))
        check_cancelled()
        if final_directory.exists():
            raise FileExistsError(f"Output directory already exists: {final_directory}")
        # Publish media, PNG archive and report as one completed job directory.
        Path(staging).rename(final_directory)
        stats = replace(stats, output_path=published_output)
        report_path = final_directory / "result.json"

    depth_label = f"Enabled (interval={depth_interval})" if enable_depth else "Disabled"
    depth_status = f"\nDepth Labeling: {depth_label}"
    status = (
        f"{output_label}\n"
        f"Model: {model}\n"
        f"Device: {actual_device}\n"
        f"Mode: {_format_mode(mode)}\n"
        f"Frames: {stats.frames}\n"
        f"Pipeline elapsed: {stats.elapsed_seconds:.3f}s\nPipeline FPS: {stats.fps:.2f}\n"
        f"Total including model loading and export: {total_seconds:.3f}s\n"
        f"Run report: {report_path}"
        + depth_status
    )
    if progress is not None:
        progress(1, desc="处理完成，结果及报告可下载。")
    return BatchDemoResult(video_path=video_path_out, artifact_path=artifact_path,
                           status=status, report_path=str(report_path))


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
    temporal_mode: str | None = None,
    temporal_motion_threshold: float | None = None,
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
        temporal_mode=temporal_mode,
        temporal_motion_threshold=temporal_motion_threshold,
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
    preserve_audio: bool | None = None,
    resize_short: int | None = None,
    background_color: str | None = None,
    progress=gr.Progress(),
    request: gr.Request = gr.Request(),
    temporal_mode: str | None = None,
    temporal_motion_threshold: float | None = None,
):
    session_id = getattr(request, "session_hash", None) or uuid4().hex
    try:
        with _BATCH_JOBS.start(session_id) as token:
            result = _run_batch_demo(
                video_path=video_path, background_path=background_path,
                model=model, mode=mode, max_frames=max_frames, device=device,
                edge_feather=edge_feather, mask_open=mask_open, mask_close=mask_close,
                temporal_smooth=temporal_smooth, temporal_blend=temporal_blend,
                enable_depth=enable_depth, depth_interval=depth_interval,
                preserve_audio=preserve_audio, resize_short=resize_short,
                background_color=background_color, progress=progress,
                should_cancel=token.is_set,
                temporal_mode=temporal_mode, temporal_motion_threshold=temporal_motion_threshold,
            )
    except ProcessingCancelled:
        return gr.update(), gr.update(), "本次处理已停止，未发布新结果。", gr.update()
    except (ValueError, OSError, RuntimeError) as exc:
        raise gr.Error(f"处理未完成：{exc}") from exc
    return (
        gr.update(value=result.video_path, visible=result.video_path is not None),
        gr.update(value=result.artifact_path, visible=result.artifact_path is not None),
        result.status,
        gr.update(value=result.report_path, visible=result.report_path is not None),
    )


def cancel_batch_for_ui(request: gr.Request = gr.Request()):
    # A None default becomes Optional[Request] in Python 3.9/3.10 type hints;
    # Gradio 4.37 only injects the session for the exact Request annotation.
    session_id = getattr(request, "session_hash", None)
    if session_id and _BATCH_JOBS.cancel(session_id):
        return "已收到停止请求；正在完成当前帧或模型加载，并清理临时文件。"
    return "当前会话没有正在处理的任务。排队中的任务开始后可停止。"


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
    temporal_mode: str | None = None,
    temporal_motion_threshold: float | None = None,
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
        temporal_mode=temporal_mode,
        temporal_motion_threshold=temporal_motion_threshold,
    )
    if cache.runner is None:
        return None, "Live model failed to initialize."

    started_at = perf_counter()
    if enable_depth and mode != "alpha":
        if cache.depth_estimator is None:
            cache.depth_estimator = DepthEstimator(device=actual_device,
                model_type=config_value(_ACTIVE_CONFIG, "depth.model", "MiDaS_small"),
                near_threshold=config_value(_ACTIVE_CONFIG, "depth.thresholds.near", 0.66),
                mid_threshold=config_value(_ACTIVE_CONFIG, "depth.thresholds.mid", 0.33))
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
        background_color_bgr=parse_background_color(config_value(_ACTIVE_CONFIG, "render.background_color", "#00FF00")),
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
    controls = args[:-1]
    temporal_settings = {}
    if len(controls) > 13:
        temporal_settings = dict(temporal_mode=controls[13], temporal_motion_threshold=controls[14])
        controls = controls[:13]
    image, status = render_live_frame(*controls, cache=cache, **temporal_settings)
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
    validate_project_config(config)
    # Publish config so batch/live callbacks resolve config-only settings
    # (e.g. matting.checkpoint) the same way the CLI does.
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    batch_model_default = _choice_default(
        config_value(config, "matting.model", "rvm_mobilenetv3"),
        MODEL_VALUES,
        "rvm_mobilenetv3",
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
    temporal_mode_default = config_value(config, "postprocess.temporal_mode", "ema")
    temporal_threshold_default = config_value(config, "postprocess.temporal_motion_threshold", 0.15)
    enable_depth_default = bool(config_value(config, "depth.enabled", False))
    depth_interval_default = int(config_value(config, "depth.interval", 5))

    with gr.Blocks(
        title="人像视频抠像工作台",
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
                            <h1 class="brand-title">人像视频抠像工作台</h1>
                            <div class="brand-subtitle">视频处理 · 背景替换 · 摄像头预览</div>
                        </div>
                    </div>
                    <div class="run-meta">
                        <span class="meta-chip">本机处理</span>
                        <span class="meta-chip">3种模型</span>
                        <span class="meta-chip">7种输出模式</span>
                        <span class="meta-chip">相对远近标注</span>
                    </div>
                </header>
                """
            )

            with gr.Tabs(elem_classes=["workspace-tabs"]):
                with gr.Tab("视频处理"):
                    gr.Markdown("上传视频 → 选择效果 → 开始处理。默认处理完整视频，结果与运行报告均可下载；透明PNG不含声音。")
                    with gr.Row(elem_classes=["workspace-row"]):
                        with gr.Column(scale=7, elem_classes=["media-column"]):
                            with gr.Group(elem_classes=["panel"]):
                                gr.HTML(_panel_header("输入素材", "视频和背景图片"))
                                with gr.Row(elem_classes=["media-grid"]):
                                    video_input = gr.File(
                                        label="上传视频",
                                        file_types=["video"],
                                        type="filepath",
                                    )
                                    input_preview = gr.Video(
                                        label="原片预览",
                                        format="mp4",
                                        include_audio=True,
                                        height=280,
                                        interactive=False,
                                    )
                                    background_input = gr.Image(
                                        label="背景图片（图片背景模式使用）",
                                        sources=["upload"],
                                        type="filepath",
                                    )

                            with gr.Row(elem_classes=["result-grid"]):
                                with gr.Group(elem_classes=["panel"]):
                                    gr.HTML(_panel_header("处理结果", "预览与下载"))
                                    output_video = gr.Video(
                                        label="输出视频",
                                        format="mp4",
                                        include_audio=True,
                                        height=320,
                                    )
                                    output_file = gr.File(
                                        label="下载透明帧压缩包",
                                        visible=False,
                                    )
                                    report_file = gr.File(label="下载运行报告", visible=False)
                                with gr.Group(elem_classes=["panel"]):
                                    gr.HTML(_panel_header("处理记录", "帧数与耗时"))
                                    status_output = gr.Textbox(
                                        label="运行状态",
                                        lines=8,
                                        elem_classes=["status-box"],
                                        show_copy_button=True,
                                    )

                        with gr.Column(scale=4, elem_classes=["control-column"]):
                            with gr.Group(elem_classes=["control-panel"]):
                                gr.HTML(_panel_header("处理设置", "选择模型与效果"))
                                with gr.Column(elem_classes=["control-stack"]):
                                    model_input = gr.Dropdown(
                                        label="抠像模型",
                                        choices=MODEL_CHOICES,
                                        value=batch_model_default,
                                        info="建议先用轻量RVM。其他模型需提前准备权重并验证效果。",
                                    )
                                    device_input = gr.Dropdown(
                                        label="处理设备",
                                        choices=["auto", "cpu", "cuda"],
                                        value=device_default,
                                    )
                                    mode_input = gr.Dropdown(
                                        label="输出效果",
                                        choices=[(MODE_LABELS[value], value) for value in BATCH_MODE_CHOICES],
                                        value=batch_mode_default,
                                        info="对比模式便于观察原片与抠像结果。",
                                    )
                                    max_frames_input = gr.Number(
                                        label="处理帧数上限",
                                        minimum=0,
                                        value=0,
                                        precision=0,
                                        info="0表示完整视频；调试时可填120等较小值。",
                                    )
                                    with gr.Accordion("导出设置", open=True):
                                        preserve_audio_input = gr.Checkbox(label="保留原视频声音",
                                            value=config_value(config, "render.preserve_audio", True))
                                        resize_short_input = gr.Number(label="输出短边上限（像素）",
                                            minimum=0, precision=0,
                                            value=config_value(config, "input.resize_short") or 0,
                                            info="0保留原尺寸；只缩小，不放大。")
                                        background_color_input = gr.ColorPicker(label="纯色背景颜色",
                                            value=config_value(config, "render.background_color", "#00FF00"))
                                    with gr.Accordion("相对远近标注", open=False):
                                        enable_depth_input = gr.Checkbox(
                                            label="启用相对远近标注",
                                            value=enable_depth_default,
                                            info="Near/Mid/Far是相对远近，不表示实际米数；开启会增加处理时间。",
                                        )
                                        depth_interval_input = gr.Slider(
                                            label="深度更新间隔（帧）",
                                            minimum=1,
                                            maximum=30,
                                            value=depth_interval_default,
                                            step=1,
                                            info="每隔这些帧重新估计一次深度。",
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
                                    with gr.Accordion("减少画面闪烁", open=True):
                                        temporal_smooth_input = gr.Checkbox(
                                            label="启用时间平滑",
                                            value=temporal_smooth_default,
                                            info="平滑可减轻遮罩闪烁；快速运动时也可能产生拖影，默认关闭。",
                                        )
                                        temporal_mode_input = gr.Dropdown(
                                            label="平滑方式",
                                            choices=[("固定平滑", "ema"), ("运动自适应（实验）", "adaptive")],
                                            value=temporal_mode_default,
                                            info="自适应方式在遮罩明显变化的位置减弱历史残留，仍需结合原片检查。",
                                        )
                                        temporal_blend_input = gr.Slider(
                                            label="历史帧权重",
                                            minimum=0.0,
                                            maximum=0.95,
                                            value=temporal_blend_default,
                                            step=0.05,
                                        )
                                        temporal_threshold_input = gr.Number(
                                            label="自适应变化阈值",
                                            value=temporal_threshold_default, minimum=0, maximum=1,
                                            info="取值大于0且不超过1，仅自适应方式使用；越小越容易跟随变化，也可能增加闪烁。",
                                        )
                                    run_button = gr.Button(
                                        "开始处理",
                                        variant="primary",
                                        elem_classes=["primary-action"],
                                    )
                                    cancel_button = gr.Button("停止当前处理", variant="secondary")

                with gr.Tab("摄像头预览"):
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
                                    "首帧需加载模型，等待时间取决于环境。"
                                    "请先允许浏览器访问摄像头；视频批处理期间，预览请求会排队。</div>"
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
                                    with gr.Accordion("减少画面闪烁", open=True):
                                        live_temporal_smooth = gr.Checkbox(
                                            label="启用时间平滑",
                                            value=temporal_smooth_default,
                                            info="可减轻闪烁，但快速运动可能有拖影；默认关闭。",
                                        )
                                        live_temporal_mode = gr.Dropdown(
                                            label="平滑方式",
                                            choices=[("固定平滑", "ema"), ("运动自适应（实验）", "adaptive")],
                                            value=temporal_mode_default,
                                            info="自适应方式在遮罩变化处减少历史残留。",
                                        )
                                        live_temporal_blend = gr.Slider(
                                            label="历史帧权重",
                                            minimum=0.0,
                                            maximum=0.95,
                                            value=temporal_blend_default,
                                            step=0.05,
                                        )
                                        live_temporal_threshold = gr.Number(
                                            label="自适应变化阈值",
                                            value=temporal_threshold_default, minimum=0, maximum=1,
                                            info="大于0且不超过1；越小越容易跟随变化，也可能增加闪烁。",
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
                    preserve_audio_input,
                    resize_short_input,
                    background_color_input,
                    temporal_mode_input,
                    temporal_threshold_input,
                ],
                outputs=[output_video, output_file, status_output, report_file],
                api_name="process_video",
                concurrency_id="inference",
                concurrency_limit=1,
            )

            cancel_button.click(fn=cancel_batch_for_ui, inputs=None, outputs=[status_output],
                                queue=False, api_name="stop_processing")

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
                    live_temporal_mode,
                    live_temporal_threshold,
                    live_session,
                ],
                outputs=[live_output, live_status, live_session],
                trigger_mode="always_last",
                concurrency_id="inference",
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
