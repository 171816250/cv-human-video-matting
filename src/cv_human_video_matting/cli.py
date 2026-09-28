"""Command line interface for iterative project milestones."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory
import sys


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def config_with_model_bundle(config: dict, bundle_path, *, select_model: bool = True) -> dict:
    """Verify a local bundle before serving and keep the caller's config unchanged."""
    from copy import deepcopy
    from cv_human_video_matting.utils.model_bundle import verify_model_bundle

    selected = deepcopy(config)
    matting = selected.setdefault("matting", {})
    expected = None
    if not select_model:
        model = matting.get("model", "rvm_mobilenetv3")
        aliases = {"rvm": "mobilenetv3", "rvm_mobilenetv3": "mobilenetv3",
                   "mobilenetv3": "mobilenetv3", "rvm_resnet50": "resnet50", "resnet50": "resnet50"}
        if model not in aliases:
            raise ValueError("Configured offline bundle requires a matching RVM model.")
        if matting.get("checkpoint") is not None:
            raise ValueError("matting.bundle and matting.checkpoint cannot be configured together.")
        expected = aliases[model]
    identity = verify_model_bundle(bundle_path, expected_model=expected)
    matting.update(model="rvm_" + identity["model_name"],
                   bundle=identity["bundle_path"], checkpoint=None)
    return selected


def create_matting_runner(**kwargs):
    """Load inference dependencies only when an inference command needs them."""
    from cv_human_video_matting.matting.factory import create_matting_runner as create
    return create(**kwargs)


class ConsoleProgress:
    """Bound terminal output to one update per second plus the first/final frame."""
    def __init__(self):
        self._last_update = 0.0

    def __call__(self, processed: int, total: int | None):
        now = perf_counter()
        if processed == 0 or processed == total or now - self._last_update >= 1:
            suffix = str(total) if total is not None else "?"
            print(f"Processing frames: {processed}/{suffix}", file=sys.stderr, flush=True)
            self._last_update = now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv-human-video-matting",
        description="Natural-scene human video matting project.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the project YAML config.",
    )

    subparsers = parser.add_subparsers(dest="command")
    summary_parser = subparsers.add_parser("summarize", help="Summarize comparable run reports.")
    summary_parser.add_argument("--reports", nargs="+", required=True, help="Run JSON paths or glob patterns.")
    summary_parser.add_argument("--output", required=True, help="Summary .html or .json path.")
    doctor_parser = subparsers.add_parser("doctor", help="Check local readiness without downloads.")
    doctor_parser.add_argument("--report", help="Save environment checks as JSON.")
    doctor_parser.add_argument("--requirements", help="Requirements profile to check (relative to current directory).")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect an input video.")
    inspect_parser.add_argument("--input", required=True, help="Input video path.")

    copy_parser = subparsers.add_parser("copy", help="Copy video frames to a new output file.")
    copy_parser.add_argument("--input", required=True, help="Input video path.")
    copy_parser.add_argument("--output", required=True, help="Output video path.")
    copy_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )

    matte_parser = subparsers.add_parser("matte", help="Run video matting on an input video.")
    matte_parser.add_argument("--input", required=True, help="Input video path.")
    matte_parser.add_argument("--report", help="Optional JSON performance report path.")
    matte_parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                              help="Show periodic frame progress on stderr (default: on).")
    matte_parser.add_argument("--resize-short", type=int, help="Limit the shorter side; no upscaling.")
    matte_parser.add_argument("--background-color", help="Solid background color, e.g. '#336699'.")
    matte_parser.add_argument("--preserve-audio", action=argparse.BooleanOptionalAction,
                              default=None, help="Keep original audio in MP4 output (default: on).")
    matte_parser.add_argument("--downsample-ratio", default=None,
                              help="RVM inference scale: auto or a number in (0, 1].")
    matte_parser.add_argument("--model-bundle", help="Verified offline RVM code and weights directory.")
    matte_parser.add_argument("--temporal-mode", choices=["ema", "adaptive"],
                              help="Smoothing method; adaptive reduces history at changing edges.")
    matte_parser.add_argument("--temporal-motion-threshold", type=float,
                              help="Normalized alpha change threshold in (0, 1], default 0.15.")
    matte_parser.add_argument(
        "--output",
        required=True,
        help="Output video path. Transparent mode writes a PNG sequence directory.",
    )
    matte_parser.add_argument(
        "--model",
        default=None,
        choices=["dummy", "rvm", "rvm_mobilenetv3", "rvm_resnet50", "birefnet"],
        help="Matting model to use; BiRefNet is an optional experimental backend.",
    )
    matte_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model checkpoint path.",
    )
    matte_parser.add_argument(
        "--device",
        default=None,
        help="Inference device: auto, cpu, or cuda.",
    )
    matte_parser.add_argument(
        "--mode",
        default=None,
        choices=["alpha", "green", "foreground", "blur", "transparent", "image", "side_by_side"],
        help="Output rendering mode. Transparent mode writes BGRA PNG frames.",
    )
    matte_parser.add_argument(
        "--background",
        default=None,
        help="Background image path used when --mode image.",
    )
    matte_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    matte_parser.add_argument(
        "--edge-feather",
        type=int,
        default=None,
        help="Optional alpha edge feather radius in pixels.",
    )
    matte_parser.add_argument(
        "--mask-open",
        type=int,
        default=None,
        help="Optional morphology opening radius for removing small foreground specks.",
    )
    matte_parser.add_argument(
        "--mask-close",
        type=int,
        default=None,
        help="Optional morphology closing radius for filling small background holes.",
    )
    matte_parser.add_argument(
        "--temporal-smooth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable temporal smoothing over alpha frames (off by default).",
    )
    matte_parser.add_argument(
        "--temporal-blend",
        type=float,
        default=None,
        help="Previous-frame blend ratio for temporal smoothing.",
    )
    matte_parser.add_argument(
        "--enable-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable distance estimation and labeling (Near/Mid/Far).",
    )
    matte_parser.add_argument(
        "--depth-interval",
        type=int,
        default=None,
        help="Run depth estimation every N frames (default: 5).",
    )

    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio demo UI.")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port.")
    ui_parser.add_argument("--model-bundle", help="Verified offline RVM bundle; selects its model and clears checkpoint.")
    ui_parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create a public Gradio share link.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    if args.command == "doctor":
        from cv_human_video_matting.diagnostics import check_environment, save_diagnostics
        report = check_environment(args.config, requirements_path=args.requirements)
        for check in report["checks"]:
            print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        if args.report:
            print(f"Environment report: {save_diagnostics(report, args.report)}")
        if not report["ready"]:
            raise SystemExit(1)
        return
    if args.command == "summarize":
        from glob import glob
        from cv_human_video_matting.evaluation import write_evaluation_summary
        paths = []
        for value in args.reports:
            matches = sorted(glob(value))
            if not matches:
                parser.error(f"No report matched: {value}")
            paths.extend(matches)
        paths = list(dict.fromkeys(paths))
        print(f"Evaluation summary: {write_evaluation_summary(paths, args.output)}")
        return
    if args.command == "ui":
        from cv_human_video_matting.utils.config import load_config

        if not 1 <= args.port <= 65535:
            parser.error("Server port must be between 1 and 65535.")
        config = load_config(args.config)
        configured_bundle = config.get("matting", {}).get("bundle")
        if args.model_bundle is not None or configured_bundle is not None:
            config = config_with_model_bundle(config, args.model_bundle or configured_bundle,
                                              select_model=args.model_bundle is not None)
        from cv_human_video_matting.ui.gradio_app import launch_demo

        launch_demo(server_name=args.host, server_port=args.port, share=args.share, config=config)
        return
    from cv_human_video_matting.io.video import copy_video, probe_video, export_video_for_delivery
    from cv_human_video_matting.pipeline import process_video, validate_video_request
    from cv_human_video_matting.utils.config import (
        config_bool, config_float, config_int, config_value, load_config, parse_background_color,
        validate_model_settings,
    )
    from cv_human_video_matting.utils.device import resolve_device
    config = load_config(args.config)

    if args.command == "inspect":
        info = probe_video(args.input)
        print(info.to_report())
        return

    if args.command == "copy":
        output = copy_video(args.input, args.output, max_frames=args.max_frames)
        print(f"Copied video to: {output}")
        return

    if args.command == "matte":
        if args.report and Path(args.report).suffix.lower() != ".json":
            parser.error("Report path must use the .json extension.")
        if args.report and Path(args.report).resolve() in {
            Path(args.input).resolve(), Path(args.output).resolve()
        }:
            parser.error("Report path must differ from input and output paths.")
        model = args.model or config_value(config, "matting.model", "rvm_mobilenetv3")
        checkpoint = args.checkpoint
        bundle_path = args.model_bundle
        if checkpoint is None and bundle_path is None:
            if model == config_value(config, "matting.model", "rvm_mobilenetv3"):
                checkpoint = config_value(config, "matting.checkpoint")
                bundle_path = config_value(config, "matting.bundle")
        requested_device = args.device or config_value(config, "device.mode", "auto")
        device = resolve_device(requested_device)
        mode = args.mode or config_value(config, "render.mode", "green")
        if mode != "transparent" and Path(args.output).suffix.lower() != ".mp4":
            parser.error("Delivery video output must use the .mp4 extension.")
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error("Input and output paths must differ.")
        resize_short = args.resize_short if args.resize_short is not None else config_value(config, "input.resize_short")
        if resize_short is not None and resize_short < 2:
            parser.error("--resize-short must be at least 2.")
        background_color = args.background_color or config_value(config, "render.background_color", "#00FF00")
        color_bgr = parse_background_color(background_color)
        preserve_audio = args.preserve_audio if args.preserve_audio is not None else config_bool(config, "render.preserve_audio", True)
        depth_model = config_value(config, "depth.model", "MiDaS_small")
        depth_near = config_float(config, "depth.thresholds.near", 0.66)
        depth_mid = config_float(config, "depth.thresholds.mid", 0.33)
        edge_feather = (
            args.edge_feather
            if args.edge_feather is not None
            else config_int(config, "postprocess.edge_feather", 0)
        )
        mask_open = (
            args.mask_open
            if args.mask_open is not None
            else config_int(config, "postprocess.mask_open", 0)
        )
        mask_close = (
            args.mask_close
            if args.mask_close is not None
            else config_int(config, "postprocess.mask_close", 0)
        )
        temporal_smooth = (
            args.temporal_smooth
            if args.temporal_smooth is not None
            else config_bool(config, "postprocess.enable_temporal_smooth", False)
        )
        temporal_blend = (
            args.temporal_blend
            if args.temporal_blend is not None
            else config_float(config, "postprocess.temporal_alpha", 0.75)
        )
        enable_depth = (
            args.enable_depth
            if args.enable_depth is not None
            else config_bool(config, "depth.enabled", False)
        )
        depth_interval = (
            args.depth_interval
            if args.depth_interval is not None
            else config_int(config, "depth.interval", 5)
        )
        temporal_mode = args.temporal_mode or config_value(config, "postprocess.temporal_mode", "adaptive")
        temporal_motion_threshold = (args.temporal_motion_threshold
            if args.temporal_motion_threshold is not None
            else config_float(config, "postprocess.temporal_motion_threshold", 0.15))
        downsample_ratio = validate_model_settings(model, checkpoint, bundle_path,
            args.downsample_ratio if args.downsample_ratio is not None
            else config_value(config, "matting.downsample_ratio", "auto"))
        prepared = validate_video_request(
            input_path=args.input, output_path=args.output, mode=mode,
            max_frames=args.max_frames, background_path=args.background,
            edge_feather=edge_feather, mask_open=mask_open, mask_close=mask_close,
            temporal_smooth=temporal_smooth, temporal_blend=temporal_blend,
            temporal_mode=temporal_mode, temporal_motion_threshold=temporal_motion_threshold,
            enable_depth=enable_depth, depth_interval=depth_interval, resize_short=resize_short,
            depth_model=depth_model, depth_near=depth_near, depth_mid=depth_mid,
            background_color_bgr=color_bgr,
        )
        from cv_human_video_matting.reporting import validate_artifact_paths
        from cv_human_video_matting.utils.publication import PublicationItem, publish_artifacts

        destination = prepared.output_path
        validate_artifact_paths(
            args.report, args.input, destination,
            asset_paths=[path for path in (checkpoint, args.background, bundle_path) if path],
        )
        started_at = perf_counter()
        runner = create_matting_runner(
            model=model,
            checkpoint=checkpoint,
            device=device,
            downsample_ratio=downsample_ratio,
            bundle_path=bundle_path,
        )
        # Prepare media and its requested report before replacing either artifact.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".matting-", dir=destination.parent) as staging:
            pipeline_output = Path(staging) / ("transparent.mp4" if mode == "transparent" else "frames.mkv")
            stats = process_video(
                input_path=args.input,
                output_path=pipeline_output,
                runner=runner,
                mode=mode,
                max_frames=args.max_frames,
                background_path=args.background,
                edge_feather=edge_feather,
                mask_open=mask_open,
                mask_close=mask_close,
                temporal_smooth=temporal_smooth,
                temporal_blend=temporal_blend,
                temporal_mode=temporal_mode,
                temporal_motion_threshold=temporal_motion_threshold,
                enable_depth=enable_depth,
                depth_interval=depth_interval,
                device=device,
                resize_short=resize_short,
                depth_model=depth_model,
                depth_near=depth_near,
                depth_mid=depth_mid,
                background_color_bgr=color_bgr,
                progress_callback=ConsoleProgress() if args.progress else None,
            )
            export_started = perf_counter()
            if mode != "transparent":
                delivered = export_video_for_delivery(stats.output_path, Path(staging) / "delivery.mp4",
                    source_path=args.input, preserve_audio=preserve_audio)
                stats = replace(stats, output_path=delivered)
            export_seconds = perf_counter() - export_started
            total_seconds = perf_counter() - started_at
            staged_report = None
            if args.report:
                from cv_human_video_matting.reporting import write_run_report

                staged_report = write_run_report(Path(staging) / "run.json", args.input, stats, {
                    "model": model, "checkpoint": checkpoint, "device": device, "mode": mode,
                    "model_bundle": str(bundle_path) if bundle_path is not None else None,
                    "downsample_ratio": downsample_ratio,
                    "max_frames": args.max_frames, "background": args.background,
                    "edge_feather": edge_feather, "mask_open": mask_open, "mask_close": mask_close,
                    "temporal_smooth": temporal_smooth, "temporal_blend": temporal_blend,
                    "temporal_mode": temporal_mode, "temporal_motion_threshold": temporal_motion_threshold,
                    "intermediate_codec": "PNG" if mode == "transparent" else "FFV1",
                    "enable_depth": enable_depth, "depth_interval": depth_interval,
                    "depth_model": depth_model, "depth_near": depth_near, "depth_mid": depth_mid,
                    "resize_short": resize_short, "background_color": background_color,
                    "preserve_audio": preserve_audio and mode != "transparent",
                }, total_seconds, export_seconds=export_seconds, published_output_path=destination,
                    model_bundle_info=getattr(runner, "bundle_info", None))
            if mode == "transparent" and destination.exists():
                # Preserve user notes in an existing sequence directory. Every
                # newly added frame is exclusive and can be rolled back alone.
                artifacts = [PublicationItem(path, destination / path.name, replace_existing=False)
                             for path in sorted(stats.output_path.glob("frame_*.png"))]
            else:
                artifacts = [PublicationItem(stats.output_path, destination,
                                             replace_existing=mode != "transparent")]
            if staged_report is not None:
                artifacts.append(PublicationItem(staged_report, Path(args.report)))
            publish_artifacts(artifacts)
            stats = replace(stats, output_path=destination)
        if args.report:
            print(f"Run report: {args.report}")
        print(f"Matting output: {stats.output_path}")
        print(f"Model: {model}")
        print(f"Mode: {mode}")
        print(f"Device: {device}")
        print(f"Total including model loading and export (excludes final publication): {total_seconds:.3f}s")
        print(
            f"Processed {stats.frames} frame(s) in {stats.elapsed_seconds:.3f}s "
            f"({stats.fps:.2f} FPS)."
        )
        return

if __name__ == "__main__":
    main()
