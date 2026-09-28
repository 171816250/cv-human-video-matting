"""Command line interface for iterative project milestones."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory
import sys


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
        default="configs/default.yaml",
        help="Path to the project YAML config.",
    )

    subparsers = parser.add_subparsers(dest="command")
    summary_parser = subparsers.add_parser("summarize", help="Summarize comparable run reports.")
    summary_parser.add_argument("--reports", nargs="+", required=True, help="Run JSON paths or glob patterns.")
    summary_parser.add_argument("--output", required=True, help="Summary .html or .json path.")
    doctor_parser = subparsers.add_parser("doctor", help="Check local readiness without downloads.")
    doctor_parser.add_argument("--report", help="Save environment checks as JSON.")

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
        help="Enable simple exponential smoothing over alpha frames.",
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
        report = check_environment(args.config)
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
    from cv_human_video_matting.io.video import copy_video, probe_video, export_video_for_delivery
    from cv_human_video_matting.pipeline import process_video
    from cv_human_video_matting.utils.config import (
        config_bool, config_float, config_int, config_value, load_config, parse_background_color,
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
        if checkpoint is None and model == config_value(config, "matting.model", "rvm_mobilenetv3"):
            checkpoint = config_value(config, "matting.checkpoint")
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
        started_at = perf_counter()
        downsample_ratio = args.downsample_ratio or config_value(config, "matting.downsample_ratio", "auto")
        runner = create_matting_runner(
            model=model,
            checkpoint=checkpoint,
            device=device,
            downsample_ratio=downsample_ratio,
        )
        # Keep any existing delivery file intact until processing and encoding succeed.
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".matting-", dir=destination.parent) as staging:
            pipeline_output = destination if mode == "transparent" else Path(staging) / "frames.mp4"
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
                delivered = export_video_for_delivery(stats.output_path, destination,
                    source_path=args.input, preserve_audio=preserve_audio)
                stats = replace(stats, output_path=delivered)
            export_seconds = perf_counter() - export_started
        total_seconds = perf_counter() - started_at
        if args.report:
            from cv_human_video_matting.reporting import write_run_report

            report = write_run_report(args.report, args.input, stats, {
                "model": model, "checkpoint": checkpoint, "device": device, "mode": mode,
                "downsample_ratio": downsample_ratio,
                "max_frames": args.max_frames, "background": args.background,
                "edge_feather": edge_feather, "mask_open": mask_open, "mask_close": mask_close,
                "temporal_smooth": temporal_smooth, "temporal_blend": temporal_blend,
                "enable_depth": enable_depth, "depth_interval": depth_interval,
                "depth_model": depth_model, "depth_near": depth_near, "depth_mid": depth_mid,
                "resize_short": resize_short, "background_color": background_color,
                "preserve_audio": preserve_audio and mode != "transparent",
            }, total_seconds, export_seconds=export_seconds)
            print(f"Run report: {report}")
        print(f"Matting output: {stats.output_path}")
        print(f"Model: {model}")
        print(f"Mode: {mode}")
        print(f"Device: {device}")
        print(f"Total including model loading and export: {total_seconds:.3f}s")
        print(
            f"Processed {stats.frames} frame(s) in {stats.elapsed_seconds:.3f}s "
            f"({stats.fps:.2f} FPS)."
        )
        return

    if args.command == "ui":
        from cv_human_video_matting.ui.gradio_app import launch_demo

        launch_demo(server_name=args.host, server_port=args.port, share=args.share, config=config)
        return

if __name__ == "__main__":
    main()
