"""Command line interface for iterative project milestones."""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_human_video_matting.io.video import copy_video, probe_video
from cv_human_video_matting.matting.factory import create_matting_runner
from cv_human_video_matting.pipeline import process_video
from cv_human_video_matting.utils.config import (
    config_bool,
    config_float,
    config_int,
    config_value,
    load_config,
)
from cv_human_video_matting.utils.device import resolve_device


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
    matte_parser.add_argument(
        "--output",
        required=True,
        help="Output video path. Transparent mode writes a PNG sequence directory.",
    )
    matte_parser.add_argument(
        "--model",
        default=None,
        choices=["dummy", "rvm", "rvm_mobilenetv3", "rvm_resnet50", "birefnet"],
        help="Matting model to use (birefnet=high precision SOTA).",
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
        model = args.model or config_value(config, "matting.model", "rvm_mobilenetv3")
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = config_value(config, "matting.checkpoint")
        requested_device = args.device or config_value(config, "device.mode", "auto")
        device = resolve_device(requested_device)
        mode = args.mode or config_value(config, "render.mode", "green")
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
        runner = create_matting_runner(
            model=model,
            checkpoint=checkpoint,
            device=device,
        )
        stats = process_video(
            input_path=args.input,
            output_path=args.output,
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
        )
        print(f"Matting output: {stats.output_path}")
        print(f"Model: {model}")
        print(f"Mode: {mode}")
        print(f"Device: {device}")
        print(
            f"Processed {stats.frames} frame(s) in {stats.elapsed_seconds:.3f}s "
            f"({stats.fps:.2f} FPS)."
        )
        return

    if args.command == "ui":
        from cv_human_video_matting.ui.gradio_app import launch_demo

        launch_demo(server_name=args.host, server_port=args.port, share=args.share, config=config)
        return

    project_name = config.get("project", {}).get("name", "cv-human-video-matting")
    parser.print_help()
    print()
    print(f"{project_name} is ready for Iteration 1.")
    print("Try: python app.py inspect --input data/input/demo.mp4")


if __name__ == "__main__":
    main()
