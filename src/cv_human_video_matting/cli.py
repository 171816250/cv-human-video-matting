"""Command line interface for iterative project milestones."""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_human_video_matting.io.video import copy_video, probe_video
from cv_human_video_matting.matting.factory import create_matting_runner
from cv_human_video_matting.pipeline import process_video
from cv_human_video_matting.utils.config import load_config


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
    matte_parser.add_argument("--output", required=True, help="Output video path.")
    matte_parser.add_argument(
        "--model",
        default="rvm_mobilenetv3",
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
        default="cpu",
        help="Inference device, such as cpu or cuda.",
    )
    matte_parser.add_argument(
        "--mode",
        default="green",
        choices=["alpha", "green", "foreground", "blur", "transparent", "image", "side_by_side"],
        help="Output rendering mode.",
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
        default=0,
        help="Optional alpha edge feather radius in pixels.",
    )
    matte_parser.add_argument(
        "--mask-open",
        type=int,
        default=0,
        help="Optional morphology opening radius for removing small foreground specks.",
    )
    matte_parser.add_argument(
        "--mask-close",
        type=int,
        default=0,
        help="Optional morphology closing radius for filling small background holes.",
    )
    matte_parser.add_argument(
        "--temporal-smooth",
        action="store_true",
        help="Enable simple exponential smoothing over alpha frames.",
    )
    matte_parser.add_argument(
        "--temporal-blend",
        type=float,
        default=0.75,
        help="Previous-frame blend ratio for temporal smoothing.",
    )
    matte_parser.add_argument(
        "--enable-depth",
        action="store_true",
        help="Enable distance estimation and labeling (Near/Mid/Far).",
    )
    matte_parser.add_argument(
        "--depth-interval",
        type=int,
        default=5,
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
        runner = create_matting_runner(
            model=args.model,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        stats = process_video(
            input_path=args.input,
            output_path=args.output,
            runner=runner,
            mode=args.mode,
            max_frames=args.max_frames,
            background_path=args.background,
            edge_feather=args.edge_feather,
            mask_open=args.mask_open,
            mask_close=args.mask_close,
            temporal_smooth=args.temporal_smooth,
            temporal_blend=args.temporal_blend,
            enable_depth=args.enable_depth,
            depth_interval=args.depth_interval,
            device=args.device,
        )
        print(f"Matting output: {stats.output_path}")
        print(
            f"Processed {stats.frames} frame(s) in {stats.elapsed_seconds:.3f}s "
            f"({stats.fps:.2f} FPS)."
        )
        return

    if args.command == "ui":
        from cv_human_video_matting.ui.gradio_app import launch_demo

        launch_demo(server_name=args.host, server_port=args.port, share=args.share)
        return

    project_name = config.get("project", {}).get("name", "cv-human-video-matting")
    parser.print_help()
    print()
    print(f"{project_name} is ready for Iteration 1.")
    print("Try: python app.py inspect --input data/input/demo.mp4")


if __name__ == "__main__":
    main()
