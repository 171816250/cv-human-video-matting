from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.ui.gradio_app import (
    build_demo,
    prepare_input_preview,
    render_live_frame,
    reset_live_state,
    run_demo,
)


def _write_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(np.full((24, 32, 3), 128, dtype=np.uint8))
    finally:
        writer.release()


def test_build_demo_returns_blocks():
    demo = build_demo()

    assert isinstance(demo, gr.Blocks)


def test_run_demo_with_dummy_runner_writes_video(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    input_path = tmp_path / "input.mp4"
    _write_test_video(input_path)

    output_path, status = run_demo(
        video_path=str(input_path),
        background_path=None,
        model="dummy",
        mode="green",
        max_frames=2,
        device="cpu",
        edge_feather=1,
        mask_open=0,
        mask_close=1,
        temporal_smooth=True,
        temporal_blend=0.5,
    )

    assert Path(output_path).exists()
    assert probe_video(output_path).frame_count == 2
    assert "FPS:" in status


def test_prepare_input_preview_returns_existing_video_path(tmp_path):
    input_path = tmp_path / "input.mp4"
    _write_test_video(input_path)

    preview_path = prepare_input_preview(str(input_path))

    assert preview_path is not None
    assert Path(preview_path).exists()


def test_render_live_frame_with_dummy_runner_returns_image():
    reset_live_state()
    frame = np.full((48, 64, 3), 128, dtype=np.uint8)

    output, status = render_live_frame(
        frame_rgb=frame,
        background_path=None,
        model="dummy",
        mode="green",
        device="cpu",
        max_live_width=64,
        edge_feather=1,
        mask_open=0,
        mask_close=1,
        temporal_smooth=True,
        temporal_blend=0.5,
    )

    assert output is not None
    assert output.shape == frame.shape
    assert "Session FPS:" in status


def test_render_live_frame_requires_background_for_image_mode():
    reset_live_state()
    frame = np.full((48, 64, 3), 128, dtype=np.uint8)

    output, status = render_live_frame(
        frame_rgb=frame,
        background_path=None,
        model="dummy",
        mode="image",
        device="cpu",
        max_live_width=64,
    )

    assert output is None
    assert "background image" in status.lower()
