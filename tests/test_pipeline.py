from pathlib import Path

import cv2
import numpy as np

from cv_human_video_matting.io.video import probe_video
from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.pipeline import process_video, render_matting_frame


def _write_test_video(path: Path, size: tuple[int, int] = (64, 48), frames: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = index * 20
            frame[:, :, 1] = 80
            frame[:, :, 2] = 160
            writer.write(frame)
    finally:
        writer.release()


def test_process_video_writes_alpha_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "alpha.mp4"
    _write_test_video(input_path)

    stats = process_video(
        input_path=input_path,
        output_path=output_path,
        runner=DummyMattingRunner(),
        mode="alpha",
        max_frames=4,
    )

    assert output_path.exists()
    assert stats.output_path == output_path
    assert stats.frames == 4
    output_info = probe_video(output_path)
    assert output_info.width == 64
    assert output_info.height == 48


def test_process_video_writes_side_by_side_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "side.mp4"
    _write_test_video(input_path)

    stats = process_video(
        input_path=input_path,
        output_path=output_path,
        runner=DummyMattingRunner(),
        mode="side_by_side",
        max_frames=3,
    )

    assert stats.frames == 3
    output_info = probe_video(output_path)
    assert output_info.width == 128
    assert output_info.height == 48


def test_process_video_writes_image_background_output(tmp_path):
    input_path = tmp_path / "input.mp4"
    background_path = tmp_path / "background.jpg"
    output_path = tmp_path / "image.mp4"
    _write_test_video(input_path)
    background = np.full((24, 96, 3), (20, 180, 40), dtype=np.uint8)
    assert cv2.imwrite(str(background_path), background)

    stats = process_video(
        input_path=input_path,
        output_path=output_path,
        runner=DummyMattingRunner(),
        mode="image",
        max_frames=4,
        background_path=background_path,
    )

    assert stats.frames == 4
    output_info = probe_video(output_path)
    assert output_info.width == 64
    assert output_info.height == 48


def test_process_video_accepts_postprocess_options(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "postprocess.mp4"
    _write_test_video(input_path)

    stats = process_video(
        input_path=input_path,
        output_path=output_path,
        runner=DummyMattingRunner(),
        mode="alpha",
        max_frames=3,
        edge_feather=1,
        mask_close=1,
        temporal_smooth=True,
        temporal_blend=0.5,
    )

    assert output_path.exists()
    assert stats.frames == 3


def test_render_matting_frame_returns_single_rendered_frame():
    frame = np.full((48, 64, 3), (20, 80, 160), dtype=np.uint8)

    rendered, alpha = render_matting_frame(
        frame,
        runner=DummyMattingRunner(),
        mode="green",
        edge_feather=1,
        mask_close=1,
    )

    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    assert alpha.shape == frame.shape[:2]
    # Alpha can be float32 or uint8 depending on postprocessing
    assert alpha.dtype in (np.float32, np.uint8)
