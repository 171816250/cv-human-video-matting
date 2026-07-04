from pathlib import Path

import cv2
import numpy as np

from cv_human_video_matting.cli import main
from cv_human_video_matting.io.video import probe_video


def _write_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(4):
            frame = np.full((24, 32, 3), 120, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_matte_command_with_dummy_runner_writes_output(tmp_path, capsys):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "out.mp4"
    _write_test_video(input_path)

    main(
        [
            "matte",
            "--model",
            "dummy",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            "green",
            "--max-frames",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert "Matting output:" in captured.out
    assert output_path.exists()
    assert probe_video(output_path).frame_count == 2


def test_matte_command_with_image_background_writes_output(tmp_path, capsys):
    input_path = tmp_path / "input.mp4"
    background_path = tmp_path / "background.jpg"
    output_path = tmp_path / "out-image.mp4"
    _write_test_video(input_path)
    background = np.full((24, 64, 3), (30, 180, 90), dtype=np.uint8)
    assert cv2.imwrite(str(background_path), background)

    main(
        [
            "matte",
            "--model",
            "dummy",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            "image",
            "--background",
            str(background_path),
            "--max-frames",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert "Matting output:" in captured.out
    assert output_path.exists()
    assert probe_video(output_path).frame_count == 2


def test_matte_command_accepts_postprocess_flags(tmp_path, capsys):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "out-postprocess.mp4"
    _write_test_video(input_path)

    main(
        [
            "matte",
            "--model",
            "dummy",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            "alpha",
            "--max-frames",
            "2",
            "--edge-feather",
            "1",
            "--mask-close",
            "1",
            "--temporal-smooth",
            "--temporal-blend",
            "0.5",
        ]
    )

    captured = capsys.readouterr()
    assert "Matting output:" in captured.out
    assert output_path.exists()
    assert probe_video(output_path).frame_count == 2
