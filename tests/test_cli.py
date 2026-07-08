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


def test_matte_command_with_transparent_mode_writes_png_sequence(tmp_path, capsys):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "transparent.mp4"
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
            "transparent",
            "--max-frames",
            "2",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "transparent_frames"
    frame_paths = sorted(output_dir.glob("frame_*.png"))
    assert "Matting output:" in captured.out
    assert not output_path.exists()
    assert len(frame_paths) == 2

    frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
    assert frame is not None
    assert frame.shape == (24, 32, 4)


def test_matte_command_uses_config_defaults(tmp_path, capsys):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "from-config.mp4"
    config_path = tmp_path / "config.yaml"
    _write_test_video(input_path)
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: test-project",
                "device:",
                "  mode: cpu",
                "matting:",
                "  model: dummy",
                "render:",
                "  mode: alpha",
                "postprocess:",
                "  enable_temporal_smooth: true",
                "  temporal_alpha: 0.5",
                "  edge_feather: 1",
                "  mask_open: 0",
                "  mask_close: 1",
                "depth:",
                "  enabled: false",
                "  interval: 5",
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            "--config",
            str(config_path),
            "matte",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--max-frames",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert "Model: dummy" in captured.out
    assert "Mode: alpha" in captured.out
    assert "Device: cpu" in captured.out
    assert output_path.exists()
    assert probe_video(output_path).frame_count == 2
