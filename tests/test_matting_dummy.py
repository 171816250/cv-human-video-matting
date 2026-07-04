import numpy as np

from cv_human_video_matting.matting.dummy import DummyMattingRunner


def test_dummy_runner_returns_center_ellipse_alpha():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    runner = DummyMattingRunner()

    result = runner.predict(frame)

    assert result.alpha.shape == (20, 30)
    assert result.alpha.dtype == np.uint8
    assert result.alpha[10, 15] == 255
    assert result.alpha[0, 0] == 0
    assert result.foreground is None


def test_dummy_runner_output_matches_input_frame_size():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    runner = DummyMattingRunner()

    result = runner.predict(frame)

    assert result.alpha.shape == frame.shape[:2]
