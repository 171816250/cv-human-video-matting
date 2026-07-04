import numpy as np
import pytest

from cv_human_video_matting.matting.dummy import DummyMattingRunner
from cv_human_video_matting.matting.factory import create_matting_runner


def test_factory_creates_dummy_runner():
    runner = create_matting_runner("dummy")

    assert isinstance(runner, DummyMattingRunner)
    result = runner.predict(np.zeros((16, 16, 3), dtype=np.uint8))
    assert result.alpha.shape == (16, 16)


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported matting model"):
        create_matting_runner("unknown-model")
