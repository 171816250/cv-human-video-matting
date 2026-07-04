import numpy as np
import pytest

from cv_human_video_matting.postprocess import TemporalAlphaSmoother


def test_temporal_smoother_returns_first_frame_unchanged():
    alpha = np.full((2, 3), 120, dtype=np.uint8)
    smoother = TemporalAlphaSmoother(blend=0.75)

    output = smoother.apply(alpha)

    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, alpha)


def test_temporal_smoother_blends_previous_and_current_alpha():
    smoother = TemporalAlphaSmoother(blend=0.75)
    previous = np.zeros((2, 2), dtype=np.uint8)
    current = np.full((2, 2), 200, dtype=np.uint8)

    smoother.apply(previous)
    output = smoother.apply(current)

    np.testing.assert_array_equal(output, np.full((2, 2), 50, dtype=np.uint8))


def test_temporal_smoother_reset_discards_previous_state():
    smoother = TemporalAlphaSmoother(blend=0.75)
    smoother.apply(np.zeros((2, 2), dtype=np.uint8))
    smoother.reset()

    output = smoother.apply(np.full((2, 2), 200, dtype=np.uint8))

    np.testing.assert_array_equal(output, np.full((2, 2), 200, dtype=np.uint8))


@pytest.mark.parametrize("blend", [-0.1, 1.1])
def test_temporal_smoother_rejects_blend_outside_unit_range(blend):
    with pytest.raises(ValueError, match="blend"):
        TemporalAlphaSmoother(blend=blend)
