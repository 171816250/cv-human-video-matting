import numpy as np

from cv_human_video_matting.postprocess import clean_alpha, feather_alpha, refine_alpha


def test_feather_alpha_softens_hard_edges_without_changing_shape_or_dtype():
    alpha = np.zeros((7, 7), dtype=np.uint8)
    alpha[:, 4:] = 255

    feathered = feather_alpha(alpha, radius=1)

    assert feathered.shape == alpha.shape
    assert feathered.dtype == np.uint8
    assert feathered.min() >= 0
    assert feathered.max() <= 255
    assert np.any((feathered > 0) & (feathered < 255))


def test_clean_alpha_removes_specks_and_fills_small_holes():
    speck = np.zeros((9, 9), dtype=np.uint8)
    speck[4, 4] = 255
    holed = np.full((9, 9), 255, dtype=np.uint8)
    holed[4, 4] = 0

    opened = clean_alpha(speck, open_radius=1)
    closed = clean_alpha(holed, close_radius=1)

    assert opened[4, 4] == 0
    assert closed[4, 4] == 255


def test_refine_alpha_applies_cleaning_before_feathering():
    alpha = np.zeros((9, 9), dtype=np.uint8)
    alpha[2:7, 2:7] = 255
    alpha[4, 4] = 0

    refined = refine_alpha(alpha, close_radius=1, feather_radius=1)

    assert refined.shape == alpha.shape
    assert refined.dtype == np.uint8
    assert refined[4, 4] > 0
    assert np.any((refined > 0) & (refined < 255))
