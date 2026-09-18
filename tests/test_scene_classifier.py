import numpy as np
import pytest

from ..calibration.scene_types import SceneType
from ..calibration.scene_classifier import HeuristicSceneClassifier


FRAME_SHAPE = (1080, 1920, 3)  # h, w, c


def _kps_confs(visible: dict, num_total: int = 29, default_conf: float = 0.9):
    """Build (29,2) keypoint array and (29,) confidence array from a dict
    of {index: (x, y)} for the visible ones; everything else near-zero
    confidence at a placeholder location."""
    kps = np.zeros((num_total, 2), dtype=np.float32)
    confs = np.full((num_total,), 0.05, dtype=np.float32)
    for idx, (x, y) in visible.items():
        kps[idx] = (x, y)
        confs[idx] = default_conf
    return kps, confs


def test_close_up_no_keypoints():
    clf = HeuristicSceneClassifier()
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=None, keypoint_confidences=None)
    assert result.scene_type == SceneType.CLOSE_UP


def test_close_up_few_keypoints():
    clf = HeuristicSceneClassifier()
    kps, confs = _kps_confs({2: (900, 500), 4: (910, 520)})  # only 2 visible
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.CLOSE_UP


def test_close_up_clustered_keypoints():
    """Enough points, but all crammed into a tiny image region on BOTH
    axes -- still a close-up, not a usable calibration frame. Locks in
    that the per-axis fix (see module docstring) doesn't accidentally
    make the classifier permissive about genuine tight clusters -- it
    only stops rejecting spreads that are wide on ONE axis."""
    clf = HeuristicSceneClassifier()
    kps, confs = _kps_confs({
        1: (900, 500), 2: (905, 505), 3: (910, 510), 4: (915, 515), 5: (920, 520),
    })
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.CLOSE_UP
    assert result.details["width_fraction"] < 0.2
    assert result.details["height_fraction"] < 0.2


def test_broadcast_wide_with_center_visible():
    clf = HeuristicSceneClassifier()
    # Spread across most of the frame, including center-circle points
    kps, confs = _kps_confs({
        0: (50, 950), 16: (1870, 950), 9: (50, 100), 25: (1870, 100),
        11: (960, 950), 12: (960, 100), 15: (960, 525),
        13: (960, 700), 14: (960, 350),
    })
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.BROADCAST_WIDE


def test_low_angle_corner_no_center_features():
    """Box + boundary points visible (one end of the pitch, close to the
    touchline), but nothing from the halfway line / center circle --
    the corner-kick signature."""
    clf = HeuristicSceneClassifier()
    kps, confs = _kps_confs({
        9: (100, 900), 0: (100, 150),         # left boundary corners
        1: (700, 150), 2: (750, 300),          # box points
        3: (700, 900), 4: (750, 750),
    })
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.LOW_ANGLE_CORNER


def test_confidence_threshold_excludes_low_confidence_points():
    clf = HeuristicSceneClassifier()
    kps = np.zeros((29, 2), dtype=np.float32)
    confs = np.full((29,), 0.3, dtype=np.float32)  # all below default threshold of 0.5
    kps[0] = (100, 100)
    kps[16] = (1800, 100)
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.CLOSE_UP  # nothing clears the confidence bar


# ---- Real-footage bug fix: per-axis coverage (see module docstring) ----

def test_wide_horizontal_spread_with_narrow_vertical_band_is_not_close_up():
    """Regression test for a real bug found on actual broadcast footage:
    keypoints spread across ~94% of the frame's WIDTH but confined to a
    narrow vertical band produced a tiny AREA-PRODUCT coverage_fraction
    under the old metric, incorrectly rejecting every frame of an entire
    session as CLOSE_UP even though the points were never a tight
    cluster -- just well-spread on one axis and narrow on the other. The
    fix (independent per-axis thresholds -- either axis is sufficient)
    must accept this pattern and route it on its actual visible-feature
    content, here landing on LOW_ANGLE_CORNER (box + boundary points, no
    center-pitch features)."""
    clf = HeuristicSceneClassifier()
    kps, confs = _kps_confs({
        0: (50, 520),      # boundary
        1: (500, 540),     # box
        17: (1400, 500),   # box
        18: (1850, 560),   # box
    })
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.LOW_ANGLE_CORNER
    assert result.details["width_fraction"] > 0.5
    assert result.details["height_fraction"] < 0.2


def test_worst_case_symmetric_split_at_the_observed_coverage_value_still_passes():
    """Directly validates the mathematical guarantee behind this fix: for
    a fixed area-product p, the LARGER of width_fraction/height_fraction
    is minimized when the two are equal (both sqrt(p)) -- so this is the
    least-favorable possible split for a given product, and it must still
    clear the new per-axis gate. Constructed at p=0.057, matching the
    real footage's observed coverage_fraction, with both axes forced to
    the exact symmetric worst case (sqrt(0.057) =~ 0.2387) -- comfortably
    above the 0.2 default on both axes, but only just."""
    clf = HeuristicSceneClassifier()
    frame_w, frame_h = 1920, 1080
    target_product = 0.057
    side = target_product ** 0.5
    bbox_w = side * frame_w
    bbox_h = side * frame_h
    kps, confs = _kps_confs({
        0: (100, 100),
        1: (100 + bbox_w, 100),
        17: (100, 100 + bbox_h),
        18: (100 + bbox_w, 100 + bbox_h),
    })
    result = clf.classify(frame=np.zeros((frame_h, frame_w, 3), dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type != SceneType.CLOSE_UP


def test_boundary_only_wide_spread_without_center_falls_back_to_broadcast_wide():
    """Exercises a branch that had ZERO test coverage before this fix:
    no center-pitch features and fewer than 2 box points (so
    LOW_ANGLE_CORNER's own routing rule doesn't fire), but >=2 boundary
    points with a wide single-axis spread -- must still resolve to
    BROADCAST_WIDE rather than falling through to UNKNOWN. This is the
    `min_broadcast_fallback_fraction` rule, which got the same
    area-product-to-per-axis fix as the main CLOSE_UP gate, for the
    identical structural reason (see module docstring)."""
    clf = HeuristicSceneClassifier()
    kps, confs = _kps_confs({
        0: (50, 950), 16: (1870, 950), 9: (50, 900), 25: (1870, 900),
    })
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.BROADCAST_WIDE