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
    """Enough points, but all crammed into a tiny image region -- still
    a close-up, not a usable calibration frame."""
    clf = HeuristicSceneClassifier()
    kps, confs = _kps_confs({
        1: (900, 500), 2: (905, 505), 3: (910, 510), 4: (915, 515), 5: (920, 520),
    })
    result = clf.classify(frame=np.zeros(FRAME_SHAPE, dtype=np.uint8),
                           keypoints_px=kps, keypoint_confidences=confs)
    assert result.scene_type == SceneType.CLOSE_UP


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