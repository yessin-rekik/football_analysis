import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..calibration.calibrator import PitchCalibrator


def _synthetic_pixel(world_xy, scale=10.0, offset=(100.0, 50.0)):
    x, y = world_xy
    return (x * scale + offset[0], y * scale + offset[1])


def _build_synthetic_frame(world_keypoints, indices_to_use, confidence=0.9):
    keypoints_px = np.zeros((29, 2), dtype=np.float32)
    confidences = np.zeros((29,), dtype=np.float32)
    for idx in indices_to_use:
        keypoints_px[idx] = _synthetic_pixel(world_keypoints[idx])
        confidences[idx] = confidence
    return keypoints_px, confidences


BROADCAST_INDICES = [0, 9, 16, 25, 11, 12, 13, 15]


# ---------------------------------------------------------------------------
# _weighted_points -- unit test of the duplication helper in isolation
# ---------------------------------------------------------------------------

def test_weighted_points_duplicates_by_confidence():
    src = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    dst = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    confidences = np.array([0.0, 1.0], dtype=np.float32)

    fit_src, fit_dst = PitchCalibrator._weighted_points(src, dst, confidences, max_multiplier=3)

    # confidence 0.0 -> 1x, confidence 1.0 -> 3x (max_multiplier)
    assert len(fit_src) == 1 + 3
    assert len(fit_dst) == 1 + 3
    assert np.sum(np.all(fit_src == src[0], axis=1)) == 1
    assert np.sum(np.all(fit_src == src[1], axis=1)) == 3


def test_weighted_points_midpoint_confidence():
    src = np.array([[5.0, 5.0]], dtype=np.float32)
    dst = np.array([[50.0, 50.0]], dtype=np.float32)
    confidences = np.array([0.5], dtype=np.float32)

    fit_src, _ = PitchCalibrator._weighted_points(src, dst, confidences, max_multiplier=3)
    # 1 + round(0.5 * 2) = 1 + 1 = 2
    assert len(fit_src) == 2


# ---------------------------------------------------------------------------
# Confidence weighting doesn't hurt clean data (sanity/regression)
# ---------------------------------------------------------------------------

def test_confidence_weighting_does_not_hurt_clean_data():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg, use_confidence_weighting=True)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert info.reprojection_error_px_mean < 1.0  # still near-zero on perfect synthetic data


# ---------------------------------------------------------------------------
# Reprojection-error rejection -- turning the diagnostic into a decision
# ---------------------------------------------------------------------------

def _broadcast_frame_with_one_bad_point(world_kps, bad_index=15, offset=(60.0, 60.0)):
    """7 perfect correspondences + 1 point nudged significantly off --
    simulates a real detection error on one keypoint, not a totally
    degenerate frame. findHomography with RANSAC will still succeed (the
    other 7 points anchor a good fit), but the per-point reprojection
    error at the bad point will be large, and averaging across ALL points
    (not just RANSAC's internal inliers) is what should surface it."""
    keypoints_px, confidences = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    bad_px = _synthetic_pixel(world_kps[bad_index])
    keypoints_px[bad_index] = (bad_px[0] + offset[0], bad_px[1] + offset[1])
    return keypoints_px, confidences


def test_no_rejection_by_default_even_with_a_bad_point():
    """Locks in backward-compatible behavior: with no thresholds set
    (the default), a technically-successful-but-imperfect calibration
    still reports CALIBRATED_FROM_KEYPOINTS, same as before this feature
    existed -- only the diagnostic values change."""
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)  # thresholds default to None
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _broadcast_frame_with_one_bad_point(world_kps)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert info.reprojection_error_px_mean > 5.0  # the bad point is visible in the diagnostic


def test_rejects_when_mean_error_exceeds_threshold():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg, max_mean_reprojection_error=5.0)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _broadcast_frame_with_one_bad_point(world_kps)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.NOT_CALIBRATED
    assert info.homography is None


def test_rejects_when_max_error_exceeds_threshold():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg, max_max_reprojection_error=50.0)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _broadcast_frame_with_one_bad_point(world_kps)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.NOT_CALIBRATED


def test_generous_threshold_still_accepts():
    """Rejection should only fire when the threshold is actually exceeded
    -- a loose threshold on the same imperfect-but-usable frame should
    still succeed."""
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg, max_mean_reprojection_error=50.0)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _broadcast_frame_with_one_bad_point(world_kps)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS


def test_clean_data_passes_strict_threshold():
    """A strict threshold shouldn't reject perfectly good data -- proves
    the rejection is about genuinely bad fits, not just being strict."""
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg, max_mean_reprojection_error=1.0, max_max_reprojection_error=2.0)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS


def test_rejected_calibration_does_not_set_homography_for_pixel_to_world():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg, max_mean_reprojection_error=5.0)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _broadcast_frame_with_one_bad_point(world_kps)
    calibrator.calibrate_frame(keypoints_px, confidences)

    assert calibrator.pixel_to_world((500, 500)) is None