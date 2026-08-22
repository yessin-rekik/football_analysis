import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import WorldPoint
from ..calibration.calibrator import PitchCalibrator


def _synthetic_pixel(world_xy, scale=10.0, offset=(100.0, 50.0)):
    """A simple, known world -> pixel mapping (10 px per meter, offset
    origin). Good enough to prove the homography math is correct without
    needing real footage -- if the calibrator can't recover THIS mapping
    exactly, it can't be trusted on anything harder."""
    x, y = world_xy
    return (x * scale + offset[0], y * scale + offset[1])


def _build_synthetic_frame(world_keypoints: np.ndarray, indices_to_use, confidence: float = 0.9):
    keypoints_px = np.zeros((29, 2), dtype=np.float32)
    confidences = np.zeros((29,), dtype=np.float32)
    for idx in indices_to_use:
        keypoints_px[idx] = _synthetic_pixel(world_keypoints[idx])
        confidences[idx] = confidence
    return keypoints_px, confidences


# A reasonable spread of 8 keypoints a broadcast-wide shot might show
BROADCAST_INDICES = [0, 9, 16, 25, 11, 12, 13, 15]


def test_successful_calibration():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    info = calibrator.calibrate_frame(keypoints_px, confidences)

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert info.num_keypoints_used == len(BROADCAST_INDICES)
    assert info.reprojection_error_px_mean < 1.0  # synthetic data is exact, should be near-zero
    assert info.homography is not None


def test_pixel_to_world_recovers_known_point():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    world_kps = cfg.world_keypoints()

    keypoints_px, confidences = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    calibrator.calibrate_frame(keypoints_px, confidences)

    field_center_world = world_kps[15]  # (52.5, 34.0)
    field_center_px = _synthetic_pixel(field_center_world)

    result = calibrator.pixel_to_world(field_center_px)
    assert result is not None
    assert result.x == pytest.approx(field_center_world[0], abs=0.5)
    assert result.y == pytest.approx(field_center_world[1], abs=0.5)


def test_insufficient_points_not_calibrated():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    keypoints_px = np.zeros((29, 2), dtype=np.float32)
    confidences = np.zeros((29,), dtype=np.float32)
    confidences[0] = 0.9
    confidences[1] = 0.9  # only 2, need >= 4 by default

    info = calibrator.calibrate_frame(keypoints_px, confidences)
    assert info.status == CalibrationStatus.NOT_CALIBRATED
    assert info.homography is None


def test_pixel_to_world_returns_none_when_not_calibrated():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    keypoints_px = np.zeros((29, 2), dtype=np.float32)
    confidences = np.zeros((29,), dtype=np.float32)
    calibrator.calibrate_frame(keypoints_px, confidences)

    assert calibrator.pixel_to_world((500, 500)) is None


def test_calibration_does_not_persist_across_failed_frame():
    """Explicitly locks in the 'no silent staleness' design decision: a
    good frame followed by a bad frame must NOT let pixel_to_world keep
    using the old homography."""
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    world_kps = cfg.world_keypoints()

    good_px, good_conf = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    info_good = calibrator.calibrate_frame(good_px, good_conf)
    assert info_good.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert calibrator.pixel_to_world((500, 500)) is not None

    bad_px = np.zeros((29, 2), dtype=np.float32)
    bad_conf = np.zeros((29,), dtype=np.float32)
    info_bad = calibrator.calibrate_frame(bad_px, bad_conf)
    assert info_bad.status == CalibrationStatus.NOT_CALIBRATED
    assert calibrator.pixel_to_world((500, 500)) is None  # must not reuse the old homography


def test_is_world_point_plausible_catches_the_known_bad_value():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    bad_point = WorldPoint(x=114.61, y=46.79)  # the exact value from earlier debugging
    assert calibrator.is_world_point_plausible(bad_point) is False

    good_point = WorldPoint(x=52.5, y=34.0)
    assert calibrator.is_world_point_plausible(good_point) is True


def test_project_pitch_outline_none_when_uncalibrated():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    assert calibrator.project_pitch_outline() is None


def test_project_pitch_outline_shape_when_calibrated():
    cfg = PitchConfig()
    calibrator = PitchCalibrator(cfg)
    world_kps = cfg.world_keypoints()
    keypoints_px, confidences = _build_synthetic_frame(world_kps, BROADCAST_INDICES)
    calibrator.calibrate_frame(keypoints_px, confidences)

    segments = calibrator.project_pitch_outline()
    assert segments is not None
    assert segments.shape == (7, 2, 2)  # 4 boundary + halfway + 2 box lines