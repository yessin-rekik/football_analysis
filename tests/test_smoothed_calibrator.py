import numpy as np
import pytest

from football_analysis.config import PitchConfig
from football_analysis.schemas.enums import CalibrationStatus
from football_analysis.calibration.calibrator import PitchCalibrator
from football_analysis.calibration.smoothed_calibrator import SmoothedPitchCalibrator
from football_analysis.calibration.propagating_calibrator import PropagatingCalibrator


FRAME_SHAPE = (480, 640, 3)
BROADCAST_INDICES = [0, 9, 16, 25, 11, 12, 13, 15]


def _synthetic_pixel(world_xy, scale=10.0, offset=(100.0, 50.0)):
    x, y = world_xy
    return (x * scale + offset[0], y * scale + offset[1])


def _build_frame(world_keypoints, indices, scale=10.0, offset=(100.0, 50.0), confidence=0.9):
    kps = np.zeros((29, 2), dtype=np.float32)
    confs = np.zeros((29,), dtype=np.float32)
    for idx in indices:
        kps[idx] = _synthetic_pixel(world_keypoints[idx], scale=scale, offset=offset)
        confs[idx] = confidence
    return kps, confs


class FakeMotionTracker:
    def __init__(self, homographies):
        self._homographies = list(homographies)
        self.reset_count = 0

    def estimate_motion(self, frame):
        return self._homographies.pop(0)

    def reset(self):
        self.reset_count += 1


# ---------------------------------------------------------------------------
# Standalone behavior
# ---------------------------------------------------------------------------

def test_alpha_validation():
    cfg = PitchConfig()
    base = PitchCalibrator(cfg)
    with pytest.raises(ValueError):
        SmoothedPitchCalibrator(base, alpha=0.0)
    with pytest.raises(ValueError):
        SmoothedPitchCalibrator(base, alpha=1.5)
    SmoothedPitchCalibrator(base, alpha=1.0)  # upper bound is inclusive, should not raise


def test_first_calibration_seeds_without_blending():
    cfg = PitchConfig()
    world_kps = cfg.world_keypoints()
    smoother = SmoothedPitchCalibrator(PitchCalibrator(cfg), alpha=0.3)

    kps, confs = _build_frame(world_kps, BROADCAST_INDICES)
    info = smoother.calibrate_frame(kps, confs)

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    raw_calibrator = PitchCalibrator(cfg)
    raw_info = raw_calibrator.calibrate_frame(kps, confs)
    assert np.allclose(np.array(info.homography), np.array(raw_info.homography), atol=1e-3)


def test_failed_calibration_clears_smoothing_state():
    cfg = PitchConfig()
    world_kps = cfg.world_keypoints()
    smoother = SmoothedPitchCalibrator(PitchCalibrator(cfg), alpha=0.3)

    kps, confs = _build_frame(world_kps, BROADCAST_INDICES)
    smoother.calibrate_frame(kps, confs)
    assert smoother.pixel_to_world((500, 500)) is not None

    bad_kps = np.zeros((29, 2), dtype=np.float32)
    bad_confs = np.zeros((29,), dtype=np.float32)
    info_bad = smoother.calibrate_frame(bad_kps, bad_confs)

    assert info_bad.status == CalibrationStatus.NOT_CALIBRATED
    assert smoother.pixel_to_world((500, 500)) is None  # no stale smoothed estimate survives


def test_reset_clears_state_directly():
    cfg = PitchConfig()
    world_kps = cfg.world_keypoints()
    smoother = SmoothedPitchCalibrator(PitchCalibrator(cfg), alpha=0.3)

    kps, confs = _build_frame(world_kps, BROADCAST_INDICES)
    smoother.calibrate_frame(kps, confs)
    assert smoother.pixel_to_world((500, 500)) is not None

    smoother.reset()
    assert smoother.pixel_to_world((500, 500)) is None


def test_blends_toward_new_estimate_gradually():
    """Two consecutive frames from slightly different (but both internally
    consistent) camera positions -- the smoothed result after frame 2
    should sit BETWEEN the two raw estimates, not jump straight to the
    new one, proving the blend is actually happening."""
    cfg = PitchConfig()
    world_kps = cfg.world_keypoints()
    smoother = SmoothedPitchCalibrator(PitchCalibrator(cfg), alpha=0.5)

    kps_a, confs_a = _build_frame(world_kps, BROADCAST_INDICES, offset=(100.0, 50.0))
    kps_b, confs_b = _build_frame(world_kps, BROADCAST_INDICES, offset=(110.0, 50.0))  # slight pan

    smoother.calibrate_frame(kps_a, confs_a)
    world_before = smoother.pixel_to_world((500, 500))

    smoother.calibrate_frame(kps_b, confs_b)
    world_after_smoothed = smoother.pixel_to_world((500, 500))

    # Compute what frame B alone (no smoothing) would say, for comparison
    raw_b = PitchCalibrator(cfg)
    raw_b.calibrate_frame(kps_b, confs_b)
    world_after_raw = raw_b.pixel_to_world((500, 500))

    # The smoothed result should differ from the pure-raw-B result (i.e.
    # smoothing is actually doing something), while still being in the
    # same ballpark (not wildly off).
    assert world_after_smoothed.x != pytest.approx(world_after_raw.x, abs=1e-6)
    assert abs(world_after_smoothed.x - world_after_raw.x) < 5.0


# ---------------------------------------------------------------------------
# Integration: composed inside PropagatingCalibrator across a real gap
# ---------------------------------------------------------------------------

def test_reset_before_reattempt_prevents_blending_across_a_gap():
    """The scenario the reset-ordering fix exists for: camera pans
    significantly during a propagated gap. When keypoints return, the
    fresh estimate must NOT get blended with the stale pre-gap smoothed
    homography -- it should re-anchor cleanly to the new position."""
    cfg = PitchConfig()
    world_kps = cfg.world_keypoints()

    raw_calibrator = PitchCalibrator(cfg)
    smoother = SmoothedPitchCalibrator(raw_calibrator, alpha=0.3)
    fake_tracker = FakeMotionTracker([None, np.eye(3), np.eye(3)])
    prop = PropagatingCalibrator(smoother, fake_tracker)

    # Frame 1: original camera position
    kps_before, confs_before = _build_frame(world_kps, BROADCAST_INDICES, offset=(100.0, 50.0))
    info1 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), kps_before, confs_before)
    assert info1.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS

    # Frame 2: gap (e.g. a duel close-up) -- propagate with identity motion
    info2 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)
    assert info2.status == CalibrationStatus.PROPAGATED

    # Frame 3: keypoints return, but the camera has actually panned a lot
    # during the gap (motion tracker couldn't see this -- e.g. it was a
    # hard cut in coverage, not a smooth pan) -- a real, different position
    kps_after, confs_after = _build_frame(world_kps, BROADCAST_INDICES, offset=(400.0, 50.0))
    info3 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), kps_after, confs_after)
    assert info3.status == CalibrationStatus.RE_ANCHORED

    # What SHOULD happen: re-anchoring uses the fresh estimate directly,
    # matching an unsmoothed calibration on the same frame 3 data.
    expected_raw = PitchCalibrator(cfg)
    expected_info = expected_raw.calibrate_frame(kps_after, confs_after)

    assert np.allclose(
        np.array(info3.homography), np.array(expected_info.homography), atol=1e-2
    ), "re-anchored homography should match the fresh fit exactly, not a blend with the stale pre-gap estimate"