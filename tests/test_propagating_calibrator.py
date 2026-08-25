import numpy as np
import pytest

from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import CalibrationInfo
from ..calibration.propagating_calibrator import PropagatingCalibrator


FRAME_SHAPE = (480, 640, 3)


class FakePitchCalibrator:
    """Scripted stand-in for PitchCalibrator -- returns a queued result per
    call rather than actually computing a homography, so the state machine
    can be tested independent of real keypoint geometry."""
    def __init__(self, results):
        self._results = list(results)
        self.call_count = 0

    def calibrate_frame(self, keypoints_px, confidences):
        self.call_count += 1
        return self._results.pop(0)


class FakeMotionTracker:
    """Scripted stand-in for CameraMotionTracker -- returns a queued
    incremental homography (or None) per call."""
    def __init__(self, incremental_homographies):
        self._homographies = list(incremental_homographies)
        self.reset_count = 0
        self.call_count = 0

    def estimate_motion(self, frame):
        self.call_count += 1
        return self._homographies.pop(0)

    def reset(self):
        self.reset_count += 1


IDENTITY = np.eye(3)


def _good_keypoint_result(homography=IDENTITY.tolist()):
    return CalibrationInfo(
        status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
        homography=homography,
        num_keypoints_used=8,
        frames_since_last_anchor=0,
    )


def _failed_keypoint_result():
    return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)


def test_first_frame_calibrated_from_keypoints():
    fake_calibrator = FakePitchCalibrator([_good_keypoint_result()])
    fake_tracker = FakeMotionTracker([None])  # first call always None (no prior frame)
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker)

    info = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8),
                               keypoints_px=np.zeros((29, 2)), keypoint_confidences=np.ones((29,)))

    assert info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert info.frames_since_last_anchor == 0


def test_propagates_through_gap_using_motion():
    identity_incremental = np.eye(3)  # camera didn't move
    fake_calibrator = FakePitchCalibrator([
        _good_keypoint_result(),   # frame 1: keypoints work
        _failed_keypoint_result(), # frame 2: keypoints gone (e.g. zoom)
        _failed_keypoint_result(), # frame 3: still gone
    ])
    fake_tracker = FakeMotionTracker([None, identity_incremental, identity_incremental])
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker)

    info1 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), np.zeros((29, 2)), np.ones((29,)))
    info2 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)
    info3 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)

    assert info1.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert info2.status == CalibrationStatus.PROPAGATED
    assert info2.frames_since_last_anchor == 1
    assert info3.status == CalibrationStatus.PROPAGATED
    assert info3.frames_since_last_anchor == 2

    # identity motion -> homography should be unchanged from the anchor
    assert np.allclose(np.array(info3.homography), IDENTITY)


def test_reanchors_after_keypoints_return():
    fake_calibrator = FakePitchCalibrator([
        _good_keypoint_result(),
        _failed_keypoint_result(),
        _good_keypoint_result(),  # keypoints come back
    ])
    fake_tracker = FakeMotionTracker([None, np.eye(3), None])
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker)

    prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), np.zeros((29, 2)), np.ones((29,)))
    # Frame 2: keypoints were ATTEMPTED but calibration failed (e.g. too
    # few confident points) -- pass real arrays, not None. None is
    # reserved for CLOSE_UP frames where the keypoint model is skipped
    # entirely; passing None here would skip calling calibrate_frame at
    # all, leaving this queued failure never consumed and desyncing the
    # queue against frame 3's call.
    prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), np.zeros((29, 2)), np.zeros((29,)))
    info3 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), np.zeros((29, 2)), np.ones((29,)))

    assert info3.status == CalibrationStatus.RE_ANCHORED
    assert info3.frames_since_last_anchor == 0
    assert fake_tracker.reset_count == 2  # once on the very first anchor, once on re-anchor


def test_no_motion_and_no_keypoints_is_not_calibrated():
    fake_calibrator = FakePitchCalibrator([_failed_keypoint_result()])
    fake_tracker = FakeMotionTracker([None])  # never anchored, no motion either
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker)

    info = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)
    assert info.status == CalibrationStatus.NOT_CALIBRATED


def test_motion_lost_mid_propagation_falls_back_to_not_calibrated():
    """Anchored, then keypoints disappear AND motion tracking itself fails
    (e.g. a hard cut) -- nothing left to propagate from, must not silently
    keep reporting a stale homography."""
    fake_calibrator = FakePitchCalibrator([_good_keypoint_result(), _failed_keypoint_result()])
    fake_tracker = FakeMotionTracker([None, None])  # motion estimation fails on frame 2
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker)

    prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), np.zeros((29, 2)), np.ones((29,)))
    info2 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)

    assert info2.status == CalibrationStatus.NOT_CALIBRATED
    assert prop.pixel_to_world((100, 100)) is None


def test_max_propagated_frames_cutoff():
    identity_incremental = np.eye(3)
    fake_calibrator = FakePitchCalibrator([
        _good_keypoint_result(),
        _failed_keypoint_result(),
        _failed_keypoint_result(),
        _failed_keypoint_result(),
    ])
    fake_tracker = FakeMotionTracker([None, identity_incremental, identity_incremental, identity_incremental])
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker, max_propagated_frames=2)

    prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), np.zeros((29, 2)), np.ones((29,)))
    info2 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)  # 1 frame propagated, OK
    info3 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)  # 2 frames, still OK (at limit)
    info4 = prop.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8), None, None)  # 3rd -- exceeds limit

    assert info2.status == CalibrationStatus.PROPAGATED
    assert info3.status == CalibrationStatus.PROPAGATED
    assert info4.status == CalibrationStatus.NOT_CALIBRATED


def test_pixel_to_world_none_before_any_calibration():
    fake_calibrator = FakePitchCalibrator([])
    fake_tracker = FakeMotionTracker([])
    prop = PropagatingCalibrator(fake_calibrator, fake_tracker)
    assert prop.pixel_to_world((10, 10)) is None