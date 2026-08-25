"""
Bridges gaps where PitchCalibrator can't calibrate from keypoints -- either
a short blip or a sustained camera zoom during a duel -- by composing
camera-motion tracking with the last known-good homography. This is the
class a real video loop should actually drive; PitchCalibrator alone is
correct but stateless, and will report NOT_CALIBRATED the instant
keypoints disappear, which is by design for that class but not what you
want frame-to-frame in a real video.

State machine:
    CALIBRATED_FROM_KEYPOINTS  -- keypoints worked this frame
            |  (keypoints fail/skipped)
            v
    PROPAGATED  -- motion-tracked from the last good homography
            |  (keypoints work again)
            v
    RE_ANCHORED  -- keypoints worked again after a PROPAGATED streak
            |
            v  (next frame, if keypoints keep working)
    CALIBRATED_FROM_KEYPOINTS

Homography composition: if H_incremental maps pixel_prev -> pixel_curr
(what CameraMotionTracker returns), then a point in the current frame maps
back to the previous frame via its inverse, and from there to world via
the last known world-homography. So:

    H_world_new = H_world_prev @ inverse(H_incremental)
"""

from typing import Optional, Tuple

import numpy as np

from .calibrator import PitchCalibrator
from .camera_motion_tracker import BaseMotionTracker, CameraMotionTracker
from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import CalibrationInfo, WorldPoint


class PropagatingCalibrator:
    def __init__(
        self,
        pitch_calibrator: PitchCalibrator,
        motion_tracker: Optional[BaseMotionTracker] = None,
        max_propagated_frames: Optional[int] = None,
    ):
        """
        max_propagated_frames: if set, stop trusting a purely
            motion-propagated homography after this many consecutive
            frames without a real re-anchor -- caps how far drift can
            accumulate. None means no limit (propagate indefinitely as
            long as motion estimation keeps succeeding).
        """
        self.pitch_calibrator = pitch_calibrator
        self.motion_tracker = motion_tracker or CameraMotionTracker()
        self.max_propagated_frames = max_propagated_frames

        self._current_homography: Optional[np.ndarray] = None
        self._frames_since_last_anchor: Optional[int] = None
        self._status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED

    def process_frame(
        self,
        frame: np.ndarray,
        keypoints_px: Optional[np.ndarray],
        keypoint_confidences: Optional[np.ndarray],
    ) -> CalibrationInfo:
        """
        Pass keypoints_px=None (and keypoint_confidences=None) for frames
        the scene classifier already flagged CLOSE_UP -- skips wasting a
        keypoint-calibration attempt and goes straight to motion
        propagation, same reasoning as the pipeline skipping a second
        model call for those frames.
        """
        # Feed the motion tracker every frame regardless of keypoint
        # outcome -- motion is a property of consecutive frames,
        # independent of whether calibration succeeds.
        incremental_H = self.motion_tracker.estimate_motion(frame)

        keypoint_result = None
        if keypoints_px is not None and keypoint_confidences is not None:
            keypoint_result = self.pitch_calibrator.calibrate_frame(keypoints_px, keypoint_confidences)

        if keypoint_result is not None and keypoint_result.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS:
            return self._handle_keypoint_success(keypoint_result)

        return self._handle_keypoint_failure(incremental_H)

    def _handle_keypoint_success(self, keypoint_result: CalibrationInfo) -> CalibrationInfo:
        was_propagated = self._status == CalibrationStatus.PROPAGATED
        self._current_homography = np.array(keypoint_result.homography, dtype=np.float64)
        self._frames_since_last_anchor = 0
        # Don't carry stale tracked feature points across a re-anchor --
        # they were tracking whatever was on screen before, which may be
        # irrelevant now.
        self.motion_tracker.reset()
        self._status = CalibrationStatus.RE_ANCHORED if was_propagated else CalibrationStatus.CALIBRATED_FROM_KEYPOINTS

        return CalibrationInfo(
            status=self._status,
            homography=keypoint_result.homography,
            reprojection_error_px_mean=keypoint_result.reprojection_error_px_mean,
            reprojection_error_px_max=keypoint_result.reprojection_error_px_max,
            num_keypoints_used=keypoint_result.num_keypoints_used,
            frames_since_last_anchor=0,
        )

    def _handle_keypoint_failure(self, incremental_H: Optional[np.ndarray]) -> CalibrationInfo:
        if self._current_homography is None or incremental_H is None:
            # Nothing to propagate from (never anchored yet, or motion
            # estimation itself failed -- e.g. a hard cut).
            self._current_homography = None
            self._frames_since_last_anchor = None
            self._status = CalibrationStatus.NOT_CALIBRATED
            return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)

        try:
            inv_incremental = np.linalg.inv(incremental_H)
        except np.linalg.LinAlgError:
            self._current_homography = None
            self._frames_since_last_anchor = None
            self._status = CalibrationStatus.NOT_CALIBRATED
            return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)

        new_homography = self._current_homography @ inv_incremental
        self._frames_since_last_anchor = (self._frames_since_last_anchor or 0) + 1

        if (self.max_propagated_frames is not None
                and self._frames_since_last_anchor > self.max_propagated_frames):
            # Drift risk too high to keep trusting a pure motion chain.
            self._current_homography = None
            self._frames_since_last_anchor = None
            self._status = CalibrationStatus.NOT_CALIBRATED
            return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)

        self._current_homography = new_homography
        self._status = CalibrationStatus.PROPAGATED
        return CalibrationInfo(
            status=CalibrationStatus.PROPAGATED,
            homography=new_homography.tolist(),
            frames_since_last_anchor=self._frames_since_last_anchor,
        )

    def pixel_to_world(self, pixel_xy: Tuple[float, float]) -> Optional[WorldPoint]:
        if self._current_homography is None:
            return None
        import cv2
        pt = np.array(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
        world = cv2.perspectiveTransform(pt, self._current_homography.astype(np.float32))[0, 0]
        return WorldPoint(x=float(world[0]), y=float(world[1]))