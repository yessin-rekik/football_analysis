"""
Wraps a PitchCalibrator and applies exponential moving average (EMA)
smoothing to the homography on successful calibrations, to reduce
frame-to-frame jitter caused by small keypoint-detection noise even when
the camera hasn't actually moved.

Exposes the SAME interface as PitchCalibrator (`calibrate_frame`,
`pixel_to_world`) plus a `reset()` method, so it's a drop-in replacement
anywhere a PitchCalibrator is expected -- including as
PropagatingCalibrator's `pitch_calibrator` argument, with zero changes
needed there.

Scope: only smooths CALIBRATED_FROM_KEYPOINTS results. A failed
calibration is passed through unchanged and clears the smoothing state --
this class follows the same "no silent staleness" rule as PitchCalibrator
itself (see that module's docstring).
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .calibrator import PitchCalibrator
from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import CalibrationInfo, WorldPoint


class SmoothedPitchCalibrator:
    def __init__(self, pitch_calibrator: PitchCalibrator, alpha: float = 0.3):
        """
        alpha: EMA weight given to the NEW homography each frame, in
            (0, 1]. Lower = smoother/slower to react to real camera
            motion, higher = snappier/less smoothing. alpha=1.0 disables
            smoothing entirely (always uses the latest raw estimate).
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self.pitch_calibrator = pitch_calibrator
        self.alpha = alpha
        self._smoothed_homography: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clears smoothing state. Call this after a gap (e.g. a
        PropagatingCalibrator re-anchor) so a fresh keypoint-based
        estimate doesn't get blended with a stale pre-gap value."""
        self._smoothed_homography = None

    def calibrate_frame(self, keypoints_px: np.ndarray, confidences: np.ndarray) -> CalibrationInfo:
        info = self.pitch_calibrator.calibrate_frame(keypoints_px, confidences)

        if info.status != CalibrationStatus.CALIBRATED_FROM_KEYPOINTS:
            self._smoothed_homography = None
            return info

        raw_H = np.array(info.homography, dtype=np.float64)

        if self._smoothed_homography is None:
            self._smoothed_homography = raw_H
        else:
            # Homographies are only defined up to scale -- normalize both
            # by their [2,2] element before blending, otherwise element-
            # wise EMA is meaningless (comparing arbitrary scale factors).
            prev = self._smoothed_homography / self._smoothed_homography[2, 2]
            curr = raw_H / raw_H[2, 2]
            self._smoothed_homography = self.alpha * curr + (1 - self.alpha) * prev

        mean_err, max_err = self._reprojection_error(
            self._smoothed_homography, keypoints_px, confidences
        )

        return CalibrationInfo(
            status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
            homography=self._smoothed_homography.tolist(),
            reprojection_error_px_mean=mean_err,
            reprojection_error_px_max=max_err,
            num_keypoints_used=info.num_keypoints_used,
            frames_since_last_anchor=info.frames_since_last_anchor,
        )

    def _reprojection_error(
        self, H: np.ndarray, keypoints_px: np.ndarray, confidences: np.ndarray
    ) -> Tuple[float, float]:
        """Re-measures error against the SMOOTHED homography, not the raw
        one -- the smoothed matrix is what's actually used downstream, so
        the reported diagnostic should reflect that, not the pre-smoothing
        fit. Duplicates a little logic from PitchCalibrator.calibrate_frame
        rather than modifying that class to expose it -- a reasonable
        trade-off for now; worth factoring into a shared helper if a third
        caller ever needs the same computation."""
        visible_indices = np.where(confidences >= self.pitch_calibrator.confidence_threshold)[0]
        src_pts = keypoints_px[visible_indices].astype(np.float32)
        dst_pts = self.pitch_calibrator.world_keypoints[visible_indices].astype(np.float32)
        H_inv = np.linalg.inv(H)
        reprojected = cv2.perspectiveTransform(
            dst_pts.reshape(-1, 1, 2), H_inv.astype(np.float32)
        ).reshape(-1, 2)
        errors = np.linalg.norm(reprojected - src_pts, axis=1)
        return float(errors.mean()), float(errors.max())

    def pixel_to_world(self, pixel_xy: Tuple[float, float]) -> Optional[WorldPoint]:
        if self._smoothed_homography is None:
            return None
        pt = np.array(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
        world = cv2.perspectiveTransform(pt, self._smoothed_homography.astype(np.float32))[0, 0]
        return WorldPoint(x=float(world[0]), y=float(world[1]))