"""
Homography-based calibrator: turns a keypoint model's raw per-frame output
into a pixel <-> world mapping, using PitchConfig.world_keypoints() as the
real-world template (never hardcoded here).

Scope, on purpose: this class ONLY answers "can I calibrate from the
keypoints in this single frame, and if so, what's the homography." It is
stateless across frames -- if a frame fails to calibrate, the previous
frame's homography is discarded, not reused. Bridging short gaps (a
keypoint model briefly losing points) or long gaps (a camera zoom during a
duel) via camera-motion propagation is Phase 1b, implemented as a separate
class composed on top of this one. That split keeps "how do I compute a
homography from points" and "what do I do when points disappear" as two
independently testable pieces of logic.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..config.pitch_config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import CalibrationInfo, WorldPoint


class PitchCalibrator:
    def __init__(
        self,
        pitch_config: PitchConfig,
        min_points: int = 4,
        confidence_threshold: float = 0.5,
        ransac_reproj_threshold: float = 5.0,
    ):
        self.pitch_config = pitch_config
        self.world_keypoints = pitch_config.world_keypoints()  # (29, 2), meters
        self.min_points = min_points
        self.confidence_threshold = confidence_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold

        # Cached only so pixel_to_world()/project_pitch_outline() can be
        # called right after calibrate_frame() without re-passing the
        # matrix -- reset to None on every failed calibration, never
        # carried forward as a stale guess (see module docstring).
        self._homography: Optional[np.ndarray] = None

    def calibrate_frame(self, keypoints_px: np.ndarray, confidences: np.ndarray) -> CalibrationInfo:
        """
        keypoints_px: (29, 2) pixel coordinates, one row per keypoint slot
        confidences:  (29,) confidence per slot, in [0, 1]

        Returns a CalibrationInfo with status CALIBRATED_FROM_KEYPOINTS on
        success or NOT_CALIBRATED on failure -- never PROPAGATED or
        RE_ANCHORED, since those states only make sense to a caller that's
        tracking history across frames (Phase 1b).
        """
        self._homography = None  # reset every call; success re-sets it below

        visible_indices = np.where(confidences >= self.confidence_threshold)[0]

        if len(visible_indices) < self.min_points:
            return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)

        src_pts = keypoints_px[visible_indices].astype(np.float32)
        dst_pts = self.world_keypoints[visible_indices].astype(np.float32)

        H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold)
        if H is None:
            return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)

        reprojected = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        errors = np.linalg.norm(reprojected - src_pts, axis=1)

        self._homography = H

        return CalibrationInfo(
            status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
            homography=H.tolist(),
            reprojection_error_px_mean=float(errors.mean()),
            reprojection_error_px_max=float(errors.max()),
            num_keypoints_used=int(len(visible_indices)),
            frames_since_last_anchor=0,
        )

    def pixel_to_world(self, pixel_xy: Tuple[float, float]) -> Optional[WorldPoint]:
        """Returns None if the most recent calibrate_frame() call failed --
        callers must check for None rather than assuming a value."""
        if self._homography is None:
            return None
        pt = np.array(pixel_xy, dtype=np.float32).reshape(1, 1, 2)
        world = cv2.perspectiveTransform(pt, self._homography)[0, 0]
        return WorldPoint(x=float(world[0]), y=float(world[1]))

    def is_world_point_plausible(self, world_point: WorldPoint, margin: float = 5.0) -> bool:
        """Sanity check for a computed world point against the pitch
        bounds -- catches a technically-successful-but-wrong homography
        the same way the (114.61, 46.79) bug was caught during debugging."""
        return self.pitch_config.in_bounds((world_point.x, world_point.y), margin=margin)

    def project_pitch_outline(self) -> Optional[np.ndarray]:
        """Projects the pitch boundary, halfway line, and penalty-area
        edges into pixel space using the current homography, for visual
        sanity-checking (same idea as the debugging overlay built earlier).
        Returns an (N, 2, 2) array of pixel-space line segments, or None if
        not currently calibrated."""
        if self._homography is None:
            return None

        L, W = self.pitch_config.pitch_length, self.pitch_config.pitch_width
        pa_depth = self.pitch_config.penalty_area_depth
        segments_world = [
            [(0, 0), (L, 0)], [(L, 0), (L, W)], [(L, W), (0, W)], [(0, W), (0, 0)],
            [(L / 2, 0), (L / 2, W)],
            [(pa_depth, 0), (pa_depth, W)],
            [(L - pa_depth, 0), (L - pa_depth, W)],
        ]
        H_inv = np.linalg.inv(self._homography)
        segments_px = []
        for p1, p2 in segments_world:
            pts = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
            segments_px.append(proj)
        return np.array(segments_px)