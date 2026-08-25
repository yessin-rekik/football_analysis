"""
Camera motion tracking: estimates how the camera itself moved between two
consecutive frames, using generic visual features -- NOT pitch keypoints.

This is what makes calibration survivable through a zoom/pan where the
keypoint model finds too few pitch markings to calibrate directly (e.g. a
tight shot on a duel). It knows nothing about the pitch; it only answers
"how did the whole frame shift/scale/rotate since last frame."

Players moving through the frame are treated as outliers by RANSAC inside
findHomography -- the majority of tracked points (grass texture, crowd,
advertising boards, pitch lines) move consistently with camera motion,
players don't, so this is the same trick video stabilization software
relies on.
"""

from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class BaseMotionTracker(ABC):
    @abstractmethod
    def estimate_motion(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns a 3x3 homography H such that, approximately,
        pixel_curr ~= H @ pixel_prev (homogeneous), describing camera
        motion from the previous call's frame to this one. Returns None on
        the first call (no previous frame yet) or when motion can't be
        reliably estimated (e.g. too few trackable points, a hard cut)."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clears internal state. Must be called after a re-anchor (fresh
        keypoint-based calibration), so stale tracked points from before a
        long propagated stretch don't bias the next incremental estimate."""
        raise NotImplementedError


class CameraMotionTracker(BaseMotionTracker):
    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 8,
        min_tracked_points: int = 15,
        ransac_reproj_threshold: float = 3.0,
        max_lk_error: float = 4.0,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_tracked_points = min_tracked_points
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_lk_error = max_lk_error

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_points = None

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners,
            qualityLevel=self.quality_level, minDistance=self.min_distance,
        )

    def estimate_motion(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        if (self._prev_gray is None or self._prev_points is None
                or len(self._prev_points) < self.min_tracked_points):
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return None

        curr_points, status, lk_error = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None
        )

        homography = None
        if curr_points is not None and status is not None:
            # status==1 only means "the solver converged," not "this is a
            # genuine match" -- on two unrelated frames, LK will happily
            # converge to nonsense correspondences. lk_error (the residual
            # patch difference at the converged location) is what actually
            # separates real tracks from garbage: empirically ~0.5 for a
            # genuine match vs ~7+ for unrelated frames, so filtering on
            # both together is what makes a hard cut correctly return None
            # instead of a homography fit to noise.
            status_mask = status.reshape(-1).astype(bool)
            error_mask = lk_error.reshape(-1) < self.max_lk_error
            combined_mask = status_mask & error_mask

            prev_valid = self._prev_points.reshape(-1, 2)[combined_mask]
            curr_valid = curr_points.reshape(-1, 2)[combined_mask]

            if len(prev_valid) >= self.min_tracked_points:
                homography, _mask = cv2.findHomography(
                    prev_valid, curr_valid, cv2.RANSAC, self.ransac_reproj_threshold
                )

        # Always refresh tracked points from the current frame for next
        # call, rather than letting the same points decay across many
        # calls -- keeps feature quality from degrading over a long
        # propagated stretch.
        self._prev_gray = gray
        self._prev_points = self._detect_features(gray)

        return homography