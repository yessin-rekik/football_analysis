"""
Scene classification: decides which keypoint model should handle a given
frame (Phase 1a routing).

Built as an abstract interface + one heuristic implementation on purpose.
The heuristic below works off keypoint output from whichever model ran
first (currently: the only model that exists, the broadcast one) --
it's a stand-in until real low-angle training data exists to either
validate this heuristic or replace it with a learned classifier. Nothing
that calls `classify()` should need to change when that swap happens.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from pydantic import BaseModel

from .scene_types import SceneType

# Re-declared here rather than imported from the detector module to avoid a
# circular import (the detector will import the classifier, not vice versa).
FIELD_BOUNDARY_INDICES = {0, 9, 16, 25}
CENTER_ELEMENT_INDICES = {11, 12, 13, 14, 15, 27, 28}
BOX_INDICES = {1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24}


class SceneClassification(BaseModel):
    scene_type: SceneType
    confidence: float
    details: dict = {}


class BaseSceneClassifier(ABC):
    """Interface every scene classifier implementation must satisfy.
    `frame` is accepted even though the heuristic implementation below
    ignores it, so a future frame-appearance-based (or learned) classifier
    is a drop-in replacement."""

    @abstractmethod
    def classify(
        self,
        frame: Optional[np.ndarray],
        keypoints_px: Optional[np.ndarray],
        keypoint_confidences: Optional[np.ndarray],
        confidence_threshold: float = 0.5,
    ) -> SceneClassification:
        raise NotImplementedError


class HeuristicSceneClassifier(BaseSceneClassifier):
    """
    First-pass classifier based purely on the spatial pattern of keypoints
    already detected by an upstream model -- no learning, no extra
    inference cost. Rules, in order:

      1. Fewer than `min_points_for_calibration` confident keypoints
         -> CLOSE_UP. Too little signal to calibrate from keypoints at
         all, regardless of which model produced them -- this is a
         Phase 1b (camera-motion propagation) case, not a routing case.
      2. Confident keypoints exist but are tightly clustered in a small
         fraction of the frame -> also CLOSE_UP (same reasoning: whatever
         model you use, there isn't a usable pitch template in view).
      3. No center-pitch features visible (halfway line / center circle)
         but box and/or boundary features are -> LOW_ANGLE_CORNER. This is
         the corner-kick/touchline signature: the camera is close to one
         end of the pitch, so center-pitch markings are out of frame.
      4. Center-pitch features visible, or boundary features visible with
         reasonable frame coverage -> BROADCAST_WIDE.
      5. Anything else -> UNKNOWN (caller should fall back to the default
         broadcast model rather than guess).

    This is a placeholder tuned by inspection, not validated against a
    labeled dataset -- treat the thresholds as a starting point to
    calibrate once you have real low-angle footage to check it against.
    """

    def __init__(self, min_points_for_calibration: int = 4, min_coverage_fraction: float = 0.15):
        self.min_points_for_calibration = min_points_for_calibration
        self.min_coverage_fraction = min_coverage_fraction

    def classify(
        self,
        frame: Optional[np.ndarray],
        keypoints_px: Optional[np.ndarray],
        keypoint_confidences: Optional[np.ndarray],
        confidence_threshold: float = 0.5,
    ) -> SceneClassification:
        if keypoints_px is None or keypoint_confidences is None or len(keypoints_px) == 0:
            return SceneClassification(
                scene_type=SceneType.CLOSE_UP,
                confidence=0.9,
                details={"reason": "no keypoint data available"},
            )

        visible_mask = keypoint_confidences >= confidence_threshold
        visible_indices = set(np.where(visible_mask)[0].tolist())
        num_visible = len(visible_indices)

        if num_visible < self.min_points_for_calibration:
            return SceneClassification(
                scene_type=SceneType.CLOSE_UP,
                confidence=min(0.95, 0.6 + 0.1 * (self.min_points_for_calibration - num_visible)),
                details={"num_visible": num_visible, "reason": "too few confident keypoints"},
            )

        visible_pts = keypoints_px[list(visible_indices)]
        frame_h, frame_w = (frame.shape[:2] if frame is not None
                             else (int(visible_pts[:, 1].max() * 1.1) + 1,
                                   int(visible_pts[:, 0].max() * 1.1) + 1))
        bbox_w = visible_pts[:, 0].max() - visible_pts[:, 0].min()
        bbox_h = visible_pts[:, 1].max() - visible_pts[:, 1].min()
        coverage_fraction = (bbox_w * bbox_h) / float(frame_w * frame_h)

        if coverage_fraction < self.min_coverage_fraction:
            return SceneClassification(
                scene_type=SceneType.CLOSE_UP,
                confidence=0.7,
                details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                         "reason": "keypoints tightly clustered -- no usable pitch template in view"},
            )

        center_visible = len(visible_indices & CENTER_ELEMENT_INDICES) > 0
        boundary_visible_count = len(visible_indices & FIELD_BOUNDARY_INDICES)
        box_visible_count = len(visible_indices & BOX_INDICES)

        if not center_visible and box_visible_count >= 2 and boundary_visible_count >= 1:
            return SceneClassification(
                scene_type=SceneType.LOW_ANGLE_CORNER,
                confidence=0.6,
                details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                         "reason": "box/boundary features visible, no center-pitch features"},
            )

        if center_visible or (boundary_visible_count >= 2 and coverage_fraction > 0.3):
            return SceneClassification(
                scene_type=SceneType.BROADCAST_WIDE,
                confidence=0.7,
                details={"num_visible": num_visible, "coverage_fraction": coverage_fraction},
            )

        return SceneClassification(
            scene_type=SceneType.UNKNOWN,
            confidence=0.3,
            details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                     "reason": "pattern doesn't clearly match a known scene type"},
        )