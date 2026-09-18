"""
Scene classification: decides which keypoint model should handle a given
frame (Phase 1a routing).

Built as an abstract interface + one heuristic implementation on purpose.
The heuristic below works off keypoint output from whichever model ran
first (currently: the only model that exists, the broadcast one) --
it's a stand-in until real low-angle training data exists to either
validate this heuristic or replace it with a learned classifier. Nothing
that calls `classify()` should need to change when that swap happens.

--- Real-footage bug fix: area-product coverage was the wrong metric ---

The original "is this frame usable, spatially" check computed a single
`coverage_fraction = (bbox_width * bbox_height) / frame_area` -- the AREA
of the box enclosing the visible keypoints, as a fraction of the whole
frame's area -- and rejected the frame as CLOSE_UP if that fraction fell
below a threshold.

This has a real structural blind spot for landscape broadcast footage:
6+ confidently-detected keypoints spread across nearly the FULL WIDTH of
the frame, but confined to a narrow horizontal band vertically (a
realistic pattern -- sustained play near one end of the pitch, where
box/touchline markings sit in a horizontal strip), produces a tiny AREA
product even though the points are perfectly fine for a homography fit.
Confirmed against real match footage: 6 visible points, every frame,
coverage_fraction consistently ~0.055-0.06 -- comfortably below the old
0.15 threshold, misclassifying every single frame as CLOSE_UP and never
producing a single calibration anchor for the whole session.

The fix: track width_fraction and height_fraction SEPARATELY, and reject
as CLOSE_UP only if BOTH axes are individually too small (a genuine tight
cluster, small in every direction) rather than requiring their PRODUCT to
clear one combined bar. A homography needs real spread in both axes to
avoid being ill-conditioned, but "spread in both axes" is a much weaker
requirement than "a large bounding AREA" -- the old metric was
accidentally testing something stricter than what's actually needed.
Mathematically: for two fractions with a fixed product p, the larger of
the two is minimized (at sqrt(p)) when they're equal, and only grows from
there -- so a real elongated-but-valid spread will always have failed the
old area-product check harder than either axis individually would fail a
per-axis check, for the same real data.

`min_broadcast_fallback_fraction` (used later, in the BROADCAST_WIDE
fallback rule) gets the identical treatment for the identical reason --
it was a second, stricter area-product threshold (0.3) gating a
previously completely UNTESTED branch (see tests/test_scene_classifier.py
for the new coverage of it).
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
      2. Confident keypoints exist, but their spread is too small on
         BOTH the width and height axes individually (see module
         docstring for why this replaced a single area-product check)
         -> also CLOSE_UP: whatever model you use, there isn't a usable
         pitch template in view. If EITHER axis clears its own
         threshold, this rule does not reject the frame, even if the
         other axis is narrow.
      3. No center-pitch features visible (halfway line / center circle)
         but box and/or boundary features are -> LOW_ANGLE_CORNER. This is
         the corner-kick/touchline signature: the camera is close to one
         end of the pitch, so center-pitch markings are out of frame.
      4. Center-pitch features visible, or boundary features visible with
         a wide spread on at least one axis -> BROADCAST_WIDE.
      5. Anything else -> UNKNOWN (caller should fall back to the default
         broadcast model rather than guess).

    This is a placeholder tuned by inspection, not validated against a
    labeled dataset -- treat the thresholds as a starting point to
    calibrate once you have real low-angle footage to check it against.
    (The width/height-fraction thresholds specifically HAVE now been
    checked against real match footage -- see module docstring -- but
    remain inspection-tuned defaults, not the product of a labeled
    validation set.)
    """

    def __init__(
        self,
        min_points_for_calibration: int = 4,
        min_width_fraction: float = 0.2,
        min_height_fraction: float = 0.2,
        min_broadcast_fallback_fraction: float = 0.3,
    ):
        """
        min_width_fraction / min_height_fraction: a frame is rejected as
            CLOSE_UP (rule 2 above) only if the visible keypoints' spread
            falls below BOTH of these, each as a fraction of the frame's
            width/height respectively. Defaulted equally (0.2/0.2) for
            now, but kept as two independent parameters rather than one
            combined value -- broadcast footage's two pixel axes don't
            have to behave symmetrically (e.g. touchline-driven
            horizontal spread vs. halfway-line-driven vertical spread),
            and a future retune might reasonably want to treat them
            differently.
        min_broadcast_fallback_fraction: separate, higher threshold used
            only by rule 4's fallback (no center-pitch features, but
            enough boundary features spread widely on at least one axis
            -> still BROADCAST_WIDE rather than falling through to
            UNKNOWN). Kept as a single value (not split by axis) since,
            unlike rule 2, this rule already explicitly checks "at least
            one axis" via the `or` below -- there's no case where
            splitting it further would currently change behavior.
        """
        self.min_points_for_calibration = min_points_for_calibration
        self.min_width_fraction = min_width_fraction
        self.min_height_fraction = min_height_fraction
        self.min_broadcast_fallback_fraction = min_broadcast_fallback_fraction

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
        width_fraction = bbox_w / float(frame_w)
        height_fraction = bbox_h / float(frame_h)
        # Kept as a derived diagnostic value only -- no longer used to
        # gate any decision below (see module docstring for why the
        # area-product metric was replaced). Still surfaced in `details`
        # so existing logging/tooling that reads this key keeps working.
        coverage_fraction = width_fraction * height_fraction

        if width_fraction < self.min_width_fraction and height_fraction < self.min_height_fraction:
            return SceneClassification(
                scene_type=SceneType.CLOSE_UP,
                confidence=0.7,
                details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                         "width_fraction": width_fraction, "height_fraction": height_fraction,
                         "reason": "keypoints tightly clustered on BOTH axes -- no usable pitch template in view"},
            )

        center_visible = len(visible_indices & CENTER_ELEMENT_INDICES) > 0
        boundary_visible_count = len(visible_indices & FIELD_BOUNDARY_INDICES)
        box_visible_count = len(visible_indices & BOX_INDICES)

        if not center_visible and box_visible_count >= 2 and boundary_visible_count >= 1:
            return SceneClassification(
                scene_type=SceneType.LOW_ANGLE_CORNER,
                confidence=0.6,
                details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                         "width_fraction": width_fraction, "height_fraction": height_fraction,
                         "reason": "box/boundary features visible, no center-pitch features"},
            )

        wide_single_axis_spread = (
            width_fraction > self.min_broadcast_fallback_fraction
            or height_fraction > self.min_broadcast_fallback_fraction
        )
        if center_visible or (boundary_visible_count >= 2 and wide_single_axis_spread):
            return SceneClassification(
                scene_type=SceneType.BROADCAST_WIDE,
                confidence=0.7,
                details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                         "width_fraction": width_fraction, "height_fraction": height_fraction},
            )

        return SceneClassification(
            scene_type=SceneType.UNKNOWN,
            confidence=0.3,
            details={"num_visible": num_visible, "coverage_fraction": coverage_fraction,
                     "width_fraction": width_fraction, "height_fraction": height_fraction,
                     "reason": "pattern doesn't clearly match a known scene type"},
        )