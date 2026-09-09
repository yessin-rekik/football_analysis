"""
Phase 3 -- pixel -> world coordinate transform.

Pure function layer: turns a frame's pixel-space `TrackedObject`s into
world-position-enriched ones, given that frame's `CalibrationInfo`. Pure
function is a deliberate design constraint (project plan, Phase 3): this
module imports ONLY from `schemas` and `config` -- never anything from
`calibration/` or `tracking/`. It doesn't need a calibrator or orchestrator
instance; it only needs `CalibrationInfo.homography`, which the calibration
layer already serializes to a plain pixel->world 3x3 list-of-lists. That's
what keeps this layer swappable/testable independent of whatever produced
the homography.

Confidence policy (confirmed design decisions):
  - CALIBRATED_FROM_KEYPOINTS -- full trust, position_confidence = 1.0.
  - RE_ANCHORED -- slightly discounted (`re_anchored_confidence`, default
    0.85) even though the homography is freshly recomputed from keypoints
    exactly like CALIBRATED_FROM_KEYPOINTS -- the frame immediately after a
    propagated gap is where an identity link (re-identification) is most
    likely to still be shaky, so downstream stats should treat it with a
    little more caution than an ordinary anchored frame.
  - PROPAGATED -- decays with frames_since_last_anchor:
    `max(propagated_confidence_floor, 1.0 - propagated_decay_per_frame *
    frames_since_last_anchor)`. Longer since the last real anchor -> more
    accumulated drift risk -> lower confidence, floored so a long-but-
    otherwise-healthy propagation isn't automatically driven to zero
    purely by elapsed time.
  - NOT_CALIBRATED (or, defensively, a CalibrationInfo with no homography
    at all despite a different status) -- world_position and
    position_confidence both None. No projection is attempted.

Out-of-bounds handling: a projected point that fails
`PitchConfig.in_bounds()` is NOT dropped -- world_position is still
populated with the (implausible) projected value, but position_confidence
is forced to 0.0. Keeping the point visible rather than nulling it out
means it still shows up for debugging/overlay purposes (e.g. spotting a
homography that's producing garbage), while the confidence signal tells
any downstream stats consumer not to trust it.
"""

from typing import List, Tuple

import cv2
import numpy as np

from ..config.pitch_config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import CalibrationInfo, TrackedObject, WorldPoint


def _project_point(pixel_xy: Tuple[float, float], homography: np.ndarray) -> WorldPoint:
    pt = np.array([[list(pixel_xy)]], dtype=np.float64)
    world = cv2.perspectiveTransform(pt, homography)[0, 0]
    return WorldPoint(x=float(world[0]), y=float(world[1]))


def transform_tracked_objects(
    tracked_objects: List[TrackedObject],
    calibration: CalibrationInfo,
    pitch_config: PitchConfig,
    *,
    re_anchored_confidence: float = 0.85,
    propagated_confidence_floor: float = 0.3,
    propagated_decay_per_frame: float = 0.02,
    out_of_bounds_margin: float = 5.0,
) -> List[TrackedObject]:
    """
    Returns a NEW list of TrackedObjects with world_position and
    position_confidence populated -- never mutates the input objects, kept
    pure so callers can still use the pre-transform (pixel-space) list
    elsewhere, e.g. for an overlay, without aliasing surprises.
    """
    if calibration.status == CalibrationStatus.NOT_CALIBRATED or calibration.homography is None:
        return [
            obj.model_copy(update={"world_position": None, "position_confidence": None})
            for obj in tracked_objects
        ]

    homography = np.asarray(calibration.homography, dtype=np.float64)

    if calibration.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS:
        base_confidence = 1.0
    elif calibration.status == CalibrationStatus.RE_ANCHORED:
        base_confidence = re_anchored_confidence
    else:  # PROPAGATED
        frames = calibration.frames_since_last_anchor or 0
        base_confidence = max(
            propagated_confidence_floor,
            1.0 - propagated_decay_per_frame * frames,
        )

    results = []
    for obj in tracked_objects:
        world_point = _project_point(obj.pixel_position.as_tuple(), homography)
        plausible = pitch_config.in_bounds(world_point.as_tuple(), margin=out_of_bounds_margin)
        position_confidence = base_confidence if plausible else 0.0

        results.append(
            obj.model_copy(update={
                "world_position": world_point,
                "position_confidence": position_confidence,
            })
        )

    return results