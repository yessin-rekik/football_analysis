from enum import Enum


class CalibrationStatus(str, Enum):
    """State of the pitch homography for a given frame (Phase 1).

    NOT_CALIBRATED         -- no usable homography this frame (world
                               positions in this frame should not be trusted)
    CALIBRATED_FROM_KEYPOINTS -- freshly computed from >=4 visible pitch
                               keypoints this frame (highest confidence)
    PROPAGATED              -- no/insufficient keypoints this frame; homography
                               was derived by composing camera-motion
                               tracking with the last known-good homography
                               (see Phase 1b -- zoom/occlusion handling)
    RE_ANCHORED              -- pitch keypoints became visible again after a
                               PROPAGATED streak and the homography was
                               recomputed from scratch, resetting drift
    """
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATED_FROM_KEYPOINTS = "calibrated_from_keypoints"
    PROPAGATED = "propagated"
    RE_ANCHORED = "re_anchored"


class ObjectClass(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    BALL = "ball"


class Team(str, Enum):
    HOME = "home"
    AWAY = "away"


class PositionProvenance(str, Enum):
    """How a tracked object's position for this frame was obtained
    (Phase 2 -- re-identification / long-gap handling).

    OBSERVED               -- normal continuous detection + tracking
    INTERPOLATED            -- filled in during a short tracking gap (e.g.
                               straight-line or velocity-seeded estimate
                               between a last-known point and a
                               re-identified reappearance)
    RE_IDENTIFIED_AFTER_GAP -- this specific frame is the reappearance point
                               where a track was matched back to a prior
                               track after a gap (position itself is
                               observed, but the identity link is inferred)
    """
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    RE_IDENTIFIED_AFTER_GAP = "re_identified_after_gap"
