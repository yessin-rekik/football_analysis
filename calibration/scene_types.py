"""
Scene type vocabulary used to route frames to the right keypoint model
(Phase 1a). Kept as its own module because it's referenced by the
classifier, the model registry, and eventually the API layer -- all of
them should import this rather than each defining their own strings.
"""

from enum import Enum


class SceneType(str, Enum):
    """
    BROADCAST_WIDE   -- standard wide broadcast angle; most/all of the
                        pitch markings the model was originally trained on
                        are visible at a normal perspective.
    LOW_ANGLE_CORNER -- corner kicks, throw-ins near the touchline, or any
                        shot where the camera is close to pitch level --
                        this is the case that currently fails.
    CLOSE_UP          -- tight shot on 1-2 players (e.g. a duel), where few
                        or no pitch keypoints are visible at all regardless
                        of angle. This is NOT a calibration-model problem --
                        no keypoint model can calibrate from markings that
                        simply aren't in frame. This case is handled by
                        Phase 1b (camera-motion propagation), not routing.
    UNKNOWN           -- not enough signal to classify confidently. Callers
                        should fall back to the default/broadcast model
                        rather than guessing.
    """
    BROADCAST_WIDE = "broadcast_wide"
    LOW_ANGLE_CORNER = "low_angle_corner"
    CLOSE_UP = "close_up"
    UNKNOWN = "unknown"