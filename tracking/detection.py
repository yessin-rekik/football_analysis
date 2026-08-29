"""
Raw per-frame detection schema -- the output of a detector, BEFORE any
tracker has assigned identity.

This is deliberately NOT `TrackedObject` (schemas/frame_result.py).
`TrackedObject` requires a `track_id` and is the canonical, cross-stage
contract; `Detection` is an internal, single-frame concept scoped to
`tracking/` only -- a detector answers "what's in this frame," a tracker
(built on top of a sequence of these) answers "which one is the same
player as last frame." Keeping them separate schemas, not one with
optional fields, means a detector can never accidentally be handed a
track_id it has no business assigning, and the tracker's job is fully
captured by its function signature: List[Detection] -> List[TrackedObject].

Reuses BoundingBox / PixelPoint / ObjectClass from `schemas/` rather than
redefining them here -- there's exactly one definition of "what a bounding
box is" in this codebase.
"""

from pydantic import BaseModel, Field

from ..schemas.frame_result import BoundingBox, PixelPoint
from ..schemas.enums import ObjectClass


class Detection(BaseModel):
    """A single detected object in a single frame, with no identity yet."""

    object_class: ObjectClass
    confidence: float = Field(ge=0, le=1, description="Detector's confidence for this box/class.")

    bounding_box: BoundingBox

    pixel_position: PixelPoint = Field(
        description="Reference point in image pixels used for tracking/calibration "
                    "-- foot/ground contact point (bottom-center of the bounding box) "
                    "for players/goalkeepers/referees, bbox center for the ball. "
                    "Matches the convention TrackedObject.pixel_position already uses, "
                    "since this value flows into it unchanged once a track_id is assigned."
    )

    @staticmethod
    def foot_point(bbox: BoundingBox) -> PixelPoint:
        """Bottom-center of a bounding box -- the standard ground-contact
        reference point for homography projection, used for players,
        goalkeepers, and referees. A `staticmethod` here (not a free
        function) because it's a detail of how a `Detection`'s
        `pixel_position` is conventionally derived, so it stays discoverable
        next to the field it feeds; detector implementations call this
        rather than each re-deriving bottom-center math independently."""
        return PixelPoint(x=(bbox.x1 + bbox.x2) / 2.0, y=bbox.y2)

    @staticmethod
    def center_point(bbox: BoundingBox) -> PixelPoint:
        """Geometric center of a bounding box -- used for the ball instead
        of `foot_point`, since the ball isn't a ground-contact object the
        way a player is (it's frequently mid-air, and its bbox bottom edge
        doesn't mean anything physically meaningful the way a foot does)."""
        return PixelPoint(x=(bbox.x1 + bbox.x2) / 2.0, y=(bbox.y1 + bbox.y2) / 2.0)