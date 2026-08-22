"""
Canonical per-frame output schema.

This is the single data contract that decouples calibration/tracking
(Phases 1-2) from coordinate transform (Phase 3) from stats (Phase 4,
not yet built) from the API layer (Phase 5). Every stage after raw
detection should produce or consume THESE models and nothing
stage-specific -- a stats function should be written against
`FrameResult` and never need to know a YOLO model or a homography
matrix exists.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from .enums import CalibrationStatus, ObjectClass, Team, PositionProvenance


class PixelPoint(BaseModel):
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class WorldPoint(BaseModel):
    """Real-world pitch coordinate in meters, using the PitchConfig
    convention: x in [0, pitch_length], y in [0, pitch_width]."""
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class CalibrationInfo(BaseModel):
    """Calibration state for a single frame. One of these per FrameResult,
    shared by every tracked object in that frame."""

    status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED

    homography: Optional[List[List[float]]] = Field(
        default=None,
        description="3x3 homography matrix (pixel -> world), row-major, or "
                    "None if NOT_CALIBRATED this frame.",
    )

    reprojection_error_px_mean: Optional[float] = Field(
        default=None, description="Mean reprojection error in pixels, when computed from keypoints."
    )
    reprojection_error_px_max: Optional[float] = None

    num_keypoints_used: Optional[int] = Field(
        default=None, description="How many pitch keypoints fed this frame's homography, if CALIBRATED_FROM_KEYPOINTS."
    )

    frames_since_last_anchor: Optional[int] = Field(
        default=None,
        description="How many frames ago the homography was last computed "
                    "from real keypoints (0 for CALIBRATED_FROM_KEYPOINTS / "
                    "RE_ANCHORED). Growing values under PROPAGATED indicate "
                    "increasing drift risk -- downstream stats can use this "
                    "as a confidence signal or cutoff.",
    )


class TrackedObject(BaseModel):
    """A single detected/tracked entity (player, goalkeeper, referee, ball)
    in a single frame."""

    track_id: int = Field(description="Stable ID across frames for this object, from the tracker (Phase 2).")
    object_class: ObjectClass
    team: Optional[Team] = Field(default=None, description="None until team classification runs or if not applicable (e.g. ball, referee).")
    jersey_number: Optional[int] = None

    pixel_position: PixelPoint = Field(description="Reference point in image pixels -- foot/ground contact point for players, not bbox center.")
    bounding_box: Optional[BoundingBox] = None

    world_position: Optional[WorldPoint] = Field(
        default=None,
        description="None if this frame's calibration status is NOT_CALIBRATED.",
    )

    detection_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    position_confidence: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Confidence in world_position specifically -- should be "
                    "discounted when CalibrationInfo.status is PROPAGATED, "
                    "independent of detection_confidence.",
    )

    provenance: PositionProvenance = PositionProvenance.OBSERVED


class FrameResult(BaseModel):
    """Canonical output for one processed video frame. A full match's
    output is a List[FrameResult] (or a stream of these, for a
    job/streaming API)."""

    frame_index: int
    timestamp_s: float

    calibration: CalibrationInfo = CalibrationInfo()
    tracked_objects: List[TrackedObject] = []

    def to_flat_records(self) -> List[dict]:
        """Flatten to one dict per tracked object, merging in frame- and
        calibration-level context. This is the shape Phase 3's CSV/Parquet
        export and most pandas-based analysis will want -- defined here,
        next to the schema, so export code in Phase 3 doesn't need to
        re-derive the flattening logic."""
        records = []
        for obj in self.tracked_objects:
            records.append({
                "frame_index": self.frame_index,
                "timestamp_s": self.timestamp_s,
                "calibration_status": self.calibration.status.value,
                "frames_since_last_anchor": self.calibration.frames_since_last_anchor,
                "track_id": obj.track_id,
                "object_class": obj.object_class.value,
                "team": obj.team.value if obj.team else None,
                "jersey_number": obj.jersey_number,
                "pixel_x": obj.pixel_position.x,
                "pixel_y": obj.pixel_position.y,
                "world_x_m": obj.world_position.x if obj.world_position else None,
                "world_y_m": obj.world_position.y if obj.world_position else None,
                "detection_confidence": obj.detection_confidence,
                "position_confidence": obj.position_confidence,
                "provenance": obj.provenance.value,
            })
        return records
