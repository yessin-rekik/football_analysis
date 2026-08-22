import pytest

from football_analysis.config import PitchConfig
from football_analysis.schemas import (
    CalibrationStatus, ObjectClass, Team, PositionProvenance,
    PixelPoint, WorldPoint, CalibrationInfo, TrackedObject, FrameResult,
    MatchMetadata,
)


def _sample_frame() -> FrameResult:
    return FrameResult(
        frame_index=10,
        timestamp_s=0.4,
        calibration=CalibrationInfo(status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS),
        tracked_objects=[
            TrackedObject(
                track_id=1,
                object_class=ObjectClass.PLAYER,
                team=Team.HOME,
                pixel_position=PixelPoint(x=100.0, y=200.0),
                world_position=WorldPoint(x=10.0, y=20.0),
                provenance=PositionProvenance.OBSERVED,
            ),
            TrackedObject(
                track_id=2,
                object_class=ObjectClass.BALL,
                pixel_position=PixelPoint(x=105.0, y=205.0),
                world_position=None,  # e.g. ball detected but off-pitch projection rejected
                provenance=PositionProvenance.OBSERVED,
            ),
        ],
    )


def test_frame_result_round_trip():
    frame = _sample_frame()
    raw = frame.model_dump_json()
    reloaded = FrameResult.model_validate_json(raw)
    assert reloaded == frame


def test_world_position_optional():
    frame = _sample_frame()
    ball = [o for o in frame.tracked_objects if o.object_class == ObjectClass.BALL][0]
    assert ball.world_position is None  # must not require a world position


def test_to_flat_records_shape():
    frame = _sample_frame()
    records = frame.to_flat_records()
    assert len(records) == 2
    assert records[0]["world_x_m"] == 10.0
    assert records[1]["world_x_m"] is None
    assert records[1]["team"] is None


def test_not_calibrated_frame_has_no_world_positions_by_convention():
    """Not a hard schema constraint (kept flexible on purpose), but this
    documents the expected usage: NOT_CALIBRATED frames should not carry
    world_position values from calibration/tracking code."""
    frame = FrameResult(
        frame_index=5,
        timestamp_s=0.2,
        calibration=CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED),
        tracked_objects=[
            TrackedObject(
                track_id=1,
                object_class=ObjectClass.PLAYER,
                pixel_position=PixelPoint(x=1.0, y=1.0),
                world_position=None,
            ),
        ],
    )
    assert frame.tracked_objects[0].world_position is None


def test_match_metadata_embeds_pitch_config():
    meta = MatchMetadata(
        match_id="m1", video_fps=25.0, video_width=1920, video_height=1080,
        pitch_config=PitchConfig(),
    )
    assert meta.pitch_config.pitch_length == 105.0
    reloaded = MatchMetadata.model_validate_json(meta.model_dump_json())
    assert reloaded == meta
