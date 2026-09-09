import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus, ObjectClass
from ..schemas.frame_result import CalibrationInfo, PixelPoint, TrackedObject
from ..coordinates.transform import transform_tracked_objects


def _pixel_for_world(world_xy, scale=10.0, offset=(100.0, 50.0)):
    """Same synthetic mapping convention as test_calibrator.py (10 px per
    meter, offset origin) -- lets us build a known-correct homography by
    hand rather than depending on PitchCalibrator to produce one."""
    x, y = world_xy
    return (x * scale + offset[0], y * scale + offset[1])


def _synthetic_homography(pitch_config: PitchConfig, scale=10.0, offset=(100.0, 50.0)):
    """pixel -> world homography for the mapping above, i.e. the inverse
    of _pixel_for_world: world = (pixel - offset) / scale."""
    return [
        [1.0 / scale, 0.0, -offset[0] / scale],
        [0.0, 1.0 / scale, -offset[1] / scale],
        [0.0, 0.0, 1.0],
    ]


def _object(track_id=1, pixel_xy=(100.0, 50.0), object_class=ObjectClass.PLAYER):
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        pixel_position=PixelPoint(x=pixel_xy[0], y=pixel_xy[1]),
    )


def test_not_calibrated_yields_no_world_position():
    cfg = PitchConfig()
    calibration = CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED)
    objs = [_object()]

    result = transform_tracked_objects(objs, calibration, cfg)

    assert result[0].world_position is None
    assert result[0].position_confidence is None


def test_calibrated_from_keypoints_recovers_known_point_and_full_confidence():
    cfg = PitchConfig()
    world_target = (52.5, 34.0)  # field center
    pixel_xy = _pixel_for_world(world_target)
    homography = _synthetic_homography(cfg)

    calibration = CalibrationInfo(
        status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
        homography=homography,
        num_keypoints_used=8,
        frames_since_last_anchor=0,
    )
    objs = [_object(pixel_xy=pixel_xy)]

    result = transform_tracked_objects(objs, calibration, cfg)

    assert result[0].world_position is not None
    assert result[0].world_position.x == pytest.approx(world_target[0], abs=0.01)
    assert result[0].world_position.y == pytest.approx(world_target[1], abs=0.01)
    assert result[0].position_confidence == pytest.approx(1.0)


def test_re_anchored_is_discounted_but_not_full_trust():
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    calibration = CalibrationInfo(
        status=CalibrationStatus.RE_ANCHORED,
        homography=homography,
        frames_since_last_anchor=0,
    )
    objs = [_object(pixel_xy=_pixel_for_world((52.5, 34.0)))]

    result = transform_tracked_objects(objs, calibration, cfg)

    assert 0.0 < result[0].position_confidence < 1.0
    assert result[0].position_confidence == pytest.approx(0.85)


def test_propagated_confidence_decays_with_frames_since_anchor():
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    pixel_xy = _pixel_for_world((52.5, 34.0))

    calibration_fresh = CalibrationInfo(
        status=CalibrationStatus.PROPAGATED,
        homography=homography,
        frames_since_last_anchor=1,
    )
    calibration_stale = CalibrationInfo(
        status=CalibrationStatus.PROPAGATED,
        homography=homography,
        frames_since_last_anchor=20,
    )

    result_fresh = transform_tracked_objects([_object(pixel_xy=pixel_xy)], calibration_fresh, cfg)
    result_stale = transform_tracked_objects([_object(pixel_xy=pixel_xy)], calibration_stale, cfg)

    assert result_fresh[0].position_confidence == pytest.approx(1.0 - 0.02 * 1)
    assert result_stale[0].position_confidence == pytest.approx(1.0 - 0.02 * 20)
    assert result_stale[0].position_confidence < result_fresh[0].position_confidence


def test_propagated_confidence_is_floored_not_zeroed():
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    pixel_xy = _pixel_for_world((52.5, 34.0))

    calibration = CalibrationInfo(
        status=CalibrationStatus.PROPAGATED,
        homography=homography,
        frames_since_last_anchor=1000,  # would go deeply negative without a floor
    )

    result = transform_tracked_objects([_object(pixel_xy=pixel_xy)], calibration, cfg)

    assert result[0].position_confidence == pytest.approx(0.3)  # default floor


def test_out_of_bounds_projection_keeps_point_but_zeroes_confidence():
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    # Pixel that maps to a world point far outside the pitch + margin
    wildly_off_pixel = _pixel_for_world((500.0, 500.0))

    calibration = CalibrationInfo(
        status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
        homography=homography,
        frames_since_last_anchor=0,
    )
    objs = [_object(pixel_xy=wildly_off_pixel)]

    result = transform_tracked_objects(objs, calibration, cfg)

    assert result[0].world_position is not None  # NOT dropped
    assert result[0].world_position.x == pytest.approx(500.0, abs=0.01)
    assert result[0].position_confidence == 0.0


def test_does_not_mutate_input_objects():
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    calibration = CalibrationInfo(
        status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
        homography=homography,
        frames_since_last_anchor=0,
    )
    original = _object(pixel_xy=_pixel_for_world((52.5, 34.0)))
    objs = [original]

    transform_tracked_objects(objs, calibration, cfg)

    assert original.world_position is None  # input untouched
    assert original.position_confidence is None


def test_preserves_object_identity_fields():
    """Transform must be additive only -- track_id, class, team, etc. must
    pass through unchanged."""
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    calibration = CalibrationInfo(
        status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
        homography=homography,
        frames_since_last_anchor=0,
    )
    obj = _object(track_id=42, pixel_xy=_pixel_for_world((10.0, 10.0)), object_class=ObjectClass.BALL)

    result = transform_tracked_objects([obj], calibration, cfg)

    assert result[0].track_id == 42
    assert result[0].object_class == ObjectClass.BALL


def test_multiple_objects_in_one_frame():
    cfg = PitchConfig()
    homography = _synthetic_homography(cfg)
    calibration = CalibrationInfo(
        status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
        homography=homography,
        frames_since_last_anchor=0,
    )
    objs = [
        _object(track_id=1, pixel_xy=_pixel_for_world((10.0, 10.0))),
        _object(track_id=2, pixel_xy=_pixel_for_world((90.0, 60.0))),
    ]

    result = transform_tracked_objects(objs, calibration, cfg)

    assert len(result) == 2
    assert result[0].world_position.x == pytest.approx(10.0, abs=0.01)
    assert result[1].world_position.x == pytest.approx(90.0, abs=0.01)