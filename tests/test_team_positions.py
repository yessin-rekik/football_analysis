import pytest

from ..schemas.enums import CalibrationStatus, ObjectClass, Team
from ..schemas.frame_result import CalibrationInfo, FrameResult, PixelPoint, TrackedObject, WorldPoint
from ..stats.team_positions import TEAM_POSITION_ELIGIBLE_CLASSES, group_positions_by_team


def _obj(track_id, object_class, team=None, world_xy=None) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        team=team,
        pixel_position=PixelPoint(x=0.0, y=0.0),
        world_position=WorldPoint(x=world_xy[0], y=world_xy[1]) if world_xy is not None else None,
    )


def _frame(objects) -> FrameResult:
    return FrameResult(
        frame_index=1,
        timestamp_s=0.04,
        calibration=CalibrationInfo(status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS),
        tracked_objects=objects,
    )


def test_separates_home_and_away():
    frame = _frame([
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 20.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (90.0, 60.0)),
    ])
    grouped = group_positions_by_team(frame)
    assert grouped[Team.HOME] == [(1, (10.0, 20.0))]
    assert grouped[Team.AWAY] == [(2, (90.0, 60.0))]


def test_goalkeeper_is_eligible_by_default():
    frame = _frame([_obj(1, ObjectClass.GOALKEEPER, Team.HOME, (5.0, 5.0))])
    grouped = group_positions_by_team(frame)
    assert grouped[Team.HOME] == [(1, (5.0, 5.0))]


def test_referee_and_ball_are_excluded_regardless_of_team():
    """Team is genuinely not applicable to these classes -- even if a
    caller somehow set .team on one (shouldn't happen upstream, but this
    function shouldn't trust it either), it must still be excluded."""
    frame = _frame([
        _obj(1, ObjectClass.REFEREE, Team.HOME, (50.0, 30.0)),
        _obj(2, ObjectClass.BALL, Team.AWAY, (52.5, 34.0)),
    ])
    grouped = group_positions_by_team(frame)
    assert grouped[Team.HOME] == []
    assert grouped[Team.AWAY] == []


def test_unassigned_team_is_excluded():
    """A player object_class is eligible, but team=None (team
    classification hasn't reached it yet) must still be excluded -- being
    the right class doesn't imply being assignable."""
    frame = _frame([_obj(1, ObjectClass.PLAYER, team=None, world_xy=(10.0, 20.0))])
    grouped = group_positions_by_team(frame)
    assert grouped[Team.HOME] == []
    assert grouped[Team.AWAY] == []


def test_missing_world_position_is_excluded():
    frame = _frame([_obj(1, ObjectClass.PLAYER, Team.HOME, world_xy=None)])
    grouped = group_positions_by_team(frame)
    assert grouped[Team.HOME] == []


def test_empty_frame_still_returns_both_team_keys():
    frame = _frame([])
    grouped = group_positions_by_team(frame)
    assert grouped == {Team.HOME: [], Team.AWAY: []}


def test_frame_with_only_one_team_present_still_returns_both_keys():
    """A caller must be able to do grouped[Team.AWAY] without a KeyError
    even when nobody on that team happened to be eligible this frame."""
    frame = _frame([_obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 20.0))])
    grouped = group_positions_by_team(frame)
    assert grouped[Team.HOME] == [(1, (10.0, 20.0))]
    assert grouped[Team.AWAY] == []


def test_object_classes_override_narrows_eligibility():
    frame = _frame([
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 20.0)),
        _obj(2, ObjectClass.GOALKEEPER, Team.HOME, (5.0, 5.0)),
    ])
    grouped = group_positions_by_team(frame, object_classes={ObjectClass.PLAYER})
    assert grouped[Team.HOME] == [(1, (10.0, 20.0))]  # goalkeeper excluded by the override


def test_default_eligible_classes_constant_matches_function_default_behavior():
    """Locks in the relationship between the exported constant and the
    function's own default -- if someone changes the constant, this test
    forces them to notice the function's behavior moved too."""
    frame = _frame([
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 20.0)),
        _obj(2, ObjectClass.GOALKEEPER, Team.HOME, (5.0, 5.0)),
        _obj(3, ObjectClass.REFEREE, Team.HOME, (50.0, 30.0)),
    ])
    default_result = group_positions_by_team(frame)
    explicit_result = group_positions_by_team(frame, object_classes=TEAM_POSITION_ELIGIBLE_CLASSES)
    assert default_result == explicit_result
    assert ObjectClass.REFEREE not in TEAM_POSITION_ELIGIBLE_CLASSES


def test_preserves_track_order_within_a_team():
    frame = _frame([
        _obj(5, ObjectClass.PLAYER, Team.HOME, (1.0, 1.0)),
        _obj(2, ObjectClass.PLAYER, Team.HOME, (2.0, 2.0)),
        _obj(9, ObjectClass.PLAYER, Team.HOME, (3.0, 3.0)),
    ])
    grouped = group_positions_by_team(frame)
    assert [track_id for track_id, _ in grouped[Team.HOME]] == [5, 2, 9]


def test_does_not_mutate_input_frame():
    frame = _frame([_obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 20.0))])
    original_tracked_objects = list(frame.tracked_objects)
    group_positions_by_team(frame)
    assert frame.tracked_objects == original_tracked_objects