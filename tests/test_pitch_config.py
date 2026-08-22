import pytest

from config import PitchConfig, PitchKeypointName


def test_default_dimensions():
    cfg = PitchConfig()
    assert cfg.pitch_length == 105.0
    assert cfg.pitch_width == 68.0


def test_world_keypoints_shape_and_key_points():
    cfg = PitchConfig()
    kps = cfg.world_keypoints()
    assert kps.shape == (29, 2)

    assert tuple(kps[PitchKeypointName.FIELD_CENTER]) == pytest.approx((52.5, 34.0))
    assert tuple(kps[PitchKeypointName.SIDELINE_TOP_LEFT]) == pytest.approx((0.0, 68.0))
    assert tuple(kps[PitchKeypointName.SIDELINE_BOTTOM_RIGHT]) == pytest.approx((105.0, 0.0))


def test_world_keypoints_symmetry():
    """Left-side and right-side keypoints should mirror around the halfway
    line, given symmetric pitch markings -- catches sign errors."""
    cfg = PitchConfig()
    kps = cfg.world_keypoints()
    L = cfg.pitch_length

    left_x = kps[PitchKeypointName.BIG_RECT_LEFT_TOP_PT2][0]
    right_x = kps[PitchKeypointName.BIG_RECT_RIGHT_TOP_PT2][0]
    assert left_x == pytest.approx(L - right_x)

    left_y = kps[PitchKeypointName.BIG_RECT_LEFT_TOP_PT2][1]
    right_y = kps[PitchKeypointName.BIG_RECT_RIGHT_TOP_PT2][1]
    assert left_y == pytest.approx(right_y)


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        PitchConfig(penalty_area_width=200.0)  # wider than the pitch itself

    with pytest.raises(ValueError):
        PitchConfig(goal_area_width=50.0)  # wider than the penalty area

    with pytest.raises(ValueError):
        PitchConfig(pitch_length=-10.0)  # gt=0 constraint


def test_in_bounds():
    cfg = PitchConfig()
    assert cfg.in_bounds((50.0, 30.0)) is True
    assert cfg.in_bounds((-2.0, 30.0)) is True  # within margin
    assert cfg.in_bounds((114.61, 46.79)) is False  # the exact bad value from earlier debugging
