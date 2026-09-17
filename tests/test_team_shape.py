import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus, ObjectClass, Team
from ..schemas.frame_result import CalibrationInfo, FrameResult, PixelPoint, TrackedObject, WorldPoint
from ..stats.radar import DEFAULT_AWAY_COLOR, DEFAULT_HOME_COLOR, render_radar_frame, world_to_pixel
from ..stats.team_shape import compute_average_positions, render_team_shape

PPM = 10.0
MARGIN_M = 3.0


def _obj(track_id, object_class, team=None, world_xy=None) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        team=team,
        pixel_position=PixelPoint(x=0.0, y=0.0),
        world_position=WorldPoint(x=world_xy[0], y=world_xy[1]) if world_xy is not None else None,
    )


def _frame(index, objects) -> FrameResult:
    return FrameResult(
        frame_index=index,
        timestamp_s=index * 0.04,
        calibration=CalibrationInfo(status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS),
        tracked_objects=objects,
    )


# ---- compute_average_positions ----

def test_average_of_multiple_frames_is_the_true_mean():
    frames = [
        _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0))]),
        _frame(1, [_obj(1, ObjectClass.PLAYER, Team.HOME, (20.0, 10.0))]),
        _frame(2, [_obj(1, ObjectClass.PLAYER, Team.HOME, (30.0, 10.0))]),
    ]
    averages = compute_average_positions(frames)
    assert averages[Team.HOME][1] == (20.0, 10.0)


def test_track_present_in_only_some_frames_is_not_diluted_by_window_length():
    """A track seen in only 1 of 3 window frames must average to its own
    single value, not be divided by the full window length as if the
    missing frames contributed a position of zero."""
    frames = [
        _frame(0, [_obj(2, ObjectClass.PLAYER, Team.AWAY, (50.0, 50.0))]),
        _frame(1, []),  # track 2 not present
        _frame(2, []),  # track 2 not present
    ]
    averages = compute_average_positions(frames)
    assert averages[Team.AWAY][2] == (50.0, 50.0)


def test_tracks_grouped_by_team_independently():
    frames = [
        _frame(0, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
            _obj(2, ObjectClass.PLAYER, Team.AWAY, (90.0, 60.0)),
        ]),
    ]
    averages = compute_average_positions(frames)
    assert 1 in averages[Team.HOME]
    assert 1 not in averages[Team.AWAY]
    assert 2 in averages[Team.AWAY]
    assert 2 not in averages[Team.HOME]


def test_track_never_calibrated_is_absent_from_averages():
    """A track that exists in the tracked_objects list every frame but
    never has a world_position (e.g. the whole window is NOT_CALIBRATED)
    must not appear in the output at all -- there's nothing to average."""
    frames = [
        _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, world_xy=None)]),
        _frame(1, [_obj(1, ObjectClass.PLAYER, Team.HOME, world_xy=None)]),
    ]
    averages = compute_average_positions(frames)
    assert 1 not in averages[Team.HOME]


def test_empty_window_returns_empty_dicts_for_both_teams():
    averages = compute_average_positions([])
    assert averages == {Team.HOME: {}, Team.AWAY: {}}


def test_multiple_tracks_per_team_averaged_independently():
    frames = [
        _frame(0, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
            _obj(2, ObjectClass.PLAYER, Team.HOME, (90.0, 60.0)),
        ]),
        _frame(1, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
            _obj(2, ObjectClass.PLAYER, Team.HOME, (90.0, 60.0)),
        ]),
    ]
    averages = compute_average_positions(frames)
    assert averages[Team.HOME][1] == (10.0, 10.0)
    assert averages[Team.HOME][2] == (90.0, 60.0)


def test_does_not_mutate_input_frames():
    frames = [_frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0))])]
    snapshot = [f.model_copy(deep=True) for f in frames]
    compute_average_positions(frames)
    assert frames == snapshot


# ---- render_team_shape ----

def test_render_returns_correct_shape_and_dtype():
    cfg = PitchConfig()
    canvas = render_team_shape([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    assert canvas.dtype == np.uint8
    assert canvas.ndim == 3 and canvas.shape[2] == 3


def test_empty_window_matches_bare_radar_pitch():
    """An empty window should render the identical bare-pitch background
    render_radar_frame produces for an empty object list -- both share the
    same draw_pitch_markings() call, so this confirms the shared code path
    is actually being reused, not re-implemented with subtly different
    defaults."""
    cfg = PitchConfig()
    team_shape_canvas = render_team_shape([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    radar_canvas = render_radar_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    assert np.array_equal(team_shape_canvas, radar_canvas)


def test_markers_drawn_at_correct_averaged_pixel_position():
    cfg = PitchConfig()
    frames = [
        _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (20.0, 10.0))]),
        _frame(1, [_obj(1, ObjectClass.PLAYER, Team.HOME, (40.0, 10.0))]),
    ]
    canvas = render_team_shape(frames, cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)

    expected_avg_world = (30.0, 10.0)  # true mean of (20,10) and (40,10)
    px = world_to_pixel(expected_avg_world, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_HOME_COLOR


def test_home_and_away_markers_use_distinct_colors():
    cfg = PitchConfig()
    frames = [
        _frame(0, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (20.0, 10.0)),
            _obj(2, ObjectClass.PLAYER, Team.AWAY, (80.0, 55.0)),
        ]),
    ]
    canvas = render_team_shape(frames, cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)

    home_px = world_to_pixel((20.0, 10.0), cfg, PPM, MARGIN_M)
    away_px = world_to_pixel((80.0, 55.0), cfg, PPM, MARGIN_M)
    assert tuple(canvas[home_px[1], home_px[0]]) == DEFAULT_HOME_COLOR
    assert tuple(canvas[away_px[1], away_px[0]]) == DEFAULT_AWAY_COLOR