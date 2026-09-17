import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus, ObjectClass, Team
from ..schemas.frame_result import CalibrationInfo, FrameResult, PixelPoint, TrackedObject, WorldPoint
from ..stats.radar import DEFAULT_AWAY_COLOR, DEFAULT_HOME_COLOR, DEFAULT_PITCH_COLOR, world_to_pixel
from ..stats.voronoi import compute_ownership_grid, render_voronoi_frame

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


# ---- compute_ownership_grid ----

def test_point_at_a_players_own_position_is_owned_by_that_players_team():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 34.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (95.0, 34.0)),
    ])
    ownership = compute_ownership_grid(frame, cfg, PPM, MARGIN_M)

    home_px = world_to_pixel((10.0, 34.0), cfg, PPM, MARGIN_M)
    away_px = world_to_pixel((95.0, 34.0), cfg, PPM, MARGIN_M)
    assert ownership[home_px[1], home_px[0]] == 0  # home index
    assert ownership[away_px[1], away_px[0]] == 1  # away index


def test_only_one_team_present_claims_the_entire_grid():
    cfg = PitchConfig()
    frame = _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (52.5, 34.0))])
    ownership = compute_ownership_grid(frame, cfg, PPM, MARGIN_M)
    assert np.all(ownership == 0)


def test_nearest_of_multiple_same_team_players_wins():
    """Two home players, one away player -- a point near the SECOND home
    player must still resolve to home (index 0) purely because home is
    nearest, not because of ordering within the team's own list."""
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (5.0, 5.0)),
        _obj(2, ObjectClass.PLAYER, Team.HOME, (100.0, 60.0)),
        _obj(3, ObjectClass.PLAYER, Team.AWAY, (52.5, 34.0)),
    ])
    ownership = compute_ownership_grid(frame, cfg, PPM, MARGIN_M)
    near_second_home_px = world_to_pixel((99.0, 59.0), cfg, PPM, MARGIN_M)
    assert ownership[near_second_home_px[1], near_second_home_px[0]] == 0


def test_no_eligible_players_returns_degenerate_zero_grid():
    """Documents the degenerate case explicitly: with nobody to own
    anything, compute_ownership_grid returns an all-zero grid rather than
    raising -- it's render_voronoi_frame's job to recognize this case and
    skip drawing a fill from it, not this function's job to signal it any
    other way."""
    cfg = PitchConfig()
    empty_frame = _frame(0, [])
    ownership = compute_ownership_grid(empty_frame, cfg, PPM, MARGIN_M)
    width_expected = int(round((cfg.pitch_length + 2 * MARGIN_M) * PPM))
    height_expected = int(round((cfg.pitch_width + 2 * MARGIN_M) * PPM))
    assert ownership.shape == (height_expected, width_expected)
    assert np.all(ownership == 0)


# ---- render_voronoi_frame ----

def test_render_returns_correct_shape_and_dtype():
    cfg = PitchConfig()
    canvas = render_voronoi_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    assert canvas.dtype == np.uint8
    assert canvas.ndim == 3 and canvas.shape[2] == 3


def test_empty_frames_sequence_renders_bare_pitch_with_no_tint():
    cfg = PitchConfig()
    canvas = render_voronoi_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    open_grass_point = (25.0, 15.0)  # clear of all drawn lines, see test_radar.py's same convention
    px = world_to_pixel(open_grass_point, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_PITCH_COLOR


def test_last_frame_with_zero_eligible_players_renders_bare_pitch():
    """Non-empty frames sequence, but the LAST frame has nobody eligible
    -- must still skip the fill entirely (degenerate ownership grid),
    same outcome as an empty sequence."""
    cfg = PitchConfig()
    frames = [
        _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (52.5, 34.0))]),
        _frame(1, []),  # last frame: nobody eligible
    ]
    canvas = render_voronoi_frame(frames, cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    open_grass_point = (25.0, 15.0)
    px = world_to_pixel(open_grass_point, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_PITCH_COLOR


def test_only_the_last_frame_is_used():
    """Two frames with different player positions -- the diagram must
    reflect only the second (last) frame, not an average or the first."""
    cfg = PitchConfig()
    frames = [
        _frame(0, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 34.0)),
            _obj(2, ObjectClass.PLAYER, Team.AWAY, (95.0, 34.0)),
        ]),
        _frame(1, [
            # Roles swapped: now HOME is on the right, AWAY on the left.
            _obj(1, ObjectClass.PLAYER, Team.AWAY, (10.0, 34.0)),
            _obj(2, ObjectClass.PLAYER, Team.HOME, (95.0, 34.0)),
        ]),
    ]
    canvas = render_voronoi_frame(
        frames, cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, fill_alpha=1.0, draw_players=False,
    )
    left_px = world_to_pixel((10.0, 34.0), cfg, PPM, MARGIN_M)
    right_px = world_to_pixel((95.0, 34.0), cfg, PPM, MARGIN_M)
    # If frame 0 had been used, left would be HOME-colored -- it must not be.
    assert tuple(canvas[left_px[1], left_px[0]]) == DEFAULT_AWAY_COLOR
    assert tuple(canvas[right_px[1], right_px[0]]) == DEFAULT_HOME_COLOR


def test_fill_alpha_zero_leaves_pitch_color_untouched():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 34.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (95.0, 34.0)),
    ])
    canvas = render_voronoi_frame(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, fill_alpha=0.0, draw_players=False,
    )
    open_home_side_point = (20.0, 50.0)  # clearly on home's side, off any drawn line
    px = world_to_pixel(open_home_side_point, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_PITCH_COLOR


def test_fill_alpha_one_is_a_solid_team_color_with_no_blending():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 34.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (95.0, 34.0)),
    ])
    canvas = render_voronoi_frame(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, fill_alpha=1.0, draw_players=False,
    )
    open_home_side_point = (20.0, 50.0)
    px = world_to_pixel(open_home_side_point, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_HOME_COLOR


def test_partial_fill_alpha_blends_toward_pitch_color():
    cfg = PitchConfig()
    alpha = 0.35
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 34.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (95.0, 34.0)),
    ])
    canvas = render_voronoi_frame(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, fill_alpha=alpha, draw_players=False,
    )
    open_home_side_point = (20.0, 50.0)
    px = world_to_pixel(open_home_side_point, cfg, PPM, MARGIN_M)
    actual = canvas[px[1], px[0]].astype(float)
    expected = (
        alpha * np.array(DEFAULT_HOME_COLOR, dtype=float)
        + (1.0 - alpha) * np.array(DEFAULT_PITCH_COLOR, dtype=float)
    )
    assert not np.array_equal(actual, np.array(DEFAULT_HOME_COLOR, dtype=float))
    assert np.allclose(actual, expected, atol=3.0)


def test_draw_players_true_adds_a_black_outline_draw_players_false_does_not():
    """Same style of check as test_radar.py's goalkeeper-ring test: scan a
    short band just outside the marker fill for a near-black outline
    pixel, rather than relying on one exact pixel matching a stroke."""
    cfg = PitchConfig()
    marker_radius = 6
    frame = _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (52.5, 34.0))])
    center_px = world_to_pixel((52.5, 34.0), cfg, PPM, MARGIN_M)

    def has_black_outline(canvas) -> bool:
        cx, cy = center_px
        for offset in range(marker_radius, marker_radius + 4):
            pixel = canvas[cy, cx + offset]
            if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < 60:
                return True
        return False

    canvas_with_players = render_voronoi_frame(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M,
        draw_players=True, player_marker_radius_px=marker_radius,
    )
    canvas_without_players = render_voronoi_frame(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M,
        draw_players=False, player_marker_radius_px=marker_radius,
    )
    assert has_black_outline(canvas_with_players) is True
    assert has_black_outline(canvas_without_players) is False