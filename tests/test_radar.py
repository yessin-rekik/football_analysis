import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import ObjectClass, Team, PositionProvenance
from ..schemas.frame_result import PixelPoint, WorldPoint, TrackedObject
from ..stats.radar import (
    get_canvas_size,
    world_to_pixel,
    render_radar_frame,
    DEFAULT_PITCH_COLOR,
    DEFAULT_HOME_COLOR,
    DEFAULT_AWAY_COLOR,
    DEFAULT_NEUTRAL_COLOR,
    DEFAULT_BALL_COLOR,
)


PPM = 10.0
MARGIN_M = 3.0


def _obj(
    track_id: int,
    object_class: ObjectClass,
    world_xy,
    team=None,
    position_confidence=None,
) -> TrackedObject:
    """Builds a minimal TrackedObject for radar tests -- pixel_position is
    required by the schema but irrelevant here, since render_radar_frame
    only ever reads world_position."""
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        team=team,
        pixel_position=PixelPoint(x=0.0, y=0.0),
        world_position=WorldPoint(x=world_xy[0], y=world_xy[1]) if world_xy is not None else None,
        position_confidence=position_confidence,
        provenance=PositionProvenance.OBSERVED,
    )


def test_canvas_size_matches_pitch_config():
    cfg = PitchConfig()  # 105 x 68
    width, height = get_canvas_size(cfg, PPM, MARGIN_M)
    assert width == int(round((105.0 + 2 * MARGIN_M) * PPM))
    assert height == int(round((68.0 + 2 * MARGIN_M) * PPM))


def test_world_to_pixel_flips_y_and_applies_margin():
    cfg = PitchConfig()
    margin_px = int(round(MARGIN_M * PPM))

    # (0, W) -- top-left corner in world terms -- must land at the pixel
    # origin plus margin (top-left of the image), confirming the flip.
    assert world_to_pixel((0.0, cfg.pitch_width), cfg, PPM, MARGIN_M) == (margin_px, margin_px)

    # (0, 0) -- bottom-left corner in world terms -- must land at the
    # BOTTOM of the image (large pixel row), not the top.
    expected_bottom_py = margin_px + int(round(cfg.pitch_width * PPM))
    assert world_to_pixel((0.0, 0.0), cfg, PPM, MARGIN_M) == (margin_px, expected_bottom_py)


def test_render_returns_correct_shape_and_dtype():
    cfg = PitchConfig()
    canvas = render_radar_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    width, height = get_canvas_size(cfg, PPM, MARGIN_M)
    assert canvas.shape == (height, width, 3)
    assert canvas.dtype == np.uint8


def test_pitch_markings_are_drawn():
    """The rendered pitch must not be a flat, uninterrupted rectangle of
    pitch_color -- boundary/halfway/circle lines must actually be drawn."""
    cfg = PitchConfig()
    canvas = render_radar_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    background = np.full_like(canvas, DEFAULT_PITCH_COLOR)
    assert not np.array_equal(canvas, background)


def test_object_without_world_position_is_skipped():
    cfg = PitchConfig()
    empty_canvas = render_radar_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)

    obj_no_position = _obj(1, ObjectClass.PLAYER, world_xy=None, team=Team.HOME)
    canvas_with_skip = render_radar_frame(
        [obj_no_position], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M
    )

    # Nothing should have been drawn for an object with no world position --
    # output must be pixel-identical to rendering an empty object list.
    assert np.array_equal(empty_canvas, canvas_with_skip)


def test_home_and_away_players_get_distinct_full_opacity_colors():
    cfg = PitchConfig()
    home = _obj(1, ObjectClass.PLAYER, (30.0, 20.0), team=Team.HOME, position_confidence=1.0)
    away = _obj(2, ObjectClass.PLAYER, (70.0, 50.0), team=Team.AWAY, position_confidence=1.0)

    canvas = render_radar_frame([home, away], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)

    home_px = world_to_pixel((30.0, 20.0), cfg, PPM, MARGIN_M)
    away_px = world_to_pixel((70.0, 50.0), cfg, PPM, MARGIN_M)

    # At full opacity, the marker's center pixel is drawn with the exact
    # color -- no blending -- so this must match exactly, not approximately.
    assert tuple(canvas[home_px[1], home_px[0]]) == DEFAULT_HOME_COLOR
    assert tuple(canvas[away_px[1], away_px[0]]) == DEFAULT_AWAY_COLOR


def test_ball_gets_ball_color():
    cfg = PitchConfig()
    ball = _obj(1, ObjectClass.BALL, (52.5, 34.0), team=None, position_confidence=1.0)
    canvas = render_radar_frame([ball], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    px = world_to_pixel((52.5, 34.0), cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_BALL_COLOR


def test_referee_gets_neutral_color():
    cfg = PitchConfig()
    referee = _obj(1, ObjectClass.REFEREE, (52.5, 34.0), team=None, position_confidence=1.0)
    canvas = render_radar_frame([referee], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    px = world_to_pixel((52.5, 34.0), cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_NEUTRAL_COLOR


def test_goalkeeper_has_a_ring_outfield_player_does_not():
    """Goalkeepers are drawn with the same team fill color as outfield
    players, distinguished only by a black outer ring -- confirm the ring
    is actually there for the keeper and absent for a plain outfield
    player of the same team."""
    cfg = PitchConfig()
    player_radius = 7
    ring_thickness = 2

    keeper = _obj(1, ObjectClass.GOALKEEPER, (30.0, 20.0), team=Team.HOME, position_confidence=1.0)
    outfield = _obj(2, ObjectClass.PLAYER, (70.0, 20.0), team=Team.HOME, position_confidence=1.0)

    canvas = render_radar_frame(
        [keeper, outfield], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M,
        player_marker_radius_px=player_radius, goalkeeper_ring_thickness_px=ring_thickness,
    )

    keeper_center = world_to_pixel((30.0, 20.0), cfg, PPM, MARGIN_M)
    outfield_center = world_to_pixel((70.0, 20.0), cfg, PPM, MARGIN_M)

    def has_black_ring(center_px) -> bool:
        cx, cy = center_px
        # Scan just outside the fill radius, out to just past where the
        # ring should be -- robust to exact anti-aliasing/stroke-width
        # pixel placement, unlike checking one single pixel.
        for offset in range(player_radius + 1, player_radius + ring_thickness + 4):
            pixel = canvas[cy, cx + offset]
            if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < 60:  # near-black
                return True
        return False

    assert has_black_ring(keeper_center) is True
    assert has_black_ring(outfield_center) is False


def test_none_confidence_is_treated_as_full_opacity():
    cfg = PitchConfig()
    obj = _obj(1, ObjectClass.PLAYER, (30.0, 20.0), team=Team.HOME, position_confidence=None)
    canvas = render_radar_frame([obj], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    px = world_to_pixel((30.0, 20.0), cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_HOME_COLOR


def test_partial_confidence_blends_toward_background():
    cfg = PitchConfig()
    confidence = 0.5
    # Deliberately NOT the pitch center (52.5, 34.0) -- that's exactly
    # where the center-circle spot is drawn, so "background" there isn't
    # plain pitch_color. This point sits in open grass: past the penalty
    # area's depth, clear of the halfway line and center circle.
    open_grass_point = (25.0, 15.0)
    obj = _obj(1, ObjectClass.BALL, open_grass_point, team=None, position_confidence=confidence)
    canvas = render_radar_frame([obj], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    px = world_to_pixel(open_grass_point, cfg, PPM, MARGIN_M)

    actual = canvas[px[1], px[0]].astype(float)
    expected = (
        confidence * np.array(DEFAULT_BALL_COLOR, dtype=float)
        + (1.0 - confidence) * np.array(DEFAULT_PITCH_COLOR, dtype=float)
    )
    # A genuine blend, not the raw marker color -- and close to the
    # arithmetic blend up to normal 8-bit rounding.
    assert not np.array_equal(actual, np.array(DEFAULT_BALL_COLOR, dtype=float))
    assert np.allclose(actual, expected, atol=3.0)


def test_zero_confidence_still_renders_via_min_draw_alpha_floor():
    """The Phase 3 out-of-bounds policy forces position_confidence to
    exactly 0.0 while deliberately keeping world_position populated, so it
    stays visible for debugging. A naive confidence->alpha mapping would
    make this point invisible (alpha=0.0) -- min_draw_alpha must prevent
    that."""
    cfg = PitchConfig()
    floor = 0.2
    # Same reasoning as above: avoid the pitch center, where the drawn
    # center spot would contaminate the "plain background" comparison.
    open_grass_point = (25.0, 15.0)
    obj = _obj(1, ObjectClass.BALL, open_grass_point, team=None, position_confidence=0.0)

    canvas = render_radar_frame(
        [obj], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, min_draw_alpha=floor
    )
    empty_canvas = render_radar_frame([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)

    px = world_to_pixel(open_grass_point, cfg, PPM, MARGIN_M)
    with_marker = canvas[px[1], px[0]]
    without_marker = empty_canvas[px[1], px[0]]

    # Must differ from the plain background -- i.e. the marker is visible,
    # even though confidence was exactly 0.0.
    assert not np.array_equal(with_marker, without_marker)

    # And the blend actually used the floor value, not alpha=0 or alpha=1.
    expected = (
        floor * np.array(DEFAULT_BALL_COLOR, dtype=float)
        + (1.0 - floor) * np.array(DEFAULT_PITCH_COLOR, dtype=float)
    )
    assert np.allclose(with_marker.astype(float), expected, atol=3.0)


def test_multiple_objects_all_render_independently():
    cfg = PitchConfig()
    objects = [
        _obj(1, ObjectClass.PLAYER, (20.0, 10.0), team=Team.HOME, position_confidence=1.0),
        _obj(2, ObjectClass.PLAYER, (85.0, 60.0), team=Team.AWAY, position_confidence=1.0),
        _obj(3, ObjectClass.BALL, (52.5, 34.0), team=None, position_confidence=1.0),
    ]
    canvas = render_radar_frame(objects, cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)

    for obj, expected_color in zip(objects, [DEFAULT_HOME_COLOR, DEFAULT_AWAY_COLOR, DEFAULT_BALL_COLOR]):
        px = world_to_pixel(obj.world_position.as_tuple(), cfg, PPM, MARGIN_M)
        assert tuple(canvas[px[1], px[0]]) == expected_color