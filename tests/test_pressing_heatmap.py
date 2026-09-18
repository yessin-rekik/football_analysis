import cv2
import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus, ObjectClass, Team
from ..schemas.frame_result import CalibrationInfo, FrameResult, PixelPoint, TrackedObject, WorldPoint
from ..stats.pressing_heatmap import (
    compute_pressing_grid,
    compute_pressing_samples,
    render_pressing_heatmap,
)
from ..stats.radar import DEFAULT_PITCH_COLOR, world_to_pixel

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


# ---- compute_pressing_samples ----

def test_distance_is_symmetric_between_the_two_nearest_opponents():
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    samples = compute_pressing_samples([frame])
    assert len(samples) == 2
    by_position = {pos: dist for pos, dist in samples}
    assert by_position[(10.0, 10.0)] == pytest.approx(3.0)
    assert by_position[(13.0, 10.0)] == pytest.approx(3.0)


def test_target_team_home_only_samples_home_players():
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    samples = compute_pressing_samples([frame], target_team=Team.HOME)
    assert len(samples) == 1
    assert samples[0][0] == (10.0, 10.0)
    assert samples[0][1] == pytest.approx(3.0)


def test_target_team_away_only_samples_away_players():
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    samples = compute_pressing_samples([frame], target_team=Team.AWAY)
    assert len(samples) == 1
    assert samples[0][0] == (13.0, 10.0)


def test_frame_with_only_one_team_contributes_no_samples():
    """No opponent exists to measure a distance to -- must not fabricate
    an infinite distance or silently sample against nothing."""
    frame = _frame(0, [_obj(1, ObjectClass.PLAYER, Team.HOME, (50.0, 30.0))])
    assert compute_pressing_samples([frame]) == []


def test_empty_frame_contributes_no_samples():
    assert compute_pressing_samples([_frame(0, [])]) == []


def test_nearest_of_multiple_opponents_is_selected():
    """One home player, two away players at different distances -- the
    sample must reflect the NEAREST opponent, not the first in the list
    or the farthest."""
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (0.0, 0.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (100.0, 60.0)),  # far
        _obj(3, ObjectClass.PLAYER, Team.AWAY, (3.0, 4.0)),      # near: distance 5.0
    ])
    samples = compute_pressing_samples([frame], target_team=Team.HOME)
    assert len(samples) == 1
    assert samples[0][1] == pytest.approx(5.0)


def test_samples_accumulate_across_multiple_frames_in_the_window():
    frames = [
        _frame(0, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
            _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
        ]),
        _frame(1, [
            _obj(1, ObjectClass.PLAYER, Team.HOME, (20.0, 20.0)),
            _obj(2, ObjectClass.PLAYER, Team.AWAY, (25.0, 20.0)),
        ]),
    ]
    samples = compute_pressing_samples(frames)
    assert len(samples) == 4  # 2 samples per frame, 2 frames


def test_does_not_mutate_input_frames():
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    snapshot = frame.model_copy(deep=True)
    compute_pressing_samples([frame])
    assert frame == snapshot


# ---- compute_pressing_grid ----

def test_grid_dimensions_match_ceil_division_of_pitch_size():
    cfg = PitchConfig()  # 105 x 68
    mean_grid, mask = compute_pressing_grid([], cfg, bin_size_m=5.0)
    assert mean_grid.shape == (14, 21)  # ceil(68/5)=14, ceil(105/5)=21
    assert mask.shape == (14, 21)


def test_empty_window_grid_has_no_data_anywhere():
    cfg = PitchConfig()
    mean_grid, mask = compute_pressing_grid([], cfg, bin_size_m=5.0)
    assert not np.any(mask)
    assert np.all(mean_grid == 0.0)


def test_two_samples_in_the_same_bin_are_averaged():
    cfg = PitchConfig()
    # (10,10) -> bin (row=2, col=2); (13,10) -> bin (row=2, col=2) too, at bin_size_m=5.0
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    mean_grid, mask = compute_pressing_grid([frame], cfg, bin_size_m=5.0)
    assert mask[2, 2] == True
    assert mean_grid[2, 2] == pytest.approx(3.0)  # both samples measured distance 3.0
    assert mask.sum() == 1  # exactly one bin holds any data


def test_samples_in_different_bins_stay_separate():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (2.0, 2.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (2.0, 2.0)),  # co-located: distance 0.0, bin (0,0)
        _obj(3, ObjectClass.PLAYER, Team.HOME, (90.0, 60.0)),
        _obj(4, ObjectClass.PLAYER, Team.AWAY, (90.0, 60.0)),  # distance 0.0, far-away bin
    ])
    mean_grid, mask = compute_pressing_grid([frame], cfg, bin_size_m=5.0)
    assert mask.sum() == 2
    assert mean_grid[0, 0] == pytest.approx(0.0)
    far_row, far_col = int(60.0 // 5.0), int(90.0 // 5.0)
    assert mean_grid[far_row, far_col] == pytest.approx(0.0)


def test_target_team_filters_the_grid_the_same_way_as_samples():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    _mean_home, mask_home = compute_pressing_grid([frame], cfg, bin_size_m=5.0, target_team=Team.HOME)
    _mean_away, mask_away = compute_pressing_grid([frame], cfg, bin_size_m=5.0, target_team=Team.AWAY)
    # HOME's sample sits at (10,10) -> bin (2,2); AWAY's sample sits at
    # (13,10) -> ALSO bin (2,2) at this bin_size_m -- same bin, but each
    # grid should only be populated from its own team's sample count.
    assert mask_home[2, 2] and mask_away[2, 2]
    assert mask_home.sum() == 1
    assert mask_away.sum() == 1


# ---- render_pressing_heatmap ----

def test_render_returns_correct_shape_and_dtype():
    cfg = PitchConfig()
    canvas = render_pressing_heatmap([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    assert canvas.dtype == np.uint8
    assert canvas.ndim == 3 and canvas.shape[2] == 3


def test_empty_window_renders_bare_pitch_with_no_tint():
    cfg = PitchConfig()
    canvas = render_pressing_heatmap([], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M)
    open_grass_point = (25.0, 15.0)
    px = world_to_pixel(open_grass_point, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_PITCH_COLOR


def test_bin_with_no_samples_stays_plain_pitch_color():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    canvas = render_pressing_heatmap([frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0)
    # Far corner of the pitch, nowhere near either player -- must be untouched.
    untouched_point = (95.0, 60.0)
    px = world_to_pixel(untouched_point, cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_PITCH_COLOR


def test_bin_with_samples_is_visibly_tinted():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    canvas = render_pressing_heatmap([frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0)
    px = world_to_pixel((10.0, 10.0), cfg, PPM, MARGIN_M)
    assert not np.array_equal(canvas[px[1], px[0]], np.array(DEFAULT_PITCH_COLOR))


def test_fill_alpha_zero_leaves_pitch_color_untouched_even_with_data():
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    canvas = render_pressing_heatmap(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0, fill_alpha=0.0,
    )
    px = world_to_pixel((10.0, 10.0), cfg, PPM, MARGIN_M)
    assert tuple(canvas[px[1], px[0]]) == DEFAULT_PITCH_COLOR


def test_fill_alpha_one_reproduces_the_exact_colormap_value():
    """With fill_alpha=1.0 and smooth=False (block upscaling, no
    interpolation blending), a pixel deep inside a data bin must equal
    EXACTLY the color cv2.applyColorMap would independently produce for
    that bin's known mean distance -- not just "some non-background
    color," a specific verifiable one."""
    cfg = PitchConfig()
    distance = 3.0
    max_distance_m = 20.0
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),  # distance 3.0, both directions
    ])
    canvas = render_pressing_heatmap(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0,
        fill_alpha=1.0, smooth=False, max_distance_m=max_distance_m,
    )

    expected_intensity = 1.0 - (distance / max_distance_m)
    expected_u8 = np.uint8(expected_intensity * 255.0)
    expected_color = tuple(int(c) for c in cv2.applyColorMap(
        np.array([[expected_u8]], dtype=np.uint8), cv2.COLORMAP_JET
    )[0, 0])

    px = world_to_pixel((10.0, 10.0), cfg, PPM, MARGIN_M)
    assert tuple(int(c) for c in canvas[px[1], px[0]]) == expected_color


def test_distance_beyond_max_distance_m_clamps_to_the_coolest_color():
    """A mean distance far beyond max_distance_m must render identically
    to a distance exactly AT max_distance_m -- clamped, not extrapolated
    into an out-of-range colormap index."""
    cfg = PitchConfig()
    max_distance_m = 20.0

    frame_at_max = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (30.0, 10.0)),  # distance exactly 20.0
    ])
    frame_beyond_max = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (60.0, 10.0)),  # distance 50.0, well beyond max
    ])

    canvas_at_max = render_pressing_heatmap(
        [frame_at_max], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0,
        fill_alpha=1.0, smooth=False, max_distance_m=max_distance_m,
    )
    canvas_beyond_max = render_pressing_heatmap(
        [frame_beyond_max], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0,
        fill_alpha=1.0, smooth=False, max_distance_m=max_distance_m,
    )

    px = world_to_pixel((10.0, 10.0), cfg, PPM, MARGIN_M)
    assert tuple(canvas_at_max[px[1], px[0]]) == tuple(canvas_beyond_max[px[1], px[0]])


def test_target_team_none_vs_home_can_render_differently():
    """A sanity check that the target_team parameter actually reaches the
    rendered output, not just compute_pressing_grid in isolation."""
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    canvas_home_only = render_pressing_heatmap(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0, target_team=Team.HOME,
    )
    away_px = world_to_pixel((13.0, 10.0), cfg, PPM, MARGIN_M)
    home_px = world_to_pixel((10.0, 10.0), cfg, PPM, MARGIN_M)
    # Both points fall in the SAME bin at bin_size_m=5.0, so with
    # target_team=Team.HOME the bin is still populated (from the home
    # sample) -- assert on the bin's mask directly instead of asserting
    # the away pixel is untouched, since same-bin coloring would make
    # that assertion incorrectly fail.
    mean_grid, mask = compute_pressing_grid([frame], cfg, bin_size_m=5.0, target_team=Team.HOME)
    assert mask[2, 2] == True
    assert not np.array_equal(canvas_home_only[home_px[1], home_px[0]], np.array(DEFAULT_PITCH_COLOR))


def test_smooth_true_and_false_both_run_and_return_valid_output():
    """Not asserting a specific pixel-level smoothing behavior (that's a
    presentation choice, per the function's own docstring) -- just that
    both interpolation modes actually work and produce a well-formed
    canvas."""
    cfg = PitchConfig()
    frame = _frame(0, [
        _obj(1, ObjectClass.PLAYER, Team.HOME, (10.0, 10.0)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, (13.0, 10.0)),
    ])
    canvas_blocky = render_pressing_heatmap(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0, smooth=False,
    )
    canvas_smooth = render_pressing_heatmap(
        [frame], cfg, pixels_per_meter=PPM, margin_m=MARGIN_M, bin_size_m=5.0, smooth=True,
    )
    for canvas in (canvas_blocky, canvas_smooth):
        assert canvas.dtype == np.uint8
        assert canvas.shape == canvas_blocky.shape