import numpy as np
import pytest

from ..config.pitch_config import PitchKeypointName
from ..schemas.enums import ObjectClass, Team
from ..schemas.frame_result import BoundingBox, PixelPoint, TrackedObject, WorldPoint
from ..stats.overlay import (
    compose_side_by_side,
    compute_combined_size,
    draw_keypoint_overlay,
    draw_tracking_overlay,
)
from ..stats.radar import DEFAULT_AWAY_COLOR, DEFAULT_BALL_COLOR, DEFAULT_HOME_COLOR, DEFAULT_NEUTRAL_COLOR
from ..stats.overlay import DEFAULT_KEYPOINT_LOW_CONFIDENCE_COLOR, DEFAULT_KEYPOINT_VISIBLE_COLOR


def _obj(
    track_id, object_class, team=None, pixel_xy=(50.0, 50.0), box=None, world_xy=None,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        team=team,
        pixel_position=PixelPoint(x=pixel_xy[0], y=pixel_xy[1]),
        bounding_box=box,
        world_position=WorldPoint(x=world_xy[0], y=world_xy[1]) if world_xy is not None else None,
    )


def _blank_frame(height=200, width=300):
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---- draw_tracking_overlay ----

def test_does_not_mutate_input_frame():
    frame = _blank_frame()
    original = frame.copy()
    obj = _obj(1, ObjectClass.PLAYER, Team.HOME, box=BoundingBox(x1=40, y1=30, x2=60, y2=70))
    draw_tracking_overlay(frame, [obj])
    assert np.array_equal(frame, original)


def test_returns_same_shape_and_dtype():
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.PLAYER, Team.HOME, box=BoundingBox(x1=40, y1=30, x2=60, y2=70))
    annotated = draw_tracking_overlay(frame, [obj])
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype


def test_home_player_box_uses_home_color():
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.PLAYER, Team.HOME, box=BoundingBox(x1=40, y1=30, x2=60, y2=70))
    annotated = draw_tracking_overlay(frame, [obj])
    assert tuple(annotated[30, 40]) == DEFAULT_HOME_COLOR  # top-left box corner


def test_away_player_box_uses_away_color():
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.PLAYER, Team.AWAY, box=BoundingBox(x1=40, y1=30, x2=60, y2=70))
    annotated = draw_tracking_overlay(frame, [obj])
    assert tuple(annotated[30, 40]) == DEFAULT_AWAY_COLOR


def test_ball_uses_ball_color():
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.BALL, team=None, box=BoundingBox(x1=40, y1=30, x2=60, y2=70))
    annotated = draw_tracking_overlay(frame, [obj])
    assert tuple(annotated[30, 40]) == DEFAULT_BALL_COLOR


def test_referee_uses_neutral_color():
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.REFEREE, team=None, box=BoundingBox(x1=40, y1=30, x2=60, y2=70))
    annotated = draw_tracking_overlay(frame, [obj])
    assert tuple(annotated[30, 40]) == DEFAULT_NEUTRAL_COLOR


def test_object_without_bounding_box_falls_back_to_circle_marker():
    """The schema allows bounding_box=None even though the tracker always
    populates one in practice -- this must still draw SOMETHING at
    pixel_position rather than silently skipping the object."""
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.BALL, team=None, pixel_xy=(150.0, 100.0), box=None)
    annotated = draw_tracking_overlay(frame, [obj])
    assert tuple(annotated[100, 150]) == DEFAULT_BALL_COLOR


def test_object_with_no_world_position_still_gets_drawn():
    """A missing calibration shouldn't hide the detection/tracking result
    -- the box must still be drawn even though the coordinate label is
    necessarily different."""
    frame = _blank_frame()
    obj = _obj(1, ObjectClass.PLAYER, Team.HOME, box=BoundingBox(x1=40, y1=30, x2=60, y2=70), world_xy=None)
    annotated = draw_tracking_overlay(frame, [obj])
    assert tuple(annotated[30, 40]) == DEFAULT_HOME_COLOR


def test_label_text_is_drawn_above_the_box():
    """Can't verify label CONTENT without OCR, but the label region (just
    above the box) must contain something other than blank background --
    confirms cv2.putText actually ran, not just the rectangle."""
    frame = _blank_frame()
    obj = _obj(
        1, ObjectClass.PLAYER, Team.HOME,
        box=BoundingBox(x1=40, y1=40, x2=80, y2=100), world_xy=(12.3, 45.6),
    )
    annotated = draw_tracking_overlay(frame, [obj])
    label_region = annotated[25:38, 38:110]  # just above the box, where the label is placed
    assert np.any(label_region != 0)


def test_multiple_objects_are_all_drawn_independently():
    frame = _blank_frame()
    objects = [
        _obj(1, ObjectClass.PLAYER, Team.HOME, box=BoundingBox(x1=10, y1=10, x2=30, y2=50)),
        _obj(2, ObjectClass.PLAYER, Team.AWAY, box=BoundingBox(x1=100, y1=100, x2=120, y2=140)),
        _obj(3, ObjectClass.BALL, team=None, pixel_xy=(200.0, 150.0), box=None),
    ]
    annotated = draw_tracking_overlay(frame, objects)
    assert tuple(annotated[10, 10]) == DEFAULT_HOME_COLOR
    assert tuple(annotated[100, 100]) == DEFAULT_AWAY_COLOR
    assert tuple(annotated[150, 200]) == DEFAULT_BALL_COLOR


def test_empty_object_list_returns_unchanged_copy():
    frame = _blank_frame()
    annotated = draw_tracking_overlay(frame, [])
    assert np.array_equal(annotated, frame)
    assert annotated is not frame  # still a copy, not the same array


# ---- draw_keypoint_overlay ----

def _blank_keypoints(num_total=29):
    kps = np.zeros((num_total, 2), dtype=np.float32)
    confs = np.zeros((num_total,), dtype=np.float32)
    return kps, confs


def test_keypoint_overlay_does_not_mutate_input_frame():
    frame = _blank_frame()
    original = frame.copy()
    kps, confs = _blank_keypoints()
    kps[15] = (150, 100)
    confs[15] = 0.9
    draw_keypoint_overlay(frame, kps, confs)
    assert np.array_equal(frame, original)


def test_none_keypoints_returns_unchanged_copy():
    frame = _blank_frame()
    annotated = draw_keypoint_overlay(frame, None, None)
    assert np.array_equal(annotated, frame)
    assert annotated is not frame


def test_visible_keypoint_uses_visible_color_and_is_labeled():
    frame = _blank_frame()
    kps, confs = _blank_keypoints()
    kps[PitchKeypointName.FIELD_CENTER] = (150, 100)
    confs[PitchKeypointName.FIELD_CENTER] = 0.9
    annotated = draw_keypoint_overlay(frame, kps, confs, confidence_threshold=0.5)
    assert tuple(annotated[100, 150]) == DEFAULT_KEYPOINT_VISIBLE_COLOR
    label_region = annotated[85:100, 155:280]
    assert np.any(label_region != 0)  # the keypoint's name got drawn somewhere in this band


def test_below_threshold_keypoint_uses_low_confidence_color_and_no_label():
    frame = _blank_frame()
    kps, confs = _blank_keypoints()
    kps[0] = (100, 100)
    confs[0] = 0.3  # detected, but below the 0.5 default threshold
    annotated = draw_keypoint_overlay(frame, kps, confs, confidence_threshold=0.5)
    assert tuple(annotated[100, 100]) == DEFAULT_KEYPOINT_LOW_CONFIDENCE_COLOR
    label_region = annotated[80:95, 105:250]
    assert not np.any(label_region != 0)  # no label drawn for a below-threshold keypoint


def test_true_non_detection_below_display_floor_is_not_drawn_at_all():
    """A slot the model never meaningfully detected (near-zero confidence,
    placeholder coordinate) must not be drawn -- distinct from a
    below-threshold-but-genuinely-detected keypoint, which IS drawn (just
    in the low-confidence color)."""
    frame = _blank_frame()
    kps, confs = _blank_keypoints()
    kps[9] = (50, 50)
    confs[9] = 0.01  # below the default min_display_confidence of 0.05
    annotated = draw_keypoint_overlay(frame, kps, confs)
    assert tuple(annotated[50, 50]) == (0, 0, 0)


def test_min_display_confidence_floor_is_tunable():
    frame = _blank_frame()
    kps, confs = _blank_keypoints()
    kps[0] = (100, 100)
    confs[0] = 0.3
    # Raising the floor above this keypoint's own confidence must exclude
    # it entirely, even though it would show as "below threshold" (red) at
    # the default floor.
    annotated = draw_keypoint_overlay(frame, kps, confs, confidence_threshold=0.5, min_display_confidence=0.5)
    assert tuple(annotated[100, 100]) == (0, 0, 0)


def test_multiple_keypoints_at_different_tiers_all_drawn_independently():
    frame = _blank_frame()
    kps, confs = _blank_keypoints()
    kps[PitchKeypointName.FIELD_CENTER] = (150, 100)
    confs[PitchKeypointName.FIELD_CENTER] = 0.9  # visible
    kps[0] = (20, 20)
    confs[0] = 0.3  # below threshold
    annotated = draw_keypoint_overlay(frame, kps, confs, confidence_threshold=0.5)
    assert tuple(annotated[100, 150]) == DEFAULT_KEYPOINT_VISIBLE_COLOR
    assert tuple(annotated[20, 20]) == DEFAULT_KEYPOINT_LOW_CONFIDENCE_COLOR


def test_show_labels_false_suppresses_all_labels():
    frame = _blank_frame()
    kps, confs = _blank_keypoints()
    kps[PitchKeypointName.FIELD_CENTER] = (150, 100)
    confs[PitchKeypointName.FIELD_CENTER] = 0.9
    annotated = draw_keypoint_overlay(frame, kps, confs, confidence_threshold=0.5, show_labels=False)
    # The dot itself still renders...
    assert tuple(annotated[100, 150]) == DEFAULT_KEYPOINT_VISIBLE_COLOR
    # ...but no label text anywhere in the region it would have occupied.
    label_region = annotated[85:100, 155:280]
    assert not np.any(label_region != 0)


# ---- compute_combined_size ----

def test_compute_combined_size_scales_up_right_to_match_left_height():
    # left: 200w x 100h ; right: 50w x 50h -> scale factor 2 -> right becomes 100w x 100h
    size = compute_combined_size((200, 100), (50, 50))
    assert size == (300, 100)


def test_compute_combined_size_scales_down_right_to_match_left_height():
    # left: 200w x 50h ; right: 100w x 100h -> scale factor 0.5 -> right becomes 50w x 50h
    size = compute_combined_size((200, 50), (100, 100))
    assert size == (250, 50)


def test_compute_combined_size_no_scaling_when_heights_already_match():
    size = compute_combined_size((200, 100), (80, 100))
    assert size == (280, 100)


def test_compute_combined_size_floors_scaled_width_at_one_pixel():
    """An extreme scale-down (right image vastly taller than left) could
    round its scaled width to zero -- must floor at 1 pixel rather than
    produce a degenerate zero-width result."""
    size = compute_combined_size((10, 1), (1000, 1000000))
    combined_width, combined_height = size
    assert combined_height == 1
    assert combined_width >= 10 + 1  # left width + at least 1 px for the scaled right image


# ---- compose_side_by_side ----

def test_compose_side_by_side_matches_compute_combined_size():
    left = np.zeros((100, 200, 3), dtype=np.uint8)
    right = np.full((50, 50, 3), 255, dtype=np.uint8)
    combined = compose_side_by_side(left, right)
    expected_width, expected_height = compute_combined_size((200, 100), (50, 50))
    assert combined.shape == (expected_height, expected_width, 3)


def test_compose_side_by_side_leaves_left_image_untouched():
    left = np.zeros((100, 200, 3), dtype=np.uint8)
    right = np.full((50, 50, 3), 255, dtype=np.uint8)
    combined = compose_side_by_side(left, right)
    assert np.array_equal(combined[:, :200], left)


def test_compose_side_by_side_right_half_is_nonzero_when_right_is_white():
    left = np.zeros((100, 200, 3), dtype=np.uint8)
    right = np.full((50, 50, 3), 255, dtype=np.uint8)
    combined = compose_side_by_side(left, right)
    assert np.all(combined[:, 200:] == 255)


def test_compose_side_by_side_with_matching_heights_is_a_plain_concat():
    left = np.zeros((100, 200, 3), dtype=np.uint8)
    right = np.full((100, 80, 3), 255, dtype=np.uint8)
    combined = compose_side_by_side(left, right)
    assert combined.shape == (100, 280, 3)
    assert np.array_equal(combined[:, :200], left)
    assert np.array_equal(combined[:, 200:], right)  # no resizing needed, right passes through as-is