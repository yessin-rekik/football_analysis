"""
Phase 4 (visualization) -- drawing tracked-object overlays directly onto
source video frames, and compositing two rendered views side by side.

Split out as its own stats/ module rather than inlined in the video-loop
driver script (examples/run_match_pipeline.py), for the same reason every
other Phase 4 visualization lives in stats/ and not in the driver:
drawing logic is real logic -- it has behavior worth testing directly
(which color a class/team gets, how a missing world_position degrades,
how two differently-sized images get composited) independent of any video
file, model, or CLI flag. The driver's only job is deciding WHEN to call
these functions and where to write the result; deciding WHAT gets drawn
belongs here, tested the same way render_radar_frame / render_team_shape
/ render_voronoi_frame / render_pressing_heatmap already are.

(An earlier draft of the annotated-video feature drew boxes/labels inline
inside the driver script itself, as a quick diff, and was deliberately
reverted before being applied -- specifically because it broke this
project's own "driver scripts own wiring only" rule. This module is that
correction, not a new pattern.)
"""

from typing import List, Tuple

import cv2
import numpy as np

from ..schemas.enums import ObjectClass, Team
from ..schemas.frame_result import TrackedObject
from .radar import DEFAULT_AWAY_COLOR, DEFAULT_BALL_COLOR, DEFAULT_HOME_COLOR, DEFAULT_NEUTRAL_COLOR


def _overlay_color(obj: TrackedObject) -> Tuple[int, int, int]:
    """Same color convention as stats/radar.py -- reusing its constants
    (not redefining new ones) so a viewer who has already seen the radar,
    team shape, or Voronoi output recognizes the same team/ball colors on
    the source footage too."""
    if obj.object_class == ObjectClass.BALL:
        return DEFAULT_BALL_COLOR
    if obj.team == Team.HOME:
        return DEFAULT_HOME_COLOR
    if obj.team == Team.AWAY:
        return DEFAULT_AWAY_COLOR
    return DEFAULT_NEUTRAL_COLOR


def draw_tracking_overlay(
    frame: np.ndarray,
    tracked_objects: List[TrackedObject],
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Draws a bounding box, track_id, and (when calibrated) real-world
    coordinates directly on a COPY of the source frame -- never mutates
    the frame passed in, since it's typically the same array
    MatchPipeline.process_frame just read from, and a caller shouldn't
    have to assume this function is destructive to use it.

    World coordinates come straight from Phase 3's output
    (TrackedObject.world_position) -- this function does no coordinate
    math of its own, only formatting and drawing. An object with no
    world_position (NOT_CALIBRATED this frame, or not projected for some
    other reason) still gets a box and its track_id, just without the
    coordinate suffix -- a missing calibration shouldn't hide the
    detection/tracking result entirely; those are two different pieces of
    information and losing one shouldn't cost you the other.

    An object with no bounding_box (the schema allows it, though the
    tracker always populates one in practice) falls back to a small
    filled circle at pixel_position instead of a rectangle -- this never
    silently skips drawing an object just because one field wasn't set.
    """
    annotated = frame.copy()
    for obj in tracked_objects:
        color = _overlay_color(obj)

        if obj.bounding_box is not None:
            p1 = (int(obj.bounding_box.x1), int(obj.bounding_box.y1))
            p2 = (int(obj.bounding_box.x2), int(obj.bounding_box.y2))
            cv2.rectangle(annotated, p1, p2, color, box_thickness)
            label_origin = (p1[0], max(0, p1[1] - 8))
        else:
            p = (int(obj.pixel_position.x), int(obj.pixel_position.y))
            cv2.circle(annotated, p, 4, color, -1)
            label_origin = (p[0], max(0, p[1] - 8))

        label = (
            f"#{obj.track_id} ({obj.world_position.x:.1f},{obj.world_position.y:.1f})m"
            if obj.world_position is not None else f"#{obj.track_id}"
        )
        cv2.putText(annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness, cv2.LINE_AA)

    return annotated


def compute_combined_size(left_size: Tuple[int, int], right_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Computes the (width, height) of compose_side_by_side()'s output, given
    the two input images' own (width, height) -- WITHOUT needing to
    actually build either image first.

    The video-loop driver needs this up front, before its frame loop
    starts, in order to open a fixed-size cv2.VideoWriter -- VideoWriter
    requires its frame size be known at construction time, not discovered
    from the first frame written to it.

    compose_side_by_side() calls this function internally for its own
    sizing math, so the two are guaranteed to agree -- a driver that
    pre-computes a size with this function will never see
    compose_side_by_side() produce a differently-shaped result.
    """
    left_width, left_height = left_size
    right_width, right_height = right_size
    scale = left_height / right_height
    scaled_right_width = max(1, int(round(right_width * scale)))
    return left_width + scaled_right_width, left_height


def compose_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Composites two BGR images side by side, scaling `right` to match
    `left`'s height (preserving `right`'s own aspect ratio -- never
    stretching or squashing it) before concatenating.

    `left` is always the fixed reference and is never resized or
    distorted; `right` adapts to it. This exists specifically for
    combining the annotated source-footage frame (`left`) with the radar
    view (`right`): the source footage should never be resized just to
    make room for a diagram next to it.

    Returns an image of shape (left.height, combined_width, 3), where
    combined_width matches what compute_combined_size() would compute for
    the same two images' sizes.
    """
    target_width, target_height = compute_combined_size(
        (left.shape[1], left.shape[0]), (right.shape[1], right.shape[0])
    )
    scaled_right_width = target_width - left.shape[1]
    right_resized = cv2.resize(right, (scaled_right_width, target_height), interpolation=cv2.INTER_AREA)
    return np.hstack([left, right_resized])