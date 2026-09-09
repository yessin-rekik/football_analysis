"""
Top-down tactical "radar" view (Phase 4a).

Pure function, same shape and same design rule as `coordinates/transform.py`:
imports ONLY from `schemas/` and `config/`, nothing from `calibration/` or
`tracking/`, and nothing from any calibrator/orchestrator/pipeline instance.
It needs world positions and team labels -- both already sit on
`TrackedObject` by the time Phase 3 has run -- and nothing else. That's what
makes this buildable now and testable purely against fabricated
`TrackedObject` lists, no video frame or homography required.

Doubles as a visual sanity check for everything upstream (per the project
plan): a jittery or drifting radar view is an early warning sign for a
calibration or tracking bug, before any numeric stat would reveal it. This
is also why `position_confidence` is rendered as opacity rather than being
used to filter points out -- see the `min_draw_alpha` note below.

Confidence -> opacity, with a floor:
    Phase 3's out-of-bounds policy deliberately keeps an implausible
    projected point's `world_position` populated but forces
    `position_confidence` to exactly 0.0, specifically so it stays visible
    for debugging (see `transform.py`'s module notes) instead of being
    silently dropped. Mapping confidence straight to alpha would make that
    exact point invisible at alpha=0.0 -- defeating the reason it was kept
    in the first place. `min_draw_alpha` (default 0.15) floors the alpha so
    a zero-confidence point still renders, faintly, rather than vanishing.
    A `None` confidence (no calibration attempted at all for that object)
    is treated as full opacity, not low confidence -- those are different
    situations and shouldn't look the same on screen.

Not included in this first pass (nice-to-haves, not blockers for the radar
being useful as a sanity check): penalty-arc curves at the edge of the
penalty area, goal-mouth rectangles. Both are easy additions later behind
the same `_draw_pitch_markings` seam if wanted.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..config.pitch_config import PitchConfig
from ..schemas.enums import ObjectClass, Team
from ..schemas.frame_result import TrackedObject

# Fixed BGR defaults (OpenCV convention, consistent with the rest of the
# project's cv2 usage). All overridable per-call -- these are just sane
# starting points, same "inspection-tuned, not final" caveat as every other
# visual/threshold constant in this project (scene classifier, green-hue
# mask, Phase 3 confidence constants).
DEFAULT_PITCH_COLOR: Tuple[int, int, int] = (60, 140, 60)      # grass green
DEFAULT_LINE_COLOR: Tuple[int, int, int] = (255, 255, 255)     # white markings
DEFAULT_HOME_COLOR: Tuple[int, int, int] = (40, 70, 220)       # red-ish
DEFAULT_AWAY_COLOR: Tuple[int, int, int] = (220, 140, 40)      # blue-ish
DEFAULT_NEUTRAL_COLOR: Tuple[int, int, int] = (170, 170, 170)  # referee / unassigned
DEFAULT_BALL_COLOR: Tuple[int, int, int] = (0, 225, 255)       # bright yellow


def get_canvas_size(
    pitch_config: PitchConfig, pixels_per_meter: float, margin_m: float
) -> Tuple[int, int]:
    """Returns (width_px, height_px) for the given pitch/scale/margin.

    Split out as its own function (rather than inlined in render_radar_
    frame) so a caller building a video overlay can size/allocate a canvas
    once, up front, without needing to call the full render function to
    find out the dimensions."""
    width = int(round((pitch_config.pitch_length + 2 * margin_m) * pixels_per_meter))
    height = int(round((pitch_config.pitch_width + 2 * margin_m) * pixels_per_meter))
    return width, height


def world_to_pixel(
    world_xy: Tuple[float, float],
    pitch_config: PitchConfig,
    pixels_per_meter: float,
    margin_m: float,
) -> Tuple[int, int]:
    """Converts a world-space (meters) point to a pixel coordinate on the
    radar canvas produced by this module.

    y is flipped (canvas row increases as world y decreases) so that
    PitchConfig's "y=0 is the bottom sideline" convention actually reads as
    the bottom of the rendered image, not the top -- gets this right once,
    here, rather than every caller needing to remember to flip it."""
    x, y = world_xy
    margin_px = margin_m * pixels_per_meter
    px = margin_px + x * pixels_per_meter
    py = margin_px + (pitch_config.pitch_width - y) * pixels_per_meter
    return int(round(px)), int(round(py))


def _draw_pitch_markings(
    canvas: np.ndarray,
    pitch_config: PitchConfig,
    pixels_per_meter: float,
    margin_m: float,
    line_color: Tuple[int, int, int],
    line_thickness: int,
) -> None:
    """Draws the pitch boundary, halfway line, center circle, and both
    penalty/goal areas directly from PitchConfig geometry -- NOT from
    PitchCalibrator.project_pitch_outline(). That method projects a
    specific frame's homography into pixel space; this is a clean top-down
    world-space draw that has nothing to do with any single frame's
    calibration, so reusing it here would be the wrong dependency
    direction (this module isn't supposed to know calibration exists)."""

    def w2p(x: float, y: float) -> Tuple[int, int]:
        return world_to_pixel((x, y), pitch_config, pixels_per_meter, margin_m)

    L = pitch_config.pitch_length
    W = pitch_config.pitch_width
    cy = W / 2.0

    # Outer boundary.
    cv2.rectangle(canvas, w2p(0, W), w2p(L, 0), line_color, line_thickness)

    # Halfway line.
    cv2.line(canvas, w2p(L / 2, 0), w2p(L / 2, W), line_color, line_thickness)

    # Center circle + center spot.
    center_px = w2p(L / 2, cy)
    radius_px = int(round(pitch_config.center_circle_radius * pixels_per_meter))
    cv2.circle(canvas, center_px, radius_px, line_color, line_thickness)
    cv2.circle(canvas, center_px, 2, line_color, -1)

    half_pa_w = pitch_config.penalty_area_width / 2.0
    half_ga_w = pitch_config.goal_area_width / 2.0
    pa_depth = pitch_config.penalty_area_depth
    ga_depth = pitch_config.goal_area_depth
    pen_spot = pitch_config.penalty_spot_distance

    # Left penalty + goal area.
    cv2.rectangle(canvas, w2p(0, cy + half_pa_w), w2p(pa_depth, cy - half_pa_w),
                  line_color, line_thickness)
    cv2.rectangle(canvas, w2p(0, cy + half_ga_w), w2p(ga_depth, cy - half_ga_w),
                  line_color, line_thickness)
    cv2.circle(canvas, w2p(pen_spot, cy), 2, line_color, -1)

    # Right penalty + goal area (mirrored).
    cv2.rectangle(canvas, w2p(L, cy + half_pa_w), w2p(L - pa_depth, cy - half_pa_w),
                  line_color, line_thickness)
    cv2.rectangle(canvas, w2p(L, cy + half_ga_w), w2p(L - ga_depth, cy - half_ga_w),
                  line_color, line_thickness)
    cv2.circle(canvas, w2p(L - pen_spot, cy), 2, line_color, -1)


def _marker_color(
    obj: TrackedObject,
    home_color: Tuple[int, int, int],
    away_color: Tuple[int, int, int],
    neutral_color: Tuple[int, int, int],
    ball_color: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    if obj.object_class == ObjectClass.BALL:
        return ball_color
    if obj.team == Team.HOME:
        return home_color
    if obj.team == Team.AWAY:
        return away_color
    return neutral_color  # referee, or a player team classification hasn't reached yet


def _draw_marker(
    canvas: np.ndarray,
    obj: TrackedObject,
    center_px: Tuple[int, int],
    color: Tuple[int, int, int],
    player_marker_radius_px: int,
    ball_marker_radius_px: int,
    goalkeeper_ring_thickness_px: int,
) -> None:
    if obj.object_class == ObjectClass.BALL:
        cv2.circle(canvas, center_px, ball_marker_radius_px, color, -1)
        return

    cv2.circle(canvas, center_px, player_marker_radius_px, color, -1)

    if obj.object_class == ObjectClass.GOALKEEPER:
        # Same fill color as their team, distinguished by an outer ring --
        # keeps goalkeeper vs. outfield visually distinct without needing
        # a third color that could be confused with a team color.
        cv2.circle(
            canvas, center_px, player_marker_radius_px + goalkeeper_ring_thickness_px,
            (0, 0, 0), goalkeeper_ring_thickness_px,
        )


def render_radar_frame(
    tracked_objects: list,
    pitch_config: PitchConfig,
    pixels_per_meter: float = 10.0,
    margin_m: float = 3.0,
    pitch_color: Tuple[int, int, int] = DEFAULT_PITCH_COLOR,
    line_color: Tuple[int, int, int] = DEFAULT_LINE_COLOR,
    line_thickness: int = 2,
    home_color: Tuple[int, int, int] = DEFAULT_HOME_COLOR,
    away_color: Tuple[int, int, int] = DEFAULT_AWAY_COLOR,
    neutral_color: Tuple[int, int, int] = DEFAULT_NEUTRAL_COLOR,
    ball_color: Tuple[int, int, int] = DEFAULT_BALL_COLOR,
    player_marker_radius_px: int = 7,
    ball_marker_radius_px: int = 5,
    goalkeeper_ring_thickness_px: int = 2,
    min_draw_alpha: float = 0.15,
) -> np.ndarray:
    """
    Renders a single top-down radar frame from a frame's tracked objects.

    tracked_objects: List[TrackedObject], typically FrameResult.tracked_objects
                      after Phase 3's transform has run. Objects with
                      world_position=None (NOT_CALIBRATED frames, or any
                      object Phase 3 didn't project) are skipped entirely --
                      there's no meaningful position to place a marker at.
    pitch_config:     drives both canvas size and the drawn markings, so a
                      non-standard pitch config renders correctly with zero
                      changes here.

    Returns a uint8 BGR image (OpenCV convention), sized via get_canvas_size
    for the same pitch_config/pixels_per_meter/margin_m.
    """
    width, height = get_canvas_size(pitch_config, pixels_per_meter, margin_m)
    canvas = np.full((height, width, 3), pitch_color, dtype=np.uint8)

    _draw_pitch_markings(canvas, pitch_config, pixels_per_meter, margin_m,
                          line_color, line_thickness)

    for obj in tracked_objects:
        if obj.world_position is None:
            continue

        confidence = obj.position_confidence if obj.position_confidence is not None else 1.0
        alpha = max(confidence, min_draw_alpha)

        center_px = world_to_pixel(obj.world_position.as_tuple(), pitch_config,
                                    pixels_per_meter, margin_m)
        color = _marker_color(obj, home_color, away_color, neutral_color, ball_color)

        # Draw the marker on a copy, then alpha-blend just that copy back
        # onto the canvas -- lets each object carry its own independent
        # alpha (driven by its own position_confidence) in a single shared
        # image, without needing a true alpha channel.
        overlay = canvas.copy()
        _draw_marker(overlay, obj, center_px, color, player_marker_radius_px,
                     ball_marker_radius_px, goalkeeper_ring_thickness_px)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)

    return canvas