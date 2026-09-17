"""
Phase 4b -- team shape / average formation.

A rolling-window quantity, unlike stats/radar.py's single-frame snapshot:
"formation" is inherently something computed over time -- averaging away
frame-to-frame noise (a player drifting for a throw-in, briefly out of
position for a foul) IS the point, not an optional smoothing layer on top.
A single frame's raw positions is just what the radar view already shows;
it isn't a formation.

Reuses this batch's shared primitive (group_positions_by_team) once per
frame in the window, and reuses draw_pitch_markings/get_canvas_size/
world_to_pixel from stats/radar.py for the pitch background and coordinate
conversion -- deliberately NOT reimplementing pitch drawing a second time.
Both stay pixel-for-pixel consistent with radar.py's visual style by
construction, not by convention.

Averaging strategy: a running mean PER TRACK_ID, not one pooled average
per team. Averaging across all of a team's players into a single blob
would erase the formation entirely -- the whole point is seeing where
each role sits (left-back stays deep and wide across the window), not one
team centroid. A track present for only part of the window (subbed on
partway through, or briefly lost by the tracker) is averaged only over
the frames it actually appeared in -- divided by its own contributing-
frame count, not the window length -- so a player seen in 10% of the
window isn't dragged toward the coordinate origin by dividing by the full
window size.
"""

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from ..config.pitch_config import PitchConfig
from ..schemas.enums import Team
from ..schemas.frame_result import FrameResult
from .radar import (
    DEFAULT_AWAY_COLOR,
    DEFAULT_HOME_COLOR,
    DEFAULT_LINE_COLOR,
    DEFAULT_PITCH_COLOR,
    draw_pitch_markings,
    get_canvas_size,
    world_to_pixel,
)
from .team_positions import group_positions_by_team


def compute_average_positions(
    frames: Sequence[FrameResult],
) -> Dict[Team, Dict[int, Tuple[float, float]]]:
    """
    Computes each track's average world position across the given window
    of frames, grouped by team.

    Returns {Team.HOME: {track_id: (avg_x, avg_y), ...}, Team.AWAY: {...}}.
    A track_id appears only if it had at least one eligible, calibrated
    world position in at least one frame of the window -- there's nothing
    to average for a track that was never seen at all in this window.

    Split out from render_team_shape() as its own function so the
    averaging math is directly testable against known input, independent
    of any rendering/canvas concerns -- same reasoning stats/radar.py
    already used splitting world_to_pixel out from render_radar_frame().
    """
    sums: Dict[Team, Dict[int, Tuple[float, float]]] = {Team.HOME: {}, Team.AWAY: {}}
    counts: Dict[Team, Dict[int, int]] = {Team.HOME: {}, Team.AWAY: {}}

    for frame in frames:
        grouped = group_positions_by_team(frame)
        for team in (Team.HOME, Team.AWAY):
            for track_id, (x, y) in grouped[team]:
                if track_id not in sums[team]:
                    sums[team][track_id] = (0.0, 0.0)
                    counts[team][track_id] = 0
                sx, sy = sums[team][track_id]
                sums[team][track_id] = (sx + x, sy + y)
                counts[team][track_id] += 1

    averages: Dict[Team, Dict[int, Tuple[float, float]]] = {Team.HOME: {}, Team.AWAY: {}}
    for team in (Team.HOME, Team.AWAY):
        for track_id, (sx, sy) in sums[team].items():
            n = counts[team][track_id]
            averages[team][track_id] = (sx / n, sy / n)

    return averages


def render_team_shape(
    frames: Sequence[FrameResult],
    pitch_config: PitchConfig,
    pixels_per_meter: float = 10.0,
    margin_m: float = 3.0,
    pitch_color: Tuple[int, int, int] = DEFAULT_PITCH_COLOR,
    line_color: Tuple[int, int, int] = DEFAULT_LINE_COLOR,
    line_thickness: int = 2,
    home_color: Tuple[int, int, int] = DEFAULT_HOME_COLOR,
    away_color: Tuple[int, int, int] = DEFAULT_AWAY_COLOR,
    marker_radius_px: int = 9,
) -> np.ndarray:
    """
    Renders each team's average formation over `frames` as a top-down
    diagram -- one dot per track_id at its averaged world position, drawn
    on the same pitch background render_radar_frame uses.

    frames: the window to average over. This function has no opinion on
             what "a window" means (last N seconds, a whole half, a whole
             match) -- the caller decides that by slicing before calling.
             An empty sequence renders just the bare pitch (no markers),
             not an error -- an empty window is a valid, if uninteresting,
             input.
    """
    width, height = get_canvas_size(pitch_config, pixels_per_meter, margin_m)
    canvas = np.full((height, width, 3), pitch_color, dtype=np.uint8)
    draw_pitch_markings(canvas, pitch_config, pixels_per_meter, margin_m,
                        line_color, line_thickness)

    averages = compute_average_positions(frames)

    for team, color in ((Team.HOME, home_color), (Team.AWAY, away_color)):
        for world_xy in averages[team].values():
            center_px = world_to_pixel(world_xy, pitch_config, pixels_per_meter, margin_m)
            cv2.circle(canvas, center_px, marker_radius_px, color, -1)

    return canvas