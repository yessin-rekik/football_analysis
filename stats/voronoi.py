"""
Phase 4b -- Voronoi space-control diagram.

Deliberately a SINGLE-FRAME snapshot, unlike team_shape.py's rolling
window. A Voronoi diagram is a geometric partition of the pitch given
player positions AT ONE INSTANT -- "the Voronoi diagram averaged over 10
seconds" isn't a coherent single geometric object the way an average
formation position is (see the Phase 4b design discussion: team shape and
the pressing heatmap both structurally need a window to be meaningful at
all; Voronoi structurally does not, and forcing it into the same windowed
shape as the other two would be the wrong fit for exactly one of the
three).

Still takes `frames: Sequence[FrameResult]` -- the same input shape as
team_shape.py and pressing_heatmap.py -- for interface consistency across
the batch, rather than a bare single FrameResult. If handed more than one
frame, ONLY THE LAST ONE is used; earlier frames are silently-by-design
ignored (documented here, not silently discarded without a trace) --
callers wanting a single frame's diagram just pass a length-1 sequence.

Ownership rule: each pixel's real-world position is assigned to whichever
team has the nearest player to it (Euclidean distance in world/meter
space, not pixel space -- meters are what's physically meaningful here,
and pixel space would distort the answer under a non-square
pixels_per_meter mapping, though this project always uses a square one).
Only Team.HOME/Team.AWAY players contribute cells -- same eligibility as
group_positions_by_team's default (the ball and referees have no team and
are excluded from the ownership computation entirely, exactly as they are
from every other Phase 4b stat).

Reuses draw_pitch_markings/get_canvas_size/world_to_pixel and the team
color constants from stats/radar.py, and group_positions_by_team from
stats/team_positions.py -- same "don't reimplement the shared pieces"
rule as team_shape.py.
"""

from typing import Dict, List, Sequence, Tuple

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

# Index convention used internally for the vectorized nearest-team lookup
# below -- 0/1 rather than the Team enum directly, since numpy needs a
# plain integer array to do argmin/indexing efficiently. Never exposed
# outside this module.
_HOME_INDEX = 0
_AWAY_INDEX = 1


def _collect_players(frame: FrameResult) -> Tuple[np.ndarray, np.ndarray]:
    """Flattens one frame's team-labeled positions into parallel arrays:
    world_xy (N, 2) float32, and team_index (N,) int (0=home, 1=away).
    Track identity isn't needed for Voronoi ownership (only position and
    team matter), so track_id is dropped here -- unlike team_shape.py,
    which needs it for per-track averaging."""
    grouped = group_positions_by_team(frame)
    world_xy: List[Tuple[float, float]] = []
    team_index: List[int] = []
    for _track_id, xy in grouped[Team.HOME]:
        world_xy.append(xy)
        team_index.append(_HOME_INDEX)
    for _track_id, xy in grouped[Team.AWAY]:
        world_xy.append(xy)
        team_index.append(_AWAY_INDEX)
    return (
        np.array(world_xy, dtype=np.float32).reshape(-1, 2),
        np.array(team_index, dtype=np.int32),
    )


def compute_ownership_grid(
    frame: FrameResult,
    pitch_config: PitchConfig,
    pixels_per_meter: float,
    margin_m: float,
) -> np.ndarray:
    """
    Computes, for every pixel of the radar-style canvas, which team's
    nearest player claims that point -- returns an (height, width) int
    array of 0 (home) / 1 (away). Returns an all-zero array of that shape
    if there are no eligible players at all this frame (nothing to
    partition; render_voronoi_frame treats this as "skip the fill
    entirely" rather than trusting this degenerate output as meaningful).

    Split out from render_voronoi_frame() as its own function so the
    ownership math is directly testable against known player positions,
    independent of any rendering/canvas/color concerns -- same reasoning
    already used for compute_average_positions() in team_shape.py.

    TODO (v2, perf): this computes a full (H*W, N_players) distance matrix
    in one vectorized shot -- fast enough at this module's default
    ~10 pixels/meter canvas resolution (roughly 800K pixels), but a much
    higher pixels_per_meter would grow that matrix quadratically. If this
    ever needs to run at higher resolution or inside a tight real-time
    loop, downsample the grid (e.g. compute ownership at half resolution
    and cv2.resize with nearest-neighbor back up) rather than computing
    every pixel directly. Not blocking at the resolutions this project
    actually renders at.
    """
    width, height = get_canvas_size(pitch_config, pixels_per_meter, margin_m)
    world_xy, team_index = _collect_players(frame)

    if len(world_xy) == 0:
        return np.zeros((height, width), dtype=np.int32)

    margin_px = margin_m * pixels_per_meter
    px_grid, py_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Inverse of world_to_pixel's mapping (see stats/radar.py): recovers
    # the real-world (x, y) that each pixel center corresponds to.
    world_x = (px_grid.astype(np.float32) - margin_px) / pixels_per_meter
    world_y = pitch_config.pitch_width - (py_grid.astype(np.float32) - margin_px) / pixels_per_meter

    world_x_flat = world_x.ravel()
    world_y_flat = world_y.ravel()

    dx = world_x_flat[:, None] - world_xy[None, :, 0]
    dy = world_y_flat[:, None] - world_xy[None, :, 1]
    dist_sq = dx * dx + dy * dy  # (H*W, N_players), squared distance is enough for argmin

    nearest_player_idx = np.argmin(dist_sq, axis=1)
    nearest_team = team_index[nearest_player_idx]

    return nearest_team.reshape(height, width)


def render_voronoi_frame(
    frames: Sequence[FrameResult],
    pitch_config: PitchConfig,
    pixels_per_meter: float = 10.0,
    margin_m: float = 3.0,
    pitch_color: Tuple[int, int, int] = DEFAULT_PITCH_COLOR,
    line_color: Tuple[int, int, int] = DEFAULT_LINE_COLOR,
    line_thickness: int = 2,
    home_color: Tuple[int, int, int] = DEFAULT_HOME_COLOR,
    away_color: Tuple[int, int, int] = DEFAULT_AWAY_COLOR,
    fill_alpha: float = 0.35,
    draw_players: bool = True,
    player_marker_radius_px: int = 6,
) -> np.ndarray:
    """
    Renders a single-frame Voronoi space-control diagram: the pitch tinted
    by whichever team's nearest player claims each region, drawn on the
    same pitch background render_radar_frame/render_team_shape use.

    frames: only frames[-1] is used -- see module docstring for why
             Voronoi doesn't take a rolling window the way team_shape.py
             and the pressing heatmap do. Pass a length-1 sequence for the
             common case of "just this frame's diagram".
    fill_alpha: opacity of the team-color tint over the grass background
             (0 = invisible fill, 1 = solid team color). Pitch markings
             are drawn AFTER the fill, so boundary/halfway/circle lines
             stay crisp and visible on top of the tinted regions rather
             than being obscured by them.
    draw_players: overlays a small marker at each contributing player's
             exact position on top of everything -- lets a viewer see
             which players are actually driving the boundaries, the same
             way render_radar_frame shows player markers. Purely visual;
             does not affect the ownership computation itself.

    An empty `frames` sequence, or a last frame with zero eligible
    players, renders the bare pitch with no tint at all -- there's no
    meaningful partition to draw, same "don't fabricate output from
    nothing" convention as compute_average_positions() returning empty
    dicts for an empty window.
    """
    width, height = get_canvas_size(pitch_config, pixels_per_meter, margin_m)
    canvas = np.full((height, width, 3), pitch_color, dtype=np.uint8)

    if len(frames) == 0:
        draw_pitch_markings(canvas, pitch_config, pixels_per_meter, margin_m, line_color, line_thickness)
        return canvas

    frame = frames[-1]
    world_xy, team_index = _collect_players(frame)

    if len(world_xy) > 0:
        ownership = compute_ownership_grid(frame, pitch_config, pixels_per_meter, margin_m)

        color_lookup = np.array([home_color, away_color], dtype=np.uint8)  # index 0/1, matches _HOME_INDEX/_AWAY_INDEX
        fill = color_lookup[ownership]  # (height, width, 3), vectorized per-pixel color lookup

        canvas = cv2.addWeighted(fill, fill_alpha, canvas, 1.0 - fill_alpha, 0)

    draw_pitch_markings(canvas, pitch_config, pixels_per_meter, margin_m, line_color, line_thickness)

    if draw_players and len(world_xy) > 0:
        team_colors = {_HOME_INDEX: home_color, _AWAY_INDEX: away_color}
        for xy, t_idx in zip(world_xy, team_index):
            center_px = world_to_pixel((float(xy[0]), float(xy[1])), pitch_config, pixels_per_meter, margin_m)
            cv2.circle(canvas, center_px, player_marker_radius_px, team_colors[t_idx], -1)
            cv2.circle(canvas, center_px, player_marker_radius_px, (0, 0, 0), 1)  # thin outline for contrast

    return canvas