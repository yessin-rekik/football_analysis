"""
Phase 4b -- pressing intensity / distance-to-nearest-opponent heatmap.

A rolling-window quantity, same reasoning as team_shape.py: a single
frame's distance-to-nearest-opponent values are just a handful of sparse
points (one per player on the pitch that instant) -- nowhere near enough
samples to call it a heatmap. The whole point of this stat is
accumulating many frames' worth of samples so a spatial pattern (where
does tight marking happen most often?) actually emerges.

Metric: for each player, the Euclidean distance (world/meter space) to
their nearest OPPONENT (a player of the other team) -- literally what the
project plan calls this stat. Lower distance = tighter marking = more
pressing at that location. A player is only sampled in frames where at
least one opponent exists to measure distance to; a frame where one team
has zero eligible players contributes no samples for anyone (there's no
opponent to be pressed by).

Binning, not per-pixel accumulation: samples are aggregated into a coarse
grid of `bin_size_m` x `bin_size_m` pitch cells (default 5m), not
per-pixel. A per-pixel bin at this module's usual ~10 pixels/meter canvas
resolution would very often hold zero or one sample across an entire
window -- statistically meaningless, and would render as pure noise. A
bin with literally zero samples stays untinted (bare pitch color) rather
than being colored as if "no pressing happened" -- those are different
things (no data vs. confirmed low pressure), same "don't fabricate output
from nothing" rule already used in team_shape.py (empty-window handling)
and voronoi.py (zero-eligible-players handling).

Reuses draw_pitch_markings/get_canvas_size from stats/radar.py and
group_positions_by_team from stats/team_positions.py, same as the rest of
this batch.
"""

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config.pitch_config import PitchConfig
from ..schemas.enums import Team
from ..schemas.frame_result import FrameResult
from .radar import DEFAULT_LINE_COLOR, DEFAULT_PITCH_COLOR, draw_pitch_markings, get_canvas_size
from .team_positions import group_positions_by_team


def compute_pressing_samples(
    frames: Sequence[FrameResult],
    target_team: Optional[Team] = None,
) -> List[Tuple[Tuple[float, float], float]]:
    """
    Collects (position, distance_to_nearest_opponent) samples across the
    given window of frames.

    target_team: None (default) samples EVERY eligible player regardless
        of team -- "how tightly is anyone marked, anywhere on the pitch."
        Team.HOME samples only home players (i.e. "where is HOME under
        the most pressure," which is equivalently a map of where AWAY is
        pressing effectively). Team.AWAY is the mirror case.

    A frame contributes zero samples for a given team if the OPPOSING
    team has no eligible player that frame -- there is no opponent to
    measure a distance to, so nothing is sampled rather than a fabricated
    "infinite distance" or a skipped-but-silent player.

    Split out from render_pressing_heatmap() as its own function so the
    sample collection is directly testable against known player
    positions, independent of binning/color/rendering concerns -- same
    pattern as compute_average_positions() and compute_ownership_grid()
    elsewhere in this batch.
    """
    samples: List[Tuple[Tuple[float, float], float]] = []

    for frame in frames:
        grouped = group_positions_by_team(frame)
        home_pts = np.array([xy for _, xy in grouped[Team.HOME]], dtype=np.float64).reshape(-1, 2)
        away_pts = np.array([xy for _, xy in grouped[Team.AWAY]], dtype=np.float64).reshape(-1, 2)

        if len(home_pts) == 0 or len(away_pts) == 0:
            continue  # no opponent to measure against for anyone, this frame

        if target_team is None or target_team == Team.HOME:
            for hx, hy in home_pts:
                dists = np.hypot(away_pts[:, 0] - hx, away_pts[:, 1] - hy)
                samples.append(((float(hx), float(hy)), float(dists.min())))

        if target_team is None or target_team == Team.AWAY:
            for ax, ay in away_pts:
                dists = np.hypot(home_pts[:, 0] - ax, home_pts[:, 1] - ay)
                samples.append(((float(ax), float(ay)), float(dists.min())))

    return samples


def compute_pressing_grid(
    frames: Sequence[FrameResult],
    pitch_config: PitchConfig,
    bin_size_m: float = 5.0,
    target_team: Optional[Team] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bins compute_pressing_samples()'s output into a coarse
    (n_rows, n_cols) grid over the pitch.

    Returns (mean_distance_grid, has_data_mask):
      mean_distance_grid: float array, mean nearest-opponent distance
          (meters) of samples falling in each bin. Value is meaningless
          (left as 0.0) wherever has_data_mask is False -- callers MUST
          check the mask, not just assume every cell holds a real value.
      has_data_mask: bool array, True where at least one sample landed in
          that bin.

    Grid indexing: row 0 = the y-in-[0, bin_size_m) strip (PitchConfig's
    "y=0 is the bottom sideline" convention), col 0 = the x-in-[0,
    bin_size_m) strip (x=0 is the left goal line). This is a DIFFERENT
    orientation from the rendered canvas (which flips y for display, see
    stats/radar.py's world_to_pixel) -- render_pressing_heatmap() handles
    that flip when it upscales this grid onto the canvas; a caller reading
    this grid directly should not assume it's already display-oriented.
    """
    n_cols = max(1, math.ceil(pitch_config.pitch_length / bin_size_m))
    n_rows = max(1, math.ceil(pitch_config.pitch_width / bin_size_m))

    sums = np.zeros((n_rows, n_cols), dtype=np.float64)
    counts = np.zeros((n_rows, n_cols), dtype=np.int64)

    for (x, y), distance in compute_pressing_samples(frames, target_team):
        col = min(n_cols - 1, max(0, int(x // bin_size_m)))
        row = min(n_rows - 1, max(0, int(y // bin_size_m)))
        sums[row, col] += distance
        counts[row, col] += 1

    has_data_mask = counts > 0
    mean_distance_grid = np.zeros((n_rows, n_cols), dtype=np.float64)
    mean_distance_grid[has_data_mask] = sums[has_data_mask] / counts[has_data_mask]

    return mean_distance_grid, has_data_mask


def render_pressing_heatmap(
    frames: Sequence[FrameResult],
    pitch_config: PitchConfig,
    pixels_per_meter: float = 10.0,
    margin_m: float = 3.0,
    bin_size_m: float = 5.0,
    target_team: Optional[Team] = None,
    max_distance_m: float = 20.0,
    fill_alpha: float = 0.6,
    smooth: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    pitch_color: Tuple[int, int, int] = DEFAULT_PITCH_COLOR,
    line_color: Tuple[int, int, int] = DEFAULT_LINE_COLOR,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Renders a pressing-intensity heatmap over `frames`: bins with data are
    tinted from cool (far from any opponent -- low pressing) to hot (close
    to an opponent -- high pressing), drawn on the same pitch background
    render_radar_frame/render_team_shape/render_voronoi_frame use. Bins
    with zero samples are left as plain pitch color -- see module
    docstring for why that's a deliberate choice, not a gap.

    max_distance_m: distances at or beyond this are clamped to the "coolest"
        end of the color scale purely for color normalization -- it does
        NOT affect compute_pressing_grid()'s actual returned distance
        values, only how this function maps them to a color.
    smooth: False (default) upscales each coarse bin as a sharp block
        (cv2.INTER_NEAREST) -- visually honest about the bin_size_m
        resolution the data actually supports, rather than implying more
        spatial precision than exists. True uses cv2.INTER_LINEAR for a
        smoother-looking gradient between bins, at the cost of that
        honesty -- a presentation choice, not a correctness one.
    """
    width, height = get_canvas_size(pitch_config, pixels_per_meter, margin_m)
    canvas = np.full((height, width, 3), pitch_color, dtype=np.uint8)

    mean_distance_grid, has_data_mask = compute_pressing_grid(frames, pitch_config, bin_size_m, target_team)

    if np.any(has_data_mask):
        # Lower distance -> higher press intensity -> higher grayscale
        # value -> "hot" end of the colormap (see COLORMAP_JET's
        # convention: 0 maps to blue/cool, 255 to red/hot).
        clamped = np.clip(mean_distance_grid, 0.0, max_distance_m)
        intensity = 1.0 - (clamped / max_distance_m)
        intensity_u8 = (intensity * 255.0).astype(np.uint8)

        colored_grid = cv2.applyColorMap(intensity_u8, colormap).reshape(mean_distance_grid.shape[0],
                                                                          mean_distance_grid.shape[1], 3)

        # Grid rows/cols run bottom-to-top in world y (row 0 = y in
        # [0, bin_size_m)) per compute_pressing_grid()'s documented
        # convention, but the canvas is drawn top-to-bottom in display
        # space with y flipped (see world_to_pixel) -- flip vertically
        # here so a bin near the bottom sideline actually renders near
        # the bottom of the image, not the top.
        colored_grid = np.flipud(colored_grid)
        alpha_grid = np.flipud(has_data_mask.astype(np.float64) * fill_alpha)

        interpolation = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
        colored_full = cv2.resize(colored_grid, (width, height), interpolation=interpolation)
        alpha_full = cv2.resize(alpha_grid, (width, height), interpolation=interpolation)

        blended = (
            alpha_full[:, :, None] * colored_full.astype(np.float64)
            + (1.0 - alpha_full[:, :, None]) * canvas.astype(np.float64)
        )
        canvas = np.clip(blended, 0, 255).astype(np.uint8)

    draw_pitch_markings(canvas, pitch_config, pixels_per_meter, margin_m, line_color, line_thickness)

    return canvas