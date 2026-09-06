"""
Team classification via jersey-color clustering (Phase 2, final piece).

Runs AFTER the tracker: it needs pixel data (the crop of each player) that
a `TrackedObject` alone doesn't carry, so its input is the raw frame plus
that frame's already-tracked objects, and its output is the same list with
`.team` filled in for players/goalkeepers. Referees and the ball are never
touched -- team is not applicable to them.

Design summary (confirmed in planning):
  - Jersey color feature per player = median BGR of the player's full
    bounding-box crop, after masking out grass-green pixels via an HSV
    threshold. Median (not mean) so a handful of leftover background/skin
    pixels can't drag the estimate -- cheap "dominant color" without
    running a second per-player clustering pass.
  - Two running reference centroids (one per team) are maintained across
    frames rather than re-clustering from scratch and hoping the labels
    land the same way twice in a row. Each frame: fit a fresh k=2 cluster
    over eligible players' colors (cv2.kmeans, no scipy/Hungarian --
    consistent with the rest of the project), reconcile the 2 fresh
    centroids against the 2 running ones (whichever of the 2 possible
    pairings has lower total distance wins -- trivial with only 2
    clusters, no need for a general assignment algorithm), then
    EMA-update the running centroids toward the matched fresh ones. This
    is the same "smooth toward new evidence, don't discard history"
    pattern already used for homography jitter in `smoothed_calibrator.py`,
    applied here to handle lighting drift over a match (shadows
    lengthening, floodlights coming on) without flipping which running
    centroid means which team.
  - Goalkeepers are excluded from the k=2 FIT (their kit is deliberately a
    different color from both outfield teams and would corrupt the
    centroids) but ARE still assigned a team via nearest-centroid once the
    centroids exist, since which side a keeper belongs to matters for
    stats (defensive shape, offside line) even though the jersey-color
    signal for keepers specifically is best-effort and can be wrong when a
    keeper kit doesn't resemble either team's outfield color at all -- a
    known limitation, not fixed here.
  - Referees and the ball are always left with `team=None`.
  - No guessing: a player whose crop has no non-grass pixels left after
    masking (bad detection box, or a genuinely all-green sliver) is
    excluded from both fitting and assignment for that frame -- team stays
    whatever it already was (None by default) rather than being set from
    noise.
  - Per-track color history (Phase 2 re-ID support): every successful
    color extraction (a `TrackedObject` that gets a team assigned) is
    also appended to that track's bounded deque of recent colors. The
    re-identification module (Phase 2 remainder, `tracking/reid.py`)
    reads from this cache via `get_color_history(track_id)` to recognise
    a returning player after a tracking gap. The deque is bounded by
    `max_color_history_per_track` (default 8 frames; 0.27s at 30fps --
    generous, enough to survive one bad frame like an all-green sliver
    without losing the signal).
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..schemas.enums import ObjectClass, Team
from ..schemas.frame_result import TrackedObject


# HSV hue range (OpenCV convention: H in [0, 179]) considered "grass green".
# Paired with a saturation floor so desaturated near-gray/white pixels that
# happen to fall in this hue band (common on jersey highlights/shadows)
# aren't wrongly treated as background.
GREEN_HUE_RANGE: Tuple[int, int] = (35, 85)
GREEN_MIN_SATURATION: int = 60

# Object classes eligible for team assignment at all. Referees and the
# ball are structurally excluded, not just left unassigned by chance.
TEAM_ELIGIBLE_CLASSES = {ObjectClass.PLAYER, ObjectClass.GOALKEEPER}


class BaseTeamClassifier(ABC):
    """Interface every team-classification implementation must satisfy."""

    @abstractmethod
    def assign_teams(
        self, frame: np.ndarray, tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        """Returns tracked_objects with `.team` filled in where possible.
        Implementations mutate the given objects in place and return the
        same list -- callers should not assume the input is left
        untouched."""
        raise NotImplementedError


class JerseyColorTeamClassifier(BaseTeamClassifier):
    def __init__(
        self,
        ema_alpha: float = 0.1,
        min_players_for_fit: int = 4,
        max_color_history_per_track: int = 8,
    ):
        """
        ema_alpha: how much each frame's freshly-fit centroids move the
            running reference centroids. Low by design -- this is meant to
            absorb slow lighting drift over a match, not react to one
            frame's noise.
        min_players_for_fit: minimum number of eligible (non-goalkeeper)
            players with a valid color this frame to attempt a k=2 fit at
            all. Below this, the frame skips fitting and falls back to
            assigning against whatever running centroids already exist (or
            leaves everyone unassigned if none exist yet).
        max_color_history_per_track: per-track deque length for the
            re-identification read path. Capped to bound memory -- 8
            frames at 30fps is 0.27s of color history, generous enough to
            survive one bad frame without losing the signal.
        """
        self.ema_alpha = ema_alpha
        self.min_players_for_fit = min_players_for_fit
        self.max_color_history_per_track = max_color_history_per_track

        # (2, 3) running BGR centroids, or None until the first successful
        # fit. Index identity (which row means "team A") is fixed at first
        # fit and preserved by reconciliation thereafter -- there is no
        # inherent mapping from cluster index to Team.HOME/AWAY, it's just
        # whichever mapping the first fit happened to produce.
        self._running_centroids: Optional[np.ndarray] = None

        # track_id -> bounded deque of BGR float64 colors, most-recent at
        # the right end. Populated lazily: a track is added the first
        # time its color is successfully extracted. Bounded by
        # max_color_history_per_track via deque(maxlen=...) so the cache
        # can't grow without limit. Consulted by the re-identification
        # module via `get_color_history(track_id)`.
        self._color_history: Dict[int, Deque[np.ndarray]] = {}

    def get_color_history(self, track_id: int) -> Optional[Deque[np.ndarray]]:
        """Public read-only view of a single track's recent jersey-color
        history (most-recent-last), or None if no color has ever been
        successfully extracted for this track.

        The returned deque is the classifier's own internal cache, NOT a
        copy -- callers MUST NOT mutate it. This is documented as a
        window onto the cache, not an export.

        Consumed by `tracking/reid.py` to recognise a returning player
        after a tracking gap. Re-ID needs to see the most recent color
        seen for a dropped track to gate the new track's color against
        it; the cache keeps the most recent N frames' worth of colors.
        """
        return self._color_history.get(track_id)

    # ---- jersey color extraction ----

    @staticmethod
    def _jersey_color(frame: np.ndarray, obj: TrackedObject) -> Optional[np.ndarray]:
        """Median BGR of the object's bbox crop after masking out
        grass-green pixels. Returns None if there's no box to crop, or if
        masking removes every pixel (nothing left to estimate a color
        from)."""
        if obj.bounding_box is None:
            return None

        box = obj.bounding_box
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hue, sat = hsv[:, :, 0], hsv[:, :, 1]
        is_green = (
            (hue >= GREEN_HUE_RANGE[0]) & (hue <= GREEN_HUE_RANGE[1])
            & (sat >= GREEN_MIN_SATURATION)
        )
        keep_mask = ~is_green

        pixels = crop[keep_mask]
        if len(pixels) == 0:
            return None

        return np.median(pixels.reshape(-1, 3), axis=0).astype(np.float64)

    # ---- centroid fit/reconcile/update ----

    @staticmethod
    def _reconcile(fresh: np.ndarray, running: np.ndarray) -> np.ndarray:
        """Returns `fresh` reordered so row i corresponds to the same team
        as running[i]. Only two possible pairings exist with k=2, so this
        is a direct comparison rather than a general assignment problem."""
        cost_identity = np.linalg.norm(fresh[0] - running[0]) + np.linalg.norm(fresh[1] - running[1])
        cost_swapped = np.linalg.norm(fresh[0] - running[1]) + np.linalg.norm(fresh[1] - running[0])
        return fresh if cost_identity <= cost_swapped else fresh[::-1]

    def _fit_and_update_centroids(self, colors: np.ndarray) -> None:
        colors32 = colors.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _compactness, _labels, centers = cv2.kmeans(
            colors32, 2, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
        )
        centers = centers.astype(np.float64)

        if self._running_centroids is None:
            self._running_centroids = centers
            return

        matched = self._reconcile(centers, self._running_centroids)
        self._running_centroids = (
            (1 - self.ema_alpha) * self._running_centroids + self.ema_alpha * matched
        )

    def _nearest_team(self, color: np.ndarray) -> Team:
        d0 = np.linalg.norm(color - self._running_centroids[0])
        d1 = np.linalg.norm(color - self._running_centroids[1])
        return Team.HOME if d0 <= d1 else Team.AWAY

    # ---- public entrypoint ----

    def assign_teams(
        self, frame: np.ndarray, tracked_objects: List[TrackedObject]
    ) -> List[TrackedObject]:
        # Colors for outfield players only -- this is the set the k=2 fit
        # is allowed to see. Goalkeepers are deliberately excluded here
        # (see module docstring) even though they'll still get assigned a
        # team below once centroids exist.
        fit_colors = []
        for obj in tracked_objects:
            if obj.object_class != ObjectClass.PLAYER:
                continue
            color = self._jersey_color(frame, obj)
            if color is not None:
                fit_colors.append(color)

        if len(fit_colors) >= self.min_players_for_fit:
            self._fit_and_update_centroids(np.array(fit_colors))

        if self._running_centroids is None:
            # Never successfully fit yet (not enough players, e.g. very
            # start of a clip) -- nothing to assign against. The per-track
            # color-history cache (re-ID support) is still populated below
            # regardless, because re-ID needs a color log per track even
            # before team centroids exist: the very first respawn of a
            # single-player scene can otherwise have no color signal to
            # match against. See `tests/test_reid.py`'s tests for the
            # concrete failure mode this guards against.
            pass
        else:
            for obj in tracked_objects:
                if obj.object_class not in TEAM_ELIGIBLE_CLASSES:
                    continue
                color = self._jersey_color(frame, obj)
                if color is None:
                    continue  # no guessing -- leave team as-is (None by default)
                obj.team = self._nearest_team(color)

        # Per-track color history for the re-identification module.
        # Populated on EVERY successful color extraction for a
        # TEAM_ELIGIBLE_CLASSES object -- independent of whether the
        # team centroids have been fit yet. Bounded by
        # max_color_history_per_track via deque(maxlen=...) so memory is
        # capped. The deque is most-recent-at-the-right; re-ID reads
        # cached[-1] for the most recent color.
        #
        # TODO (v2, perf): the assignment loop above and this loop both
        # call `_jersey_color` on the same objects when centroids
        # exist. For 22 players at 25fps that's ~550 extra
        # cv2.cvtColor+median calls per second of video. The clean
        # refactor is to compute each player's color once into a local
        # dict and reuse it in both loops. Not blocking -- the cost is
        # sub-millisecond per player and well within real-time budget
        # at broadcast frame rates -- but worth doing before the
        # pipeline gets pushed above 30fps or player counts grow.
        for obj in tracked_objects:
            if obj.object_class not in TEAM_ELIGIBLE_CLASSES:
                continue
            color = self._jersey_color(frame, obj)
            if color is None:
                continue
            self._color_history.setdefault(
                obj.track_id, deque(maxlen=self.max_color_history_per_track)
            ).append(color)

        return tracked_objects
