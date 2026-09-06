"""
Re-identification after long tracking gaps (final piece of Phase 2).

Scope, on purpose: this class answers "is this frame's brand-new track
actually a returning player that the tracker already dropped?" -- exactly
the case `ByteTracker` deliberately does NOT handle (it gives every track
30 frames of grace and then forgets it). Re-ID sits BETWEEN the tracker
and the team classifier's world-position / stats consumers, in the
pipeline order:

    detector.detect(frame)            -> List[Detection]
    tracker.update(detections)        -> List[TrackedObject]
    team_clf.assign_teams(frame, t)   -> mutates obj.team, populates
                                         team_clf._color_history
    reid.update(t, cal_info, fps)     -> re-stamps track_id + provenance
                                         where applicable

Re-ID depends on the team classifier having run this frame, because the
team classifier already extracts a per-object jersey color and
(after the additive change in `JerseyColorTeamClassifier.get_color_history`)
keeps a per-track color deque -- re-ID consults that cache to recognise a
returning player. Re-ID ALSO keeps its own per-track history, separately,
because it needs:
  - a per-track world position snapshot from the moment the track dropped
    (the team classifier's cache doesn't carry world coords), and
  - a fallback color snapshot in case the team classifier's cache is
    empty for an old track (e.g. the team classifier was instantiated
    after the track was first seen, or it skipped the color extraction
    that frame due to an all-green crop).
This mirrors the `_Track` pattern in `ByteTracker`: deliberately NOT
pydantic, deliberately not exported, mutable per-track bookkeeping.

Handoff with ByteTracker (min_reid_gap_frames):
    `ByteTracker` is deliberately given `max_age` frames of internal
    grace before it forgets a track -- including a stage-2 low-confidence
    recovery pass specifically meant to survive brief (1-frame-ish)
    occlusions without ever handing the problem to re-ID at all. Without
    a floor here, re-ID was racing that recovery: a track missing for a
    SINGLE frame was immediately eligible for re-identification, and
    could lose to an older, unrelated dropped track that happened to have
    a marginally closer jersey-color match (color distance is the primary
    ranking key; recency is only a tiebreak on exact ties -- see step 5
    below). `min_reid_gap_frames` closes that gap: a track isn't offered
    as re-ID-eligible until it's been gone at least as long as
    `ByteTracker` would already have needed to give up on it internally.
    Construct this with `min_reid_gap_frames=tracker.max_age` so the two
    stay in sync by construction rather than as two independently-tuned
    numbers that can silently drift apart.

Algorithm per frame (in order):
  0. Identify newly-spawned tracks (in current frame, not in previous).
     Exclude BALL and REFEREE -- no reliable identity signal in v1.
  1. Garbage-collect stale history entries past `max_reid_gap_s`.
  2. Build candidate set from dropped tracks, filtering by object_class
     AND by `min_reid_gap_frames` (must have been dropped at least this
     long -- see "Handoff with ByteTracker" above).
  3. World-speed gate: reject candidates whose implied speed since drop
     exceeds `max_plausible_speed_mps`. SKIPPED if no homography this
     frame (re-ID is most useful exactly when calibration is bad).
  4. Color gate: reject candidates whose jersey color differs from the
     incoming track's by more than `color_match_max_distance_bgr` in BGR
     Euclidean distance. (v1 limitation: the color distance IS the team
     signal; an explicit team-equality check is v2.)
  5. Pick best surviving candidate (lowest color distance, ties broken by
     most-recent).
  6. Re-stamp: set new_obj.track_id = old.stored_track_id, new_obj.provenance
     = RE_IDENTIFIED_AFTER_GAP. Move old to "consumed" (removed from the
     candidate set) and re-add the re-linked new_obj to re-ID's "still
     alive" history.
  7. Update history for continuous tracks (refresh last-seen, last color,
     last world position). Mark previously-alive tracks missing this
     frame as "dropped" with a timestamp.

The per-frame cost is O(D + N) where D is the dropped-history size
(bounded by `max_reid_gap_s * fps`) and N is `len(tracked_objects)`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..schemas.enums import ObjectClass, PositionProvenance, Team
from ..schemas.frame_result import CalibrationInfo, TrackedObject
from .team_classifier import JerseyColorTeamClassifier


# Object classes for which re-ID is meaningful. Ball has no identity
# signal at all (it's a small sphere; distinguishing "this ball" from
# "that ball" across a gap is pure guesswork). Referee re-entry after a
# zoom is not necessarily the same referee -- and a wrong referee merge
# is worse than a missed re-ID since the field is usually small enough
# that a new ID is cheap.
REID_ELIGIBLE_CLASSES = {ObjectClass.PLAYER, ObjectClass.GOALKEEPER}


@dataclass
class _ReIdTrackHistory:
    """Private, mutable per-track state re-ID keeps externally because
    `ByteTracker._Track` is private and can't be extended. Same pattern as
    `_Track`: deliberately NOT pydantic, deliberately not exported."""

    track_id: int
    object_class: ObjectClass
    last_pixel_xy: Tuple[float, float]
    last_world_xy: Optional[Tuple[float, float]]  # None if not calibrated last seen
    last_known_color: Optional[np.ndarray]        # (3,) BGR float64
    last_team: Optional[Team]
    last_seen_frame_index: int
    last_seen_timestamp_s: float
    is_dropped: bool
    time_of_drop_s: Optional[float] = None        # set when is_dropped transitions to True


class BaseReIdentifier(ABC):
    """Interface every re-identification implementation must satisfy."""

    @abstractmethod
    def update(
        self,
        tracked_objects: List[TrackedObject],
        calibration_info: Optional[CalibrationInfo],
        video_fps: float,
    ) -> List[TrackedObject]:
        """Advance re-ID by one frame. `tracked_objects` is this frame's
        tracker output (post-team-classifier). `calibration_info` is the
        per-frame calibration result from the orchestrator (re-ID uses
        only the homography, if present, to project the incoming track's
        pixel position into world coordinates for the speed gate).

        Mutates `tracked_objects` in place where re-ID matches are found
        (re-stamps `track_id` and `provenance`), and returns the same
        list. Callers should not assume the input is left untouched."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-track history, starting fresh as if no frames
        had been processed yet -- e.g. after a hard scene cut, where
        carrying pre-cut identity links forward would be actively wrong."""
        raise NotImplementedError


class ReIdentifier(BaseReIdentifier):
    def __init__(
        self,
        team_classifier: JerseyColorTeamClassifier,
        min_reid_gap_frames: int = 30,
        max_reid_gap_s: float = 5.0,
        max_plausible_speed_mps: float = 12.0,
        color_match_max_distance_bgr: float = 60.0,
        appearance_embedder: Optional[
            Callable[[TrackedObject, TrackedObject, np.ndarray, np.ndarray], float]
        ] = None,
        appearance_match_threshold: float = 0.0,
    ):
        """
        team_classifier: the JerseyColorTeamClassifier instance the runner
            is using this frame. Re-ID does NOT depend on its centroids
            being fit; it only consults the per-track color cache via
            `get_color_history(track_id)`. Passing the instance (not just
            the class) keeps the option open for a future seam that needs
            to call instance methods (e.g. `_nearest_team` for a v2 team-
            equality gate).

        min_reid_gap_frames: a dropped track is NOT offered as a re-ID
            candidate until it has been gone at least this many frames.
            This is the floor that keeps re-ID out of ByteTracker's
            territory -- ByteTracker already has `max_age` frames of
            internal grace (including a stage-2 low-confidence recovery
            pass built specifically to survive brief occlusions), so
            re-ID engaging on a 1-frame gap doesn't just duplicate that
            effort, it can actively steal the correct track_id if an
            older, unrelated dropped track happens to have a marginally
            closer color match (color distance is the primary ranking
            key; recency is only a tiebreak on exact ties). Construct
            this as `ReIdentifier(..., min_reid_gap_frames=tracker.max_age)`
            so the two components stay in sync by construction, not as
            two independently-tuned numbers that can drift apart.
            Default of 30 matches `ByteTracker`'s own default `max_age`;
            override if you construct `ByteTracker` with a non-default
            value.

        max_reid_gap_s: how long a dropped track is kept in re-ID's
            candidate history before being garbage-collected. After this
            many seconds, even a perfect color match is too stale to
            trust. 5.0 is generous; well past a typical `max_age` window
            in seconds but still a reasonable "this is probably still the
            same person" budget.

        max_plausible_speed_mps: world-coordinate speed ceiling for the
            position gate. 12.0 m/s is faster than any realistic sprint
            (Bolt's peak is ~12.2; professional match sprints top out
            around 8-9). Generous on purpose: re-ID gates are biased
            toward over-accepting, because the cost of a false positive
            (two players get merged) is recoverable, and the cost of a
            false negative (a player loses their identity across a zoom)
            is irrecoverable without a fresh detection.

        color_match_max_distance_bgr: BGR Euclidean distance threshold
            for the color gate. 60.0 is comfortably within "same team"
            intra-cluster variance; distinct team centroids typically
            sit >100 apart in BGR Euclidean distance.

        appearance_embedder: optional Callable invoked once per surviving
            candidate (after speed + color gates) as a future-pluggable
            tiebreaker. Signature: (new_obj, old_obj, new_color, old_color)
            -> similarity_score, where higher means more likely same
            identity. If the returned score is below
            `appearance_match_threshold`, the candidate is rejected.
            None means no tiebreaker (v1 default).

        appearance_match_threshold: minimum similarity score for the
            appearance embedder to accept a candidate. Only meaningful
            when `appearance_embedder` is set.
        """
        self._team_classifier = team_classifier
        self.min_reid_gap_frames = min_reid_gap_frames
        self.max_reid_gap_s = max_reid_gap_s
        self.max_plausible_speed_mps = max_plausible_speed_mps
        self.color_match_max_distance_bgr = color_match_max_distance_bgr
        self.appearance_embedder = appearance_embedder
        self.appearance_match_threshold = appearance_match_threshold

        self._history: Dict[int, _ReIdTrackHistory] = {}
        self._previous_track_ids: set = set()
        self._previous_timestamp_s: float = -1.0  # sentinel: "no previous frame"

    def reset(self) -> None:
        self._history = {}
        self._previous_track_ids = set()
        self._previous_timestamp_s = -1.0

    # ---- internals ----

    @staticmethod
    def _project_pixel(pixel_xy: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
        """Project a single (x, y) pixel point through a 3x3 homography.
        Returns the world (x, y) in meters. Caller is responsible for
        ensuring the homography is non-None first."""
        pt = np.array([pixel_xy], dtype=np.float32).reshape(1, 1, 2)
        world = cv2.perspectiveTransform(pt, homography.astype(np.float32))[0, 0]
        return (float(world[0]), float(world[1]))

    def _get_homography(self, calibration_info: Optional[CalibrationInfo]) -> Optional[np.ndarray]:
        """Extract the per-frame homography as a 3x3 ndarray, or None if
        calibration is unavailable this frame (NOT_CALIBRATED) or the
        field is None."""
        if calibration_info is None:
            return None
        h = calibration_info.homography
        if h is None:
            return None
        return np.array(h, dtype=np.float64)

    def _garbage_collect(self, now_s: float) -> None:
        """Drop history entries past `max_reid_gap_s` of age. Bounded
        cost: O(history size), and the size itself is bounded by
        `max_reid_gap_s * fps` because new entries can only be added
        when a track is alive OR is a recently-dropped candidate."""
        to_drop = [
            tid for tid, h in self._history.items()
            if (now_s - h.last_seen_timestamp_s) > self.max_reid_gap_s
        ]
        for tid in to_drop:
            del self._history[tid]

    def _get_old_color(self, old_history: _ReIdTrackHistory) -> Optional[np.ndarray]:
        """Pick the best available color for a dropped track: prefer the
        team classifier's recent cache (most-recent at the right end),
        fall back to the snapshot re-ID captured at the moment of drop."""
        cached = self._team_classifier.get_color_history(old_history.track_id)
        if cached is not None and len(cached) > 0:
            return cached[-1]  # most-recent
        return old_history.last_known_color

    def _get_new_color(self, new_obj: TrackedObject) -> Optional[np.ndarray]:
        """Pick the best available color for an incoming brand-new track.
        Source of truth: the team classifier's cache for this track_id,
        which was just populated by `assign_teams` earlier this frame.
        If empty (e.g. all-green crop), returns None -- the match is
        then doomed in the color gate, which is the right outcome."""
        cached = self._team_classifier.get_color_history(new_obj.track_id)
        if cached is not None and len(cached) > 0:
            return cached[-1]
        return None

    def _is_eligible(self, object_class: ObjectClass) -> bool:
        return object_class in REID_ELIGIBLE_CLASSES

    # ---- main per-frame entrypoint ----

    def update(
        self,
        tracked_objects: List[TrackedObject],
        calibration_info: Optional[CalibrationInfo],
        video_fps: float,
    ) -> List[TrackedObject]:
        # Compute the "now" timestamp for this frame. Re-ID doesn't have
        # access to a frame index; it derives a synthetic clock from
        # video_fps. First call: now_s = 0.0 (no previous frame).
        # Subsequent calls: now_s = previous + 1/fps.
        if self._previous_timestamp_s < 0:
            now_s = 0.0
        else:
            now_s = self._previous_timestamp_s + (1.0 / max(float(video_fps), 1e-6))

        # Step 1: garbage-collect stale history.
        self._garbage_collect(now_s)

        # Step 0: identify newly-spawned tracks.
        current_track_ids = {obj.track_id for obj in tracked_objects}
        if self._previous_timestamp_s < 0:
            # First call: every track in this frame is "new" by
            # construction (no previous frame to compare against).
            new_track_ids = current_track_ids
        else:
            new_track_ids = current_track_ids - self._previous_track_ids

        # Step 7 (partial): mark previously-alive tracks missing this
        # frame as "dropped". Done before step 0's matching loop so the
        # dropped set is ready.
        for tid, h in self._history.items():
            if not h.is_dropped and tid not in current_track_ids:
                h.is_dropped = True
                h.time_of_drop_s = h.last_seen_timestamp_s

        # Compute this frame's homography once; reused across candidates.
        H_now = self._get_homography(calibration_info)

        # Frame-count equivalent of "now", used only for the
        # min_reid_gap_frames floor below. Derived from the same
        # synthetic clock as now_s, so it stays consistent even though
        # re-ID has no real frame_index available.
        frame_interval_s = 1.0 / max(float(video_fps), 1e-6)

        # Steps 2-6: for each newly-spawned eligible track, try to match
        # against the dropped-history candidate set.
        for new_obj in tracked_objects:
            if new_obj.track_id not in new_track_ids:
                continue
            if not self._is_eligible(new_obj.object_class):
                continue

            # Step 2: candidates = dropped entries of matching class,
            # dropped at least `min_reid_gap_frames` ago (floor -- keeps
            # re-ID out of ByteTracker's own recovery window, see class
            # docstring), and still within the re-ID window (ceiling).
            # Linear scan; bounded by max_reid_gap_s * fps.
            candidates = [
                (tid, h) for tid, h in self._history.items()
                if h.is_dropped and h.object_class == new_obj.object_class
                and (now_s - h.last_seen_timestamp_s) <= self.max_reid_gap_s
                and (now_s - h.time_of_drop_s) / frame_interval_s >= self.min_reid_gap_frames
            ]

            if not candidates:
                continue

            # Compute the new track's world position for the speed gate.
            new_world_xy: Optional[Tuple[float, float]] = None
            if H_now is not None and new_obj.pixel_position is not None:
                new_world_xy = self._project_pixel(
                    (new_obj.pixel_position.x, new_obj.pixel_position.y), H_now
                )

            new_color = self._get_new_color(new_obj)

            best: Optional[Tuple[int, float, float]] = None  # (track_id, color_dist, dt)

            for old_tid, old_h in candidates:
                # Step 3: world-speed gate.
                if new_world_xy is not None and old_h.last_world_xy is not None:
                    dt = now_s - old_h.last_seen_timestamp_s
                    if dt <= 0:
                        continue
                    dx = new_world_xy[0] - old_h.last_world_xy[0]
                    dy = new_world_xy[1] - old_h.last_world_xy[1]
                    implied_speed = float(np.hypot(dx, dy)) / dt
                    if implied_speed > self.max_plausible_speed_mps:
                        continue
                else:
                    dt = now_s - old_h.last_seen_timestamp_s  # for tie-breaking

                # Step 4: color gate.
                old_color = self._get_old_color(old_h)
                if new_color is None or old_color is None:
                    continue
                color_dist = float(np.linalg.norm(new_color - old_color))
                if color_dist > self.color_match_max_distance_bgr:
                    continue

                # Optional step 4.5: appearance embedder tiebreaker.
                if self.appearance_embedder is not None:
                    score = self.appearance_embedder(new_obj, old_h, new_color, old_color)
                    if score < self.appearance_match_threshold:
                        continue

                # Surviving candidate. Track the best.
                if best is None or color_dist < best[1]:
                    best = (old_tid, color_dist, dt)
                elif color_dist == best[1] and dt < best[2]:
                    best = (old_tid, color_dist, dt)

            if best is None:
                continue

            # Step 6: re-stamp.
            old_tid = best[0]
            old_h = self._history[old_tid]
            new_obj.track_id = old_h.track_id
            new_obj.provenance = PositionProvenance.RE_IDENTIFIED_AFTER_GAP

            # Remove the consumed old entry, then add the re-linked
            # new_obj as "still alive" (will be refreshed again in the
            # step-7 loop below, but writing it here keeps the
            # invariant that the re-linked track is in history before
            # any later matching on this same frame).
            del self._history[old_tid]
            self._history[new_obj.track_id] = _ReIdTrackHistory(
                track_id=new_obj.track_id,
                object_class=new_obj.object_class,
                last_pixel_xy=(new_obj.pixel_position.x, new_obj.pixel_position.y),
                last_world_xy=new_world_xy,
                last_known_color=new_color,
                last_team=new_obj.team,
                last_seen_frame_index=-1,
                last_seen_timestamp_s=now_s,
                is_dropped=False,
            )

        # Step 7 (full): refresh history for every alive track this frame.
        for new_obj in tracked_objects:
            existing = self._history.get(new_obj.track_id)
            if existing is not None and not existing.is_dropped:
                existing.last_pixel_xy = (new_obj.pixel_position.x, new_obj.pixel_position.y)
                existing.last_team = new_obj.team
                existing.last_seen_timestamp_s = now_s
                existing.last_seen_frame_index = -1
                if H_now is not None:
                    existing.last_world_xy = self._project_pixel(
                        (new_obj.pixel_position.x, new_obj.pixel_position.y), H_now
                    )
                cached = self._team_classifier.get_color_history(new_obj.track_id)
                if cached is not None and len(cached) > 0:
                    existing.last_known_color = cached[-1]

        # Add brand-new eligible tracks that didn't get re-linked (and
        # didn't have a prior history entry) as "still alive".
        for new_obj in tracked_objects:
            if new_obj.track_id in self._history:
                continue
            if not self._is_eligible(new_obj.object_class):
                continue
            new_color_for_history = self._get_new_color(new_obj)
            new_world = None
            if H_now is not None:
                new_world = self._project_pixel(
                    (new_obj.pixel_position.x, new_obj.pixel_position.y), H_now
                )
            self._history[new_obj.track_id] = _ReIdTrackHistory(
                track_id=new_obj.track_id,
                object_class=new_obj.object_class,
                last_pixel_xy=(new_obj.pixel_position.x, new_obj.pixel_position.y),
                last_world_xy=new_world,
                last_known_color=new_color_for_history,
                last_team=new_obj.team,
                last_seen_frame_index=-1,
                last_seen_timestamp_s=now_s,
                is_dropped=False,
            )

        self._previous_track_ids = current_track_ids
        self._previous_timestamp_s = now_s
        return tracked_objects