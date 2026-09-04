"""
Multi-object tracking (Phase 2): turns a per-frame stream of `Detection`s
(no identity) into `TrackedObject`s (stable `track_id`s across frames).

Scope, on purpose: this class ONLY answers "which detection this frame is
the same physical object as a detection from a previous frame." It never
fabricates a position for a frame where a track wasn't actually matched --
a missed track is simply absent from that frame's output, not interpolated.
Bridging longer gaps (re-identification after a camera zoom loses
everyone, per the project plan's "Handling long tracking gaps" section) is
a deliberately separate, later piece of Phase 2 -- it needs team/jersey
color and appearance matching this class knows nothing about, and mixing
that logic in here would make both halves harder to test independently.

Association strategy (IoU-based, ByteTrack-style two-stage, no scipy/
Hungarian algorithm -- consistent with the project's minimal-dependency
philosophy):

  1. Group active tracks and this frame's detections by `object_class` --
     a goalkeeper detection can never match a player's track, full stop.
  2. Within each class group, match HIGH-confidence detections to tracks
     via greedy IoU (highest-IoU pairs claimed first, no reuse).
  3. Whatever tracks are STILL unmatched get a second chance against
     LOW-confidence detections in that same class -- this is the ByteTrack
     insight: a real object that's motion-blurred or partially occluded
     often only clears a low confidence threshold, but its position is
     still informative enough to keep the track alive instead of losing
     it and eventually respawning a new ID for the same player.
  4. Any HIGH-confidence detection still unmatched after that starts a
     brand-new track. LOW-confidence detections never spawn new tracks --
     they're only trusted to extend an identity that's already
     established, not to originate one.
  5. Tracks unmatched for more than `max_age` consecutive frames are
     dropped.

One deliberate simplification vs. textbook ByteTrack: matching against a
track uses its LAST-KNOWN bounding box as-is, with no Kalman-filter motion
prediction. Good enough to bridge a handful of frames of blur/occlusion at
broadcast frame rates (players don't teleport frame-to-frame); genuinely
long gaps (a multi-second camera zoom) are exactly the case the project
plan hands off to re-identification instead, not this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from ..schemas.enums import ObjectClass, PositionProvenance
from ..schemas.frame_result import BoundingBox, TrackedObject
from .detection import Detection


class BaseTracker(ABC):
    """Interface every tracker implementation must satisfy."""

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Advance the tracker by one frame. `detections` are this frame's
        raw detector output (no identity). Returns the `TrackedObject`s
        for tracks that were actually matched THIS frame -- a track that
        wasn't matched is simply not in the returned list."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clears all track state, starting fresh as if no frames had been
        processed yet -- e.g. after a hard scene cut, where continuing to
        match against pre-cut tracks would be actively wrong."""
        raise NotImplementedError


class _Track:
    """Internal, mutable per-track state. Deliberately NOT a pydantic
    model -- this is the tracker's private bookkeeping (hit counts, age,
    last-matched bbox), not part of any cross-stage contract. Only
    `ByteTracker.update()` ever turns one of these into the public
    `TrackedObject` schema, and only once it's confirmed."""

    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.object_class = detection.object_class
        self.bounding_box = detection.bounding_box
        self.pixel_position = detection.pixel_position
        self.confidence = detection.confidence
        self.hits = 1
        self.time_since_update = 0

    def update_with_detection(self, detection: Detection) -> None:
        self.bounding_box = detection.bounding_box
        self.pixel_position = detection.pixel_position
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1


class ByteTracker(BaseTracker):
    def __init__(
        self,
        high_conf_threshold: float = 0.6,
        low_conf_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        """
        high_conf_threshold: detections at/above this are eligible both to
            match existing tracks in stage 1 AND to spawn new tracks.
        low_conf_threshold: detections in [low_conf_threshold,
            high_conf_threshold) are only used in stage 2, to keep an
            already-established track alive -- never to spawn a new one.
            Detections below this are ignored entirely (pure noise).
        iou_threshold: minimum IoU for a track/detection pair to be
            considered a candidate match at all.
        max_age: consecutive missed frames before a track is dropped.
        min_hits: consecutive matched frames (including creation) required
            before a track is confirmed and starts appearing in output --
            filters out single-frame spurious detections spawning IDs that
            immediately vanish.
        """
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: List[_Track] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 1

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        xx1 = max(a.x1, b.x1)
        yy1 = max(a.y1, b.y1)
        xx2 = min(a.x2, b.x2)
        yy2 = min(a.y2, b.y2)

        inter_w = max(0.0, xx2 - xx1)
        inter_h = max(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
        area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
        union = area_a + area_b - inter_area

        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _greedy_iou_match(
        tracks: List[_Track],
        detections: List[Detection],
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy (not globally optimal) IoU association -- highest-IoU
        candidate pairs are claimed first, and once a track or detection
        is claimed it's removed from further consideration. Deliberately
        not Hungarian-algorithm-optimal, consistent with the project's
        no-scipy dependency rule; for per-frame association with small
        inter-frame displacement this is sufficient in practice.

        Returns (matches, unmatched_track_indices, unmatched_detection_indices),
        all indices local to the `tracks`/`detections` lists passed in.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        candidates = []
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                iou = ByteTracker._iou(track.bounding_box, det.bounding_box)
                if iou >= iou_threshold:
                    candidates.append((iou, t_idx, d_idx))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_tracks, matched_dets = set(), set()
        matches: List[Tuple[int, int]] = []
        for _iou_val, t_idx, d_idx in candidates:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matches.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        classes = {t.object_class for t in self._tracks} | {d.object_class for d in detections}

        matched_track_idx: set = set()
        matched_det_idx: set = set()
        all_matches: List[Tuple[int, int]] = []  # (global track idx, global detection idx)

        for cls in classes:
            cls_track_idx = [i for i, t in enumerate(self._tracks) if t.object_class == cls]
            high_det_idx = [
                i for i, d in enumerate(detections)
                if d.object_class == cls and d.confidence >= self.high_conf_threshold
            ]

            cls_tracks = [self._tracks[i] for i in cls_track_idx]
            high_dets = [detections[i] for i in high_det_idx]

            matches, unmatched_local_tracks, _unmatched_high = self._greedy_iou_match(
                cls_tracks, high_dets, self.iou_threshold
            )
            for t_local, d_local in matches:
                t_global, d_global = cls_track_idx[t_local], high_det_idx[d_local]
                all_matches.append((t_global, d_global))
                matched_track_idx.add(t_global)
                matched_det_idx.add(d_global)

            # Stage 2: tracks stage 1 left unmatched get a second chance
            # against this class's low-confidence detections.
            remaining_track_idx = [cls_track_idx[i] for i in unmatched_local_tracks]
            low_det_idx = [
                i for i, d in enumerate(detections)
                if d.object_class == cls and self.low_conf_threshold <= d.confidence < self.high_conf_threshold
            ]
            remaining_tracks = [self._tracks[i] for i in remaining_track_idx]
            low_dets = [detections[i] for i in low_det_idx]

            matches2, _unmatched2, _unused = self._greedy_iou_match(
                remaining_tracks, low_dets, self.iou_threshold
            )
            for t_local, d_local in matches2:
                t_global, d_global = remaining_track_idx[t_local], low_det_idx[d_local]
                all_matches.append((t_global, d_global))
                matched_track_idx.add(t_global)
                matched_det_idx.add(d_global)

        for t_global, d_global in all_matches:
            self._tracks[t_global].update_with_detection(detections[d_global])

        for i, track in enumerate(self._tracks):
            if i not in matched_track_idx:
                track.mark_missed()

        # New tracks ONLY from unmatched high-confidence detections --
        # a low-confidence detection is never trusted to originate an ID.
        for d_idx, det in enumerate(detections):
            if d_idx in matched_det_idx:
                continue
            if det.confidence < self.high_conf_threshold:
                continue
            self._tracks.append(_Track(self._next_track_id, det))
            self._next_track_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        results: List[TrackedObject] = []
        for track in self._tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                results.append(TrackedObject(
                    track_id=track.track_id,
                    object_class=track.object_class,
                    pixel_position=track.pixel_position,
                    bounding_box=track.bounding_box,
                    detection_confidence=track.confidence,
                    provenance=PositionProvenance.OBSERVED,
                ))
        return results