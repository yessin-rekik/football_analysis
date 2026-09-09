"""
Phase 2 + Phase 3 integration -- the single per-frame entrypoint for real
video use, tying together everything built so far:

    detector.detect(frame)              -> List[Detection]
    tracker.update(detections)          -> List[TrackedObject]
    team_clf.assign_teams(frame, t)     -> same list, .team filled in
    calibration_orchestrator.process_frame(frame) -> OrchestratedFrameResult
    reid.update(t, calibration_info, fps) -> re-stamps track_id/provenance
                                              where a returning player is
                                              recognised
    transform_tracked_objects(t, calibration_info, pitch_config)
                                         -> world_position/position_confidence
                                            filled in
    FrameResult(...)                    -> the canonical Phase 0 output

This ordering is not arbitrary -- it's the exact sequence documented in
`tracking/reid.py`'s own docstring, and it's load-bearing: re-ID reads the
team classifier's per-track color cache (`get_color_history`), so team
classification MUST run first; re-ID's world-speed gate needs this frame's
homography, so calibration MUST run before re-ID; and the Phase 3
transform is last because it's the one step that turns pixel-space
identity/team data into the fully-populated `TrackedObject`s the canonical
schema expects.

Same seam pattern as `calibration/orchestrator.py`, just one layer up:
each component (detector, tracker, team classifier, re-identifier,
calibration orchestrator) stays independently testable, and this class's
only job is calling them in the right order with the right handoffs --
never reimplementing any of their logic.

frame_index / timestamp_s are supplied by the CALLER on every
process_frame() call, not tracked internally. A real video loop already
knows which frame it's on; an internal counter would silently desync from
reality the moment a frame gets skipped upstream (e.g. a frame-sampling
step that processes every 2nd frame) -- exactly the kind of silent
staleness this project has avoided everywhere else (calibration state,
CSV export). The tradeoff: `timestamp_s` passed in here should stay
consistent with the fps used to construct `reid`, since `ReIdentifier`
keeps its own internal synthetic clock derived from `video_fps` rather
than from `frame_index` -- this class does not (and cannot) reconcile the
two if they drift, since `reid.update()`'s interface doesn't accept a
frame index at all.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .calibration.orchestrator import VideoCalibrationOrchestrator
from .calibration.scene_classifier import SceneClassification
from .config.pitch_config import PitchConfig
from .coordinates.transform import transform_tracked_objects
from .schemas.frame_result import FrameResult
from .tracking.detector import BaseObjectDetector
from .tracking.reid import BaseReIdentifier
from .tracking.team_classifier import BaseTeamClassifier
from .tracking.tracker import BaseTracker


@dataclass
class PipelineFrameOutput:
    """Wraps the canonical FrameResult alongside calibration debug info
    (scene classification, raw keypoints) that overlay/debugging tools
    may want but that has no place in the canonical schema -- same
    reasoning as `OrchestratedFrameResult` keeping this data separate
    from `CalibrationInfo`."""
    frame_result: FrameResult
    scene_classification: SceneClassification
    keypoints_px: Optional[np.ndarray] = None
    keypoint_confidences: Optional[np.ndarray] = None


class MatchPipeline:
    def __init__(
        self,
        detector: BaseObjectDetector,
        tracker: BaseTracker,
        team_classifier: BaseTeamClassifier,
        reid: BaseReIdentifier,
        calibration_orchestrator: VideoCalibrationOrchestrator,
        pitch_config: PitchConfig,
        video_fps: float,
        *,
        re_anchored_confidence: float = 0.85,
        propagated_confidence_floor: float = 0.3,
        propagated_decay_per_frame: float = 0.02,
        out_of_bounds_margin: float = 5.0,
    ):
        """
        video_fps: passed to `reid.update()` every frame -- must match
            the fps `reid` was tuned against (its gap-timing math derives
            directly from this value, not from frame_index).
        re_anchored_confidence / propagated_confidence_floor /
            propagated_decay_per_frame / out_of_bounds_margin: forwarded
            unchanged to `transform_tracked_objects()` every frame. Kept
            as constructor params (not hardcoded) so a caller can tune
            them without reaching into this class's internals -- same
            inspection-tuned-default caveat as everywhere else in this
            project.
        """
        self.detector = detector
        self.tracker = tracker
        self.team_classifier = team_classifier
        self.reid = reid
        self.calibration_orchestrator = calibration_orchestrator
        self.pitch_config = pitch_config
        self.video_fps = video_fps

        self.re_anchored_confidence = re_anchored_confidence
        self.propagated_confidence_floor = propagated_confidence_floor
        self.propagated_decay_per_frame = propagated_decay_per_frame
        self.out_of_bounds_margin = out_of_bounds_margin

    def process_frame(
        self, frame: np.ndarray, frame_index: int, timestamp_s: float,
    ) -> PipelineFrameOutput:
        # Calibration runs first -- re-ID's speed gate needs this frame's
        # homography, and it has no way to ask for it later.
        orchestrated = self.calibration_orchestrator.process_frame(frame)
        calibration_info = orchestrated.calibration_info

        detections = self.detector.detect(frame)
        tracked_objects = self.tracker.update(detections)

        # Team classification before re-ID -- re-ID reads the color cache
        # this call populates (`get_color_history`), for both the
        # incoming new track and whatever dropped track it's compared
        # against.
        tracked_objects = self.team_classifier.assign_teams(frame, tracked_objects)

        tracked_objects = self.reid.update(tracked_objects, calibration_info, self.video_fps)

        tracked_objects = transform_tracked_objects(
            tracked_objects,
            calibration_info,
            self.pitch_config,
            re_anchored_confidence=self.re_anchored_confidence,
            propagated_confidence_floor=self.propagated_confidence_floor,
            propagated_decay_per_frame=self.propagated_decay_per_frame,
            out_of_bounds_margin=self.out_of_bounds_margin,
        )

        frame_result = FrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            calibration=calibration_info,
            tracked_objects=tracked_objects,
        )

        return PipelineFrameOutput(
            frame_result=frame_result,
            scene_classification=orchestrated.scene_classification,
            keypoints_px=orchestrated.keypoints_px,
            keypoint_confidences=orchestrated.keypoint_confidences,
        )

    def reset(self) -> None:
        """Resets every stateful component that has a `reset()` -- for a
        hard scene cut, where carrying tracker IDs, re-ID history, or
        camera-motion tracking state across the cut would be actively
        wrong. Uses duck-typed `getattr(..., "reset", None)`, same
        pattern as `PropagatingCalibrator`'s calibrator-reset handling --
        not every injected component is guaranteed to expose one (e.g. a
        stateless detector never needs it), so this skips silently rather
        than requiring every component to implement a no-op.

        Note: `calibration_orchestrator` is NOT reset here. It has no
        public `reset()` of its own -- its internal state (via
        `PropagatingCalibrator`) already self-corrects the moment fresh
        keypoints are seen again (RE_ANCHORED), which is the right
        behavior across a cut too. If that ever changes, this is the
        place to add it.
        """
        for component in (self.tracker, self.reid):
            reset_fn = getattr(component, "reset", None)
            if reset_fn is not None:
                reset_fn()