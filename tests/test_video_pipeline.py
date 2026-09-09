import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus, ObjectClass, Team, PositionProvenance
from ..schemas.frame_result import CalibrationInfo, PixelPoint, TrackedObject
from ..schemas.frame_result import BoundingBox, CalibrationInfo, PixelPoint, TrackedObject
from ..calibration.orchestrator import OrchestratedFrameResult
from ..calibration.scene_classifier import SceneClassification
from ..calibration.scene_types import SceneType
from ..tracking.detection import Detection
from ..tracking.detector import BaseObjectDetector
from ..tracking.tracker import BaseTracker
from ..tracking.team_classifier import BaseTeamClassifier
from ..tracking.reid import BaseReIdentifier
from ..video_pipeline import MatchPipeline, PipelineFrameOutput


# ---- synthetic homography, same convention as test_calibrator.py /
# test_transform.py: 10 px per meter, offset origin ----

SCALE = 10.0
OFFSET = (100.0, 50.0)


def _pixel_for_world(world_xy):
    x, y = world_xy
    return (x * SCALE + OFFSET[0], y * SCALE + OFFSET[1])


def _synthetic_homography():
    return [
        [1.0 / SCALE, 0.0, -OFFSET[0] / SCALE],
        [0.0, 1.0 / SCALE, -OFFSET[1] / SCALE],
        [0.0, 0.0, 1.0],
    ]


def _tracked_object(track_id, world_xy, object_class=ObjectClass.PLAYER):
    px, py = _pixel_for_world(world_xy)
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        pixel_position=PixelPoint(x=px, y=py),
        provenance=PositionProvenance.OBSERVED,
    )


# ---- fakes for every injected component, each appending to a shared
# call_log so tests can assert the exact call order the pipeline uses ----

class FakeDetector(BaseObjectDetector):
    def __init__(self, call_log, detections):
        self.call_log = call_log
        self._detections = detections
        self.received_frame = None

    def detect(self, frame):
        self.call_log.append("detector")
        self.received_frame = frame
        return self._detections


class FakeTracker(BaseTracker):
    def __init__(self, call_log, tracked_objects):
        self.call_log = call_log
        self._tracked_objects = tracked_objects
        self.received_detections = None
        self.reset_count = 0

    def update(self, detections):
        self.call_log.append("tracker")
        self.received_detections = detections
        return self._tracked_objects

    def reset(self):
        self.reset_count += 1


class FakeTeamClassifier(BaseTeamClassifier):
    """Mutates every PLAYER to Team.HOME -- lets tests confirm this
    mutation is visible to re-ID, which runs immediately after."""

    def __init__(self, call_log):
        self.call_log = call_log
        self.received_tracked_objects = None

    def assign_teams(self, frame, tracked_objects):
        self.call_log.append("team_classifier")
        for obj in tracked_objects:
            if obj.object_class == ObjectClass.PLAYER:
                obj.team = Team.HOME
        self.received_tracked_objects = tracked_objects
        return tracked_objects


class FakeReIdentifier(BaseReIdentifier):
    """Optionally re-stamps the first object's track_id, so tests can
    confirm a re-ID relink survives all the way into the final
    FrameResult. Also records exactly what it was handed, so tests can
    confirm the calibration_info/video_fps handoff."""

    def __init__(self, call_log, restamp_first_track_id_to=None):
        self.call_log = call_log
        self._restamp_to = restamp_first_track_id_to
        self.received_calibration_info = None
        self.received_video_fps = None
        self.received_teams = None
        self.reset_count = 0

    def update(self, tracked_objects, calibration_info, video_fps):
        self.call_log.append("reid")
        self.received_calibration_info = calibration_info
        self.received_video_fps = video_fps
        # Confirms team classification already ran: capture .team values
        # visible to re-ID at the moment it's called.
        self.received_teams = [obj.team for obj in tracked_objects]

        if self._restamp_to is not None and tracked_objects:
            tracked_objects[0].track_id = self._restamp_to
            tracked_objects[0].provenance = PositionProvenance.RE_IDENTIFIED_AFTER_GAP

        return tracked_objects

    def reset(self):
        self.reset_count += 1


class FakeCalibrationOrchestrator:
    """Not a subclass of VideoCalibrationOrchestrator -- MatchPipeline
    only relies on `process_frame(frame)` returning an
    OrchestratedFrameResult-shaped object, so a plain duck-typed fake is
    enough and keeps this test independent of the real calibration
    pipeline's internals."""

    def __init__(self, call_log, orchestrated_result):
        self.call_log = call_log
        self._result = orchestrated_result
        self.received_frame = None

    def process_frame(self, frame):
        self.call_log.append("calibration")
        self.received_frame = frame
        return self._result


def _build_pipeline(call_log, tracked_objects, calibration_status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
                     frames_since_last_anchor=0, restamp_first_track_id_to=None):
    detections = [Detection(
        object_class=ObjectClass.PLAYER,
        confidence=0.9,
        bounding_box=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        pixel_position=PixelPoint(x=0.0, y=0.0),
    )]

    calibration_info = CalibrationInfo(
        status=calibration_status,
        homography=_synthetic_homography() if calibration_status != CalibrationStatus.NOT_CALIBRATED else None,
        frames_since_last_anchor=frames_since_last_anchor,
    )
    scene_classification = SceneClassification(scene_type=SceneType.BROADCAST_WIDE, confidence=0.9)
    orchestrated_result = OrchestratedFrameResult(
        scene_classification=scene_classification,
        calibration_info=calibration_info,
        keypoints_px=np.zeros((29, 2), dtype=np.float32),
        keypoint_confidences=np.ones((29,), dtype=np.float32),
    )

    detector = FakeDetector(call_log, detections)
    tracker = FakeTracker(call_log, tracked_objects)
    team_classifier = FakeTeamClassifier(call_log)
    reid = FakeReIdentifier(call_log, restamp_first_track_id_to=restamp_first_track_id_to)
    orchestrator = FakeCalibrationOrchestrator(call_log, orchestrated_result)

    pipeline = MatchPipeline(
        detector=detector,
        tracker=tracker,
        team_classifier=team_classifier,
        reid=reid,
        calibration_orchestrator=orchestrator,
        pitch_config=PitchConfig(),
        video_fps=25.0,
    )
    return pipeline, dict(
        detector=detector, tracker=tracker, team_classifier=team_classifier,
        reid=reid, orchestrator=orchestrator, calibration_info=calibration_info,
    )


def test_calls_components_in_correct_order():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, _ = _build_pipeline(call_log, tracked_objects)

    pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    assert call_log == ["calibration", "detector", "tracker", "team_classifier", "reid"]


def test_calibration_info_is_forwarded_to_reid():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, parts = _build_pipeline(call_log, tracked_objects)

    pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    assert parts["reid"].received_calibration_info is parts["calibration_info"]


def test_video_fps_is_forwarded_to_reid():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, parts = _build_pipeline(call_log, tracked_objects)

    pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    assert parts["reid"].received_video_fps == 25.0


def test_team_classification_is_visible_to_reid():
    """Proves the load-bearing ordering constraint: re-ID must see teams
    already assigned, since it runs after team_classifier."""
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0), object_class=ObjectClass.PLAYER)]
    pipeline, parts = _build_pipeline(call_log, tracked_objects)

    pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    assert parts["reid"].received_teams == [Team.HOME]


def test_reid_restamped_track_id_survives_into_final_frame_result():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, _ = _build_pipeline(call_log, tracked_objects, restamp_first_track_id_to=999)

    output = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    obj = output.frame_result.tracked_objects[0]
    assert obj.track_id == 999
    assert obj.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP


def test_world_position_is_computed_via_real_transform():
    """transform_tracked_objects is NOT faked -- this confirms the real
    Phase 3 function actually runs on calibration's real homography."""
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, _ = _build_pipeline(call_log, tracked_objects)

    output = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    obj = output.frame_result.tracked_objects[0]
    assert obj.world_position is not None
    assert obj.world_position.x == pytest.approx(52.5, abs=0.01)
    assert obj.world_position.y == pytest.approx(34.0, abs=0.01)
    assert obj.position_confidence == pytest.approx(1.0)


def test_propagated_confidence_decay_params_are_forwarded():
    """Confirms the pipeline's confidence-policy constructor args actually
    reach transform_tracked_objects, using a custom decay rate that would
    produce a different confidence than the default."""
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, _ = _build_pipeline(
        call_log, tracked_objects,
        calibration_status=CalibrationStatus.PROPAGATED,
        frames_since_last_anchor=5,
    )
    pipeline.propagated_decay_per_frame = 0.1  # override for this test

    output = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    obj = output.frame_result.tracked_objects[0]
    assert obj.position_confidence == pytest.approx(1.0 - 0.1 * 5)


def test_not_calibrated_frame_yields_no_world_position():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, _ = _build_pipeline(
        call_log, tracked_objects, calibration_status=CalibrationStatus.NOT_CALIBRATED,
    )

    output = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    obj = output.frame_result.tracked_objects[0]
    assert obj.world_position is None
    assert obj.position_confidence is None


def test_frame_result_uses_caller_supplied_frame_index_and_timestamp():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, _ = _build_pipeline(call_log, tracked_objects)

    output = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=42, timestamp_s=1.68)

    assert output.frame_result.frame_index == 42
    assert output.frame_result.timestamp_s == pytest.approx(1.68)


def test_pipeline_frame_output_carries_debug_fields():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, parts = _build_pipeline(call_log, tracked_objects)

    output = pipeline.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), frame_index=0, timestamp_s=0.0)

    assert isinstance(output, PipelineFrameOutput)
    assert output.scene_classification.scene_type == SceneType.BROADCAST_WIDE
    assert output.keypoints_px is not None
    assert output.keypoint_confidences is not None


def test_reset_resets_tracker_and_reid_only():
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, parts = _build_pipeline(call_log, tracked_objects)

    pipeline.reset()

    assert parts["tracker"].reset_count == 1
    assert parts["reid"].reset_count == 1
    # Neither the fake detector, team classifier, nor calibration
    # orchestrator expose a reset() at all in this test -- reset() must
    # not raise trying to call one that doesn't exist.


def test_reset_does_not_call_calibration_orchestrator():
    """Explicitly locks in the documented design decision: calibration
    self-corrects via RE_ANCHORED and is deliberately NOT reset here."""
    call_log = []
    tracked_objects = [_tracked_object(1, (52.5, 34.0))]
    pipeline, parts = _build_pipeline(call_log, tracked_objects)

    # Attach a reset() that would fail the test if ever called.
    def _should_not_be_called():
        raise AssertionError("calibration_orchestrator.reset() should never be called")
    parts["orchestrator"].reset = _should_not_be_called

    pipeline.reset()  # must not raise