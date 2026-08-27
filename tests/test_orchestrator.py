import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..calibration.scene_types import SceneType
from ..calibration.keypoint_model import BaseKeypointModel
from ..calibration.model_registry import KeypointModelRegistry
from ..calibration.calibrator import PitchCalibrator
from ..calibration.pipeline import CalibrationPipeline
from ..calibration.propagating_calibrator import PropagatingCalibrator
from ..calibration.orchestrator import VideoCalibrationOrchestrator


FRAME_SHAPE = (1080, 1920, 3)


class FixedOutputModel(BaseKeypointModel):
    def __init__(self, keypoints_px, confidences):
        self._keypoints_px = keypoints_px
        self._confidences = confidences
        self.call_count = 0

    def predict(self, frame):
        self.call_count += 1
        return self._keypoints_px, self._confidences


class CountingPitchCalibrator:
    """Wraps a real PitchCalibrator and counts calls -- lets tests prove
    calibrate_frame was (or wasn't) invoked for a given frame, which is
    exactly what matters for verifying the CLOSE_UP -> None translation
    actually prevents a wasted/wrong calibration attempt."""
    def __init__(self, real):
        self._real = real
        self.call_count = 0

    def calibrate_frame(self, keypoints_px, confidences):
        self.call_count += 1
        return self._real.calibrate_frame(keypoints_px, confidences)

    def pixel_to_world(self, pixel_xy):
        return self._real.pixel_to_world(pixel_xy)


class FakeMotionTracker:
    def __init__(self, homographies):
        self._homographies = list(homographies)
        self.reset_count = 0

    def estimate_motion(self, frame):
        return self._homographies.pop(0)

    def reset(self):
        self.reset_count += 1


def _kps_confs(visible: dict, default_conf: float = 0.9):
    kps = np.zeros((29, 2), dtype=np.float32)
    confs = np.full((29,), 0.05, dtype=np.float32)
    for idx, (x, y) in visible.items():
        kps[idx] = (x, y)
        confs[idx] = default_conf
    return kps, confs


BROADCAST_PATTERN = _kps_confs({
    0: (50, 950), 16: (1870, 950), 9: (50, 100), 25: (1870, 100),
    11: (960, 950), 12: (960, 100), 15: (960, 525),
    13: (960, 700), 14: (960, 350),
})

CLOSE_UP_PATTERN = _kps_confs({2: (900, 500), 4: (910, 520)})


def _build_orchestrator(broadcast_model, motion_homographies):
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "broadcast.pt"},
        model_loader=lambda path: broadcast_model,
    )
    cfg = PitchConfig()
    pipeline = CalibrationPipeline(cfg, registry)
    counting_calibrator = CountingPitchCalibrator(PitchCalibrator(cfg))
    fake_tracker = FakeMotionTracker(motion_homographies)
    propagating = PropagatingCalibrator(counting_calibrator, fake_tracker)
    orchestrator = VideoCalibrationOrchestrator(pipeline, propagating)
    return orchestrator, counting_calibrator, fake_tracker


def test_broadcast_frame_calibrates_through_the_full_chain():
    broadcast_model = FixedOutputModel(*BROADCAST_PATTERN)
    orchestrator, counting_calibrator, _ = _build_orchestrator(broadcast_model, [None])

    result = orchestrator.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))

    assert result.scene_classification.scene_type == SceneType.BROADCAST_WIDE
    assert result.calibration_info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert counting_calibrator.call_count == 1


def test_close_up_never_reaches_calibrate_frame():
    """The core thing this orchestrator exists to guarantee: a CLOSE_UP
    frame's keypoints (however garbage) must never be handed to
    calibrate_frame at all -- PropagatingCalibrator should go straight to
    motion propagation (or NOT_CALIBRATED, with no prior anchor)."""
    broadcast_model = FixedOutputModel(*CLOSE_UP_PATTERN)
    orchestrator, counting_calibrator, _ = _build_orchestrator(broadcast_model, [None])

    result = orchestrator.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))

    assert result.scene_classification.scene_type == SceneType.CLOSE_UP
    assert result.calibration_info.status == CalibrationStatus.NOT_CALIBRATED
    assert counting_calibrator.call_count == 0  # never invoked
    # the raw (garbage) keypoints are still surfaced for debugging/overlay
    assert result.keypoints_px is not None


def test_full_sequence_broadcast_gap_reanchor():
    """End-to-end proof: broadcast frame anchors, a close-up gap
    propagates, and a broadcast frame reappearing re-anchors -- routing
    decisions and the state machine working together correctly."""
    broadcast_model = FixedOutputModel(*BROADCAST_PATTERN)
    identity = np.eye(3)
    orchestrator, counting_calibrator, fake_tracker = _build_orchestrator(
        broadcast_model, [None, identity, identity]
    )

    result1 = orchestrator.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))
    assert result1.calibration_info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS

    # Switch the registered model's output to a close-up pattern for frame 2
    broadcast_model._keypoints_px, broadcast_model._confidences = CLOSE_UP_PATTERN
    result2 = orchestrator.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))
    assert result2.scene_classification.scene_type == SceneType.CLOSE_UP
    assert result2.calibration_info.status == CalibrationStatus.PROPAGATED

    # Switch back to broadcast pattern for frame 3 -- should re-anchor
    broadcast_model._keypoints_px, broadcast_model._confidences = BROADCAST_PATTERN
    result3 = orchestrator.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))
    assert result3.scene_classification.scene_type == SceneType.BROADCAST_WIDE
    assert result3.calibration_info.status == CalibrationStatus.RE_ANCHORED

    # calibrate_frame was invoked for frames 1 and 3 only, never frame 2
    assert counting_calibrator.call_count == 2
    assert fake_tracker.reset_count == 2  # once on initial anchor, once on re-anchor