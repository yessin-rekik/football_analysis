import numpy as np
import pytest

from ..config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..calibration.scene_types import SceneType
from ..calibration.keypoint_model import BaseKeypointModel
from ..calibration.model_registry import KeypointModelRegistry
from ..calibration.pipeline import CalibrationPipeline


FRAME_SHAPE = (1080, 1920, 3)


class FixedOutputModel(BaseKeypointModel):
    """A fake model that always returns the same, test-defined keypoints --
    lets us script exactly what 'the broadcast model' vs 'the low-angle
    model' sees, and count how many times each was actually invoked."""

    def __init__(self, keypoints_px: np.ndarray, confidences: np.ndarray):
        self._keypoints_px = keypoints_px
        self._confidences = confidences
        self.call_count = 0

    def predict(self, frame):
        self.call_count += 1
        return self._keypoints_px, self._confidences


def _kps_confs(visible: dict, default_conf: float = 0.9):
    kps = np.zeros((29, 2), dtype=np.float32)
    confs = np.full((29,), 0.05, dtype=np.float32)
    for idx, (x, y) in visible.items():
        kps[idx] = (x, y)
        confs[idx] = default_conf
    return kps, confs


# Same broadcast-wide pattern as test_scene_classifier.py
BROADCAST_PATTERN = _kps_confs({
    0: (50, 950), 16: (1870, 950), 9: (50, 100), 25: (1870, 100),
    11: (960, 950), 12: (960, 100), 15: (960, 525),
    13: (960, 700), 14: (960, 350),
})

# Same corner pattern as test_scene_classifier.py: box/boundary visible, no center
CORNER_PATTERN_FROM_BROADCAST_MODEL = _kps_confs({
    9: (100, 900), 0: (100, 150),
    1: (700, 150), 2: (750, 300),
    3: (700, 900), 4: (750, 750),
})

# A fuller, higher-quality set the (hypothetical) low-angle model would
# produce for the same corner shot
CORNER_PATTERN_FROM_LOW_ANGLE_MODEL = _kps_confs({
    9: (100, 900), 0: (100, 150),
    1: (700, 150), 2: (750, 300),
    3: (700, 900), 4: (750, 750),
    16: (1800, 150), 25: (1800, 900),
})

CLOSE_UP_PATTERN = _kps_confs({2: (900, 500), 4: (910, 520)})  # only 2 visible


def _registry_with_loader(paths_to_models: dict):
    def loader(path):
        return paths_to_models[path]
    return loader


def test_broadcast_frame_uses_broadcast_model_only():
    broadcast_model = FixedOutputModel(*BROADCAST_PATTERN)
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "broadcast.pt"},
        model_loader=_registry_with_loader({"broadcast.pt": broadcast_model}),
    )
    pipeline = CalibrationPipeline(PitchConfig(), registry)

    result = pipeline.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))

    assert result.scene_classification.scene_type == SceneType.BROADCAST_WIDE
    assert result.calibration_info.status == CalibrationStatus.CALIBRATED_FROM_KEYPOINTS
    assert broadcast_model.call_count == 1


def test_close_up_skips_calibration_and_second_model_call():
    broadcast_model = FixedOutputModel(*CLOSE_UP_PATTERN)
    low_angle_model = FixedOutputModel(*BROADCAST_PATTERN)  # should never be used
    registry = KeypointModelRegistry(
        model_paths={
            SceneType.BROADCAST_WIDE: "broadcast.pt",
            SceneType.LOW_ANGLE_CORNER: "low_angle.pt",
        },
        model_loader=_registry_with_loader({
            "broadcast.pt": broadcast_model, "low_angle.pt": low_angle_model,
        }),
    )
    pipeline = CalibrationPipeline(PitchConfig(), registry)

    result = pipeline.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))

    assert result.scene_classification.scene_type == SceneType.CLOSE_UP
    assert result.calibration_info.status == CalibrationStatus.NOT_CALIBRATED
    assert broadcast_model.call_count == 1
    assert low_angle_model.call_count == 0  # never invoked -- no wasted inference


def test_corner_frame_reruns_with_low_angle_model_when_available():
    broadcast_model = FixedOutputModel(*CORNER_PATTERN_FROM_BROADCAST_MODEL)
    low_angle_model = FixedOutputModel(*CORNER_PATTERN_FROM_LOW_ANGLE_MODEL)
    registry = KeypointModelRegistry(
        model_paths={
            SceneType.BROADCAST_WIDE: "broadcast.pt",
            SceneType.LOW_ANGLE_CORNER: "low_angle.pt",
        },
        model_loader=_registry_with_loader({
            "broadcast.pt": broadcast_model, "low_angle.pt": low_angle_model,
        }),
    )
    pipeline = CalibrationPipeline(PitchConfig(), registry)

    result = pipeline.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))

    assert result.scene_classification.scene_type == SceneType.LOW_ANGLE_CORNER
    assert broadcast_model.call_count == 1
    assert low_angle_model.call_count == 1
    # calibration must reflect the LOW-ANGLE model's output (8 points), not
    # the broadcast model's original guess (6 points) -- proves the rerun
    # result was actually used, not just computed and discarded
    assert result.calibration_info.num_keypoints_used == 8


def test_corner_frame_without_low_angle_model_does_not_double_call():
    """The exact situation you're in right now: no low-angle model trained
    yet. The pipeline should fall back to using the broadcast model's
    (imperfect) corner-scene output rather than calling the same model
    twice on identical input."""
    broadcast_model = FixedOutputModel(*CORNER_PATTERN_FROM_BROADCAST_MODEL)
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "broadcast.pt"},
        # LOW_ANGLE_CORNER intentionally not registered
        model_loader=_registry_with_loader({"broadcast.pt": broadcast_model}),
    )
    pipeline = CalibrationPipeline(PitchConfig(), registry)

    result = pipeline.process_frame(np.zeros(FRAME_SHAPE, dtype=np.uint8))

    assert result.scene_classification.scene_type == SceneType.LOW_ANGLE_CORNER
    assert broadcast_model.call_count == 1  # NOT called twice
    assert result.calibration_info.num_keypoints_used == 6  # used the original 6 points