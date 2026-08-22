# Phase 1a -- scene routing (SceneType, classifiers, model registry) and
# keypoint-based homography calibration.
# Phase 1b (camera-motion propagation for zoom/occlusion gaps) will extend
# this package with a class that wraps PitchCalibrator, not modify it.

from .scene_types import SceneType
from .scene_classifier import BaseSceneClassifier, HeuristicSceneClassifier, SceneClassification
from .keypoint_model import BaseKeypointModel, YoloKeypointModel
from .model_registry import KeypointModelRegistry, ModelNotAvailableError
from .calibrator import PitchCalibrator
from .pipeline import CalibrationPipeline, CalibrationPipelineResult

__all__ = [
    "SceneType",
    "BaseSceneClassifier",
    "HeuristicSceneClassifier",
    "SceneClassification",
    "BaseKeypointModel",
    "YoloKeypointModel",
    "KeypointModelRegistry",
    "ModelNotAvailableError",
    "PitchCalibrator",
    "CalibrationPipeline",
    "CalibrationPipelineResult",
]