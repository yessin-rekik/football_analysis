"""
Orchestrates a single frame through: initial keypoint detection -> scene
classification -> (conditionally) a second, scene-specific keypoint
detection -> homography calibration.

Chicken-and-egg note: classifying the scene needs SOME keypoint output to
look at, but choosing which model to run needs to already know the scene.
This is resolved by always running the broadcast model first (the one
model guaranteed to exist) and classifying off that output. A second
inference call only happens for LOW_ANGLE_CORNER frames -- CLOSE_UP frames
never get a second model call, since no keypoint model helps when no pitch
markings are in frame at all.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .scene_types import SceneType
from .scene_classifier import BaseSceneClassifier, HeuristicSceneClassifier, SceneClassification
from .model_registry import KeypointModelRegistry
from .calibrator import PitchCalibrator
from ..config.pitch_config import PitchConfig
from ..schemas.enums import CalibrationStatus
from ..schemas.frame_result import CalibrationInfo


@dataclass
class CalibrationPipelineResult:
    """Internal result type for callers of the pipeline (e.g. a video
    player script) that may want the raw keypoints for debugging/overlay
    purposes -- NOT part of the canonical FrameResult schema. Whatever
    stage turns this into a FrameResult (Phase 3) picks out just
    `calibration_info` and discards the rest."""
    scene_classification: SceneClassification
    calibration_info: CalibrationInfo
    keypoints_px: Optional[np.ndarray] = None
    keypoint_confidences: Optional[np.ndarray] = None


class CalibrationPipeline:
    def __init__(
        self,
        pitch_config: PitchConfig,
        model_registry: KeypointModelRegistry,
        scene_classifier: Optional[BaseSceneClassifier] = None,
        calibrator: Optional[PitchCalibrator] = None,
        confidence_threshold: float = 0.5,
    ):
        self.registry = model_registry
        self.classifier = scene_classifier or HeuristicSceneClassifier()
        self.calibrator = calibrator or PitchCalibrator(pitch_config, confidence_threshold=confidence_threshold)
        self.confidence_threshold = confidence_threshold

    def process_frame(self, frame: np.ndarray) -> CalibrationPipelineResult:
        # Step 1: always start with the broadcast model -- it's the one
        # guaranteed to be registered, and its output is what the
        # classifier needs to make a routing decision at all.
        initial_model = self.registry.get_model(SceneType.BROADCAST_WIDE)
        keypoints_px, confidences = initial_model.predict(frame)

        scene_result = self.classifier.classify(
            frame, keypoints_px, confidences, self.confidence_threshold
        )

        if scene_result.scene_type == SceneType.CLOSE_UP:
            # No pitch markings in view, regardless of which model runs.
            # Don't waste a second inference call -- this frame is a
            # Phase 1b (camera-motion propagation) case, not a routing case.
            return CalibrationPipelineResult(
                scene_classification=scene_result,
                calibration_info=CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED),
                keypoints_px=keypoints_px,
                keypoint_confidences=confidences,
            )

        if scene_result.scene_type == SceneType.LOW_ANGLE_CORNER:
            low_angle_model = self.registry.get_model(SceneType.LOW_ANGLE_CORNER)
            # If no low-angle model is registered yet, get_model() already
            # fell back to returning the SAME broadcast instance -- `is`
            # comparison detects that and skips a pointless duplicate
            # inference call on identical input.
            if low_angle_model is not initial_model:
                keypoints_px, confidences = low_angle_model.predict(frame)

        calibration_info = self.calibrator.calibrate_frame(keypoints_px, confidences)

        return CalibrationPipelineResult(
            scene_classification=scene_result,
            calibration_info=calibration_info,
            keypoints_px=keypoints_px,
            keypoint_confidences=confidences,
        )