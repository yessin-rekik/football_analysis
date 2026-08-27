"""
Ties Phase 1a (scene routing + model selection) together with Phase 1b
(gap-bridging via camera motion) into a single per-frame entrypoint for
real video use.

Why this is a separate class rather than folding one into the other:
CalibrationPipeline's job is "which model should see this frame, and what
does it detect" -- it has no concept of frame-to-frame history.
PropagatingCalibrator's job is "given this frame's keypoints (or none),
what's the best calibration right now, accounting for recent history" --
it has no concept of scene types or model routing. Keeping them separate
means each stays independently testable (already proven: 4 pipeline
tests, 13 propagation-related tests, neither needed to know the other
existed). This class is the seam between the two, and nothing else needs
to be.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .pipeline import CalibrationPipeline
from .propagating_calibrator import PropagatingCalibrator
from .scene_classifier import SceneClassification
from .scene_types import SceneType
from ..schemas.frame_result import CalibrationInfo


@dataclass
class OrchestratedFrameResult:
    scene_classification: SceneClassification
    calibration_info: CalibrationInfo
    keypoints_px: Optional[np.ndarray] = None
    keypoint_confidences: Optional[np.ndarray] = None


class VideoCalibrationOrchestrator:
    def __init__(self, pipeline: CalibrationPipeline, propagating_calibrator: PropagatingCalibrator):
        self.pipeline = pipeline
        self.propagating_calibrator = propagating_calibrator

    def process_frame(self, frame: np.ndarray) -> OrchestratedFrameResult:
        scene_result, keypoints_px, keypoint_confidences = self.pipeline.route_and_detect(frame)

        # Translate CLOSE_UP into the explicit None/None PropagatingCalibrator
        # expects -- this is the ONE place that conversion needs to happen,
        # since it's a Phase 1b contract, not something routing itself
        # should have to know about.
        if scene_result.scene_type == SceneType.CLOSE_UP:
            propagation_input: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        else:
            propagation_input = (keypoints_px, keypoint_confidences)

        calibration_info = self.propagating_calibrator.process_frame(frame, *propagation_input)

        return OrchestratedFrameResult(
            scene_classification=scene_result,
            calibration_info=calibration_info,
            keypoints_px=keypoints_px,  # always the raw detection, for debugging/overlay
            keypoint_confidences=keypoint_confidences,
        )

    def pixel_to_world(self, pixel_xy):
        return self.propagating_calibrator.pixel_to_world(pixel_xy)