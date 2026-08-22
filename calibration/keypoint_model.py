"""
Keypoint model interface.

`BaseKeypointModel` is the contract the registry (and everything above it)
depends on. `YoloKeypointModel` is the concrete implementation wrapping
your existing SoccerKeypointDetector logic. The split matters because the
low-angle model is a SEPARATE set of weights, not a different interface --
both the broadcast and low-angle models will be `YoloKeypointModel`
instances pointed at different `.pt` files, since you confirmed the output
schema (29 keypoints, same order) is identical for both.

If the low-angle approach ever changes architecture (e.g. a non-YOLO
model), only a new BaseKeypointModel subclass is needed -- the registry
and calibrator never import YOLO directly, only this interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseKeypointModel(ABC):
    """A keypoint model takes a BGR frame and returns:
        keypoints_px: (29, 2) float array of pixel coordinates
        confidences:  (29,)   float array in [0, 1]
    for all 29 keypoint slots, regardless of visibility -- low-confidence
    slots should still have SOME coordinate (even a placeholder), with the
    confidence value being what the caller uses to decide visibility."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class YoloKeypointModel(BaseKeypointModel):
    """Wraps an ultralytics YOLO-pose model. This is the same detection
    logic as your working SoccerKeypointDetector, just conforming to the
    BaseKeypointModel interface so the registry can treat it
    interchangeably with any future model type.

    ultralytics is imported lazily (inside __init__, not at module level)
    so that anything that only needs BaseKeypointModel/the registry's
    routing logic -- e.g. unit tests -- doesn't require ultralytics or a
    GPU to be present at all.
    """

    def __init__(self, model_path: str, num_keypoints: int = 29):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required to load a YoloKeypointModel. "
                "Install it with: pip install ultralytics"
            ) from e

        self.model_path = model_path
        self.num_keypoints = num_keypoints
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}") from e

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        results = self.model(frame, verbose=False)

        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        confidences = np.zeros((self.num_keypoints,), dtype=np.float32)

        for result in results:
            if result.keypoints is not None:
                kpts = result.keypoints.data.cpu().numpy()
                if len(kpts) > 0:
                    kp_data = kpts[0]
                    n = min(self.num_keypoints, kp_data.shape[0])
                    keypoints[:n] = kp_data[:n, :2]
                    confidences[:n] = kp_data[:n, 2]

        return keypoints, confidences