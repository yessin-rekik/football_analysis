"""
Object detector interface (Phase 2 player/ball detection).

Same shape as `calibration/keypoint_model.py`: an abstract interface the
rest of the pipeline depends on, plus a concrete YOLO implementation
wrapping your existing player/ball detection model. If detection ever
needs a non-YOLO architecture, only a new `BaseObjectDetector` subclass is
needed -- nothing that calls `detect()` should need to change.

Unlike the keypoint model (which returns a fixed-size array of 29 named
slots), a detector returns a variable-length list -- however many objects
it found this frame, however many players/refs/balls actually happen to
be visible.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from ..schemas.enums import ObjectClass
from ..schemas.frame_result import BoundingBox
from .detection import Detection


class BaseObjectDetector(ABC):
    """A detector takes a BGR frame and returns every object it found this
    frame, as `Detection`s -- no identity/tracking, that's the tracker's
    job, built on top of a sequence of these."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class YoloObjectDetector(BaseObjectDetector):
    """Wraps an ultralytics YOLO detection model.

    `class_id_map` is REQUIRED and deliberately has no default: it maps
    your specific trained model's integer class IDs to `ObjectClass`
    values (e.g. {0: ObjectClass.BALL, 1: ObjectClass.GOALKEEPER, ...}),
    and that mapping is a property of your training data, not something
    this code can guess. Guessing wrong here would silently mislabel every
    detection with no error to ever catch it -- far worse than the
    up-front friction of having to specify it. A class ID that appears in
    the model's output but ISN'T in this map is skipped, not guessed at.

    `ultralytics` is imported lazily (inside __init__, not at module
    level) for the same reason as `YoloKeypointModel` -- so anything that
    only needs `BaseObjectDetector` (e.g. tracker unit tests using a fake
    detector) doesn't require ultralytics or a GPU to be present at all.
    """

    def __init__(
        self,
        model_path: str,
        class_id_map: Dict[int, ObjectClass],
        confidence_threshold: float = 0.3,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required to load a YoloObjectDetector. "
                "Install it with: pip install ultralytics"
            ) from e

        self.model_path = model_path
        self.class_id_map = dict(class_id_map)
        self.confidence_threshold = confidence_threshold
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}") from e

    @staticmethod
    def _boxes_to_detections(
        xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_id_map: Dict[int, ObjectClass],
        confidence_threshold: float,
    ) -> List[Detection]:
        """Pure parsing logic, split out from `detect()` so it's testable
        with plain numpy arrays -- no ultralytics `Results` object, no
        loaded model, no GPU needed to verify this logic is correct.

        xyxy:        (N, 4) pixel box corners
        confidences: (N,)   float in [0, 1]
        class_ids:   (N,)   int, this model's own class-ID convention
        """
        detections: List[Detection] = []
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confidences, class_ids):
            if conf < confidence_threshold:
                continue

            object_class = class_id_map.get(int(cls_id))
            if object_class is None:
                # Unmapped class ID -- skip rather than guess. Could be a
                # class the caller intentionally left unmapped (e.g. a
                # "referee assistant" class they don't care about yet).
                continue

            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            pixel_position = (
                Detection.center_point(bbox)
                if object_class == ObjectClass.BALL
                else Detection.foot_point(bbox)
            )

            detections.append(Detection(
                object_class=object_class,
                confidence=float(conf),
                bounding_box=bbox,
                pixel_position=pixel_position,
            ))

        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, verbose=False)

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            detections.extend(self._boxes_to_detections(
                xyxy, confidences, class_ids, self.class_id_map, self.confidence_threshold,
            ))

        return detections