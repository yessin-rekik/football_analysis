"""
Maps SceneType -> the right keypoint model, with lazy loading (a model
isn't loaded into memory/GPU until a frame actually needs it) and an
explicit fallback for scene types that don't have a dedicated model yet.

This is intentionally the ONLY place that knows "LOW_ANGLE_CORNER goes to
this specific weights file" -- the classifier doesn't know model paths
exist, and the calibrator (next file) won't know SceneType exists at all,
it just asks the registry for "the model for this scene" and gets one back.
"""

from typing import Callable, Dict, Optional

from .scene_types import SceneType
from .keypoint_model import BaseKeypointModel


# Scene types that never map to a keypoint model at all -- CLOSE_UP has no
# pitch markings in frame regardless of which model you use, so requesting
# a model for it is almost certainly a caller bug (it should have been
# routed to camera-motion propagation, Phase 1b, instead).
NO_MODEL_SCENE_TYPES = {SceneType.CLOSE_UP}


class ModelNotAvailableError(Exception):
    """Raised when a scene type has no registered model and no usable
    fallback -- distinct from a generic KeyError so callers can catch this
    specifically and decide what to do (e.g. skip the frame, log a metric)."""


class KeypointModelRegistry:
    def __init__(
        self,
        model_paths: Dict[SceneType, str],
        fallback_scene_type: SceneType = SceneType.BROADCAST_WIDE,
        model_loader: Optional[Callable[[str], BaseKeypointModel]] = None,
    ):
        """
        model_paths: e.g. {SceneType.BROADCAST_WIDE: "weights/broadcast.pt"}.
                     LOW_ANGLE_CORNER can simply be omitted until that model
                     exists -- requests for it will fall back cleanly rather
                     than crashing the pipeline.
        fallback_scene_type: used for UNKNOWN, and for any registered-but-
                     requested scene type whose model fails to load.
        model_loader: callable that turns a path into a BaseKeypointModel.
                     Defaults to YoloKeypointModel, but tests (and any
                     future non-YOLO model) can inject a different loader
                     without the registry needing to know anything changed.
        """
        self.model_paths = dict(model_paths)
        self.fallback_scene_type = fallback_scene_type

        if model_loader is None:
            from .keypoint_model import YoloKeypointModel
            model_loader = YoloKeypointModel
        self._model_loader = model_loader

        self._loaded: Dict[SceneType, BaseKeypointModel] = {}

    def available_scene_types(self) -> set:
        return set(self.model_paths.keys())

    def get_model(self, scene_type: SceneType) -> BaseKeypointModel:
        if scene_type in NO_MODEL_SCENE_TYPES:
            raise ModelNotAvailableError(
                f"{scene_type.value} has no associated keypoint model by design -- "
                f"this scene type should be routed to camera-motion propagation "
                f"instead of requesting a model."
            )

        resolved_type = scene_type
        if resolved_type not in self.model_paths:
            if self.fallback_scene_type not in self.model_paths:
                raise ModelNotAvailableError(
                    f"No model registered for {scene_type.value}, and fallback "
                    f"{self.fallback_scene_type.value} is also unregistered. "
                    f"Registered types: {sorted(t.value for t in self.model_paths)}"
                )
            resolved_type = self.fallback_scene_type

        if resolved_type in self._loaded:
            return self._loaded[resolved_type]

        path = self.model_paths[resolved_type]
        try:
            model = self._model_loader(path)
        except Exception as e:
            if resolved_type == self.fallback_scene_type:
                raise  # the fallback itself failed -- nothing left to fall back to
            # the specific model failed to load; try the fallback once
            fallback_model = self.get_model(self.fallback_scene_type)
            self._loaded[resolved_type] = fallback_model
            return fallback_model

        self._loaded[resolved_type] = model
        return model