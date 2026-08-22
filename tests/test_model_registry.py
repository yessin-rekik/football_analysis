import numpy as np
import pytest

from ..calibration.scene_types import SceneType
from ..calibration.keypoint_model import BaseKeypointModel
from ..calibration.model_registry import (
    KeypointModelRegistry, ModelNotAvailableError,
)


class FakeKeypointModel(BaseKeypointModel):
    """Stands in for YoloKeypointModel in tests -- records which path it
    was 'loaded' from so tests can assert routing went to the right one."""
    def __init__(self, path: str):
        self.path = path
        self.load_count = 1

    def predict(self, frame):
        return np.zeros((29, 2), dtype=np.float32), np.zeros((29,), dtype=np.float32)


def _fake_loader_factory():
    """Returns (loader_fn, calls_list) so tests can assert how many times
    and with what paths the loader was actually invoked (lazy-loading check)."""
    calls = []

    def loader(path):
        calls.append(path)
        return FakeKeypointModel(path)

    return loader, calls


def test_routes_to_correct_model_per_scene_type():
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(
        model_paths={
            SceneType.BROADCAST_WIDE: "weights/broadcast.pt",
            SceneType.LOW_ANGLE_CORNER: "weights/low_angle.pt",
        },
        model_loader=loader,
    )

    broadcast_model = registry.get_model(SceneType.BROADCAST_WIDE)
    low_angle_model = registry.get_model(SceneType.LOW_ANGLE_CORNER)

    assert broadcast_model.path == "weights/broadcast.pt"
    assert low_angle_model.path == "weights/low_angle.pt"
    assert broadcast_model is not low_angle_model


def test_lazy_loading_only_loads_requested_model():
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(
        model_paths={
            SceneType.BROADCAST_WIDE: "weights/broadcast.pt",
            SceneType.LOW_ANGLE_CORNER: "weights/low_angle.pt",
        },
        model_loader=loader,
    )

    registry.get_model(SceneType.BROADCAST_WIDE)
    assert calls == ["weights/broadcast.pt"]  # low_angle never touched


def test_caches_loaded_model_does_not_reload():
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "weights/broadcast.pt"},
        model_loader=loader,
    )

    m1 = registry.get_model(SceneType.BROADCAST_WIDE)
    m2 = registry.get_model(SceneType.BROADCAST_WIDE)

    assert m1 is m2
    assert calls == ["weights/broadcast.pt"]  # loaded exactly once


def test_falls_back_when_low_angle_model_not_yet_available():
    """The exact scenario you're in right now: no low-angle model trained
    yet. Requesting LOW_ANGLE_CORNER should silently and safely use the
    broadcast model rather than crashing the pipeline."""
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "weights/broadcast.pt"},
        # LOW_ANGLE_CORNER intentionally omitted
        model_loader=loader,
    )

    model = registry.get_model(SceneType.LOW_ANGLE_CORNER)
    assert model.path == "weights/broadcast.pt"


def test_unknown_scene_type_falls_back_to_broadcast():
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "weights/broadcast.pt"},
        model_loader=loader,
    )
    model = registry.get_model(SceneType.UNKNOWN)
    assert model.path == "weights/broadcast.pt"


def test_close_up_raises_by_design():
    """CLOSE_UP should never reach the registry in normal operation -- if it
    does, that's a caller bug (it should have routed to camera-motion
    propagation instead), so this must raise loudly, not silently fall
    back to a model that can't help anyway."""
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: "weights/broadcast.pt"},
        model_loader=loader,
    )
    with pytest.raises(ModelNotAvailableError):
        registry.get_model(SceneType.CLOSE_UP)


def test_raises_when_nothing_registered_at_all():
    loader, calls = _fake_loader_factory()
    registry = KeypointModelRegistry(model_paths={}, model_loader=loader)
    with pytest.raises(ModelNotAvailableError):
        registry.get_model(SceneType.BROADCAST_WIDE)


def test_failed_model_load_falls_back():
    """If a registered model's file is missing/corrupt, the registry
    should fall back rather than take down the whole pipeline on one bad
    scene type."""
    def flaky_loader(path):
        if "low_angle" in path:
            raise RuntimeError("simulated corrupt weights file")
        return FakeKeypointModel(path)

    registry = KeypointModelRegistry(
        model_paths={
            SceneType.BROADCAST_WIDE: "weights/broadcast.pt",
            SceneType.LOW_ANGLE_CORNER: "weights/low_angle_corrupt.pt",
        },
        model_loader=flaky_loader,
    )

    model = registry.get_model(SceneType.LOW_ANGLE_CORNER)
    assert model.path == "weights/broadcast.pt"  # fell back successfully