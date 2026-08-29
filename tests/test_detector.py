import numpy as np
import pytest

from ..schemas.enums import ObjectClass
from ..tracking.detector import YoloObjectDetector


# A representative class-ID convention: doesn't need to match any real
# trained model, just needs to exercise the mapping logic.
CLASS_MAP = {
    0: ObjectClass.BALL,
    1: ObjectClass.GOALKEEPER,
    2: ObjectClass.PLAYER,
    3: ObjectClass.REFEREE,
}


def test_maps_class_ids_to_object_classes():
    xyxy = np.array([
        [790.0, 430.0, 810.0, 450.0],   # ball
        [100.0, 200.0, 130.0, 280.0],   # goalkeeper
    ])
    confs = np.array([0.9, 0.85])
    cls_ids = np.array([0, 1])

    dets = YoloObjectDetector._boxes_to_detections(xyxy, confs, cls_ids, CLASS_MAP, 0.3)

    assert len(dets) == 2
    assert dets[0].object_class == ObjectClass.BALL
    assert dets[1].object_class == ObjectClass.GOALKEEPER


def test_ball_uses_center_point_others_use_foot_point():
    xyxy = np.array([
        [790.0, 430.0, 810.0, 450.0],   # ball
        [800.0, 400.0, 824.0, 460.0],   # player
    ])
    confs = np.array([0.9, 0.9])
    cls_ids = np.array([0, 2])

    dets = YoloObjectDetector._boxes_to_detections(xyxy, confs, cls_ids, CLASS_MAP, 0.3)

    ball_det = dets[0]
    assert ball_det.pixel_position.x == pytest.approx(800.0)
    assert ball_det.pixel_position.y == pytest.approx(440.0)  # bbox center

    player_det = dets[1]
    assert player_det.pixel_position.x == pytest.approx(812.0)
    assert player_det.pixel_position.y == pytest.approx(460.0)  # bbox bottom


def test_below_threshold_confidence_dropped():
    xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
    confs = np.array([0.2])
    cls_ids = np.array([2])

    dets = YoloObjectDetector._boxes_to_detections(xyxy, confs, cls_ids, CLASS_MAP, 0.3)
    assert dets == []


def test_unmapped_class_id_skipped_not_guessed():
    """A class ID the caller didn't map to anything -- must be dropped,
    never silently assigned to some default ObjectClass."""
    xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
    confs = np.array([0.99])
    cls_ids = np.array([99])  # not in CLASS_MAP

    dets = YoloObjectDetector._boxes_to_detections(xyxy, confs, cls_ids, CLASS_MAP, 0.3)
    assert dets == []


def test_empty_input_returns_empty_list():
    xyxy = np.zeros((0, 4))
    confs = np.zeros((0,))
    cls_ids = np.zeros((0,), dtype=int)

    dets = YoloObjectDetector._boxes_to_detections(xyxy, confs, cls_ids, CLASS_MAP, 0.3)
    assert dets == []


def test_mixed_batch_preserves_order_and_filters_correctly():
    xyxy = np.array([
        [0.0, 0.0, 10.0, 10.0],     # low confidence player -- dropped
        [50.0, 50.0, 70.0, 90.0],   # referee -- kept
        [20.0, 20.0, 24.0, 24.0],   # ball -- kept
        [30.0, 30.0, 34.0, 34.0],   # unmapped class -- dropped
    ])
    confs = np.array([0.1, 0.95, 0.6, 0.99])
    cls_ids = np.array([2, 3, 0, 42])

    dets = YoloObjectDetector._boxes_to_detections(xyxy, confs, cls_ids, CLASS_MAP, 0.3)

    assert len(dets) == 2
    assert dets[0].object_class == ObjectClass.REFEREE
    assert dets[1].object_class == ObjectClass.BALL


def test_missing_ultralytics_raises_clear_error(monkeypatch):
    """Constructing a YoloObjectDetector without ultralytics installed
    should fail with a clear, actionable message -- not a bare ImportError
    from deep inside ultralytics' own import chain."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("no module named ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="ultralytics is required"):
        YoloObjectDetector(model_path="fake.pt", class_id_map=CLASS_MAP)