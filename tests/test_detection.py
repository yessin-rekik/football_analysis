import pytest

from ..schemas.enums import ObjectClass
from ..schemas.frame_result import BoundingBox, PixelPoint
from ..tracking.detection import Detection


def test_detection_round_trip():
    det = Detection(
        object_class=ObjectClass.PLAYER,
        confidence=0.93,
        bounding_box=BoundingBox(x1=800.0, y1=400.0, x2=824.0, y2=460.0),
        pixel_position=PixelPoint(x=812.0, y=460.0),
    )
    raw = det.model_dump_json()
    reloaded = Detection.model_validate_json(raw)
    assert reloaded == det


def test_confidence_bounds_enforced():
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    with pytest.raises(ValueError):
        Detection(
            object_class=ObjectClass.BALL,
            confidence=1.5,  # out of [0, 1]
            bounding_box=bbox,
            pixel_position=PixelPoint(x=5, y=5),
        )
    with pytest.raises(ValueError):
        Detection(
            object_class=ObjectClass.BALL,
            confidence=-0.1,
            bounding_box=bbox,
            pixel_position=PixelPoint(x=5, y=5),
        )


def test_foot_point_is_bottom_center():
    bbox = BoundingBox(x1=800.0, y1=400.0, x2=824.0, y2=460.0)
    pt = Detection.foot_point(bbox)
    assert pt.x == pytest.approx(812.0)
    assert pt.y == pytest.approx(460.0)


def test_foot_point_works_for_asymmetric_box():
    bbox = BoundingBox(x1=100.0, y1=50.0, x2=101.0, y2=999.0)
    pt = Detection.foot_point(bbox)
    assert pt.x == pytest.approx(100.5)
    assert pt.y == pytest.approx(999.0)


def test_all_object_classes_are_valid():
    """Detection should accept every ObjectClass the schema defines --
    catches a future ObjectClass addition that Detection wasn't updated
    to allow (it shouldn't need updating, since it takes the enum
    directly, but this locks that in as a regression test)."""
    bbox = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    for oc in ObjectClass:
        det = Detection(
            object_class=oc,
            confidence=0.5,
            bounding_box=bbox,
            pixel_position=PixelPoint(x=0.5, y=1.0),
        )
        assert det.object_class == oc