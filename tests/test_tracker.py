import pytest

from ..schemas.enums import ObjectClass
from ..schemas.frame_result import BoundingBox, PixelPoint
from ..tracking.detection import Detection
from ..tracking.tracker import ByteTracker


def _det(object_class, x1, y1, x2, y2, confidence=0.9):
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    pixel_position = Detection.center_point(bbox) if object_class == ObjectClass.BALL else Detection.foot_point(bbox)
    return Detection(object_class=object_class, confidence=confidence, bounding_box=bbox, pixel_position=pixel_position)


def _shift(det: Detection, dx: float, dy: float) -> Detection:
    b = det.bounding_box
    return _det(det.object_class, b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy, det.confidence)


# --- IoU / greedy matching, tested directly (pure functions, no tracker state) ---

def test_iou_identical_boxes_is_one():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    assert ByteTracker._iou(a, a) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=100, y1=100, x2=110, y2=110)
    assert ByteTracker._iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)   # area 100
    b = BoundingBox(x1=5, y1=0, x2=15, y2=10)   # area 100, overlap 5x10=50
    # union = 100 + 100 - 50 = 150, iou = 50/150
    assert ByteTracker._iou(a, b) == pytest.approx(50.0 / 150.0)


# --- Tracker behavior, via update() over a sequence of frames ---

def _confirmed_tracker(min_hits=1, **kwargs):
    """min_hits=1 by default in most tests, so a track is emitted the very
    frame it's created -- keeps the tests focused on association logic
    rather than confirmation-delay logic (that gets its own dedicated
    test below)."""
    return ByteTracker(min_hits=min_hits, **kwargs)


def test_new_track_created_for_unmatched_high_conf_detection():
    tracker = _confirmed_tracker()
    det = _det(ObjectClass.PLAYER, 100, 100, 130, 180, confidence=0.9)

    result = tracker.update([det])

    assert len(result) == 1
    assert result[0].object_class == ObjectClass.PLAYER
    assert result[0].track_id == 1


def test_low_conf_detection_never_spawns_new_track():
    tracker = _confirmed_tracker()
    det = _det(ObjectClass.PLAYER, 100, 100, 130, 180, confidence=0.2)  # below high threshold

    result = tracker.update([det])

    assert result == []


def test_track_id_stable_across_frames_for_continuously_tracked_object():
    tracker = _confirmed_tracker()
    det1 = _det(ObjectClass.PLAYER, 100, 100, 130, 180)
    r1 = tracker.update([det1])
    track_id = r1[0].track_id

    det2 = _shift(det1, 5, 5)  # small movement frame to frame
    r2 = tracker.update([det2])

    assert len(r2) == 1
    assert r2[0].track_id == track_id


def test_cross_class_detections_never_match_each_others_tracks():
    tracker = _confirmed_tracker()
    player_det = _det(ObjectClass.PLAYER, 100, 100, 130, 180)
    r1 = tracker.update([player_det])
    player_track_id = r1[0].track_id

    # Same location, next frame, but now labeled GOALKEEPER -- must NOT
    # reuse the player's track_id; must instead spawn its own new track.
    gk_det = _det(ObjectClass.GOALKEEPER, 100, 100, 130, 180)
    r2 = tracker.update([gk_det])

    assert len(r2) == 1
    assert r2[0].object_class == ObjectClass.GOALKEEPER
    assert r2[0].track_id != player_track_id


def test_track_survives_one_low_confidence_frame():
    """The core ByteTrack idea: a real object that's momentarily only
    detected at low confidence (blur/partial occlusion) should keep its
    track alive via stage-2 matching, not lose it."""
    tracker = _confirmed_tracker()
    det1 = _det(ObjectClass.PLAYER, 100, 100, 130, 180, confidence=0.9)
    r1 = tracker.update([det1])
    track_id = r1[0].track_id

    # Same object, barely moved, but this frame only clears low confidence
    low_conf_det = _shift(det1, 2, 2)
    low_conf_det = _det(
        low_conf_det.object_class,
        low_conf_det.bounding_box.x1, low_conf_det.bounding_box.y1,
        low_conf_det.bounding_box.x2, low_conf_det.bounding_box.y2,
        confidence=0.2,  # between low and high threshold
    )
    r2 = tracker.update([low_conf_det])

    assert len(r2) == 1
    assert r2[0].track_id == track_id  # same identity preserved


def test_track_dropped_after_max_age_missed_frames():
    tracker = _confirmed_tracker(max_age=2)
    det = _det(ObjectClass.PLAYER, 100, 100, 130, 180)
    tracker.update([det])  # frame 1: created

    tracker.update([])  # frame 2: missed (age 1)
    tracker.update([])  # frame 3: missed (age 2, at limit)
    assert len(tracker._tracks) == 1  # still alive, at the limit

    tracker.update([])  # frame 4: exceeds max_age
    assert len(tracker._tracks) == 0  # dropped


def test_track_not_emitted_on_a_missed_frame():
    tracker = _confirmed_tracker()
    det = _det(ObjectClass.PLAYER, 100, 100, 130, 180)
    tracker.update([det])

    result = tracker.update([])  # nothing detected this frame
    assert result == []  # absent, not interpolated


def test_min_hits_delays_first_emission():
    tracker = ByteTracker(min_hits=3)
    det = _det(ObjectClass.PLAYER, 100, 100, 130, 180)

    r1 = tracker.update([det])
    assert r1 == []  # hit 1 of 3 -- not confirmed yet

    r2 = tracker.update([_shift(det, 1, 1)])
    assert r2 == []  # hit 2 of 3

    r3 = tracker.update([_shift(det, 2, 2)])
    assert len(r3) == 1  # hit 3 of 3 -- now confirmed


def test_two_simultaneous_players_get_distinct_stable_ids():
    tracker = _confirmed_tracker()
    det_a = _det(ObjectClass.PLAYER, 100, 100, 130, 180)
    det_b = _det(ObjectClass.PLAYER, 500, 500, 530, 580)

    r1 = tracker.update([det_a, det_b])
    assert len(r1) == 2
    ids_frame1 = {r.track_id for r in r1}

    r2 = tracker.update([_shift(det_a, 2, 2), _shift(det_b, 2, 2)])
    ids_frame2 = {r.track_id for r in r2}

    assert ids_frame1 == ids_frame2  # same two IDs, no swap


def test_ball_uses_center_point_in_output():
    tracker = _confirmed_tracker()
    det = _det(ObjectClass.BALL, 790, 430, 810, 450)

    result = tracker.update([det])

    assert result[0].pixel_position.x == pytest.approx(800.0)
    assert result[0].pixel_position.y == pytest.approx(440.0)


def test_reset_clears_all_tracks():
    tracker = _confirmed_tracker()
    det = _det(ObjectClass.PLAYER, 100, 100, 130, 180)
    r1 = tracker.update([det])
    original_id = r1[0].track_id

    tracker.reset()
    r2 = tracker.update([det])

    assert r2[0].track_id == original_id  # id counter also reset, starts back at 1
    assert len(tracker._tracks) == 1


def test_ambiguous_overlap_greedy_prefers_higher_iou_pair():
    """Two tracks both plausibly overlap two detections -- greedy matching
    should award the highest-IoU pair first, not just first-index-wins."""
    tracker = _confirmed_tracker()
    # Track A and B start far apart
    det_a = _det(ObjectClass.PLAYER, 0, 0, 10, 10)
    det_b = _det(ObjectClass.PLAYER, 100, 100, 110, 110)
    r1 = tracker.update([det_a, det_b])
    id_a = [r.track_id for r in r1 if r.pixel_position.x < 50][0]
    id_b = [r.track_id for r in r1 if r.pixel_position.x >= 50][0]

    # Next frame: detection near A's old position has higher IoU with A,
    # detection near B's old position has higher IoU with B (despite some
    # overlap ambiguity introduced by a small nudge toward each other).
    next_det_a = _det(ObjectClass.PLAYER, 1, 1, 11, 11)     # near A
    next_det_b = _det(ObjectClass.PLAYER, 99, 99, 109, 109)  # near B
    r2 = tracker.update([next_det_a, next_det_b])

    r2_by_id = {r.track_id: r for r in r2}
    assert r2_by_id[id_a].pixel_position.x < 50
    assert r2_by_id[id_b].pixel_position.x >= 50