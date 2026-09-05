import numpy as np
import pytest

from ..schemas.enums import ObjectClass, Team
from ..schemas.frame_result import TrackedObject, BoundingBox, PixelPoint
from ..tracking.team_classifier import JerseyColorTeamClassifier


GRASS_GREEN = (60, 150, 60)  # BGR, falls inside GREEN_HUE_RANGE by design
BLUE = (200, 50, 50)
RED = (50, 50, 200)


def _make_frame(height: int = 100, width: int = 100, background_bgr=GRASS_GREEN) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = background_bgr
    return frame


def _player(
    frame, track_id, x1, y1, x2, y2, color_bgr,
    object_class=ObjectClass.PLAYER, has_box: bool = True,
) -> TrackedObject:
    """Paints a solid-color patch directly into `frame` at the given box,
    and returns the matching TrackedObject. Since the patch exactly fills
    the bbox with a single color, the expected jersey_color is exact
    (no green mixed in), letting tests assert precise values rather than
    fuzzy ranges."""
    frame[y1:y2, x1:x2] = color_bgr
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        pixel_position=PixelPoint(x=(x1 + x2) / 2.0, y=float(y2)),
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) if has_box else None,
    )


def _two_team_frame():
    """4 outfield players (2 blue, 2 red), no goalkeeper/referee/ball --
    enough to clear the default min_players_for_fit=4."""
    frame = _make_frame()
    objs = [
        _player(frame, 1, 10, 10, 20, 20, BLUE),
        _player(frame, 2, 30, 10, 40, 20, BLUE),
        _player(frame, 3, 50, 10, 60, 20, RED),
        _player(frame, 4, 70, 10, 80, 20, RED),
    ]
    return frame, objs


# ---- jersey color extraction (static method, tested directly) ----

def test_jersey_color_extracts_median_ignoring_green_background():
    frame = _make_frame()
    # a red patch sitting inside a larger box that's otherwise background
    # green -- if masking works, the green majority must not pull the
    # result away from the exact red value.
    frame[15:25, 15:25] = (10, 10, 220)
    obj = TrackedObject(
        track_id=1, object_class=ObjectClass.PLAYER,
        pixel_position=PixelPoint(x=20.0, y=30.0),
        bounding_box=BoundingBox(x1=10, y1=10, x2=30, y2=30),
    )
    color = JerseyColorTeamClassifier._jersey_color(frame, obj)
    assert color is not None
    assert np.allclose(color, (10, 10, 220), atol=1.0)


def test_jersey_color_returns_none_for_fully_green_crop():
    frame = _make_frame()
    obj = TrackedObject(
        track_id=1, object_class=ObjectClass.PLAYER,
        pixel_position=PixelPoint(x=15.0, y=15.0),
        bounding_box=BoundingBox(x1=10, y1=10, x2=20, y2=20),
    )
    assert JerseyColorTeamClassifier._jersey_color(frame, obj) is None


def test_jersey_color_returns_none_without_bounding_box():
    frame = _make_frame()
    obj = TrackedObject(
        track_id=1, object_class=ObjectClass.PLAYER,
        pixel_position=PixelPoint(x=15.0, y=15.0),
        bounding_box=None,
    )
    assert JerseyColorTeamClassifier._jersey_color(frame, obj) is None


# ---- assign_teams behavior ----

def test_insufficient_players_leaves_everyone_unassigned():
    clf = JerseyColorTeamClassifier(min_players_for_fit=4)
    frame = _make_frame()
    objs = [
        _player(frame, 1, 10, 10, 20, 20, BLUE),
        _player(frame, 2, 30, 10, 40, 20, RED),
    ]  # only 2, below the default threshold of 4
    result = clf.assign_teams(frame, objs)
    assert all(o.team is None for o in result)


def test_two_teams_separated_correctly():
    clf = JerseyColorTeamClassifier(min_players_for_fit=4)
    frame, objs = _two_team_frame()
    result = clf.assign_teams(frame, objs)

    blue_teams = {o.team for o in result if o.track_id in (1, 2)}
    red_teams = {o.team for o in result if o.track_id in (3, 4)}

    assert None not in blue_teams and None not in red_teams
    assert len(blue_teams) == 1  # both blue players got the same team
    assert len(red_teams) == 1   # both red players got the same team
    assert blue_teams != red_teams  # the two teams are actually different


def test_team_labels_stable_across_frames_despite_reordering():
    """Re-fitting every frame must not flip which label means which team,
    even if cv2.kmeans happens to return its two clusters in a different
    order than last time -- that's exactly what the reconciliation step
    exists to prevent."""
    clf = JerseyColorTeamClassifier(min_players_for_fit=4)

    frame1, objs1 = _two_team_frame()
    result1 = clf.assign_teams(frame1, objs1)
    blue_team_frame1 = result1[0].team  # track_id=1, a blue player
    red_team_frame1 = result1[2].team   # track_id=3, a red player

    # Same colors again, but listed in reverse order -- if the classifier
    # were naively trusting cv2.kmeans' output order each time, this is
    # exactly the kind of perturbation that could expose a label flip.
    frame2 = _make_frame()
    objs2 = [
        _player(frame2, 4, 70, 10, 80, 20, RED),
        _player(frame2, 3, 50, 10, 60, 20, RED),
        _player(frame2, 2, 30, 10, 40, 20, BLUE),
        _player(frame2, 1, 10, 10, 20, 20, BLUE),
    ]
    result2 = clf.assign_teams(frame2, objs2)
    blue_obj_frame2 = next(o for o in result2 if o.track_id == 1)
    red_obj_frame2 = next(o for o in result2 if o.track_id == 3)

    assert blue_obj_frame2.team == blue_team_frame1
    assert red_obj_frame2.team == red_team_frame1


def test_goalkeeper_excluded_from_fit_but_assigned_and_does_not_skew_centroids():
    clf = JerseyColorTeamClassifier(min_players_for_fit=4)
    frame, objs = _two_team_frame()
    # A goalkeeper in a third, unrelated kit color -- if this leaked into
    # the k=2 fit, it would drag one of the two centroids toward yellow.
    goalkeeper_color = (0, 220, 220)  # yellow-ish, BGR
    objs.append(_player(frame, 5, 10, 40, 20, 50, goalkeeper_color, object_class=ObjectClass.GOALKEEPER))

    result = clf.assign_teams(frame, objs)

    # Centroids must still land exactly on the true blue/red values --
    # proof the goalkeeper's color never entered the fit.
    centroids = clf._running_centroids
    dists_to_blue = [np.linalg.norm(c - np.array(BLUE)) for c in centroids]
    dists_to_red = [np.linalg.norm(c - np.array(RED)) for c in centroids]
    assert min(dists_to_blue) < 1.0
    assert min(dists_to_red) < 1.0

    # The goalkeeper still gets *some* team assignment (best-effort).
    gk = next(o for o in result if o.track_id == 5)
    assert gk.team is not None


def test_referee_and_ball_never_assigned_a_team():
    clf = JerseyColorTeamClassifier(min_players_for_fit=4)
    frame, objs = _two_team_frame()
    objs.append(_player(frame, 5, 10, 40, 20, 50, (0, 0, 0), object_class=ObjectClass.REFEREE))
    objs.append(_player(frame, 6, 30, 40, 40, 50, (255, 255, 255), object_class=ObjectClass.BALL))

    result = clf.assign_teams(frame, objs)

    referee = next(o for o in result if o.track_id == 5)
    ball = next(o for o in result if o.track_id == 6)
    assert referee.team is None
    assert ball.team is None


def test_object_missing_bounding_box_stays_unassigned_even_when_others_fit():
    clf = JerseyColorTeamClassifier(min_players_for_fit=4)
    frame, objs = _two_team_frame()
    boxless = _player(frame, 5, 0, 0, 0, 0, BLUE, has_box=False)
    objs.append(boxless)

    result = clf.assign_teams(frame, objs)

    boxless_result = next(o for o in result if o.track_id == 5)
    assert boxless_result.team is None
    # meanwhile the properly-boxed players still got assigned
    assert all(o.team is not None for o in result if o.track_id != 5)


def test_ema_smoothing_moves_centroids_gradually_not_instantly():
    clf = JerseyColorTeamClassifier(ema_alpha=0.1, min_players_for_fit=4)

    frame1, objs1 = _two_team_frame()
    clf.assign_teams(frame1, objs1)
    baseline = clf._running_centroids.copy()
    blue_idx = int(np.argmin([np.linalg.norm(c - np.array(BLUE)) for c in baseline]))
    red_idx = 1 - blue_idx

    shifted_blue = (180, 60, 60)
    shifted_red = (70, 60, 180)
    frame2 = _make_frame()
    objs2 = [
        _player(frame2, 1, 10, 10, 20, 20, shifted_blue),
        _player(frame2, 2, 30, 10, 40, 20, shifted_blue),
        _player(frame2, 3, 50, 10, 60, 20, shifted_red),
        _player(frame2, 4, 70, 10, 80, 20, shifted_red),
    ]
    clf.assign_teams(frame2, objs2)
    updated = clf._running_centroids

    expected_blue = 0.9 * np.array(BLUE) + 0.1 * np.array(shifted_blue)
    expected_red = 0.9 * np.array(RED) + 0.1 * np.array(shifted_red)

    assert np.allclose(updated[blue_idx], expected_blue, atol=1.0)
    assert np.allclose(updated[red_idx], expected_red, atol=1.0)
    # must not have jumped all the way to the new frame's raw colors
    assert not np.allclose(updated[blue_idx], shifted_blue, atol=1.0)