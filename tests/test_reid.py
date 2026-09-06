"""
Tests for `tracking.reid.ReIdentifier` -- the re-identification module
that reconnects dropped tracks back to their original `track_id` after
the tracker has forgotten them.

All tests use:
  - synthetic frames (a 100x100 BGR array, painted with solid colors)
  - synthetic `TrackedObject`s
  - a real `JerseyColorTeamClassifier` to populate the per-track color
    cache (re-ID depends on this cache; testing without it would be
    testing a different code path than the real one)
  - a synthetic `CalibrationInfo` with either a real homography (for
    the world-speed gate) or `homography=None` (to test the
    uncalibrated branch)
No ultralytics, no real video.

A note on `min_reid_gap_frames`: several tests below explicitly pass a
small `min_reid_gap_frames` (e.g. 1) even though the default is 30. This
is intentional, not an oversight -- those tests exist to isolate one
specific gate (color, speed, appearance embedder) using short synthetic
gaps of 5-9 frames, and the floor (added to fix a real bug where re-ID
was racing ByteTracker's own recovery window) would otherwise reject the
match before that gate is ever exercised. Tests for the floor itself are
grouped near the end of this file and deliberately use the DEFAULT
min_reid_gap_frames=30 to lock in that exact behavior.
"""

from collections import deque

import numpy as np
import pytest

from ..schemas.enums import ObjectClass, PositionProvenance, Team
from ..schemas.frame_result import (
    BoundingBox,
    CalibrationInfo,
    CalibrationStatus,
    PixelPoint,
    TrackedObject,
)
from ..tracking.team_classifier import JerseyColorTeamClassifier
from ..tracking.reid import ReIdentifier


# Standard test colors (BGR). Distinct enough that BGR Euclidean
# distance is meaningful.
GRASS_GREEN = (60, 150, 60)  # falls inside GREEN_HUE_RANGE
BLUE = (200, 50, 50)
RED = (50, 50, 200)
YELLOW = (50, 200, 200)


# Identity homography: 1 pixel = 1 meter in both x and y. Lets the
# world-speed gate reason directly in pixel-space displacement / dt.
IDENTITY_H = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


# ---- helpers ----


def _make_frame(bgr=GRASS_GREEN) -> np.ndarray:
    """100x100 solid-color BGR frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8) + np.array(bgr, dtype=np.uint8)


def _paint(frame, x1, y1, x2, y2, bgr):
    """Paint a solid color into a rectangular region of the frame."""
    frame[y1:y2, x1:x2] = np.array(bgr, dtype=np.uint8)


def _player(track_id, x1, y1, x2, y2, object_class=ObjectClass.PLAYER) -> TrackedObject:
    """Build a TrackedObject with a foot_point for a given bbox."""
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        pixel_position=PixelPoint(x=(x1 + x2) / 2.0, y=float(y2)),
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _calibrated(homography=IDENTITY_H, status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS) -> CalibrationInfo:
    return CalibrationInfo(
        status=status,
        homography=homography,
        frames_since_last_anchor=0,
        num_keypoints_used=4,
    )


def _uncalibrated() -> CalibrationInfo:
    return CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED, homography=None)


def _run_pipeline_step(
    reid: ReIdentifier,
    team_clf: JerseyColorTeamClassifier,
    frame: np.ndarray,
    tracked: list,
    fps: float,
    cal_info: CalibrationInfo,
):
    """One full frame step: team classifier first (to populate the per-
    track color cache that re-ID depends on), then re-ID."""
    team_clf.assign_teams(frame, tracked)
    reid.update(tracked, cal_info, fps)


# ---- tests ----


def test_no_match_leaves_new_track_unchanged():
    """Empty re-ID history: a brand-new track should keep its
    track_id and stay provenance=OBSERVED."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf)

    frame = _make_frame()
    _paint(frame, 10, 10, 20, 20, BLUE)
    obj = _player(track_id=42, x1=10, y1=10, x2=20, y2=20)

    _run_pipeline_step(reid, team_clf, frame, [obj], fps=10.0, cal_info=_calibrated())

    assert obj.track_id == 42
    assert obj.provenance == PositionProvenance.OBSERVED


def test_color_match_accepts_same_jersey_within_threshold():
    """A track that drops and respawns with the same jersey color, at
    a plausible position, should be re-linked. min_reid_gap_frames=1
    isolates the color gate from the (separately-tested) gap floor --
    this test's 9-frame gap is intentionally shorter than the default
    floor of 30."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=1)
    fps = 10.0
    cal_info = _calibrated()

    # Frame 0: blue player at (10..20, 10..20).
    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player_1 = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player_1], fps, cal_info)

    # Frames 1-9: player is missing (re-ID marks them as dropped after
    # one missed frame).
    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    # Frame 10: same blue player respawns at (12..22, 10..20) -- 2
    # pixels right, plausible motion. Tracker would have given it a
    # brand-new track_id (we simulate with track_id=999).
    frame1 = _make_frame()
    _paint(frame1, 12, 10, 22, 20, BLUE)
    respawned = _player(track_id=999, x1=12, y1=10, x2=22, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 7, "expected the respawned track to be re-stamped to the original track_id"
    assert respawned.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP


def test_color_mismatch_rejects():
    """A respawning track with a clearly different jersey color should
    NOT be re-linked."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=1)
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)

    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    # Respawns as RED. BGR distance from BLUE (200,50,50) to RED
    # (50,50,200) is sqrt(150^2 + 0^2 + 150^2) = ~212 -- way over 60.
    frame1 = _make_frame()
    _paint(frame1, 10, 10, 20, 20, RED)
    respawned = _player(track_id=999, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 999, "track_id should NOT be re-stamped on a color mismatch"
    assert respawned.provenance == PositionProvenance.OBSERVED


def test_world_speed_gate_rejects_implausibly_fast_re_identification():
    """A re-ID candidate that implies a 30 m/s sprint should be
    rejected, even with a matching color. Positive control (5 m/s)
    should accept. min_reid_gap_frames=1 isolates the speed gate from
    the gap floor -- both cases here use a 9-frame gap."""
    team_clf = JerseyColorTeamClassifier()
    fps = 10.0
    cal_info = _calibrated()

    # --- Negative case: 30 m/s, must reject ---
    reid_fast = ReIdentifier(team_classifier=team_clf, max_plausible_speed_mps=12.0, min_reid_gap_frames=1)
    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid_fast, team_clf, frame0, [blue_player], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid_fast, team_clf, frame0, [], fps, cal_info)

    # 1 second gap (10 frames at 10 fps), respawn 30 pixels away
    # in pixel space == 30 meters with identity homography == 30 m/s.
    frame1 = _make_frame()
    _paint(frame1, 40, 10, 50, 20, BLUE)
    respawned = _player(track_id=999, x1=40, y1=10, x2=50, y2=20)
    _run_pipeline_step(reid_fast, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 999, "30 m/s should be rejected by the speed gate"
    assert respawned.provenance == PositionProvenance.OBSERVED

    # --- Positive control: 5 m/s, should accept ---
    team_clf2 = JerseyColorTeamClassifier()
    reid_slow = ReIdentifier(team_classifier=team_clf2, max_plausible_speed_mps=12.0, min_reid_gap_frames=1)
    frame0b = _make_frame()
    _paint(frame0b, 10, 10, 20, 20, BLUE)
    blue_player_2 = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid_slow, team_clf2, frame0b, [blue_player_2], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid_slow, team_clf2, frame0b, [], fps, cal_info)

    # 1 second gap, respawn 5 pixels right == 5 m/s.
    frame1b = _make_frame()
    _paint(frame1b, 15, 10, 25, 20, BLUE)
    respawned_2 = _player(track_id=999, x1=15, y1=10, x2=25, y2=20)
    _run_pipeline_step(reid_slow, team_clf2, frame1b, [respawned_2], fps, cal_info)

    assert respawned_2.track_id == 7, "5 m/s should be accepted by the speed gate"
    assert respawned_2.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP


def test_max_gap_exceeded_rejects():
    """A respawning track with matching color but past max_reid_gap_s
    should NOT be re-linked (the dropped history was garbage-collected).
    Not affected by the gap floor -- the 10-second gap here is already
    past garbage collection at the default max_reid_gap_s=1.0 long
    before the floor would even be checked."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, max_reid_gap_s=1.0)  # 1 second window
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)

    # 100 missed frames = 10 seconds. 9x past max_reid_gap_s=1.0.
    for _ in range(100):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    # Respawns with matching color. Should be too stale to match.
    frame1 = _make_frame()
    _paint(frame1, 10, 10, 20, 20, BLUE)
    respawned = _player(track_id=999, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 999
    assert respawned.provenance == PositionProvenance.OBSERVED


def test_full_gap_then_reattach_scenario():
    """End-to-end: two players established, one drops, a new detection
    that looks like the dropped player reattaches, the other player's
    track_id is untouched. min_reid_gap_frames=1 isolates this
    end-to-end scenario from the gap floor -- the 5-frame gap here is
    intentionally shorter than the default floor of 30."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=1)
    fps = 10.0
    cal_info = _calibrated()

    # Frame 0: player A (blue, id=7) and player B (red, id=11).
    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    _paint(frame0, 60, 10, 70, 20, RED)
    player_a = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    player_b = _player(track_id=11, x1=60, y1=10, x2=70, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [player_a, player_b], fps, cal_info)

    # Player A disappears for 5 frames (0.5s); player B persists.
    for i in range(5):
        frame = _make_frame()
        _paint(frame, 60, 10, 70, 20, RED)
        b = _player(track_id=11, x1=60, y1=10, x2=70, y2=20)
        _run_pipeline_step(reid, team_clf, frame, [b], fps, cal_info)

    # Player A respawns as a new track.
    frame2 = _make_frame()
    _paint(frame2, 11, 10, 21, 20, BLUE)
    _paint(frame2, 60, 10, 70, 20, RED)
    respawned_a = _player(track_id=42, x1=11, y1=10, x2=21, y2=20)
    b_again = _player(track_id=11, x1=60, y1=10, x2=70, y2=20)
    _run_pipeline_step(reid, team_clf, frame2, [respawned_a, b_again], fps, cal_info)

    assert respawned_a.track_id == 7, "player A should be re-stamped to its original id"
    assert respawned_a.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP
    assert b_again.track_id == 11, "player B's track_id should be untouched"
    assert b_again.provenance == PositionProvenance.OBSERVED


def test_ball_and_referee_never_re_identified():
    """Ball and referee are structurally excluded from re-ID. Even with
    a perfect color match and a plausible position, no re-linking
    should occur. Unaffected by the gap floor -- class exclusion is
    checked before candidates are ever built."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf)
    fps = 10.0
    cal_info = _calibrated()

    # --- Ball ---
    frame0 = _make_frame()
    _paint(frame0, 10, 10, 14, 14, YELLOW)
    ball_1 = _player(track_id=1, x1=10, y1=10, x2=14, y2=14, object_class=ObjectClass.BALL)
    _run_pipeline_step(reid, team_clf, frame0, [ball_1], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    frame1 = _make_frame()
    _paint(frame1, 11, 10, 15, 14, YELLOW)
    ball_2 = _player(track_id=99, x1=11, y1=10, x2=15, y2=14, object_class=ObjectClass.BALL)
    _run_pipeline_step(reid, team_clf, frame1, [ball_2], fps, cal_info)
    assert ball_2.track_id == 99
    assert ball_2.provenance == PositionProvenance.OBSERVED

    # --- Referee ---
    reid2 = ReIdentifier(team_classifier=JerseyColorTeamClassifier())
    frame0r = _make_frame()
    _paint(frame0r, 10, 10, 20, 20, YELLOW)
    ref_1 = _player(track_id=2, x1=10, y1=10, x2=20, y2=20, object_class=ObjectClass.REFEREE)
    _run_pipeline_step(reid2, team_clf, frame0r, [ref_1], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid2, team_clf, frame0r, [], fps, cal_info)

    frame1r = _make_frame()
    _paint(frame1r, 11, 10, 21, 20, YELLOW)
    ref_2 = _player(track_id=100, x1=11, y1=10, x2=21, y2=20, object_class=ObjectClass.REFEREE)
    _run_pipeline_step(reid2, team_clf, frame1r, [ref_2], fps, cal_info)
    assert ref_2.track_id == 100
    assert ref_2.provenance == PositionProvenance.OBSERVED


def test_cross_class_match_rejected():
    """A dropped PLAYER cannot re-link to a respawning GOALKEEPER even
    with identical color. Unaffected by the gap floor -- class mismatch
    is a separate, unconditional filter on the candidate set."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf)
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20, object_class=ObjectClass.PLAYER)
    _run_pipeline_step(reid, team_clf, frame0, [player], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    frame1 = _make_frame()
    _paint(frame1, 11, 10, 21, 20, BLUE)
    keeper = _player(track_id=99, x1=11, y1=10, x2=21, y2=20, object_class=ObjectClass.GOALKEEPER)
    _run_pipeline_step(reid, team_clf, frame1, [keeper], fps, cal_info)

    assert keeper.track_id == 99
    assert keeper.provenance == PositionProvenance.OBSERVED


def test_color_history_requested_from_team_classifier():
    """When the team classifier has a color cached for the dropped
    track, re-ID should use that cached color (most-recent).
    min_reid_gap_frames=1 isolates this from the gap floor -- the
    9-frame gap here is intentionally shorter than the default of 30."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=1)
    fps = 10.0
    cal_info = _calibrated()

    # Frame 0: blue player.
    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)

    # Verify the team classifier's cache now has the blue color.
    cached = team_clf.get_color_history(7)
    assert cached is not None
    assert len(cached) >= 1

    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    # Respawn with the same blue. Re-ID must use the team-classifier
    # cache as its source of truth for the old color.
    frame1 = _make_frame()
    _paint(frame1, 11, 10, 21, 20, BLUE)
    respawned = _player(track_id=999, x1=11, y1=10, x2=21, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 7
    assert respawned.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP


def test_reset_clears_all_history():
    """After reset(), a respawning track that would otherwise match
    should NOT be re-linked -- the history is gone. min_reid_gap_frames=1
    isolates this from the gap floor, same reasoning as elsewhere."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=1)
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    reid.reset()

    frame1 = _make_frame()
    _paint(frame1, 11, 10, 21, 20, BLUE)
    respawned = _player(track_id=999, x1=11, y1=10, x2=21, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 999
    assert respawned.provenance == PositionProvenance.OBSERVED


def test_speed_gate_skipped_when_uncalibrated():
    """When the calibration_info has no homography (NOT_CALIBRATED),
    the speed gate is skipped -- even a far-away respawn with a
    matching color should be re-linked. min_reid_gap_frames=1 isolates
    this from the gap floor -- the 9-frame gap here is intentionally
    shorter than the default of 30."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=1)
    fps = 10.0
    cal_info = _uncalibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    # Respawns far away (50 pixels right). With a homography that
    # would imply 50 m/s, but with no homography the speed gate is
    # skipped, so the color gate alone is enough.
    frame1 = _make_frame()
    _paint(frame1, 60, 10, 70, 20, BLUE)
    respawned = _player(track_id=999, x1=60, y1=10, x2=70, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 7
    assert respawned.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP


def test_appearance_embedder_seam_is_called_when_provided():
    """A custom appearance_embedder is invoked and its threshold
    enforced. min_reid_gap_frames=1 isolates this from the gap floor --
    the 9-frame gap here is intentionally shorter than the default of
    30."""
    team_clf = JerseyColorTeamClassifier()
    call_log = []

    def fake_embedder(new_obj, old_obj, new_color, old_color):
        call_log.append((new_obj.track_id, old_obj.track_id))
        # Always return 0.5 -- above the threshold below, so it
        # doesn't reject, but we can verify it was called.
        return 0.5

    reid = ReIdentifier(
        team_classifier=team_clf,
        appearance_embedder=fake_embedder,
        appearance_match_threshold=0.0,
        min_reid_gap_frames=1,
    )
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)
    for _ in range(9):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    frame1 = _make_frame()
    _paint(frame1, 11, 10, 21, 20, BLUE)
    respawned = _player(track_id=999, x1=11, y1=10, x2=21, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 7
    assert respawned.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP
    # Embedder was called at least once (could be more if it was
    # invoked during the alive-track refresh path -- we just check
    # it was called during the match).
    assert len(call_log) >= 1
    # The most recent call should involve the new (respawned) track
    # and the old (original) track.
    assert any(new == 999 and old == 7 for new, old in call_log), \
        f"expected a call with (999, 7), got {call_log}"


# ---- gap floor tests (min_reid_gap_frames) ----
#
# These lock in the actual bug fix: re-ID must not compete with
# ByteTracker's own recovery window on short gaps. Both tests below use
# the DEFAULT min_reid_gap_frames=30 deliberately -- unlike every test
# above, which overrides it to 1 to isolate a different gate.


def test_gap_shorter_than_floor_is_rejected_even_with_perfect_match():
    """A track dropped for fewer frames than min_reid_gap_frames must
    NOT be re-identified, even with an identical color and zero
    displacement (the easiest possible match). This is the exact bug
    reported against real footage: a 1-frame occlusion was winning
    against ByteTracker's own recovery and getting mis-stamped to an
    unrelated older track."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf)  # default min_reid_gap_frames=30
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)

    # Only 5 missed frames -- well short of the default floor of 30.
    for _ in range(5):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    # Respawns at the exact same position with the exact same color --
    # the easiest possible match on every other gate.
    frame1 = _make_frame()
    _paint(frame1, 10, 10, 20, 20, BLUE)
    respawned = _player(track_id=999, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 999, "a gap shorter than min_reid_gap_frames must not be re-identified"
    assert respawned.provenance == PositionProvenance.OBSERVED


def test_gap_at_floor_threshold_is_accepted():
    """A track dropped for exactly min_reid_gap_frames should be
    eligible for re-identification (the floor is a >=, not a strict >)."""
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf)  # default min_reid_gap_frames=30
    fps = 10.0
    cal_info = _calibrated()

    frame0 = _make_frame()
    _paint(frame0, 10, 10, 20, 20, BLUE)
    blue_player = _player(track_id=7, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame0, [blue_player], fps, cal_info)

    # Exactly 29 missed frames, so the respawn attempt below lands on
    # the 30th frame since last-seen -- exactly at the default floor.
    for _ in range(29):
        _run_pipeline_step(reid, team_clf, frame0, [], fps, cal_info)

    frame1 = _make_frame()
    _paint(frame1, 10, 10, 20, 20, BLUE)
    respawned = _player(track_id=999, x1=10, y1=10, x2=20, y2=20)
    _run_pipeline_step(reid, team_clf, frame1, [respawned], fps, cal_info)

    assert respawned.track_id == 7, "a gap exactly at min_reid_gap_frames should be accepted"
    assert respawned.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP