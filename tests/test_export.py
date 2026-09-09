import csv
import io
from pathlib import Path

import pytest

from ..schemas.enums import CalibrationStatus, ObjectClass, Team, PositionProvenance
from ..schemas.frame_result import (
    CalibrationInfo, PixelPoint, WorldPoint, TrackedObject, FrameResult,
)
from ..coordinates.export import FrameResultCSVWriter, export_frames_to_csv, FIELDNAMES


def _frame_with_objects(frame_index=0, timestamp_s=0.0) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        calibration=CalibrationInfo(
            status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
            frames_since_last_anchor=0,
        ),
        tracked_objects=[
            TrackedObject(
                track_id=1,
                object_class=ObjectClass.PLAYER,
                team=Team.HOME,
                jersey_number=9,
                pixel_position=PixelPoint(x=100.0, y=200.0),
                world_position=WorldPoint(x=10.0, y=20.0),
                detection_confidence=0.9,
                position_confidence=1.0,
                provenance=PositionProvenance.OBSERVED,
            ),
            TrackedObject(
                track_id=2,
                object_class=ObjectClass.BALL,
                pixel_position=PixelPoint(x=105.0, y=205.0),
                world_position=None,
                provenance=PositionProvenance.OBSERVED,
            ),
        ],
    )


def _empty_frame(frame_index=1, timestamp_s=0.04) -> FrameResult:
    return FrameResult(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        calibration=CalibrationInfo(status=CalibrationStatus.NOT_CALIBRATED),
        tracked_objects=[],
    )


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_header_matches_fieldnames(tmp_path):
    path = tmp_path / "tracks.csv"
    with FrameResultCSVWriter(path):
        pass  # header written on __enter__, before any frame

    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == FIELDNAMES


def test_write_frame_returns_row_count(tmp_path):
    path = tmp_path / "tracks.csv"
    with FrameResultCSVWriter(path) as writer:
        count = writer.write_frame(_frame_with_objects())
    assert count == 2


def test_empty_frame_contributes_zero_rows(tmp_path):
    path = tmp_path / "tracks.csv"
    with FrameResultCSVWriter(path) as writer:
        n1 = writer.write_frame(_frame_with_objects(frame_index=0))
        n2 = writer.write_frame(_empty_frame(frame_index=1))

    assert n1 == 2
    assert n2 == 0
    rows = _read_rows(path)
    assert len(rows) == 2  # empty frame added no rows at all
    assert {r["frame_index"] for r in rows} == {"0"}  # frame 1 never appears


def test_row_values_match_flat_record_content(tmp_path):
    path = tmp_path / "tracks.csv"
    frame = _frame_with_objects()
    with FrameResultCSVWriter(path) as writer:
        writer.write_frame(frame)

    rows = _read_rows(path)
    player_row = next(r for r in rows if r["track_id"] == "1")

    assert player_row["object_class"] == "player"
    assert player_row["team"] == "home"
    assert player_row["jersey_number"] == "9"
    assert player_row["pixel_x"] == "100.0"
    assert player_row["world_x_m"] == "10.0"
    assert player_row["calibration_status"] == "calibrated_from_keypoints"
    assert player_row["provenance"] == "observed"


def test_none_values_render_as_empty_string(tmp_path):
    path = tmp_path / "tracks.csv"
    with FrameResultCSVWriter(path) as writer:
        writer.write_frame(_frame_with_objects())

    rows = _read_rows(path)
    ball_row = next(r for r in rows if r["track_id"] == "2")

    assert ball_row["world_x_m"] == ""  # None -> empty CSV field
    assert ball_row["team"] == ""
    assert ball_row["jersey_number"] == ""


def test_multiple_frames_accumulate_in_order(tmp_path):
    path = tmp_path / "tracks.csv"
    with FrameResultCSVWriter(path) as writer:
        writer.write_frame(_frame_with_objects(frame_index=0, timestamp_s=0.0))
        writer.write_frame(_frame_with_objects(frame_index=1, timestamp_s=0.04))

    rows = _read_rows(path)
    assert len(rows) == 4
    assert [r["frame_index"] for r in rows] == ["0", "0", "1", "1"]


def test_write_frame_without_context_manager_raises():
    writer = FrameResultCSVWriter("unused.csv")
    with pytest.raises(RuntimeError):
        writer.write_frame(_frame_with_objects())


def test_file_is_readable_mid_run_before_exit(tmp_path):
    """Proves the flush-per-frame behavior: content must be on disk and
    readable even before the writer's context manager has exited, since
    that's the whole point of streaming (a crash mid-run shouldn't lose
    already-written frames sitting in an unflushed buffer)."""
    path = tmp_path / "tracks.csv"
    with FrameResultCSVWriter(path) as writer:
        writer.write_frame(_frame_with_objects(frame_index=0))
        # Read the file WHILE still inside the context manager, i.e.
        # before __exit__ has had a chance to close/flush anything itself.
        rows_mid_run = _read_rows(path)

    assert len(rows_mid_run) == 2


def test_export_frames_to_csv_with_list(tmp_path):
    path = tmp_path / "tracks.csv"
    frames = [_frame_with_objects(frame_index=0), _frame_with_objects(frame_index=1)]

    total = export_frames_to_csv(frames, path)

    assert total == 4
    rows = _read_rows(path)
    assert len(rows) == 4


def test_export_frames_to_csv_with_generator(tmp_path):
    """Must accept a generator, not just a list -- a real frame-processing
    loop shouldn't be forced to materialize the whole match in memory
    first just to call export."""
    path = tmp_path / "tracks.csv"

    def frame_stream():
        yield _frame_with_objects(frame_index=0)
        yield _empty_frame(frame_index=1)
        yield _frame_with_objects(frame_index=2)

    total = export_frames_to_csv(frame_stream(), path)

    assert total == 4  # 2 + 0 + 2
    rows = _read_rows(path)
    assert len(rows) == 4


def test_export_frames_to_csv_accepts_path_as_string(tmp_path):
    path_str = str(tmp_path / "tracks.csv")
    total = export_frames_to_csv([_frame_with_objects()], path_str)
    assert total == 2
    assert Path(path_str).exists()