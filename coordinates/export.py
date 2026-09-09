"""
Phase 3 -- CSV export.

Serializes FrameResult objects (Phase 0's canonical schema) to CSV, one row
per tracked object per frame, using FrameResult.to_flat_records() -- that
flattening logic already lives on the schema itself (Phase 0), so this
module's only job is turning a stream of dicts into a well-formed CSV file.
It deliberately knows nothing about calibration, tracking, or the
transform layer -- it consumes FrameResult and nothing else, same
"stats/export code should never import from calibration/ or tracking/"
rule the whole project has followed since Phase 0.

Streaming by design: a full match can be tens of thousands of frames, so
this writes incrementally (one frame's rows at a time, flushed
immediately) rather than accepting the whole match in memory at once.
That also means a crash partway through a long run leaves a valid,
readable partial CSV instead of losing everything -- the CSV-writing
analogue of the "no silent staleness" principle used elsewhere in this
project.

Parquet is intentionally NOT supported here -- it would pull in
pyarrow/fastparquet for a format nothing downstream currently consumes.
If/when there's an actual need for it, add a ParquetExporter alongside
this one behind the same usage pattern; don't preemptively add the
dependency.
"""

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Union

from ..schemas.frame_result import FrameResult

# Explicit, fixed column order -- deliberately NOT inferred from the first
# frame's records. Inferring from the first frame is fragile: if frame 0
# happens to have no tracked objects (a NOT_CALIBRATED frame with a total
# detection miss, however rare), a header could get skipped or built from
# the wrong shape entirely. This list must stay in sync with the dict keys
# FrameResult.to_flat_records() produces.
FIELDNAMES: List[str] = [
    "frame_index",
    "timestamp_s",
    "calibration_status",
    "frames_since_last_anchor",
    "track_id",
    "object_class",
    "team",
    "jersey_number",
    "pixel_x",
    "pixel_y",
    "world_x_m",
    "world_y_m",
    "detection_confidence",
    "position_confidence",
    "provenance",
]


class FrameResultCSVWriter:
    """
    Incremental CSV writer for FrameResult objects. Use as a context
    manager so the file handle is always closed properly, even if the
    caller's frame loop raises partway through a match:

        with FrameResultCSVWriter("tracks.csv") as writer:
            for frame in frame_stream:
                writer.write_frame(frame)

    A frame with zero tracked_objects (e.g. a totally missed detection on
    a NOT_CALIBRATED frame) simply contributes zero rows -- there is no
    per-frame placeholder row. The CSV is object-centric, not frame-
    centric; a frame's absence from the file is recoverable by diffing
    frame_index values against your video's frame count if that ever
    matters, but isn't reconstructed here since nothing downstream has
    asked for it yet.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._file = None
        self._writer: Optional[csv.DictWriter] = None

    def __enter__(self) -> "FrameResultCSVWriter":
        self._file = open(self.path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        return self

    def write_frame(self, frame_result: FrameResult) -> int:
        """Writes this frame's tracked objects as rows. Returns the number
        of rows written (0 for a frame with no tracked objects), mainly so
        a caller can log/accumulate a running row count without needing to
        call to_flat_records() itself."""
        if self._writer is None:
            raise RuntimeError(
                "FrameResultCSVWriter must be used as a context manager "
                "(`with FrameResultCSVWriter(path) as writer:`) before "
                "write_frame() is called."
            )
        records = frame_result.to_flat_records()
        for record in records:
            self._writer.writerow(record)
        # Flush (not just rely on Python's internal buffering) after every
        # frame -- the whole point of streaming is that a crash mid-run
        # shouldn't lose already-processed frames sitting in an unflushed
        # buffer.
        self._file.flush()
        return len(records)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


def export_frames_to_csv(frames: Iterable[FrameResult], path: Union[str, Path]) -> int:
    """
    Convenience wrapper for the common case: you already have a complete
    Iterable[FrameResult] (a List, or any generator) and just want a CSV
    written in one call, without managing the writer object yourself.

    Accepts any Iterable rather than requiring a List specifically, so a
    generator-based frame-processing loop can be passed directly without
    forcing the caller to materialize the whole match into a list first.

    Returns the total number of rows written across all frames.
    """
    total_rows = 0
    with FrameResultCSVWriter(path) as writer:
        for frame in frames:
            total_rows += writer.write_frame(frame)
    return total_rows