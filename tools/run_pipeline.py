"""
End-to-end executable for the football_analysis pipeline.

Run a full broadcast video through:
  Phase 1: scene-routing + keypoint calibration + camera-motion propagation
  Phase 2: YOLO player/ball detection + ByteTrack tracking +
           jersey-color team classification + re-identification after gaps
  Phase 3: per-object pixel->world projection

and write:
  <output-dir>/tracks.csv      one row per (frame_index, track_id) per
                               frame, via FrameResult.to_flat_records()
  <output-dir>/debug.mp4       (optional, --no-debug-video to skip)
                               per-frame annotated overlay
  <output-dir>/metadata.json   MatchMetadata sidecar (pitch, model
                               versions, processed frame count)

This is the first real end-to-end executable. The CLI uses argparse (no
existing CLI convention in this project; tools/distance_tool.py is
interactive, examples/demo_schema_usage.py is a sanity check). The CSV
writer uses stdlib `csv` only -- no pandas, no pyarrow, no new
dependencies. metadata.json uses the existing `MatchMetadata` schema
written via pydantic's `model_dump_json()`. FrameResult rows are
produced via the schema's `to_flat_records()` so the export is
contract-stable.

The per-frame pipeline order is:
  1. orchestrator.process_frame(frame)        -- calibration
  2. detector.detect(frame)                  -- YOLO detections
  3. tracker.update(detections)              -- ByteTrack
  4. team_clf.assign_teams(frame, tracked)   -- mutates obj.team,
                                                populates _color_history
  5. reid.update(tracked, cal_info, fps)     -- re-stamps track_id +
                                                provenance where applicable
  6. world projection per tracked object     -- via orchestrator.pixel_to_world
  7. FrameResult + to_flat_records + CSV write
  8. (optional) draw_overlay + VideoWriter.write

position_confidence (per TrackedObject) is computed in step 6:
  - 1.0 for CALIBRATED_FROM_KEYPOINTS / RE_ANCHORED
  - 0.6 for PROPAGATED
  - 0.0 for NOT_CALIBRATED (or no homography)
  - multiplied by 0.6 on top if provenance == RE_IDENTIFIED_AFTER_GAP,
    because a reappearance is genuinely less trustworthy than a
    continuous track even when the new frame's pixel observation is
    high-confidence.

This is the v1 deliverable. Future work:
  - pandas / pyarrow exporters for Phase 4 stats
  - a real appearance-embedding tiebreaker for re-ID
  - a low-angle keypoint model once trained
  - real-data-driven threshold tuning
"""

import argparse
import csv
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calibration.calibrator import PitchCalibrator
from ..calibration.model_registry import KeypointModelRegistry
from ..calibration.orchestrator import VideoCalibrationOrchestrator
from ..calibration.pipeline import CalibrationPipeline
from ..calibration.propagating_calibrator import PropagatingCalibrator
from ..calibration.scene_types import SceneType
from ..calibration.smoothed_calibrator import SmoothedPitchCalibrator
from ..config.pitch_config import PitchConfig
from ..schemas.enums import CalibrationStatus, ObjectClass, PositionProvenance, Team
from ..schemas.frame_result import FrameResult, TrackedObject
from ..schemas.match_metadata import MatchMetadata, ModelVersions
from ..tracking.detector import YoloObjectDetector
from ..tracking.reid import ReIdentifier
from ..tracking.team_classifier import JerseyColorTeamClassifier
from ..tracking.tracker import ByteTracker


# CLI help text strings kept module-level so the parser is readable
# without scrolling through the body of main().
VIDEO_HELP = "Path to input video file (e.g. broadcast .mp4)."
KEYPOINT_MODEL_HELP = "Path to Soccana YOLO keypoint model weights (.pt)."
DETECTOR_MODEL_HELP = "Path to YOLO player/ball detection model weights (.pt)."
CLASS_ID_MAP_HELP = (
    'Detector class-id -> ObjectClass mapping, comma-separated "int:label" pairs '
    '(lowercase enum values). E.g. "0:ball,1:player,2:goalkeeper,3:referee".'
)
OUTPUT_DIR_HELP = "Directory to write tracks.csv, debug.mp4, metadata.json into."
NO_DEBUG_VIDEO_HELP = "Skip writing the annotated debug video (faster)."
MAX_FRAMES_HELP = "Stop after this many frames (for smoke tests / dev)."
SKIP_FRAMES_HELP = "Skip the first N frames of input before processing."


# Debug-video overlay color scheme. BGR for OpenCV.
TEAM_BGR = {
    Team.HOME: (255, 128, 0),    # blue
    Team.AWAY: (0, 0, 255),      # red
    None:       (200, 200, 200),  # gray: unassigned / referee / ball
}


# ---- CLI plumbing ----


def parse_class_id_map(s: str) -> Dict[int, ObjectClass]:
    """Parse '0:ball,1:player,2:goalkeeper,3:referee' into
    Dict[int, ObjectClass]. Bad input raises SystemExit with a clear
    message (argparse convention for CLI validation errors)."""
    label_to_class = {
        "ball": ObjectClass.BALL,
        "player": ObjectClass.PLAYER,
        "goalkeeper": ObjectClass.GOALKEEPER,
        "referee": ObjectClass.REFEREE,
    }
    out: Dict[int, ObjectClass] = {}
    for raw_pair in s.split(","):
        pair = raw_pair.strip()
        if ":" not in pair:
            raise SystemExit(
                f"--class-id-map: expected 'int:label' pairs, got {raw_pair!r}"
            )
        k, v = pair.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v not in label_to_class:
            raise SystemExit(
                f"--class-id-map: unknown label {v!r}; "
                f"must be one of {sorted(label_to_class)}"
            )
        try:
            int_k = int(k)
        except ValueError:
            raise SystemExit(
                f"--class-id-map: expected integer key, got {k!r}"
            )
        out[int_k] = label_to_class[v]
    if not out:
        raise SystemExit("--class-id-map: no valid pairs parsed")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Kept as a separate function so it
    can be unit-tested or imported for --help generation without
    triggering the full main() flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the football_analysis pipeline end-to-end on a broadcast "
            "video. Writes a per-frame CSV of tracked objects, an optional "
            "debug video with bounding boxes, and a metadata.json sidecar."
        )
    )
    parser.add_argument("--video", required=True, help=VIDEO_HELP)
    parser.add_argument("--keypoint-model", required=True, help=KEYPOINT_MODEL_HELP)
    parser.add_argument("--detector-model", required=True, help=DETECTOR_MODEL_HELP)
    parser.add_argument("--class-id-map", required=True, help=CLASS_ID_MAP_HELP)
    parser.add_argument("--pitch-length", type=float, default=105.0, help="Pitch length in meters (default 105, FIFA).")
    parser.add_argument("--pitch-width", type=float, default=68.0, help="Pitch width in meters (default 68, FIFA).")
    parser.add_argument("--output-dir", required=True, help=OUTPUT_DIR_HELP)
    parser.add_argument("--no-debug-video", action="store_true", help=NO_DEBUG_VIDEO_HELP)
    parser.add_argument("--max-frames", type=int, default=None, help=MAX_FRAMES_HELP)
    parser.add_argument("--skip-frames", type=int, default=0, help=SKIP_FRAMES_HELP)
    return parser


def iter_frames(cap: cv2.VideoCapture, skip: int, max_frames: Optional[int]):
    """Yield (frame_index, frame) from `cap`, skipping the first
    `skip` frames and stopping after `max_frames` frames. `max_frames`
    of None means "no limit"."""
    frame_idx = -1
    yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        frame_idx += 1
        if frame_idx < skip:
            continue
        yield frame_idx, frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            return


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    return cap


# ---- per-frame world projection + position_confidence ----


def _calibration_confidence(status: CalibrationStatus) -> float:
    """Base position_confidence derived from CalibrationInfo.status.
    Independent of detection_confidence (a perfectly-tracked player can
    have low position_confidence if calibration is propagated)."""
    if status in (CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
                  CalibrationStatus.RE_ANCHORED):
        return 1.0
    if status == CalibrationStatus.PROPAGATED:
        return 0.6
    return 0.0  # NOT_CALIBRATED


def _project_and_confidence(
    obj: TrackedObject,
    orchestrator: VideoCalibrationOrchestrator,
    status: CalibrationStatus,
) -> None:
    """Mutate `obj` in place: set `world_position` and
    `position_confidence`. Re-ID reappearance further discounts
    position_confidence on top of the calibration-derived value."""
    base_conf = _calibration_confidence(status)

    wp = orchestrator.pixel_to_world((obj.pixel_position.x, obj.pixel_position.y))
    if wp is not None:
        obj.world_position = wp

    final_conf = base_conf
    if obj.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP:
        # Identity is inferred; even with high-confidence pixel observation,
        # the trust in the world position is reduced.
        final_conf *= 0.6
    obj.position_confidence = final_conf


# ---- CSV writer (stdlib csv only) ----


CSV_COLUMNS = [
    "frame_index", "timestamp_s", "calibration_status", "frames_since_last_anchor",
    "track_id", "object_class", "team", "jersey_number",
    "pixel_x", "pixel_y", "world_x_m", "world_y_m",
    "detection_confidence", "position_confidence", "provenance",
]


class CsvWriter:
    """Append-only writer for FrameResult.to_flat_records() output.
    Stdlib csv only -- no pandas. Tracks the number of rows written for
    the final summary."""

    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self.row_count = 0

    def write_rows(self, rows: List[dict]) -> None:
        self._writer.writerows(rows)
        self.row_count += len(rows)

    def close(self) -> None:
        self._f.close()


# ---- Debug video overlay ----


def draw_overlay(
    frame: np.ndarray,
    tracked: List[TrackedObject],
    status: CalibrationStatus,
    frames_since_last_anchor: Optional[int],
    frame_idx: int,
    timestamp_s: float,
) -> np.ndarray:
    """Render an annotated copy of `frame` for the debug video.

    Per tracked object: a colored bounding box (color = team), and a
    label above the box ('#<track_id> <object_class> [team]' plus
    '[REID]' if provenance == RE_IDENTIFIED_AFTER_GAP). The '[REID]'
    suffix is the single visual signal a developer needs to confirm
    re-identification is firing on the right track.

    Top-left: calibration status (CALIBRATED / PROPAGATED (N frames) /
    RE_ANCHORED / NOT CALIBRATED). Bottom-left: frame index and
    timestamp. BGR colors throughout.
    """
    display = frame.copy()

    for obj in tracked:
        if obj.bounding_box is None:
            continue
        x1, y1 = int(obj.bounding_box.x1), int(obj.bounding_box.y1)
        x2, y2 = int(obj.bounding_box.x2), int(obj.bounding_box.y2)
        color = TEAM_BGR.get(obj.team, TEAM_BGR[None])
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        label = f"#{obj.track_id} {obj.object_class.value}"
        if obj.team is not None:
            label += f" {obj.team.value}"
        if obj.provenance == PositionProvenance.RE_IDENTIFIED_AFTER_GAP:
            label += " [REID]"
        cv2.putText(
            display, label, (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    # Calibration status, top-left.
    status_text = status.value.upper()
    if frames_since_last_anchor:
        status_text += f" ({frames_since_last_anchor} frames)"
    cv2.putText(
        display, status_text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    # Frame counter, bottom-left.
    cv2.putText(
        display, f"frame {frame_idx} t={timestamp_s:.2f}s",
        (10, display.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return display


# ---- Pipeline construction ----


def build_orchestrator(pitch: PitchConfig, keypoint_model_path: str) -> VideoCalibrationOrchestrator:
    """Wire the Phase 1 pipeline: model registry -> CalibrationPipeline
    (routing) -> SmoothedPitchCalibrator (jitter reduction) ->
    PropagatingCalibrator (gap bridging) -> VideoCalibrationOrchestrator
    (the single entrypoint)."""
    registry = KeypointModelRegistry(
        model_paths={SceneType.BROADCAST_WIDE: keypoint_model_path},
    )
    pipeline = CalibrationPipeline(
        pitch_config=pitch, model_registry=registry,
    )
    base_calibrator = PitchCalibrator(pitch)
    smooth_cal = SmoothedPitchCalibrator(base_calibrator)
    propagating = PropagatingCalibrator(pitch_calibrator=smooth_cal)
    return VideoCalibrationOrchestrator(
        pipeline=pipeline, propagating_calibrator=propagating,
    )


def build_phase2(class_id_map: Dict[int, ObjectClass], detector_model_path: str) -> Tuple[
    YoloObjectDetector, ByteTracker, JerseyColorTeamClassifier, ReIdentifier
]:
    """Wire the Phase 2 stack: YOLO detector -> ByteTracker ->
    JerseyColorTeamClassifier -> ReIdentifier (depends on the
    team_classifier instance for the per-track color cache)."""
    detector = YoloObjectDetector(
        model_path=detector_model_path, class_id_map=class_id_map,
    )
    tracker = ByteTracker()
    team_clf = JerseyColorTeamClassifier()
    reid = ReIdentifier(team_classifier=team_clf, min_reid_gap_frames=tracker.max_age)
    return detector, tracker, team_clf, reid


# ---- Main ----


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_id_map = parse_class_id_map(args.class_id_map)

    pitch = PitchConfig(
        pitch_length=args.pitch_length, pitch_width=args.pitch_width,
    )

    # --- Build the pipeline ---
    orchestrator = build_orchestrator(pitch, args.keypoint_model)
    detector, tracker, team_clf, reid = build_phase2(
        class_id_map, args.detector_model,
    )

    # --- Open the video ---
    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Open the CSV writer (always) and the optional debug-video writer ---
    csv_path = out_dir / "tracks.csv"
    video_path = out_dir / "debug.mp4"
    metadata_path = out_dir / "metadata.json"

    writer = CsvWriter(csv_path)
    video_writer: Optional[cv2.VideoWriter] = None
    if not args.no_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    print(f"  class_id_map: { {k: v.value for k, v in class_id_map.items()} }")
    print(f"  output_dir: {out_dir}")
    print(f"  no_debug_video: {args.no_debug_video}")
    print(f"  max_frames: {args.max_frames}, skip_frames: {args.skip_frames}")

    processed = 0
    try:
        for frame_idx, frame in iter_frames(
            cap, skip=args.skip_frames, max_frames=args.max_frames,
        ):
            timestamp_s = frame_idx / fps

            # --- Phase 1: calibration ---
            orch_result = orchestrator.process_frame(frame)
            cal_info = orch_result.calibration_info

            # --- Phase 2: detection + tracking + team classification + re-ID ---
            detections = detector.detect(frame)
            tracked = tracker.update(detections)
            team_clf.assign_teams(frame, tracked)
            reid.update(tracked, cal_info, fps)

            # --- Phase 3 (in-script for now): per-object world projection ---
            for obj in tracked:
                _project_and_confidence(obj, orchestrator, cal_info.status)

            # --- Emit per-frame output ---
            frame_result = FrameResult(
                frame_index=frame_idx,
                timestamp_s=timestamp_s,
                calibration=cal_info,
                tracked_objects=tracked,
            )
            writer.write_rows(frame_result.to_flat_records())

            if video_writer is not None:
                overlay = draw_overlay(
                    frame, tracked, cal_info.status,
                    cal_info.frames_since_last_anchor, frame_idx, timestamp_s,
                )
                video_writer.write(overlay)

            processed += 1
            if processed % 30 == 0:
                print(
                    f"  frame {frame_idx} @ {timestamp_s:.2f}s, "
                    f"{writer.row_count} rows so far"
                )
    finally:
        writer.close()
        if video_writer is not None:
            video_writer.release()
        cap.release()

    # --- Metadata sidecar (MatchMetadata) ---
    metadata = MatchMetadata(
        match_id=str(uuid.uuid4()),
        video_fps=fps,
        video_width=width,
        video_height=height,
        pitch_config=pitch,
        model_versions=ModelVersions(
            keypoint_model=Path(args.keypoint_model).name,
            detection_model=Path(args.detector_model).name,
            tracker="ByteTracker",
            reid="ReIdentifier",
        ),
    )
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    # --- Final summary ---
    print(f"Done. Processed {processed} frames.")
    print(f"  {csv_path} ({writer.row_count} rows)")
    if video_writer is not None:
        print(f"  {video_path}")
    print(f"  {metadata_path} (run_id={metadata.match_id}, "
          f"processed_at={datetime.datetime.utcnow().isoformat()}Z)")


if __name__ == "__main__":
    main()
