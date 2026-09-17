"""
End-to-end video-loop driver: reads a video file frame by frame, runs it
through `MatchPipeline` (Phase 1-3, already fully wired), writes the
resulting per-frame tracks to CSV via `FrameResultCSVWriter`, and
optionally renders a radar-only video via `stats/radar.py` for visual
sanity-checking.

This is a driver script, not a library module -- same role as
`examples/demo_schema_usage.py`, just exercising the real pipeline instead
of hand-built sample data. It owns wiring and CLI plumbing ONLY; every
piece of actual logic (detection, tracking, calibration, re-ID, transform,
radar rendering, CSV export) already exists and is independently tested
elsewhere. If something here looks like real logic, it's misplaced --
open an issue against the module it actually belongs in instead of fixing
it here.

Run as a module (so the relative imports below resolve the same way they
do for the test suite), from the directory ONE LEVEL ABOVE the
`football_analysis` package:

    python -m football_analysis.examples.run_match_pipeline \\
        --video path/to/match.mp4 \\
        --keypoint-model weights/soccana_keypoint.pt \\
        --detection-model weights/player_ball.pt \\
        --class-id-map 0:ball,1:goalkeeper,2:player,3:referee \\
        --output-csv tracks.csv \\
        --output-radar-video radar.mp4

Known gaps, left for later (not silently papered over here):
  - No scene-cut detection. `MatchPipeline.reset()` exists specifically
    for that case, but nothing in this loop calls it -- a hard cut mid-
    video will carry tracker/re-ID state across the cut incorrectly until
    a scene-cut detector is built and wired in here.
  - No low-angle keypoint model exists yet (per the project status), so
    `--low-angle-keypoint-model` is optional; omitting it means
    `KeypointModelRegistry` transparently falls back to the broadcast
    model for LOW_ANGLE_CORNER frames, exactly as designed.
  - The real `class_id_map` for your trained detection model hasn't been
    finalized -- `--class-id-map` is a required CLI arg specifically so
    nothing here guesses at it.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2

from ..calibration.keypoint_model import YoloKeypointModel

from ..calibration.calibrator import PitchCalibrator
from ..calibration.camera_motion_tracker import CameraMotionTracker
from ..calibration.model_registry import KeypointModelRegistry
from ..calibration.orchestrator import VideoCalibrationOrchestrator
from ..calibration.pipeline import CalibrationPipeline
from ..calibration.propagating_calibrator import PropagatingCalibrator
from ..calibration.scene_types import SceneType
from ..config.pitch_config import PitchConfig
from ..coordinates.export import FrameResultCSVWriter
from ..schemas.enums import ObjectClass
from ..stats.radar import get_canvas_size, render_radar_frame
from ..tracking.detector import YoloObjectDetector
from ..tracking.reid import ReIdentifier
from ..tracking.team_classifier import JerseyColorTeamClassifier
from ..tracking.tracker import ByteTracker
from ..video_pipeline import MatchPipeline

# Fallback used only when the video container doesn't report a usable fps
# (some codecs/containers report 0 via cv2.CAP_PROP_FPS) -- affects both
# per-frame timestamp_s computation and the output radar video's fps, and
# ReIdentifier's internal gap-timing math, so getting a real fps from the
# source is always preferable to relying on this.
FALLBACK_FPS = 25.0


def parse_class_id_map(raw: str) -> Dict[int, ObjectClass]:
    """Parses the project's established inline colon-separated CLI
    convention for a class-ID map (e.g. "0:ball,1:player,2:goalkeeper,
    3:referee") -- same format already used elsewhere in this project to
    avoid Windows shell quoting issues with JSON-shaped arguments.

    Raises ValueError with a clear message on malformed input or an
    unrecognized class name, rather than silently skipping a bad entry --
    a mistyped class-ID map here would silently mislabel every detection
    of that class for the entire run, exactly the failure mode
    `YoloObjectDetector`'s docstring warns about."""
    mapping: Dict[int, ObjectClass] = {}
    for raw_pair in raw.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Malformed --class-id-map entry {pair!r} -- expected "
                f"'<int>:<class_name>', e.g. '0:ball'."
            )
        id_part, name_part = pair.split(":", 1)
        try:
            class_id = int(id_part.strip())
        except ValueError:
            raise ValueError(f"Class ID {id_part!r} in --class-id-map is not an integer.")
        name = name_part.strip().lower()
        try:
            object_class = ObjectClass(name)
        except ValueError:
            valid = ", ".join(c.value for c in ObjectClass)
            raise ValueError(f"Unrecognized class name {name!r} in --class-id-map. Valid: {valid}.")
        mapping[class_id] = object_class
    if not mapping:
        raise ValueError("--class-id-map produced an empty mapping.")
    return mapping


def build_pitch_config(args: argparse.Namespace) -> PitchConfig:
    """Only overrides the fields the caller actually passed -- every
    PitchConfig field otherwise keeps its FIFA-standard default, so a
    caller who only cares about pitch_length doesn't need to know or
    supply every other geometry field."""
    overrides = {}
    if args.pitch_length is not None:
        overrides["pitch_length"] = args.pitch_length
    if args.pitch_width is not None:
        overrides["pitch_width"] = args.pitch_width
    return PitchConfig(**overrides)


def build_match_pipeline(args: argparse.Namespace, pitch_config: PitchConfig, video_fps: float) -> MatchPipeline:
    """Constructs every Phase 1-3 component and wires them into a single
    MatchPipeline, in the order MatchPipeline itself expects them
    constructed (though MatchPipeline.process_frame -- not this function
    -- is what enforces call ORDER at runtime)."""

    # ---- Phase 1: calibration ----
    model_paths = {SceneType.BROADCAST_WIDE: args.keypoint_model}
    if args.low_angle_keypoint_model:
        model_paths[SceneType.LOW_ANGLE_CORNER] = args.low_angle_keypoint_model
    model_registry = KeypointModelRegistry(model_paths=model_paths, model_loader=lambda path: YoloKeypointModel(path, device=args.device))

    calibration_pipeline = CalibrationPipeline(pitch_config, model_registry)

    # Explicit PitchCalibrator INSTANCE, not the PitchConfig itself --
    # PropagatingCalibrator needs something with a .calibrate_frame()
    # method. Passing PitchConfig here directly was a real bug caught
    # previously in this project; this is the fix, applied at
    # construction time so it can't recur.
    base_calibrator = PitchCalibrator(pitch_config)
    camera_motion_tracker = CameraMotionTracker()
    propagating_calibrator = PropagatingCalibrator(
        base_calibrator, camera_motion_tracker, max_propagated_frames=args.max_propagated_frames,
    )

    calibration_orchestrator = VideoCalibrationOrchestrator(calibration_pipeline, propagating_calibrator)

    # ---- Phase 2: detection, tracking, team classification, re-ID ----
    class_id_map = parse_class_id_map(args.class_id_map)
    detector = YoloObjectDetector(
        args.detection_model, class_id_map=class_id_map,
        confidence_threshold=args.detection_confidence_threshold,
        device=args.device,
    )

    tracker = ByteTracker(max_age=args.tracker_max_age, min_hits=args.tracker_min_hits)

    team_classifier = JerseyColorTeamClassifier()

    # Structurally coupled to tracker.max_age by construction (not an
    # independently-tuned number) -- see ReIdentifier's own docstring for
    # why: without this, re-ID can steal a track_id ByteTracker's own
    # stage-2 recovery pass was already about to handle internally.
    reid = ReIdentifier(team_classifier, min_reid_gap_frames=tracker.max_age)

    return MatchPipeline(
        detector=detector,
        tracker=tracker,
        team_classifier=team_classifier,
        reid=reid,
        calibration_orchestrator=calibration_orchestrator,
        pitch_config=pitch_config,
        video_fps=video_fps,
    )


def resolve_video_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # covers 0, negative, and NaN
        print(
            f"Warning: source video reported an unusable fps ({fps!r}); "
            f"falling back to {FALLBACK_FPS}. This affects timestamp_s, "
            f"the output radar video's fps, and ReIdentifier's internal "
            f"gap-timing math -- fix the source file/container if this "
            f"matters for your run.",
            file=sys.stderr,
        )
        return FALLBACK_FPS
    return float(fps)


def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    video_fps = resolve_video_fps(cap)
    pitch_config = build_pitch_config(args)
    match_pipeline = build_match_pipeline(args, pitch_config, video_fps)

    radar_writer: Optional[cv2.VideoWriter] = None
    if args.output_radar_video:
        canvas_size = get_canvas_size(pitch_config, args.radar_pixels_per_meter, args.radar_margin_m)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        radar_writer = cv2.VideoWriter(args.output_radar_video, fourcc, video_fps, canvas_size)
        if not radar_writer.isOpened():
            raise RuntimeError(
                f"Could not open radar output video for writing: "
                f"{args.output_radar_video} (codec availability varies by "
                f"OpenCV build/platform -- try a different extension/codec "
                f"if this fails)."
            )

    frame_index = 0
    total_rows = 0
    start_time = time.monotonic()

    try:
        with FrameResultCSVWriter(args.output_csv) as csv_writer:
            while True:
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_s = frame_index / video_fps
                output = match_pipeline.process_frame(frame, frame_index, timestamp_s)
                total_rows += csv_writer.write_frame(output.frame_result)

                if radar_writer is not None:
                    radar_frame = render_radar_frame(
                        output.frame_result.tracked_objects, pitch_config,
                        pixels_per_meter=args.radar_pixels_per_meter,
                        margin_m=args.radar_margin_m,
                    )
                    radar_writer.write(radar_frame)

                    if args.display:
                        cv2.imshow("Radar", radar_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Display window closed by user (q) -- stopping early.")
                            break

                if frame_index % args.log_every == 0:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"frame {frame_index} | scene={output.scene_classification.scene_type.value} "
                        f"| calibration={output.frame_result.calibration.status.value} "
                        f"| tracked_objects={len(output.frame_result.tracked_objects)} "
                        f"| rows_so_far={total_rows} | elapsed={elapsed:.1f}s"
                    )

                frame_index += 1
    finally:
        cap.release()
        if radar_writer is not None:
            radar_writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    print(
        f"Done. Processed {frame_index} frames in {elapsed:.1f}s "
        f"({frame_index / elapsed if elapsed > 0 else 0:.1f} fps). "
        f"Wrote {total_rows} rows to {args.output_csv}."
    )
    if radar_writer is not None:
        print(f"Radar video written to {args.output_radar_video}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full football_analysis pipeline over a video file, "
                    "writing per-frame tracks to CSV and (optionally) a radar-view video."
    )

    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--keypoint-model", required=True, help="Path to broadcast keypoint model weights (.pt).")
    parser.add_argument("--low-angle-keypoint-model", default=None,
                        help="Path to low-angle keypoint model weights (.pt). Omit if not trained yet -- "
                             "the registry falls back to the broadcast model automatically.")
    parser.add_argument("--detection-model", required=True, help="Path to player/ball detection model weights (.pt).")
    parser.add_argument("--class-id-map", required=True,
                        help="Inline colon-separated class-ID map, e.g. '0:ball,1:goalkeeper,2:player,3:referee'.")

    parser.add_argument("--output-csv", default="tracks.csv", help="Output CSV path (default: tracks.csv).")
    parser.add_argument("--output-radar-video", default=None,
                        help="Optional output path for a radar-only .mp4. Omit to skip radar rendering entirely.")
    parser.add_argument("--display", action="store_true",
                        help="Show a live radar preview window while processing (requires --output-radar-video "
                             "or renders radar frames purely for display if omitted). Press 'q' to stop early.")

    parser.add_argument("--pitch-length", type=float, default=None, help="Override pitch length in meters.")
    parser.add_argument("--pitch-width", type=float, default=None, help="Override pitch width in meters.")

    parser.add_argument("--max-propagated-frames", type=int, default=None,
                        help="Cap on consecutive PROPAGATED frames before falling back to NOT_CALIBRATED. "
                             "Default: no cap.")
    parser.add_argument("--detection-confidence-threshold", type=float, default=0.3)
    parser.add_argument("--tracker-max-age", type=int, default=30)
    parser.add_argument("--tracker-min-hits", type=int, default=3)
    parser.add_argument("--device", default=None, help="Inference device for both models, e.g. 'cuda:0'. Default: ultralytics auto-detect.")

    parser.add_argument("--radar-pixels-per-meter", type=float, default=10.0)
    parser.add_argument("--radar-margin-m", type=float, default=3.0)

    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames (for quick tests).")
    parser.add_argument("--log-every", type=int, default=50, help="Print progress every N frames.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # --display without --output-radar-video still renders radar frames
    # for the preview window, just without writing them to disk -- doing
    # that render is only worth the cost if the caller actually asked to
    # see something, so we validate the combination rather than silently
    # rendering nothing.
    if args.display and not args.output_radar_video:
        print(
            "Note: --display was set without --output-radar-video -- "
            "a radar frame will still be rendered every frame for the "
            "preview window, but nothing will be saved to disk.",
            file=sys.stderr,
        )

    if not Path(args.video).exists():
        raise SystemExit(f"--video path does not exist: {args.video}")

    run(args)


if __name__ == "__main__":
    main()