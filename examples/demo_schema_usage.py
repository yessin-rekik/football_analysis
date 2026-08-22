"""
Demonstrates the Phase 0 schema end-to-end, standing in for what Phases 1-3
will eventually produce. Run this to sanity-check the schema shape and
serialization before any real calibration/tracking code exists.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from football_analysis.config import PitchConfig
from football_analysis.schemas import (
    CalibrationStatus, ObjectClass, Team, PositionProvenance,
    PixelPoint, WorldPoint, CalibrationInfo, TrackedObject, FrameResult,
    ModelVersions, MatchMetadata,
)


def main():
    # --- Phase 0: config ---
    pitch_config = PitchConfig()  # FIFA-standard defaults
    print(f"Pitch: {pitch_config.pitch_length}m x {pitch_config.pitch_width}m")

    world_kps = pitch_config.world_keypoints()
    print(f"world_keypoints shape: {world_kps.shape}")
    print(f"  field_center -> {tuple(world_kps[15])}  (expect ~52.5, 34.0)")
    print(f"  sideline_top_right -> {tuple(world_kps[16])}  (expect 105.0, 68.0)")

    # --- match metadata ---
    metadata = MatchMetadata(
        match_id="demo-match-001",
        video_fps=25.0,
        video_width=1920,
        video_height=1080,
        pitch_config=pitch_config,
        model_versions=ModelVersions(
            keypoint_model="soccana_keypoint_v1",
            detection_model="yolo_player_ball_v1",
            tracker="bytetrack_v1",
        ),
    )

    # --- a normal, well-calibrated frame ---
    frame_ok = FrameResult(
        frame_index=120,
        timestamp_s=4.8,
        calibration=CalibrationInfo(
            status=CalibrationStatus.CALIBRATED_FROM_KEYPOINTS,
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # placeholder
            reprojection_error_px_mean=2.3,
            reprojection_error_px_max=5.1,
            num_keypoints_used=9,
            frames_since_last_anchor=0,
        ),
        tracked_objects=[
            TrackedObject(
                track_id=17,
                object_class=ObjectClass.PLAYER,
                team=Team.HOME,
                jersey_number=9,
                pixel_position=PixelPoint(x=812.0, y=430.0),
                world_position=WorldPoint(x=58.2, y=31.4),
                detection_confidence=0.93,
                position_confidence=0.95,
                provenance=PositionProvenance.OBSERVED,
            ),
            TrackedObject(
                track_id=42,
                object_class=ObjectClass.BALL,
                pixel_position=PixelPoint(x=790.0, y=440.0),
                world_position=WorldPoint(x=56.9, y=30.8),
                detection_confidence=0.88,
                position_confidence=0.90,
                provenance=PositionProvenance.OBSERVED,
            ),
        ],
    )

    # --- a frame during a camera zoom, calibration propagated from motion tracking ---
    frame_propagated = FrameResult(
        frame_index=145,
        timestamp_s=5.8,
        calibration=CalibrationInfo(
            status=CalibrationStatus.PROPAGATED,
            homography=[[1.02, 0.01, 3.0], [0.0, 1.01, -1.5], [0.0, 0.0, 1.0]],  # placeholder
            frames_since_last_anchor=25,
        ),
        tracked_objects=[
            TrackedObject(
                track_id=17,
                object_class=ObjectClass.PLAYER,
                team=Team.HOME,
                jersey_number=9,
                pixel_position=PixelPoint(x=940.0, y=520.0),
                world_position=WorldPoint(x=61.0, y=29.9),
                detection_confidence=0.91,
                position_confidence=0.55,  # discounted: calibration is propagated, not anchored
                provenance=PositionProvenance.OBSERVED,
            ),
        ],
    )

    # --- a frame that is the reappearance point after a long tracking gap ---
    frame_reappear = FrameResult(
        frame_index=200,
        timestamp_s=8.0,
        calibration=CalibrationInfo(
            status=CalibrationStatus.RE_ANCHORED,
            homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            num_keypoints_used=11,
            frames_since_last_anchor=0,
        ),
        tracked_objects=[
            TrackedObject(
                track_id=17,  # same track_id -- re-identification matched it back
                object_class=ObjectClass.PLAYER,
                team=Team.HOME,
                jersey_number=9,
                pixel_position=PixelPoint(x=1100.0, y=600.0),
                world_position=WorldPoint(x=70.5, y=25.0),
                detection_confidence=0.9,
                position_confidence=0.6,  # identity link inferred, not certain
                provenance=PositionProvenance.RE_IDENTIFIED_AFTER_GAP,
            ),
        ],
    )

    frames = [frame_ok, frame_propagated, frame_reappear]

    print("\n--- MatchMetadata JSON ---")
    print(metadata.model_dump_json(indent=2)[:400], "...")

    print("\n--- FrameResult JSON (frame_ok) ---")
    print(frame_ok.model_dump_json(indent=2))

    print("\n--- flattened records (what Phase 3 CSV/Parquet export will use) ---")
    for f in frames:
        for record in f.to_flat_records():
            print(record)

    # Round-trip check: serialize then parse back, to make sure the schema
    # is actually a stable contract and not just "looks right in this script"
    raw = frame_propagated.model_dump_json()
    reloaded = FrameResult.model_validate_json(raw)
    assert reloaded == frame_propagated
    print("\nRound-trip serialize/deserialize check: OK")


if __name__ == "__main__":
    main()
