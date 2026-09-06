# CLAUDE.md -- football_analysis

Project-specific guidance for Claude Code. Companion to `status.md`, which
holds the live "where are we right now" snapshot of phases and test counts.
Read both at the start of any session.

**Note: the project `README.md` is stale (still describes Phase 0 only).
Do not rely on it for the current layout. `status.md` and this file are the
authoritative descriptions of what's actually in the repo.**

## Project shape

A football broadcast video -> per-frame tactical analysis pipeline, built in
named phases. Each phase produces a class/script that fits into one slot
in the pipeline. Schemas (`schemas/`) are the contract that holds all
phases together -- downstream code reads `FrameResult` and never imports
from `calibration/` or `tracking/`.

| Phase | Slot | Status |
|-------|------|--------|
| 0     | Foundations: `config/pitch_config.py`, `schemas/` | Complete |
| 1a    | Scene routing + keypoint calibration (`calibration/`) | Complete |
| 1b    | Camera-motion propagation (zoom/duel gap) | Complete |
| 1c    | Robustness polish (confidence weighting, reproj rejection, EMA smoothing) | Complete |
| 2     | Detection & tracking (`tracking/`) -- detector, `ByteTracker`, `JerseyColorTeamClassifier` | Functionally complete; only re-identification after long gaps is not built |
| 3     | Coordinate transform layer (`coordinates/`) | Not started; up next |
| 4     | Stats/analysis (radar, Voronoi, pressing, offside, passes) | Design only |
| 5     | API/microservice layer (`api/`) | Not started |

## Layout (live, not the README)

```
football_analysis/
  config/pitch_config.py          PitchConfig (pydantic), FIFA defaults
  schemas/                         Canonical output contract -- FrameResult,
                                   TrackedObject, CalibrationInfo, enums
  calibration/                     Phase 1
    scene_types.py                 SceneType enum
    scene_classifier.py            HeuristicSceneClassifier (NOT validated
                                   against real low-angle footage; treat
                                   as placeholder)
    keypoint_model.py              BaseKeypointModel + YoloKeypointModel
                                   (ultralytics lazy-imported)
    model_registry.py              KeypointModelRegistry (lazy load, fallback
                                   to broadcast, CLOSE_UP raises)
    calibrator.py                  PitchCalibrator (keypoints -> homography;
                                   has opt-in confidence weighting and
                                   reprojection-error rejection)
    smoothed_calibrator.py         SmoothedPitchCalibrator (EMA on
                                   homography, drop-in PitchCalibrator
                                   replacement; ONLY smooths
                                   CALIBRATED_FROM_KEYPOINTS, clears state
                                   on failure)
    camera_motion_tracker.py       CameraMotionTracker (LK optical flow +
                                   RANSAC homography, filters by LK error
                                   not just status==1)
    propagating_calibrator.py      PropagatingCalibrator state machine:
                                   CALIBRATED_FROM_KEYPOINTS -> PROPAGATED
                                   -> RE_ANCHORED
    pipeline.py                    CalibrationPipeline (routing; has
                                   route_and_detect() seam + process_frame)
    orchestrator.py                VideoCalibrationOrchestrator -- the one
                                   entrypoint: process_frame(frame)
  tracking/                        Phase 2
    detection.py                   Detection schema (separate from
                                   TrackedObject by design)
    detector.py                    BaseObjectDetector + YoloObjectDetector;
                                   class_id_map is REQUIRED, no default
    tracker.py                     BaseTracker + ByteTracker (IoU greedy,
                                   class-scoped, no Kalman, no
                                   interpolation, min_hits confirmation
                                   delay, max_age eviction)
    team_classifier.py             BaseTeamClassifier +
                                   JerseyColorTeamClassifier (median BGR,
                                   green-mask, EMA-smoothed running
                                   centroids with reconciliation;
                                   goalkeepers excluded from fit but still
                                   assigned via nearest-centroid;
                                   referees/ball structurally excluded)
    __init__.py                    Exports updated by the user directly
  coordinates/                     Phase 3 -- empty stub
  api/                             Phase 5 -- empty stub
  tools/distance_tool.py           Standalone "click two points, get real
                                   distance" QA tool (tested against real
                                   FIFA distances through actual
                                   PitchCalibrator homography)
  examples/demo_schema_usage.py
  tests/                           pytest suite (see status.md for counts)
  status.md                        Live status -- READ THIS for current
                                   test counts, bugs caught, design notes
  README.md                        Stale (Phase 0 framing). Do not follow.
```

## Hard project conventions

- **No scipy / no Hungarian.** Greedy matching only. `cv2.kmeans` is fine
  because OpenCV is already a hard dependency. See `ByteTracker` and
  `JerseyColorTeamClassifier` for the established pattern.
- **`ultralytics` is always lazy-imported** inside the concrete
  `YoloKeypointModel` / `YoloObjectDetector` classes. Nothing else in the
  project should `import ultralytics` at module top level.
- **pydantic for every schema** (free validation, free JSON round-trip,
  drops into FastAPI in Phase 5 with zero rework).
- **Provenance is two independent fields, never combined.** `CalibrationInfo.
  status` (geometric trust: not_calibrated / calibrated_from_keypoints /
  propagated / re_anchored) and `TrackedObject.provenance` (identity
  continuity: observed / interpolated / re_identified_after_gap) measure
  orthogonal failure modes and must stay separate.
- **`None` vs. empty array is meaningful.** `None` = "keypoint/detection
  model skipped entirely" (e.g. CLOSE_UP routing). Empty array = "model
  attempted and produced nothing." Conflating them has caused real bugs
  in this project; respect the contract in any new code that bridges
  detection results to downstream consumers.
- **Confidence-threshold tunables are inspection-tuned against synthetic
  data, not validated against real footage.** The two known cases are
  `HeuristicSceneClassifier`'s thresholds and `JerseyColorTeamClassifier`'s
  green-hue mask (`GREEN_HUE_RANGE = (35, 85)`, `GREEN_MIN_SATURATION = 60`).
  Both work in tests and are honest starting points, but should be sanity-
  checked on real broadcast lighting when available -- particularly
  artificial turf or unusually yellow/dry-grass pitches near the hue
  boundary.
- **Every meaningful class has a duck-typed interface** (`BaseObjectDetector`,
  `BaseTracker`, `BaseTeamClassifier`, `BaseKeypointModel`,
  `BaseMotionTracker`, `PitchCalibrator`-shaped). New implementations
  subclass the interface; orchestration code never depends on a concrete
  class.

## Testing approach

- One dedicated `tests/test_<module>.py` per new capability.
- Verify against synthetic data + fakes/mocks, not real model weights.
  Trained weights (`best.pt` / Soccana_Keypoint and the player/ball
  detection model) only exist on the user's machine -- everything is
  built so the suite runs without them.
- Real numerical targets where the ground truth is known: distance tool
  tested against actual FIFA distances (7.32m goal-post, 40.32m penalty
  box) through a real `PitchCalibrator` homography, not fakes.
- **Sandbox caveat:** the in-sandbox pydantic/pytest stand-in used when
  PyPI was unavailable does not support `model_dump_json` or real
  validators, so `test_frame_result.py` and `test_pitch_config.py` show
  as failing in-sandbox. Both are known-good from the user's local
  pytest runs. Do NOT treat those two in-sandbox failures as regressions
  when re-verifying -- confirm against the user's local checkout.

## Real model weights / data

- Trained weights live on the user's machine, not in this directory.
- The player/ball detection model's actual `class_id_map` (which integer
  means ball/player/goalkeeper/referee) has not yet been supplied.
  `YoloObjectDetector` cannot be instantiated against the real model
  until that mapping is provided, though all parsing logic is already
  tested independent of it.
- The eventual low-angle keypoint model MUST output the same 29-keypoint
  schema, same order, as the broadcast model -- this is what keeps
  `KeypointModelRegistry` able to treat both interchangeably. Confirmed
  with the user.

## Current entrypoints

- Calibration: `VideoCalibrationOrchestrator.process_frame(frame) ->
  OrchestratedFrameResult` (the seam that wires `CalibrationPipeline`
  routing into `PropagatingCalibrator` gap-bridging).
- Tracking: `ByteTracker.update(detections) -> List[TrackedObject]`,
  followed by `JerseyColorTeamClassifier.assign_teams(frame,
  tracked_objects)`.

## Working style (also in memory)

See `status.md`'s "Working style" section and the cross-session feedback
memories under `C:\Users\enica\.claude\projects\...\memory\`. Key rules:

- One file per response (sized for the user's daily free-tier budget).
- Re-show the full file on any edit to an already-delivered file.
- Propose explicit numbered design choices before non-trivial code.
- Local checkout is the source of truth over any session-memory snapshot.
