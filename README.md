# football_analysis
 
Modular pipeline for football match video analysis: pitch calibration,
player/ball detection and tracking, pixel→world coordinate transform, and
tactical visualization (radar view, team shape, Voronoi space-control,
pressing-intensity heatmaps). See `football_tracking_project_plan.md` for
the original phase-by-phase design plan and `football_analysis_status.md`
for the full, detailed running log of what's been built, tested, and
debugged session by session. This file is the top-level orientation: what
exists, how it's laid out, how to run it, and where it honestly stands.
 
## Current status
 
| Phase | Scope | Status |
|---|---|---|
| 0 | Pitch config schema, canonical per-frame output schema | Done, tested |
| 1 | Scene routing, keypoint model, homography calibration, camera-motion propagation | Done, tested end to end |
| 2 | Player/ball detection, multi-object tracking, team classification, re-identification | Functionally complete |
| 3 | Pixel→world coordinate transform, CSV export | Done, tested |
| 4a/4b | Radar view, average team shape, Voronoi diagram, pressing heatmap | Implemented and tested |
| 4c | Offside line, pass detection | Not started (depends on reliable ball tracking) |
| 5 | API/microservice layer | Not started |
 
The pipeline runs end to end on real video today, with GPU-accelerated
inference, producing per-frame CSV output and any combination of a
radar-only video, an annotated source-footage video (boxes, track IDs,
world coordinates, optional raw-keypoint diagnostic overlay), and a
combined side-by-side video.
 
## Layout
 
```
football_analysis/
  config/
    pitch_config.py       PitchConfig -- pitch dimensions (meters), FIFA
                           defaults, validated, world_keypoints() producing
                           the (29,2) meter-coordinate array for the
                           standard keypoint schema.
  schemas/
    enums.py               CalibrationStatus, ObjectClass, Team, PositionProvenance
    frame_result.py        FrameResult, TrackedObject, CalibrationInfo,
                           PixelPoint, WorldPoint, BoundingBox -- the
                           canonical per-frame output contract
    match_metadata.py      MatchMetadata, ModelVersions
  calibration/             Phase 1 -- scene routing + keypoint calibration
    scene_types.py          SceneType enum
    scene_classifier.py     HeuristicSceneClassifier (per-axis coverage
                           check -- see "Known limitations" below)
    keypoint_model.py       BaseKeypointModel, YoloKeypointModel
    model_registry.py       KeypointModelRegistry -- lazy loading, fallback
    calibrator.py            PitchCalibrator -- keypoints -> homography
    camera_motion_tracker.py CameraMotionTracker -- generic optical-flow
                           motion estimation, independent of pitch keypoints
    propagating_calibrator.py PropagatingCalibrator -- the
                           CALIBRATED_FROM_KEYPOINTS -> PROPAGATED ->
                           RE_ANCHORED state machine
    pipeline.py              CalibrationPipeline -- routes a frame to the
                           right keypoint model
    orchestrator.py          VideoCalibrationOrchestrator -- the single
                           real per-frame calibration entrypoint
  tracking/                 Phase 2 -- detection & tracking
    detection.py             Detection -- raw per-frame detector output
    detector.py               BaseObjectDetector, YoloObjectDetector
    tracker.py                BaseTracker, ByteTracker
    team_classifier.py        JerseyColorTeamClassifier
    reid.py                   ReIdentifier -- re-identification after gaps
  coordinates/               Phase 3 -- coordinate transform + export
    transform.py              transform_tracked_objects() -- pixel -> world
    export.py                  FrameResultCSVWriter, export_frames_to_csv()
  stats/                     Phase 4 -- visualization & analytics
    radar.py                  Top-down radar view (single-frame snapshot)
    team_positions.py         Shared team-labeled-position grouping primitive
    team_shape.py              Average formation (rolling window)
    voronoi.py                  Space-control diagram (single-frame snapshot)
    pressing_heatmap.py          Distance-to-nearest-opponent heatmap (rolling window)
    overlay.py                    Source-footage overlays (tracking boxes,
                               raw keypoint diagnostics) + side-by-side compositing
  video_pipeline.py          MatchPipeline -- the single real per-frame
                           entrypoint tying Phases 1-3 together
  api/                       Phase 5 (empty stub)
  examples/
    demo_schema_usage.py       Phase 0 schema usage demo
    run_match_pipeline.py       Real end-to-end video-loop driver -- see
                           "Running things" below
  tools/
    distance_tool.py            Standalone click-two-points, get real
                           distance in meters dev/QA tool
  tests/                       pytest suite covering every module above
```
 
## Why this shape
 
- **`PitchConfig` flows into everything.** Created once (from API input in
  Phase 5, or FIFA-standard defaults elsewhere) and passed down rather
  than any stage hardcoding 105x68 or box dimensions.
- **`FrameResult` is the contract, not an implementation detail.** Every
  stage after raw detection produces or consumes this schema and nothing
  stage-specific -- `stats/` never imports from `calibration/` or
  `tracking/`, `coordinates/` never imports from `calibration/`, etc.
  This is what let Phase 4's four visualization modules get built and
  tested completely independently of the real detection/tracking stack.
- **Provenance is tracked at the field level.** `CalibrationInfo.status`
  and `TrackedObject.provenance` exist so "which numbers can I actually
  trust" is answered by the schema itself, not reconstructed after the
  fact.
- **Dependency injection throughout.** Every stage's real implementation
  sits behind a small interface (`BaseKeypointModel`, `BaseObjectDetector`,
  `BaseTracker`, `BaseTeamClassifier`, `BaseReIdentifier`,
  `BaseSceneClassifier`), which is what made a from-scratch sandbox
  reconstruction of the whole test suite possible without real model
  weights or a GPU at every step of development.
- **Everything is a pydantic model.** Free validation, free JSON
  serialization, and drops straight into a future FastAPI layer with no
  rework.
## Running things
 
```bash
pip install -r requirements.txt
 
# see the Phase 0 schema in isolation
python examples/demo_schema_usage.py
 
# run the full test suite
pytest tests/ -v
 
# run the real end-to-end pipeline on a video file
python -m football_analysis.examples.run_match_pipeline \
    --video path/to/match.mp4 \
    --keypoint-model weights/keypoint_broadcast.pt \
    --detection-model weights/detector_player_ball.pt \
    --class-id-map 0:player,1:referee,2:ball \
    --output-csv tracks.csv \
    --output-combined-video combined.mp4 \
    --device cuda:0
```
 
Key flags worth knowing about on `run_match_pipeline.py` (run `--help` for
the full list):
 
- `--class-id-map` -- **must match your specific trained detector's class
  ID convention exactly.** Getting this wrong doesn't error, it silently
  mislabels every detection (this has already happened once during
  development -- see "Known limitations").
- `--output-radar-video` / `--output-annotated-video` /
  `--output-combined-video` -- independently requestable; a view is only
  ever rendered once per frame even if multiple outputs need it.
- `--show-keypoints` -- overlays the raw per-slot keypoint detections
  (green = counted as visible, red = detected but below threshold) onto
  the annotated/combined video, for diagnosing calibration problems.
- `--calibration-confidence-threshold`, `--scene-min-points`,
  `--scene-min-width-fraction`, `--scene-min-height-fraction` -- tunables
  for the scene classifier and calibrator, added specifically because the
  original hardcoded defaults were inspection-tuned guesses that turned
  out to be wrong on real footage (see below).
- `--log-every 1 --max-frames N` -- per-frame diagnostic logging
  (scene classification confidence/details, calibration status,
  frames-since-anchor) for debugging a specific stretch of a clip.
## Known limitations (honest, as of this writing)
 
- **The scene classifier is a hand-tuned heuristic, not a learned or
  validated one.** One real bug in it was found and fixed against actual
  match footage (an area-product spatial-coverage check was rejecting
  frames whose keypoints were well-spread on one axis but narrow on the
  other -- replaced with an independent per-axis check), but the
  thresholds are still inspection-tuned defaults, not the product of a
  labeled validation set.
- **No dedicated low-angle keypoint model exists yet.** `LOW_ANGLE_CORNER`
  frames transparently fall back to the broadcast model, which was never
  trained for that framing.
- **The keypoint model degrades badly on some non-standard camera
  orientations.** Under investigation -- current working hypothesis is a
  video decode/orientation-metadata issue effectively handing the model a
  90°-rotated frame in some clips, not a fundamental model limitation. A
  rotate-and-retry fallback (applied only to the keypoint model's input,
  never to the detector's) is in design.
- **Reprojection-error-based rejection of bad homographies exists in
  `PitchCalibrator` but isn't wired into the real video-loop driver yet.**
  A geometrically bad calibration can currently still be reported as
  `CALIBRATED_FROM_KEYPOINTS` rather than being rejected.
- **Team classification and re-identification thresholds are also
  inspection-tuned**, not validated against labeled data.
- **No quantitative accuracy evaluation exists.** All validation to date
  has been visual/spot-check against real footage and synthetic-data unit
  tests -- there is no benchmark dataset or ground-truth comparison yet.
- **Throughput is below real-time** on a full match video in current
  testing; no formal profiling or optimization pass has been done.
## Next up
 
1. Root-cause and fix the camera-orientation keypoint issue (rotate-and-
   retry fallback, isolated to the keypoint model only).
2. Wire reprojection-error rejection into the real pipeline.
3. Investigate synthetic perspective-augmentation training data, generated
   from already-correctly-calibrated frames via the existing homography
   math, to improve keypoint model robustness without new manual labeling.
4. Phase 4c -- offside line visualization, pass detection.
5. A real quantitative accuracy/throughput benchmark.
6. Phase 5 -- API/microservice layer.