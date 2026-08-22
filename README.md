# football_analysis -- Phase 0

Foundations for the football tracking/tactical analysis project: the config
schema and the canonical output schema that every later phase (calibration,
tracking, coordinates, stats, API) will read and write. See the project
plan markdown for full phase-by-phase context; this package implements
Phase 0 specifically.

## Layout

```
football_analysis/
  config/
    pitch_config.py      PitchConfig -- pitch dimensions (meters), FIFA
                          defaults, validated, with a world_keypoints()
                          method producing the (29,2) meter-coordinate
                          array for the standard keypoint schema.
  schemas/
    enums.py              CalibrationStatus, ObjectClass, Team, PositionProvenance
    frame_result.py        FrameResult, TrackedObject, CalibrationInfo,
                          PixelPoint, WorldPoint, BoundingBox -- the
                          canonical per-frame output contract
    match_metadata.py     MatchMetadata, ModelVersions -- match-level context
  calibration/            Phase 1 (empty stub)
  tracking/                Phase 2 (empty stub)
  coordinates/             Phase 3 (empty stub)
  api/                     Phase 5 (empty stub)
  examples/
    demo_schema_usage.py   End-to-end usage demo, run this first
  tests/                   pytest suite locking the schema contract
```

## Why this shape

- **`PitchConfig` flows into everything.** It's created once (from API
  input in Phase 5, or FIFA-standard defaults elsewhere) and passed down
  rather than any stage hardcoding 105x68 or box dimensions. Its
  `world_keypoints()` method is now the *only* place pitch geometry is
  turned into keypoint coordinates -- Phase 1's calibrator will call this
  instead of recomputing geometry itself.
- **`FrameResult` is the contract, not an implementation detail.** Stats
  code (Phase 4, not built yet) should never need to import anything from
  `calibration/` or `tracking/` -- it reads `FrameResult` objects only.
  This is what makes "modular and scalable" concrete: any future consumer
  (a new stat, a different frontend, a different sport even) just needs to
  produce or consume this schema.
- **Provenance is tracked at the field level, not bolted on later.**
  `CalibrationInfo.status` (not_calibrated / calibrated_from_keypoints /
  propagated / re_anchored) and `TrackedObject.provenance` (observed /
  interpolated / re_identified_after_gap) exist because retrofitting "which
  numbers can I actually trust" after Phases 1-2 are built is much harder
  than designing for it from the start -- directly motivated by the
  low-angle-model and zoom/occlusion-gap discussions from planning.
- **Everything is a pydantic model.** Free validation (e.g. `PitchConfig`
  rejects a penalty area wider than the pitch), free JSON
  serialization/deserialization (round-trip tested), and this drops
  straight into FastAPI request/response models in Phase 5 with no rework.

## Running things

```bash
pip install -r requirements.txt

# see the schema in action
python examples/demo_schema_usage.py

# run the test suite
pytest tests/ -v
```

## Next: Phase 1a/1b

Calibration layer -- scene routing for low camera angles, keypoint-based
homography (porting/replacing the existing `PitchCalibrator` prototype to
consume `PitchConfig` instead of hardcoded constants), and camera-motion
propagation for the zoom/occlusion gap problem, emitting `CalibrationInfo`
per frame.
