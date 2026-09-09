# football_analysis -- Current Status
 
Companion to the project plan doc. Read that first for the full phase-by-phase
architecture; this file is just "where are we right now."
 
## Completed
 
### Phase 0 -- Foundations
- `config/pitch_config.py` -- `PitchConfig` (pydantic), FIFA-standard defaults,
  fully overridable, validated (rejects e.g. a penalty area wider than the
  pitch). `world_keypoints()` is the ONLY place pitch geometry becomes the
  29-keypoint meter template.
- `schemas/` -- canonical output contract: `FrameResult`, `TrackedObject`,
  `CalibrationInfo`, `PixelPoint`/`WorldPoint`, `MatchMetadata`. Provenance is
  first-class: `CalibrationInfo.status` (not_calibrated / calibrated_from_
  keypoints / propagated / re_anchored) and `TrackedObject.provenance`
  (observed / interpolated / re_identified_after_gap) -- these two are
  deliberately independent fields, not one combined status, since a
  perfectly-tracked player can still have a low-confidence world position if
  calibration is propagated, and vice versa.
- Fully tested (pydantic round-trip, validation rejection cases).
### Phase 1a -- Scene routing + keypoint calibration
- `calibration/scene_types.py` -- `SceneType` enum (BROADCAST_WIDE,
  LOW_ANGLE_CORNER, CLOSE_UP, UNKNOWN).
- `calibration/scene_classifier.py` -- `HeuristicSceneClassifier`, a
  placeholder rule-based classifier working off keypoint spatial pattern
  (count, spread, which named keypoint groups are visible). NOT validated
  against real low-angle footage yet -- thresholds are inspection-tuned,
  treat as a starting point.
- `calibration/keypoint_model.py` -- `BaseKeypointModel` interface,
  `YoloKeypointModel` wraps ultralytics (imported lazily so nothing else
  needs it installed).
- `calibration/model_registry.py` -- `KeypointModelRegistry`: lazy loading,
  automatic fallback to broadcast for unregistered/failed scene types,
  `CLOSE_UP` raises loudly by design (should never reach the registry).
- `calibration/calibrator.py` -- `PitchCalibrator`: keypoints -> homography
  via `cv2.findHomography`, stateless per frame (no propagation -- that's
  Phase 1b). Extended in Phase 1c (see below).
- `calibration/pipeline.py` -- `CalibrationPipeline`: orchestrates
  broadcast-model-first -> classify -> conditionally rerun with low-angle
  model -> calibrate. Skips a second inference call entirely for CLOSE_UP
  frames. **Refactored** to split routing from calibration -- see
  "Pipeline/propagation integration" below.
- All fully tested with fakes/mocks (no real model weights needed).
### Phase 1b -- Camera-motion propagation (the zoom/duel gap fix)
- `calibration/camera_motion_tracker.py` -- `BaseMotionTracker` interface,
  `CameraMotionTracker` (Lucas-Kanade optical flow + RANSAC homography,
  generic frame features, NOT pitch keypoints). Filters tracked points by
  LK error, not just the `status` convergence flag -- `status==1` only
  means the solver converged, not that the match is genuine; on unrelated
  frames it happily converges to nonsense. Empirically ~0.5 error for a
  real match vs ~7+ for garbage.
- `calibration/propagating_calibrator.py` -- `PropagatingCalibrator`: the
  full state machine `CALIBRATED_FROM_KEYPOINTS -> PROPAGATED ->
  RE_ANCHORED`, composing `PitchCalibrator` (or any duck-typed equivalent,
  see Phase 1c) + `CameraMotionTracker`. Composition math:
  `H_world_new = H_world_prev @ inverse(H_incremental)`.
- Tested (7/7 passing, confirmed via the user's own local pytest run). One
  real test bug caught and fixed: a test had passed `keypoints_px=None` for
  a "keypoints attempted but failed" frame, but `None` is reserved for
  "keypoint model skipped entirely" (CLOSE_UP) -- passing None skips
  calling `calibrate_frame` at all, which desynced a scripted fake's result
  queue. Fixed in the test, not production code -- this `None` = skipped
  vs. real-but-failing-array = attempted-and-failed distinction turned out
  to matter again later (see orchestrator section).
### Phase 1c -- Robustness polish
- `calibrator.py` extended with two **opt-in, default-off** additions so
  existing behavior/tests are unaffected unless explicitly enabled:
  1. `use_confidence_weighting` -- duplicates higher-confidence point pairs
     in the fitting set (OpenCV's `findHomography` has no native per-point
     weight param), biasing the least-squares refinement toward trusted
     points. Diagnostic reprojection error is always measured against the
     original undupli­cated points, so weighting can't make a bad
     calibration look artificially good on its own metric.
  2. `max_mean_reprojection_error` / `max_max_reprojection_error` -- turns
     the reprojection-error diagnostic into an actual rejection decision (a
     technically-successful-but-geometrically-bad homography now returns
     `NOT_CALIBRATED` instead of silently reporting
     `CALIBRATED_FROM_KEYPOINTS`). Tested with a deliberately-planted bad
     correspondence (one keypoint nudged 60px off among 8) -- confirmed the
     bad point alone pushes mean error to ~10.6px / max to ~85px, well
     above clean synthetic data's sub-pixel error, and that thresholds
     reject it while default (`None`) behavior stays unchanged. 17/17
     calibrator tests passing (8 original + 9 new).
- `calibration/smoothed_calibrator.py` -- `SmoothedPitchCalibrator`: EMA
  smoothing on the homography to reduce frame-to-frame jitter from keypoint
  detection noise. Wraps a `PitchCalibrator` and exposes the exact same
  interface (`calibrate_frame`, `pixel_to_world`, plus `reset()`), so it's
  a **drop-in replacement** anywhere a `PitchCalibrator` is expected --
  including as `PropagatingCalibrator`'s `pitch_calibrator` argument, with
  zero changes needed there. Only smooths `CALIBRATED_FROM_KEYPOINTS`
  results; a failed calibration clears smoothing state immediately (same
  "no silent staleness" rule as `PitchCalibrator` itself).
  - Real bug caught while integrating this: if `SmoothedPitchCalibrator`
    is injected into `PropagatingCalibrator` and the camera pans
    significantly during a `PROPAGATED` gap, the fresh keypoint-based
    estimate on re-anchor would get wrongly blended with the stale
    pre-gap smoothed homography -- because by the time `_handle_keypoint_
    success` runs, the blend has already happened. Fixed by resetting the
    injected calibrator's smoothing state (via `getattr(..., "reset",
    None)`, a no-op for plain `PitchCalibrator`) **before** attempting
    calibration whenever the prior status wasn't already
    `CALIBRATED_FROM_KEYPOINTS`, not after. An integration test
    (`test_reset_before_reattempt_prevents_blending_across_a_gap`) proves
    the fix.
  - 6/6 new smoother tests passing, 7/7 existing `PropagatingCalibrator`
    tests still passing (regression-checked after the reset-ordering
    change).
- `tools/distance_tool.py` -- standalone "click two points, get real
  distance in meters" dev/QA tool. Pure measurement logic
  (`compute_distance`, `measure_from_pixels`) split from the interactive
  OpenCV shell so it's unit-testable without a display. Deliberately uses
  a single calibrator/model directly, NOT the full pipeline -- validating
  homography accuracy on one frame doesn't need scene routing or gap
  propagation. Tested against REAL FIFA distances through an actual
  `PitchCalibrator` homography, not fakes: synthetic goal-post positions
  (7.32m apart) and penalty-box edges (40.32m apart) both recovered within
  5cm. 7/7 tests passing.
### Pipeline/propagation integration (the seam between Phase 1a and 1b)
This didn't have a name in the original phase plan -- it surfaced as a real
gap while building the distance tool: `CalibrationPipeline` (routing) and
`PropagatingCalibrator` (gap-bridging) had never actually been wired
together. Each called `.calibrate_frame()` on its own held calibrator
independently; nothing connected scene-routing output into the
gap-bridging state machine for continuous video use.
 
Resolved with a non-invasive seam rather than merging the two classes:
- `pipeline.py` refactored: `process_frame()` split so routing/detection
  now lives in a new `route_and_detect()` method, with `process_frame()`
  calling it and doing exactly what it did before. Verified
  behavior-identical against all 4 existing pipeline tests before building
  anything on top of it.
- `calibration/orchestrator.py` (new) -- `VideoCalibrationOrchestrator`:
  calls `pipeline.route_and_detect()`, translates `CLOSE_UP` into the
  explicit `(None, None)` `PropagatingCalibrator` expects (this translation
  belongs here, not in either existing class, since it's specifically
  about satisfying Phase 1b's contract), then calls
  `propagating_calibrator.process_frame()`. Single entrypoint:
  `process_frame(frame) -> OrchestratedFrameResult` (dataclass with
  `scene_classification`, `calibration_info`, `keypoints_px`,
  `keypoint_confidences`). Also exposes `pixel_to_world()`, delegating to
  the wrapped `PropagatingCalibrator`.
- 3 new tests, including a full broadcast -> close-up gap -> re-anchor
  sequence proving routing decisions and the state machine work correctly
  together end to end. 62/64 tests passing project-wide (the 2 "failures"
  are known limitations of the in-sandbox pydantic test stand-in used when
  PyPI access was unavailable -- see below -- not real regressions).
**Phase 1 is complete end-to-end**, with one real usable entrypoint:
`VideoCalibrationOrchestrator.process_frame(frame)`.
 
### Phase 2 -- Detection & tracking (nearly complete)
- `tracking/detection.py` -- `Detection`: the raw per-frame output of a
  detector, BEFORE any tracker has assigned identity. Deliberately a
  separate schema from `TrackedObject` (schemas/frame_result.py), not
  `TrackedObject` with an optional `track_id` -- a detector's job is "what
  objects are in this frame," a tracker's job is "which one is the same
  player as last frame," and keeping them as two schemas means a detector
  can never accidentally be handed a track_id it has no business assigning.
  Reuses `BoundingBox`/`PixelPoint`/`ObjectClass` from `schemas/` rather
  than redefining them. Two static helpers live on the class:
  `foot_point()` (bbox bottom-center, used for players/goalkeepers/
  referees) and `center_point()` (bbox center, used for the ball, since a
  ball's bbox bottom edge isn't a meaningful ground-contact point the way
  a foot is).
- `tracking/detector.py` -- `BaseObjectDetector` interface + concrete
  `YoloObjectDetector`, same shape as `calibration/keypoint_model.py`
  (`ultralytics` imported lazily so nothing else needs it installed).
  `class_id_map: Dict[int, ObjectClass]` is a **required** constructor
  arg with no default -- your trained model's class-ID convention (which
  integer means ball/player/goalkeeper/referee) is training-data-specific
  and can't be guessed at; an unmapped class ID is skipped, never
  misassigned to a default. The actual box/confidence/class-ID parsing
  logic is split into a plain static method
  (`YoloObjectDetector._boxes_to_detections`) operating on raw numpy
  arrays rather than an ultralytics `Results` object -- this is what makes
  it unit-testable without ultralytics or a GPU installed at all, same
  testability goal as the rest of the project, just achieved by isolating
  parsing logic instead of injecting a loader (there was no obvious
  loader-injection seam here the way there was for `KeypointModelRegistry`).
- `tracking/tracker.py` -- `BaseTracker` interface + `ByteTracker`: turns a
  frame's `List[Detection]` into `List[TrackedObject]` with stable
  `track_id`s. IoU-based greedy association (no scipy/Hungarian algorithm,
  same minimal-dependency rule as the rest of the project), class-scoped
  (a goalkeeper detection can never steal a player's track), ByteTrack-
  style two-stage confidence split: high-confidence detections match
  first and are the only thing allowed to spawn a brand-new track;
  detections in a lower confidence band only get a second chance to keep
  an *already-established* track alive (survives one blurry/partially-
  occluded frame without a dropped-and-respawned ID). `min_hits` delays a
  new track's first appearance in output until it's matched that many
  consecutive frames, so one spurious detection can't mint a flickering
  ID. Deliberately never emits an interpolated position -- a track missed
  this frame is simply absent from this frame's output, not guessed at;
  bridging longer gaps is explicitly out of scope for this class (that's
  the project plan's separate re-identification piece, needing team/
  jersey/appearance signals this class knows nothing about). One
  documented simplification vs. textbook ByteTrack: matching against a
  track uses its last-known bbox as-is, no Kalman-filter motion
  prediction -- adequate for bridging a handful of blur/occlusion frames
  at broadcast frame rate, not intended for multi-second gaps.
  `tests/test_tracker.py` covers IoU math directly, ID stability across
  frames, cross-class isolation, the stage-2 low-confidence recovery
  case, max_age eviction, min_hits confirmation delay, and greedy-match
  tie-breaking with two simultaneous tracks. **Confirmed passing via the
  user's own local pytest run.**
- `tracking/team_classifier.py` -- `BaseTeamClassifier` interface +
  `JerseyColorTeamClassifier`: runs after the tracker, since it needs
  pixel data (`frame`) alongside that frame's `List[TrackedObject]` --
  something neither `Detection` nor `TrackedObject` alone carries.
  - **Jersey color feature**: median BGR of each object's full bbox crop,
    after masking out grass-green pixels via an HSV hue+saturation
    threshold (`GREEN_HUE_RANGE`/`GREEN_MIN_SATURATION`). Median rather
    than mean so leftover skin/sock/advertising-board pixels can't drag
    the estimate -- a cheap "dominant color" without a second per-player
    clustering pass. A crop with no non-green pixels left (bad box, or a
    genuinely all-green sliver) returns `None` and that object is skipped
    entirely for the frame -- no guessing.
  - **Frame-to-frame stability**: rather than re-clustering cold every
    frame and hoping the two labels land the same way twice in a row, the
    classifier maintains two running reference centroids across the whole
    session. Each frame with enough eligible players: fit a fresh k=2
    cluster (`cv2.kmeans`, no scipy/Hungarian, consistent with the
    project's minimal-dependency rule) over outfield players' colors,
    **reconcile** the 2 fresh centroids against the 2 running ones
    (trivial with only 2 clusters -- compare the 2 possible pairings by
    total distance and keep the cheaper one), then **EMA-update**
    (`ema_alpha`, default 0.1) the running centroids toward the matched
    fresh ones. Same "smooth toward new evidence, don't discard history"
    pattern as `smoothed_calibrator.py`, applied here to absorb slow
    lighting drift over a match (lengthening shadows, floodlights coming
    on) without ever flipping which running centroid means which team.
  - **Goalkeeper handling**: excluded from the k=2 *fit* (a keeper's kit
    is deliberately a different color from both outfield teams and would
    corrupt the two centroids) but still gets a team assigned via
    nearest-centroid once centroids exist, since which side a keeper is
    on matters for stats (defensive shape, offside line) even though the
    jersey-color signal specifically is best-effort for keepers -- a kit
    that resembles neither outfield color is a known, undocumented-fix
    limitation.
  - Referees and the ball are structurally excluded (`TEAM_ELIGIBLE_
    CLASSES` doesn't include them) -- `team` stays `None`, always, not
    just unassigned by chance.
  - `tests/test_team_classifier.py` -- 10 tests: direct static-method
    tests of the green-masking + median extraction (including the
    all-green-crop -> `None` and no-bbox -> `None` cases), insufficient-
    players-skips-fit, correct two-team separation, label stability
    across frames despite reordered/re-clustered input (the reconciliation
    guarantee, tested by asserting the same color maps to the same team
    label across two separate `assign_teams()` calls), goalkeeper
    exclusion-from-fit verified numerically (centroids land within 1.0 of
    the true blue/red BGR values even with a yellow-kit goalkeeper
    present), referee/ball never assigned, a boxless object staying
    unassigned while others still fit, and EMA smoothing verified
    numerically (centroid moves exactly toward `0.9*old + 0.1*new`, not
    all the way to the new frame's raw color). **Confirmed passing via
    the user's own local pytest run.**
- `tracking/__init__.py` updated (by the user directly) to export
  `Detection`, `BaseObjectDetector`, `YoloObjectDetector`, `BaseTracker`,
  `ByteTracker`, `BaseTeamClassifier`, `JerseyColorTeamClassifier`.
- One real bug hit and fixed earlier in this phase: `Detection.
  center_point()` was added to `detection.py` in a follow-up edit, but the
  updated file was never re-shown in full for copying -- so the user's
  checked-out copy still had the original version, and `test_detector.py`
  (written against `center_point` already existing) failed with
  `AttributeError: center_point` on 3 tests. Not a logic bug -- fixed by
  re-supplying the complete updated `detection.py`. This is what motivated
  the "always re-show the full file on any edit" rule below, and the
  even stricter one-file-per-prompt rule.
**Phase 2 is functionally complete.** The only item from the original plan
not yet built is re-identification after long tracking gaps (see "Not
started yet" below) -- everything needed to produce a fully-populated
`TrackedObject` (identity, class, team) for a normally-tracked frame exists
and is tested.
 
### Phase 3 -- Coordinate transform layer + export
- `coordinates/transform.py` -- `transform_tracked_objects(tracked_objects,
  calibration, pitch_config, ...)`: pure function, batch pixel -> world
  transform for a frame's `List[TrackedObject]`. **Deliberately imports
  ONLY from `schemas/` and `config/`** -- no dependency on `calibration/`
  or on any calibrator/orchestrator instance at all. This was a real design
  simplification discovered while starting Phase 3, not the original plan:
  `CalibrationInfo.homography` is already a serialized 3x3 list from the
  calibration layer, so the transform doesn't need a `PitchCalibrator` or
  `VideoCalibrationOrchestrator` object in hand -- it can project points
  directly via `cv2.perspectiveTransform` on that raw matrix. This keeps
  the "every stage after config/schemas talks only to those two packages"
  rule from the project plan literally true for Phase 3, not just
  aspirational.
  - Never mutates input objects -- returns a new list via
    `TrackedObject.model_copy(update=...)`, so a pre-transform pixel-space
    list stays usable elsewhere (e.g. an overlay) without aliasing bugs.
  - **Confidence policy** (design decisions confirmed with the user):
    - `CALIBRATED_FROM_KEYPOINTS` -> `position_confidence = 1.0` (full
      trust).
    - `RE_ANCHORED` -> `position_confidence = 0.85` by default
      (`re_anchored_confidence` param) -- discounted despite being a fresh
      keypoint recompute, because the frame right after a propagated gap
      is exactly where a re-identification identity link is most likely
      to still be shaky.
    - `PROPAGATED` -> decays with `frames_since_last_anchor`:
      `max(propagated_confidence_floor, 1.0 -
      propagated_decay_per_frame * frames)`, defaults `floor=0.3`,
      `decay_per_frame=0.02`. Floored so a long-but-otherwise-healthy
      propagation isn't driven all the way to zero by elapsed time alone.
    - `NOT_CALIBRATED` (or a `CalibrationInfo` with no homography despite
      a different status) -> `world_position = None`,
      `position_confidence = None`, no projection attempted.
  - **Out-of-bounds handling** (confirmed): a projected point failing
    `PitchConfig.in_bounds()` is NOT dropped -- `world_position` is still
    populated with the (implausible) value, but `position_confidence` is
    forced to `0.0`. Keeps the point visible for debugging/overlay (e.g.
    spotting a homography producing garbage) while telling downstream
    stats not to trust it.
  - `tests/test_transform.py` -- 9 tests: `NOT_CALIBRATED` passthrough,
    exact pixel->world recovery + full confidence for
    `CALIBRATED_FROM_KEYPOINTS` (synthetic 10px/meter homography, same
    convention as `test_calibrator.py`), the `RE_ANCHORED` discount value,
    `PROPAGATED` decay at two different gap lengths plus the floor clamp,
    out-of-bounds keep-but-zero-confidence behavior, non-mutation of
    inputs, identity-field (track_id/class) passthrough, and multi-object
    batching. **Confirmed passing via the user's own local pytest run.**
- `coordinates/export.py` -- `FrameResultCSVWriter` (context-manager
  streaming writer) + `export_frames_to_csv()` (convenience wrapper over
  an `Iterable[FrameResult]`, so a generator-based frame loop can be
  passed directly without materializing the whole match into a list
  first). Uses `FrameResult.to_flat_records()` (already built in Phase 0)
  for the actual flattening -- this module's only job is dict-stream ->
  CSV.
  - **Streaming, not batch-in-memory, by design**: a full match can be
    tens of thousands of frames. Writing incrementally with a flush after
    every frame means (a) the whole match's tracked-object history is
    never held in RAM at once, and (b) a crash partway through a long run
    leaves a valid, readable partial CSV instead of losing everything --
    the export-layer analogue of the project's existing "no silent
    staleness" rule, applied to I/O instead of calibration state.
  - **CSV only, no Parquet**, on purpose -- Parquet needs
    `pyarrow`/`fastparquet`, a new dependency with no current downstream
    consumer. CSV needs nothing beyond the stdlib `csv` module, consistent
    with the project's minimal-dependency rule. Add a Parquet exporter
    later, behind the same interface, if/when something actually needs it.
  - Fixed, explicit `FIELDNAMES` list (not inferred from the first
    frame's records) -- inferring is fragile if frame 0 happens to have
    zero tracked objects.
  - A frame with zero tracked objects contributes zero rows (no
    placeholder row) -- the CSV is object-centric, not frame-centric.
  - `tests/test_export.py` -- 11 tests: header correctness, row counts
    (including the zero-row empty-frame case), value fidelity against
    `to_flat_records()`, `None` -> empty-string CSV rendering, frame
    ordering across multiple `write_frame()` calls, the
    "must-be-used-as-context-manager" guard, the mid-run flush guarantee
    (read the file while still inside the `with` block, before `__exit__`
    runs), both the list and generator paths through
    `export_frames_to_csv()`, and string-path support. Written this
    session -- pending the user's own local pytest confirmation.
**Phase 3 is functionally complete.** Per-track summary CSV remains
deliberately deferred (trivially a `groupby("track_id")` over this
module's CSV output via pandas, not worth bespoke code).
 
### Phase 2 (remainder) -- Re-identification after long tracking gaps
- `tracking/reid.py` -- `BaseReIdentifier` interface + `ReIdentifier`:
  answers "is this frame's brand-new track actually a returning player
  `ByteTracker` already dropped?" Sits in the pipeline between the team
  classifier and Phase 3's transform:
  `detector -> tracker -> team_classifier -> reid -> transform`. This
  order is load-bearing, not incidental -- re-ID reads the team
  classifier's per-track jersey-color cache
  (`JerseyColorTeamClassifier.get_color_history()`, added additively to
  support this) for both the incoming new track and whatever dropped
  track it's compared against, so team classification must run first;
  and its world-speed gate needs the current frame's homography, so
  calibration must run before re-ID too.
  - **`min_reid_gap_frames` structural coupling**: constructed as
    `ReIdentifier(..., min_reid_gap_frames=tracker.max_age)` on purpose --
    without this floor, a track missing for a single frame was
    immediately re-ID-eligible and could lose to an older, unrelated
    dropped track with a marginally closer color match, racing
    `ByteTracker`'s own stage-2 low-confidence recovery pass that's
    specifically built to survive that exact 1-frame case internally.
    Tying the two constants together by construction (not as two
    independently-tuned numbers) prevents this class of bug from
    recurring.
  - **Gating pipeline per candidate**: world-speed gate (rejects implied
    speeds over `max_plausible_speed_mps`, default 12.0 -- generous on
    purpose, since a false negative here is irrecoverable but a false
    positive is not) -> color gate
    (`color_match_max_distance_bgr`, default 60.0) -> optional pluggable
    `appearance_embedder` tiebreaker (v1 default: none). Best surviving
    candidate wins by lowest color distance, ties broken by recency.
  - Re-stamps `track_id` and sets `provenance = RE_IDENTIFIED_AFTER_GAP`
    on a match; otherwise the new track keeps its own ID untouched.
  - Keeps its own per-track history (`_ReIdTrackHistory`, same
    deliberately-non-pydantic private-bookkeeping pattern as
    `ByteTracker._Track`) rather than relying solely on the team
    classifier's cache, since it needs a world-position snapshot at drop
    time and a color fallback the team classifier's cache doesn't
    guarantee to have.
  - Derives its own synthetic clock from `video_fps` (`0, 1/fps, 2/fps,
    ...`) rather than accepting a frame index -- this only stays correct
    if it's called exactly once per consecutive video frame, never
    skipping. Documented as a real constraint on the caller, not papered
    over.
### Phase 2 + Phase 3 integration -- `MatchPipeline`
The seam wiring tracking and coordinate-transform together into one
real per-frame video-loop entrypoint, same shape as `orchestrator.py`
being the seam between Phase 1a and 1b.
- `video_pipeline.py` (project root, not inside any single package --
  it's the one piece that legitimately needs to know about
  `calibration/`, `tracking/`, `coordinates/`, `config/`, and `schemas/`
  all at once) -- `MatchPipeline.process_frame(frame, frame_index,
  timestamp_s) -> PipelineFrameOutput`. Fixed call order per frame:
  1. `calibration_orchestrator.process_frame(frame)` -- runs FIRST,
     since re-ID's speed gate needs this frame's homography and has no
     way to ask for it later.
  2. `detector.detect(frame) -> List[Detection]`
  3. `tracker.update(detections) -> List[TrackedObject]`
  4. `team_classifier.assign_teams(frame, tracked_objects)` -- populates
     `.team` AND the per-track color cache re-ID reads next.
  5. `reid.update(tracked_objects, calibration_info, video_fps)` --
     re-stamps track_id/provenance where a returning player is
     recognised.
  6. `transform_tracked_objects(tracked_objects, calibration_info,
     pitch_config, ...)` -- Phase 3, fills in `world_position` /
     `position_confidence`.
  7. Wraps the result in `FrameResult(frame_index, timestamp_s,
     calibration_info, tracked_objects)` -- the canonical Phase 0 output.
  - `frame_index`/`timestamp_s` are supplied by the CALLER on every call,
    not tracked internally -- an internal counter would silently desync
    from reality the moment a frame gets skipped upstream (e.g. a
    frame-sampling step processing every 2nd frame), the same "no silent
    staleness" principle used for calibration state and CSV export
    elsewhere in this project. Tradeoff documented in the module
    docstring: the caller-supplied `timestamp_s` must stay consistent
    with the `video_fps` `reid` was constructed with, since `reid`
    derives its own internal clock from `video_fps` rather than from
    `frame_index` -- `MatchPipeline` cannot reconcile the two if they
    drift, because `ReIdentifier.update()`'s interface doesn't accept a
    frame index at all.
  - `PipelineFrameOutput` wraps the canonical `FrameResult` alongside
    calibration debug info (`scene_classification`, raw keypoints) that
    overlay/debugging tools may want but that has no place in the
    canonical schema -- same reasoning as `OrchestratedFrameResult`
    keeping that data separate from `CalibrationInfo`.
  - `reset()` resets `tracker` and `reid` only (via duck-typed
    `getattr(..., "reset", None)`, same pattern as
    `PropagatingCalibrator`'s calibrator-reset handling) -- for a hard
    scene cut, where carrying tracker IDs or re-ID history across the
    cut would be actively wrong. Deliberately does NOT reset
    `calibration_orchestrator`: its internal state already self-corrects
    the moment fresh keypoints reappear (`RE_ANCHORED`), which is the
    right behavior across a cut too.
  - `tests/test_video_pipeline.py` -- 12 tests, all against fakes/duck
    types for every injected component except the real (unfaked)
    `transform_tracked_objects` call: exact 5-component call order,
    the calibration-info-reaches-reid handoff, video_fps forwarding, the
    team-classification-visible-to-reid ordering guarantee, a re-ID
    relink surviving into the final `FrameResult`, real-transform
    world-position correctness (including a `NOT_CALIBRATED` case and a
    `PROPAGATED` case with a custom decay override), caller-supplied
    `frame_index`/`timestamp_s` passthrough, `PipelineFrameOutput`'s
    debug fields, and `reset()` touching only `tracker`/`reid` -- not
    the calibration orchestrator. One real test bug caught: the fake
    detection in the test setup passed `bounding_box=None`, but unlike
    `TrackedObject.bounding_box` (optional), `Detection.bounding_box` is
    a required field -- fixed in the test, not production code.
    **Confirmed passing via the user's own local pytest run: 155/155
    tests project-wide.**
**Phase 2 is now fully complete, including re-identification.** The
Phase 1a/1b and Phase 2/3 integration seams both now exist, giving the
project one real usable entrypoint end to end:
`MatchPipeline.process_frame(frame, frame_index, timestamp_s)`.
 
## Not started yet
 
- Phase 4 -- stats/analysis layer (radar view, team shape, Voronoi,
  pressing heatmaps, offside line, pass detection) -- design-only so far,
  see the plan doc. This is the next real design work, now that
  `MatchPipeline` actually produces the `FrameResult`-shaped, fully
  world-position-and-identity-enriched data it depends on.
- Phase 5 -- API/microservice layer.
## Standing facts worth knowing
 
- No trained low-angle model exists yet. The registry already handles this
  cleanly (falls back to broadcast, zero code changes needed once trained)
  -- confirmed by `test_falls_back_when_low_angle_model_not_yet_available`
  and the pipeline-level equivalent.
- The eventual low-angle model MUST output the same 29-keypoint schema,
  same order, as the broadcast model -- confirmed with the user. This is
  what keeps the registry/pipeline able to treat both models
  interchangeably behind `BaseKeypointModel`.
- Real trained model weights (`best.pt` / Soccana_Keypoint, and the
  player/ball detection model referenced in Phase 2) only exist on the
  user's machine -- not available to test against directly in a sandbox
  environment. Everything built so far is verified against synthetic data
  and fakes/mocks (dependency injection throughout was a deliberate design
  choice specifically to make this possible).
- The player/ball detection model's actual `class_id_map` (which integer
  class ID means ball/player/goalkeeper/referee) has not yet been supplied
  -- needed before `YoloObjectDetector` can be instantiated against the
  real model, though all of its parsing logic is already tested
  independent of that mapping.
- `JerseyColorTeamClassifier`'s green-hue mask (`GREEN_HUE_RANGE = (35,
  85)`, `GREEN_MIN_SATURATION = 60`) is inspection-tuned against synthetic
  test colors, same caveat as the scene classifier's thresholds -- worth
  sanity-checking (or retuning) against real broadcast footage lighting
  once available, particularly artificial-turf or unusually
  yellow/dry-grass pitches that might sit closer to the hue boundary.
- The Phase 3 confidence-policy constants (`re_anchored_confidence=0.85`,
  `propagated_confidence_floor=0.3`, `propagated_decay_per_frame=0.02`)
  are inspection-chosen defaults, same caveat as every other
  inspection-tuned threshold in this project (scene classifier, green-hue
  mask) -- worth revisiting once real footage/tracking data exists to
  check whether the decay rate and floor actually match observed drift
  behavior, rather than treating them as final.
- `ReIdentifier`'s tuning constants (`max_plausible_speed_mps=12.0`,
  `color_match_max_distance_bgr=60.0`, `max_reid_gap_s=5.0`) are likewise
  inspection-chosen, not validated against real footage yet.
  `min_reid_gap_frames` is the one exception -- it's deliberately NOT an
  independent inspection-tuned constant, it's structurally coupled to
  `ByteTracker.max_age` by construction (`min_reid_gap_frames=
  tracker.max_age`) specifically to prevent re-ID and the tracker's own
  recovery window from fighting each other.
- `requirements.txt` is current: `pydantic`, `numpy`, `opencv-python` (core,
  needed for the whole test suite -- non-lazy import in
  `tracking/team_classifier.py` for `cv2.kmeans`, `coordinates/
  transform.py` for `cv2.perspectiveTransform`, and `tracking/reid.py`
  for the same, same as it already was in `calibration/`), `ultralytics`
  (only needed by `YoloKeypointModel`/`YoloObjectDetector` for real
  inference, lazily imported so nothing else requires it), `pytest`
  (dev/testing). No new dependency was added for Phase 3 or the reid/
  integration work -- CSV export uses only the stdlib `csv` module.
## Working style for this project (updated)
 
- Build and verify one file/feature at a time rather than batching
  multiple phases into one response; write a dedicated test file per new
  capability; explain design reasoning (especially scope boundaries
  between classes) before writing code.
- When an already-delivered file is edited, re-show the full file rather
  than just describing the change, since there's no live sync to the
  user's actual checkout.
- **One file per prompt, hard limit.** The user is on Anthropic's daily
  free-tier usage limit, and a single prompt that writes multiple
  substantial files (e.g. a tracker implementation + its full test file in
  one response) can burn the entire day's allowance before the user gets
  to actually run and confirm anything. Going forward:
  - If a planned step naturally involves several files (e.g. an
    implementation file + its test file), split it into that many
    separate prompts/responses -- one file delivered, then stop, even if
    the next file is small and "obviously" comes next.
  - If a single file is itself large enough to plausibly exceed one
    day's free-tier budget on its own, split THAT file into multiple
    parts across multiple responses (e.g. class skeleton + docstrings in
    one response, method bodies in the next) rather than trying to push
    it through in one shot.
  - This means the natural unit of work per response shrinks: don't plan
    "implementation + tests" as one deliverable anymore -- plan
    "implementation" as one deliverable, confirm, then "tests" as the
    next.
  - Existing already-completed files are unaffected by this -- it only
    changes how NEW work gets chunked from here forward.
  - Trivial, additive changes (e.g. a one-line `__init__.py` export
    addition) can be handled by the user directly without spending a
    prompt on them, as happened with `tracking/__init__.py` this session.
- Design decision points (e.g. jersey-color feature extraction, centroid
  stability strategy, goalkeeper handling, and -- this session -- the
  Phase 3 confidence-discount policy and out-of-bounds handling) are
  proposed as explicit, numbered choices and confirmed by the user BEFORE
  any code is written.
- Sandbox resilience note: mid-project, the sandbox environment reset
  (lost all files, then lost package-registry access) -- full project was
  reconstructed from conversation history alone and verification continued
  using only `numpy`/`opencv` plus a minimal in-memory pydantic/pytest
  stand-in. That stand-in doesn't support `model_dump_json` or real
  pydantic validators, so `test_frame_result.py` and `test_pitch_config.py`
  show as failing in-sandbox -- both are known-good from the user's own
  local `pytest` runs and unaffected by anything built since. The user's
  own local environment (Windows, real venv) has confirmed Phase 1, all
  of Phase 2 (detection/detector/tracker, team classifier, re-ID), Phase
  3 (transform + export), and the `MatchPipeline` integration layer all
  pass for real via `pytest` -- 155/155 project-wide as of the
  `MatchPipeline` session.
- **Reminder for next session**: project knowledge base files (this status
  doc, and any copies of source files kept there) can drift out of sync
  with the user's actual local repo -- confirmed firsthand earlier this
  project when an attached `tracking/__init__.py` snapshot didn't match
  what the status doc claimed was already exported, and again when this
  session started with only the plan/status docs in the knowledge base and
  no actual `.py` source files (including `calibration/orchestrator.py`,
  needed for Phase 3 and only obtained by the user pasting it directly).
  Treat the user's local checkout as the source of truth whenever the two
  disagree, and ask/diff rather than overwrite blind. Consider adding
  actual source files to the project knowledge base (not just these two
  planning docs) if repeatedly re-pasting files each session becomes a
  recurring cost.
## Next up
 
Phase 4 -- stats/analysis layer design, starting with 4a (the top-down
tactical "radar" view, per the plan doc's suggested build order -- cheap,
needs only Phase 3 world positions, and doubles as a visual sanity check
for everything upstream: a jittery or drifting radar view is an early
warning sign for a calibration or tracking bug before any numeric stat
would reveal it). `MatchPipeline` now produces real
`FrameResult`-shaped, world-position-and-identity-enriched data end to
end, so this is genuinely buildable now, not just designable.
 
Also still open, not urgent: actually hooking `FrameResultCSVWriter` up
to a real `MatchPipeline`-driven video loop (both are independently
tested and the wiring is trivial -- `writer.write_frame(output.
frame_result)` per iteration -- just hasn't been written as an
end-to-end script/example yet).